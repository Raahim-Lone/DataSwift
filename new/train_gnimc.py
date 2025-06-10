#Training script for the adapted IMC
from __future__ import annotations


import os, csv, sys, json, time, random, re, joblib, argparse, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch_geometric.data import Batch
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss, HuberLoss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from modules.parse_plan import parse_plan, canonical, _unwrap_plan
from modules.op_maps import NODE_TYPE_MAP, EDGE_TYPE_MAP
from modules.plan2vec import Plan2VecEncoder


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


FINE_TUNE_AT = 90         
FINE_TUNE_LR = 8e-5     
FINE_TUNE_TMAX = 45      
RANK_START = 90      

def fine_tune_phase(opt, head):
  """Lower LR and unfreeze variance head."""
  for g in opt.param_groups:
      g["lr"] = FINE_TUNE_LR
  for p in head.var_raw.parameters():
      p.requires_grad = True
  print("🔄  Fine-tune phase — lr → %.1e, var_raw unfrozen" % FINE_TUNE_LR)



def ranking_accuracy(mu: torch.Tensor, y: torch.Tensor) -> float:
  B = mu.size(0)
  if B < 2:
      return float("nan")
  idx = torch.randperm(B, device=mu.device)
  half = B // 2
  good = (
      ((y[idx[:half]] < y[idx[half:]]) & (mu[idx[:half]] < mu[idx[half:]]))
      | ((y[idx[:half]] > y[idx[half:]]) & (mu[idx[:half]] > mu[idx[half:]]))
  )
  return good.float().mean().item()



def classification_accuracy(logits, labels):
  return (logits.argmax(1) == labels).float().mean().item()



def warm_up_maps(plan_root: Path) -> None:
  for p in plan_root.rglob("*.json"):
      for rec in (
          json.load(open(p))
          if isinstance(json.load(open(p)), list)
          else [json.load(open(p))]
      ):
          stk = [_unwrap_plan(rec)]
          while stk:
              node = stk.pop()
              NODE_TYPE_MAP.setdefault(
                  canonical(node.get("Node Type", "unknown")), len(NODE_TYPE_MAP)
              )
              EDGE_TYPE_MAP.setdefault(node.get("Join Type", "child"), len(EDGE_TYPE_MAP))
              stk.extend(node.get("Plans", []))
  if NODE_TYPE_MAP.get("unknown", 0) != 0:  # ensure unknown=0
      ordered = ["unknown"] + [k for k in NODE_TYPE_MAP if k != "unknown"]
      NODE_TYPE_MAP.clear()
      for i, k in enumerate(ordered):
          NODE_TYPE_MAP[k] = i
  EDGE_TYPE_MAP.setdefault("child", 0)








def rebuild_sqls(plan_root: Path):
  sqls, mapping = [], {}
  for p in plan_root.rglob("*.json"):
      for rec in (
          json.load(open(p))
          if isinstance(json.load(open(p)), list)
          else [json.load(open(p))]
      ):
          q = rec.get("sql", "").strip()
          if q and q not in mapping:
              mapping[q] = str(p)
              sqls.append(q)
  return sqls, mapping








def build_sql_vocab(plan_root: Path, max_size: int = 10_000):
  tok2idx = {"<pad>": 0, "<unk>": 1}
  tokenise = re.compile(r"\w+").findall
  for p in plan_root.rglob("*.json"):
      for rec in (
          json.load(open(p))
          if isinstance(json.load(open(p)), list)
          else [json.load(open(p))]
      ):
          for t in tokenise(rec.get("sql", "")):
              if t not in tok2idx:
                  tok2idx[t] = len(tok2idx)
                  if len(tok2idx) >= max_size:
                      return tok2idx
  return tok2idx








class PairDS(Dataset):
  def __init__(
      self,
      obs,
      sqls,
      sql2path,
      W,
      Y,
      base_log,
      mu,
      sig,
      vocab,
      maxlen: int = 128,
  ):
      self.obs, self.sqls, self.sql2path = obs, sqls, sql2path
      self.W, self.Y = W, Y
      self.base_log = base_log
      self.mu, self.sig = mu, sig
      self.vocab, self.maxlen = vocab, maxlen
      self.tok = re.compile(r"\w+").findall
      self.errors = [0.0] * len(obs)




  def __len__(self):
      return len(self.obs)




  def __getitem__(self, k):
      q_idx, h_idx = self.obs[k]
      recs = json.load(open(self.sql2path[self.sqls[q_idx]]))
      rec = recs[0] if isinstance(recs, list) else recs
      g = parse_plan(rec)
      h = torch.from_numpy(self.Y[h_idx]).float()
      y_raw = np.log1p(self.W[q_idx, h_idx]) - self.base_log[q_idx]
      y = (y_raw - self.mu) / self.sig




      toks = self.tok(rec.get("sql", "").lower())
      ids = [self.vocab.get(t, 1) for t in toks[: self.maxlen]]
      mask = [1] * len(ids)
      ids.extend([0] * (self.maxlen - len(ids)))
      mask.extend([0] * (self.maxlen - len(mask)))




      return (
          q_idx,
          g,
          torch.tensor(ids),
          torch.tensor(mask),
          h,
          torch.tensor(y, dtype=torch.float32),
      )








def collate(batch):
  qs, gs, ids, mask, hs, ys = zip(*batch)
  return (
      torch.tensor(qs, dtype=torch.long),
      Batch.from_data_list(gs),
      torch.stack(ids),
      torch.stack(mask),
      torch.stack(hs),
      torch.stack(ys),
  )








class GatedFusion(nn.Module):
  def __init__(self, struct_dim: int, text_dim: int):
      super().__init__()
      self.gate = nn.Sequential(nn.Linear(text_dim, struct_dim), nn.Sigmoid())




  def forward(self, z_struct, z_text):
      return z_struct * self.gate(z_text) + z_struct








class GNIMCModel(nn.Module):
  def __init__(self, q_dim: int, h_dim: int, rank: int, num_hints: int):
      super().__init__()
      self.U = nn.Parameter(torch.randn(q_dim, rank) * 0.2)
      self.V = nn.Parameter(torch.randn(h_dim, rank) * 0.2)
      self.bias_z = nn.Linear(1, 1)
      self.bias_h = nn.Linear(1, 1)
      self.var_raw = nn.Linear(2 * rank + 1, 1)
      nn.init.zeros_(self.var_raw.bias)
      self.classifier = nn.Linear(q_dim, num_hints)




  def forward(self, z, h):
      if z.dim() == 1:
          z = z.unsqueeze(0)
      if h.dim() == 1:
          h = h.unsqueeze(0)
      if z.size(0) != h.size(0):
          if z.size(0) == 1:
              z = z.expand(h.size(0), -1)
          if h.size(0) == 1:
              h = h.expand(z.size(0), -1)
      u, v = z @ self.U, h @ self.V
      inter = (u * v).sum(1)
      ones = torch.ones_like(inter).unsqueeze(-1)
      mu = inter + self.bias_z(ones).squeeze(-1) + self.bias_h(ones).squeeze(-1)
      sig = (
          nn.functional.softplus(
              self.var_raw(torch.cat([u, v, (u - v).abs().sum(1, keepdim=True)], 1))
          ).squeeze(-1)
          + 1e-4
      )
      return mu, sig, self.classifier(z)








class GNIMCWithFusion(nn.Module):
  def __init__(self, enc, fus, head):
      super().__init__()
      self.e = enc
      self.f = fus
      self.h = head




  def forward(self, gb, ids, mask, z_text_all, q_idx, h_emb):
      z_struct = self.e(gb, ids, mask)
      z_text = z_text_all[q_idx]
      z_cat = torch.cat([self.f(z_struct, z_text), z_text], 1)
      return self.h(z_cat, h_emb)

parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true",
                  help="resume from models/gnimc_ckpt.pt")
args = parser.parse_args()

def main():

  t0 = time.time()
  HOME = Path.home()
  data_root = HOME / "New_Data"
  W = np.load(data_root / "Wnew.npy")
  M = np.load(data_root / "Mnew.npy")
  Y = np.load(data_root / "Y_scalednew.npz")["Y"]
  Y = StandardScaler().fit_transform(Y)
  num_q, _ = W.shape
  default_log = np.log1p(W[:, 0])


  if (M > 1).any():
      rep_idx = np.argwhere(M > 1)
      noise_floor = np.std(np.log1p(W[M > 0]) - default_log[rep_idx[:, 0]])
      print(f"≈ intrinsic σ (noise floor): {noise_floor:.3f}")
  else:
      print("No repeat trials → noise floor unknown")


  plan_root = HOME / "Downloads" / "dsb"
  warm_up_maps(plan_root)
  sql_vocab = build_sql_vocab(plan_root)
  sqls, sql2path = rebuild_sqls(plan_root)
  assert len(sqls) == num_q


  models_dir = HOME / "models"
  models_dir.mkdir(exist_ok=True)
  metrics = []  

  metrics_csv_path = models_dir / "metrics.csv"
  with open(metrics_csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "epoch",
        "loss_pred", "loss_rank", "loss_ce", "loss_total",
        "mae", "medae", "lr"
    ])
    writer.writeheader()



  json.dump(NODE_TYPE_MAP, open(models_dir / "op_map.json", "w"), indent=2)
  json.dump(EDGE_TYPE_MAP, open(models_dir / "edge_map.json", "w"), indent=2)
  json.dump(sql_vocab, open(models_dir / "sql_vocab.json", "w"), indent=2)
  json.dump({canonical(s): i for i, s in enumerate(sqls)},
            open(models_dir / "sql2idx.json", "w"), indent=2)

  hint_ids_raw = [
      "",
      "hashjoin,indexonlyscan",
      "hashjoin,indexonlyscan,indexscan",
      "hashjoin,indexonlyscan,indexscan,mergejoin",
      "hashjoin,indexonlyscan,indexscan,mergejoin,nestloop",
      "hashjoin,indexonlyscan,indexscan,mergejoin,nestloop,seqscan",
      "hashjoin,indexonlyscan,indexscan,nestloop",
      "hashjoin,indexonlyscan,indexscan,nestloop,seqscan",
      "hashjoin,indexonlyscan,indexscan,seqscan",
      "hashjoin,indexonlyscan,mergejoin",
      "hashjoin,indexonlyscan,mergejoin,nestloop",
      "hashjoin,indexonlyscan,mergejoin,nestloop,seqscan",
      "hashjoin,indexonlyscan,mergejoin,seqscan",
      "hashjoin,indexonlyscan,nestloop",
      "hashjoin,indexonlyscan,nestloop,seqscan",
      "hashjoin,indexonlyscan,seqscan",
      "hashjoin,indexscan",
      "hashjoin,indexscan,mergejoin",
      "hashjoin,indexscan,mergejoin,nestloop",
      "hashjoin,indexscan,mergejoin,nestloop,seqscan",
      "hashjoin,indexscan,mergejoin,seqscan",
      "hashjoin,indexscan,nestloop",
      "hashjoin,indexscan,nestloop,seqscan",
      "hashjoin,indexscan,seqscan",
      "hashjoin,mergejoin,nestloop,seqscan",
      "hashjoin,mergejoin,seqscan",
      "hashjoin,nestloop,seqscan",
      "hashjoin,seqscan",
      "indexonlyscan,indexscan,mergejoin",
      "indexonlyscan,indexscan,mergejoin,nestloop",
      "indexonlyscan,indexscan,mergejoin,nestloop,seqscan",
      "indexonlyscan,indexscan,mergejoin,seqscan",
      "indexonlyscan,indexscan,nestloop",
      "indexonlyscan,indexscan,nestloop,seqscan",
      "indexonlyscan,mergejoin",
      "indexonlyscan,mergejoin,nestloop",
      "indexonlyscan,mergejoin,nestloop,seqscan",
      "indexonlyscan,mergejoin,seqscan",
      "indexonlyscan,nestloop",
      "indexonlyscan,nestloop,seqscan",
      "indexscan,mergejoin",
      "indexscan,mergejoin,nestloop",
      "indexscan,mergejoin,nestloop,seqscan",
      "indexscan,mergejoin,seqscan",
      "indexscan,nestloop",
      "indexscan,nestloop,seqscan",
      "mergejoin,nestloop,seqscan",
      "mergejoin,seqscan",
      "nestloop,seqscan",
  ]
  hint_ids = sorted(
      set(
          ",".join(sorted(filter(None, c.replace("seqcan", "seqscan").split(","))))
          for c in hint_ids_raw
      )
  )
  json.dump(hint_ids, open(models_dir / "hint_ids.json", "w"), indent=2)


  # embeddings
  embed_path = HOME / "rahhh" / "new" / "embeddings" / "syntaxA_embedding.npy"
  X_np = np.load(embed_path)
  pca = Pipeline([("scl", StandardScaler()), ("pca", PCA(n_components=120))])
  X_pca = torch.from_numpy(pca.fit_transform(X_np)).float()
  joblib.dump(pca, models_dir / "pipeline.pkl")


  obs = np.argwhere(M > 0)
  lat = np.log1p(W[M > 0]) - default_log[obs[:, 0]]
  # robust scaling avoids the “blow-up when tails return” problem
  mu_y  = np.median(lat)
  sig_y = 1.4826 * np.median(np.abs(lat - mu_y))  # ≈ MAD→σ
  ds = PairDS(obs, sqls, sql2path, W, Y, default_log, mu_y, sig_y, sql_vocab)
  sampler = WeightedRandomSampler(np.ones(len(ds)), len(ds), replacement=True)
  dl = DataLoader(ds, batch_size=32, sampler=sampler, collate_fn=collate, num_workers=4)

  best_labels = torch.from_numpy(np.argmin(W, 1)).long()


  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  log_var_pred = torch.nn.Parameter(torch.zeros(()).to(device))
  log_var_rank = torch.nn.Parameter(torch.zeros(()).to(device))

  enc = Plan2VecEncoder(len(NODE_TYPE_MAP),
                        ds[0][1].x.size(1) - 1,
                        len(sql_vocab), 64, 512, 4, 512).to(device)
  fus = GatedFusion(enc.out_dim, 120).to(device)
  head = GNIMCModel(enc.out_dim + 120, Y.shape[1], 288, len(hint_ids)).to(device)
  model = GNIMCWithFusion(enc, fus, head).to(device)
  head.var_raw.bias.data.fill_(math.log(math.expm1(sig_y)))
  head.bias_z.weight.data.zero_()
  head.bias_z.bias.data.fill_(mu_y)
  head.bias_h.weight.data.zero_()
  head.bias_h.bias.data.zero_()

  opt = AdamW([
          {"params": enc.parameters(), "lr": 3e-3},
          {"params": fus.parameters(), "lr": 3e-3},
          {"params": head.U,           "lr": 2e-3},
          {"params": head.V,           "lr": 2e-3},
          {"params": head.bias_z.parameters(), "lr": 1e-1},
          {"params": head.bias_h.parameters(), "lr": 1e-1},
          {"params": head.var_raw.parameters(),"lr": 2e-7},
          {"params": head.classifier.parameters(),"lr": 1e-3},
          {"params": [log_var_pred, log_var_rank], "lr": 1e-3},
      ], weight_decay=1e-5)
      
  scheduler = CosineAnnealingLR(opt, T_max=120, eta_min=1e-6)

  huber_fn = HuberLoss(delta=0.25)
  ce_fn    = CrossEntropyLoss()

  for p in head.var_raw.parameters():
      p.requires_grad = False

  start_epoch = 1
  if args.resume and (models_dir / "gnimc_ckpt.pt").exists():
      ckpt = torch.load(models_dir / "gnimc_ckpt.pt", map_location=device)
      head.load_state_dict(ckpt["model_state_dict"])
      try:  # Plan2Vec encoder state
          enc.load_state_dict(torch.load(models_dir / "plan2vec_ckpt.pt",
                                         map_location=device))
      except Exception:
          pass
      start_epoch = ckpt.get("epoch", 0) + 1
      for _ in range(start_epoch - 1):
          scheduler.step()
      print(f"🔄  Resumed at epoch {start_epoch-1:02d}  (lr={opt.param_groups[0]['lr']:.2e})")
      if start_epoch > FINE_TUNE_AT:
          fine_tune_phase(opt, head)
          scheduler = CosineAnnealingLR(opt, T_max=FINE_TUNE_TMAX, eta_min=5e-6)

  print("✅ operator map size:", len(NODE_TYPE_MAP))
  print("🚀 Training…")


  for ep in range(start_epoch, 115):
      if ep >= 4:
          errs = np.abs(np.array(ds.errors))
          if errs.size > 0 and errs.mean() > 0:
              new_w = 0.75 + 0.25 * (errs / errs.mean())**2
              sampler.weights = torch.tensor(new_w, dtype=torch.double)
      ds.errors.clear()


      model.train()
      sum_pred = sum_rank = sum_ce = sum_tot = 0.0
      abs_errs = []

      for q_idx, g, ids, mask, h, y in dl:
          B   = y.size(0)
          q_idx = q_idx.to(device)
          g     = g.to(device)
          ids   = ids.to(device)
          mask  = mask.to(device)
          h     = h.to(device)
          y     = y.to(device)
          lbl   = best_labels[q_idx].to(device)

          mu, sig, logits = model(g, ids, mask, X_pca, q_idx, h)

          nll = 0.5 * torch.log(sig) + 0.5 * ((mu - y) ** 2) / sig
          err = (y - mu).abs()
          keep = (err < 2.5 * sig).float()      # keep ≈95 % in-liers
          if keep.sum() == 0:                    # all points were flagged as outliers
              loss_pred = nll.mean()             # fallback: use every sample
          else:
              loss_pred = (keep * nll).sum() / keep.sum()
          K = min(5, B)                               # K = 5 or batch size if smaller
          top_idx   = torch.argsort(y)[:K]            # indices of K lowest true latencies
          prob_true = F.softmax(-y[top_idx], dim=0)   # lower latency ⇒ higher prob
          prob_pred = F.softmax(-mu[top_idx], dim=0)
          loss_rank = F.kl_div(prob_pred.log(), prob_true, reduction="batchmean")

          ds.errors.extend((y - mu).abs().cpu().tolist())
          abs_errs.extend((y - mu).abs().cpu().tolist())


          loss_ce = ce_fn(logits, lbl)
          weighted_pred = torch.exp(-log_var_pred) * loss_pred + log_var_pred
          weighted_rank = torch.exp(-log_var_rank) * loss_rank + log_var_rank

          if ep < RANK_START:          # stage-A: learn µ only
              weighted_rank = torch.zeros_like(weighted_rank)

          loss = weighted_pred + weighted_rank + loss_ce     # CE already very small / 0


          opt.zero_grad()
          loss.backward()
          grad_norm = clip_grad_norm_(model.parameters(), 5.0)
          opt.step()


          sum_pred += loss_pred.item() * B
          sum_rank += loss_rank.item() * B
          sum_ce   += loss_ce.item()   * B
          sum_tot  += loss.item()      * B

      mae   = float(np.mean(abs_errs))
      medae = float(np.median(abs_errs))
      lr_now = opt.param_groups[0]["lr"]
      N = len(ds)
      epoch_metrics = {
          "epoch": ep,
          "loss_pred": sum_pred / N,
          "loss_rank": sum_rank / N,
          "loss_ce":   sum_ce   / N,
          "loss_total":sum_tot  / N,
          "mae": mae,
          "medae": medae,
          "lr": lr_now
      }
      print(
         f"Epoch {ep:03d}  lr={lr_now:.2e}  "
         f"pred={epoch_metrics['loss_pred']:.4f}  "
         f"rank={epoch_metrics['loss_rank']:.4f}  "
         f"ce={epoch_metrics['loss_ce']:.4f}  "
         f"total={epoch_metrics['loss_total']:.4f}  "
         f"MAE={epoch_metrics['mae']:.3f}  "
         f"MedAE={epoch_metrics['medae']:.3f}"
      )

      metrics.append(epoch_metrics)

      with open(metrics_csv_path, "a", newline="") as f:
          writer = csv.DictWriter(f, fieldnames=epoch_metrics.keys())
          writer.writerow(epoch_metrics)

      with open(models_dir / "metrics.json", "w") as f:
          json.dump(metrics, f, indent=2)

      ckpt = {
          "epoch": ep,
          "model_state_dict": head.state_dict(),
          "optimizer_state_dict": opt.state_dict(),
          "scheduler_state_dict": scheduler.state_dict()
      }
      torch.save(enc.state_dict(),  models_dir / "plan2vec_ckpt.pt")
      torch.save(ckpt,             models_dir / f"gnimc_ckpt_ep{ep:03d}.pt")

      scheduler.step()

      if ep == FINE_TUNE_AT:
          fine_tune_phase(opt, head)
          scheduler = CosineAnnealingLR(opt, T_max=FINE_TUNE_TMAX, eta_min=5e-6)
      torch.save(enc.state_dict(), models_dir / "plan2vec_ckpt.pt")
      torch.save(
          {"model_state_dict": head.state_dict(), "epoch": ep},
          models_dir / "gnimc_ckpt.pt",
      )
      torch.save(
          {"model_state_dict": head.state_dict(), "epoch": ep},
          models_dir / f"gnimc_ckpt_ep{ep:03d}.pt",
      )

  print("🏁 Done in", round(time.time() - t0, 1), "s")


# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
  print("YES")
  main()


