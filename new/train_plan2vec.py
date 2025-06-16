#Used for training plan2vec model (need pt files)

import os, sys, glob, argparse
from pathlib import Path
from typing import List

import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader as GeoLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from modules.plan2vec import Plan2VecEncoder    

class PlanDataset(Dataset):
    def __init__(self, root: str):
        super().__init__(root)
        self.files: List[str] = sorted(
            glob.glob(os.path.join(root, "**", "*.pt"), recursive=True)
        )
        if not self.files:
            raise RuntimeError(f"No .pt files found under {root!r}")

    def len(self) -> int:
        return len(self.files)

    def get(self, idx):
        return torch.load(self.files[idx], weights_only=False)  # full Data

def mse_loss(pred, tgt): return ((pred.squeeze(-1) - tgt.squeeze(-1)) ** 2).mean()

def train(opts):
    dl = GeoLoader(PlanDataset(opts.data_root),
                   batch_size=opts.batch_size,
                   shuffle=True,
                   num_workers=opts.workers)

    sample = next(iter(dl))
    opts.numeric_dim = sample.x.size(1) - 1
    print(f"[*] Detected numeric_dim = {opts.numeric_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = Plan2VecEncoder(
        num_op_types = opts.num_op_types,
        numeric_dim  = opts.numeric_dim,
        vocab_size   = opts.vocab_size,
        text_dim     = opts.text_dim,
        hidden_dim   = opts.hidden_dim,
        num_layers   = opts.num_layers,
        out_dim      = 1,
    ).to(device)

    opt   = torch.optim.AdamW(model.parameters(), lr=opts.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=opts.epochs)

    for epoch in range(1, opts.epochs + 1):
        model.train(); epoch_loss = 0.0
        for batch in dl:
            batch = batch.to(device)
            y     = batch.y                                

            if hasattr(batch, "sql_ids") and batch.sql_ids.ndim == 2:
                sql_ids  = batch.sql_ids.to(device)
                sql_mask = batch.sql_mask.to(device)
            else:
                B        = batch.num_graphs
                sql_ids  = torch.zeros((B, 1), dtype=torch.long, device=device)
                sql_mask = torch.zeros_like(sql_ids)

            if model.edge_embed.num_embeddings == 1:
                batch.edge_attr = torch.zeros_like(batch.edge_attr)

            pred = model(batch, sql_ids, sql_mask)
            loss = mse_loss(pred, y)

            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item() * batch.num_graphs

        sched.step()
        avg = epoch_loss / len(dl.dataset)
        print(f"[{epoch:02d}/{opts.epochs}] loss = {avg:.4f}")

    ckpt = Path(opts.ckpt); ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(),
                "num_op_types": opts.num_op_types,
                "numeric_dim" : opts.numeric_dim,
                "vocab_size"  : opts.vocab_size},
               ckpt)
    print(f"✓ saved checkpoint → {ckpt}")

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", required=True, help="Directory of *.pt graphs")
    p.add_argument("--ckpt",      default="models/plan2vec_ckpt.pt")
    p.add_argument("--num-op-types", type=int, default=47)
    p.add_argument("--vocab-size",   type=int, default=8000)
    p.add_argument("--text-dim",     type=int, default=64)
    p.add_argument("--hidden-dim",   type=int, default=256)
    p.add_argument("--num-layers",   type=int, default=3)

    p.add_argument("--batch-size",   type=int, default=64)
    p.add_argument("--epochs",       type=int, default=5)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--workers",      type=int, default=4)
    return p.parse_args()

if __name__ == "__main__":
    train(parse_args())
