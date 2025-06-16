# Correct script fo training
import os, inspect, sys, json, time, random, re, hashlib, logging, argparse, joblib, csv, statistics, re
from pathlib import Path
from getpass import getuser
from collections import defaultdict

import numpy as np, torch, psycopg2
import pandas as pd
from sentence_transformers import SentenceTransformer
import matplotlib      
matplotlib.use("Agg")    
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA_ROOT   = Path.home() / "New_Data"
W, M        = np.load(DATA_ROOT/"Wnew.npy"), np.load(DATA_ROOT/"Mnew.npy")
default_log = np.log1p(W[:, 0])                          # log1p(seconds)
obs    = np.argwhere(M > 0)                              # shape (N,2): (q_idx,h_idx)
deltas = np.log1p(W[M > 0]) - default_log[obs[:, 0]]     # lift off Δlog1p(lat_sec)
mu_y_train  = np.median(deltas)
sig_y_train = 1.4826 * np.median(np.abs(deltas - mu_y_train))

from modules.parse_plan  import parse_plan, canonical as _can
from modules.plan2vec    import Plan2VecEncoder
from modules.train_gnimc import GNIMCModel
from modules.eim         import EmbeddingMemory
from modules.bandit      import ThompsonBandit
print(f"[probe] running file: {__file__}")
print(f"[probe] argv seen by the script: {sys.argv!r}")

src = inspect.getsource(sys.modules[__name__])
opts = set(re.findall(r'--[A-Za-z0-9\-\u2010-\u2015]+', src))
print("[probe] option strings embedded in the file:")
for o in sorted(opts):
    codes = " ".join(f"{ord(c):04X}" for c in o)
    print(f"  {o:<15}  {codes}")
print()  

# Adapt to your directory
RAW_SQL_DIR = Path.home() / "rahhh" / "Data_Gathering" / "raw_sql_queries"
OUT_DIR     = Path.home() / "rahhh" / "new" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


DEBUG      = False   
DBG_THRESH = 3.0      
NO_CSV     = False   
RESUME     = False    
SAFETY_THRESH = 0.95   
BASELINE_CUTOFF_MIN_MS = 13_300
BASELINE_CUTOFF_MAX_MS = 20_000
print(f"RAW_SQL_DIR={RAW_SQL_DIR}")
print(f"OUT_DIR={OUT_DIR}")

BASELINE_F   = OUT_DIR / "baseline.json"
RESULTS_CSV  = OUT_DIR / "results.csv"
SUMMARY_CSV  = OUT_DIR / "summary.csv"
QUERY_RESULTS_CSV = OUT_DIR / "query_results.csv"

if not QUERY_RESULTS_CSV.exists():
    with open(QUERY_RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "query_file",
            "variant", "arm", "hint_combo",
            "lat_ms", "baseline_ms", "speedup"
        ])

try:
    baseline_map = json.load(open(BASELINE_F))
except FileNotFoundError:
    baseline_map = {}

calib = {"alpha": 0.1, "c": 1.0}

torch.manual_seed(0); np.random.seed(0); random.seed(0)

log_lvl = logging.DEBUG if os.getenv("IMC_DEBUG") else logging.INFO
logging.basicConfig(level=log_lvl,
                    format="%(levelname)s: %(message)s",
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler(OUT_DIR / "run.log", mode="w")
                    ])
L   = logging.getLogger("IMC")
BAD = logging.getLogger("IMC.bad")
if DEBUG:
    BAD.setLevel(logging.WARNING)

MODELS_DIR = Path.home() / "models"
DATA_ROOT  = Path.home() / "New_Data"
PLAN_ROOT  = Path.home() / "Downloads" / "dsb"

HINT_IDS  = json.load(open(MODELS_DIR / "hint_ids.json"))
SQL2IDX   = json.load(open(MODELS_DIR / "sql2idx.json"))
PIPELINE  = joblib.load(MODELS_DIR / "pipeline.pkl")
OP_MAP    = json.load(open(MODELS_DIR / "op_map.json"))
SQL_VOCAB = json.load(open(MODELS_DIR / "sql_vocab.json"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMBED_PATH = Path.home() / "rahhh" / "new" / "embeddings" / "syntaxA_embedding.npy"
embed_train = torch.from_numpy(
    PIPELINE.transform(np.load(str(EMBED_PATH)))
).float().to(DEVICE)

Y_scaled = np.load(DATA_ROOT / "Y_scalednew.npz")["Y"]
H_ALL    = torch.from_numpy(Y_scaled).float().to(DEVICE)

SQL_TOK_MAXLEN = 128
HINT_TARGETS   = [
    "hashjoin", "indexonlyscan", "indexscan",
    "mergejoin", "nestloop", "seqscan",
]


def _build_models():
    sample_plan = parse_plan(
        json.load(open(next(PLAN_ROOT.rglob("*.json"))))
    )
    numeric_dim = sample_plan.x.size(1) - 1

    enc = Plan2VecEncoder(
        num_op_types=len(OP_MAP),
        numeric_dim=numeric_dim,
        vocab_size=len(SQL_VOCAB),
        text_dim=64,
        hidden_dim=512,      # ⟵ make sure to match training run
        num_layers=4,        # ⟵ make sure to match training run
        out_dim=512,
    ).to(DEVICE)
    head = GNIMCModel(
        q_dim=enc.out_dim + 120,
        h_dim=H_ALL.size(1),
        rank=288,            # ⟵ make sure to match training run
        num_hints=len(HINT_IDS),
    ).to(DEVICE)
    epoch = 110 # Choose your best performing epoch
    enc_ck  = torch.load(MODELS_DIR / "plan2vec_ckpt.pt", map_location=DEVICE)
    head_ck = torch.load(
        MODELS_DIR / f"gnimc_ckpt_ep{epoch:03d}.pt",
        map_location=DEVICE
    )

    if enc_ck["edge_embed.weight"].size(0) != enc.edge_embed.weight.size(0):
        new_sz   = enc_ck["edge_embed.weight"].size(0)
        emb_dim  = enc.edge_embed.weight.size(1)
        enc.edge_embed = torch.nn.Embedding(new_sz, emb_dim).to(DEVICE)

    enc.load_state_dict(enc_ck, strict=False)
    head.load_state_dict(head_ck["model_state_dict"])
    enc.eval(); head.eval()

    return enc, head

ENC, HEAD = _build_models()

EIM    = EmbeddingMemory(dim=ENC.out_dim + 120, tau=8.0)
BANDIT = ThompsonBandit()

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def _pca_vec(sql: str) -> torch.Tensor:
    can = _can(sql)
    if can in SQL2IDX:
        return embed_train[SQL2IDX[can]: SQL2IDX[can] + 1]
    vec = embedder.encode([sql], normalize_embeddings=True)
    return torch.from_numpy(PIPELINE.transform(vec)).float().to(DEVICE)

def _sql_tokens(sql: str):
    toks = re.findall(r"\w+", sql.lower())[:SQL_TOK_MAXLEN]
    ids  = [SQL_VOCAB.get(t, 1) for t in toks] + [0]*(SQL_TOK_MAXLEN-len(toks))
    mask = [1]*len(toks) + [0]*(SQL_TOK_MAXLEN-len(toks))
    return (
        torch.tensor([ids],  dtype=torch.long,   device=DEVICE),
        torch.tensor([mask], dtype=torch.float32,device=DEVICE),
    )

def _hash(sql: str) -> str:
    return hashlib.sha1(sql.encode()).hexdigest()

def _toggles_from_hint(hint_combo: str, force_disable: bool = False) -> str:

    parts = {h.strip() for h in hint_combo.split(",") if h.strip()}
    cmds = [f"SET enable_{h}=on;" for h in parts]
    if force_disable:
        cmds += [f"SET enable_{h}=off;"
                 for h in HINT_TARGETS if h not in parts]
    return " ".join(cmds)

# Use your PostgreSQL connection
conn = psycopg2.connect(
    dbname="database name",
    user=getuser(),
    host="host",
    port="password",
    password=os.getenv("PGPASSWORD", ""),
)
conn.autocommit = True
cur = conn.cursor()

def _parse_explain(raw):
    obj = json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else raw
    return obj[0] if isinstance(obj, list) else obj

if not NO_CSV and not RESULTS_CSV.exists():
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "query_file", "variant", "arm", "hint_combo",
            "lat_ms", "baseline_ms", "speedup", "pred_mu_ms", "pred_sigma_ms",
            "ok_base", "ok_pred"
        ])


def choose_hint(sql_text: str):
    qkey = _hash(sql_text)

    cur.execute(f"EXPLAIN (FORMAT JSON) {sql_text}")
    g = parse_plan(_parse_explain(cur.fetchone()[0])).to(DEVICE)
    g.batch = torch.zeros(g.x.size(0), dtype=torch.long, device=DEVICE)
    sql_ids, sql_mask = _sql_tokens(sql_text)

    with torch.no_grad():
        z_core = ENC(g, sql_ids, sql_mask)
    z = torch.cat([z_core, _pca_vec(sql_text)], dim=1)

    arms, mu_sigma = {}, {}
    with torch.no_grad():
        mu, sig, _ = HEAD(z, H_ALL)
    mu_np, sig_np = mu.cpu().numpy(), sig.cpu().numpy()
    log_base = np.log1p(true_base_ms / 1e3)               # sec-domain
    log_pred = mu_np * sig_y_train + mu_y_train + log_base
    lat_mu   = np.expm1(log_pred) * 1e3                  # back to ms
    lat_mu   = BANDIT._cal(lat_mu)
    idx      = int(np.argsort(lat_mu)[0])

    default_idx  = HINT_IDS.index("")
    base_pred_ms = float(lat_mu[0])

    n_idx, n_lat = EIM.query(z.cpu().numpy())
    if n_idx is not None and n_idx != default_idx and n_lat < base_pred_ms:
        arms["cache"]     = (n_idx, n_lat)
        mu_sigma["cache"] = (float(n_lat), 0.0)

    kn_idx, kn_lat = EIM.best_of_knn(z.cpu().numpy(), k=10)
    if (
        kn_idx is not None and len(EIM._vals) >= 8
        and kn_lat < base_pred_ms * 0.95 and kn_idx != default_idx
    ):
        arms["knn"]       = (kn_idx, kn_lat)
        mu_sigma["knn"]   = (float(kn_lat), 0.0)

    arms["default"]     = (default_idx, base_pred_ms)
    mu_sigma["default"] = (base_pred_ms, 0.0)

    K = 5
    for rank, idx in enumerate(np.argsort(lat_mu)[:K], 1):
        arm_name = f"imc{rank}"
        sigma_ms = np.expm1(sig_np[idx] * sig_y_train) * 1e3
        penalty  = 1 + 0.5 * (sigma_ms / max(lat_mu[idx], 1e-3))
        arms[arm_name]     = (int(idx), float(lat_mu[idx] * penalty))
        mu_sigma[arm_name] = (float(lat_mu[idx]), float(sig_np[idx]))

    if qkey in BEST_HINT:
        arms["best"] = BEST_HINT[qkey]

    arm = BANDIT.choose(
        qkey,
        arms,
        base_pred_ms,
        mu_sigma,
        actual_base_ms=true_base_ms,  
        z=z.cpu().numpy(),
    )
    hint_idx      = arms[arm][0]
    pred_mu_ms    = float(mu_sigma.get(arm, (arms[arm][1], 0.0))[0])
    pred_sigma_ms = float(mu_sigma.get(arm, (0.0, 0.0))[1])

    return (HINT_IDS[hint_idx], z, qkey, arm,
            base_pred_ms, pred_mu_ms, pred_sigma_ms)

BEST_HINT: dict[str, tuple[int, float]] = {}

all_lat      = defaultdict(list)         
all_base_lat = defaultdict(list)        
def run_variant(sql_text: str,
                variant: str,
                true_base_ms: float,
                sql_file_name: str):
    z_vec = None                    

    if variant == "default":
        toggles     = ""             
        arm         = "default"
        hint_combo  = ""
        pred_mu_ms  = pred_sigma_ms = 0.0


    else:  
        (hint_combo, z_vec, qkey, arm,
         base_pred_ms, pred_mu_ms, pred_sigma_ms) = choose_hint(sql_text)
        toggles = _toggles_from_hint(hint_combo)
    if variant == "full":
        if (true_base_ms < BASELINE_CUTOFF_MIN_MS
                or true_base_ms > BASELINE_CUTOFF_MAX_MS):
            force_disable = False
        else:
            force_disable = (pred_mu_ms < true_base_ms * SAFETY_THRESH)
        toggles = _toggles_from_hint(hint_combo, force_disable)


    cur.execute("RESET ALL")

    t0 = time.time()
    cur.execute(toggles + " " + sql_text)
    lat_ms = (time.time() - t0) * 1e3

    ok_base = lat_ms <= true_base_ms * 1.05
    ok_pred = (variant == "full"  # only meaningful when we have a model μ
               and lat_ms <= pred_mu_ms * DBG_THRESH) or variant != "full"

    return (variant, arm, hint_combo, lat_ms,
            pred_mu_ms, pred_sigma_ms, ok_base, ok_pred, z_vec)

debug_rows = [] if DEBUG else None

try:
    VARIANTS = ("default", "full")     # ← fixed order

    for sql_file in sorted(RAW_SQL_DIR.rglob("*.sql")):
        sql_text = sql_file.read_text()
        qkey     = _hash(sql_text)

        cnt = baseline_map.get(qkey, {}).get("count", 0) + 1
        if cnt == 1 or cnt % 100 == 0 or (
            time.time() - baseline_map.get(qkey, {}).get("ts", 0) > 86_400
        ):
            cur.execute("RESET ALL")
            t0 = time.time()
            cur.execute(sql_text)                       # DEFAULT PLAN
            b_ms = (time.time() - t0) * 1e3
            baseline_map[qkey] = {"lat": b_ms,
                                  "count": cnt,
                                  "ts": time.time()}
            with open(BASELINE_F, "w") as f:
                json.dump(baseline_map, f)

        true_base_ms = baseline_map[qkey]["lat"] * calib["c"]

        for variant in VARIANTS:
            (variant, arm, hint_combo, lat_ms,
            pred_mu_ms, pred_sigma_ms, ok_base, ok_pred, z_vec) = \
                run_variant(sql_text, variant,
                            true_base_ms, sql_file.name)

            if variant == "full":
                idx_hint = HINT_IDS.index(hint_combo)
                ok       = ok_base
                if z_vec is not None:                 # safety: should always be true here
                    EIM.update_stats(z_vec.cpu().numpy(), idx_hint, lat_ms, ok)
                    if ok:
                        EIM.add(z_vec.cpu().numpy(), idx_hint, lat_ms, ok)
                    BANDIT.update(qkey, arm, lat_ms, true_base_ms,
                                z=z_vec.cpu().numpy())

            tag = variant                     
            all_lat      .setdefault(tag, []).append(lat_ms)
            all_base_lat .setdefault(tag, []).append(true_base_ms)

            if not NO_CSV:
                speedup = lat_ms / max(true_base_ms, 1e-6)
                with open(QUERY_RESULTS_CSV, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        int(time.time()), sql_file.name,
                        variant, arm, hint_combo,
                        f"{lat_ms:.2f}", f"{true_base_ms:.2f}",
                        f"{speedup:.4f}"
                    ])

            L.info(f"{sql_file.name:20s}  [{variant}] "
                   f"arm={arm:6s}  "
                   f"hints=({hint_combo or 'none':30s})  "
                   f"lat={lat_ms:7.2f} ms   base={true_base_ms:7.2f}")

finally:
    cur.close(); conn.close()
    try:
        df = pd.read_csv(QUERY_RESULTS_CSV)
        df.to_csv(OUT_DIR / "query_results.csv", index=False)
        L.info(f"💾 wrote query results → {OUT_DIR/'query_results.csv'}")

        matplotlib.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'serif'],
            'pdf.fonttype': 42,    # embed fonts in vector outputs
            'ps.fonttype': 42,
        })
        SPINE_KWARGS = {'linewidth': 1.2, 'color': 'black'}

        summary = (
            df
            .groupby("variant").speedup.agg(
                num_queries        = "count",
                mean_speedup       = "mean",
                p99_speedup        = lambda s: s.quantile(0.99),
                regressions_gt20pct= lambda s: (s > 1.20).sum(),
                worst_regression   = "max"
            )
            .reset_index()
        )
        summary.to_csv(OUT_DIR / "summary.csv", index=False)
        L.info(f"📊 wrote summary → {OUT_DIR/'summary.csv'}")

        totals = df.assign(lat_s=lambda d: d.lat_ms/1000.0) \
                   .groupby("variant").lat_s.sum()
        fig, ax = plt.subplots(figsize=(4,3.5))
        totals.plot.bar(ax=ax, edgecolor='black', linewidth=1.5)
        ax.set_xlabel("Variant", fontsize=12, fontweight='bold')
        ax.set_ylabel("Total Wall-Time [s]", fontsize=12, fontweight='bold')
        ax.set_title("Cumulative Runtime Per Variant", fontsize=14, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_linewidth(SPINE_KWARGS['linewidth'])
            spine.set_color(SPINE_KWARGS['color'])
        plt.tight_layout()
        plt.savefig(OUT_DIR / "total_latency_bar.png", dpi=300)
        plt.close(fig)
        L.info("🖼️  wrote total_latency_bar.png")

        perc = (
            df.groupby("variant").lat_ms
              .quantile([0.50, 0.95])
              .unstack(level=-1)
              .divide(1000.0)
        )
        perc_T = perc.T.copy()
        perc_T.index = [f"{int(q*100)}th Percentile" for q in perc_T.index]
        perc_T.columns = [v.title() for v in perc_T.columns]
        fig, ax = plt.subplots(figsize=(4,3.5))
        perc_T.plot.bar(ax=ax, edgecolor='black', linewidth=1.5)
        ax.set_xlabel("Percentile", fontsize=12, fontweight='bold')
        ax.set_ylabel("Latency [s]", fontsize=12, fontweight='bold')
        ax.set_title("Latency By Percentile", fontsize=14, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_linewidth(SPINE_KWARGS['linewidth'])
            spine.set_color(SPINE_KWARGS['color'])
        leg = ax.legend(title="", fontsize=10)
        leg.get_frame().set_linewidth(0)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "percentile_latency_bar.png", dpi=300)
        plt.close(fig)
        L.info("🖼️  wrote percentile_latency_bar.png")

    except Exception as e:
        L.error(f"Failed to regenerate summary/charts: {e}")
    if NO_CSV:
        sys.exit(0)
