#This is an outdated version of testing. It is kept here purely for reference
import os, sys, json, time, random, re, hashlib, logging, joblib, argparse
from pathlib import Path
from getpass import getuser
from collections import defaultdict
import numpy as np, torch, psycopg2
from sentence_transformers import SentenceTransformer

# For allowing to import modules.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.parse_plan  import parse_plan, canonical as _can
from modules.plan2vec    import Plan2VecEncoder
from modules.train_gnimc import GNIMCModel
from modules.eim         import EmbeddingMemory
from modules.bandit      import ThompsonBandit

argp = argparse.ArgumentParser()
argp.add_argument("--sql-dir", required=True, help="Directory containing *.sql files")
argp.add_argument("--out",     default="./results", help="Where to write run log")
argp.add_argument("--debug",   action="store_true",
                  help="Emit extra diagnostics and capture ‘bad-hint’ rows")
argp.add_argument("--dbg-thresh", type=float, default=2.0,
                  help="Flag when actual_ms > pred_ms × THRESH (default 2)")
ARGS = argp.parse_args()

RAW_SQL_DIR = Path(ARGS.sql_dir).expanduser()
OUT_DIR     = Path(ARGS.out).expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_F = OUT_DIR / "baseline.json"
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
L = logging.getLogger("IMC")
if ARGS.debug:                       # extra channel for anomalies
    BAD = logging.getLogger("IMC.bad")
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
    PIPELINE.transform(
        np.load(str(EMBED_PATH))
    )
).float().to(DEVICE)

Y_scaled = np.load(DATA_ROOT / "Y_scalednew.npz")["Y"]
H_ALL    = torch.from_numpy(Y_scaled).float().to(DEVICE)

W, M = np.load(DATA_ROOT / "Wnew.npy"), np.load(DATA_ROOT / "Mnew.npy")
MU_Y, SIG_Y = np.log1p(W[M > 0]).mean(), np.log1p(W[M > 0]).std()

SQL_TOK_MAXLEN = 128
HINT_TARGETS   = ["hashjoin", "mergejoin", "nestloop"]

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
        hidden_dim=512,
        num_layers=4,
        out_dim=512,
    ).to(DEVICE)
    head = GNIMCModel(
        q_dim=enc.out_dim + 120,
        h_dim=H_ALL.size(1),
        rank=288,          
        num_hints=len(HINT_IDS),
    ).to(DEVICE)

    enc_ck  = torch.load(MODELS_DIR / "plan2vec_ckpt.pt", map_location=DEVICE)
    head_ck = torch.load(MODELS_DIR / "gnimc_ckpt.pt",    map_location=DEVICE)

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


def _toggles_from_hint(hint_combo: str) -> str:

    toggles = ["RESET ALL;"]
    if not hint_combo:
        return " ".join(toggles)
    parts = hint_combo.split(",")
    for h in HINT_TARGETS:
        on_off = "on" if h in parts else "off"
        toggles.append(f"SET enable_{h}={on_off};")
    return " ".join(toggles)

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
    lat_mu = np.expm1((mu_np * SIG_Y) + MU_Y) * 1e3  

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

    # default always present
    arms["default"]     = (default_idx, base_pred_ms)
    mu_sigma["default"] = (base_pred_ms, 0.0)

    # top-K IMC suggestions
    K = 5
    for rank, idx in enumerate(np.argsort(lat_mu)[:K], 1):
        arm_name = f"imc{rank}"
        sigma_ms = np.expm1(sig_np[idx] * SIG_Y) * 1e3
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
        actual_base_ms=true_base_ms,   # pulled from caller’s global
        z=z.cpu().numpy(),
    )
    hint_idx      = arms[arm][0]
    pred_mu_ms    = float(mu_sigma.get(arm, (arms[arm][1], 0.0))[0])
    pred_sigma_ms = float(mu_sigma.get(arm, (0.0, 0.0))[1])
    return (HINT_IDS[hint_idx], z, qkey, arm,
            base_pred_ms, pred_mu_ms, pred_sigma_ms)

BEST_HINT: dict[str, tuple[int, float]] = {}

debug_rows = [] if ARGS.debug else None

try:
    for sql_file in sorted(RAW_SQL_DIR.rglob("*.sql")):
        sql_text = sql_file.read_text()
        qkey     = _hash(sql_text)

        cnt = baseline_map.get(qkey, {}).get("count", 0) + 1
        if cnt == 1 or cnt % 100 == 0 or \
            (time.time() - baseline_map.get(qkey, {}).get("ts", 0) > 86_400):
            t0 = time.time()                    
            cur.execute(sql_text)
            b_ms = (time.time() - t0) * 1e3
            baseline_map[qkey] = {"lat": b_ms, "count": cnt, "ts": time.time()}
            with open(BASELINE_F, "w") as f:
                json.dump(baseline_map, f)
        true_base_ms = baseline_map[qkey]["lat"] * calib["c"]

        (hint_combo, z_vec, qkey, arm,
         base_pred_ms, pred_ms, pred_sigma) = choose_hint(sql_text)
        SPEEDUP_THRESHOLD = 1.08
        if base_pred_ms / pred_ms < SPEEDUP_THRESHOLD:
            hint_combo   = ""
            arm          = "default"
            pred_ms      = 0.0
            pred_sigma   = 0.0

        toggles = _toggles_from_hint(hint_combo)
        t0 = time.time()
        cur.execute(toggles + " " + sql_text)
        lat_ms = (time.time() - t0) * 1e3

        # feedback
        idx_hint = HINT_IDS.index(hint_combo)
        ok_base  = lat_ms <= true_base_ms * 1.05
        ok_pred  = lat_ms <= pred_ms * ARGS.dbg_thresh
        ok       = ok_base                      # contract with user

        EIM.update_stats(z_vec.cpu().numpy(), idx_hint, lat_ms, ok)
        if ok:                                  # store only good outcomes
            EIM.add(z_vec.cpu().numpy(), idx_hint, lat_ms, ok)
        BANDIT.update(
            qkey,
            arm,
            lat_ms,
            true_base_ms,
            z=z_vec.cpu().numpy(),
        )

        if ok and (qkey not in BEST_HINT or lat_ms < BEST_HINT[qkey][1]):
            BEST_HINT[qkey] = (idx_hint, lat_ms)

        if ARGS.debug and not ok_pred:
            rec = {
                "sql_file":   sql_file.name,
                "arm":        arm,
                "hint_combo": hint_combo,
                "lat_ms":     lat_ms,
                "pred_ms":    pred_ms,
                "pred_sigma": pred_sigma,
                "ratio":      lat_ms / max(pred_ms, 1e-6),
                "base_ms":    true_base_ms,
            }
            debug_rows.append(rec)
            BAD.warning(f"[BAD] {rec}")

        msg = (f"{sql_file.name:20s}  arm={arm:5s} "
               f"hints=({hint_combo or 'none':30s})  "
               f"lat={lat_ms:7.2f} ms   base={base_pred_ms:7.2f}")
        L.info(msg)


finally:
    cur.close(); conn.close()
    if ARGS.debug and debug_rows:
        dbg_f = OUT_DIR / "debug.jsonl"
        with open(dbg_f, "w") as f:
            for r in debug_rows:
                f.write(json.dumps(r) + "\n")
        L.info(f"⚠️  wrote {len(debug_rows)} anomalous rows → {dbg_f}")
