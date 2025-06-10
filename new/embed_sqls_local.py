
import os
import json
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def load_sqls(plan_root: Path):
    sqls = []
    seen = set()
    for path in plan_root.rglob("*.json"):
        recs = json.load(open(path))
        recs = recs if isinstance(recs, list) else [recs]
        for rec in recs:
            q = rec.get("sql", "").strip()
            if q and q not in seen:
                seen.add(q)
                sqls.append(q)
    return sqls

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan-root", type=Path, required=True)
    p.add_argument("--out-dir",   type=Path, default=Path("./embeddings"))
    p.add_argument("--model",     type=str, default="all-MiniLM-L6-v2")
    args = p.parse_args()

    sqls = load_sqls(args.plan_root)
    print(f"Found {len(sqls)} distinct SQL queries under {args.plan_root!r}.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "syntaxA_embedding.npy"

    print(f"Loading SentenceTransformer(\"{args.model}\")…")
    model = SentenceTransformer(args.model)

    print("Computing embeddings in batches…")
    embeds = model.encode(sqls, batch_size=64, show_progress_bar=True)
    X = np.array(embeds, dtype=np.float32)

    np.save(out_path, X)
    print(f"Saved local embeddings to {out_path!r} (shape {X.shape}).")

if __name__ == "__main__":
    main()
