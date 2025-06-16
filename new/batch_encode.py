#!/usr/bin/env python3

import os
import sys
import argparse
from torch_geometric.data import Batch

import torch
import numpy as np
def configure_environment():
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    scripts_path = os.path.join(project_root, "scripts")

    if scripts_path in sys.path:
        sys.path.remove(scripts_path)

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

#Here bec problems with env configurations when importing modules
configure_environment()

from modules.plan2vec import Plan2VecEncoder

def main(input_dir: str, output_dir: str, model_ckpt: str, device: str):
    print(f"[DEBUG] Input directory: {input_dir}")
    if not os.path.isdir(input_dir):
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        sys.exit(1)

    print(f"[DEBUG] Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[DEBUG] Checkpoint path: {model_ckpt}")
    ckpt_exists = os.path.exists(model_ckpt)
    print(f"[DEBUG] Checkpoint exists: {ckpt_exists}")


    dev = torch.device(device if (device == 'cuda' and torch.cuda.is_available()) else 'cpu')

    ckpt = torch.load(model_ckpt, map_location=dev)
    encoder = Plan2VecEncoder(
        num_op_types = ckpt["num_op_types"],
        numeric_dim  = ckpt["numeric_dim"],
        vocab_size   = ckpt["vocab_size"],
        text_dim     = 64,
        hidden_dim   = 256,
        num_layers   = 3,
        out_dim      = ckpt["model_state"]["mlp.3.bias"].numel(),
    ).to(dev).eval()
    encoder.load_state_dict(ckpt["model_state"], strict=True)
    print(f"✅ Loaded Plan2VecEncoder from {model_ckpt}")

    pt_files = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if fn.lower().endswith('.pt'):
                pt_files.append(os.path.join(root, fn))
    print(f"[DEBUG] Found {len(pt_files)} .pt files to process")
    if not pt_files:
        print("[ERROR] No .pt Data files found in input directory. Exiting.")
        return

    for in_path in pt_files:
        rel_dir = os.path.relpath(os.path.dirname(in_path), input_dir)
        out_subdir = os.path.join(output_dir, rel_dir)
        os.makedirs(out_subdir, exist_ok=True)

        base = os.path.splitext(os.path.basename(in_path))[0]
        out_path = os.path.join(out_subdir, base + '.npy')
        print(f"[DEBUG] Loading {in_path} -> {out_path}")

        try:
            data = torch.load(in_path, weights_only=False)
        except TypeError:
            data = torch.load(in_path)
        batch = Batch.from_data_list([data]).to(dev)

        # 2) These are dummies - can be deleted
        sql_ids  = torch.zeros((1, 1), dtype=torch.long, device=dev)
        sql_mask = torch.ones((1, 1), dtype=torch.float, device=dev)

        try:
            with torch.no_grad():
                z = encoder(batch, sql_ids, sql_mask)  # → shape [1, out_dim]
        except Exception as e:
            print(f"[ERROR] Encoding failed for {in_path}: {e}")
            continue

        arr = z.squeeze(0).cpu().numpy()
        try:
            np.save(out_path, arr)
            print(f"✅ Saved embedding to {out_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save embedding for {in_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch-encode parsed plan Data objects with Plan2VecEncoder"
    )
    parser.add_argument("--input_dir", required=True,
                        help="Root folder of your .pt Data files")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write .npy embedding files")
    parser.add_argument("--model_ckpt", required=True,
                        help="Path to plan2vec checkpoint (or fresh if missing)")
    parser.add_argument("--device", default="cpu",
                        help="‘cuda’ or ‘cpu’; uses cuda only if available")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.model_ckpt, args.device)
