#!/usr/bin/env python3
import os, sys, json, glob, argparse, pathlib, torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from modules.parse_plan  import parse_plan                          # noqa: E402
from modules.plan2vec    import Plan2VecEncoder                     # noqa: E402

_encoder = None      

def get_encoder(ckpt_path: str) -> Plan2VecEncoder:
    global _encoder
    if _encoder is not None:
        return _encoder

    ckpt = torch.load(ckpt_path, map_location="cpu")
    _encoder = Plan2VecEncoder(
        num_op_types = ckpt["num_op_types"],
        numeric_dim  = ckpt["numeric_dim"],
        vocab_size   = ckpt["vocab_size"],
        text_dim     = 64,
        hidden_dim   = 256,
        num_layers   = 3,
        out_dim      = 256,
    ).eval()
    _encoder.load_state_dict(ckpt["model_state"], strict=True)
    return _encoder


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input-root",  default="~/Downloads/dsb",
                    help="Root folder that contains *.json plans")
    ap.add_argument("--output-root", default="~/Downloads/parsed_dsb",
                    help="Destination folder for *.pt graphs")
    ap.add_argument("--encode", action="store_true",
                    help="also attach 256-D embedding using Plan2Vec checkpoint")
    ap.add_argument("--ckpt", default="models/plan2vec_ckpt.pt",
                    help="Path to encoder checkpoint (only if --encode)")
    args = ap.parse_args()

    INPUT_ROOT  = os.path.expanduser(args.input_root)
    OUTPUT_ROOT = os.path.expanduser(args.output_root)

    json_files = glob.glob(os.path.join(INPUT_ROOT, "**", "*.json"), recursive=True)
    if not json_files:
        print(f"⚠️  No JSON files found under {INPUT_ROOT!r}")
        return

    for src in json_files:
        rel   = os.path.relpath(src, INPUT_ROOT)
        dst   = os.path.splitext(os.path.join(OUTPUT_ROOT, rel))[0] + ".pt"
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        # load JSON (handle list-wrapper formats)
        with open(src, "r") as f:
            raw = json.load(f)
        rec = raw[0] if isinstance(raw, list) else raw

        data = parse_plan(rec)



        if args.encode:
            try:
                enc = get_encoder(os.path.expanduser(args.ckpt))
                with torch.no_grad():
                    # enc expects (Batched) Data; wrap single graph in list
                    emb = enc(data, data.sql_ids.unsqueeze(0), data.sql_mask.unsqueeze(0))
                    data.emb = emb.squeeze(0).cpu()            # shape [256]
            except FileNotFoundError:
                print(f"⚠️  Encoder checkpoint {args.ckpt} not found – skipping embeddings.")

        torch.save(data, dst)
        print(f"✔ Parsed {src} → {dst}")

if __name__ == "__main__":
    main()
