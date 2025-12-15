"""Validation bits-per-byte for a checkpoint — the v1 fidelity gate.

v1 paper anchor: val_bpb in [1.149, 1.152] for all seeds. Metric matches
nanochat's evaluate_bpb (see fogen.evals.bpb).

Usage:
  python scripts/eval_bpb.py --ckpt runs/v1_repro_s42/ckpts/step004500.safetensors \
      [--config runs/v1_repro_s42/config_used.yaml] [--max-windows 2048]
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from fogen.data import load_tokenizer
from fogen.evals.bpb import evaluate_bpb, token_byte_table, val_stream
from fogen.model import GPT, ModelConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None,
                    help="defaults to config_used.yaml two dirs above ckpt")
    ap.add_argument("--val-shards", default="data/tinystories/bpe8192/val_shards")
    ap.add_argument("--tokenizer-dir", default="data/tinystories/bpe8192")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--doc-aligned", action="store_true",
                    help="one doc per window from position 0 (in-distribution "
                         "for one-doc-trained models); --max-windows caps docs")
    ap.add_argument("--out", default=None, help="append JSON result to this file")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    cfg_path = Path(args.config) if args.config else ckpt.parent.parent / "config_used.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPT(ModelConfig(**cfg["model"])).to(device)
    from safetensors.torch import load_file
    state = {k: v.float() for k, v in load_file(str(ckpt)).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith("rope_") for k in missing), missing
    model.eval()

    tokenizer = load_tokenizer(args.tokenizer_dir)
    tb = token_byte_table(tokenizer)
    assert tb[0] == 0, "<|endoftext|> must have 0 bytes (masked from bpb)"
    stream = val_stream(args.val_shards)
    print(f"val stream: {len(stream):,} tokens; ckpt: {ckpt.name}")

    if args.doc_aligned:
        from fogen.evals.bpb import evaluate_bpb_docs
        res = evaluate_bpb_docs(model, stream, tb, cfg["model"]["ctx_len"],
                                batch_size=args.batch_size,
                                max_docs=args.max_windows, device=device)
    else:
        res = evaluate_bpb(model, stream, tb, cfg["model"]["ctx_len"],
                           batch_size=args.batch_size,
                           max_windows=args.max_windows, device=device)
    res = {"ckpt": str(ckpt), **res}
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
