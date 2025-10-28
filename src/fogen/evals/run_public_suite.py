"""Probe battery over public checkpointed suites (Pythia/OLMo).

Pythia exposes checkpoints as HF revisions: step0, step1, ..., step143000.
  python -m fogen.evals.run_public_suite --model EleutherAI/pythia-70m \
      --battery data/probes/v1/battery.jsonl --out results/pythia70m \
      --steps 0,512,1000,2000,4000,8000,16000,33000,66000,99000,143000
"""

import argparse
import json
from pathlib import Path

import torch

from fogen.evals.scoring import aggregate, hf_scorer, load_battery


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--battery", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", required=True,
                    help="comma-separated checkpoint steps (HF revision step<N>)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    battery = load_battery(args.battery)
    tok = AutoTokenizer.from_pretrained(args.model)

    for step in [int(s) for s in args.steps.split(",")]:
        rev = f"step{step}"
        dst = out / f"step{step:07d}.jsonl"
        if dst.exists():
            print(f"skip {rev} (exists)")
            continue
        print(f"loading {args.model}@{rev}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=rev, torch_dtype=torch.float32)
        sc = hf_scorer(model, tok, device=args.device)
        rows = sc.score_items(battery)
        with dst.open("w") as f:
            for r in rows:
                f.write(json.dumps({"model": args.model, "step": step, **r}) + "\n")
        for a in aggregate(rows):
            print(f"  {a['probe']:28s} {a['split']:8s} acc={a['argmax_acc']:.2f}")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
