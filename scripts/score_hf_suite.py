"""Score an rvp battery across a public model's training checkpoints.

Writes probe_log_<tag>.jsonl in the same format as fogen training runs
("step" = checkpoint revision step), so scripts/gate_a_classify.py works
unchanged on public-suite trajectories (Pythia revisions: step0, step1,
..., step512 log-spaced, then step1000..step143000).

  python scripts/score_hf_suite.py --model EleutherAI/pythia-70m \
      --battery data/probes/rvp3/battery.jsonl --tag rvp3 \
      --out runs/pythia-70m --revisions auto
"""

import argparse
import json
import re
from pathlib import Path

import torch

from fogen.evals.scoring import aggregate, hf_scorer, load_battery


def auto_revisions() -> list[str]:
    """Pythia public grid, thinned: log2 to 512, then every 5k, plus final."""
    steps = [0] + [2 ** k for k in range(10)] + list(range(1000, 143000, 5000))
    steps.append(143000)
    return [f"step{s}" for s in sorted(set(steps))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--battery", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--revisions", default="auto",
                    help='"auto" or comma-separated revision names')
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--per-template", action="store_true")
    ap.add_argument("--scratch-cache", default=None,
                    help="HF cache dir wiped after each revision (bounds "
                         "disk for large models; ~full re-download per rev)")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    revs = auto_revisions() if args.revisions == "auto" else args.revisions.split(",")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    battery = load_battery(args.battery)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / f"probe_log_{args.tag}.jsonl").open("a")
    tlog = (out / f"probe_log_{args.tag}_templates.jsonl").open("a") \
        if args.per_template else None

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    for rev in revs:
        # leading step number only: OLMo revisions look like
        # "step99000-tokens415B" and must not absorb the token count
        step = int(re.match(r"step(\d+)", rev).group(1))
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, revision=rev, dtype=torch.float32,
                cache_dir=args.scratch_cache)
        except Exception as e:
            print(f"skip {rev}: {e}")
            continue
        scorer = hf_scorer(model, tokenizer, device=device,
                           batch_size=args.batch_size)
        rows = scorer.score_items(battery)
        for a in aggregate(rows):
            log.write(json.dumps({"step": step, **a}) + "\n")
        log.flush()
        if tlog:
            from collections import defaultdict
            groups = defaultdict(list)
            for r in rows:
                groups[(r["probe"], r["template_id"])].append(r["argmax_acc"])
            for (probe, tid), accs in sorted(groups.items()):
                tlog.write(json.dumps({
                    "step": step, "probe": probe, "template_id": tid,
                    "n": len(accs), "argmax_acc": sum(accs) / len(accs)}) + "\n")
            tlog.flush()
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        if args.scratch_cache:
            import shutil
            shutil.rmtree(args.scratch_cache, ignore_errors=True)
        print(f"scored {args.model}@{rev}", flush=True)


if __name__ == "__main__":
    main()
