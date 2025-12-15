"""Offline probe-battery scoring over a run's saved checkpoints.

Rescores dense checkpoints with any battery (e.g. rev2, which has larger
splits than the frozen rev1 file used in-loop), writing one probe_log-format
jsonl. Lets us test whether in-loop transience flags survive a higher-power
battery without retraining.

  python scripts/score_ckpts.py runs/databudget_dn5_s42 \
      --battery data/probes/v2/battery.jsonl --tag rev2
"""

import argparse
import json
import re
from pathlib import Path

import torch
import yaml

from fogen.data import load_tokenizer
from fogen.evals.scoring import aggregate, fogen_scorer, load_battery
from fogen.model import GPT, ModelConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--battery", required=True)
    ap.add_argument("--tag", required=True, help="suffix: probe_log_<tag>.jsonl")
    ap.add_argument("--tokenizer-dir", default=None,
                    help="default: data.tokenizer_dir from config_used.yaml")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--every", type=int, default=1, help="score every Nth ckpt")
    ap.add_argument("--per-template", action="store_true",
                    help="also write per-(probe,template_id) aggregates")
    args = ap.parse_args()

    run = Path(args.run_dir)
    cfg = yaml.safe_load(open(run / "config_used.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(ModelConfig(**cfg["model"])).to(device)
    tokenizer = load_tokenizer(args.tokenizer_dir or cfg["data"]["tokenizer_dir"])
    battery = load_battery(args.battery)
    scorer = fogen_scorer(model, tokenizer, device=device,
                          batch_size=args.batch_size)

    ckpts = sorted((run / "ckpts").glob("step*.safetensors"))[::args.every]
    out_path = run / f"probe_log_{args.tag}.jsonl"
    tmpl_path = run / f"probe_log_{args.tag}_templates.jsonl"
    from safetensors.torch import load_file
    tmpl_out = tmpl_path.open("w") if args.per_template else None
    with out_path.open("w") as out:
        for ck in ckpts:
            step = int(re.search(r"step(\d+)", ck.name).group(1))
            state = {k: v.float() for k, v in load_file(str(ck)).items()}
            missing, unexpected = model.load_state_dict(state, strict=False)
            assert not unexpected, unexpected
            assert all(k.startswith("rope_") for k in missing), missing
            rows = scorer.score_items(battery)
            for a in aggregate(rows):
                out.write(json.dumps({"step": step, **a}) + "\n")
            out.flush()
            if tmpl_out:
                from collections import defaultdict
                groups = defaultdict(list)
                for r in rows:
                    groups[(r["probe"], r["template_id"])].append(r["argmax_acc"])
                for (probe, tid), accs in sorted(groups.items()):
                    tmpl_out.write(json.dumps(
                        {"step": step, "probe": probe, "template_id": tid,
                         "n": len(accs),
                         "argmax_acc": sum(accs) / len(accs)}) + "\n")
                tmpl_out.flush()
            print(f"scored step {step:6d} ({ck.name})")
    if tmpl_out:
        tmpl_out.close()
        print(f"wrote {tmpl_path}")
    print(f"wrote {out_path} ({len(ckpts)} ckpts)")


if __name__ == "__main__":
    main()
