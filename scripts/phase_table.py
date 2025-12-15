"""Family x cell phase table with per-seed outcome strings.

Applies the registered classifier (gate_a_classify.classify, constants
unchanged) to every (cell, seed) run dir and prints one outcome letter per
seed: R/D/P/N/U (lowercase if control-invalid). Also emits a JSON artifact
for plotting.

  python scripts/phase_table.py --seeds 42 43 44 --tag rvp1 \
      packed=runs/v1_repro_s{seed} dn5=runs/databudget_dn5_s{seed} \
      dn15=runs/databudget_dn15_s{seed} web=runs/web_packed_v2_s{seed} \
      webdn5=runs/web_dn5_v2_s{seed} webdn15=runs/web_dn15_v2_s{seed} \
      [--out runs/phase_table_rvp1.json]
"""

import argparse
import json
from pathlib import Path

from gate_a_classify import EXPLORATORY, classify, load_traj

LETTER = {"RECOVERED": "R", "DISPLACED": "D", "PARTIAL": "P",
          "NEVER": "N", "UNSTABLE": "U"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="+", metavar="name=run_dir_template",
                    help="run dir with {seed} placeholder")
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--tag", default="rvp1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pairs = [c.split("=", 1) for c in args.cells]
    results = {}  # (cell, seed) -> {family: dict}
    missing = []
    for name, tmpl in pairs:
        for seed in args.seeds:
            d = tmpl.format(seed=seed)
            if not (Path(d) / f"probe_log_{args.tag}.jsonl").exists():
                missing.append(d)
                continue
            results[(name, seed)] = classify(load_traj(d, args.tag))

    fams = sorted({f for r in results.values() for f in r})
    cells = [n for n, _ in pairs]
    print(f"{'family':26s} " + "".join(f"{c:>10s}" for c in cells)
          + f"   (seeds {' '.join(map(str, args.seeds))}; lowercase = control-invalid)")
    table = {}
    for fam in fams:
        row = f"{fam:26s} "
        for c in cells:
            s = ""
            for seed in args.seeds:
                r = results.get((c, seed), {}).get(fam)
                if r is None:
                    s += "."
                else:
                    l = LETTER[r["class"]]
                    s += l if r["valid"] else l.lower()
            row += f"{s:>10s}"
            table[f"{fam}|{c}"] = s
        print(row + ("  [exploratory]" if fam in EXPLORATORY else ""))
    if missing:
        print("\nmissing runs: " + ", ".join(missing))

    if args.out:
        Path(args.out).write_text(json.dumps({
            "tag": args.tag, "seeds": args.seeds, "cells": cells,
            "table": table,
            "detail": {f"{c}|s{seed}": res for (c, seed), res in
                       sorted((((c, s), r) for (c, s), r in results.items()))},
        }, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
