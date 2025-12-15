"""Compact text summary of the borrowed mechanism analyses
(DECISIONS.md 2026-06-11, post-hoc/descriptive) across runs.

Reads runs/<run>/mech_heads.jsonl, runs/<run>/mech_patch.json and
runs/bootstrap_cis_g*.json (all produced by the registered scripts;
pull from s3://fogen-phase first). Prints, per run: the dominant
last-position head by direct-logit attribution at the CM peak and at
the final step, its zero-ablation effect, its OV-cosine, the top
patching components in both directions, and the direction-specificity
contrast; then the bootstrap CI table per family and the seed
envelopes. Pure reporting; no thresholds, no verdicts.

  python scripts/mech_summary.py runs/v1_repro_s42 [runs/...]
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def head_row(rows, step):
    return next(r for r in rows if r["step"] == step)


def top_head(r):
    c = np.array(r["C_heads"])  # [L][H]
    l, h = np.unravel_index(np.abs(c).argmax(), c.shape)
    return int(l), int(h), float(c[l, h])


def fmt_ci(x, ci):
    return f"{x:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]"


def summarize_run(run):
    run = Path(run)
    hp = run / "mech_heads.jsonl"
    pp = run / "mech_patch.json"
    print(f"\n=== {run.name}")
    if not hp.exists() or not pp.exists():
        print("  (artifacts missing; skipped)")
        return
    rows = sorted((json.loads(l) for l in open(hp)),
                  key=lambda r: r["step"])
    pat = json.loads(pp.read_text())
    peak_step, final_step = pat["peak_step"], pat["final_step"]

    for name, step in (("peak", peak_step), ("final", final_step)):
        r = head_row(rows, step)
        l, h, c = top_head(r)
        abl = r["abl_CM"][l][h] - r["CM"]
        ov = r["ov_cos"][l][h]
        tot = float(np.abs(np.array(r["C_heads"])).sum())
        share = abs(c) / tot if tot else float("nan")
        print(f"  {name:5s} step {step:4d}  CM={r['CM']:+.3f}  "
              f"C_dir={r['C_dir']:+.3f}  top-head L{l}H{h} "
              f"C={c:+.3f} ({share:.0%} of head |C|)  "
              f"ablate dCM={abl:+.3f}  ov_cos={ov:+.3f}")

    for d in ("peak_into_final", "final_into_peak"):
        res = pat["patch"][d]
        top3 = sorted(res, key=lambda k: -(res[k]["recovery"] or -9e9))[:3]
        s = "  ".join(f"{k} {res[k]['recovery']:+.2f}" for k in top3)
        print(f"  patch {d:16s} top: {s}")

    for name in ("peak", "final"):
        sp = pat["specificity"][name]
        for r in sp["per_k"]:
            if r["dCM"] is None:
                continue
            print(f"  spec[{name}] k={r['k']}  dCM={r['dCM']:+.3f}  "
                  f"other-fam mean|dmargin|="
                  f"{r['other_fam_mean_absdelta']:.3f}  "
                  f"d_val_bpb={r['d_val_bpb']:+.4f}")


def summarize_cis():
    merged = {}
    for f in sorted(glob.glob("runs/bootstrap_cis_g*.json")) or \
            ["runs/bootstrap_cis.json"]:
        if Path(f).exists():
            d = json.loads(Path(f).read_text())
            for k, v in d.items():
                if k != "cells":
                    merged[k] = v
    if not merged:
        print("\n(no bootstrap_cis files found)")
        return
    print("\n=== bootstrap CIs (10k, stratified, paired drop)")
    for name in sorted(merged):
        print(f"  {name}")
        for fam, r in sorted(merged[name]["families"].items()):
            print(f"    {fam:24s} n={r['n_items']:3d}  "
                  f"peak {fmt_ci(r['peak_acc'], r['peak_ci95'])}  "
                  f"final {fmt_ci(r['final_acc'], r['final_ci95'])}  "
                  f"drop {fmt_ci(r['drop'], r['drop_ci95'])}")
    cells = {}
    import re
    for name, r in merged.items():
        m = re.match(r"(.+)_s(\d+)$", name)
        if m:
            cells.setdefault(m.group(1), []).append(r)
    print("\n=== seed envelopes (min/max over seeds, not CIs)")
    for cell, rs in sorted(cells.items()):
        fams = set.intersection(*[set(r["families"]) for r in rs])
        for fam in sorted(fams):
            dr = [r["families"][fam]["drop"] for r in rs]
            fi = [r["families"][fam]["final_acc"] for r in rs]
            print(f"  {cell:20s} {fam:24s} n_seeds={len(rs)}  "
                  f"final [{min(fi):.3f},{max(fi):.3f}]  "
                  f"drop [{min(dr):.3f},{max(dr):.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="*",
                    default=sorted(glob.glob("runs/*_s4[234]")))
    args = ap.parse_args()
    for run in args.run_dirs:
        summarize_run(run)
    summarize_cis()


if __name__ == "__main__":
    main()
