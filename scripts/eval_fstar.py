"""Frozen evaluator for the registered Step-4 f* ordering prediction.

Written 2026-06-10 BEFORE any support-count JSON was read (support side
UNREAD; the conflict_final side was already read in the Step-3 governed
read — disclosed: only the support ranks remain blind, so the Spearman
cannot be tuned via the support axis).

Registered (RESEARCH_LOG 2026-06-10, prereg §4):
  rule_support(F,C)  = bigram aggregate count F.conflict.correct
  prior_support(F,C) = bigram aggregate count F.conflict.distractor
  support_ratio      = (rule+1)/(prior+1)
  Gender families (pronoun_gender_ref, reflexive_gender — both name-cued)
  use the f*-v2 WINDOWED ratio instead: combine girl_names + boy_names
  classes congruently (rule = girls.first_she + boys.first_he,
  prior = girls.first_he + boys.first_she).
  PREDICTION: Spearman rho(support_ratio, conflict_final) > 0 in every
  web cell of the Step-3 grid, among control-valid non-exploratory
  families; DISPLACED/low families (pronoun_gender_ref, reflexive_gender)
  have the LOWEST web support_ratio among non-exploratory families.
  FALSIFIER: any web cell where a bottom-2 support_ratio family is
  RECOVERED while a top-2 family is DISPLACED (both control-valid).

Operationalization fixed at write time (before support read):
  conflict_final(F, cell) = mean over seeds; control-valid = valid in
  >= 2/3 seeds; classification tag rvp31 (the battery the counts derive
  from). Per-seed rhos and a no-_v2 sensitivity are reported descriptively.

  python scripts/eval_fstar.py --seeds 42 43 44 \
      --ts-bigram runs/support_tinystories_rvp31.json \
      --web-bigram runs/support_web_rvp31.json \
      --ts-window runs/support_ts_window.json \
      --web-window runs/support_web_window.json \
      --web packed=runs/web_packed_v2_s{seed} dn5=runs/web_dn5_v2_s{seed} \
            dn15=runs/web_dn15_v2_s{seed} [--out runs/eval_fstar.json]
"""

import argparse
import json
from pathlib import Path

from eval_public_suite import spearman
from gate_a_classify import EXPLORATORY, classify, load_traj

GENDER_FAMS = ["pronoun_gender_ref", "reflexive_gender"]


def support_ratios(bigram_path, window_path):
    agg = json.load(open(bigram_path))["aggregate"]
    fams = sorted({k.rsplit(".", 2)[0] for k in agg})
    ratios = {}
    for f in fams:
        rule = agg.get(f"{f}.conflict.correct", 0)
        prior = agg.get(f"{f}.conflict.distractor", 0)
        ratios[f] = (rule + 1) / (prior + 1)
    w = json.load(open(window_path))["classes"]
    rule = w["girl_names"]["first_she"] + w["boy_names"]["first_he"]
    prior = w["girl_names"]["first_he"] + w["boy_names"]["first_she"]
    for f in GENDER_FAMS:
        if f in ratios:
            ratios[f] = (rule + 1) / (prior + 1)
    return ratios


def verdict(name, ok, detail):
    print(f"{name}: {'PASS' if ok else 'FAIL'}  {detail}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--tag", default="rvp31")
    ap.add_argument("--ts-bigram", required=True)
    ap.add_argument("--web-bigram", required=True)
    ap.add_argument("--ts-window", required=True)
    ap.add_argument("--web-window", required=True)
    ap.add_argument("--web", nargs=3, metavar="name=tmpl", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sr = {"ts": support_ratios(args.ts_bigram, args.ts_window),
          "web": support_ratios(args.web_bigram, args.web_window)}
    fams = sorted(f for f in sr["web"] if f not in EXPLORATORY)
    print("support_ratio (web | ts):")
    for f in sorted(fams, key=lambda f: sr["web"][f]):
        print(f"  {f:26s} {sr['web'][f]:10.4f} | {sr['ts'].get(f, float('nan')):10.4f}")

    cells = dict(c.split("=", 1) for c in args.web)
    res = {n: {s: classify(load_traj(t.format(seed=s), args.tag))
               for s in args.seeds} for n, t in cells.items()}

    report = {"support_ratio": sr, "cells": {}}
    all_pos, falsified = True, []
    for n in cells:
        valid_f = [f for f in fams
                   if sum(res[n][s][f]["valid"] for s in args.seeds) >= 2]
        finals = {f: sum(res[n][s][f]["conflict_final"]
                         for s in args.seeds) / len(args.seeds)
                  for f in valid_f}
        rho = spearman([sr["web"][f] for f in valid_f],
                       [finals[f] for f in valid_f])
        per_seed = {s: spearman(
            [sr["web"][f] for f in fams if res[n][s][f]["valid"]],
            [res[n][s][f]["conflict_final"] for f in fams
             if res[n][s][f]["valid"]]) for s in args.seeds}
        no_v2 = [f for f in valid_f if not f.endswith("_v2")]
        rho_no_v2 = spearman([sr["web"][f] for f in no_v2],
                             [finals[f] for f in no_v2])
        all_pos &= rho > 0
        ranked = sorted(valid_f, key=lambda f: sr["web"][f])
        bot2, top2 = ranked[:2], ranked[-2:]
        for s in args.seeds:
            br = [f for f in bot2 if res[n][s][f]["class"] == "RECOVERED"
                  and res[n][s][f]["valid"]]
            td = [f for f in top2 if res[n][s][f]["class"] == "DISPLACED"
                  and res[n][s][f]["valid"]]
            if br and td:
                falsified.append((n, s, br, td))
        report["cells"][n] = {
            "valid_families": valid_f, "finals": finals, "rho": rho,
            "rho_per_seed": per_seed, "rho_no_v2": rho_no_v2,
            "bottom2": bot2, "top2": top2}
        print(f"cell {n}: rho={rho:+.3f} (no_v2 {rho_no_v2:+.3f}; "
              f"per-seed {per_seed}) n_fams={len(valid_f)} "
              f"bot2={bot2} top2={top2}")

    report["F1_spearman_all_web_cells"] = verdict(
        "F1 (rho>0 every web cell)", all_pos,
        f"rhos={ {n: report['cells'][n]['rho'] for n in cells} }")
    web_rank = sorted(fams, key=lambda f: sr["web"][f])
    lowest_ok = set(web_rank[:2]) == set(GENDER_FAMS)
    report["F2_displaced_lowest"] = verdict(
        "F2 (gender fams lowest web support)", lowest_ok,
        f"bottom2={web_rank[:2]}")
    report["FALSIFIER"] = verdict(
        "FALSIFIER (bottom-2 R while top-2 D, both valid)", not falsified,
        f"hits={falsified}") and not falsified
    report["FALSIFIER"] = not falsified

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
