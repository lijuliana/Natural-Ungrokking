"""Frozen evaluator for registered M4 (prior-pull margin predictions).

Registered 2026-06-10 BEFORE any margin computation. Reads
mech_margins.jsonl (scripts/mech_margins.py) per run.

INSTRUMENT-VALID per run: max smoothed CM >= +0.5 nats at step >= MIN_STEP;
violation -> INSTRUMENT-INVALID (never PASS/FAIL).
M4a (CONSISTENCY, behavior already read): web packed CM_final < 0, 3/3;
     TS packed CM_final > 0, 3/3.
M4b (blind, timing): web packed last step with smoothed CM >= 0 within
     [0.5x, 2x] of behavioral collapse onset (last step with smoothed
     pronoun conflict heldout acc >= 0.5, tag rvp1 — registered M2
     definition), >= 2/3 seeds.
M4c (blind, prior strengthening): web packed PM_final > PM at CM-peak
     step, 3/3 seeds.
FALSIFIER: CM_final > 0 on web packed >= 2/3 seeds.

  python scripts/eval_m4.py --seeds 42 43 44 \
      --web runs/web_packed_v2_s{seed} --ts runs/v1_repro_s{seed} \
      [--out runs/eval_m4.json]
"""

import argparse
import json
from pathlib import Path

from gate_a_classify import MIN_STEP, load_traj, smooth

PRON = "pronoun_gender_ref"


def final_window(vals):
    k = max(3, len(vals) // 10)
    return sum(vals[-k:]) / k


def load_margins(run_dir):
    rows = sorted((json.loads(l) for l in
                   (Path(run_dir) / "mech_margins.jsonl")
                   .read_text().splitlines()), key=lambda r: r["step"])
    steps = [r["step"] for r in rows]
    cm_s = smooth([r["CM"] for r in rows])
    pm = [r["PM"] for r in rows]
    return steps, cm_s, pm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--web", required=True)
    ap.add_argument("--ts", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows, invalid = [], []
    for grp, tmpl in (("web", args.web), ("ts", args.ts)):
        for s in args.seeds:
            d = tmpl.format(seed=s)
            steps, cm_s, pm = load_margins(d)
            cand = [(c, st) for st, c in zip(steps, cm_s) if st >= MIN_STEP]
            peak, peak_step = max(cand)
            valid = peak >= 0.5
            if not valid:
                invalid.append((grp, s, peak))
            e = {"grp": grp, "seed": s, "valid": valid,
                 "cm_peak": peak, "cm_peak_step": peak_step,
                 "cm_final": final_window(cm_s),
                 "pm_final": final_window(pm),
                 "pm_at_cm_peak": pm[steps.index(peak_step)]}
            if grp == "web":
                e["cm_zero_cross"] = max(
                    (st for st, c in zip(steps, cm_s) if c >= 0),
                    default=None)
                tr = load_traj(d, "rvp1").get((PRON, "conflict"), [])
                accs = smooth([a for _, a in tr])
                e["collapse_onset"] = max(
                    (st for (st, _), a in zip(tr, accs) if a >= 0.5),
                    default=None)
            rows.append(e)
            print(f"{grp} s{s}: valid={valid} CM peak={peak:+.3f}@"
                  f"{peak_step} final={e['cm_final']:+.3f} "
                  f"PM final={e['pm_final']:+.3f} "
                  f"PM@peak={e['pm_at_cm_peak']:+.3f} "
                  + (f"zero_cross={e['cm_zero_cross']} "
                     f"onset={e['collapse_onset']}" if grp == "web" else ""))

    report = {"rows": rows, "instrument_invalid": invalid}
    if invalid:
        print(f"INSTRUMENT-INVALID runs: {invalid} — no PASS/FAIL issued "
              f"for predicates touching these runs")
    web = [r for r in rows if r["grp"] == "web" and r["valid"]]
    ts = [r for r in rows if r["grp"] == "ts" and r["valid"]]
    n = len(args.seeds)
    if len(web) == n and len(ts) == n:
        report["M4a"] = (all(r["cm_final"] < 0 for r in web)
                         and all(r["cm_final"] > 0 for r in ts))
        m4b = [r["seed"] for r in web
               if r["cm_zero_cross"] is not None
               and r["collapse_onset"] is not None
               and 0.5 * r["collapse_onset"] <= r["cm_zero_cross"]
               <= 2 * r["collapse_onset"]]
        report["M4b"] = len(m4b) >= 2
        report["M4c"] = all(r["pm_final"] > r["pm_at_cm_peak"] for r in web)
        fals = [r["seed"] for r in web if r["cm_final"] > 0]
        report["FALSIFIER_triggered"] = len(fals) >= 2
        for k in ("M4a", "M4b", "M4c"):
            print(f"{k}: {'PASS' if report[k] else 'FAIL'}")
        print(f"M4b hits={m4b}  FALSIFIER: "
              f"{'TRIGGERED ' + str(fals) if report['FALSIFIER_triggered'] else 'no'}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
