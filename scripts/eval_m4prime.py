"""Frozen evaluator for the registered M4-prime extension (RESEARCH_LOG
2026-06-11, registered before any dn5/dn15 margin computation; committed
before the margin values are read).

M4pA:  >= 4/6 web dn5+dn15 runs instrument-valid (max smoothed CM >= 0.5
       nats at step >= MIN_STEP) AND CM_final < 0 AND PM_final > PM at
       CM-peak step.
M4pTS: >= 4/6 TS dn5+dn15 runs instrument-valid AND CM_final > 0.
FALSIFIER: any instrument-valid web run with CM_final > 0.

  python scripts/eval_m4prime.py --seeds 42 43 44 \
      --web-dn5 runs/web_dn5_v2_s{seed} --web-dn15 runs/web_dn15_v2_s{seed} \
      --ts-dn5 runs/databudget_dn5_s{seed} --ts-dn15 runs/databudget_dn15_s{seed} \
      [--out runs/eval_m4prime.json]
"""

import argparse
import json
from pathlib import Path

from eval_m4 import final_window, load_margins
from gate_a_classify import MIN_STEP

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--web-dn5", required=True)
    ap.add_argument("--web-dn15", required=True)
    ap.add_argument("--ts-dn5", required=True)
    ap.add_argument("--ts-dn15", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    groups = {("web", "dn5"): args.web_dn5, ("web", "dn15"): args.web_dn15,
              ("ts", "dn5"): args.ts_dn5, ("ts", "dn15"): args.ts_dn15}
    rows = []
    for (grp, bud), tmpl in groups.items():
        for s in args.seeds:
            d = tmpl.format(seed=s)
            steps, cm_s, pm = load_margins(d)
            cand = [(c, st) for st, c in zip(steps, cm_s) if st >= MIN_STEP]
            peak, peak_step = max(cand)
            e = {"grp": grp, "budget": bud, "seed": s,
                 "valid": peak >= 0.5, "cm_peak": peak,
                 "cm_peak_step": peak_step, "cm_final": final_window(cm_s),
                 "pm_final": final_window(pm),
                 "pm_at_cm_peak": pm[steps.index(peak_step)]}
            rows.append(e)
            print(f"{grp} {bud} s{s}: valid={e['valid']} "
                  f"CM peak={peak:+.3f}@{peak_step} "
                  f"final={e['cm_final']:+.3f} PM final={e['pm_final']:+.3f} "
                  f"PM@peak={e['pm_at_cm_peak']:+.3f}")

    web = [r for r in rows if r["grp"] == "web"]
    ts = [r for r in rows if r["grp"] == "ts"]
    m4pa_hits = [(r["budget"], r["seed"]) for r in web
                 if r["valid"] and r["cm_final"] < 0
                 and r["pm_final"] > r["pm_at_cm_peak"]]
    m4pts_hits = [(r["budget"], r["seed"]) for r in ts
                  if r["valid"] and r["cm_final"] > 0]
    fals = [(r["budget"], r["seed"]) for r in web
            if r["valid"] and r["cm_final"] > 0]
    report = {"rows": rows,
              "M4pA": len(m4pa_hits) >= 4, "m4pa_hits": m4pa_hits,
              "M4pTS": len(m4pts_hits) >= 4, "m4pts_hits": m4pts_hits,
              "FALSIFIER_triggered": bool(fals), "falsifier_hits": fals}
    print(f"M4pA: {'PASS' if report['M4pA'] else 'FAIL'} "
          f"hits={m4pa_hits} ({len(m4pa_hits)}/6, need >=4)")
    print(f"M4pTS: {'PASS' if report['M4pTS'] else 'FAIL'} "
          f"hits={m4pts_hits} ({len(m4pts_hits)}/6, need >=4)")
    print(f"FALSIFIER: {'TRIGGERED ' + str(fals) if fals else 'no'}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
