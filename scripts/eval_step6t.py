"""Frozen evaluator for the registered Step-6T rescue-timing predictions
(AMENDMENT 2026-06-11, prereg/PREREGISTRATION.md; committed BEFORE any
Step-6T corpus, run, probe value, or margin value exists).

Cells (seeds 42/43/44): rescue_d100_early (injection exposure steps
0-2400, total dose 1.0 x delta_TS), rescue_d100_late (exposure steps
2800-4400, same total dose). Imports the frozen conventions unchanged
(gate_a_classify rvp31 behavioral verdicts; eval_m4 CM smoothing,
final_window, 0.5-nat instrument guard). scripts/eval_step6.py is not
modified and stays frozen over the original 12 predictions.

Gates (violations -> INVALID/UNSCOREABLE, never PASS/FAIL):
  G1 disjointness artifact (runs/step6t_disjointness.json) must exist
     and pass; logged pre-launch.
  G2 windowed-loader unit test passing at the registered commit
     (pre-launch gate, logged in DECISIONS.md; not re-derivable from
     run artifacts and therefore recorded, not recomputed, here).
  G3 intervention validity per cell: det_an_choice AND a_an_adjective
     RECOVERED+valid in >= 2/3 seeds; both families failing in a cell
     -> INTERVENTION-INVALID, no behavioral verdict for that cell.
  G4 instrument: >= 2 seeds with max smoothed CM >= 0.5 nat at
     step >= MIN_STEP, else INSTRUMENT-INVALID for CM components.
Artifact clause: conflict_final >= 0.8 with control-invalid agree in
>= 2/3 seeds = blanket-she artifact, not recovery.

Predictions (>= 2/3 seeds; "seed-mean CM_final" = mean over
instrument-valid seeds, per the amendment text):
  T1_beh early: pronoun RECOVERED+valid.
  T1_cm  early: seed-mean CM_final > 0 (given G4).
  T2_beh late:  conflict_final <= 0.6 (stays dead; supporting arm).
  T3_ord seed-mean CM_final(early) > seed-mean CM_final(late) AND
         seed-mean CM_final(early) > +0.16 (uniform-d100 read value,
         a registered constant); requires G4 in both cells.
T-FALSIFIER iff (i) early INTERVENTION-VALID and (ii) RECOVERED+valid
in <= 1/3 early seeds and (iii) G4 holds with seed-mean CM_final(early)
<= +0.66. (i)+(ii) with G4 failing = "behavioral non-rescue, mechanism
unscoreable" (reported, falsifier NOT triggered). Named intermediate
outcome: (i)+(ii) and CM_final(early) > +0.66 = "timing moves the
mechanism but is behaviorally insufficient at d=1". T2/T3 misses do
NOT trigger the falsifier.
Descriptive (registered named readings, not verdicts): per-seed
in-window smoothed conflict peak for the early cell (steps <= 2400);
reading (B) "rescued in-window then re-collapsed after withdrawal" =
peak >= 0.8 in-window with final below; reading (A) "no in-window
recovery" otherwise.

  python scripts/eval_step6t.py --runs-root runs --seeds 42 43 44 \
      [--tag rvp31] [--g1-json runs/step6t_disjointness.json] \
      [--out runs/eval_step6t.json]
"""

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, "scripts")
from eval_m4 import final_window, load_margins
from gate_a_classify import MIN_STEP, classify, load_traj, smooth

PRON = "pronoun_gender_ref"
SPEC = ["det_an_choice", "a_an_adjective"]
EARLY, LATE = "rescue_d100_early", "rescue_d100_late"
EARLY_WINDOW_END = 2400
UNIFORM_D100_CM = 0.16   # registered constant from the Step-6 read
FALSIFIER_CM_BUF = 0.66  # +0.16 + 0.5-nat instrument-guard buffer


def run_row(d, tag):
    traj = load_traj(d, tag)
    res = classify(traj)
    steps, cm_s, pm = load_margins(d)
    cand = [(c, st) for st, c in zip(steps, cm_s) if st >= MIN_STEP]
    peak, _ = max(cand)
    p = res.get(PRON, {})
    pron_conf = smooth([a for _, a in traj.get((PRON, "conflict"), [])])
    pron_steps = [s for s, _ in traj.get((PRON, "conflict"), [])]
    inwin = [a for s, a in zip(pron_steps, pron_conf)
             if s <= EARLY_WINDOW_END]
    return {"dir": str(d), "pron_class": p.get("class"),
            "pron_valid": p.get("valid"),
            "conflict_final": p.get("conflict_final"),
            "inwindow_peak_conflict": max(inwin) if inwin else None,
            "cm_valid": peak >= 0.5, "cm_peak": peak,
            "cm_final": final_window(cm_s), "pm_final": final_window(pm),
            "spec": {f: (res[f]["class"] == "RECOVERED" and res[f]["valid"])
                     for f in SPEC if f in res}}


def cell_stats(rows):
    rec = sum(r["pron_class"] == "RECOVERED" and r["pron_valid"]
              for r in rows)
    die = sum(r["conflict_final"] is not None and r["conflict_final"] <= 0.6
              for r in rows)
    art = sum(r["conflict_final"] is not None and r["conflict_final"] >= 0.8
              and not r["pron_valid"] for r in rows)
    cmv = [r["cm_final"] for r in rows if r["cm_valid"]]
    spec_ok = {f: sum(r["spec"].get(f, False) for r in rows) >= 2
               for f in SPEC}
    return {"n": len(rows), "recovered_valid": rec, "died": die,
            "artifact": art, "cm_n_valid": len(cmv),
            "cm_final_validmean": (sum(cmv) / len(cmv)) if cmv else None,
            "spec_ok": spec_ok,
            "intervention_valid": sum(not v for v in spec_ok.values()) < 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--tag", default="rvp31")
    ap.add_argument("--g1-json", default="runs/step6t_disjointness.json")
    ap.add_argument("--out", default="runs/eval_step6t.json")
    args = ap.parse_args()

    root = Path(args.runs_root)
    g1 = None
    if Path(args.g1_json).exists():
        g1 = json.load(open(args.g1_json))
    g1_pass = bool(g1) and g1.get("pass", g1.get("ok", False))

    cells = {}
    for name in (EARLY, LATE):
        rows = [run_row(root / f"{name}_s{s}", args.tag)
                for s in args.seeds]
        cells[name] = {"rows": rows, "stats": cell_stats(rows)}
        st = cells[name]["stats"]
        print(f"{name}: rec+valid={st['recovered_valid']}/3 "
              f"died={st['died']}/3 cm_valid={st['cm_n_valid']} "
              f"validmeanCM={st['cm_final_validmean']} "
              f"G3={'OK' if st['intervention_valid'] else 'INVALID'}")

    e, l = cells[EARLY]["stats"], cells[LATE]["stats"]
    V = {"G1_disjointness": g1_pass,
         "G2_loader_test": "pre-launch gate (DECISIONS.md, registered commit)",
         "G3_early": e["intervention_valid"],
         "G3_late": l["intervention_valid"],
         "G4_early": e["cm_n_valid"] >= 2, "G4_late": l["cm_n_valid"] >= 2}

    def beh(st, pred):
        if not st["intervention_valid"]:
            return "INTERVENTION-INVALID"
        return pred(st)

    V["T1_beh"] = beh(e, lambda s: s["recovered_valid"] >= 2)
    V["T1_cm"] = (e["cm_final_validmean"] > 0) if V["G4_early"] \
        else "INSTRUMENT-INVALID"
    V["T2_beh"] = beh(l, lambda s: s["died"] >= 2)
    if V["G4_early"] and V["G4_late"]:
        V["T3_ord"] = (e["cm_final_validmean"] > l["cm_final_validmean"]
                       and e["cm_final_validmean"] > UNIFORM_D100_CM)
    else:
        V["T3_ord"] = "INSTRUMENT-INVALID"

    non_rescue = e["intervention_valid"] and e["recovered_valid"] <= 1
    if non_rescue and V["G4_early"]:
        V["T_FALSIFIER_triggered"] = \
            e["cm_final_validmean"] <= FALSIFIER_CM_BUF
        V["named_outcome"] = (
            "timing moves the mechanism but is behaviorally insufficient "
            "at d=1" if e["cm_final_validmean"] > FALSIFIER_CM_BUF
            else None)
    elif non_rescue:
        V["T_FALSIFIER_triggered"] = False
        V["named_outcome"] = "behavioral non-rescue, mechanism unscoreable"
    else:
        V["T_FALSIFIER_triggered"] = False
        V["named_outcome"] = None
    V["artifact_flag"] = {n: cells[n]["stats"]["artifact"] >= 2
                          for n in (EARLY, LATE)}

    # registered descriptive readings (early arm), not verdicts
    readings = []
    for r in cells[EARLY]["rows"]:
        pk, fin = r["inwindow_peak_conflict"], r["conflict_final"]
        if pk is not None and pk >= 0.8 and fin is not None and fin < 0.8:
            readings.append("B: rescued in-window, re-collapsed after "
                            "withdrawal")
        else:
            readings.append("A: no in-window recovery"
                            if (pk is None or pk < 0.8) else
                            "recovered and held")
    V["early_inwindow_readings"] = readings

    scored = [(k, v) for k, v in V.items()
              if k.startswith("T") and not k.startswith("T_FALS")
              and isinstance(v, bool)]
    V["hit_rate"] = {"hits": sum(v for _, v in scored),
                     "scored": len(scored), "registered": 4}

    for k in ("T1_beh", "T1_cm", "T2_beh", "T3_ord"):
        v = V[k]
        print(f"{k}: {v if not isinstance(v, bool) else 'PASS' if v else 'FAIL'}")
    print(f"T_FALSIFIER: "
          f"{'TRIGGERED' if V['T_FALSIFIER_triggered'] else 'no'}")
    if V["named_outcome"]:
        print(f"named outcome: {V['named_outcome']}")
    print(f"early in-window readings: {readings}")

    Path(args.out).write_text(json.dumps(
        {"registered": "AMENDMENT 2026-06-11 Step-6T (pre-launch)",
         "cells": cells, "verdicts": V}, indent=2, default=str))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
