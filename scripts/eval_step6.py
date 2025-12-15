"""Frozen evaluator for the registered Step-6 causal-control predictions
(RESEARCH_LOG amendment 2026-06-10; committed BEFORE any Step-6 run is
launched or any Step-6 margin/probe value is read).

Cells (seeds 42/43/44 each; baselines are the existing Step-3 runs):
  RESCUE (base web_packed_v2): rescue_d001/d010/d100/d300
          = injected dose 0.01 / 0.1 / 1.0 / 3.0 x TS girl-class density.
  KILL   (base v1_repro):      kill_p437/p645/p1000
          = girl-cue first-pronoun flip rate 0.437 / 0.645 / 1.0.

Behavioral verdicts use the registered classifier (gate_a_classify, tag
rvp31 offline rescore); CM verdicts use mech_margins + eval_m4
conventions (smoothed CM, final_window, instrument guard >= 0.5 nat).
Behavioral and CM components are scored INDEPENDENTLY; a cell-level
mechanism agreement requires both.

Registered predicates (>= 2/3 seeds unless stated):
  R1  rescue_d100: pronoun RECOVERED+valid           AND CM_final > 0
  R1c rescue_d300: same                              (ceiling dose)
  R2  rescue_d001: conflict_final <= 0.6             AND CM_final < 0
  R3  Spearman(dose, seed-mean CM_final) >= +0.8 over
      {0(web_packed_v2), .01, .1, 1, 3}
  RS  det_an_choice AND a_an_adjective RECOVERED+valid in every rescue cell
  R-FALSIFIER: R1-style predicate fails in BOTH d100 and d300
               (behavioral part), OR any rescue cell shows
               conflict_final >= 0.8 with control-invalid agree in >= 2/3
               (blanket-she artifact).
  K1  kill_p645:  pronoun conflict_final <= 0.6 (not RECOVERED+valid)
                                                     AND CM_final < 0
  K1c kill_p1000: same                               (reversal ceiling)
  K2  Spearman(p, seed-mean CM_final) <= -0.8 over
      {0(v1_repro), .437, .645, 1}
  KS  det_an_choice, a_an_adjective, irregular_past, negation_bare_verb
      all RECOVERED+valid in every kill cell; >= 2 of them failing in a
      cell -> that cell INTERVENTION-INVALID (no K verdicts from it)
  K-FALSIFIER: pronoun RECOVERED+valid in >= 2/3 seeds in BOTH kill_p645
               and kill_p1000.
  Instrument guard: runs with max smoothed CM < 0.5 nat at step >=
  MIN_STEP are INSTRUMENT-INVALID for CM components only; a cell CM
  component needs >= 2 valid seeds, else no CM verdict is issued.

Blind-prediction hit rate = fraction of the directional predictions
below that come out true (each cell x {behavioral direction, CM sign}
for R1/R1c/R2/K1/K1c, plus R3 and K2): 12 predictions total.

  python scripts/eval_step6.py --runs-root runs --seeds 42 43 44 \
      [--tag rvp31] [--out runs/eval_step6.json]
"""

import argparse
import json
from pathlib import Path

from eval_m3 import bootstrap_ci, template_traj
from eval_m4 import final_window, load_margins
from eval_public_suite import spearman
from gate_a_classify import MIN_STEP, classify, load_traj

PRON = "pronoun_gender_ref"
RESCUE = [("rescue_d001", 0.01), ("rescue_d010", 0.1),
          ("rescue_d100", 1.0), ("rescue_d300", 3.0)]
KILL = [("kill_p437", 0.437), ("kill_p645", 0.645), ("kill_p1000", 1.0)]
R_SPEC = ["det_an_choice", "a_an_adjective"]
K_SPEC = ["det_an_choice", "a_an_adjective", "irregular_past",
          "negation_bare_verb"]


def run_row(d, tag):
    res = classify(load_traj(d, tag))
    steps, cm_s, pm = load_margins(d)
    cand = [(c, st) for st, c in zip(steps, cm_s) if st >= MIN_STEP]
    peak, peak_step = max(cand)
    p = res.get(PRON, {})
    row = {"dir": str(d), "pron_class": p.get("class"),
           "pron_valid": p.get("valid"),
           "conflict_final": p.get("conflict_final"),
           "cm_valid": peak >= 0.5, "cm_peak": peak,
           "cm_final": final_window(cm_s), "pm_final": final_window(pm),
           "spec": {f: (res[f]["class"] == "RECOVERED" and res[f]["valid"])
                    for f in set(R_SPEC + K_SPEC) if f in res}}
    try:
        t_steps, t_acc, per_step = template_traj(d, tag)
        if t_steps:
            k = max(3, len(t_acc) // 10)
            row["conflict_final_ci95"] = list(
                bootstrap_ci(per_step, t_steps, k))
    except FileNotFoundError:
        pass
    return row


def cell_stats(rows):
    rec = sum(r["pron_class"] == "RECOVERED" and r["pron_valid"]
              for r in rows)
    die = sum(r["conflict_final"] is not None and r["conflict_final"] <= 0.6
              for r in rows)
    art = sum(r["conflict_final"] is not None and r["conflict_final"] >= 0.8
              and not r["pron_valid"] for r in rows)
    cmv = [r for r in rows if r["cm_valid"]]
    pos = sum(r["cm_final"] > 0 for r in cmv)
    neg = sum(r["cm_final"] < 0 for r in cmv)
    return {"n": len(rows), "recovered_valid": rec, "died": die,
            "artifact": art, "cm_n_valid": len(cmv), "cm_pos": pos,
            "cm_neg": neg,
            "cm_final_mean": (sum(r["cm_final"] for r in rows)
                              / len(rows)) if rows else None}


def cm_verdict(st, want_pos):
    if st["cm_n_valid"] < 2:
        return "INSTRUMENT-INVALID"
    hits = st["cm_pos"] if want_pos else st["cm_neg"]
    return hits >= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--tag", default="rvp31")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.runs_root)
    cells = {}
    for name, x in (RESCUE + KILL
                    + [("web_packed_v2", 0.0), ("v1_repro", 0.0)]):
        rows = [run_row(root / f"{name}_s{s}", args.tag)
                for s in args.seeds]
        cells[name] = {"x": x, "rows": rows, "stats": cell_stats(rows)}
        st = cells[name]["stats"]
        print(f"{name}: rec+valid={st['recovered_valid']}/3 "
              f"died={st['died']}/3 cm_valid={st['cm_n_valid']} "
              f"cm+={st['cm_pos']} cm-={st['cm_neg']} "
              f"meanCM={st['cm_final_mean']:+.3f}")

    V = {}
    preds = []  # (name, outcome True/False/'INSTRUMENT-INVALID')

    def add(name, outcome):
        preds.append((name, outcome))
        V[name] = outcome

    for cell, tagn in (("rescue_d100", "R1"), ("rescue_d300", "R1c")):
        st = cells[cell]["stats"]
        add(f"{tagn}_beh", st["recovered_valid"] >= 2)
        add(f"{tagn}_cm", cm_verdict(st, want_pos=True))
    st = cells["rescue_d001"]["stats"]
    add("R2_beh", st["died"] >= 2)
    add("R2_cm", cm_verdict(st, want_pos=False))

    r_doses = [0.0] + [x for _, x in RESCUE]
    r_cms = [cells["web_packed_v2"]["stats"]["cm_final_mean"]] + \
            [cells[n]["stats"]["cm_final_mean"] for n, _ in RESCUE]
    rho_r = spearman(r_doses, r_cms)
    add("R3_graded", rho_r >= 0.8)

    rs = all(sum(r["spec"].get(f, False) for r in cells[n]["rows"]) >= 2
             for n, _ in RESCUE for f in R_SPEC)
    V["RS_specificity"] = rs
    r_fals = ((cells["rescue_d100"]["stats"]["recovered_valid"] < 2)
              and (cells["rescue_d300"]["stats"]["recovered_valid"] < 2)) \
        or any(cells[n]["stats"]["artifact"] >= 2 for n, _ in RESCUE)
    V["R_FALSIFIER_triggered"] = r_fals

    kill_valid = {}
    for n, _ in KILL:
        bad = sum(sum(r["spec"].get(f, False)
                      for r in cells[n]["rows"]) < 2 for f in K_SPEC)
        kill_valid[n] = bad < 2
    V["kill_cell_intervention_valid"] = kill_valid

    for cell, tagn in (("kill_p645", "K1"), ("kill_p1000", "K1c")):
        st = cells[cell]["stats"]
        if not kill_valid[cell]:
            add(f"{tagn}_beh", "INTERVENTION-INVALID")
            add(f"{tagn}_cm", "INTERVENTION-INVALID")
            continue
        add(f"{tagn}_beh", st["died"] >= 2)
        add(f"{tagn}_cm", cm_verdict(st, want_pos=False))

    k_ps = [0.0] + [x for _, x in KILL]
    k_cms = [cells["v1_repro"]["stats"]["cm_final_mean"]] + \
            [cells[n]["stats"]["cm_final_mean"] for n, _ in KILL]
    rho_k = spearman(k_ps, k_cms)
    add("K2_graded", rho_k <= -0.8)

    k_fals = all(cells[n]["stats"]["recovered_valid"] >= 2
                 for n in ("kill_p645", "kill_p1000"))
    V["K_FALSIFIER_triggered"] = k_fals

    scored = [(n, o) for n, o in preds if isinstance(o, bool)]
    hits = sum(o for _, o in scored)
    V["hit_rate"] = {"hits": hits, "scored": len(scored),
                     "registered": len(preds),
                     "unscoreable": [n for n, o in preds
                                     if not isinstance(o, bool)]}
    V["rho_rescue"], V["rho_kill"] = rho_r, rho_k

    for n, o in preds:
        print(f"{n}: "
              f"{o if not isinstance(o, bool) else 'PASS' if o else 'FAIL'}")
    print(f"RS_specificity: {'PASS' if rs else 'FAIL'}")
    print(f"R_FALSIFIER: {'TRIGGERED' if r_fals else 'no'}  "
          f"K_FALSIFIER: {'TRIGGERED' if k_fals else 'no'}")
    print(f"rho_rescue={rho_r:+.3f} rho_kill={rho_k:+.3f}")
    print(f"hit rate: {hits}/{len(scored)} scored "
          f"({len(preds)} registered)")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"cells": cells, "verdicts": V}, indent=2, default=str))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
