"""Independent Step-6 verifier (second implementation, written from the
registered definitions in prereg/PREREGISTRATION.md "AMENDMENT 2026-06-10
— Step 6" and the registered instrument conventions; shares NO code with
the frozen evaluator chain eval_step6/gate_a_classify/eval_m4/eval_m3).

Purpose: recompute every per-run row, cell statistic, predicate verdict,
falsifier, and the blind-prediction hit rate directly from the raw logs
(probe_log_rvp31.jsonl, probe_log_rvp31_templates.jsonl,
mech_margins.jsonl) and compare against the frozen evaluator's output.
Any disagreement is a bug in one of the two implementations and must be
adjudicated against the registered text BEFORE results are used. This
script must never be edited to match eval_step6.py outputs without that
adjudication being logged.

Registered conventions re-implemented here:
- behavioral: heldout split only, steps >= 100; centered moving average,
  window 3; final = mean of last max(3, round(0.10*len)) checkpoints;
  RECOVERED final >= 0.8; DISPLACED emerged (max >= 0.8) and final
  <= 0.6; UNSTABLE if smoothed conflict range >= 0.2 over the final
  window; control-valid = agree final >= 0.8 and (max - final) < 0.15.
- CM: smooth CM with the same window-3 average over the step-sorted
  series; CM_final = mean of last max(3, len//10) entries (eval_m4
  convention); instrument guard = max smoothed CM at step >= 100 must
  reach 0.5 nat; PM_final from the raw (unsmoothed) PM series.
- graded: Spearman rank correlation (average ranks on ties) between
  knob value (incl. the 0-knob baseline cell) and seed-mean CM_final
  (mean over all 3 seeds, valid or not, per the registered text).
- template CI: stratified bootstrap over template ids, 1000 resamples,
  random.Random(0), 95% interval of the final-window weighted mean.

  python scripts/verify_step6.py --runs-root runs --seeds 42 43 44
"""

import argparse
import json
import random
from pathlib import Path

W_SMOOTH = 3
STEP_MIN = 100
THR_REC = 0.8
THR_DISP = 0.6
THR_AGREE = 0.8
THR_AGREE_DROP = 0.15
THR_UNSTABLE = 0.2
THR_CM_GUARD = 0.5
PRONOUN = "pronoun_gender_ref"
RESCUE_CELLS = {"rescue_d001": 0.01, "rescue_d010": 0.1,
                "rescue_d100": 1.0, "rescue_d300": 3.0}
KILL_CELLS = {"kill_p437": 0.437, "kill_p645": 0.645, "kill_p1000": 1.0}
R_CONTROLS = ("det_an_choice", "a_an_adjective")
K_CONTROLS = ("det_an_choice", "a_an_adjective", "irregular_past",
              "negation_bare_verb")


def moving_avg(xs):
    h = W_SMOOTH // 2
    return [sum(xs[max(0, i - h): i + h + 1])
            / (min(len(xs), i + h + 1) - max(0, i - h))
            for i in range(len(xs))]


def tail_mean_round(xs):
    n = max(3, int(round(len(xs) * 0.10)))
    return sum(xs[-n:]) / n, n


def tail_mean_floor(xs):
    n = max(3, len(xs) // 10)
    return sum(xs[-n:]) / n


def rank_avg(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    ranks = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman_rho(xs, ys):
    rx, ry = rank_avg(list(xs)), rank_avg(list(ys))
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    var = (sum((a - mx) ** 2 for a in rx)
           * sum((b - my) ** 2 for b in ry)) ** 0.5
    return cov / var if var else 0.0


def probe_trajectories(run_dir, tag):
    """{(family, cond): [(step, acc)] step-sorted}, heldout, step>=100."""
    by_key = {}
    for line in (Path(run_dir) / f"probe_log_{tag}.jsonl").open():
        r = json.loads(line)
        if r.get("split") != "heldout" or r["step"] < STEP_MIN:
            continue
        fam, _, cond = r["probe"].rpartition(".")
        by_key.setdefault((fam, cond), {})[r["step"]] = r["argmax_acc"]
    return {k: sorted(d.items()) for k, d in by_key.items()}


def classify_family(traj, fam):
    sm = {}
    for cond in ("conflict", "agree"):
        accs = [a for _, a in traj.get((fam, cond), [])]
        sm[cond] = moving_avg(accs) if accs else []
    cf, ag = sm["conflict"], sm["agree"]
    cf_final, n_tail = tail_mean_round(cf) if cf else (float("nan"), 0)
    ag_final, _ = tail_mean_round(ag) if ag else (float("nan"), 0)
    valid = bool(ag) and ag_final >= THR_AGREE \
        and (max(ag) - ag_final) < THR_AGREE_DROP
    tail = cf[-n_tail:] if cf else []
    if tail and max(tail) - min(tail) >= THR_UNSTABLE:
        cls = "UNSTABLE"
    elif cf and cf_final >= THR_REC:
        cls = "RECOVERED"
    elif cf and max(cf) >= THR_REC and cf_final <= THR_DISP:
        cls = "DISPLACED"
    elif cf and max(cf) >= THR_REC:
        cls = "PARTIAL"
    else:
        cls = "NEVER"
    return cls, valid, cf_final


def margins(run_dir):
    rows = sorted((json.loads(l) for l in
                   (Path(run_dir) / "mech_margins.jsonl").open()),
                  key=lambda r: r["step"])
    steps = [r["step"] for r in rows]
    cm_sm = moving_avg([r["CM"] for r in rows])
    pm = [r["PM"] for r in rows]
    guard_vals = [c for s, c in zip(steps, cm_sm) if s >= STEP_MIN]
    peak = max(guard_vals)
    return {"cm_valid": peak >= THR_CM_GUARD, "cm_peak": peak,
            "cm_final": tail_mean_floor(cm_sm),
            "pm_final": tail_mean_floor(pm)}


def template_ci(run_dir, tag):
    path = Path(run_dir) / f"probe_log_{tag}_templates.jsonl"
    if not path.exists():
        return None
    per_step = {}
    for line in path.open():
        r = json.loads(line)
        if r["probe"] != f"{PRONOUN}.conflict" or r["step"] < STEP_MIN:
            continue
        per_step.setdefault(r["step"], []).append(
            (r["template_id"], r["argmax_acc"], r["n"]))
    steps = sorted(per_step)
    if not steps:
        return None
    k = max(3, len(steps) // 10)
    fin = steps[-k:]
    tids = sorted({t for s in fin for t, _, _ in per_step[s]})
    rng = random.Random(0)
    ests = []
    for _ in range(1000):
        pick = [tids[rng.randrange(len(tids))] for _ in tids]
        means = []
        for s in fin:
            at_s = {t: (a, n) for t, a, n in per_step[s]}
            num = sum(at_s[t][0] * at_s[t][1] for t in pick if t in at_s)
            den = sum(at_s[t][1] for t in pick if t in at_s)
            if den:
                means.append(num / den)
        if means:
            ests.append(sum(means) / len(means))
    ests.sort()
    return [ests[int(0.025 * len(ests))], ests[int(0.975 * len(ests))]]


def run_row(run_dir, tag):
    traj = probe_trajectories(run_dir, tag)
    fams = sorted({f for f, _ in traj})
    pron_cls, pron_valid, pron_final = classify_family(traj, PRONOUN)
    row = {"dir": str(run_dir), "pron_class": pron_cls,
           "pron_valid": pron_valid, "conflict_final": pron_final,
           **margins(run_dir),
           "spec": {}}
    for fam in sorted(set(R_CONTROLS) | set(K_CONTROLS)):
        if fam in fams:
            cls, valid, _ = classify_family(traj, fam)
            row["spec"][fam] = (cls == "RECOVERED" and valid)
    ci = template_ci(run_dir, tag)
    if ci is not None:
        row["conflict_final_ci95"] = ci
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--tag", default="rvp31")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.runs_root)

    cells = {}
    for name in (list(RESCUE_CELLS) + list(KILL_CELLS)
                 + ["web_packed_v2", "v1_repro"]):
        rows = [run_row(root / f"{name}_s{s}", args.tag)
                for s in args.seeds]
        n_rec = sum(r["pron_class"] == "RECOVERED" and r["pron_valid"]
                    for r in rows)
        n_die = sum(r["conflict_final"] <= THR_DISP for r in rows
                    if r["conflict_final"] == r["conflict_final"])
        n_art = sum(r["conflict_final"] >= THR_REC and not r["pron_valid"]
                    for r in rows
                    if r["conflict_final"] == r["conflict_final"])
        valid_rows = [r for r in rows if r["cm_valid"]]
        cells[name] = {
            "rows": rows, "recovered_valid": n_rec, "died": n_die,
            "artifact": n_art, "cm_n_valid": len(valid_rows),
            "cm_pos": sum(r["cm_final"] > 0 for r in valid_rows),
            "cm_neg": sum(r["cm_final"] < 0 for r in valid_rows),
            "cm_final_mean": sum(r["cm_final"] for r in rows) / len(rows)}

    def cm_call(cell, positive):
        c = cells[cell]
        if c["cm_n_valid"] < 2:
            return "INSTRUMENT-INVALID"
        return (c["cm_pos"] if positive else c["cm_neg"]) >= 2

    verdicts, directional = {}, []

    def predict(name, outcome):
        verdicts[name] = outcome
        directional.append((name, outcome))

    predict("R1_beh", cells["rescue_d100"]["recovered_valid"] >= 2)
    predict("R1_cm", cm_call("rescue_d100", True))
    predict("R1c_beh", cells["rescue_d300"]["recovered_valid"] >= 2)
    predict("R1c_cm", cm_call("rescue_d300", True))
    predict("R2_beh", cells["rescue_d001"]["died"] >= 2)
    predict("R2_cm", cm_call("rescue_d001", False))

    doses = [0.0] + list(RESCUE_CELLS.values())
    dose_cms = [cells["web_packed_v2"]["cm_final_mean"]] + \
        [cells[c]["cm_final_mean"] for c in RESCUE_CELLS]
    rho_r = spearman_rho(doses, dose_cms)
    predict("R3_graded", rho_r >= 0.8)

    verdicts["RS_specificity"] = all(
        sum(r["spec"].get(f, False) for r in cells[c]["rows"]) >= 2
        for c in RESCUE_CELLS for f in R_CONTROLS)
    verdicts["R_FALSIFIER_triggered"] = (
        (cells["rescue_d100"]["recovered_valid"] < 2
         and cells["rescue_d300"]["recovered_valid"] < 2)
        or any(cells[c]["artifact"] >= 2 for c in RESCUE_CELLS))

    kill_ok = {}
    for c in KILL_CELLS:
        n_bad = sum(sum(r["spec"].get(f, False)
                        for r in cells[c]["rows"]) < 2 for f in K_CONTROLS)
        kill_ok[c] = n_bad < 2
    verdicts["kill_cell_intervention_valid"] = kill_ok

    for cell, tagn in (("kill_p645", "K1"), ("kill_p1000", "K1c")):
        if not kill_ok[cell]:
            predict(f"{tagn}_beh", "INTERVENTION-INVALID")
            predict(f"{tagn}_cm", "INTERVENTION-INVALID")
        else:
            predict(f"{tagn}_beh", cells[cell]["died"] >= 2)
            predict(f"{tagn}_cm", cm_call(cell, False))

    rates = [0.0] + list(KILL_CELLS.values())
    rate_cms = [cells["v1_repro"]["cm_final_mean"]] + \
        [cells[c]["cm_final_mean"] for c in KILL_CELLS]
    rho_k = spearman_rho(rates, rate_cms)
    predict("K2_graded", rho_k <= -0.8)

    verdicts["K_FALSIFIER_triggered"] = all(
        cells[c]["recovered_valid"] >= 2
        for c in ("kill_p645", "kill_p1000"))

    scoreable = [(n, o) for n, o in directional if isinstance(o, bool)]
    verdicts["hit_rate"] = {
        "hits": sum(o for _, o in scoreable), "scored": len(scoreable),
        "registered": len(directional),
        "unscoreable": [n for n, o in directional
                        if not isinstance(o, bool)]}
    verdicts["rho_rescue"], verdicts["rho_kill"] = rho_r, rho_k

    for n, o in directional:
        label = o if not isinstance(o, bool) else ("PASS" if o else "FAIL")
        print(f"{n}: {label}")
    print(f"RS_specificity: "
          f"{'PASS' if verdicts['RS_specificity'] else 'FAIL'}")
    print(f"R_FALSIFIER: "
          f"{'TRIGGERED' if verdicts['R_FALSIFIER_triggered'] else 'no'}  "
          f"K_FALSIFIER: "
          f"{'TRIGGERED' if verdicts['K_FALSIFIER_triggered'] else 'no'}")
    print(f"rho_rescue={rho_r:+.3f} rho_kill={rho_k:+.3f}")
    hr = verdicts["hit_rate"]
    print(f"hit rate: {hr['hits']}/{hr['scored']} scored "
          f"({hr['registered']} registered)")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"cells": cells, "verdicts": verdicts}, indent=2, default=str))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
