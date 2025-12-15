"""Evaluator for the registered rvp5 OOD-transfer predictions
(prereg AMENDMENT 2026-06-10; thresholds fixed before any rvp5 scoring).

Implements exactly:
  RV5-V  validity: run is RVP5-VALID iff pronoun_gender_ref_ood.agree
         heldout final >= 0.7 (smoothing + final window per the
         registered classifier conventions in gate_a_classify).
  RV5-P1 competence transfers: every cell with rvp31 pronoun_gender_ref
         RECOVERED+valid in >= 2/3 seeds has OOD conflict final >= 0.7
         in >= 2/3 of RVP5-VALID seeds.
  RV5-P2 failure transfers: every cell with rvp31 DISPLACED+valid in
         >= 2/3 seeds has OOD conflict final <= 0.6 in >= 2/3 of
         RVP5-VALID seeds.
  RV5-P3 transience transfers: every cell-seed whose rvp31 pronoun
         conflict trajectory is transient (smoothed max >= 0.8 and
         final <= 0.6) has rvp5 OOD smoothed conflict max >= final+0.15.
  RV5-F  falsifier: >= 2 distinct RECOVERED+valid cells where >= 2/3 of
         RVP5-VALID seeds have OOD conflict final < 0.6.
reflexive_gender_ood and plural_was_were_ood are reported descriptively
(no registered thresholds). Outcomes reported for every cell.

Scope guard: Step-3/baseline cells, seeds 42-44 only. Step-6 cells are
REFUSED unless --include-step6, which additionally requires (i) a logged
governed read (runs/eval_step6.json exists) and (ii) the disjointness
gate report with ok=true (runs/rvp5_disjointness.json).

  python scripts/eval_rvp5.py --runs-root runs --seeds 42 43 44 \
      [--out runs/eval_rvp5.json] [--include-step6]
"""

import argparse
import json
from pathlib import Path

from gate_a_classify import classify, final_mean, load_traj, smooth

PRON_ID = "pronoun_gender_ref"
PRON_OOD = "pronoun_gender_ref_ood"
BREADTH = ["reflexive_gender_ood", "plural_was_were_ood"]
BASE_CELLS = ["v1_repro", "databudget_dn5", "databudget_dn15",
              "web_packed_v2", "web_dn5_v2", "web_dn15_v2",
              "ts_packed_armB"]
STEP6_CELLS = ["rescue_d001", "rescue_d010", "rescue_d100", "rescue_d300",
               "kill_p437", "kill_p645", "kill_p1000"]
ALLOWED_SEEDS = {42, 43, 44}
VALID_THR = 0.7        # RV5-V agree bound
TRANSFER_THR = 0.7     # RV5-P1 conflict bound
FAIL_THR = 0.6         # RV5-P2 bound (= registered DISPLACED bound)
P3_GAP = 0.15
FRAC = 2 / 3


def fam_stats(traj, fam):
    out = {}
    for cond in ("conflict", "agree"):
        accs = smooth([a for _, a in traj.get((fam, cond), [])])
        out[cond] = {"final": final_mean(accs) if accs else None,
                     "max": max(accs) if accs else None,
                     "n_ckpts": len(accs)}
    return out


def seed_row(run_dir):
    t5 = load_traj(run_dir, "rvp5")
    pron = fam_stats(t5, PRON_OOD)
    ag, cf = pron["agree"], pron["conflict"]
    rvp5_valid = ag["final"] is not None and ag["final"] >= VALID_THR
    id_res = classify(load_traj(run_dir, "rvp31")).get(PRON_ID, {})
    transient_id = (id_res.get("conflict_max") is not None
                    and id_res["conflict_max"] >= 0.8
                    and id_res["conflict_final"] <= 0.6)
    return {"dir": str(run_dir),
            "rvp5_valid": rvp5_valid,
            "ood_agree_final": ag["final"],
            "ood_conflict_final": cf["final"],
            "ood_conflict_max": cf["max"],
            "p3_applicable": transient_id,
            "p3_pass": (cf["max"] >= cf["final"] + P3_GAP
                        if transient_id and cf["max"] is not None else None),
            "id_class": id_res.get("class"),
            "id_valid": id_res.get("valid"),
            "breadth": {f: fam_stats(t5, f) for f in BREADTH}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--out", default=None)
    ap.add_argument("--include-step6", action="store_true")
    args = ap.parse_args()

    bad = set(args.seeds) - ALLOWED_SEEDS
    assert not bad, f"seeds {sorted(bad)} are not in the registered " \
                    f"allowlist {sorted(ALLOWED_SEEDS)} (quarantine!)"

    root = Path(args.runs_root)
    cells = list(BASE_CELLS)
    if args.include_step6:
        gov = root / "eval_step6.json"
        dis = root / "rvp5_disjointness.json"
        assert gov.exists(), \
            "REFUSED: no governed Step-6 read on record (eval_step6.json)"
        assert dis.exists() and json.loads(dis.read_text())["ok"], \
            "REFUSED: disjointness gate report missing or not ok"
        cells += STEP6_CELLS

    report = {"cells": {}, "skipped": []}
    for cell in cells:
        rows = []
        for s in args.seeds:
            d = root / f"{cell}_s{s}"
            if not (d / "probe_log_rvp5.jsonl").is_file():
                report["skipped"].append(f"{cell}_s{s}")
                continue
            rows.append(seed_row(d))
        if not rows:
            continue
        n = len(rows)
        id_rec = sum(r["id_class"] == "RECOVERED" and r["id_valid"]
                     for r in rows)
        id_dis = sum(r["id_class"] == "DISPLACED" and r["id_valid"]
                     for r in rows)
        valid = [r for r in rows if r["rvp5_valid"]]
        hi = sum(r["ood_conflict_final"] >= TRANSFER_THR for r in valid)
        lo = sum(r["ood_conflict_final"] <= FAIL_THR for r in valid)
        below = sum(r["ood_conflict_final"] < FAIL_THR for r in valid)
        report["cells"][cell] = {
            "rows": rows, "n_seeds": n,
            "id_recovered_valid": id_rec, "id_displaced_valid": id_dis,
            "n_rvp5_valid": len(valid),
            "p1_applicable": id_rec >= FRAC * n,
            "p1_pass": (hi >= FRAC * len(valid)) if valid else None,
            "p2_applicable": id_dis >= FRAC * n,
            "p2_pass": (lo >= FRAC * len(valid)) if valid else None,
            "falsifier_cell": (id_rec >= FRAC * n and bool(valid)
                               and below >= FRAC * len(valid)),
        }

    C = report["cells"]
    p1_cells = [c for c in C if C[c]["p1_applicable"]]
    p2_cells = [c for c in C if C[c]["p2_applicable"]]
    p3_rows = [r for c in C for r in C[c]["rows"] if r["p3_applicable"]]
    verdicts = {
        "RV5_P1": (all(C[c]["p1_pass"] for c in p1_cells)
                   if p1_cells else "NOT-APPLICABLE"),
        "RV5_P2": (all(C[c]["p2_pass"] for c in p2_cells)
                   if p2_cells else "NOT-APPLICABLE"),
        "RV5_P3": (all(r["p3_pass"] for r in p3_rows)
                   if p3_rows else "NOT-APPLICABLE"),
        "RV5_F_triggered": sum(C[c]["falsifier_cell"] for c in C) >= 2,
        "p1_cells": p1_cells, "p2_cells": p2_cells,
        "n_p3_seeds": len(p3_rows),
    }
    report["verdicts"] = verdicts

    for cell in C:
        st = C[cell]
        vfin = [f"{r['ood_conflict_final']:.2f}" if r["rvp5_valid"]
                else f"({r['ood_conflict_final']:.2f}!)" for r in st["rows"]]
        print(f"{cell}: id_rec+v={st['id_recovered_valid']}/{st['n_seeds']} "
              f"id_dis+v={st['id_displaced_valid']}/{st['n_seeds']} "
              f"rvp5_valid={st['n_rvp5_valid']} ood_cf_final={','.join(vfin)}")
    for k in ("RV5_P1", "RV5_P2", "RV5_P3"):
        v = verdicts[k]
        print(f"{k}: {v if not isinstance(v, bool) else 'PASS' if v else 'FAIL'}")
    print(f"RV5_F: {'TRIGGERED' if verdicts['RV5_F_triggered'] else 'no'}")
    if report["skipped"]:
        print(f"skipped (no rvp5 log): {', '.join(report['skipped'])}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
