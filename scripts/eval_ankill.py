"""Score the registered a/an-kill blind predictions AK1-AK5
(DECISIONS.md 2026-06-11, registered before any ankill run existed).

Reads runs/ankill_p*_s4?/probe_log_rvp31.jsonl, runs/v1_repro_s4?/
probe_log_rvp31.jsonl, runs/ankill_manifests/p*.json. Applies the
frozen gate_a_classify conventions; no new thresholds beyond the two
operationalizations documented below, fixed at scoring time and
logged in DECISIONS.md:
  - AK4 "agree-validity gate to fail at high doses" is scored as: the
    frozen control-validity check fails for det_an_choice in >=2/3
    seeds at the highest dose (p=1.0).
  - AK5 "within noise of v1_repro" is scored as: |dose-cell mean -
    base-cell mean| <= 0.05 for every dose, per family.
Writes runs/eval_ankill.json.

  python scripts/eval_ankill.py
"""

import json
from pathlib import Path

import sys
sys.path.insert(0, "scripts")
from gate_a_classify import classify, load_traj

SEEDS = (42, 43, 44)
DOSES = [(0.5, 500), (0.667, 667), (0.75, 750), (0.9, 900), (1.0, 1000)]
AK5_FAMS = ("pronoun_gender_ref", "negation_bare_verb", "irregular_past")
AK5_TOL = 0.05


def main():
    res = {"v1_repro": [classify(load_traj(f"runs/v1_repro_s{s}", "rvp31"))
                        for s in SEEDS]}
    for _, tag in DOSES:
        res[f"p{tag}"] = [
            classify(load_traj(f"runs/ankill_p{tag}_s{s}", "rvp31"))
            for s in SEEDS]

    def finals(cell, fam):
        return [r[fam]["conflict_final"] for r in res[cell]]

    def mean(cell, fam):
        return sum(finals(cell, fam)) / 3

    cells = ["v1_repro"] + [f"p{t}" for _, t in DOSES]
    da_means = [mean(c, "det_an_choice") for c in cells]

    report = {}
    report["AK1"] = all(a > b for a, b in zip(da_means, da_means[1:]))
    p1000 = res["p1000"]
    report["AK2"] = (
        sum(r["det_an_choice"]["class"] == "DISPLACED" for r in p1000) >= 2
        or sum(f < 0.1 for f in finals("p1000", "det_an_choice")) >= 2)
    report["AK3"] = (
        sum(f >= 0.8 for f in finals("p500", "det_an_choice")) >= 2
        and sum(f <= 0.6 for f in finals("p750", "det_an_choice")) >= 2)
    report["AK4"] = (
        sum(not r["det_an_choice"]["valid"] for r in p1000) >= 2)
    report["AK5"] = all(
        abs(mean(f"p{t}", fam) - mean("v1_repro", fam)) <= AK5_TOL
        for _, t in DOSES for fam in AK5_FAMS)

    out = {
        "registered": "DECISIONS.md 2026-06-11 (pre-run, blind)",
        "report": report,
        "det_an_choice": {
            c: {"finals": finals(c, "det_an_choice"),
                "mean": mean(c, "det_an_choice"),
                "classes": [r["det_an_choice"]["class"] for r in res[c]],
                "agree_finals": [r["det_an_choice"]["agree_final"]
                                 for r in res[c]]}
            for c in cells},
        "a_an_adjective_means": {c: mean(c, "a_an_adjective")
                                 for c in cells},
        "ak5_families": {
            fam: {c: mean(c, fam) for c in cells} for fam in AK5_FAMS},
        "post_kill_ratios": {
            f"p{t}": json.load(
                open(f"runs/ankill_manifests/p{t}.json"))["an_kill"][
                    "post_kill_ratio"]
            for _, t in DOSES},
    }
    Path("runs/eval_ankill.json").write_text(json.dumps(out, indent=1))
    for k, v in report.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")
    print("det_an cell means:",
          " ".join(f"{m:.3f}" for m in da_means))


if __name__ == "__main__":
    main()
