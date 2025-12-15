"""Gate A outcome classification (metric registered 2026-06-10, RESEARCH_LOG).

Reads probe_log_rvp1.jsonl per run, applies the registered metric to the
heldout split, prints per-(family x cell) classifications and the gate
verdict. Registered constants are deliberately hardcoded here to match the
log entry; do not tune them post hoc.

  python scripts/gate_a_classify.py runs/v1_repro_s42 runs/databudget_dn5_s42 \
      runs/databudget_dn15_s42
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

SMOOTH_K = 3
EMERGED_THR = 0.8
FINAL_FRAC = 0.10
FINAL_MIN_CKPTS = 3
RECOVERED_THR = 0.8
DISPLACED_THR = 0.6
AGREE_VALID_THR = 0.8
AGREE_DROP_THR = 0.15
MIN_STEP = 100
# UNSTABLE amendment v2 (RESEARCH_LOG 2026-06-10, post-hoc, anti-
# confirmatory): smoothed conflict range >= 0.2 over the SAME final-10%
# window that defines `final` -> UNSTABLE (still moving at the stop point;
# the final-state label would depend on where training stopped). v1 of the
# rule (range >= 0.3 over final 25%) misfired on settled trajectories with
# mid-tail wander; both forms and their disagreements are logged.
UNSTABLE_RANGE = 0.2
EXPLORATORY = {"comparative_er"}
ANCHOR = "pronoun_gender_ref"


def smooth(xs, k=SMOOTH_K):
    out = []
    for i in range(len(xs)):
        lo, hi = max(0, i - k // 2), min(len(xs), i + k // 2 + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out


def final_mean(xs):
    n = max(FINAL_MIN_CKPTS, int(round(len(xs) * FINAL_FRAC)))
    return sum(xs[-n:]) / n


def load_traj(run_dir, tag="rvp1"):
    """-> {(family, condition): [(step, heldout_acc), ...] sorted}"""
    rows = defaultdict(dict)
    for line in (Path(run_dir) / f"probe_log_{tag}.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("split") != "heldout" or r["step"] < MIN_STEP:
            continue
        fam, cond = r["probe"].rsplit(".", 1)
        rows[(fam, cond)][r["step"]] = r["argmax_acc"]
    return {k: sorted(v.items()) for k, v in rows.items()}


def classify(traj):
    """traj: {(family, condition): [(step, acc)]} -> {family: dict}"""
    fams = sorted({f for f, _ in traj})
    out = {}
    for fam in fams:
        res = {}
        for cond in ("conflict", "agree"):
            steps_accs = traj.get((fam, cond), [])
            accs = smooth([a for _, a in steps_accs])
            res[cond] = {"final": final_mean(accs) if accs else float("nan"),
                         "max": max(accs) if accs else float("nan"),
                         "emerged": bool(accs) and max(accs) >= EMERGED_THR}
        ag = res["agree"]
        valid = (ag["final"] >= AGREE_VALID_THR
                 and ag["max"] - ag["final"] < AGREE_DROP_THR)
        cf = res["conflict"]
        cf_accs = smooth([a for _, a in traj.get((fam, "conflict"), [])])
        nf = max(FINAL_MIN_CKPTS, int(round(len(cf_accs) * FINAL_FRAC)))
        tail = cf_accs[-nf:]
        if tail and max(tail) - min(tail) >= UNSTABLE_RANGE:
            cls = "UNSTABLE"
        elif cf["final"] >= RECOVERED_THR:
            cls = "RECOVERED"
        elif cf["emerged"] and cf["final"] <= DISPLACED_THR:
            cls = "DISPLACED"
        elif cf["emerged"]:
            cls = "PARTIAL"
        else:
            cls = "NEVER"
        out[fam] = {"class": cls, "valid": valid,
                    "conflict_final": cf["final"], "conflict_max": cf["max"],
                    "agree_final": ag["final"], "agree_max": ag["max"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="+", metavar="name=run_dir",
                    help="e.g. packed=runs/v1_repro_s42 web=runs/web_packed_s42")
    ap.add_argument("--tag", default="rvp1")
    ap.add_argument("--gate", nargs=2, metavar=("HIGH", "LOW"), default=None,
                    help="amended gate (RESEARCH_LOG 2026-06-10): count "
                         "families RECOVERED in HIGH but DISPLACED/NEVER in "
                         "LOW, control-valid in both. Default: first two cells")
    args = ap.parse_args()

    pairs = [c.split("=", 1) for c in args.cells]
    cells = [n for n, _ in pairs]
    results = {n: classify(load_traj(d, args.tag)) for n, d in pairs}
    high, low = args.gate or cells[:2]

    fams = sorted(results[cells[0]])
    hdr = f"{'family':22s} " + "".join(f"{c:>26s}" for c in cells)
    print(hdr)
    for fam in fams:
        row = f"{fam:22s} "
        for c in cells:
            r = results[c][fam]
            v = "" if r["valid"] else "!CTRL"
            row += f"{r['class']:>10s}({r['conflict_final']:.2f}){v:>6s}    "
        flag = " [exploratory]" if fam in EXPLORATORY else ""
        print(row + flag)

    anchor_ok = all(results[c][ANCHOR]["class"] == "RECOVERED"
                    and results[c][ANCHOR]["valid"] for c in cells)
    print(f"\nanchor ({ANCHOR}) RECOVERED+valid everywhere: {anchor_ok}")

    passing, judgment = [], []
    for fam in fams:
        if fam in EXPLORATORY:
            continue
        h, l = results[high][fam], results[low][fam]
        if not (h["valid"] and l["valid"]):
            continue
        if h["class"] == "RECOVERED" and l["class"] in ("DISPLACED", "NEVER"):
            passing.append(fam)
        elif (l["class"] == "PARTIAL"
              and h["conflict_final"] - l["conflict_final"] >= 0.2):
            judgment.append(fam)

    print(f"PASS families (RECOVERED@{high}, DISPLACED/NEVER@{low}, valid): {passing}")
    print(f"JUDGMENT-ZONE families (PARTIAL@{low}, gap>=0.2): {judgment}")
    if len(passing) >= 2:
        print("GATE A: PASS")
    elif len(passing) == 1 or len(judgment) >= 2:
        print("GATE A: JUDGMENT ZONE (proceed, flag prominently)")
    else:
        print("GATE A: 0 passing families in this comparison (kill requires "
              "zero control-valid corpus/budget-dependent outcomes across ALL "
              "cells — judge from the full table, per the registered rule)")


if __name__ == "__main__":
    main()
