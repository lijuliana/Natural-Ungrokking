"""Seed-42 (or any seed) fidelity check vs v1 paper anchors.

Compares a finished run against reference/v1_table2_anchors.json:
  1. val_bpb (pass --bpb with the value from scripts/eval_bpb.py)
  2. peak / final / drop for the three v1 transient probes
  3. qualitative anchors (close_quote < 0.6 peak, reflexive at chance,
     relative_clause never emerges)
Emits a matched / MISSED / ambiguous verdict per anchor and writes
<run_dir>/replication_summary.md.

  python scripts/check_replication.py runs/v1_repro_s42 --seed 42 [--bpb 1.151]
"""

import argparse
import json
from pathlib import Path

from fogen.analysis.transience import load_probe_log, transience_score

TRANSIENTS = ["end_of_sentence", "modal_continuation", "adjective_order"]


def verdict(ok: bool, ambiguous: bool = False) -> str:
    return "ambiguous" if ambiguous else ("matched" if ok else "**MISSED**")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--seed", required=True)
    ap.add_argument("--bpb", type=float, default=None,
                    help="val_bpb from scripts/eval_bpb.py on the final ckpt")
    ap.add_argument("--split", default="train",
                    help="v1 reported full-battery accs; our train split is closest")
    args = ap.parse_args()

    run = Path(args.run_dir)
    anchors = json.loads(Path("reference/v1_table2_anchors.json").read_text())
    traj = load_probe_log(run / "probe_log.jsonl")
    lines = [f"# Replication check — {run.name} vs v1 anchors", ""]
    misses = 0

    lo, hi = anchors["fidelity"]["val_bpb_range"]
    if args.bpb is not None:
        ok = lo <= args.bpb <= hi
        near = abs(args.bpb - (lo + hi) / 2) <= 0.02
        misses += not (ok or near)
        lines.append(f"- val_bpb = {args.bpb:.4f} vs v1 [{lo}, {hi}]: "
                     f"{verdict(ok, ambiguous=(not ok and near))}")
    else:
        lines.append(f"- val_bpb: NOT PROVIDED (v1 anchor [{lo}, {hi}]) — run eval_bpb.py")

    lines.append("")
    lines.append("| probe | ours peak | v1 peak | ours final | v1 final | ours drop | v1 drop [CI] | verdict |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for probe in TRANSIENTS:
        pts = traj[(probe, args.split)]
        steps, accs = zip(*pts)
        r = transience_score(list(steps), list(accs))
        a = anchors["table2"][probe][str(args.seed)]
        drop = r["peak"] - r["final"]
        # matched if our drop falls inside v1's bootstrap CI and the shape
        # is peak-then-collapse; ambiguous if collapse exists but magnitude
        # differs (reconstructed items differ in difficulty from v1's)
        in_ci = a["drop_ci"][0] <= drop <= a["drop_ci"][1]
        collapses = r["transient"]
        misses += not collapses
        lines.append(
            f"| {probe} | {r['peak']:.2f} @{r['peak_step']} | {a['peak']:.2f} "
            f"[{a['peak_ci'][0]:.2f},{a['peak_ci'][1]:.2f}] | {r['final']:.2f} | "
            f"{a['final']:.2f} | {drop:.2f} | {a['drop']:.2f} "
            f"[{a['drop_ci'][0]:.2f},{a['drop_ci'][1]:.2f}] | "
            f"{verdict(in_ci and collapses, ambiguous=(collapses and not in_ci))} |")

    lines.append("")
    lines.append("Qualitative anchors:")

    def final_acc(probe):
        pts = traj.get((probe, args.split))
        if not pts:
            return None
        accs = [a for _, a in pts]
        return sum(accs[-max(1, len(accs) // 4):]) / max(1, len(accs) // 4)

    def peak_acc(probe):
        pts = traj.get((probe, args.split))
        steps, accs = zip(*pts)
        r = transience_score(list(steps), list(accs))
        return r.get("peak")

    cq = peak_acc("close_quote")
    lines.append(f"- close_quote peak {cq:.2f}; v1: never >= 0.6 — "
                 f"{verdict(cq is not None and cq < 0.6)}")
    rf = peak_acc("reflexive_pronoun")
    lines.append(f"- reflexive_pronoun peak {rf:.2f}; v1: at chance throughout — "
                 f"{verdict(rf is not None and rf < 0.65)}")
    rc = final_acc("relative_clause_agreement")
    lines.append(f"- relative_clause final {rc:.2f}; v1 Table 9: emerges late "
                 f"to final ~0.89 — {verdict(rc is not None and rc > 0.75)}")

    lines.append("")
    lines.append("Table 9 fingerprint (our final vs v1 cross-seed final, train split):")
    lines.append("")
    lines.append("| probe | ours final | v1 final | ours peak | v1 peak |")
    lines.append("|---|---|---|---|---|")
    t9 = anchors["table9_cross_seed"]
    ours_f, v1_f = [], []
    for probe in sorted(k for k in t9 if not k.startswith("_")):
        if (probe, args.split) not in traj:
            continue
        of, pk = final_acc(probe), peak_acc(probe)
        ours_f.append(of); v1_f.append(t9[probe]["final"])
        lines.append(f"| {probe} | {of:.2f} | {t9[probe]['final']:.2f} | "
                     f"{pk:.2f} | {t9[probe]['peak']:.2f} |")
    import numpy as np
    def rank(v):
        v = np.asarray(v, dtype=float)
        order = np.argsort(v)
        r = np.empty(len(v))
        r[order] = np.arange(len(v), dtype=float)
        for val in np.unique(v):       # average ranks over ties
            m = v == val
            r[m] = r[m].mean()
        return r
    ra, rb = rank(ours_f), rank(v1_f)
    rho = float(np.corrcoef(ra, rb)[0, 1])
    lines.append("")
    lines.append(f"Spearman rank correlation of per-probe finals vs v1: "
                 f"rho = {rho:.2f} (n={len(ours_f)}; high rho = same relative "
                 f"difficulty ordering even if absolute levels differ)")
    lines.append("")
    lines.append(f"Hard misses (no peak-then-collapse where v1 had one, or bpb "
                 f"far off): {misses}")
    lines.append("Interpretation rule (pre-stated in RESEARCH_LOG 2026-06-10): "
                 "ceiling-saturated reconstructed items that never collapse "
                 "indicate item-difficulty mismatch, not refuted transience; "
                 "check logprob_diff before concluding.")
    out = run / "replication_summary.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
