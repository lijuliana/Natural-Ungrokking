"""Plot per-probe accuracy trajectories from a run's probe_log.jsonl and emit
a transience summary table (markdown).

  python scripts/plot_trajectories.py runs/v1_repro_s42 [--split train]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fogen.analysis.transience import load_probe_log, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--split", default="train", choices=["train", "heldout"])
    args = ap.parse_args()
    run = Path(args.run_dir)
    traj = load_probe_log(run / "probe_log.jsonl")

    probes = sorted({p for (p, s) in traj if s == args.split})
    n = len(probes)
    ncol = 4
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow),
                             squeeze=False)
    for i, probe in enumerate(probes):
        ax = axes[i // ncol][i % ncol]
        for split, style in (("train", "-"), ("heldout", "--")):
            pts = traj.get((probe, split))
            if pts:
                steps, accs = zip(*pts)
                ax.plot(steps, accs, style, lw=1, label=split)
        ax.axhline(0.5, color="gray", lw=0.5)
        ax.set_title(probe, fontsize=8)
        ax.set_ylim(0, 1)
        if i == 0:
            ax.legend(fontsize=6)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.tight_layout()
    out_png = run / f"trajectories_{args.split}.png"
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")

    rows = summarize(run / "probe_log.jsonl")
    md = [f"# Transience summary — {run.name}", "",
          "| probe | split | peak | peak_step | final | transience | transient? |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        if "peak" not in r:
            continue
        md.append(f"| {r['probe']} | {r['split']} | {r['peak']:.2f} | "
                  f"{r['peak_step']} | {r['final']:.2f} | {r['transience']:.2f} | "
                  f"{'**YES**' if r['transient'] else 'no'} |")
    out_md = run / "transience_summary.md"
    out_md.write_text("\n".join(md) + "\n")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
