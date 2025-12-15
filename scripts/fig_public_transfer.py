"""Public-checkpoint transfer figure: focal-rule held-out conflict
accuracy across Pythia (70M-1.4B) and OLMo-1B training revisions,
smoothed with the frozen k=3 convention, steps on a log axis.

Reads runs/<model>/probe_log_rvp31.jsonl (scripts/eval_public_suite.py
artifacts). Agree controls are intentionally not plotted; the caption
reports their status, and the raw logs carry them.

  python scripts/fig_public_transfer.py [--out figures/public_transfer.pdf]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "scripts")
from gate_a_classify import MIN_STEP, smooth  # frozen conventions

# Pythia = sequential blues by scale (light = small); OLMo = neutral
# dashed. Green/vermillion stay reserved for survives/collapses in
# the rest of the paper's figures.
MODELS = [
    ("pythia-70m", "70M", "#9ecae1", "-"),
    ("pythia-160m", "160M", "#6baed6", "-"),
    ("pythia-410m", "410M", "#3182bd", "-"),
    ("pythia-1b", "1B", "#08519c", "-"),
    ("pythia-1.4b", "1.4B", "#08306b", "-"),
    ("olmo-1b", "OLMo-1B", "#555555", "--"),
]
PROBE = "pronoun_gender_ref.conflict"

from fig_style import use_paper_style

use_paper_style()


def series(model):
    rows = [json.loads(l) for l in open(f"runs/{model}/probe_log_rvp31.jsonl")]
    pg = sorted((r for r in rows
                 if r["probe"] == PROBE and r["split"] == "heldout"
                 and r["step"] >= MIN_STEP),
                key=lambda r: r["step"])
    return [r["step"] for r in pg], smooth([r["argmax_acc"] for r in pg])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/public_transfer.pdf")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(3.35, 2.05))
    last_step = 0
    for model, label, color, ls in MODELS:
        steps, acc = series(model)
        ax.plot(steps, acc, color=color, ls=ls, lw=1.1, label=label)
        last_step = max(last_step, steps[-1])

    ax.set_xscale("log")
    ax.set_xlim(right=last_step * 1.15)
    ax.set_ylim(-0.03, 1.06)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("training step (log scale)")
    ax.set_ylabel("held-out conflict acc.")
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="lower right",
              handlelength=1.6, columnspacing=1.0, labelspacing=0.3)
    fig.tight_layout(pad=0.3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_name(out.stem + "_preview.png"), dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
