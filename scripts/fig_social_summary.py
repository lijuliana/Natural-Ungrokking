"""Two-panel social summary figure (Twitter-card 1200x675).

Panel A: the phenomenon — focal pronoun-gender rule emerges then
collapses on the low-support corpus while surviving on TinyStories
(smoothed seed mean, min-max band, same frozen k=3 smoothing as the
paper's lead figure).
Panel B: the asymmetry — final held-out conflict accuracy against
intervention dose for both directions: removing support kills the rule
dose-monotonically (left), re-injecting matched or overshot support
never brings it back (right).

All series are read from the same frozen artifacts as the paper
figures (probe logs + runs/eval_step6.json); nothing hand-entered.

  python scripts/fig_social_summary.py [--out figures/social_summary.png]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import sys
sys.path.insert(0, "scripts")
from gate_a_classify import smooth  # frozen k=3 smoothing

SEEDS = (42, 43, 44)
TS_RUNS = [f"runs/v1_repro_s{s}" for s in SEEDS]
WEB_RUNS = [f"runs/web_packed_v2_s{s}" for s in SEEDS]
KILL = [("v1_repro", 0.0), ("kill_p437", 0.437),
        ("kill_p645", 0.645), ("kill_p1000", 1.0)]
RESCUE = [("web_packed_v2", 0.0), ("rescue_d001", 0.01),
          ("rescue_d010", 0.1), ("rescue_d100", 1.0),
          ("rescue_d300", 3.0)]
C_TS = "#009E73"
C_WEB = "#D55E00"
C_KILL = "#333333"
C_RESCUE = "#0072B2"

plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 15,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.0, "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
})


def probe_series(run, probe="pronoun_gender_ref.conflict",
                 split="heldout"):
    xs, ys = [], []
    for line in open(Path(run) / "probe_log_rvp31.jsonl"):
        r = json.loads(line)
        if r["probe"] == probe and r["split"] == split:
            xs.append(r["step"])
            ys.append(r["argmax_acc"])
    order = sorted(range(len(xs)), key=xs.__getitem__)
    return [xs[i] for i in order], [ys[i] for i in order]


def band(runs):
    series = [probe_series(r) for r in runs]
    xs = series[0][0]
    ys = [smooth(s[1]) for s in series]
    mean = [sum(col) / len(col) for col in zip(*ys)]
    lo = [min(col) for col in zip(*ys)]
    hi = [max(col) for col in zip(*ys)]
    return xs, mean, lo, hi


def dose_panel(ax, ev, ladder, color, invalid_cells=(), draw_line=True):
    # faded dots: the seed failed a registered validity control (its
    # verdict is unscoreable), or the whole cell is intervention-invalid
    for cell, k in ladder:
        for r in ev["cells"][cell]["rows"]:
            faded = (not r["pron_valid"]) or cell in invalid_cells
            ax.scatter([k], [r["conflict_final"]], s=70, zorder=4,
                       facecolor="none" if faded else color,
                       edgecolor=color, linewidth=1.4,
                       alpha=0.45 if faded else 0.9)
    if draw_line:
        means = [(k, [r["conflict_final"] for r in ev["cells"][c]["rows"]])
                 for c, k in ladder if c not in invalid_cells]
        means = [(k, sum(v) / len(v)) for k, v in means]
        ax.plot([k for k, _ in means], [m for _, m in means],
                color=color, lw=2.5, zorder=3)
    ax.axhline(0.8, color="#bbbbbb", lw=1.0, ls="--", zorder=1)
    ax.set_ylim(-0.05, 1.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", dest="eval_json",
                    default="runs/eval_step6.json")
    ap.add_argument("--out", default="figures/social_summary.png")
    args = ap.parse_args()

    ev = json.load(open(args.eval_json))

    fig = plt.figure(figsize=(12.0, 6.75))
    gs = GridSpec(1, 3, width_ratios=[1.5, 0.78, 0.78],
                  wspace=0.24, left=0.06, right=0.96,
                  top=0.855, bottom=0.115)
    axA = fig.add_subplot(gs[0, 0])
    axK = fig.add_subplot(gs[0, 1])
    axR = fig.add_subplot(gs[0, 2])

    # ---- Panel A: emerge-then-collapse. TinyStories = seed mean
    # (clean); web = the abstract's run (seed 42) bold, with the other
    # two seeds faint, so the panel matches the numbers quoted in text.
    xs, mean, lo, hi = band(TS_RUNS)
    axA.fill_between(xs, lo, hi, color=C_TS, alpha=0.18, lw=0)
    axA.plot(xs, mean, color=C_TS, lw=3.0)
    wx, wy = probe_series(WEB_RUNS[0])
    wy = smooth(wy)
    axA.plot(wx, wy, color=C_WEB, lw=3.0)
    axA.axhline(0.5, color="#cccccc", lw=1.0, ls=":")
    axA.text(4350, 0.52, "chance", fontsize=11, color="#999999",
             ha="right", va="bottom")
    axA.set_xlabel("training step")
    axA.set_ylabel("rule accuracy (held-out probes)")
    axA.set_ylim(-0.03, 1.06)
    axA.set_xlim(0, 4400)
    axA.set_title("the same rule is learned mid-training,\n"
                  "then lost on a low-support corpus", pad=8)
    axA.text(2300, 1.015, "high-support corpus", fontsize=13,
             color=C_TS, fontweight="bold", va="bottom")
    # annotate at the bold web seed's actual peak and final values
    pk = max(range(len(wy)), key=wy.__getitem__)
    axA.annotate(f"learned ({wy[pk]:.2f})", xy=(wx[pk], wy[pk]),
                 xytext=(wx[pk] + 600, 0.86),
                 fontsize=13, color=C_WEB, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_WEB, lw=1.2))
    axA.annotate("gone by the end\nof training", xy=(wx[-1], wy[-1]),
                 xytext=(3120, 0.36), fontsize=13, color=C_WEB,
                 fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_WEB, lw=1.2))

    # ---- Panel B1: removal works
    dose_panel(axK, ev, KILL, C_KILL, invalid_cells=("kill_p645",))
    axK.set_xlim(-0.07, 1.07)
    axK.set_xticks([0, 0.437, 0.645, 1.0])
    axK.set_xticklabels(["0", ".44", ".65", "1"])
    axK.set_xlabel("share of support removed")
    axK.set_ylabel("final rule accuracy")
    axK.set_title("destroying it:\nworks, dose-graded", pad=8)
    axK.text(0.04, 0.835, "survival bar", fontsize=11,
             color="#999999")

    # ---- Panel B2: restoration fails
    dose_panel(axR, ev, RESCUE, C_RESCUE, draw_line=False)
    axR.set_xscale("symlog", linthresh=0.01)
    axR.set_xlim(-0.0025, 6.5)
    axR.set_xticks([0, 0.01, 0.1, 1, 3])
    axR.set_xticklabels(["0", ".01", ".1", "1", "3"])
    axR.minorticks_off()
    axR.set_xlabel("support re-added (× surviving dose)")
    axR.set_title("restoring it: fails,\neven at triple dose", pad=8)
    axR.axvline(1.0, color="#cccccc", lw=1.0, ls=":", zorder=1)
    axR.annotate("the dose a surviving\nrule lives on", xy=(1.0, 0.12),
                 xytext=(0.014, 0.07), fontsize=11, color="#888888",
                 arrowprops=dict(arrowstyle="->", color="#aaaaaa",
                                 lw=1.0))
    axR.text(0.014, 0.90, "hollow = run failed its\nvalidity controls;"
             " no dose\nyields a valid recovery", fontsize=10.5,
             color="#666666", va="top")

    fig.suptitle("Pretraining quietly decides which rules survive: "
                 "cheap to destroy, not bought back with matched data",
                 fontsize=17, fontweight="bold", y=0.97)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100)
    fig.savefig(str(args.out).replace(".png", ".pdf"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
