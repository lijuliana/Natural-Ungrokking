"""Lead figure: emerge-then-collapse trajectories of the focal rule.

Panel (a): pronoun_gender_ref conflict accuracy (heldout) over training,
TinyStories vs web, smoothed seed mean with a min-max band across seeds;
agree-condition controls as one light line per corpus. Panel (b): small
multiples, one strip per web seed, each with its smoothed contrast
margin, behavioral collapse onset (vertical line), and CM zero crossing
(dot), so the crossing/onset coincidence is checkable seed by seed.
Everything is read from per-run probe/margin logs and the frozen M4
evaluator output; no hand-entered numbers.

  python scripts/fig_trajectories.py [--out figures/trajectories.pdf]
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
from gate_a_classify import smooth  # frozen classifier smoothing, k=3

SEEDS = (42, 43, 44)
TS_RUNS = {s: f"runs/v1_repro_s{s}" for s in SEEDS}
WEB_RUNS = {s: f"runs/web_packed_v2_s{s}" for s in SEEDS}
C_TS = "#009E73"     # Okabe-Ito green   = survives
C_WEB = "#D55E00"    # Okabe-Ito vermillion = displaced
C_GREY = "#9a9a9a"

from fig_style import use_paper_style

use_paper_style()


def probe_series(run, probe, split="heldout"):
    xs, ys = [], []
    for line in open(Path(run) / "probe_log_rvp31.jsonl"):
        r = json.loads(line)
        if r["probe"] == probe and r["split"] == split:
            xs.append(r["step"])
            ys.append(r["argmax_acc"])
    order = sorted(range(len(xs)), key=xs.__getitem__)
    return [xs[i] for i in order], [ys[i] for i in order]


def margin_series(run):
    xs, ys = [], []
    for line in open(Path(run) / "mech_margins.jsonl"):
        r = json.loads(line)
        xs.append(r["step"])
        ys.append(r["CM"])
    order = sorted(range(len(xs)), key=xs.__getitem__)
    return [xs[i] for i in order], [ys[i] for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/trajectories.pdf")
    args = ap.parse_args()

    fig = plt.figure(figsize=(6.9, 2.45))
    gs = GridSpec(2, 2, width_ratios=[1.25, 1.0], hspace=0.35,
                  wspace=0.26, left=0.075, right=0.995, top=0.96,
                  bottom=0.21)
    axA = fig.add_subplot(gs[:, 0])
    strips = [fig.add_subplot(gs[i, 1]) for i in range(2)]
    # panel (b) shows only the two instrument-valid seeds; the third
    # (seed 44, fails the registered gate) is reported in the text
    B_SEEDS = (42, 43)

    m4 = {(r["grp"], r["seed"]): r
          for r in json.load(open("runs/eval_m4.json"))["rows"]}

    # ---- Panel (a): TinyStories = seed mean + min-max band; web =
    # per-seed traces with the focal seed (the run the abstract
    # quotes) emphasized. Peak annotation is computed from the same
    # probe log; nothing hand-entered.
    for runs, color, name in ((TS_RUNS, C_TS, "TinyStories"),
                              (WEB_RUNS, C_WEB, "web")):
        acc, agr, steps = [], [], None
        for run in runs.values():
            xs, ys = probe_series(run, "pronoun_gender_ref.conflict")
            xa, ya = probe_series(run, "pronoun_gender_ref.agree")
            steps = xs
            acc.append(smooth(ys))
            agr.append(smooth(ya))
        agr_mean = [sum(v) / len(v) for v in zip(*agr)]
        if name == "TinyStories":
            lo = [min(v) for v in zip(*acc)]
            hi = [max(v) for v in zip(*acc)]
            mean = [sum(v) / len(v) for v in zip(*acc)]
            axA.fill_between(steps, lo, hi, color=color, alpha=0.16,
                             lw=0)
            axA.plot(steps, mean, color=color, lw=1.4)
        else:
            for other in acc[1:]:
                axA.plot(steps, other, color=color, lw=0.7, alpha=0.3)
            focal = acc[0]   # seed 42, first in SEEDS
            axA.plot(steps, focal, color=color, lw=1.4)
            ipk = max(range(len(focal)), key=focal.__getitem__)
            axA.plot([steps[ipk]], [focal[ipk]], marker="o", ms=3.2,
                     mfc="white", mec=color, mew=0.9, zorder=6)
            # white pads lift the labels off the busy line work
            pad = dict(facecolor="white", alpha=0.8,
                       edgecolor="none", pad=1.4)
            axA.annotate(f"learned ({focal[ipk]:.2f})",
                         xy=(steps[ipk] + 30, focal[ipk] + 0.012),
                         xytext=(1480, 1.055), fontsize=6.5,
                         color="#666666", zorder=7, bbox=pad,
                         arrowprops=dict(arrowstyle="->",
                                         color="#aaaaaa", lw=0.6))
            axA.annotate(f"gone ({focal[-1]:.2f})",
                         xy=(steps[-1] - 40, focal[-1] + 0.012),
                         xytext=(3500, 0.33), fontsize=6.5,
                         color="#666666", zorder=7, bbox=pad,
                         arrowprops=dict(arrowstyle="->",
                                         color="#aaaaaa", lw=0.6))
        axA.plot(steps, agr_mean, color=color, lw=0.8, ls=(0, (1, 1.2)),
                 alpha=0.55)

    axA.axhline(0.5, color="#c8c8c8", lw=0.6, ls=":", zorder=1)
    axA.text(4350, 0.515, "chance", color="#999999", fontsize=6.5,
             ha="right", va="bottom")
    axA.set_xlabel("training step")
    axA.set_ylabel("conflict accuracy (held out)")
    axA.set_xlim(0, 4400)
    axA.set_ylim(-0.02, 1.13)
    axA.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axA.text(2450, 0.93, "TinyStories", color=C_TS, ha="left",
             fontsize=8.5, fontweight="bold")
    axA.text(4300, 0.07, "web", color=C_WEB, ha="right", fontsize=8.5,
             fontweight="bold")
    axA.annotate("agree controls", xy=(3400, 0.985), xytext=(2710, 0.74),
                 fontsize=6.5, color="#888888",
                 bbox=dict(facecolor="white", alpha=0.8,
                           edgecolor="none", pad=1.4),
                 arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5))
    axA.text(-0.115, -0.16, "(a)", transform=axA.transAxes, fontsize=9.5,
             fontweight="bold")

    # ---- Panel (b): one strip per instrument-valid web seed
    for ax, seed in zip(strips, B_SEEDS):
        row = m4[("web", seed)]
        xm, ym = margin_series(WEB_RUNS[seed])
        sm = smooth(ym)
        ax.axhline(0, color="#bbbbbb", lw=0.5)
        ax.plot(xm, sm, color=C_WEB, lw=1.1)
        ax.axvline(row["collapse_onset"], color="#555555", lw=0.8,
                   alpha=0.9)
        cross = row.get("cm_zero_cross")
        if cross is not None:
            ax.plot([cross], [0], marker="o", ms=3.5, color="black",
                    zorder=5, clip_on=False)
        ax.set_xlim(0, 4400)
        ax.set_ylim(-1.6, 1.6)
        ax.set_yticks([-1, 0, 1])
        ax.set_ylabel("CM (nats)", fontsize=8)
        ax.text(0.015, 0.86, f"seed {seed}", transform=ax.transAxes,
                fontsize=7, color="#444444")
        if seed != B_SEEDS[-1]:
            ax.tick_params(labelbottom=False)
    strips[0].text(0.99, 0.86, "onset", transform=strips[0].transAxes,
                   fontsize=6.5, color="#555555", ha="right")
    strips[1].set_xlabel("training step")
    strips[1].text(-0.135, -0.36, "(b)", transform=strips[1].transAxes,
                   fontsize=9.5, fontweight="bold")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    png = str(args.out).replace(".pdf", "_preview.png")
    fig.savefig(png, dpi=170)
    print(f"wrote {args.out} and {png}")


if __name__ == "__main__":
    main()
