"""Second mechanism measure (post-hoc): decomposition + decodability.

Panel (a): per web seed, the contextual-path contribution C_ctx
(attention + MLP) and the direct-path contribution C_dir to the
pre-softcap she-he gap, with the behavioral collapse onset. C_dir is
flat at zero: the margin lives in the contextual path, and collapse
is that path reversing sign. Defined on all seeds (no CM peak gate).
Panel (b): cross-frame NCM decodability of the cue's gender at the
prediction site (final layer), web vs TinyStories, chance at 0.5.

Reads runs/<run>/mech_decomp.jsonl (scripts/mech_decomp.py) and the
frozen M4 evaluator output for onsets. TinyStories runs still being
rescored are skipped with a notice, so the figure can be regenerated
as results land.

  python scripts/fig_mech_decomp.py [--out figures/mech_decomp.pdf]
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
C_TS = "#009E73"
C_WEB = "#D55E00"
C_DIR = "#555555"

from fig_style import use_paper_style

use_paper_style()


def series(run):
    p = Path(run) / "mech_decomp.jsonl"
    if not p.exists():
        return None
    rows = sorted((json.loads(l) for l in open(p)),
                  key=lambda r: r["step"])
    return ([r["step"] for r in rows],
            smooth([r["C_ctx"] for r in rows]),
            smooth([r["C_dir"] for r in rows]),
            smooth([r["dec_acc"][-1] for r in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/mech_decomp.pdf")
    args = ap.parse_args()

    m4 = {(r["grp"], r["seed"]): r
          for r in json.load(open("runs/eval_m4.json"))["rows"]}

    fig = plt.figure(figsize=(6.9, 2.3))
    gs = GridSpec(3, 2, width_ratios=[1.0, 1.0], hspace=0.5,
                  wspace=0.24, left=0.07, right=0.99, top=0.93,
                  bottom=0.22)
    strips = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    axB = fig.add_subplot(gs[:, 1])

    for ax, seed in zip(strips, SEEDS):
        s = series(f"runs/web_packed_v2_s{seed}")
        steps, cctx, cdir, _ = s
        ax.axhline(0, color="#bbbbbb", lw=0.5)
        ax.plot(steps, cctx, color=C_WEB, lw=1.1)
        ax.plot(steps, cdir, color=C_DIR, lw=0.8)
        ax.axvline(m4[("web", seed)]["collapse_onset"], color="#555555",
                   lw=0.8, alpha=0.9)
        ax.set_xlim(0, 4400)
        ax.set_ylim(-5.0, 4.5)
        ax.set_yticks([-3, 0, 3])
        # late-training C_ctx sits well below zero, so the top-right
        # corner is clear of both curves in every seed strip
        ax.text(0.985, 0.8, f"seed {seed}", transform=ax.transAxes,
                fontsize=6.5, color="#444444", ha="right")
        if seed != SEEDS[-1]:
            ax.tick_params(labelbottom=False)
    strips[0].text(2280, 2.6, r"$C_{\mathrm{ctx}}$", color=C_WEB,
                   fontsize=7.5)
    strips[0].text(700, -3.4, r"$C_{\mathrm{dir}}$", color=C_DIR,
                   fontsize=7.5)
    strips[1].set_ylabel("contribution (pre-softcap nats)", fontsize=8)
    strips[2].set_xlabel("training step")
    strips[2].text(-0.13, -0.62, "(a)", transform=strips[2].transAxes,
                   fontsize=9.5, fontweight="bold")

    missing = []
    for grp, runs, color in (
            ("web", {s: f"runs/web_packed_v2_s{s}" for s in SEEDS}, C_WEB),
            ("TinyStories", {s: f"runs/v1_repro_s{s}" for s in SEEDS},
             C_TS)):
        decs, steps = [], None
        for seed, run in runs.items():
            s = series(run)
            if s is None:
                missing.append(run)
                continue
            steps, _, _, dec = s
            decs.append(dec)
        if not decs:
            continue
        lo = [min(v) for v in zip(*decs)]
        hi = [max(v) for v in zip(*decs)]
        mean = [sum(v) / len(v) for v in zip(*decs)]
        axB.fill_between(steps, lo, hi, color=color, alpha=0.16, lw=0)
        axB.plot(steps, mean, color=color, lw=1.4)
    axB.axhline(0.5, color="#c8c8c8", lw=0.6, ls=":")
    axB.text(4350, 0.515, "chance", color="#999999", fontsize=6.5,
             ha="right", va="bottom")
    axB.set_xlim(0, 4400)
    axB.set_ylim(0.3, 1.04)
    axB.set_yticks([0.5, 0.75, 1.0])
    axB.set_xlabel("training step")
    axB.set_ylabel("cue-gender decodability")
    axB.text(4300, 0.855, "TinyStories", color=C_TS, fontsize=8.5,
             fontweight="bold", ha="right")
    axB.text(3300, 0.40, "web", color=C_WEB, fontsize=8.5,
             fontweight="bold")
    axB.text(-0.10, -0.155, "(b)", transform=axB.transAxes, fontsize=9.5,
             fontweight="bold")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    png = str(args.out).replace(".pdf", "_preview.png")
    fig.savefig(png, dpi=170)
    note = f" (missing, skipped: {missing})" if missing else ""
    print(f"wrote {args.out} and {png}{note}")


if __name__ == "__main__":
    main()
