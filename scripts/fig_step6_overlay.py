"""Step-6 figures: base phase grid, and dose-response of both
interventions (two separate output files; the grid sits in §4.1 next
to the phase table, the dose panels in §4.3).

figures/phase_grid.pdf: the Step-3 base grid in (corpus x D/N) space,
one outcome dot per seed per cell, color+letter coded, faded if
control-invalid in the frozen phase table.

figures/step6_overlay.pdf, panels (a,b): one column per intervention
direction (kill on TinyStories, rescue on web); top strip shows the
per-seed behavioral outcome at each dose including the un-intervened
base, bottom panel shows final CM against the same dose axis (hollow =
fails the CM instrument gate; line = mean over gate-passing seeds).

Inputs (all artifacts; nothing hand-entered at result time):
- runs/phase_table_rvp31.json   Step-3 outcomes (frozen classifier)
- runs/eval_step6.json          frozen evaluator output

  python scripts/fig_step6_overlay.py [--out figures/step6_overlay.pdf]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

N_PARAMS = 11.5e6
TS_TOKENS = 466_769_099
WEB_TOKENS = 341.0e6   # climbmix budget from the rescue dose accounting
# base-grid cells: (x = corpus column, y = unique-data/params)
CELLS = {"packed": (0, TS_TOKENS / N_PARAMS), "dn5": (0, 5.0),
         "dn15": (0, 1.5), "web": (1, WEB_TOKENS / N_PARAMS),
         "webdn5": (1, 5.0), "webdn15": (1, 1.5)}
RESCUE = {"rescue_d001": 0.01, "rescue_d010": 0.1,
          "rescue_d100": 1.0, "rescue_d300": 3.0}
KILL = {"kill_p437": 0.437, "kill_p645": 0.645, "kill_p1000": 1.0}
# Okabe-Ito; R/D match the trajectory figure's corpus colors on purpose
# (green = survives, vermillion = displaced). Letters drawn inside the
# markers are the redundant greyscale channel.
COLOR = {"R": "#009E73", "D": "#D55E00", "P": "#E69F00",
         "N": "#7f7f7f", "U": "#CC79A7"}
LONG = {"RECOVERED": "R", "DISPLACED": "D", "PARTIAL": "P",
        "NEVER": "N", "UNSTABLE": "U"}
C_KILL = "#333333"
C_RESCUE = "#0072B2"

import sys
sys.path.insert(0, "scripts")
from fig_style import use_paper_style

use_paper_style()


def dot(ax, x, y, letter, faded, size=80):
    ax.scatter([x], [y], s=size, marker="o",
               facecolor=COLOR[letter.upper()],
               edgecolor="#999999" if faded else "black",
               linewidth=0.5, zorder=5, alpha=0.45 if faded else 1.0,
               clip_on=False)
    # faded markers get a solid grey letter so the verdict stays
    # readable at column width
    ax.text(x, y, letter.upper(), fontsize=5.5, ha="center", va="center",
            color="#666666" if faded else "white", zorder=6,
            fontweight="bold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", dest="eval_json",
                    default="runs/eval_step6.json")
    ap.add_argument("--phase-table", default="runs/phase_table_rvp31.json")
    ap.add_argument("--out", default="figures/step6_overlay.pdf")
    ap.add_argument("--grid-out", default="figures/phase_grid.pdf")
    args = ap.parse_args()

    pt = json.load(open(args.phase_table))
    ev = json.load(open(args.eval_json))

    # ---- Standalone figure: base phase grid, corpus x D/N (§4.1)
    # Ordinal row spacing (the log axis left a dead band between
    # D/N 5 and 30); rows labeled by their D/N value, packed rows
    # marked, legend in the empty top-left cell.
    ROW = {1.5: 0, 5.0: 1, WEB_TOKENS / N_PARAMS: 2,
           TS_TOKENS / N_PARAMS: 3}
    figA = plt.figure(figsize=(3.25, 2.05))
    axA = figA.add_axes([0.135, 0.13, 0.84, 0.78])
    for cell, (cx, dn) in CELLS.items():
        letters = pt["table"][f"pronoun_gender_ref|{cell}"]
        for i, ch in enumerate(letters):
            dot(axA, cx + (i - 1) * 0.16, ROW[dn], ch,
                faded=ch.islower())
    axA.set_xlim(-0.55, 1.55)
    axA.set_ylim(-0.55, 3.55)
    axA.set_xticks([0, 1])
    axA.set_xticklabels(["TinyStories", "web"])
    axA.set_yticks([0, 1, 2, 3])
    axA.set_yticklabels(["1.5", "5", "30\n(packed)", "40\n(packed)"])
    axA.minorticks_off()
    axA.set_ylabel("unique data / params")
    axA.set_title("focal rule outcome, base grid (no intervention)",
                  fontsize=8, color="#444444", pad=3)
    handles = [Line2D([], [], marker="o", ls="", ms=6, mec="black",
                      mew=0.5, mfc=COLOR[k], label=lbl)
               for k, lbl in (("R", "survives (R)"), ("U", "unstable (U)"),
                              ("D", "displaced (D)"), ("N", "never (N)"))]
    # the web column has no packed-TS row and the TS column no
    # packed-web row, so the off-diagonal corners are empty
    axA.legend(handles=handles, loc="upper right", fontsize=6.5,
               frameon=False, borderpad=0.1, handletextpad=0.3,
               labelspacing=0.45, bbox_to_anchor=(1.005, 1.02), ncol=1,
               columnspacing=1.0)
    Path(args.grid_out).parent.mkdir(parents=True, exist_ok=True)
    figA.savefig(args.grid_out)
    figA.savefig(str(args.grid_out).replace(".pdf", "_preview.png"),
                 dpi=170)

    # ---- Dose-response figure (§4.3)
    fig = plt.figure(figsize=(6.9, 2.45))
    gs = GridSpec(2, 2, height_ratios=[1, 2.1],
                  width_ratios=[1.0, 1.0], hspace=0.12,
                  wspace=0.26, left=0.07, right=0.99, top=0.88,
                  bottom=0.20)
    axKo = fig.add_subplot(gs[0, 0])
    axKc = fig.add_subplot(gs[1, 0], sharex=axKo)
    axRo = fig.add_subplot(gs[0, 1])
    axRc = fig.add_subplot(gs[1, 1], sharex=axRo)

    # ---- Panels (a,b): one column per direction; outcome strip on top
    # of the CM dose-response, sharing the dose axis.
    for axo, axc, cells_knob, base, color, title, lab, panel in (
            (axKo, axKc, KILL, "v1_repro", C_KILL,
             "removal: pronoun flips in TinyStories",
             "kill flip rate $p$", "(a)"),
            (axRo, axRc, RESCUE, "web_packed_v2", C_RESCUE,
             "restoration: support injected into web",
             r"rescue dose ($\times\,\delta_{\mathrm{TS}}$)", "(b)")):
        ladder = [(base, 0.0)] + sorted(cells_knob.items(),
                                        key=lambda kv: kv[1])
        for cell, k in ladder:
            rows = ev["cells"][cell]["rows"]
            for i, r in enumerate(rows):
                dot(axo, k, 2 - i, LONG[r["pron_class"]],
                    faded=(not r["pron_valid"]) or cell == "kill_p645",
                    size=52)
                axc.scatter([k], [r["cm_final"]], s=16, zorder=4,
                            facecolor=color if r["cm_valid"] else "none",
                            edgecolor=color, linewidth=0.9,
                            alpha=0.85 if r["cm_valid"] else 0.55)
        means = [(k, [r["cm_final"] for r in ev["cells"][c]["rows"]
                      if r["cm_valid"]]) for c, k in ladder]
        means = [(k, sum(v) / len(v)) for k, v in means if v]
        axc.plot([k for k, _ in means], [m for _, m in means],
                 color=color, lw=1.3, zorder=3)
        axc.axhline(0, color="#bbbbbb", lw=0.6)
        axo.set_ylim(-0.7, 2.7)
        axo.set_yticks([2, 1, 0])
        axo.set_yticklabels(["s42", "s43", "s44"], fontsize=6)
        axo.tick_params(length=0, labelbottom=False)
        for s in ("left", "bottom"):
            axo.spines[s].set_visible(False)
        axo.set_title(title, fontsize=8, pad=3)
        axc.set_xlabel(lab, labelpad=1.5)
        axc.text(-0.16, -0.42, panel, transform=axc.transAxes,
                 fontsize=9.5, fontweight="bold")

    axKc.set_ylabel("final CM (nats)")
    axRc.set_ylabel("final CM (nats)")
    axKo.set_xlim(-0.07, 1.07)
    axKc.set_xticks([0, 0.437, 0.645, 1.0])
    # the p=0.645 cell is intervention-invalid (registered KS gate);
    # its behavioral verdicts are voided, margins are not
    axKc.set_xticklabels(["0", ".437", ".645$^\\dag$", "1"])
    axKc.set_ylim(-4.7, 4.9)
    axRo.set_xscale("symlog", linthresh=0.01)
    axRo.set_xlim(-0.0025, 6.5)
    axRc.set_xticks([0, 0.01, 0.1, 1, 3])
    axRc.set_xticklabels(["0", ".01", ".1", "1", "3"])
    axRc.set_ylim(-0.8, 0.8)
    axRc.set_yticks([-0.5, 0, 0.5])
    axRc.annotate("dose that sustains\nthe rule in TS", xy=(1.0, 0.62),
                  xytext=(0.035, 0.42), fontsize=6, color="#888888",
                  arrowprops=dict(arrowstyle="->", color="#aaaaaa",
                                  lw=0.6))
    axRc.axvline(1.0, color="#cccccc", lw=0.6, ls=":", zorder=1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    png = str(args.out).replace(".pdf", "_preview.png")
    fig.savefig(png, dpi=170)
    print(f"wrote {args.out} and {png}")


if __name__ == "__main__":
    main()
