"""Shared matplotlib style for paper figures.

Serif (STIX, Times-compatible) to match the ICML body font, thin
axes, Okabe-Ito palette, frameless legends. Styling only — every
number in every figure still comes from frozen artifacts.
"""

import matplotlib.pyplot as plt

# Okabe-Ito, blue/vermillion first for maximum contrast
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#E69F00",
             "#CC79A7", "#56B4E9", "#F0E442", "#000000"]

PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.frameon": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.8, "ytick.major.size": 2.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
}

# data lines ~1.4-1.5pt, reference/baseline lines thin and grey
LW_DATA = 1.4
LW_REF = 0.6
C_REF = "#bbbbbb"


def use_paper_style():
    plt.rcParams.update(PAPER_RC)
