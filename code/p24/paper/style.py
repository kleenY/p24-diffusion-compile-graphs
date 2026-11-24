"""Shared deterministic matplotlib style for the paper figures (S5 P0-4).  CPU only; no torch.

Determinism (goals/S5 item 16): SOURCE_DATE_EPOCH fixed, pdf.fonttype 42, fixed font (DejaVu Sans, shipped with matplotlib),
savefig(metadata={'CreationDate': None}).  Palette = the dataviz categorical order used by reproduce_main_figure (colour-blind
friendly blue / orange / aqua / yellow / ...), never a rainbow map.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1700000000")

ROOT = Path(__file__).resolve().parents[3]                       # <project>/code/p24/paper/style.py -> <project>
FIG_DIR = ROOT / "paper/paper/figs"
TAB_DIR = ROOT / "paper/paper/tables"
COL_W, FULL_W = 3.5, 7.16                                         # IEEEtran single column / full width (in)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]


def mpl():
    """Import matplotlib (Agg) with the fixed rcParams; returns pyplot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["DejaVu Sans"], "pdf.fonttype": 42, "ps.fonttype": 42, "axes.edgecolor": AXIS,
                         "text.color": INK, "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
                         "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6.2, "mathtext.fontset": "dejavusans"})
    return plt


def style_ax(ax):
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
        ax.spines[sp].set_linewidth(0.6)
    ax.grid(True, color=GRID, linewidth=0.5, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, length=2, width=0.5, pad=1.5)


def save_pdf(fig, out: Path):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", metadata={"CreationDate": None}, facecolor="white")
