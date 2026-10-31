"""Fig 1a — Inputs and preprocessing (KATv2 manuscript panel).

Single-cell RNA counts -> whole-transcriptome library size -> log offset
o_i = log(L_i / median(L)) -> CORE17 cell vector. Teal schematic panel.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from fig_style_v2 import (apply_style, rbox, arrow, panel_title, save_panel,
                       SINGLE, TEAL, LTEAL, GRAY, DK)

apply_style()
fig = plt.figure(figsize=(SINGLE, 3.90))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 112)
ax.axis("off")
panel_title(ax, "Inputs and preprocessing")

rng = np.random.default_rng(3)

# ---- expression matrix (upper left) ----
mx, my, mw, mh = 3, 42, 42, 46
nrow, ncol = 8, 11
cw, ch = mw / ncol, mh / nrow
for j in range(ncol):
    core = j >= ncol - 3
    for i in range(nrow):
        ax.add_patch(mpatches.Rectangle(
            (mx + j * cw, my + i * ch), cw, ch,
            fc=LTEAL if core else "#FFFFFF",
            ec=TEAL if core else "#C9CFD6", lw=0.45, zorder=2))
for j in range(ncol):
    for i in range(nrow):
        if rng.random() < (0.55 if j >= ncol - 3 else 0.30):
            r = rng.choice([0.30, 0.55, 0.85]) * min(cw, ch) * 0.32
            ax.add_patch(Circle((mx + j * cw + cw / 2, my + i * ch + ch / 2),
                                r, fc=TEAL if j >= ncol - 3 else "#455A64",
                                ec="none", zorder=3))

# CORE17 bracket + labels
bx0 = mx + (ncol - 3) * cw
ax.plot([bx0, bx0, mx + mw, mx + mw],
        [my + mh + 1.0, my + mh + 2.8, my + mh + 2.8, my + mh + 1.0],
        color=TEAL, lw=1.0)
ax.text(bx0 + 1.5 * cw, my + mh + 4.2, "CORE17\ncircadian\nanchor genes",
        fontsize=5.8, color=TEAL, ha="center", va="bottom",
        fontweight="bold")
ax.text(15, my + mh + 4.2, "all genes\n(transcriptome)", fontsize=5.6,
        color=GRAY, ha="center", va="bottom")

# axis labels
ax.text(1.7, my + mh / 2, "cells", rotation=90, fontsize=6.2,
        ha="center", va="center", color=GRAY)
ax.text(mx + mw / 2, my - 2.6, "genes", fontsize=6.2, ha="center",
        va="top", color=GRAY)
ax.text(mx + mw / 2, my - 7.2,
        r"single-cell RNA counts  $X \in \mathbb{N}^{N \times G}$",
        fontsize=6.6, ha="center", va="top", color=DK)

# ---- sparsity inset (upper right) ----
ix, iy, iw, ih = 56, 52, 41, 34
ax.add_patch(mpatches.Rectangle((ix, iy), iw, ih, fc="#FFFFFF", ec=GRAY,
                                lw=0.7, zorder=2))
ax.text(ix + iw / 2, iy + ih + 2.0, "sparse counts,\nheterogeneous cells",
        fontsize=5.6, color=GRAY, ha="center", va="bottom")
for r_, inten in enumerate([0.95, 0.35, 0.65, 0.18]):
    yy = iy + ih - (r_ + 1) * (ih / 4.6) - 1
    for c_ in range(16):
        if rng.random() < inten * 0.75:
            d = 0.35 + inten * 0.55
            ax.add_patch(Circle((ix + 2.5 + c_ * (iw - 5) / 16, yy),
                                d * 0.75, fc="#546E7A", ec="none",
                                zorder=3))
arrow(ax, mx + mw - 1, my + mh - 6, ix - 1, iy + ih - 2,
      color=GRAY, lw=0.8, ms=7)

# ---- preprocessing box (bottom) ----
py = 3
rbox(ax, 2, py, 96, 26, fc=LTEAL, ec=TEAL, lw=1.0)
ax.text(50, py + 20.8, "Preprocessing", fontsize=7.6, fontweight="bold",
        color=TEAL, ha="center")
ax.text(50, py + 14.6,
        r"library size extraction:  $L_i = \sum_{g} x_{ig}$",
        fontsize=6.4, ha="center", va="center", color=DK)
ax.text(50, py + 8.6,
        r"log-library-size offset:  $o_i = \log\!\left( L_i \, / \, \mathrm{median}(L) \right)$",
        fontsize=6.8, ha="center", va="center", color=TEAL,
        fontweight="bold")
ax.text(50, py + 2.8,
        r"extract CORE17 columns  $\rightarrow$  cell-level expression vector $x_i$",
        fontsize=6.4, ha="center", va="center", color=DK)
arrow(ax, 49, 33.0, 49, 29.6, color=TEAL, lw=1.2)

save_panel(fig, "fig1a_inputs_preprocessing_v2")
