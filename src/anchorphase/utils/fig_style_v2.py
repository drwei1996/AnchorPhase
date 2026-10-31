"""Shared publication style + QA helpers for KATv2 manuscript panels.

All panel scripts import from here so that fonts, sizes, colors, line
widths and the overlap audit stay identical across the whole figure set.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
# Portable local/server root. If scripts live in <ROOT>/scripts, parent is ROOT.
ROOT = os.environ.get("KATV2_ROOT", os.path.dirname(HERE))
PANELS_DIR = os.path.join(ROOT, "figures", "panels_v2")
LOGS_DIR = os.path.join(ROOT, "figures", "logs_v2")
RESULTS = os.path.join(ROOT, "results", "single_cell_phase_rebuild")
T11B = os.path.join(RESULTS, "11B_KAT_V2_SAFE_PHASE_INFORMATION_20261007")
T12 = os.path.join(RESULTS, "12_KAT_V2_EXTERNAL_VALIDATION_20261008")

# unified palette (same logic as the accepted KATv2 schematic)
NAVY = "#1F4E79"; BLUE = "#2E6FB7"; LBLUE = "#EAF1F9"; MBBLUE = "#D7E5F5"
TEAL = "#0E7C7B"; LTEAL = "#E3F2F1"
GREEN = "#2E7D52"; LGREEN = "#E9F4EC"
ORANGE = "#D97B14"; LORANGE = "#FDF1E0"; DORANGE = "#B05E08"
RED = "#B33A2B"; LRED = "#F9E9E7"
GRAY = "#8A8F98"; LGRAY = "#F4F5F7"; DK = "#1A1A1A"

MM = 1.0 / 25.4
SINGLE = 89 * MM    # 3.50 in  — Nature single column
MID = 120 * MM      # 4.72 in  — 1.5 column
DOUBLE = 183 * MM   # 7.20 in  — full width


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.2,
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 0.6,
        "axes.edgecolor": DK,
        "axes.labelcolor": DK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.color": DK, "ytick.color": DK,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.labelsize": 7.4,
        "legend.fontsize": 6.4,
        "legend.frameon": False,
        "savefig.dpi": 400,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def rbox(ax, x, y, w, h, fc="white", ec=NAVY, lw=1.0, dashed=False,
         r=1.6, z=2, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha,
                       linestyle=(0, (4, 2.4)) if dashed else "solid",
                       mutation_aspect=1.0)
    ax.add_patch(p)
    return p


def arrow(ax, x1, y1, x2, y2, color=NAVY, lw=1.4, dashed=False,
          style="-|>", ms=9, z=5, rad=0.0):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=ms, lw=lw, color=color,
                        linestyle=(0, (4, 2.6)) if dashed else "solid",
                        connectionstyle=f"arc3,rad={rad}", zorder=z,
                        shrinkA=1.5, shrinkB=1.5)
    ax.add_patch(a)
    return a


def panel_title(ax, s, color=DK, y=103.5):
    """Left-aligned bold internal panel title."""
    ax.text(1, y, s, fontsize=9.0, fontweight="bold", color=color,
            ha="left", va="bottom")


def chip(ax, x, y, text, ok=True, w=None, fs=5.8):
    """Small PASS/FAIL status chip."""
    col = GREEN if ok else RED
    bg = LGREEN if ok else LRED
    if w is None:
        w = 2.1 + 3.35 * len(text)
    rbox(ax, x, y, w, 6.2, fc=bg, ec=col, lw=0.7, r=1.2)
    ax.text(x + w / 2, y + 3.1, text, fontsize=fs, color=col,
            ha="center", va="center", fontweight="bold")


# ---------------- deterministic layout audit ----------------

def _inter(b1, b2):
    return (max(0, min(b1.x1, b2.x1) - max(b1.x0, b2.x0)) *
            max(0, min(b1.y1, b2.y1) - max(b1.y0, b2.y0)))


def audit(fig):
    """Text-overlap and out-of-bounds audit. Returns list of problems."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    problems = []
    W, H = fig.canvas.get_width_height()

    def texts_of(ax):
        out = []
        for t in ax.texts:
            if t.get_text().strip():
                out.append((t.get_text()[:26].replace("\n", "|"),
                            t.get_window_extent(rend)))
        leg = ax.get_legend()
        if leg is not None:
            out.append(("<legend>", leg.get_window_extent(rend)))
        return out

    for ax in fig.axes:
        ts = texts_of(ax)
        for i in range(len(ts)):
            bb = ts[i][1]
            if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1:
                problems.append(f"out-of-bounds '{ts[i][0]}'")
            for j in range(i + 1, len(ts)):
                if _inter(ts[i][1], ts[j][1]) > 0:
                    problems.append(
                        f"overlap '{ts[i][0]}' x '{ts[j][0]}'")
    # figure-level texts vs everything (titles, annotations)
    fts = [(t.get_text()[:26], t.get_window_extent(rend))
           for t in fig.texts if t.get_text().strip()]
    for i in range(len(fts)):
        bb = fts[i][1]
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1:
            problems.append(f"out-of-bounds figtext '{fts[i][0]}'")
        for j in range(i + 1, len(fts)):
            if _inter(fts[i][1], fts[j][1]) > 0:
                problems.append(f"overlap figtext '{fts[i][0]}' '{fts[j][0]}'")
        for ax in fig.axes:
            for lbl, bb2 in texts_of(ax):
                if _inter(bb, bb2) > 0:
                    problems.append(
                        f"overlap figtext '{fts[i][0]}' x '{lbl}'")
    return problems


def save_panel(fig, name):
    """Audit, then write PDF + 400dpi PNG into figures/panels_v2."""
    os.makedirs(PANELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    problems = audit(fig)
    pdf = os.path.join(PANELS_DIR, name + ".pdf")
    png = os.path.join(PANELS_DIR, name + ".png")
    # Keep the requested physical panel size; do not use bbox_inches='tight',
    # which can silently change panel dimensions across files.
    fig.savefig(pdf, format="pdf", facecolor="white")
    fig.savefig(png, dpi=400, facecolor="white")
    if problems:
        print(f"[{name}] SAVED WITH {len(problems)} AUDIT WARNINGS:")
        for p in problems:
            print("   -", p)
    else:
        print(f"[{name}] saved clean (pdf + png)")
    return problems
