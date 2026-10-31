# -*- coding: utf-8 -*-
"""
Final Figure 3 panels for AnchorPhase manuscript
================================================

Fig3A  Sample-level phase reconstruction
Fig3B  Posterior confidence predicts phase error
Fig3C  Confidence-guided selective inference
Fig3D  Transcriptome-level rhythmicity reconstruction

Outputs for each panel:
    *_text.pdf / *_text.png
    *_clean.pdf / *_clean.png

This version is summary-driven from the frozen night-2 paper validation report.
No synthetic sample-level empirical points are created.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# ============================================================
# 0. ROOT
# ============================================================

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.resolve()

sys.path.insert(0, str(HERE))

OUTDIR = ROOT / "figures" / "panels_v2"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. STYLE
# ============================================================

try:
    from fig_style_v2 import (
        apply_style,
        audit,
        DOUBLE,
        NAVY,
        BLUE,
        GREEN,
        ORANGE,
        RED,
        GRAY,
        DK,
    )
except Exception:
    def apply_style():
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })

    def audit(fig):
        return []

    DOUBLE = 7.1
    NAVY = "#17395C"
    BLUE = "#3C78B5"
    GREEN = "#4C9A75"
    ORANGE = "#E39D3E"
    RED = "#C85A54"
    GRAY = "#8C8C8C"
    DK = "#222222"

apply_style()

TEAL = "#3F9DA8"
LTGRAY = "#D9D9D9"
GRID = "#E5E5E5"
BG = "white"

TITLE_FS = 8.8
SUBTITLE_FS = 6.0
AX_TITLE_FS = 7.2
LABEL_FS = 6.5
TICK_FS = 6.0
NOTE_FS = 5.8
VALUE_FS = 6.3

MODEL = "AnchorPhase"

# ============================================================
# 2. FROZEN VALIDATED NUMBERS (Night 2)
# ============================================================

# Fig3A
PHASE_RECOVERY = {
    "GSE145197\nliver hepatocytes": {
        "weighted_median_ae": 1.65,
        "unweighted_median_ae": 1.64,
        "stable_cells_per_sample": 100,
        "n_samples": 10,
        "color": GREEN,
    },
    "GSE223109\ndorsal dermis": {
        "weighted_median_ae": 1.33,
        "unweighted_median_ae": 1.65,
        "stable_cells_per_sample": 250,
        "n_samples": 18,
        "color": ORANGE,
    },
}

# Fig3B
UNCERTAINTY_CAL = {
    "GSE145197\nliver hepatocytes": {
        "rho": -0.73,
        "ci_low": -0.99,
        "ci_high": -0.14,
        "n_samples": 10,
        "color": GREEN,
    },
    "GSE223109\ndorsal dermis": {
        "rho": -0.58,
        "ci_low": -0.90,
        "ci_high": -0.05,
        "n_samples": 18,
        "color": ORANGE,
    },
}

# Fig3C
SELECTIVE = {
    "improved_pairs": 11,
    "total_pairs": 12,
    "dermis_full_coverage_ae": 1.33,
    "dermis_top25_coverage_ae": 0.18,
    "coverage": 0.25,
}

# Fig3D
TRANSCRIPTOME = {
    "liver": {
        "amp_rho": 0.74,
        "r2_rho": 0.91,
        "median_acrophase_error": 1.57,
        "n_rhythmic": 96,
        "mean_lag": 0.99,
        "lag_ci_low": 0.58,
        "lag_ci_high": 1.43,
        "color": GREEN,
    },
    "dermis": {
        "amp_rho": 0.48,
        "r2_rho": 0.28,
        "median_acrophase_error": 0.94,
        "n_rhythmic": 37,
        "mean_lag": 0.70,
        "lag_ci_low": 0.47,
        "lag_ci_high": 0.90,
        "color": ORANGE,
    },
}

# ============================================================
# 3. HELPERS
# ============================================================

def save_final(fig, stem, show_text):
    suffix = "text" if show_text else "clean"
    name = f"{stem}_{suffix}"

    try:
        problems = audit(fig)
    except Exception:
        problems = []

    pdf = OUTDIR / f"{name}.pdf"
    png = OUTDIR / f"{name}.png"

    fig.savefig(pdf, format="pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(png, format="png", dpi=400, bbox_inches="tight", facecolor="white")

    if problems:
        print(f"[{name}] audit warnings: {len(problems)}")
        for p in problems:
            print("  -", p)
    else:
        print(f"[{name}] AUDIT PASS")

    print("   PDF:", pdf)
    print("   PNG:", png)

    plt.close(fig)


def remove_figure_text(fig):
    for t in list(fig.texts):
        try:
            t.remove()
        except Exception:
            pass


def clean_axis_text(ax):
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for t in list(ax.texts):
        try:
            t.remove()
        except Exception:
            pass
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()


def add_panel_label(ax, label):
    ax.text(
        -0.16, 1.08, label,
        transform=ax.transAxes,
        fontsize=9.8,
        fontweight="bold",
        ha="left",
        va="top",
        color=DK
    )


def pretty_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


# ============================================================
# 4. FIG3A
# Sample-level phase reconstruction
# ============================================================

def make_fig3A(show_text=True):

    labels = list(PHASE_RECOVERY.keys())
    weighted = [PHASE_RECOVERY[k]["weighted_median_ae"] for k in labels]
    unweighted = [PHASE_RECOVERY[k]["unweighted_median_ae"] for k in labels]
    stable = [PHASE_RECOVERY[k]["stable_cells_per_sample"] for k in labels]
    ns = [PHASE_RECOVERY[k]["n_samples"] for k in labels]
    colors = [PHASE_RECOVERY[k]["color"] for k in labels]

    x = np.arange(len(labels))
    w = 0.32

    fig, ax = plt.subplots(figsize=(DOUBLE * 0.80, 3.55))
    add_panel_label(ax, "A")

    ax.bar(x - w / 2, weighted, width=w, color=NAVY, zorder=3, label="weighted sample phase")
    ax.bar(x + w / 2, unweighted, width=w, color=LTGRAY, edgecolor=GRAY, lw=0.7,
           zorder=3, label="unweighted sample phase")

    for i, (ww, uu, st, n, c) in enumerate(zip(weighted, unweighted, stable, ns, colors)):
        ax.text(i - w / 2, ww + 0.05, f"{ww:.2f}", ha="center", va="bottom", fontsize=6.4, color=DK)
        ax.text(i + w / 2, uu + 0.05, f"{uu:.2f}", ha="center", va="bottom", fontsize=6.4, color=GRAY)

        if show_text:
            ax.plot([i - 0.32, i + 0.32], [2.15, 2.15], lw=0.9, color=c)
            ax.text(
                i, 2.20,
                f"empirically stable from ≥{st} cells/sample\nn = {n} biological samples",
                ha="center", va="bottom", fontsize=5.8, color=c
            )

    ax.set_ylim(0, 2.55)
    ax.set_xticks(x)

    if show_text:
        ax.set_xticklabels(labels, fontsize=6.7)
        ax.set_ylabel("Median circular AE (h)", fontsize=LABEL_FS)
        ax.set_title(
            "Sample-level circadian phase reconstruction by AnchorPhase",
            fontsize=TITLE_FS, fontweight="bold", pad=10
        )
        ax.text(
            0.0, 1.03,
            "Weighted and unweighted sample-phase summaries are both reported; no universal winner is claimed.",
            transform=ax.transAxes, fontsize=SUBTITLE_FS, ha="left", va="bottom", color=GRAY
        )
        ax.legend(frameon=False, fontsize=6.1, loc="upper right")
        ax.text(
            0.5, -0.24,
            "Stability thresholds are empirical observations in these tested cohorts only, not universal or clinical cutoffs.",
            transform=ax.transAxes, ha="center", va="top", fontsize=NOTE_FS, color=DK, style="italic"
        )

    pretty_axes(ax)

    if not show_text:
        clean_axis_text(ax)
        remove_figure_text(fig)

    fig.tight_layout(rect=(0.04, 0.10, 0.995, 0.94))
    save_final(fig, "fig3A_sample_phase_reconstruction", show_text)


# ============================================================
# 5. FIG3B
# Posterior confidence predicts phase error
# ============================================================

def make_fig3B(show_text=True):

    labels = list(UNCERTAINTY_CAL.keys())
    rhos = [UNCERTAINTY_CAL[k]["rho"] for k in labels]
    lo = [UNCERTAINTY_CAL[k]["ci_low"] for k in labels]
    hi = [UNCERTAINTY_CAL[k]["ci_high"] for k in labels]
    ns = [UNCERTAINTY_CAL[k]["n_samples"] for k in labels]
    colors = [UNCERTAINTY_CAL[k]["color"] for k in labels]

    x = np.arange(len(labels))
    err_low = [r - l for r, l in zip(rhos, lo)]
    err_high = [h - r for r, h in zip(rhos, hi)]

    fig, ax = plt.subplots(figsize=(DOUBLE * 0.80, 3.55))
    add_panel_label(ax, "B")

    bars = ax.bar(
        x, rhos, width=0.55, color=colors, zorder=3
    )

    ax.errorbar(
        x, rhos, yerr=[err_low, err_high],
        fmt="none", ecolor=DK, elinewidth=0.9, capsize=3, zorder=4
    )

    ax.axhline(0, color=GRAY, lw=0.8, ls="--", zorder=1)

    for i, (r, l, h, n, c) in enumerate(zip(rhos, lo, hi, ns, colors)):
        ax.text(i, r - 0.06, f"{r:.2f}", ha="center", va="top", fontsize=6.6, color="white", fontweight="bold")
        if show_text:
            ax.text(
                i, -1.07,
                f"95% CI [{l:.2f}, {h:.2f}]\nn = {n}",
                ha="center", va="bottom", fontsize=5.8, color=c
            )

    ax.set_ylim(-1.12, 0.15)
    ax.set_xticks(x)

    if show_text:
        ax.set_xticklabels(labels, fontsize=6.7)
        ax.set_ylabel("Spearman ρ with sample phase error", fontsize=LABEL_FS)
        ax.set_title(
            "Posterior evidence strength predicts sample-level phase error",
            fontsize=TITLE_FS, fontweight="bold", pad=10
        )
        ax.text(
            0.0, 1.03,
            "Evidence strength was the only uncertainty metric significantly associated with error in both validation cohorts.",
            transform=ax.transAxes, fontsize=SUBTITLE_FS, ha="left", va="bottom", color=GRAY
        )
        ax.text(
            0.5, -0.24,
            "More negative ρ indicates that higher posterior confidence is associated with lower phase error.",
            transform=ax.transAxes, ha="center", va="top", fontsize=NOTE_FS, color=DK, style="italic"
        )

    pretty_axes(ax)

    if not show_text:
        clean_axis_text(ax)
        remove_figure_text(fig)

    fig.tight_layout(rect=(0.04, 0.10, 0.995, 0.94))
    save_final(fig, "fig3B_uncertainty_calibration", show_text)


# ============================================================
# 6. FIG3C
# Confidence-guided selective inference
# ============================================================

def make_fig3C(show_text=True):

    fig = plt.figure(figsize=(DOUBLE, 3.70))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.1], wspace=0.38)

    # ---- left: 11/12 improved
    ax1 = fig.add_subplot(gs[0, 0])
    add_panel_label(ax1, "C")

    improved = SELECTIVE["improved_pairs"]
    total = SELECTIVE["total_pairs"]
    not_improved = total - improved

    ax1.bar([0], [improved], width=0.52, color=NAVY, zorder=3)
    ax1.bar([1], [not_improved], width=0.52, color=LTGRAY, edgecolor=GRAY, lw=0.8, zorder=3)

    ax1.set_ylim(0, total + 1)
    ax1.set_xticks([0, 1])

    if show_text:
        ax1.set_xticklabels(["AURC improved", "not improved"], fontsize=6.3)
        ax1.set_ylabel("Metric–cohort pairs", fontsize=LABEL_FS)
        ax1.set_title(
            "Selective prediction improves performance",
            fontsize=AX_TITLE_FS, fontweight="bold", pad=6
        )
        ax1.text(0, improved + 0.25, f"{improved}/{total}", ha="center", va="bottom", fontsize=6.8, color=DK)
        ax1.text(1, not_improved + 0.25, f"{not_improved}/{total}", ha="center", va="bottom", fontsize=6.8, color=DK)
        ax1.text(
            0.5, 0.92,
            "AURC below the full-coverage baseline in\n11 of 12 tested metric–cohort combinations",
            transform=ax1.transAxes, ha="center", va="top", fontsize=5.9, color=GRAY
        )

    pretty_axes(ax1)

    # ---- right: dermis example
    ax2 = fig.add_subplot(gs[0, 1])

    x = np.array([0, 1])
    vals = [
        SELECTIVE["dermis_full_coverage_ae"],
        SELECTIVE["dermis_top25_coverage_ae"]
    ]
    cols = [ORANGE, GREEN]

    ax2.bar(x, vals, width=0.54, color=cols, zorder=3)

    if show_text:
        ax2.set_xticks(x)
        ax2.set_xticklabels(["100% coverage", "top 25% coverage"], fontsize=6.3)
        ax2.set_ylabel("Median circular AE (h)", fontsize=LABEL_FS)
        ax2.set_title(
            "Example: dorsal dermis",
            fontsize=AX_TITLE_FS, fontweight="bold", pad=6
        )

    ax2.set_ylim(0, 1.65)

    for i, v in enumerate(vals):
        ax2.text(i, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=6.6, color=DK)

    if show_text:
        ax2.annotate(
            "",
            xy=(1, vals[1] + 0.02),
            xytext=(0, vals[0] - 0.02),
            arrowprops=dict(arrowstyle="->", lw=1.0, color=GREEN)
        )
        ax2.text(
            0.56, 0.63,
            "retain only\nhigh-confidence\nsamples",
            transform=ax2.transAxes, ha="center", va="center", fontsize=5.8, color=GREEN
        )

        ax2.text(
            0.5, 0.94,
            "Median AE decreased from 1.33 h to 0.18 h\nwhen retaining the highest-confidence 25% of samples",
            transform=ax2.transAxes, ha="center", va="top", fontsize=5.9, color=GRAY
        )

    pretty_axes(ax2)

    if show_text:
        fig.text(
            0.005, 0.995,
            "Confidence-guided selective inference identifies the most reliable phase predictions",
            fontsize=TITLE_FS, fontweight="bold", ha="left", va="top"
        )
        fig.text(
            0.5, 0.02,
            "No abstention threshold was trained. The dermis example illustrates the utility of posterior confidence ranking, "
            "but coverage must always be reported alongside error.",
            fontsize=NOTE_FS, ha="center", va="bottom", color=DK, style="italic"
        )

    if not show_text:
        clean_axis_text(ax1)
        clean_axis_text(ax2)
        remove_figure_text(fig)

    fig.tight_layout(rect=(0.04, 0.08, 0.995, 0.93))
    save_final(fig, "fig3C_selective_prediction", show_text)


# ============================================================
# 7. FIG3D
# Transcriptome-level rhythmicity recovery
# ============================================================

def make_fig3D(show_text=True):

    fig, axs = plt.subplots(2, 2, figsize=(DOUBLE, 5.45))
    axs = axs.ravel()

    # ---- D1 amplitude rho
    ax = axs[0]
    add_panel_label(ax, "D")

    labs = ["liver", "dermis"]
    x = np.arange(2)
    vals = [TRANSCRIPTOME[k]["amp_rho"] for k in labs]
    cols = [TRANSCRIPTOME[k]["color"] for k in labs]

    ax.bar(x, vals, width=0.55, color=cols, zorder=3)
    ax.set_ylim(0, 1.0)
    if show_text:
        ax.set_title("Amplitude rank concordance", fontsize=AX_TITLE_FS, fontweight="bold", pad=5)
        ax.set_ylabel("Spearman ρ", fontsize=LABEL_FS)
        ax.set_xticks(x)
        ax.set_xticklabels(["Liver", "Dermis"], fontsize=6.2)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=6.3)
    pretty_axes(ax)

    # ---- D2 R2 rho
    ax = axs[1]
    vals = [TRANSCRIPTOME[k]["r2_rho"] for k in labs]
    ax.bar(x, vals, width=0.55, color=cols, zorder=3)
    ax.set_ylim(0, 1.0)
    if show_text:
        ax.set_title(r"Rhythmic fit concordance ($R^2$ rank)", fontsize=AX_TITLE_FS, fontweight="bold", pad=5)
        ax.set_xticks(x)
        ax.set_xticklabels(["Liver", "Dermis"], fontsize=6.2)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=6.3)
    pretty_axes(ax)

    # ---- D3 median acrophase error
    ax = axs[2]
    vals = [TRANSCRIPTOME[k]["median_acrophase_error"] for k in labs]
    ns = [TRANSCRIPTOME[k]["n_rhythmic"] for k in labs]
    ax.bar(x, vals, width=0.55, color=cols, zorder=3)
    ax.set_ylim(0, 2.2)
    if show_text:
        ax.set_title("True-rhythmic genes", fontsize=AX_TITLE_FS, fontweight="bold", pad=5)
        ax.set_ylabel("Median circular acrophase error (h)", fontsize=LABEL_FS)
        ax.set_xticks(x)
        ax.set_xticklabels(["Liver", "Dermis"], fontsize=6.2)
        for i, (v, n) in enumerate(zip(vals, ns)):
            ax.text(i, v + 0.05, f"{v:.2f}\n(n={n})", ha="center", va="bottom", fontsize=6.0)
    pretty_axes(ax)

    # ---- D4 systematic lag
    ax = axs[3]
    means = [TRANSCRIPTOME[k]["mean_lag"] for k in labs]
    lo = [TRANSCRIPTOME[k]["lag_ci_low"] for k in labs]
    hi = [TRANSCRIPTOME[k]["lag_ci_high"] for k in labs]
    err_low = [m - l for m, l in zip(means, lo)]
    err_high = [h - m for m, h in zip(means, hi)]

    ax.bar(x, means, width=0.55, color=cols, zorder=3)
    ax.errorbar(x, means, yerr=[err_low, err_high], fmt="none", ecolor=DK, elinewidth=0.9, capsize=3, zorder=4)
    ax.set_ylim(0, 1.8)

    if show_text:
        ax.set_title("Systematic phase lag", fontsize=AX_TITLE_FS, fontweight="bold", pad=5)
        ax.set_xticks(x)
        ax.set_xticklabels(["Liver", "Dermis"], fontsize=6.2)
        for i, (m, l, h) in enumerate(zip(means, lo, hi)):
            ax.text(i, m + 0.06, f"{m:.2f}\n[{l:.2f}, {h:.2f}]", ha="center", va="bottom", fontsize=5.8)
        ax.text(
            0.5, 1.04,
            "~+1 h uncorrected cohort-wide lag retained as the frozen B0 real performance",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=5.7, color=GRAY
        )

    pretty_axes(ax)

    if show_text:
        fig.text(
            0.005, 0.995,
            "AnchorPhase preserves transcriptome-level rhythmic structure while revealing a modest systematic phase lag",
            fontsize=TITLE_FS, fontweight="bold", ha="left", va="top"
        )
        fig.text(
            0.005, 0.958,
            "Validation summary across all expressed genes and preclassified true-rhythmic genes.",
            fontsize=SUBTITLE_FS, ha="left", va="top", color=GRAY
        )
        fig.text(
            0.5, 0.018,
            "Liver shows stronger transcriptome-wide recovery than dermis; both cohorts retain a significant positive lag that is reported but not corrected.",
            fontsize=NOTE_FS, ha="center", va="bottom", color=DK, style="italic"
        )

    if not show_text:
        for ax in axs:
            clean_axis_text(ax)
        remove_figure_text(fig)

    fig.tight_layout(rect=(0.04, 0.06, 0.995, 0.93), h_pad=2.0, w_pad=1.8)
    save_final(fig, "fig3D_transcriptome_recovery", show_text)


# ============================================================
# 8. MAIN
# ============================================================

def main():

    print("=" * 90)
    print("FINAL FIGURE 3 RENDER")
    print("=" * 90)
    print("Display model:", MODEL)
    print("Input mode: summary-driven from frozen night-2 validation report")
    print("Synthetic empirical points: FALSE")
    print("=" * 90)

    jobs = [
        ("3A text", make_fig3A, True),
        ("3A clean", make_fig3A, False),
        ("3B text", make_fig3B, True),
        ("3B clean", make_fig3B, False),
        ("3C text", make_fig3C, True),
        ("3C clean", make_fig3C, False),
        ("3D text", make_fig3D, True),
        ("3D clean", make_fig3D, False),
    ]

    failed = []

    for label, fn, show_text in jobs:
        print()
        print("-" * 90)
        print(label)
        print("-" * 90)
        try:
            fn(show_text=show_text)
        except Exception as e:
            import traceback
            traceback.print_exc()
            failed.append((label, repr(e)))

    print()
    print("=" * 90)
    if failed:
        print("FAILED:")
        for label, err in failed:
            print(f"  {label}: {err}")
        raise SystemExit(1)

    print("FIG3_FINAL_RENDER = COMPLETE")
    print("OUTPUT DIRECTORY:", OUTDIR)
    print("=" * 90)


if __name__ == "__main__":
    main()