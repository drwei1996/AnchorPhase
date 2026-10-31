from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# Figure 1D
# Mapping single cells onto the 24-h circadian phase space
#
# Conceptual / illustrative panel
# NOT empirical data
#
# Script location:
#   sckat_server_package/scripts/
#
# Output:
#   sckat_server_package/figures/panels_v2/
# ============================================================


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

OUTDIR = ROOT / "figures" / "panels_v2"
OUTDIR.mkdir(parents=True, exist_ok=True)

PNG_OUT = OUTDIR / "fig1d_circular_phase_map_v2.png"
PDF_OUT = OUTDIR / "fig1d_circular_phase_map_v2.pdf"


# ------------------------------------------------------------
# Style
# Matched to existing KATv2 panels
# ------------------------------------------------------------
NAVY = "#255784"
BLUE = "#2B6FB8"
TEAL = "#128486"
GREEN = "#2F855A"

TEXT = "#202020"
GREY = "#8C939D"
LIGHT_GREY = "#CBD2D9"
VERY_LIGHT_GREY = "#EEF1F3"

PALE_BLUE = "#E7EFF8"
PALE_TEAL = "#E4F2F1"
PALE_GREEN = "#E8F2EB"

WHITE = "#FFFFFF"


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 22,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "savefig.facecolor": WHITE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def hour_to_theta(hours):
    """
    Convert circadian hours to radians.
    0 h = top; phase increases clockwise.
    """
    return 2.0 * np.pi * np.asarray(hours) / 24.0


def wrapped_arc(center_h, width_h, n=160):
    """
    Arc centered at center_h with total width width_h.
    Returns theta values in radians.
    """
    hrs = np.linspace(
        center_h - width_h / 2.0,
        center_h + width_h / 2.0,
        n
    )
    return hour_to_theta(hrs % 24.0)


# ------------------------------------------------------------
# Synthetic illustrative cells
# ------------------------------------------------------------
# Important:
# These are NOT empirical observations.
# They simply illustrate how KATv2 outputs map onto circular time.
# ------------------------------------------------------------
rng = np.random.default_rng(20261009)

groups = [
    dict(center=3.5,  n=22, sd=0.85, r_low=0.72, r_high=0.93),
    dict(center=9.5,  n=18, sd=1.05, r_low=0.48, r_high=0.77),
    dict(center=15.0, n=21, sd=1.05, r_low=0.58, r_high=0.86),
    dict(center=20.5, n=17, sd=1.25, r_low=0.22, r_high=0.57),
]

phase_h = []
confidence = []

for g in groups:
    phase_h.extend(
        (rng.normal(g["center"], g["sd"], g["n"]) % 24).tolist()
    )
    confidence.extend(
        rng.uniform(g["r_low"], g["r_high"], g["n"]).tolist()
    )

phase_h = np.asarray(phase_h)
confidence = np.asarray(confidence)

# radial jitter only for visualization
radius = 0.77 + rng.uniform(-0.075, 0.075, len(phase_h))

# posterior confidence controls marker size and opacity
point_size = 16 + 54 * confidence
point_alpha = 0.25 + 0.65 * confidence


# ------------------------------------------------------------
# Representative posterior examples
# Narrow -> high confidence
# Broad -> low confidence
# ------------------------------------------------------------
examples = [
    {
        "phase": 3.5,
        "R": 0.86,
        "width_h": 1.4,
        "radius": 1.03,
        "color": GREEN,
        "label": "high confidence",
    },
    {
        "phase": 10.5,
        "R": 0.55,
        "width_h": 4.0,
        "radius": 1.075,
        "color": TEAL,
        "label": "mid confidence",
    },
    {
        "phase": 20.0,
        "R": 0.22,
        "width_h": 7.2,
        "radius": 1.12,
        "color": NAVY,
        "label": "low confidence",
    },
]


# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------
fig = plt.figure(
    figsize=(8.5, 8.5),
    facecolor=WHITE
)

# Main title — same visual language as current panels
fig.text(
    0.055,
    0.958,
    "24-h circular phase map",
    ha="left",
    va="top",
    fontsize=24,
    fontweight="bold",
    color=TEXT,
)

fig.text(
    0.055,
    0.914,
    "mapping single cells onto an anchored circadian phase space",
    ha="left",
    va="top",
    fontsize=12.5,
    color=GREY,
)


# ------------------------------------------------------------
# Main polar plot
# ------------------------------------------------------------
ax = fig.add_axes(
    [0.13, 0.16, 0.74, 0.72],
    projection="polar"
)

ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

ax.set_ylim(0, 1.18)
ax.set_yticks([])

# Circadian clock labels
major_hours = np.array([0, 6, 12, 18])
minor_hours = np.array([3, 9, 15, 21])

ax.set_xticks(hour_to_theta(major_hours))
ax.set_xticklabels(
    [f"{x} h" for x in major_hours],
    fontsize=13,
    color=TEXT,
)

# Custom minor-hour labels
for h in minor_hours:
    theta = hour_to_theta(h)
    ax.text(
        theta,
        1.255,
        f"{h} h",
        ha="center",
        va="center",
        fontsize=11,
        color=GREY,
        clip_on=False,
    )

# Grid
ax.grid(
    color=LIGHT_GREY,
    linewidth=1.0,
    alpha=0.75
)

ax.spines["polar"].set_color(LIGHT_GREY)
ax.spines["polar"].set_linewidth(1.4)


# ------------------------------------------------------------
# Soft phase manifold band
# ------------------------------------------------------------
theta_full = np.linspace(0, 2 * np.pi, 720)

ax.fill_between(
    theta_full,
    np.full_like(theta_full, 0.64),
    np.full_like(theta_full, 0.90),
    color=PALE_TEAL,
    alpha=0.65,
    zorder=0,
)

ax.plot(
    theta_full,
    np.full_like(theta_full, 0.64),
    color=LIGHT_GREY,
    lw=1.0,
    zorder=1,
)

ax.plot(
    theta_full,
    np.full_like(theta_full, 0.90),
    color=LIGHT_GREY,
    lw=1.0,
    zorder=1,
)


# ------------------------------------------------------------
# Single-cell posterior mean phases
# ------------------------------------------------------------
theta_cells = hour_to_theta(phase_h)

for th, rr, size, alpha in zip(
    theta_cells,
    radius,
    point_size,
    point_alpha
):
    ax.scatter(
        th,
        rr,
        s=size,
        color=TEAL,
        alpha=alpha,
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
        clip_on=True,
    )


# ------------------------------------------------------------
# Representative posterior arcs
# ------------------------------------------------------------
for ex in examples:

    theta_arc = wrapped_arc(
        ex["phase"],
        ex["width_h"]
    )

    rr = ex["radius"]

    inner = np.full_like(theta_arc, rr - 0.045)
    outer = np.full_like(theta_arc, rr + 0.018)

    ax.fill_between(
        theta_arc,
        inner,
        outer,
        color=ex["color"],
        alpha=0.13,
        zorder=1,
        clip_on=True,
    )

    ax.plot(
        theta_arc,
        outer,
        color=ex["color"],
        linewidth=2.2,
        solid_capstyle="butt",
        zorder=2,
        clip_on=True,
    )

    ax.scatter(
        hour_to_theta(ex["phase"]),
        rr - 0.012,
        s=66,
        color=ex["color"],
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
        clip_on=True,
    )


# ------------------------------------------------------------
# Center annotation
# ------------------------------------------------------------
ax.text(
    0.5,
    0.535,
    "single-cell\ncircadian\nphase space",
    transform=ax.transAxes,
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
    color=NAVY,
    linespacing=1.10,
)

ax.text(
    0.5,
    0.355,
    "angle = inferred phase\n"
    "posterior concentration = confidence",
    transform=ax.transAxes,
    ha="center",
    va="center",
    fontsize=11.5,
    color=GREY,
    linespacing=1.30,
)


# ------------------------------------------------------------
# Small explanatory labels
# ------------------------------------------------------------
fig.text(
    0.145,
    0.822,
    r"$R_i^{B0}=0.86$",
    color=GREEN,
    fontsize=11.5,
    fontweight="bold",
)

fig.text(
    0.145,
    0.797,
    "narrow posterior",
    color=GREY,
    fontsize=10.5,
)

fig.text(
    0.855,
    0.817,
    r"$R_i^{B0}=0.55$",
    color=TEAL,
    fontsize=11.5,
    fontweight="bold",
    ha="right",
)

fig.text(
    0.855,
    0.792,
    "intermediate posterior",
    color=GREY,
    fontsize=10.5,
    ha="right",
)

fig.text(
    0.835,
    0.232,
    r"$R_i^{B0}=0.22$",
    color=NAVY,
    fontsize=11.5,
    fontweight="bold",
    ha="right",
)

fig.text(
    0.835,
    0.207,
    "broad posterior",
    color=GREY,
    fontsize=10.5,
    ha="right",
)


# ------------------------------------------------------------
# Minimal legend
# ------------------------------------------------------------
legend_handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="none",
        markerfacecolor=TEAL,
        markeredgecolor="white",
        markersize=7,
        label="cell posterior mean phase",
    ),
    Line2D(
        [0], [0],
        color=GREEN,
        linewidth=2.5,
        label="representative posterior arc",
    ),
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.105),
    ncol=2,
    frameon=False,
    fontsize=10.5,
    handlelength=2.2,
    columnspacing=2.0,
)


# ------------------------------------------------------------
# Bottom explanatory line
# ------------------------------------------------------------
fig.text(
    0.5,
    0.065,
    "angular position = inferred circadian phase; "
    "posterior concentration reflects phase confidence",
    ha="center",
    va="center",
    fontsize=10.8,
    color=GREY,
)

fig.text(
    0.5,
    0.035,
    "illustrative schematic; not empirical data",
    ha="center",
    va="center",
    fontsize=9.6,
    color=GREY,
    style="italic",
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
fig.savefig(
    PNG_OUT,
    dpi=400,
    bbox_inches="tight",
    pad_inches=0.08,
    facecolor=WHITE,
)

fig.savefig(
    PDF_OUT,
    bbox_inches="tight",
    pad_inches=0.08,
    facecolor=WHITE,
)

plt.close(fig)

print("=" * 72)
print("Figure 1D complete")
print("=" * 72)
print(f"PNG: {PNG_OUT}")
print(f"PDF: {PDF_OUT}")