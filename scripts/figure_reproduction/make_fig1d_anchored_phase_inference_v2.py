from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from matplotlib.colors import to_rgba

# ============================================================
# Figure 1D — clean publication layout
# Anchored circular phase inference
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

OUTDIR = ROOT / "figures" / "panels_v2"
OUTDIR.mkdir(parents=True, exist_ok=True)

PNG_OUT = OUTDIR / "fig1d_anchored_phase_inference_v2.png"
PDF_OUT = OUTDIR / "fig1d_anchored_phase_inference_v2.pdf"

# ------------------------------------------------------------
# Style
# ------------------------------------------------------------
NAVY = "#255784"
TEAL = "#128486"
GREEN = "#2F855A"

TEXT = "#202020"
GREY = "#8C939D"
LIGHT_GREY = "#D5DCE3"
PALE_BLUE = "#E8F0F8"
PALE_TEAL = "#E6F2F2"
WHITE = "#FFFFFF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "savefig.facecolor": WHITE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------
CX = 0.50
CY = 0.525

# much smaller center
R_CENTER = 0.080

# posterior ring
R_POST_IN = 0.205
R_POST_OUT = 0.325

# clock outline
R_CLOCK = 0.365

# anchor ticks
R_ANCHOR_1 = 0.377
R_ANCHOR_2 = 0.400


def hour_to_deg(h):
    # 0 h at top, clockwise
    return 90.0 - 15.0 * h


def hour_to_rad(h):
    return np.deg2rad(hour_to_deg(h))


def xy(h, r):
    a = hour_to_rad(h)
    return CX + r * np.cos(a), CY + r * np.sin(a)


# ============================================================
# Illustrative posterior
# ============================================================
phase_grid_h = np.arange(0, 24, 0.5)
theta = 2 * np.pi * phase_grid_h / 24.0

target_phase_h = 14.5
target_theta = 2 * np.pi * target_phase_h / 24.0
target_R = 0.86


def posterior(kappa):
    logq = kappa * np.cos(theta - target_theta)
    logq -= logq.max()
    q = np.exp(logq)
    return q / q.sum()


best_q = None
best_kappa = None
best_diff = np.inf

for kappa in np.linspace(0.05, 20, 5000):
    q0 = posterior(kappa)
    z0 = np.sum(q0 * np.exp(1j * theta))
    diff = abs(abs(z0) - target_R)

    if diff < best_diff:
        best_diff = diff
        best_q = q0
        best_kappa = kappa

q = best_q
z = np.sum(q * np.exp(1j * theta))

phase_hat = (np.angle(z) % (2*np.pi)) * 24 / (2*np.pi)
R_hat = abs(z)

q_scaled = q / q.max()

# ------------------------------------------------------------
# Correct representative frozen anchors
# ------------------------------------------------------------
# Frozen B0 anchor convention:
# ARNTL = 0 h
# NR1D1 = 8 h
# PER2  = 14 h
# CRY1  = 18 h
anchors = [
    ("ARNTL", 0.0),
    ("NR1D1", 8.0),
    ("PER2", 14.0),
    ("CRY1", 18.0),
]

# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------
fig = plt.figure(figsize=(8.3, 7.9), facecolor=WHITE)

ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.axis("off")

# ============================================================
# Title block
# ============================================================
ax.text(
    0.055, 0.955,
    "Anchored circular phase inference",
    ha="left",
    va="top",
    fontsize=22,
    fontweight="bold",
    color=TEXT
)

ax.text(
    0.055, 0.918,
    "48-bin posterior over biologically anchored circadian time",
    ha="left",
    va="top",
    fontsize=12.5,
    color=GREY
)

# ============================================================
# Clock outline
# ============================================================
ax.add_patch(
    Circle(
        (CX, CY),
        R_CLOCK,
        facecolor="none",
        edgecolor=LIGHT_GREY,
        linewidth=1.4,
        zorder=1
    )
)

# ============================================================
# Posterior ring background
# ============================================================
ax.add_patch(
    Wedge(
        (CX, CY),
        R_POST_OUT,
        0, 360,
        width=R_POST_OUT - R_POST_IN,
        facecolor=PALE_TEAL,
        edgecolor="none",
        zorder=1
    )
)

# ============================================================
# Posterior ring: 48 bins
# ============================================================
bin_deg = 360 / 48

for h, val in zip(phase_grid_h, q_scaled):
    cdeg = hour_to_deg(h)

    alpha = 0.10 + 0.85 * (val ** 0.80)

    ax.add_patch(
        Wedge(
            (CX, CY),
            R_POST_OUT,
            cdeg - bin_deg/2 + 0.30,
            cdeg + bin_deg/2 - 0.30,
            width=R_POST_OUT - R_POST_IN,
            facecolor=to_rgba(TEAL, alpha),
            edgecolor=WHITE,
            linewidth=0.42,
            zorder=3
        )
    )

# borders
for r in [R_POST_IN, R_POST_OUT]:
    ax.add_patch(
        Circle(
            (CX, CY), r,
            facecolor="none",
            edgecolor=LIGHT_GREY,
            linewidth=0.9,
            zorder=4
        )
    )

# ============================================================
# Four major clock axes
# ============================================================
for h in [0, 6, 12, 18]:
    x1, y1 = xy(h, R_CENTER + 0.012)
    x2, y2 = xy(h, R_POST_OUT)

    ax.plot(
        [x1, x2], [y1, y2],
        color=LIGHT_GREY,
        linewidth=0.8,
        zorder=1
    )

# ============================================================
# Clock labels — deliberately separated from gene labels
# ============================================================
time_positions = {
    0:  (0.50, 0.865, "center", "bottom"),
    6:  (0.895, CY, "left", "center"),
    12: (0.50, 0.142, "center", "top"),
    18: (0.105, CY, "right", "center"),
}

for h, (xx, yy, ha, va) in time_positions.items():
    ax.text(
        xx, yy,
        f"{h} h",
        ha=ha,
        va=va,
        fontsize=13,
        color=TEXT,
        zorder=20
    )

# ============================================================
# Anchor ticks + labels
# No anchor shares the same text position as time labels
# ============================================================
anchor_text_pos = {
    "ARNTL": (0.415, 0.844, "right", "center"),
    "NR1D1": (0.847, 0.315, "left", "center"),
    "PER2":  (0.360, 0.170, "right", "center"),
    "CRY1":  (0.145, 0.575, "right", "center"),
}

for gene, h in anchors:

    x1, y1 = xy(h, R_ANCHOR_1)
    x2, y2 = xy(h, R_ANCHOR_2)

    ax.plot(
        [x1, x2], [y1, y2],
        color=TEAL,
        linewidth=2.2,
        solid_capstyle="round",
        zorder=10
    )

    ax.scatter(
        [x2], [y2],
        s=28,
        color=TEAL,
        edgecolor=WHITE,
        linewidth=0.5,
        zorder=11
    )

    tx, ty, ha, va = anchor_text_pos[gene]

    ax.text(
        tx, ty,
        gene,
        ha=ha,
        va=va,
        fontsize=11.5,
        fontweight="bold",
        color=TEAL,
        zorder=12
    )

# ============================================================
# Minimal ring annotations
# ============================================================
ax.text(
    0.50, 0.802,
    "fixed CORE17 anchors",
    ha="center",
    va="center",
    fontsize=13.5,
    fontweight="bold",
    color=TEAL
)

ax.text(
    0.50, 0.765,
    r"posterior $q_i(\theta)$",
    ha="center",
    va="center",
    fontsize=15,
    fontweight="bold",
    color=NAVY
)

# ============================================================
# Tiny central cell
# ============================================================
ax.add_patch(
    Circle(
        (CX, CY),
        R_CENTER,
        facecolor=PALE_BLUE,
        edgecolor=NAVY,
        linewidth=1.8,
        zorder=6
    )
)

ax.text(
    CX, CY + 0.020,
    r"cell $i$",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
    color=NAVY,
    zorder=8
)

ax.text(
    CX, CY - 0.022,
    "CORE17",
    ha="center",
    va="center",
    fontsize=12.5,
    color=TEXT,
    zorder=8
)

# ============================================================
# Inferred phase highlight
# ============================================================
x1, y1 = xy(phase_hat, R_POST_IN - 0.006)
x2, y2 = xy(phase_hat, R_POST_OUT + 0.012)

ax.plot(
    [x1, x2],
    [y1, y2],
    color=NAVY,
    linewidth=3.2,
    solid_capstyle="round",
    zorder=20
)

xp, yp = xy(
    phase_hat,
    (R_POST_IN + R_POST_OUT) / 2
)

ax.scatter(
    [xp], [yp],
    s=66,
    color=NAVY,
    edgecolor=WHITE,
    linewidth=0.8,
    zorder=21
)

# ============================================================
# Compact output below main circle
# No box, no overlap
# ============================================================
ax.text(
    0.50, 0.095,
    r"inferred phase   "
    rf"$\hat{{\theta}}_i={phase_hat:.1f}$ h"
    r"      "
    rf"$R_i={R_hat:.2f}$",
    ha="center",
    va="center",
    fontsize=14,
    color=TEXT
)

# color-emphasis line underneath
ax.plot(
    [0.355, 0.645],
    [0.073, 0.073],
    color=LIGHT_GREY,
    linewidth=1.0
)

ax.text(
    0.50, 0.045,
    "illustrative schematic",
    ha="center",
    va="center",
    fontsize=9.6,
    color=GREY,
    style="italic"
)

# ============================================================
# Save
# ============================================================
fig.savefig(
    PNG_OUT,
    dpi=400,
    bbox_inches="tight",
    pad_inches=0.05,
    facecolor=WHITE
)

fig.savefig(
    PDF_OUT,
    bbox_inches="tight",
    pad_inches=0.05,
    facecolor=WHITE
)

plt.close(fig)

print("=" * 72)
print("Figure 1D complete — clean publication layout")
print("=" * 72)
print(f"theta_hat = {phase_hat:.3f} h")
print(f"R_i       = {R_hat:.3f}")
print(f"PNG: {PNG_OUT}")
print(f"PDF: {PDF_OUT}")