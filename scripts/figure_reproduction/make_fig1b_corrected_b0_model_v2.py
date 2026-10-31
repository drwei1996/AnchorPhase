"""Fig 1b — Corrected anchored B0 model (primary model; core panel).

NB likelihood with log-library-size offset
    log mu_ig(theta_i) = o_i + beta_g + a_g cos(theta_i - varphi_g),
exact-grid posterior over CORE17, anchored acrophase priors.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from fig_style_v2 import (apply_style, rbox, panel_title, save_panel, MID,
                       NAVY, BLUE, LBLUE, MBBLUE, GRAY, DK)

apply_style()
fig = plt.figure(figsize=(MID, 4.65))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 108)
ax.axis("off")
panel_title(ax, "Corrected anchored B0 model", color=NAVY)

rbox(ax, 1, 2.5, 98, 97.5, fc=LBLUE, ec=NAVY, lw=1.4, r=2.0, alpha=0.95)

# ---- likelihood equations ----
rbox(ax, 6, 66, 88, 22, fc="white", ec=BLUE, lw=0.9)
ax.text(50, 81.5,
        r"$x_{ig} \;\sim\; \mathrm{NB}\left(\mu_{ig}(\theta_i),\; \kappa_g\right)$",
        fontsize=9.0, ha="center", color=DK)
ax.text(50, 71.5,
        r"$\log \mu_{ig}(\theta_i) \;=\; o_i \;+\; \beta_g \;+\; a_g \cos(\theta_i - \varphi_g)$",
        fontsize=9.0, ha="center", color=DK)

# ---- parameter legend (2 columns x 3 rows) ----
rows_y = [61.0, 55.4, 49.8]
items = [
    (rows_y[0], 6.0, r"$o_i$", "log-library-size offset",
     52.0, r"$\varphi_g$", "anchored gene acrophase (fixed prior)"),
    (rows_y[1], 6.0, r"$\beta_g$", "baseline expression",
     52.0, r"$\theta_i$", "latent circadian phase of cell $i$"),
    (rows_y[2], 6.0, r"$a_g$", "rhythmic amplitude",
     52.0, r"$\kappa_g$", "NB dispersion"),
]
for y, x1, s1, t1, x2, s2, t2 in items:
    ax.text(x1, y, s1, fontsize=6.4, color=GRAY)
    ax.text(x1 + 7.0, y, t1, fontsize=6.2, color=GRAY)
    ax.text(x2, y, s2, fontsize=6.4, color=GRAY)
    ax.text(x2 + 7.0, y, t2, fontsize=6.2, color=GRAY)

# ---- cosine wave with anchored acrophase marker ----
wx, wy, ww, wh = 62, 25.5, 32, 16
tt = np.linspace(0, 2 * np.pi, 240)
ax.add_patch(mpatches.Rectangle((wx, wy), ww, wh, fc="white",
                                ec="#C9CFD6", lw=0.6, zorder=2))
ax.plot(wx + tt / (2 * np.pi) * ww,
        wy + wh / 2 + np.cos(tt - 0.9) * wh * 0.32,
        color=BLUE, lw=1.2, zorder=3)
ph_x = wx + (0.9 / (2 * np.pi)) * ww
peak_y = wy + wh / 2 + wh * 0.32
# Stop the dashed guide just below the acrophase marker.  In the previous
# version the guide continued through the marker and into the phi_g label,
# producing a visual collision at the curve/guide intersection.
ax.plot([ph_x, ph_x], [wy + 1.0, peak_y - 0.45], color=NAVY, lw=0.9,
        ls=(0, (3, 2)), zorder=4, solid_capstyle="butt", clip_on=True)
ax.plot(ph_x, peak_y, marker="o", ms=3.0, color=NAVY, zorder=5,
        clip_on=True)
# Place phi_g above-right of the marker with a tiny white knockout so the
# symbol can never sit on the curve or guide line.
ax.text(ph_x + 1.35, peak_y + 0.60, r"$\varphi_g$", fontsize=6.4,
        color=NAVY, ha="left", va="bottom", zorder=6,
        bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.96))
ax.text(wx + ww / 2, wy + wh + 1.45,
        "acrophase priors anchor absolute orientation",
        fontsize=5.7, color=NAVY, ha="center", va="bottom")

# ---- posterior formula ----
rbox(ax, 5, 24.5, 53, 22, fc=MBBLUE, ec=NAVY, lw=1.0)
ax.text(31.5, 41.8, "Posterior inference (exact phase grid)",
        fontsize=7.2, fontweight="bold", color=NAVY, ha="center")
ax.text(31.5, 32.3,
        r"$q_i(\theta) \;\propto\; p(\theta) "
        r"\prod_{g \,\in\, \mathrm{CORE17}} \mathrm{NB}(x_{ig} \mid \mu_{ig}(\theta), \kappa_g)$",
        fontsize=7.4, ha="center", color=DK)

# ---- key properties box ----
rbox(ax, 5, 3.5, 53, 18.5, fc="white", ec=BLUE, lw=1.0)
ax.text(8, 19.6, "Key properties (corrected B0)", fontsize=7.2,
        fontweight="bold", color=BLUE, va="top")
bullets = ["negative binomial count model",
           "whole-transcriptome offset correction",
           "biologically anchored absolute phase inference",
           "cell-level posterior distribution"]
for k, b in enumerate(bullets):
    ax.text(8.5, 15.6 - k * 3.2, "•", fontsize=6, color=BLUE, va="center")
    ax.text(12.0, 15.6 - k * 3.2, b, fontsize=6.2, color=DK, va="center")

# ---- italic note ----
ax.text(78, 14.0,
        "Biologically anchored\nabsolute phase inference\nfrom core clock genes",
        fontsize=6.4, style="italic", color=NAVY, ha="center", va="center")

save_panel(fig, "fig1b_corrected_b0_model_v2")
