"""Fig 1c — Posterior outputs and uncertainty (KATv2 panel).

Circular posteriors (sharp / intermediate / broad), posterior circular
mean phase theta_B0, concentration R_B0 and entropy H_B0; final default
deployed model = anchored B0.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from fig_style_v2 import (apply_style, rbox, panel_title, save_panel, SINGLE,
                       GREEN, NAVY, GRAY, DK)

apply_style()
fig = plt.figure(figsize=(SINGLE, 3.95))
ax = fig.add_axes([0.005, 0, 0.99, 1]); ax.set_xlim(0, 100)
ax.set_ylim(0, 108); ax.axis("off")
panel_title(ax, "Posterior outputs and uncertainty")

pA = ax.get_position()


def icon_axes(cx_data, cy_data, d_in=0.86):
    fx = pA.x0 + cx_data / 100 * pA.width
    fy = pA.y0 + cy_data / 108 * pA.height
    w, h = d_in / fig.get_size_inches()[0], d_in / fig.get_size_inches()[1]
    a = fig.add_axes([fx - w / 2, fy - h / 2, w, h], zorder=5)
    a.set_xlim(-1.25, 1.25); a.set_ylim(-1.25, 1.25)
    a.set_aspect("equal"); a.axis("off")
    return a


def draw_posterior(a, mean, kappa):
    rim = Circle((0, 0), 1.0, fc="white", ec="#B9C4CE", lw=1.0,
                 zorder=2)
    a.add_patch(rim)
    th = np.linspace(0, 2 * np.pi, 300)
    dens = np.exp(kappa * (np.cos(th - mean) - 1))
    dens = dens / dens.max()
    for t_, d_ in zip(th, dens):
        if d_ > 0.06:
            wedge = mpatches.Wedge(
                (0, 0), 0.985, np.degrees(t_) - 1.4, np.degrees(t_) + 1.4,
                fc=GREEN, ec="none", alpha=0.15 + 0.8 * d_, zorder=3,
                clip_on=True)
            wedge.set_clip_path(rim)
            a.add_patch(wedge)
    # Keep the mean indicator fully inside the circumference.  Ending a thick
    # line exactly at r=1 with a round cap created the small protruding dot seen
    # outside each circle in the previous render.
    mean_line, = a.plot([np.cos(mean) * 0.82, np.cos(mean) * 0.945],
                        [np.sin(mean) * 0.82, np.sin(mean) * 0.945],
                        color=GREEN, lw=1.55, zorder=5,
                        solid_capstyle="butt", clip_on=True)
    mean_line.set_clip_path(rim)


icons = [(r"$\hat{\theta}^{B0}_i$ = 14.5 h" + "\n" +
          r"$R^{B0}_i$ = 0.86,  $H^{B0}_i$ low", 8.0, 0.9),
         (r"$\hat{\theta}^{B0}_i$ = 21.0 h" + "\n" +
          r"$R^{B0}_i$ = 0.54,  $H^{B0}_i$ mid", 2.5, 2.6),
         (r"$\hat{\theta}^{B0}_i$ = 5.5 h" + "\n" +
          r"$R^{B0}_i$ = 0.18,  $H^{B0}_i$ high", 1.0, -0.6)]
for cx, (lab, kap, mu) in zip([18, 50, 82], icons):
    draw_posterior(icon_axes(cx, 78), mu, kap)
    ax.text(cx, 62.5, lab, fontsize=6.0, ha="center", va="top",
            color=DK)

ax.text(50, 49.0,
        r"$\hat{\theta}^{B0}_i$: posterior circular mean phase",
        fontsize=6.6, ha="center", color=GRAY, style="italic")

ax.text(50, 43.2, r"$R_i \;=\; \left| \mathbb{E}_{q_i}\!\left[ e^{i\theta} \right] \right|$",
        fontsize=8.2, ha="center", color=DK)
ax.text(50, 35.6, r"$H_i \;=\; -\int q_i(\theta)\, \log q_i(\theta)\, d\theta$",
        fontsize=8.2, ha="center", color=DK)

ax.text(50, 27.8,
        r"outputs:  $\hat{\theta}^{B0}_i$,  $R^{B0}_i$ (confidence),  $H^{B0}_i$ (entropy)",
        fontsize=6.8, ha="center", color=GREEN)

rbox(ax, 6, 4, 88, 17, fc=NAVY, ec=NAVY, lw=0)
ax.text(50, 15.8, "Final default deployed model", fontsize=7.4,
        fontweight="bold", color="white", ha="center", va="center")
ax.text(50, 9.2, "anchored B0", fontsize=9.4, fontweight="bold",
        color="white", ha="center", va="center")

save_panel(fig, "fig1c_posterior_outputs_uncertainty_v2")
