#!/usr/bin/env python3
"""GSE166555 Figure 4A: compact reference-style two-column alluvial.

Only 5434 genes identified as phase-associated in at least one condition.
R=phase-associated by current analysis; AR=not detected as phase-associated.
Not evidence of objectively measured circadian rhythm or gene conversion.
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MP
from matplotlib.patches import PathPatch, Rectangle

SHARED = 2097  # Normal R -> Tumor R
NORMAL_ONLY = 1611  # Normal R -> Tumor AR
TUMOR_ONLY = 1726  # Normal AR -> Tumor R
N = SHARED + NORMAL_ONLY + TUMOR_ONLY

# Colors intentionally muted like the supplied K1/K2 reference.
AR_BAR = '#c9d6e9'
R_BAR = '#e9baa4'
SHARED_BAND = '#a9bed9'
GAIN_BAND = '#dce5f2'
LOSS_BAND = '#e8c9b8'
EDGE = '#565d6a'

def flow(ax, x0, x1, yl0, yh0, yl1, yh1, color, alpha, z):
    """Closed smooth band with exactly preserved thickness at both ends."""
    d = x1 - x0
    b = d * .44
    vertices = [(x0, yl0), (x0+b, yl0), (x1-b, yl1), (x1, yl1),
                (x1, yh1), (x1-b, yh1), (x0+b, yh0), (x0, yh0), (x0, yl0)]
    codes = [MP.MOVETO, MP.CURVE4, MP.CURVE4, MP.CURVE4,
             MP.LINETO, MP.CURVE4, MP.CURVE4, MP.CURVE4, MP.CLOSEPOLY]
    ax.add_patch(PathPatch(MP(vertices, codes), fc=color, ec='none', alpha=alpha, zorder=z))


def draw(output, with_text, dpi=600):
    # Close layout / narrow flow gap; columns intentionally taller and clearly segmented.
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    fig.patch.set_facecolor('white')
    col_width = 0.115
    lx, rx = 0.27, 0.71
    l_inner, r_inner = lx + col_width, rx
    left_split = SHARED + NORMAL_ONLY
    right_split = SHARED + TUMOR_ONLY

    # The width of every band is its gene count. Stable R->R is the base;
    # the two switching bands are very pale so their crossing is not dominant.
    flow(ax, l_inner, r_inner, 0, SHARED, 0, SHARED,
         SHARED_BAND, .55, 1)
    flow(ax, l_inner, r_inner, SHARED, left_split, right_split, N,
         LOSS_BAND, .47, 2)
    flow(ax, l_inner, r_inner, left_split, N, SHARED, right_split,
         GAIN_BAND, .50, 3)

    for x, split in ((lx, left_split), (rx, right_split)):
        # Separate opaque R and AR parts, not a dark undifferentiated bar.
        ax.add_patch(Rectangle((x, 0), col_width, split,
                               fc=R_BAR, ec='none', zorder=4))
        ax.add_patch(Rectangle((x, split), col_width, N-split,
                               fc=AR_BAR, ec='none', zorder=4))
        ax.add_patch(Rectangle((x, 0), col_width, N,
                               fc='none', ec=EDGE, lw=1.25, zorder=6))
        ax.plot([x, x+col_width], [split, split], color='white', lw=2.0, zorder=7)
        if with_text:
            ax.text(x+col_width/2, split/2, 'R', ha='center', va='center',
                    fontsize=10.5, weight='bold', color='#29313a', zorder=8)
            ax.text(x+col_width/2, (split+N)/2, 'AR', ha='center', va='center',
                    fontsize=10.5, weight='bold', color='#29313a', zorder=8)

    ax.set_xlim(0.16, 0.94)
    ax.set_ylim(0, N*1.025)
    ax.grid(False)
    ax.spines[['top', 'right']].set_visible(False)
    if with_text:
        ax.set_xticks([lx+col_width/2, rx+col_width/2], ['Normal', 'Tumor'])
        ax.set_yticks([0, 1000, 2000, 3000, 4000, 5000])
        ax.tick_params(axis='both', labelsize=9, width=.8, length=3)
        ax.set_ylabel('Gene count', fontsize=10)
        ax.set_title('Normal vs Tumor', fontsize=12, pad=8)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[['left', 'bottom']].set_visible(False)

    fig.subplots_adjust(left=.20 if with_text else .05,
                        right=.98 if with_text else .98,
                        top=.91 if with_text else .98,
                        bottom=.13 if with_text else .035)
    name = 'FIG4A_REFERENCE_WITH_LABELS' if with_text else 'FIG4A_REFERENCE_NO_LABELS'
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'{name}.{ext}', dpi=dpi, facecolor='white')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--outdir', default='16_FIG4_FIG5_APPLICATIONS/GSE166555/FIG4A_FINAL')
    ap.add_argument('--dpi', type=int, default=600)
    a=ap.parse_args()
    dest=Path(a.outdir)
    dest.mkdir(parents=True, exist_ok=True)
    draw(dest, True, a.dpi)
    draw(dest, False, a.dpi)
    assert SHARED + NORMAL_ONLY == 3708
    assert SHARED + TUMOR_ONLY == 3823
    print(f'DONE {dest.resolve()} | union={N}, shared={SHARED}, normal-only={NORMAL_ONLY}, tumor-only={TUMOR_ONLY}')

if __name__ == '__main__':
    main()
