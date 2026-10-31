#!/usr/bin/env python3
"""Figure 4A: minimalist two-stratum alluvial, styled after user's reference image.
Counts from GSE166555 (union of genes associated with inferred phase):
  AR -> R: 1726; R -> AR: 1611; R -> R: 2097.
The AR/R labeling is operational significance status, not demonstrated absence/presence of biological rhythmicity.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle

FLOWS = {'AR_to_R': 1726, 'R_to_AR': 1611, 'R_to_R': 2097}
TOTAL = sum(FLOWS.values())

def ribbon(ax, x0, x1, source_lo, source_hi, target_lo, target_hi, color, alpha=.68):
    dx = x1-x0
    bend = .44 * dx
    verts = [(x0,source_lo), (x0+bend,source_lo), (x1-bend,target_lo), (x1,target_lo),
             (x1,target_hi), (x1-bend,target_hi), (x0+bend,source_hi), (x0,source_hi),
             (x0,source_lo)]
    codes = [MplPath.MOVETO] + [MplPath.CURVE4]*3 + [MplPath.LINETO] + [MplPath.CURVE4]*3 + [MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts,codes),facecolor=color,edgecolor='none',alpha=alpha,zorder=1))

def draw(outdir, show_text=True, dpi=450):
    fig,ax = plt.subplots(figsize=(4.1,5.0))
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    a,b=0.17,0.82
    barw=.115
    xl=a+barw/2; xr=b-barw/2
    rr=FLOWS['R_to_R']; ra=FLOWS['R_to_AR']; ar=FLOWS['AR_to_R']
    # Stack from bottom to top: R below AR (as in the supplied image).
    # At left, lower R contains R->R then R->AR; upper AR contains AR->R.
    # At right, lower R contains R->R then AR->R; upper AR contains R->AR.
    left_r=rr+ra
    right_r=rr+ar
    # Draw large stable flow first, then two pale crossing flows.
    ribbon(ax,xl,xr,0,rr,0,rr,'#c3d2e8',.74)            # R -> R
    ribbon(ax,xl,xr,rr,left_r,right_r,TOTAL,'#e7c7b5',.67) # R -> AR
    ribbon(ax,xl,xr,left_r,TOTAL,rr,right_r,'#dce4f1',.80) # AR -> R
    # Draw columns on top of ribbons: outlines and segment-specific subtle fills.
    for x,threshold in [(a,left_r),(b,right_r)]:
        ax.add_patch(Rectangle((x,0),barw,threshold,facecolor='#e6b99f',edgecolor='none',zorder=3,alpha=.85))
        ax.add_patch(Rectangle((x,threshold),barw,TOTAL-threshold,facecolor='#bbcae0',edgecolor='none',zorder=3,alpha=.95))
        ax.add_patch(Rectangle((x,0),barw,TOTAL,fill=False,edgecolor='#43484f',linewidth=1.05,zorder=5))
        ax.hlines(threshold,x,x+barw,color='white',linewidth=1.7,zorder=6)
    if show_text:
        ax.text(a+barw/2,left_r/2,'R',ha='center',va='center',fontsize=10,weight='bold',zorder=7)
        ax.text(a+barw/2,(left_r+TOTAL)/2,'AR',ha='center',va='center',fontsize=10,weight='bold',zorder=7)
        ax.text(b+barw/2,right_r/2,'R',ha='center',va='center',fontsize=10,weight='bold',zorder=7)
        ax.text(b+barw/2,(right_r+TOTAL)/2,'AR',ha='center',va='center',fontsize=10,weight='bold',zorder=7)
        ax.set_title('Normal vs Tumor',fontsize=11,weight='bold',pad=9)
        ax.set_ylabel('Gene counts',fontsize=9)
        ax.set_xticks([a+barw/2,b+barw/2],['Normal','Tumor'])
    else:
        ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(.02,1.08)
    ax.set_ylim(0,TOTAL*1.02)
    ax.grid(False)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#777'); ax.spines['left'].set_color('#777')
    ax.tick_params(axis='both',labelsize=8,length=3)
    fig.subplots_adjust(left=.18 if show_text else .07,right=.97,bottom=.11,top=.91 if show_text else .97)
    stem='FIG4A_REFERENCE_WITH_LABELS' if show_text else 'FIG4A_REFERENCE_NO_LABELS'
    for ext in ('png','pdf','svg'):
        fig.savefig(outdir/f'{stem}.{ext}',dpi=dpi,facecolor='white')
    plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--outdir',default='FIG4A_REFERENCE_STYLE_OUTPUT')
    p.add_argument('--dpi',type=int,default=450)
    args=p.parse_args(); o=Path(args.outdir);o.mkdir(parents=True,exist_ok=True)
    draw(o,True,args.dpi);draw(o,False,args.dpi)
    print(f'DONE: {o.resolve()} | union {TOTAL}; shared {FLOWS["R_to_R"]}; normal-only {FLOWS["R_to_AR"]}; tumor-only {FLOWS["AR_to_R"]}')
if __name__=='__main__':main()
