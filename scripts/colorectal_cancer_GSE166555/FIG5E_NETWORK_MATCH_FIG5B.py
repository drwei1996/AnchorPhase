#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 5E: minimal target–molecule network, consistent with Figure 5B.
Uses OBSERVED ChEMBL-matched edges in the existing V7 export, not invented links.
No grey boxes, no legend, no grid. PNG/PDF/SVG and selected-edge audit CSV.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

TARGETS = ['ABCB1','CDA','MGLL','PLA2G2A','SLC9A3','MAP2K2']
PREFERRED = {
    'ABCB1': ['ENCEQUIDAR','BIRICODAR'],
    'CDA': ['CEDAZURIDINE'],
    'MGLL': ['ELCUBRAGISTAT'],
    'PLA2G2A': ['VARESPLADIB METHYL','VARESPLADIB'],
    'SLC9A3': ['SQUALAMINE LACTATE','TENAPANOR HYDROCHLORIDE'],
    'MAP2K2': ['BINIMETINIB','WX-554'],
}
CATEGORY = {
    'ABCB1':'Exposure / resistance', 'CDA':'Exposure / resistance',
    'MGLL':'Direct / investigational', 'PLA2G2A':'Direct / multi-PLA2',
    'SLC9A3':'Non-oncology physiological', 'MAP2K2':'Direct / oncology pathway',
}
# Match the V7 Figure 5B palette; keep category differences subtle but distinguishable.
COLORS = {
    'Exposure / resistance':'#447ca8',
    'Direct / investigational':'#bf704b',
    'Direct / multi-PLA2':'#bf704b',
    'Non-oncology physiological':'#9993b5',
    'Direct / oncology pathway':'#bf704b',
}

def build_edges(df, max_per_target=2, strict=True):
    required = {'gene','drug_name'}
    if not required.issubset(df.columns):
        raise ValueError(f'Expected columns {required}; found {list(df.columns)}')
    data = df.copy()
    data['gene'] = data['gene'].astype(str).str.upper().str.strip()
    data['drug_name'] = data['drug_name'].astype(str).str.upper().str.strip()
    data = data.loc[data.gene.isin(TARGETS) & ~data.drug_name.isin(['','NAN','NONE'])].drop_duplicates(['gene','drug_name'])
    edges=[]
    for gene in TARGETS:
        present = set(data.loc[data.gene.eq(gene),'drug_name'])
        preferred = [d for d in PREFERRED[gene] if d in present]
        if strict:
            selected=preferred[:max_per_target]
        else:
            selected=(preferred + sorted(present-set(preferred)))[:max_per_target]
        if not selected:
            print(f'WARNING: no observed preferred molecule for {gene}; no edge drawn')
        for drug in selected:
            edges.append({'gene':gene,'drug_name':drug,'mechanism_group':CATEGORY[gene],
                          'observed_in_input':True,'selection_rule':'preferred observed' if drug in PREFERRED[gene] else 'fallback observed'})
    return pd.DataFrame(edges, columns=['gene','drug_name','mechanism_group','observed_in_input','selection_rule'])

def bezier_link(ax, x0,y0,x1,y1,color):
    dx=x1-x0
    verts=[(x0,y0),(x0+0.32*dx,y0),(x1-0.32*dx,y1),(x1,y1)]
    path=MplPath(verts,[MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4])
    ax.add_patch(PathPatch(path,facecolor='none',edgecolor=color,lw=1.6,alpha=0.55,
                           capstyle='round',zorder=1))

def draw(edges,out,show_counts=False):
    if edges.empty:
        raise ValueError('No verified molecule links in input. Do not create an empty or fabricated network.')
    # Each molecule gets one distinct slot; shared drug molecules get multiple inbound edges.
    gene_order=[g for g in TARGETS if g in set(edges.gene)]
    drug_order=list(dict.fromkeys(edges.drug_name))
    n=max(len(drug_order),len(gene_order))
    height=max(5.0, 0.51*n+1.9)
    fig,ax=plt.subplots(figsize=(12.7,height),facecolor='white')
    fig.subplots_adjust(left=0.055,right=0.965,top=0.87,bottom=0.14)
    x0,x1=0.31,0.67
    yg={g:y for g,y in zip(gene_order,np.linspace(0.90,0.10,len(gene_order)))}
    yd={d:y for d,y in zip(drug_order,np.linspace(0.90,0.10,len(drug_order)))}
    for r in edges.itertuples(index=False):
        bezier_link(ax,x0,yg[r.gene],x1,yd[r.drug_name],COLORS[r.mechanism_group])
    for g in gene_order:
        c=COLORS[CATEGORY[g]]
        ax.scatter([x0],[yg[g]],s=155,color=c,edgecolor='white',linewidth=1.35,zorder=4)
        ax.text(x0-0.021,yg[g],g,ha='right',va='center',fontsize=12,fontweight='semibold',color='#252525')
        if show_counts:
            count=int(edges.gene.eq(g).sum())
            ax.text(x0-0.021,yg[g]-0.037,f'{count} shown',ha='right',va='top',fontsize=8,color='#777777')
    for d in drug_order:
        groups=edges.loc[edges.drug_name.eq(d),'mechanism_group']
        color=COLORS[groups.iloc[0]]
        ax.scatter([x1],[yd[d]],s=110,color=color,edgecolor='white',linewidth=1.2,zorder=4)
        ax.text(x1+0.018,yd[d],d,ha='left',va='center',fontsize=10.4,color='#252525')
    ax.text(x0-0.02,1.005,'TARGET',ha='right',va='bottom',fontsize=11,weight='bold')
    ax.text(x1+0.018,1.005,'ChEMBL MOLECULE',ha='left',va='bottom',fontsize=11,weight='bold')
    ax.set_xlim(0.01,1.015);ax.set_ylim(0,1.045)
    ax.axis('off')
    ax.set_title('Figure 5E | Target–molecule network',fontsize=15,pad=14,fontweight='medium')
    fig.text(.5,.035,'Only observed ChEMBL links; maximum two molecules per target. Not evidence of drug efficacy.',
             ha='center',fontsize=8.5,color='#6a6a6a')
    out.mkdir(parents=True,exist_ok=True)
    stem='FIG5E_TARGET_MOLECULE_NETWORK_CLEAN'
    for suffix in ('png','pdf','svg'):
        kwargs={'dpi':300} if suffix=='png' else {}
        fig.savefig(out/f'{stem}.{suffix}',bbox_inches='tight',facecolor='white',**kwargs)
    plt.close(fig)
    edges.to_csv(out/f'{stem}_DISPLAYED_EDGES.csv',index=False)
    print('SUCCESS:',out/stem,'[png/pdf/svg]')
    print('Verified edges:',len(edges),'Targets:',edges.gene.nunique(),'Molecules:',edges.drug_name.nunique())

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='16_FIG4_FIG5_APPLICATIONS/GSE166555')
    p.add_argument('--input',default=None,help='Override ChEMBL matched V7 table')
    p.add_argument('--output',default=None)
    p.add_argument('--max-per-target',type=int,default=2)
    p.add_argument('--fallback-observed',action='store_true',help='Use other observed molecules if preferred are missing')
    p.add_argument('--show-counts',action='store_true')
    a=p.parse_args();root=Path(a.root)
    inp=Path(a.input) if a.input else root/'FIG45_MECHANISM_PHASE_v2/06_FIG5_ALL_MECHANISM_QUALIFIED_MATCHES.csv.gz'
    out=Path(a.output) if a.output else root/'FIG5E_CLEAN_NETWORK'
    if not inp.exists():raise FileNotFoundError(f'Observed-molecule table not found: {inp}')
    df=pd.read_csv(inp,low_memory=False)
    ed=build_edges(df,a.max_per_target,strict=not a.fallback_observed)
    draw(ed,out,a.show_counts)

if __name__=='__main__': main()
