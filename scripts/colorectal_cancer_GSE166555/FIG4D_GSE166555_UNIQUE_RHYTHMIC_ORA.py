#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GSE166555 Figure 4D: Normal-only vs Tumor-only phase-associated genes ORA.
Uses frozen original V1 non-CORE17 calls and MSigDB v2026.1 GMT.
Descriptive conditional phase association, NOT independently verified 24-hour rhythmicity.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import textwrap

FOCUS_CORE={'ARNTL','BHLHE41','CLOCK','CRY1','CRY2','DBP','FBXO21','HLF','NFIL3','NPAS2','NR1D1','NR1D2','PER1','PER2','PER3','RORC','TEF'}

def bh(p):
    p=np.asarray(p,dtype=float)
    if not len(p):return p
    order=np.argsort(p,kind='stable')
    sorted_p=p[order]
    q=np.minimum.accumulate((sorted_p*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    out=np.empty(len(p));out[order]=np.clip(q,0,1)
    return out

def load_calls(file):
    df=pd.read_csv(file,low_memory=False)
    required={'gene','eligible_nonCORE','rhythmic_candidate_nonCORE'}
    if not required.issubset(df.columns):
        raise ValueError(f'{file} must contain {required}; columns={df.columns.tolist()}')
    df['gene']=df.gene.astype(str).str.strip().str.upper()
    if df.gene.duplicated().any():raise ValueError(f'Duplicate genes in {file}')
    def strict_bool(s):
        x=s.astype(str).str.strip().str.lower()
        invalid=~x.isin(['true','false','1','0'])
        if invalid.any():raise ValueError(f'Unrecognized flags in {file}: {s[invalid].head().tolist()}')
        return x.isin(['true','1'])
    eligible=strict_bool(df.eligible_nonCORE)
    hits=strict_bool(df.rhythmic_candidate_nonCORE)
    if (hits & ~eligible).any():raise ValueError('Rhythmic genes outside eligible genes')
    U=set(df.loc[eligible,'gene'])-FOCUS_CORE
    R=set(df.loc[hits,'gene']) & U
    return U,R

def read_gmt(files,universe,min_size,max_size):
    sets={}
    for fp in files:
        if not fp.is_file():raise FileNotFoundError(f'GMT missing: {fp}')
        collection='Hallmark' if fp.name.startswith('h.all.') else 'Reactome'
        with fp.open(encoding='utf-8') as f:
            for line in f:
                tok=line.rstrip('\n').split('\t')
                if len(tok)<3:continue
                name=tok[0]
                gs={v.upper() for v in tok[2:] if v} & universe
                if min_size<=len(gs)<=max_size:
                    key=(collection,name)
                    if key in sets:raise ValueError(f'Duplicate GMT pathway {key}')
                    sets[key]=gs
    if not sets:raise RuntimeError('No valid pathways after overlap with gene universe')
    return sets

def ora(query,U,sets):
    N=len(U); n=len(query)
    if n==0:raise ValueError('Empty query, cannot do ORA')
    rows=[]
    for (db,path),gs in sets.items():
        K=len(gs); hits=gs & query; k=len(hits)
        p=float(hypergeom.sf(k-1,N,K,n)) if k else 1.
        fe=(k/n)/(K/N) if K else float('nan')
        rows.append({'collection':db,'pathway':path,'overlap_n':k,
                     'pathway_in_universe_n':K,'query_n':n,'universe_n':N,
                     'fold_enrichment':fe,'p':p,'overlap_genes':';'.join(sorted(hits))})
    d=pd.DataFrame(rows); d['q_collection']=np.nan
    # Preserve previously established collection-wise FDR family; test all eligible sets, including k=0.
    for collection,ind in d.groupby('collection').groups.items():
        d.loc[ind,'q_collection']=bh(d.loc[ind,'p'].values)
    d['q_combined']=bh(d.p.values)
    return d.sort_values(['q_combined','p','overlap_n'],ascending=[True,True,False]).reset_index(drop=True)

def nice(s):
    for prefix in ('REACTOME_','HALLMARK_'):
        if s.startswith(prefix):s=s[len(prefix):];break
    return s.replace('_',' ').title().replace('Eif2Ak4','EIF2AK4').replace('Gcn2','GCN2').replace('Mrna','mRNA').replace('Rrna','rRNA').replace('Nfkb','NF-κB')

def choose(d,qcut,top,min_overlap,metric):
    candidates=d[(d.q_combined<qcut)&(d.overlap_n>=min_overlap)&(d.fold_enrichment>1)].copy()
    if metric=='enrichment':candidates=candidates.sort_values(['q_combined','fold_enrichment'],ascending=[True,False])
    else:candidates=candidates.sort_values(['q_combined','p'],ascending=[True,True])
    return candidates.head(top)

def draw(left,right,out,top,annotate=True):
    fig,axs=plt.subplots(1,2,figsize=(15.5,8),gridspec_kw={'wspace':.92})
    colors=['#C77950','#327F88']
    for ax,dat,title,color in zip(axs,[left,right],['Normal-only phase-associated','Tumor-only phase-associated'],colors):
        ax.set_title(title,fontsize=14,pad=12)
        if dat.empty:
            ax.text(.5,.5,'No pathways pass prespecified FDR and overlap filters',ha='center',va='center',transform=ax.transAxes,wrap=True,fontsize=11)
            ax.set_yticks([]);ax.set_xlim(0,1)
        else:
            data=dat.iloc[::-1].copy(); y=np.arange(len(data))
            marker_size=np.clip(data.overlap_n.to_numpy()*9+45,65,340)
            ax.scatter(data.fold_enrichment,y,s=marker_size,c=color,edgecolor='white',linewidth=1.1,zorder=3)
            ax.set_yticks(y)
            if annotate:
                labels=[textwrap.fill(nice(z),width=38,break_long_words=False) for z in data.pathway]
                ax.set_yticklabels(labels,fontsize=9)
            else:ax.set_yticklabels(['']*len(data))
            ax.set_xlim(0,max(1.05,data.fold_enrichment.max()*1.18))
            ax.set_ylim(-.6,len(data)-.4)
            if annotate:
                for x,yy,nn in zip(data.fold_enrichment,y,data.overlap_n):
                    ax.annotate(f'{int(nn)}',xy=(x,yy),xytext=(6,0),textcoords='offset points',fontsize=8,va='center')
        ax.grid(False)
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
        ax.set_xlabel('ORA fold enrichment',fontsize=11)
    if annotate:
        fig.suptitle('Figure 4D | Enrichment of condition-specific phase-associated genes',fontsize=17)
        fig.text(.5,.015,'Hypergeometric ORA; BH across all eligible Hallmark + Reactome pathways. Descriptive associations, not 24-hour verified rhythms.',ha='center',fontsize=9)
        fig.subplots_adjust(left=.28,right=.98,bottom=.11,top=.89,wspace=.95)
    else:
        fig.subplots_adjust(left=.07,right=.98,bottom=.12,top=.91,wspace=.23)
        for ax in axs:ax.set_title('');ax.set_xlabel('');ax.set_xticklabels([])
    for ext in ['png','pdf']:
        fig.savefig(out.with_suffix('.'+ext),dpi=320,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',default='.',help='katv2 project root')
    ap.add_argument('--normal',default=None)
    ap.add_argument('--tumor',default=None)
    ap.add_argument('--out',default=None)
    ap.add_argument('--min-set',type=int,default=10)
    ap.add_argument('--max-set',type=int,default=500)
    ap.add_argument('--min-overlap',type=int,default=5)
    ap.add_argument('--top',type=int,default=12)
    ap.add_argument('--fdr',type=float,default=.05)
    ap.add_argument('--sort',choices=['q','enrichment'],default='q')
    args=ap.parse_args()
    base=Path(args.base).expanduser().resolve()
    study=base/'16_FIG4_FIG5_APPLICATIONS'
    original=study/'GSE166555/fig4_DETERMINISTIC_COSINOR_v1'
    nf=Path(args.normal) if args.normal else original/'01_Normal_gene_rhythmicity.csv.gz'
    tf=Path(args.tumor) if args.tumor else original/'01_Tumor_gene_rhythmicity.csv.gz'
    gmt=[study/'data/h.all.v2026.1.Hs.symbols.gmt',study/'data/c2.cp.reactome.v2026.1.Hs.symbols.gmt']
    out=Path(args.out) if args.out else study/'GSE166555/FIG4D_UNIQUE_RHYTHMIC_ORA_v2'
    out.mkdir(parents=True,exist_ok=True)
    Un,Rn=load_calls(nf);Ut,Rt=load_calls(tf)
    universe=Un&Ut
    if len(universe)<1000:raise RuntimeError(f'Common tested gene universe unexpectedly small ({len(universe)}). Check input.')
    Rn=Rn&universe;Rt=Rt&universe
    nor=Rn-Rt;tum=Rt-Rn;both=Rn&Rt
    if len(nor)==0 or len(tum)==0:raise RuntimeError('One condition-specific set is empty')
    sets=read_gmt(gmt,universe,args.min_set,args.max_set)
    dn=ora(nor,universe,sets);dt=ora(tum,universe,sets)
    for prefix,data in [('Normal_only',dn),('Tumor_only',dt)]:
        data.to_csv(out/f'{prefix}_ALL_TESTED_pathways.csv',index=False)
        select=choose(data,args.fdr,args.top,args.min_overlap,args.sort)
        select.to_csv(out/f'{prefix}_PLOTTED_pathways.csv',index=False)
    for name,genes in [('Normal_only',nor),('Tumor_only',tum),('Shared',both),('Common_tested_background',universe)]:
        (out/f'{name}_genes.txt').write_text('\n'.join(sorted(genes))+'\n',encoding='utf-8')
    left=choose(dn,args.fdr,args.top,args.min_overlap,args.sort)
    right=choose(dt,args.fdr,args.top,args.min_overlap,args.sort)
    draw(left,right,out/'FIG4D_UNIQUE_RHYTHMIC_ORA_WITH_LABELS',args.top,True)
    draw(left,right,out/'FIG4D_UNIQUE_RHYTHMIC_ORA_NO_LABELS',args.top,False)
    audit={'normal_rhythmic_common_background':len(Rn),'tumor_rhythmic_common_background':len(Rt),'shared':len(both),'normal_only':len(nor),'tumor_only':len(tum),'common_tested_noncore_background':len(universe),'pathways_tested':len(sets),'significant_normal_only':int((dn.q_combined<args.fdr).sum()),'significant_tumor_only':int((dt.q_combined<args.fdr).sum()),'normal_plotted':len(left),'tumor_plotted':len(right),'q_definition':'BH across BOTH GMT collections and ALL gene sets passing size filter, including zero overlaps','caveat':'Exploratory plug-in cosinor inferred-phase association; unique means category-specific q<0.05, not statistically proven group interaction; no measured 24-h time series','source_files':[str(nf),str(tf)]+[str(x) for x in gmt]}
    (out/'AUDIT.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(audit,indent=2,ensure_ascii=False))
    print('SUCCESS',out)

if __name__=='__main__':main()
