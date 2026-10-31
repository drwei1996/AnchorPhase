#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 4D: gene-resolved phase-association heatmaps for top 4 Tumor-only ORA pathways.

Uses ORIGINAL GSE166555 patient-harmonic fit (GSE166555_FIG4_FUNCTIONAL_PHASE_V2.py),
the original GSE166555 q-based gene calls and top 4 pathway ORA.
Rows are Tumor-only ORA overlap genes, NOT whole pathway. Left/right show exactly
same genes. Entries are fitted first-harmonic deviations of log1p(CP10K), pooled
per-gene scaled to preserve Normal/Tumor amplitude differences within each gene.
Selection is outcome dependent: strictly exploratory/descriptive, not independent validation.
"""
import os
for _k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k,'1')
import argparse, json, importlib.util, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BLUE='#2877ac'; ORANGE='#d8753a'
CMAP=LinearSegmentedColormap.from_list('blue_white_orange',[BLUE,'#ffffff',ORANGE],N=256)
SEED=20261023

def module(path):
    spec=importlib.util.spec_from_file_location('source_patient_harmonic',str(path))
    if spec is None or spec.loader is None: raise RuntimeError(f'Could not load: {path}')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def pretty(path):
    return re.sub(r'^(HALLMARK_|REACTOME_)','',path).replace('_',' ').title()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',default='.')
    ap.add_argument('--ora-dir',default=None)
    ap.add_argument('--out',default=None)
    ap.add_argument('--top',type=int,default=4)
    ap.add_argument('--bootstrap',type=int,default=0,help='Optional, not used for color matrix; kept for CLI compatibility')
    ap.add_argument('--min-cells',type=int,default=100)
    ap.add_argument('--min-detection',type=float,default=0.01,help='Min detectable fraction in combined eligible cells')
    ap.add_argument('--min-paired',type=int,default=5)
    ap.add_argument('--max-genes',type=int,default=60,help='Max rows per pathway. Pick from ORA overlapping genes in tumor q order, NOT group contrast')
    ap.add_argument('--min-overlap',type=int,default=5)
    ap.add_argument('--fdr',type=float,default=.05)
    args=ap.parse_args()
    base=Path(args.base).expanduser().resolve()
    root=base/'16_FIG4_FIG5_APPLICATIONS'/'GSE166555'
    ora_dir=Path(args.ora_dir).resolve() if args.ora_dir else root/'FIG4D_UNIQUE_RHYTHMIC_ORA_v2'
    ora_file=ora_dir/'Tumor_only_ALL_TESTED_pathways.csv'
    data=root/'anchorphase_project'/'data'
    rhythm=root/'fig4_DETERMINISTIC_COSINOR_v1'
    helper=next((p for p in [base/'GSE166555_FIG4_FUNCTIONAL_PHASE_V2.py',Path(__file__).with_name('GSE166555_FIG4_FUNCTIONAL_PHASE_V2.py')] if p.exists()),None)
    for fp in (ora_file,data/'count_matrix.h5ad',data/'PREDICTIONS_BLINDED.csv.gz',rhythm/'01_Normal_gene_rhythmicity.csv.gz',rhythm/'01_Tumor_gene_rhythmicity.csv.gz'):
        if not fp.exists():raise FileNotFoundError(str(fp))
    if helper is None:raise FileNotFoundError('Original fit module GSE166555_FIG4_FUNCTIONAL_PHASE_V2.py required beside script or katv2 root')
    h=module(helper)
    out=Path(args.out).resolve() if args.out else root/'FIG4D_TUMOR_ONLY_TOP4_GENE_HEATMAP'
    out.mkdir(parents=True,exist_ok=True)
    ora=pd.read_csv(ora_file)
    needed={'collection','pathway','q_combined','fold_enrichment','overlap_n','overlap_genes'}
    if miss:=needed-set(ora.columns):raise ValueError(f'ORA columns missing {miss}; columns={ora.columns.tolist()}')
    ora=ora[(pd.to_numeric(ora.q_combined,errors='coerce')<args.fdr)&(pd.to_numeric(ora.overlap_n,errors='coerce')>=args.min_overlap)&(pd.to_numeric(ora.fold_enrichment,errors='coerce')>1)]
    ora=ora.sort_values(['q_combined','fold_enrichment','overlap_n','collection','pathway'],ascending=[True,False,False,True,True]).head(args.top)
    if len(ora)!=args.top:raise RuntimeError(f'Only {len(ora)} valid Tumor-only enriched sets; expected {args.top}')
    ora.to_csv(out/'01_SELECTED_TUMOR_ONLY_ORA.csv',index=False)
    normal=pd.read_csv(rhythm/'01_Normal_gene_rhythmicity.csv.gz',usecols=['gene','eligible_nonCORE','rhythmic_candidate_nonCORE'])
    tumor=pd.read_csv(rhythm/'01_Tumor_gene_rhythmicity.csv.gz',usecols=['gene','eligible_nonCORE','rhythmic_candidate_nonCORE'])
    for x in (normal,tumor):
        x['gene']=x.gene.astype(str).str.upper().str.strip()
        for flag in ('eligible_nonCORE','rhythmic_candidate_nonCORE'):
            x[flag]=x[flag].astype(str).str.lower().isin(('true','1'))
    bg=set(normal.loc[normal.eligible_nonCORE,'gene'])&set(tumor.loc[tumor.eligible_nonCORE,'gene'])
    nset=set(normal.loc[normal.rhythmic_candidate_nonCORE,'gene'])&bg
    tset=set(tumor.loc[tumor.rhythmic_candidate_nonCORE,'gene'])&bg
    tumor_only=tset-nset
    print(f'Common tested universe={len(bg)}; Tumor-only genes={len(tumor_only)}',flush=True)
    import anndata as ad
    adata=ad.read_h5ad(data/'count_matrix.h5ad',backed='r')
    try:
        var=pd.Index(adata.var_names.astype(str).str.upper())
        if not var.is_unique:raise ValueError('Duplicate var names after upper-case conversion')
        prediction=pd.read_csv(data/'PREDICTIONS_BLINDED.csv.gz',usecols=['cell_id','phase_mean_h'])
        if prediction.cell_id.duplicated().any():raise ValueError('Non-unique prediction cell IDs')
        match=pd.Index(prediction.cell_id.astype(str)).get_indexer(adata.obs_names.astype(str))
        if (match<0).any():raise ValueError(f'Missing predictions: {(match<0).sum()}')
        phase=np.mod(prediction.phase_mean_h.to_numpy(dtype=float)[match],24.)
        if not np.isfinite(phase).all():raise ValueError('Nonfinite phase')
        obs=adata.obs
        for key in ('patient_id','condition','epithelial_subtype'):
            if key not in obs:raise KeyError(f'Missing adata.obs[{key}]')
        patient=obs.patient_id.astype(str).to_numpy()
        condition=obs.condition.astype(str).to_numpy()
        subtype=obs.epithelial_subtype.astype('string').fillna('Unknown').astype(str).to_numpy()
        counts=pd.crosstab(patient,condition).reindex(columns=['Normal','Tumor'],fill_value=0)
        eligible=counts.index[(counts.Normal>=args.min_cells)&(counts.Tumor>=args.min_cells)].tolist()
        if len(eligible)<args.min_paired:raise RuntimeError('Insufficient eligible matched patients')
        keep=np.isin(patient,eligible)&np.isin(condition,['Normal','Tumor'])
        idx=np.flatnonzero(keep)
        # Compute library size once, chunked to avoid backing the whole raw count matrix into RAM.
        lib=np.empty(len(adata),dtype=float)
        for start in range(0,len(adata),1000):
            end=min(len(adata),start+1000)
            lib[start:end]=np.asarray(adata.X[start:end,:].sum(axis=1)).ravel()
        if (lib[idx]<=0).any():raise ValueError('Zero library in eligible patients')
        theta=np.arange(0,24,.5)
        c=np.cos(2*np.pi*theta/24);s=np.sin(2*np.pi*theta/24)
        unique_genes=set()
        pathways=[]
        tumor_cols=tumor.set_index('gene')
        for r in ora.itertuples(index=False):
            genes=[g.strip().upper() for g in str(r.overlap_genes).split(';') if g.strip()]
            genes=sorted(set(genes)&tumor_only&set(var))
            # Prespecified cap uses tumor gene association q if source provides it, otherwise alphabetical.
            if len(genes)>args.max_genes:
                qcol=next((q for q in ['q','qvalue','q_value','q_bh','q_harmonic','padj','fdr','FDR'] if q in tumor_cols.columns),None)
                genes=sorted(genes,key=lambda g:(float(tumor_cols.loc[g,qcol]) if qcol and pd.notna(tumor_cols.loc[g,qcol]) else 1.0,g))[:args.max_genes]
            if not genes:raise RuntimeError(f'No tumor-only overlap genes for {r.pathway}')
            pathways.append((r,genes)); unique_genes.update(genes)
        gene_fits={}
        longfits=[]
        for g in sorted(unique_genes):
            col=var.get_loc(g)
            raw=adata.X[:,col]
            if sparse.issparse(raw):raw=np.asarray(raw.toarray()).ravel()
            else:raw=np.asarray(raw).ravel()
            if (raw[idx]>0).mean()<args.min_detection:
                print(f'LOW_DETECTION: {g}',flush=True)
                continue
            score=np.log1p(1e4*np.divide(raw,lib,out=np.zeros_like(raw,dtype=float),where=lib>0))
            fit={}
            for p in eligible:
                for cond in ('Normal','Tumor'):
                    cells=np.flatnonzero((patient==p)&(condition==cond))
                    item=h.patient_harmonic(score[cells],phase[cells],lib[cells],subtype[cells])
                    if item is not None:fit[(p,cond)]=item
            paired=[p for p in eligible if (p,'Normal') in fit and (p,'Tumor') in fit]
            if len(paired)<args.min_paired:
                print(f'SKIP insufficient paired fits: {g} n={len(paired)}',flush=True)
                continue
            gene_fits[g]={}
            for cond in ('Normal','Tumor'):
                betas=np.array([[fit[(p,cond)]['bcos'],fit[(p,cond)]['bsin']] for p in paired])
                b=betas.mean(axis=0)
                gene_fits[g][cond]=dict(bcos=float(b[0]),bsin=float(b[1]),amp=float(np.hypot(*b)),n_paired=len(paired),peak_h=float(np.mod(np.arctan2(b[1],b[0]),2*np.pi)*24/(2*np.pi)))
                longfits.append(dict(gene=g,condition=cond,**gene_fits[g][cond]))
            print(f'Fit {g}: paired={len(paired)}',flush=True)
        pd.DataFrame(longfits).to_csv(out/'02_GENE_HARMONIC_COEFFICIENTS.csv',index=False)
        if not gene_fits:raise RuntimeError('No genes had valid patient paired harmonic fits')
        panels=[]; audit=[]; matrix=[]
        for r,allgenes in pathways:
            valid=[g for g in allgenes if g in gene_fits]
            if not valid:raise RuntimeError(f'No valid fitted Tumor-only genes in {r.pathway}')
            # Order genes by tumor peak phase, with gene name secondary tie-break.
            valid=sorted(valid,key=lambda g:(gene_fits[g]['Tumor']['peak_h'],g))
            panels.append((r,valid))
            for g in allgenes:
                audit.append(dict(collection=r.collection,pathway=r.pathway,gene=g,in_heatmap=g in valid,tumor_only=True))
            for g in valid:
                waves={cond:(gene_fits[g][cond]['bcos']*c+gene_fits[g][cond]['bsin']*s) for cond in ('Normal','Tumor')}
                scale=max(float(np.max(np.abs(np.concatenate(list(waves.values()))))),1e-9)
                for cond in ('Normal','Tumor'):
                    for hh,rawval in zip(theta,waves[cond]):
                        matrix.append(dict(collection=r.collection,pathway=r.pathway,gene=g,condition=cond,phase_h=hh,centered_logCP10K=rawval,pooled_gene_maxabs=scale,display_scaled=rawval/scale))
        pd.DataFrame(audit).to_csv(out/'03_PATHWAY_TUMOR_ONLY_MEMBERSHIP.csv',index=False)
        cv=pd.DataFrame(matrix);cv.to_csv(out/'04_HEATMAP_VALUES_LONG.csv',index=False)
        # Panel height follows number of genes rather than stretching small pathways.
        heights=[max(1.35,.26*len(genes)+.7) for _,genes in panels]
        for show_text in (True,False):
            fig=plt.figure(figsize=(11 if show_text else 8, sum(heights)+1.2),facecolor='white')
            gs=fig.add_gridspec(len(panels),2,height_ratios=heights,hspace=.62,wspace=.035,left=.23 if show_text else .08,right=.94,bottom=.045,top=.94)
            for k,(r,genes) in enumerate(panels):
                for j,cond in enumerate(('Normal','Tumor')):
                    ax=fig.add_subplot(gs[k,j])
                    sub=cv[(cv.collection==r.collection)&(cv.pathway==r.pathway)&(cv.condition==cond)]
                    m=sub.pivot(index='gene',columns='phase_h',values='display_scaled').reindex(index=genes,columns=theta).to_numpy()
                    ax.imshow(m,aspect='auto',interpolation='nearest',origin='upper',extent=[0,24,len(genes)-.5,-.5],vmin=-1,vmax=1,cmap=CMAP)
                    # imshow extent inverted y manually to keep first gene at top
                    ax.set_ylim(len(genes)-.5,-.5)
                    ax.set_xlim(0,24)
                    ax.set_xticks([0,6,12,18,24]);ax.tick_params(axis='x',labelsize=8,length=2)
                    if show_text:
                        ax.set_title((pretty(str(r.pathway))+'  (n='+str(len(genes))+')\n' if j==0 else '')+cond,fontsize=10,color=BLUE if j==0 else ORANGE,pad=8)
                        if j==0:
                            ax.set_yticks(np.arange(len(genes)))
                            ax.set_yticklabels(genes,fontsize=7)
                        else:ax.set_yticks([])
                        if k==len(panels)-1:ax.set_xlabel('AnchorPhase inferred phase (h)',fontsize=9)
                    else:ax.set_xticklabels([]);ax.set_yticks([])
                    for spine in ax.spines.values():spine.set_visible(False)
            if show_text:
                fig.suptitle('Figure 4D | Tumor-only phase-associated genes within enriched pathways',fontsize=13,y=.985)
                fig.text(.51,.012,'Centered per-gene first-harmonic fits; gene-wise pooled Normal/Tumor scaling. Descriptive, not pathway activity.',ha='center',fontsize=8)
            stem='FIG4D_TUMOR_ONLY_GENE_HEATMAP_'+('WITH_TEXT' if show_text else 'NO_TEXT')
            for ext in ('.png','.pdf','.svg'):
                fig.savefig(out/(stem+ext),dpi=300,bbox_inches='tight',facecolor='white')
            plt.close(fig)
        (out/'RUN_MANIFEST.json').write_text(json.dumps(dict(source_ora=str(ora_file),source_rhythmic_calls=str(rhythm),expression=str(data/'count_matrix.h5ad'),phase=str(data/'PREDICTIONS_BLINDED.csv.gz'),source_original_fit=str(helper),selection='Tumor-only ORA top four by combined BH q, FE, overlap; overlap members only',gene_order='Tumor patient mean fitted peak phase, within each pathway',normalization='harmonic centered log1p(CP10K), each gene divided by its maximum absolute fitted value across both conditions; retains between-condition relative amplitude within gene only',not_independent_validation=True,colors={'negative':BLUE,'positive':ORANGE},n_eligible_patients=len(eligible),max_genes=args.max_genes),ensure_ascii=False,indent=2),encoding='utf-8')
        print('SUCCESS:',out,flush=True)
    finally:
        adata.file.close()

if __name__=='__main__':main()
