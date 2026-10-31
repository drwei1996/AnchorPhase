#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replot GSE166555 Figure 4D using TOP FOUR Tumor-only ORA pathways.
Same four stacked panels, Normal blue, Tumor orange, per-patient harmonic
regression, paired sign-flip p and patient-bootstrap 95% CI as previous Fig4D.
No alteration to Figure 4B/C or drug candidate selection.
Inputs: precomputed Tumor_only_ALL_TESTED_pathways.csv from unique-rhythm ORA,
2026.1 GMT, h5ad expression, BLINDED phase predictions.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(k,'1')
import argparse, sys, textwrap, json, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NORMAL='#2877ac'
TUMOR='#d8753a'
SEED=20261023

def load_helper(path, name):
    spec=importlib.util.spec_from_file_location(name,str(path))
    if spec is None or spec.loader is None: raise RuntimeError(f'Cannot load helper script: {path}')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def nice(s):
    s=str(s)
    for pre in ('HALLMARK_','REACTOME_'):
        if s.startswith(pre): s=s[len(pre):];break
    return s.replace('_',' ').title()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',default='.',help='katv2 project root')
    ap.add_argument('--ora-dir',default=None,help='directory containing Tumor_only_ALL_TESTED_pathways.csv')
    ap.add_argument('--out',default=None,help='output directory')
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--min-cells',type=int,default=100)
    ap.add_argument('--fdr',type=float,default=.05)
    ap.add_argument('--min-overlap',type=int,default=5)
    ap.add_argument('--top',type=int,default=4)
    ap.add_argument('--allow-nonsignificant',action='store_true',help='Only for diagnostic use; default requires FDR significant ORA')
    args=ap.parse_args()
    base=Path(args.base).expanduser().resolve()
    root=base/'16_FIG4_FIG5_APPLICATIONS'/'GSE166555'
    ora_dir=Path(args.ora_dir).resolve() if args.ora_dir else root/'FIG4D_UNIQUE_RHYTHMIC_ORA_v2'
    ora_file=ora_dir/'Tumor_only_ALL_TESTED_pathways.csv'
    if not ora_file.exists():
        raise FileNotFoundError(f'Missing {ora_file}\nRun FIG4D_GSE166555_UNIQUE_RHYTHMIC_ORA.py first.')
    path_ora=pd.read_csv(ora_file)
    needed={'collection','pathway','q_combined','fold_enrichment','overlap_n'}
    if missing:=needed-set(path_ora.columns): raise ValueError(f'ORA missing columns {missing}')
    x=path_ora[(pd.to_numeric(path_ora.overlap_n,errors='coerce')>=args.min_overlap)&
                (pd.to_numeric(path_ora.fold_enrichment,errors='coerce')>1)].copy()
    if not args.allow_nonsignificant:x=x[x.q_combined<args.fdr]
    # Prespecified top four: combined BH q, then fold enrichment, then overlap.
    x=x.sort_values(['q_combined','fold_enrichment','overlap_n','collection','pathway'],
                    ascending=[True,False,False,True,True]).drop_duplicates(['collection','pathway']).head(args.top)
    if len(x)!=args.top:raise RuntimeError(f'Only {len(x)} qualifying pathways; requested {args.top}. Check ORA. No cherry picking.')
    out=Path(args.out).resolve() if args.out else root/'FIG4D_TUMOR_ONLY_TOP4_PHASE_CURVES'
    out.mkdir(parents=True,exist_ok=True)
    x.to_csv(out/'01_PRESPECIFIED_TUMOR_ONLY_TOP4_ORA.csv',index=False)
    helper_candidates=[base/'GSE166555_FIG4_FUNCTIONAL_PHASE_V2.py',Path(__file__).with_name('GSE166555_FIG4_FUNCTIONAL_PHASE_V2.py')]
    helper_path=next((p for p in helper_candidates if p.is_file()),None)
    if helper_path is None:raise FileNotFoundError('Place GSE166555_FIG4_FUNCTIONAL_PHASE_V2.py next to this script or in katv2 root; it supplies the ORIGINAL patient-harmonic algorithm.')
    h=load_helper(helper_path,'original_fig4_functional_patient_v2')
    gmt_dir=base/'16_FIG4_FIG5_APPLICATIONS'/'data'
    gmts=h.gmt_read(gmt_dir)
    selections=[]
    for r in x.itertuples(index=False):
        key=(str(r.collection),str(r.pathway));gene_set=gmts.get(key)
        if gene_set is None:raise KeyError(f'Top-4 pathway not found in 2026.1 GMT: {key}')
        selections.append((key,gene_set,r))
    import anndata as ad
    path_data=root/'anchorphase_project'/'data'
    a=ad.read_h5ad(path_data/'count_matrix.h5ad')
    p=pd.read_csv(path_data/'PREDICTIONS_BLINDED.csv.gz')
    if 'cell_id' not in p or 'phase_mean_h' not in p: raise ValueError('Prediction file needs cell_id and phase_mean_h')
    if not p.cell_id.is_unique or not a.obs_names.is_unique:raise ValueError('Cell IDs not unique')
    m=pd.Index(p.cell_id.astype(str)).get_indexer(a.obs_names.astype(str))
    if (m<0).any(): raise ValueError(f'{(m<0).sum()} missing prediction cell IDs')
    phase=np.mod(p.iloc[m].phase_mean_h.to_numpy(float),24.)
    if not np.isfinite(phase).all():raise ValueError('Nonfinite inferred phase values')
    obs=a.obs.copy()
    for k in ('patient_id','condition','epithelial_subtype'):
        if k not in obs:raise ValueError('H5AD .obs missing '+k)
    patients=obs.patient_id.astype(str).to_numpy()
    conditions=obs.condition.astype(str).to_numpy()
    subtypes=obs.epithelial_subtype.astype('string').fillna('Unknown').astype(str).to_numpy()
    if set(conditions)!={'Normal','Tumor'}:raise ValueError(f'Unexpected conditions: {set(conditions)}')
    counts=pd.crosstab(patients,conditions).reindex(columns=['Normal','Tumor'],fill_value=0)
    eligible=counts[(counts.Normal>=args.min_cells)&(counts.Tumor>=args.min_cells)].index.tolist()
    if len(eligible)<5:raise RuntimeError(f'Only {len(eligible)} eligible paired patients')
    scores,lib,members=h.get_module_scores(a,selections)
    members.to_csv(out/'02_PATHWAY_MEMBERS.csv',index=False)
    if len(scores)!=args.top:raise RuntimeError('One or more selected gene sets have <10 expressed genes in H5AD; no silent substitution')
    theta=np.arange(0,24.0001,.25)
    ct=np.cos(2*np.pi*theta/24.);st=np.sin(2*np.pi*theta/24.)
    rng=np.random.default_rng(SEED)
    records=[];curves=[];stats=[]
    for (collection,pathway),score in scores:
        fitted={}
        for patient in eligible:
            for cond in ('Normal','Tumor'):
                ix=np.flatnonzero((patients==patient)&(conditions==cond))
                fit=h.patient_harmonic(score[ix],phase[ix],lib[ix],subtypes[ix])
                if fit is not None:fitted[(patient,cond)]=fit
                records.append(dict(collection=collection,pathway=pathway,patient=patient,condition=cond,fit_ok=(fit is not None),n_cells=len(ix),**(fit or {})))
        paired=[patient for patient in eligible if all((patient,c) in fitted for c in ('Normal','Tumor'))]
        if len(paired)<5:raise RuntimeError(f'{pathway}: only {len(paired)} fitted paired patients, cannot plot valid comparison')
        b={cond:np.asarray([[fitted[(patient,cond)]['bcos'],fitted[(patient,cond)]['bsin']] for patient in paired]) for cond in ('Normal','Tumor')}
        delta=b['Tumor']-b['Normal']
        pvalue=h.paired_signflip(delta)
        stats.append(dict(collection=collection,pathway=pathway,n_paired_fitted=len(paired),paired_p=pvalue,paired_delta_norm=float(np.linalg.norm(delta.mean(axis=0)))))
        for cond in ('Normal','Tumor'):
            y=b[cond][:,0,None]*ct+b[cond][:,1,None]*st
            yboot=np.mean(y[rng.integers(0,len(paired),(args.bootstrap,len(paired)))],axis=1)
            low,high=np.percentile(yboot,[2.5,97.5],axis=0)
            for hour,mean,lo,hi in zip(theta,y.mean(axis=0),low,high):
                curves.append(dict(collection=collection,pathway=pathway,condition=cond,phase_h=hour,mean_centered_module_logCP10K=mean,bootstrap_lower=lo,bootstrap_upper=hi,n_paired=len(paired)))
        print(f'{collection} | {pathway}: {len(paired)} pairs, paired p={pvalue:.5g}',flush=True)
    pd.DataFrame(records).to_csv(out/'03_PATIENT_FITS.csv',index=False)
    pd.DataFrame(stats).to_csv(out/'04_PAIRED_TESTS.csv',index=False)
    cv=pd.DataFrame(curves);cv.to_csv(out/'05_PATIENT_EQUAL_CURVES.csv',index=False)
    for with_text in (True,False):
        fig,axes=plt.subplots(args.top,1,figsize=(8,2.3*args.top),sharex=True,squeeze=False)
        for i,((coll,path),_,_) in enumerate(selections):
            ax=axes[i,0]
            sub=cv[(cv.collection==coll)&(cv.pathway==path)]
            for condition,color in (('Normal',NORMAL),('Tumor',TUMOR)):
                d=sub[sub.condition==condition].sort_values('phase_h')
                hours=d.phase_h.to_numpy(float)
                ax.plot(hours,d.mean_centered_module_logCP10K.to_numpy(float),color=color,lw=2,label=condition)
                ax.fill_between(hours,d.bootstrap_lower.to_numpy(float),d.bootstrap_upper.to_numpy(float),color=color,alpha=.16)
            ax.axhline(0,color='grey',lw=.55);ax.set_xlim(0,24);ax.set_xticks([0,6,12,18,24]);ax.grid(False)
            if with_text:
                pv=next(z['paired_p'] for z in stats if z['collection']==coll and z['pathway']==path)
                title=f'{nice(path)}   |   paired p = {pv:.3g} (exploratory)'
                ax.set_title('\n'.join(textwrap.wrap(title,width=95,break_long_words=False)),fontsize=10)
                ax.set_ylabel('Centered\nmodule score',fontsize=9)
            else:
                ax.set_title('');ax.set_ylabel('');ax.set_yticklabels([])
            if i==0 and with_text:ax.legend(frameon=False,ncol=2,loc='upper right',fontsize=9)
            for side in ('top','right'):ax.spines[side].set_visible(False)
        axes[-1,0].set_xlabel('AnchorPhase inferred phase (h)' if with_text else '')
        if with_text:fig.suptitle('Figure 4D | Inferred-phase pathway associations (95% patient bootstrap CI)',fontsize=12)
        fig.tight_layout()
        name='FIG4D_TUMOR_ONLY_TOP4_'+('WITH_TEXT' if with_text else 'NO_TEXT')
        for ext in ('png','pdf','svg'):
            fig.savefig(out/(name+'.'+ext),dpi=300,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    (out/'RUN_MANIFEST.json').write_text(json.dumps(dict(selection='Tumor-only ORA, combined BH q ascending, fold enrichment descending, overlap descending; top 4, no p-value based reselection',selection_file=str(ora_file),gmt_dir=str(gmt_dir),expression_file=str(path_data/'count_matrix.h5ad'),phase_file=str(path_data/'PREDICTIONS_BLINDED.csv.gz'),patient_min_cells=args.min_cells,bootstrap=args.bootstrap,bootstrap_seed=SEED,selected=[dict(collection=k[0],pathway=k[1]) for k,gs,r in selections],interpretation='Descriptive association along inferred phase, not measured 24-hour rhythm; paired p exploratory, pathway selection uses same data'),indent=2,ensure_ascii=False),encoding='utf-8')
    print('SUCCESS:',out,flush=True)

if __name__=='__main__':main()
