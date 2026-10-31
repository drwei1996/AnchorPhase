#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""All Hallmark + Reactome pathways: ORIGINAL Figure-4H method A at scale.

Same biological/statistical definitions as GSE166555_FIG4_FUNCTIONAL_PHASE_V2.py:
  module = mean_g log1p(10000 * UMI_g / library_size) among observed GMT members;
  patient/condition fit: module ~ intercept + standardized log1p(library) +
  epithelial_subtype dummies + cos(2*pi*phase/24) + sin(2*pi*phase/24);
  reject ill-conditioned partialled-out phase regressors (sigma_min < 0.025);
  difference = mean over eligible paired donors of (Tumor - Normal) [bcos, bsin];
  effect = Euclidean norm of that mean; exact sign-flip p when n<=15;
  separate BH-FDR for full Hallmark family and full Reactome family.

IMPORTANT: inferred phase is NOT observed time. This analysis is exploratory;
patient pairing preserves subject as the independent observational unit.
"""
import os
for _k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse, json, gzip, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy import linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_gmt(folder):
    sets = {}
    paths = sorted(list(folder.glob('*.gmt'))+list(folder.glob('*.gmt.gz')))
    for p in paths:
        n=p.name.lower()
        coll = 'Hallmark' if ('h.all.' in n or 'hallmark' in n) else ('Reactome' if 'reactome' in n else None)
        if not coll: continue
        opener=gzip.open if p.suffix == '.gz' else open
        with opener(p,'rt',encoding='utf8') as fh:
            for line in fh:
                a=line.rstrip('\n').split('\t')
                if len(a)>3:
                    key=(coll,a[0])
                    if key in sets and sets[key]!=set(a[2:]):
                        raise ValueError(f'Duplicate conflicting GMT pathway: {key}')
                    sets[key]=set(a[2:])
    if not sets: raise FileNotFoundError(f'No Hallmark or Reactome GMT found in {folder}')
    return sets, [str(x) for x in paths]


def bh(p):
    x=np.asarray(p,float);out=np.full(x.shape,np.nan)
    ii=np.where(np.isfinite(x))[0]
    if len(ii):
        order=ii[np.argsort(x[ii])]
        q=np.minimum.accumulate((x[order]*len(order)/np.arange(1,len(order)+1))[::-1])[::-1]
        out[order]=np.minimum(1,q)
    return out


def fit_design(phase,lib,subtype, threshold):
    phase=np.mod(np.asarray(phase,float),24)
    c=np.cos(2*np.pi*phase/24);s=np.sin(2*np.pi*phase/24)
    loglib=np.log1p(np.asarray(lib,float))
    loglib=(loglib-loglib.mean())/(loglib.std()+1e-12)
    dummy=pd.get_dummies(pd.Series(subtype,dtype='str'),drop_first=True,dtype=float)
    nuisance=np.column_stack([np.ones(len(phase)),loglib,dummy.to_numpy()])
    q,r,piv=linalg.qr(nuisance,mode='economic',pivoting=True)
    tol=max(nuisance.shape)*np.finfo(float).eps*max(abs(r[0,0]),1.)
    rank=int(np.sum(abs(np.diag(r))>tol))
    nuisance=nuisance[:,sorted(piv[:rank])]
    residual=np.column_stack([c,s])-nuisance@np.linalg.lstsq(nuisance,np.column_stack([c,s]),rcond=None)[0]
    sigma=float(np.linalg.svd(residual,compute_uv=False)[-1]/max(np.sqrt(len(phase)),1))
    if sigma<threshold: return None,sigma
    design=np.column_stack([nuisance,c,s])
    return np.linalg.pinv(design)[-2:,:],sigma


def signflip_exact(d,rng):
    n=len(d)
    if n<4:return np.nan
    v=float(np.linalg.norm(d.mean(axis=0)))
    if n<=15:
        bits=np.arange(2**n,dtype=np.uint32)[:,None]
        signs=np.where(((bits>>np.arange(n))&1)==1,1.,-1.)
        null=np.linalg.norm(signs@d/n,axis=1)
        return float((null>=v-1e-12).mean())
    # Same MC count and seed as original implementation; local reseeding intentional.
    local=np.random.default_rng(12043)
    signs=local.choice([-1.,1.],size=(10000,n))
    null=np.linalg.norm(signs@d/n,axis=1)
    return float((1+(null>=v-1e-12).sum())/(1+len(null)))


def nice(s):
    return re.sub(r'^(HALLMARK_|REACTOME_)','',s).replace('_',' ').title()


def plot_top(frame,collection,out,top):
    d=frame[(frame.collection==collection)&frame.paired_p.notna()].sort_values(['paired_p','paired_delta_norm'],ascending=[True,False]).head(top)
    if len(d)==0:return
    fig,ax=plt.subplots(figsize=(10,max(5,0.36*len(d)+1.5)))
    y=np.arange(len(d));vals=d.paired_delta_norm.to_numpy(float)
    q=d.q_bh_within_collection.to_numpy(float)
    colors=np.where(q<.05,'#D26A23','#4C78A8')
    ax.barh(y,vals,color=colors,height=.72)
    ax.set_yticks(y,[nice(z)[:67] for z in d.pathway],fontsize=8)
    ax.invert_yaxis();ax.set_xlim(0,max(vals.max()*1.36,1e-5))
    for yi,x,p,qq in zip(y,vals,d.paired_p,q):
        ax.text(x+max(vals.max()*.025,1e-6),yi,f'p={p:.3g}; q={qq:.3g}',va='center',fontsize=7.7)
    ax.set_xlabel('Norm of patient-mean (Tumor - Normal) harmonic coefficients [log1p(CP10K)]',fontsize=9)
    ax.set_title(f'{collection}: top {len(d)} by paired sign-flip P (exploratory)\n'
                 'All eligible pathways tested; BH correction within collection; inferred phase, not clock time',fontsize=10)
    fig.tight_layout()
    for ext in ('png','pdf','svg'):
        fig.savefig(out/f'SUPP_{collection.upper()}_TOP{len(d)}_PAIRED_CONTRAST.{ext}',dpi=260,bbox_inches='tight')
    plt.close(fig)


def plot_overview(frame,out):
    fig,axs=plt.subplots(1,2,figsize=(11,4.6),constrained_layout=True)
    for ax,collection in zip(axs,['Hallmark','Reactome']):
        d=frame[(frame.collection==collection)&frame.paired_p.notna()].copy()
        if not len(d):continue
        x=d.paired_delta_norm.to_numpy(float);y=-np.log10(np.maximum(d.paired_p.to_numpy(float),1e-12))
        sig=d.q_bh_within_collection.to_numpy(float)<.05
        ax.scatter(x[~sig],y[~sig],s=15,color='#8CA9B9',alpha=.75,lw=0)
        if sig.any():ax.scatter(x[sig],y[sig],s=23,color='#D26A23',alpha=.9,lw=0,label='BH q < 0.05')
        ax.axhline(-np.log10(.05),ls='--',lw=.7,color='#999999')
        ax.set_title(f'{collection} (n={len(d)} tested)')
        ax.set_xlabel('Paired mean harmonic contrast norm')
        ax.set_ylabel('-log10(exact sign-flip P)')
        if sig.any():ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Supplementary | All-pathway paired inferred-phase association contrasts',fontsize=12)
    for ext in ('png','pdf','svg'):
        fig.savefig(out/f'SUPP_ALL_PATHWAYS_OVERVIEW.{ext}',dpi=260,bbox_inches='tight')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',default='.',help='katv2 project base directory')
    ap.add_argument('--root',default=None,help='Override GSE166555 root')
    ap.add_argument('--gmt-dir',default=None,help='Override GMT directory')
    ap.add_argument('--out',default=None)
    ap.add_argument('--min-genes',type=int,default=10,help='Same as original Figure4H: >=10 present members')
    ap.add_argument('--min-cells-per-condition',type=int,default=100)
    ap.add_argument('--min-pairs',type=int,default=5)
    ap.add_argument('--phase-sigma-min',type=float,default=.025)
    ap.add_argument('--batch-size',type=int,default=60)
    ap.add_argument('--top',type=int,default=25)
    args=ap.parse_args()
    base=Path(args.base).resolve()
    root=Path(args.root).resolve() if args.root else base/'16_FIG4_FIG5_APPLICATIONS/GSE166555'
    data=root/'anchorphase_project/data'
    gmt=Path(args.gmt_dir).resolve() if args.gmt_dir else base/'16_FIG4_FIG5_APPLICATIONS/data'
    out=Path(args.out).resolve() if args.out else root/'SUPP_ALL_PATHWAYS_PAIRED_MODULE_HARMONIC'
    out.mkdir(parents=True,exist_ok=True)
    for p in [data/'count_matrix.h5ad',data/'PREDICTIONS_BLINDED.csv.gz']:
        if not p.is_file():raise FileNotFoundError(f'Missing input: {p}')
    sets,gmt_files=read_gmt(gmt)
    print(f'GMT collections: {pd.Series([k[0] for k in sets]).value_counts().to_dict()}',flush=True)
    import anndata as ad
    a=ad.read_h5ad(data/'count_matrix.h5ad')
    pred=pd.read_csv(data/'PREDICTIONS_BLINDED.csv.gz',usecols=['cell_id','phase_mean_h'])
    if not a.obs_names.is_unique or not pred.cell_id.is_unique:raise ValueError('Duplicate cell IDs')
    ix=pd.Index(pred.cell_id.astype(str)).get_indexer(a.obs_names.astype(str))
    if (ix<0).any(): raise ValueError(f'Missing predictions for {(ix<0).sum()} H5AD cells')
    phase=np.mod(pred.iloc[ix].phase_mean_h.to_numpy(float),24)
    if not np.isfinite(phase).all():raise ValueError('Nonfinite phase')
    obs=a.obs.copy()
    for col in ('patient_id','condition','epithelial_subtype'):
        if col not in obs:raise ValueError(f'Missing adata.obs column: {col}')
    patient=obs.patient_id.astype(str).to_numpy()
    condition=obs.condition.astype(str).to_numpy()
    subtype=obs.epithelial_subtype.fillna('Unknown').astype(str).to_numpy()
    if set(condition)!={'Normal','Tumor'}:raise ValueError(f'Unexpected conditions {set(condition)}')
    count=pd.crosstab(patient,condition).reindex(columns=['Normal','Tumor'],fill_value=0)
    pairs=count[(count.Normal>=args.min_cells_per_condition)&(count.Tumor>=args.min_cells_per_condition)].index.tolist()
    if len(pairs)<args.min_pairs:raise RuntimeError(f'Only {len(pairs)} paired patients')
    X=a.X
    if sparse.issparse(X):
        X=X.tocsr().astype(np.float64)
        if X.nnz and (X.data<0).any():raise ValueError('count_matrix.X contains negative values; expected raw nonnegative counts')
        lib=np.asarray(X.sum(axis=1)).ravel()
    else:
        X=np.asarray(X,dtype=float)
        if np.any(X<0): raise ValueError('count_matrix.X contains negative values')
        lib=X.sum(axis=1)
    if not np.all(np.isfinite(lib)&(lib>0)):raise ValueError('Invalid library sizes')
    print(f'Cells={a.n_obs}, genes={a.n_vars}, eligible paired patients={len(pairs)}',flush=True)
    # Exact original module score: mean of gene-wise log1p(CP10K).
    if sparse.issparse(X):
        X=X.copy(); rowids=np.repeat(np.arange(X.shape[0]),np.diff(X.indptr))
        X.data=np.log1p(1e4*X.data/lib[rowids]);del rowids
    else:
        X=np.log1p(1e4*X/lib[:,None])
    gene_index=pd.Index(a.var_names.astype(str))
    if not gene_index.is_unique:raise ValueError('Duplicated var_names')
    eligible=[];skipped=[]
    for (collection,name),members in sorted(sets.items()):
        indices=gene_index.get_indexer(sorted(members)); indices=indices[indices>=0]
        record=dict(collection=collection,pathway=name,n_gmt_genes=len(members),n_genes_present=len(indices))
        if len(indices)<args.min_genes:
            skipped.append({**record,'reason':'below_min_genes'});continue
        eligible.append((collection,name,indices,record))
    print(f'Eligible pathways={len(eligible)}; skipped={len(skipped)}',flush=True)
    fits={};qc=[]
    for donor in pairs:
        for cond in ('Normal','Tumor'):
            rows=np.flatnonzero((patient==donor)&(condition==cond))
            proj,sigma=fit_design(phase[rows],lib[rows],subtype[rows],args.phase_sigma_min)
            fits[(donor,cond)]=(rows,proj)
            qc.append(dict(patient=donor,condition=cond,n_cells=len(rows),phase_sigma_min=sigma,fit_ok=(proj is not None)))
    pd.DataFrame(qc).to_csv(out/'00_PATIENT_CONDITION_QC.csv',index=False)
    if sum(fits[(p,'Normal')][1] is not None and fits[(p,'Tumor')][1] is not None for p in pairs)<args.min_pairs:
        raise RuntimeError('Too few donor pairs with identifiable phase regressors; inspect QC')
    recs=[];donor_rows=[]
    for start in range(0,len(eligible),args.batch_size):
        batch=eligible[start:start+args.batch_size]
        ridx=[];cidx=[];weight=[]
        for j,(_,_,indices,_) in enumerate(batch):
            ridx.extend(indices.tolist());cidx.extend([j]*len(indices));weight.extend([1/len(indices)]*len(indices))
        weights=sparse.csc_matrix((np.array(weight),(np.array(ridx),np.array(cidx))),shape=(a.n_vars,len(batch)))
        score=X@weights
        if sparse.issparse(score):score=score.toarray()
        else:score=np.asarray(score)
        b={}
        for (donor,cond),(rows,proj) in fits.items():
            if proj is not None:
                b[(donor,cond)]=proj@score[rows,:]
        for j,(collection,name,indices,record) in enumerate(batch):
            good=[donor for donor in pairs if (donor,'Normal') in b and (donor,'Tumor') in b]
            result=dict(**record,n_paired_fitted=len(good),paired_p=np.nan,paired_delta_norm=np.nan,
                        mean_delta_cos=np.nan,mean_delta_sin=np.nan,
                        Normal_mean_vector_amplitude=np.nan,Tumor_mean_vector_amplitude=np.nan,
                        Normal_mean_individual_amplitude=np.nan,Tumor_mean_individual_amplitude=np.nan)
            if len(good)>=args.min_pairs:
                norm=np.array([b[(donor,'Normal')][:,j] for donor in good])
                tum=np.array([b[(donor,'Tumor')][:,j] for donor in good])
                delta=tum-norm
                result.update(paired_p=signflip_exact(delta,None),
                    paired_delta_norm=float(np.linalg.norm(delta.mean(axis=0))),
                    mean_delta_cos=float(delta[:,0].mean()),mean_delta_sin=float(delta[:,1].mean()),
                    Normal_mean_vector_amplitude=float(np.linalg.norm(norm.mean(axis=0))),
                    Tumor_mean_vector_amplitude=float(np.linalg.norm(tum.mean(axis=0))),
                    Normal_mean_individual_amplitude=float(np.linalg.norm(norm,axis=1).mean()),
                    Tumor_mean_individual_amplitude=float(np.linalg.norm(tum,axis=1).mean()))
                for k,donor in enumerate(good):
                    donor_rows.append(dict(collection=collection,pathway=name,patient=donor,
                         Normal_bcos=norm[k,0],Normal_bsin=norm[k,1],Tumor_bcos=tum[k,0],Tumor_bsin=tum[k,1],
                         delta_bcos=delta[k,0],delta_bsin=delta[k,1]))
            recs.append(result)
        if start==0 or (start//args.batch_size+1)%5==0 or start+len(batch)==len(eligible):
            print(f'Analyzed {min(start+len(batch),len(eligible))}/{len(eligible)} pathways',flush=True)
    result=pd.DataFrame(recs)
    result['q_bh_within_collection']=np.nan
    for col in ('Hallmark','Reactome'):
        mask=result.collection.eq(col)
        result.loc[mask,'q_bh_within_collection']=bh(result.loc[mask,'paired_p'].to_numpy())
    result=result.sort_values(['collection','paired_p','paired_delta_norm'],ascending=[True,True,False])
    result.to_csv(out/'01_ALL_PATHWAY_PAIRED_HARMONIC_TESTS.csv',index=False)
    pd.DataFrame(donor_rows).to_csv(out/'02_PATIENT_LEVEL_HARMONIC_COEFFICIENTS.csv.gz',index=False,compression='gzip')
    pd.DataFrame(skipped,columns=['collection','pathway','n_gmt_genes','n_genes_present','reason']).to_csv(out/'03_EXCLUDED_PATHWAYS.csv',index=False)
    for coll in ('Hallmark','Reactome'):plot_top(result,coll,out,args.top)
    plot_overview(result,out)
    info=dict(source='GSE166555 original Figure 4H method A, generalized from 10 selected pathways to all detectable GMT pathways',
        n_cells=a.n_obs,n_genes=a.n_vars,n_paired_eligible=len(pairs),paired_patients=pairs,
        n_pathways_tested=result.groupby('collection').paired_p.count().to_dict(),gmt_files=gmt_files,
        model='Patient x condition: mean[log1p(CP10K)] pathway score ~ nuisance intercept + z(log1p(library)) + epithelial subtype dummies + cos + sin',
        score='Full pathway genes intersected with observed H5AD genes, >=min-genes. This is NOT Tumor-only ORA.',
        statistic='Euclidean norm of mean paired Tumor-Normal [bcos, bsin] differences',
        p_value='Exact sign-flip over donor pairs if <=15; otherwise 10,000 reproducible sign-flip Monte Carlo',
        multiple_testing='BH correction separately across all valid tested pathways in each collection, no ORA preselection',
        phase='AnchorPhase deterministic inferred phase_mean_h, not observed circadian time',
        limitation='Exploratory association; fitted phase may share expression information with tested genes; no true differential biological rhythmicity claim',
        min_genes=args.min_genes,min_cells_per_condition=args.min_cells_per_condition,
        min_pairs=args.min_pairs,phase_sigma_min=args.phase_sigma_min)
    (out/'README_ANALYSIS.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf8')
    print('COMPLETE:',out,flush=True)
    print(result.groupby('collection').agg(n_tested=('paired_p','count'),n_q_lt_05=('q_bh_within_collection',lambda x:int((x<.05).sum()))).to_string(),flush=True)

if __name__=='__main__':main()
