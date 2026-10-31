#!/usr/bin/env python3
"""GSE166555 Figure4 V1: deterministic AnchorPhase phase_mean_h plug-in Cosinor.
Group-specific exploratory OLS partial-F q-values, patient-robust sensitivity p, CORE17 curves, ORA.
CORE17 is used to infer phase and therefore is an IN-SAMPLE DIAGNOSTIC.
"""
import os
for x in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[x]='1'
import argparse, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import sparse
from scipy.stats import f as fdist, hypergeom
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

CORE=['ARNTL','BHLHE41','CLOCK','CRY1','CRY2','DBP','FBXO21','HLF','NFIL3','NPAS2','NR1D1','NR1D2','PER1','PER2','PER3','RORC','TEF']

def bh(v):
 v=np.array(v,float); q=np.full(len(v),np.nan); ix=np.where(np.isfinite(v))[0]
 if len(ix):
  order=ix[np.argsort(v[ix])]; tmp=np.minimum.accumulate((v[order]*len(ix)/np.arange(1,len(ix)+1))[::-1])[::-1]
  q[order]=np.minimum(tmp,1)
 return q

def read_gmt(folder):
 files=list(folder.rglob('*.gmt'))+list(folder.rglob('*.gmt.gz'))
 selected=[]
 for p in files:
  name=p.name.lower()
  typ='Hallmark' if ('h.all.' in name or 'hallmark' in name) else ('Reactome' if 'reactome' in name else None)
  if typ: selected.append((p,typ))
 if not selected: raise RuntimeError(f'No Hallmark/Reactome GMT found in {folder}')
 import gzip
 sets={}
 for path,typ in selected:
  op=gzip.open if path.suffix=='.gz' else open
  with op(path,'rt',encoding='utf8') as fh:
   for line in fh:
    tok=line.rstrip('\n').split('\t')
    if len(tok)>3:sets[(typ,tok[0])]=set(tok[2:])
 return sets,[(str(p),t) for p,t in selected]

def construct_design(meta, lib):
 # Within-condition patient and subtype fixed effects, plus library covariate.
 patient=pd.get_dummies(meta['patient_id'].astype(str),drop_first=True,dtype=float)
 subtype=pd.get_dummies(meta['cell_subtype'].astype(str),drop_first=True,dtype=float)
 if subtype.shape[1]>0:
  # Remove redundant columns with QR pivoted rank selection.
  import scipy.linalg as la
  z=np.column_stack([np.ones(len(meta)), patient.to_numpy(),subtype.to_numpy(),np.log1p(lib)])
 else:z=np.column_stack([np.ones(len(meta)),patient.to_numpy(),np.log1p(lib)])
 import scipy.linalg as la
 _,r,piv=la.qr(z,mode='economic',pivoting=True)
 rank=int((abs(np.diag(r))>1e-9*max(abs(r[0,0]),1)).sum())
 z=z[:,sorted(piv[:rank])]
 return z

def fit_condition(a, idx, phases, batch, out, cond, min_detect):
 import scipy.linalg as la
 from scipy.stats import f as fdis
 subset=a[idx,:]
 phi=2*np.pi*phases[idx]/24
 C=np.cos(phi); S=np.sin(phi)
 lib=np.asarray(subset.X.sum(axis=1)).ravel().astype(float)
 meta=subset.obs.reset_index(drop=True)
 z=construct_design(meta,lib)
 X=np.column_stack([z,C,S])
 q,r=np.linalg.qr(X,mode='reduced'); k=X.shape[1]; n=len(idx)
 if np.linalg.matrix_rank(r)<k:raise RuntimeError(f'{cond} design rank deficient')
 ix=np.arange(k-2,k)
 invxx=np.linalg.inv(X.T@X)
 V=invxx[-2:,-2:]; Vi=np.linalg.inv(V)
 # Patient scores, sandwich covariance of last two coefficients.
 patients=meta['patient_id'].astype(str).to_numpy(); unique=np.unique(patients); G=len(unique)
 Xp=X[:,-2:]
 res=[]; core_fits={}
 print(f'{cond}: {n} cells, {G} patients, {a.n_vars} genes, design rank {k}',flush=True)
 for start in range(0,a.n_vars,batch):
  cols=np.arange(start,min(start+batch,a.n_vars))
  arr=subset.X[:,cols]
  if sparse.issparse(arr):arr=arr.toarray()
  arr=np.asarray(arr,float)
  y=np.log1p(1e4*arr/np.maximum(lib[:,None],1))
  coef=np.linalg.solve(r,q.T@y)
  pred=X@coef
  resid=y-pred
  # Primary screening: 2-df partial F test, deterministic phase plug-in.
  # Conditional on inferred phase; cell-level p-values are exploratory, not donor-level confirmation.
  rss=np.sum(resid*resid,axis=0)
  bt=coef[-2:,:]  # harmonic coefficients must be defined before calculating F
  explained=np.einsum("ac,ab,bc->c",bt,Vi,bt)
  f_ols=np.maximum(explained/2,0)/np.maximum(rss/(n-k),1e-15)
  p_ols=fdis.sf(f_ols,2,n-k)
  B=bt.shape[1]
  # Cluster robust requires all nuisance coeff contributions: full scores.
  scores_full=np.zeros((G,k,B))
  for gi,pat in enumerate(unique):
   mask=patients==pat
   scores_full[gi]=X[mask].T@resid[mask]
  transformed=np.einsum('ab,gbc->gac',invxx[-2:,:],scores_full,optimize=True)
  factor=G/(G-1)*(n-1)/(n-k) if G>1 and n>k else 1
  cov=factor*np.einsum('gac,gbc->cab',transformed,transformed,optimize=True)
  a11,a12,a22=cov[:,0,0],cov[:,0,1],cov[:,1,1]
  det=a11*a22-a12*a12
  good=(det>1e-15)&(a11>0)&(a22>0)&(G>=4)
  stat=np.full(B,np.nan)
  stat[good]=(a22[good]*bt[0,good]**2-2*a12[good]*bt[0,good]*bt[1,good]+a11[good]*bt[1,good]**2)/det[good]/2
  p=np.full(B,np.nan);p[good]=fdis.sf(np.maximum(stat[good],0),2,G-1)
  for j,g in enumerate(cols):
   name=str(a.var_names[g]); bc=float(bt[0,j]);bs=float(bt[1,j])
   if name in CORE:core_fits[name]={'bcos':bc,'bsin':bs,'cov':cov[j].copy(),'baseline':float(y[:,j].mean())}
   res.append({'gene':name,'condition':cond,'n_cells':n,'n_patients':G,
               'detected_frac':float((arr[:,j]>0).mean()),'mean_CP10K_log1p':float(y[:,j].mean()),
               'beta_cos':bc,'beta_sin':bs,'amplitude':float(np.hypot(bc,bs)),
               'peak_phase_h':float(np.mod(np.arctan2(bs,bc),2*np.pi)*24/(2*np.pi)),
               'patient_cluster_p':float(p[j]) if np.isfinite(p[j]) else np.nan,
               'cosinor_F':float(f_ols[j]),'cosinor_p':float(p_ols[j])})
  if start%max(batch*20,1)==0:print(' ',cond,start+len(cols),'/',a.n_vars,flush=True)
 df=pd.DataFrame(res)
 eligible=(df.detected_frac>=min_detect)&(~df.gene.isin(CORE))
 df['eligible_nonCORE']=eligible
 df['q_nonCORE']=np.nan
 df.loc[eligible,'q_nonCORE']=bh(df.loc[eligible,'cosinor_p'])
 df['rhythmic_candidate_nonCORE']=(df.q_nonCORE<.05)&eligible
 df['q_patient_robust_exploratory']=np.nan
 df.loc[eligible,'q_patient_robust_exploratory']=bh(df.loc[eligible,'patient_cluster_p'])
 # CORE are in-sample diagnostics only, separate BH within 17 input genes.
 coremask=df.gene.isin(CORE)
 df['core_in_sample_q']=np.nan
 df.loc[coremask,'core_in_sample_q']=bh(df.loc[coremask,'cosinor_p'])
 df.to_csv(out/f'01_{cond}_gene_rhythmicity.csv.gz',index=False)
 return df,core_fits

def ora(selected,universe,sets,condition,out,minsize,maxsize):
 U=set(universe);S=set(selected)&U
 rows=[]
 for (typ,name),genes in sets.items():
  gs=set(genes)&U
  if not minsize<=len(gs)<=maxsize:continue
  hit=gs&S
  k=len(hit);K=len(gs); n=len(S);N=len(U)
  pv=float(hypergeom.sf(k-1,N,K,n)) if n else 1.
  rows.append({'condition':condition,'collection':typ,'pathway':name,'universe_size':N,
    'rhythmic_total':n,'pathway_size':K,'overlap_n':k,'gene_ratio':k/max(n,1),
    'fold_enrichment':(k/max(n,1))/(K/max(N,1)),
    'p':pv,'hit_genes':';'.join(sorted(hit))})
 df=pd.DataFrame(rows)
 for typ in ['Hallmark','Reactome']:
  mask=df.collection==typ
  df.loc[mask,'q_collection']=bh(df.loc[mask,'p'])
  df.loc[mask].sort_values(['q_collection','p']).to_csv(out/f'07_{condition}_ORA_{typ}.csv',index=False)
 return df

def plot_ora(df,cond,path):
 fig,axs=plt.subplots(1,2,figsize=(15,7))
 for ax,typ in zip(axs,['Hallmark','Reactome']):
  d=df[(df.collection==typ)&(df.overlap_n>=2)].sort_values('q_collection').head(15).copy()
  if len(d):
   d=d.iloc[::-1]; yy=np.arange(len(d));sizes=30+np.minimum(d.overlap_n.to_numpy(),40)*5
   col=-np.log10(np.maximum(d.q_collection.to_numpy(),1e-300))
   ax.scatter(d.fold_enrichment,yy,s=sizes,c=col,cmap='viridis')
   ax.set_yticks(yy);ax.set_yticklabels([s.replace('REACTOME_','').replace('HALLMARK_','').replace('_',' ')[:43] for s in d.pathway],fontsize=8)
   ax.set_xlabel('Fold enrichment');ax.set_title(f'{cond} rhythmic candidates — {typ}')
  else:ax.text(.5,.5,'No testable hits',ha='center',va='center',transform=ax.transAxes)
 fig.tight_layout();fig.savefig(path,dpi=220,bbox_inches='tight');plt.close(fig)

def core_plots(a,dfn,dft,fn,ft,out):
 data=pd.concat([dfn[dfn.gene.isin(CORE)],dft[dft.gene.isin(CORE)]])
 data.to_csv(out/'03_CORE17_rhythmicity_summary_long.csv',index=False)
 fig,ax=plt.subplots(figsize=(5.5,9));mat=np.full((17,2),np.nan)
 for j,cond in enumerate(['Normal','Tumor']):
  df=dfn if cond=='Normal' else dft
  ix=df.set_index('gene')
  for i,g in enumerate(CORE):
   q=ix.loc[g,'core_in_sample_q'] if g in ix.index else np.nan
   mat[i,j]=-np.log10(max(float(q),1e-8)) if pd.notna(q) else np.nan
 im=ax.imshow(mat,aspect='auto',cmap='viridis',vmin=0,vmax=3)
 ax.set_xticks([0,1],['Normal','Tumor']);ax.set_yticks(np.arange(17),CORE)
 for i,g in enumerate(CORE):
  for j in range(2): ax.text(j,i,'*' if np.isfinite(mat[i,j]) and mat[i,j]>-np.log10(.05) else 'ns',ha='center',va='center',color='white',fontsize=8)
 ax.set_title('CORE17 in-sample phase association (circular validation)');fig.colorbar(im,ax=ax,label='−log10(core-set BH q)')
 fig.tight_layout();fig.savefig(out/'FIG4B_CORE17_SUMMARY.png',dpi=230);plt.close(fig)
 theta=np.arange(0,24.01,.25)
 fig,ax=plt.subplots(17,2,figsize=(13,42),sharex=True)
 records=[]
 for i,g in enumerate(CORE):
  ymax=[];ymin=[]
  for j,(cond,fits,df) in enumerate([('Normal',fn,dfn),('Tumor',ft,dft)]):
   fit=fits[g]; ang=2*np.pi*theta/24
   wave=fit['baseline']+fit['bcos']*np.cos(ang)+fit['bsin']*np.sin(ang)
   D=np.stack([np.cos(ang),np.sin(ang)],axis=1)
   var=np.einsum('pa,ab,pb->p',D,fit['cov'],D)
   # coefficient uncertainty only; omits baseline and inferred-phase errors.
   se=np.sqrt(np.maximum(var,0))
   lower=wave-1.96*se;upper=wave+1.96*se
   q=float(df.set_index('gene').loc[g,'core_in_sample_q'])
   ax[i,j].plot(theta,wave,color='#1774B8' if j==0 else '#CE6E20',lw=1.7)
   ax[i,j].fill_between(theta,lower,upper,color='#1774B8' if j==0 else '#CE6E20',alpha=.14)
   ax[i,j].set_title(f'{g} | {cond} | plug-in q={q:.2g} | amplitude={np.hypot(fit["bcos"],fit["bsin"]):.3f}',fontsize=9)
   ax[i,j].set_xlim(0,24);ax[i,j].set_xticks([0,6,12,18,24]);ax[i,j].set_xlabel('AnchorPhase inferred phase (h)')
   for t,v,lo,hi in zip(theta,wave,lower,upper):records.append({'gene':g,'condition':cond,'phase_h':t,'fitted_log1p_CP10K':v,'lower_coef_only':lo,'upper_coef_only':hi})
   ymax.append(max(upper));ymin.append(min(lower))
  for j in range(2):ax[i,j].set_ylim(min(ymin)-.05,max(ymax)+.05);ax[i,j].set_ylabel('Fitted log1p(CP10K)')
 fig.suptitle('CORE17: descriptive harmonic fits to the same genes used in phase inference\nNOT independent evidence of rhythmicity',fontsize=13)
 fig.tight_layout(rect=[0,.01,1,.98]);fig.savefig(out/'FIG4C_CORE17_ALL_17x2.png',dpi=170,bbox_inches='tight')
 with PdfPages(out/'FIG4C_CORE17_ALL_17x2.pdf') as pdf:
  # split into pages of 5 genes for legible PDF
  for s in range(0,17,5):
   ff,aa=plt.subplots(min(5,17-s),2,figsize=(13, min(5,17-s)*2.6),squeeze=False)
   for ii in range(min(5,17-s)):
    gene=CORE[s+ii]
    for jj,c in enumerate(['Normal','Tumor']):
     d=[r for r in records if r['gene']==gene and r['condition']==c]
     aa[ii,jj].plot([r['phase_h'] for r in d],[r['fitted_log1p_CP10K'] for r in d])
     aa[ii,jj].fill_between([r['phase_h'] for r in d],[r['lower_coef_only'] for r in d],[r['upper_coef_only'] for r in d],alpha=.2)
     aa[ii,jj].set_title(gene+' — '+c);aa[ii,jj].set_xlabel('Inferred phase (h)');aa[ii,jj].set_xlim(0,24)
   ff.tight_layout();pdf.savefig(ff);plt.close(ff)
 plt.close(fig)
 pd.DataFrame(records).to_csv(out/'04_CORE17_curves_long.csv',index=False)

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--root',default='16_FIG4_FIG5_APPLICATIONS/GSE166555')
 p.add_argument('--gmt-dir',default='16_FIG4_FIG5_APPLICATIONS/data')
 p.add_argument('--out',default=None)
 p.add_argument('--batch',type=int,default=96)
 p.add_argument('--phase-col',choices=['phase_mean_h','phase_MAP_h'],default='phase_mean_h')
 p.add_argument('--min-detect',type=float,default=.01)
 p.add_argument('--min-set',type=int,default=10)
 p.add_argument('--max-set',type=int,default=500)
 args=p.parse_args()
 import anndata as ad
 root=Path(args.root);data=root/'anchorphase_project/data'; out=Path(args.out) if args.out else root/'fig4_DETERMINISTIC_COSINOR_v1'
 out.mkdir(parents=True,exist_ok=True)
 a=ad.read_h5ad(data/'count_matrix.h5ad')
 pred=pd.read_csv(data/'PREDICTIONS_BLINDED.csv.gz')
 assert pred.cell_id.is_unique and a.obs_names.is_unique
 positions=pd.Index(pred.cell_id.astype(str)).get_indexer(a.obs_names.astype(str))
 assert (positions>=0).all(), 'H5AD vs predictions: missing cell IDs'
 phases=pred.iloc[positions][args.phase_col].to_numpy(float)
 assert np.isfinite(phases).all() and ((phases>=0)&(phases<=24)).all(), 'invalid phase'
 phases=np.mod(phases,24.0)
 assert all(g in a.var_names for g in CORE), 'Missing CORE17 genes'
 assert set(['patient_id','epithelial_subtype','condition']).issubset(a.obs.columns)
 assert a.n_obs==len(pred), f'Prediction cells {len(pred)} != H5AD {a.n_obs}'
 print('INPUT VALIDATION PASS:',a.shape,'matching prediction cells:',len(pred),flush=True)
 a.obs['patient_id']=a.obs['patient_id'].astype(str)
 a.obs['cell_subtype']=a.obs['epithelial_subtype'].fillna('Unknown').astype(str)
 cond=a.obs['condition'].astype(str).to_numpy()
 assert set(cond)=={'Normal','Tumor'}
 allresults={};fits={}
 for c in ['Normal','Tumor']:
  ix=np.where(cond==c)[0]
  allresults[c],fits[c]=fit_condition(a,ix,phases,args.batch,out,c,args.min_detect)
 rhythmic={c:set(d.loc[d.rhythmic_candidate_nonCORE,'gene']) for c,d in allresults.items()}
 common=rhythmic['Normal']&rhythmic['Tumor']
 counts={'Normal':len(rhythmic['Normal']),'Tumor':len(rhythmic['Tumor']), 'shared':len(common),'normal_only':len(rhythmic['Normal']-common),'tumor_only':len(rhythmic['Tumor']-common)}
 for c in ['Normal','Tumor']:(out/f'05_{c}_rhythmic_gene_list.txt').write_text('\n'.join(sorted(rhythmic[c]))+'\n')
 (out/'00_COUNTS.json').write_text(json.dumps(counts,indent=2))
 (out/'05_Shared_rhythmic_gene_list.txt').write_text('\n'.join(sorted(common))+'\n')
 fig,ax=plt.subplots(figsize=(8,5));ax.bar(['Normal','Tumor'],[counts['Normal'],counts['Tumor']],color=['#1774B8','#CE6E20']);
 for i,c in enumerate(['Normal','Tumor']):ax.text(i,counts[c],str(counts[c]),ha='center',va='bottom')
 ax.set_ylabel('Plug-in cosinor BH q<0.05 (exploratory)');ax.set_title(f'Shared={counts["shared"]} | Normal-only={counts["normal_only"]} | Tumor-only={counts["tumor_only"]}')
 fig.tight_layout();fig.savefig(out/'FIG4A_RHYTHMIC_GENE_COUNTS.png',dpi=250);plt.close(fig)
 core_plots(a,allresults['Normal'],allresults['Tumor'],fits['Normal'],fits['Tumor'],out)
 sets,gmtfiles=read_gmt(Path(args.gmt_dir))
 ora_results={}
 for c in ['Normal','Tumor']:
  universe=allresults[c].loc[allresults[c].eligible_nonCORE,'gene'].tolist()
  ora_results[c]=ora(rhythmic[c],universe,sets,c,out,args.min_set,args.max_set)
  plot_ora(ora_results[c],c,out/f'FIG4{"D" if c=="Normal" else "E"}_{c}_ORA.png')
 combined=pd.concat([d.assign(condition=c) for c,d in ora_results.items()]);combined.to_csv(out/'08_Combined_ORA.csv',index=False)
 top=combined.sort_values('q_collection').drop_duplicates('pathway').head(24).pathway.tolist()
 mat=np.full((len(top),2),np.nan)
 for j,c in enumerate(['Normal','Tumor']):
  indexed=ora_results[c].set_index('pathway')
  for i,g in enumerate(top):
   if g in indexed.index:mat[i,j]=min(-np.log10(max(indexed.loc[g,'q_collection'],1e-50)),20)
 fig,ax=plt.subplots(figsize=(8,9));im=ax.imshow(mat,aspect='auto',cmap='viridis',vmin=0,vmax=10)
 ax.set_xticks([0,1],['Normal','Tumor']);ax.set_yticks(np.arange(len(top)),[x.replace('REACTOME_','').replace('HALLMARK_','').replace('_',' ')[:55] for x in top],fontsize=8)
 ax.set_title('ORA of separately discovered phase-associated genes');fig.colorbar(im,ax=ax,label='−log10(BH q), capped at 20');fig.tight_layout();fig.savefig(out/'FIG4F_NORMAL_TUMOR_PATHWAY_HEATMAP.png',dpi=220,bbox_inches='tight');plt.close(fig)
 with PdfPages(out/'FIGURE4_OVERVIEW.pdf') as pdf:
  import matplotlib.image as mpimg
  for fname in ['FIG4A_RHYTHMIC_GENE_COUNTS.png','FIG4B_CORE17_SUMMARY.png','FIG4D_Normal_ORA.png','FIG4E_Tumor_ORA.png','FIG4F_NORMAL_TUMOR_PATHWAY_HEATMAP.png']:
   ff,aa=plt.subplots(figsize=(13,9));aa.imshow(mpimg.imread(out/fname));aa.axis('off');ff.tight_layout();pdf.savefig(ff);plt.close(ff)
 report={'counts':counts,'gmt_files':gmtfiles,'phase_column':args.phase_col,'core_in_sample_not_independent':True,'dataset':'GSE166555','method':'Deterministic phase_mean_h plug-in cosinor OLS partial F(2,n-k), patient/subtype fixed effects and log-lib covariate; BH q separately per group. Cluster-patient robust Wald sensitivity p included but not primary discovery.','caveats':['CORE17 used to infer phases, thus core gene q/curves are circular in-sample diagnostics','Normal and Tumor separate significance does NOT imply differential rhythmicity','Primary cell-level partial F p-values are conditional exploratory tests, NOT patient-independent biological replication. Patient cluster robust p columns are secondary sensitivity and must be interpreted cautiously.','Gene sets ORA BH q accounts for pathways but not gene correlation','Normal and Tumor gene universe/detection differ; compare counts descriptively only']}
 (out/'README_RESULTS.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
 print('SUCCESS',out,counts,flush=True)
if __name__=='__main__':main()
