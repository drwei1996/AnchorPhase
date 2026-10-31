#!/usr/bin/env python3
"""GSE166555 Figure 4/5 unified visual-only V7. Reuses existing analyses; no B0/refit/drug downloads.
Outputs PNG + PDF. Descriptive, inferred-phase associations; not verified 24h rhythms or efficacy.
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path as MPath
from matplotlib.colors import TwoSlopeNorm
N='#2877ac'; T='#d8753a'; D='#985052'; GREEN='#3b9890'
FOCUS=['ABCB1','CDA','MGLL','PLA2G2A','SLC9A3','MAP2K2']
FIG5C=['ABCB1','MGLL','PLA2G2A','SLC9A3']  # Strong subtype sensitivity for SLC9A3; descriptive only
REFERENCE={
'ABCB1':('ENCEQUIDAR','Exposure / resistance'),
'CDA':('CEDAZURIDINE','Exposure / resistance'),
'MGLL':('ELCUBRAGISTAT','Direct / investigational'),
'PLA2G2A':('VARESPLADIB METHYL','Direct / multi-PLA2'),
'SLC9A3':('TENAPANOR','Non-oncology physiological'),
'MAP2K2':('BINIMETINIB','Direct / oncology pathway'),
'LGALS3':('BELAPECTIN','Direct / investigational'),
'ODC1':('EFLORNITHINE','Direct / investigational'),
'EPCAM':('ADECATUMUMAB','Surface-targeted delivery'),
}
REFERENCE['EPCAM']=('ADECATUMUMAB','Surface-targeted delivery')

def load(p, mandatory=True):
 if not p.exists():
  if mandatory:raise FileNotFoundError(str(p))
  return None
 return pd.read_csv(p,low_memory=False)

def save(fig,out,stem):
 fig.savefig(out/(stem+'.png'),dpi=240,bbox_inches='tight',facecolor='white')
 fig.savefig(out/(stem+'.pdf'),bbox_inches='tight',facecolor='white')
 plt.close(fig)

def fmt(x):
 return 'NA' if not np.isfinite(x) else f'{x:.3g}'

def fig4a(root,out):
 v1=root/'fig4_DETERMINISTIC_COSINOR_v1'
 n=load(v1/'01_Normal_gene_rhythmicity.csv.gz');t=load(v1/'01_Tumor_gene_rhythmicity.csv.gz')
 for d in (n,t):
  if not {'gene','q'}.issubset(d.columns) and 'rhythmic_candidate_nonCORE' not in d.columns:
   q=next((c for c in ['q_BH','BH_q','qvalue','q_nonCORE','q_bh','qvalue_BH','qval'] if c in d),None)
   if q is None:raise ValueError('Rhythmic gene tables have no recognized BH q column: '+str(list(d.columns)))
   d.rename(columns={q:'q'},inplace=True)
 # match original plug-in criteria if a hit flag exists
 def rhythmic(d):
  flag=next((c for c in ['rhythmic','is_rhythmic','rhythmic_hit','pass_q','significant'] if c in d),None)
  if 'rhythmic_candidate_nonCORE' in d:return set(d.loc[d['rhythmic_candidate_nonCORE'].astype(str).str.lower().isin(['true','1','yes']),'gene'].astype(str).str.upper())
  if flag is not None:return set(d.loc[d[flag].astype(str).str.lower().isin(['true','1','yes']),'gene'].astype(str).str.upper())
  return set(d.loc[pd.to_numeric(d.q,errors='coerce')<.05,'gene'].astype(str).str.upper())
 a,b=rhythmic(n),rhythmic(t);both=len(a&b);onlyn=len(a-b);onlyt=len(b-a)
 print(f'4A overlap from source: shared={both}, normal_only={onlyn}, tumor_only={onlyt}',flush=True)
 # The union is the universe, not two disjoint copies of shared genes.
 total=both+onlyn+onlyt
 if not total:raise ValueError('No rhythmic genes')
 fig,ax=plt.subplots(figsize=(10,5.4));ax.set_xlim(-.17,1.18);ax.set_ylim(-.08,1.09);ax.axis('off')
 # left Normal Yes (shared first, normal-only second) and Normal No (tumor-only)
 # right Tumor Yes (shared then tumor-only) and Tumor No (normal-only)
 segments=[('Shared',both,GREEN,.25),('Normal only',onlyn,N,.73),('Tumor only',onlyt,T,.47)]
 edges={'shared':(0,both/total,0,both/total), 'nonly':(both/total,(both+onlyn)/total,(both+onlyt)/total,1),
        'tonly':((both+onlyn)/total,1,both/total,(both+onlyt)/total)}
 for key,col in [('shared',GREEN),('nonly',N),('tonly',T)]:
  l0,l1,r0,r1=edges[key]
  verts=[(.07,l0),(.40,l0),(.60,r0),(.93,r0),(.93,r1),(.60,r1),(.40,l1),(.07,l1),(.07,l0)]
  codes=[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.LINETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.CLOSEPOLY]
  ax.add_patch(PathPatch(MPath(verts,codes),facecolor=col,edgecolor='white',lw=1.2,alpha=.56))
 # Group rectangles represent Normal yes/no and Tumor yes/no; stack aligns exact counts.
 for x,labels in [(.04,[('Normal rhythmic',len(a),0,(both+onlyn)/total),('Normal not rhythmic',onlyt,(both+onlyn)/total,1)]),
                  (.94,[('Tumor rhythmic',len(b),0,(both+onlyt)/total),('Tumor not rhythmic',onlyn,(both+onlyt)/total,1)])]:
  for label,v,y0,y1 in labels:
   ax.add_patch(Rectangle((x,y0),.025,y1-y0,color='#465c75',lw=0))
   if x<.5: ax.text(x-.025,(y0+y1)/2,f'{label}\n{v:,}',ha='right',va='center',fontsize=11)
   else:ax.text(x+.055,(y0+y1)/2,f'{label}\n{v:,}',ha='left',va='center',fontsize=11)
 ax.text(.5,1.055,'Phase-associated genes: Normal → Tumor',ha='center',fontsize=15,weight='bold')
 ax.text(.5,1.005,f'Union {total:,}  |  Shared {both:,}  |  Normal only {onlyn:,}  |  Tumor only {onlyt:,}',ha='center',fontsize=10)
 ax.text(.5,-.065,'Shared genes are counted ONCE in the union; set membership is not gene conversion.',ha='center',fontsize=9,color='#555')
 save(fig,out,'FIG4A_ALLUVIAL');return dict(shared=both,normal_only=onlyn,tumor_only=onlyt,union=total)

def fig4b(root,out,top=25):
 ora=load(root/'fig4_DETERMINISTIC_COSINOR_v1/08_Combined_ORA.csv')
 must={'collection','pathway','condition','q_collection','fold_enrichment'}
 if not must.issubset(ora):raise ValueError(f'4B missing {must-set(ora)}')
 ora['q_collection']=pd.to_numeric(ora.q_collection,errors='coerce');ora['fold_enrichment']=pd.to_numeric(ora.fold_enrichment,errors='coerce')
 if 'overlap_n' in ora:d=ora[(ora.q_collection<.05)&(pd.to_numeric(ora.overlap_n,errors='coerce')>=5)].copy()
 else:d=ora[ora.q_collection<.05].copy()
 d=d.sort_values('q_collection').drop_duplicates(['collection','pathway']).head(top)
 keys=list(zip(d.collection,d.pathway));mat=np.full((len(keys),2),np.nan);q=np.full_like(mat,np.nan)
 for i,(coll,name) in enumerate(keys):
  for j,cond in enumerate(['Normal','Tumor']):
   u=ora[(ora.collection==coll)&(ora.pathway==name)&(ora.condition==cond)]
   if len(u):mat[i,j]=np.log2(max(float(u.iloc[0].fold_enrichment),1e-9));q[i,j]=float(u.iloc[0].q_collection)
 if len(keys)==0:raise ValueError('No ORA pathways eligible for heatmap')
 mx=max(np.nanmax(abs(mat)),.2);fig,ax=plt.subplots(figsize=(8,max(6,0.39*len(keys)+1)))
 im=ax.imshow(np.ma.masked_invalid(mat),aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-mx,vcenter=0,vmax=mx))
 ax.set_xticks([0,1],['Normal','Tumor']);ax.set_yticks(np.arange(len(keys)),[k[1].replace('_',' ').title()[:56] for k in keys],fontsize=8)
 for i in range(len(keys)):
  for j in range(2):
   if np.isfinite(q[i,j]) and q[i,j]<.05:ax.text(j,i,'*',ha='center',va='center',fontsize=13,weight='bold',color='black' if abs(mat[i,j])<mx*.7 else 'white')
 ax.set_title('Figure 4B | Pathway enrichment: effect size',fontsize=13)
 fig.colorbar(im,ax=ax,shrink=.8,label='log2(ORA fold enrichment)')
 fig.text(.49,.005,'* Within-condition ORA BH q < 0.05; NOT a between-condition test',ha='center',fontsize=9)
 save(fig,out,'FIG4B_ORA_EFFECTSIZE_STARS_ONLY')

def choose_pathways(curves,stats):
 # Prioritize an interesting contrast plus functional diversity; avoid claiming significance if not established.
 pref=['HALLMARK_MYC_TARGETS_V1','HALLMARK_OXIDATIVE_PHOSPHORYLATION','HALLMARK_TNFA_SIGNALING_VIA_NFKB','HALLMARK_XENOBIOTIC_METABOLISM',
       'MYC_TARGETS_V1','OXIDATIVE_PHOSPHORYLATION','TNFA_SIGNALING_VIA_NFKB','XENOBIOTIC_METABOLISM']
 av=curves[['collection','pathway']].drop_duplicates()
 chosen=[]
 for key in pref:
  hit=av[av.pathway.astype(str).str.upper()==key]
  if len(hit):
   tup=tuple(hit.iloc[0][['collection','pathway']]);
   if tup not in chosen:chosen.append(tup)
 for r in stats.sort_values('paired_p',na_position='last').itertuples():
  tup=(r.collection,r.pathway)
  if tup not in chosen and tup in set(map(tuple,av.to_numpy())):chosen.append(tup)
  if len(chosen)>=4:break
 return chosen[:4]

def fig4c(root,out):
 directory=root/'fig4_FUNCTIONAL_PHASE_PATIENT_v2'
 curves=load(directory/'04_PATIENT_EQUAL_MODULE_CURVES.csv');stats=load(directory/'03_PAIRED_PATIENT_MODULE_COMPARISON.csv')
 selected=choose_pathways(curves,stats)
 if not selected:raise ValueError('4C no pathways with fitted curves')
 for with_text in [True,False]:
  fig,axs=plt.subplots(len(selected),1,figsize=(8,2.3*len(selected)),sharex=True,squeeze=False)
  for i,(coll,path) in enumerate(selected):
   ax=axs[i,0];sub=curves[(curves.collection==coll)&(curves.pathway==path)]
   for cond,col in [('Normal',N),('Tumor',T)]:
    d=sub[sub.condition==cond].sort_values('phase_h');x=d.phase_h.to_numpy(float)
    if len(d)==0:continue
    ax.plot(x,d.mean_centered_module_logCP10K.to_numpy(float),color=col,lw=2,label=cond)
    ax.fill_between(x,d.bootstrap_lower.to_numpy(float),d.bootstrap_upper.to_numpy(float),color=col,alpha=.16)
   ax.axhline(0,color='grey',lw=.55);ax.set_xlim(0,24);ax.set_xticks([0,6,12,18,24])
   if with_text:
    k=stats[(stats.collection==coll)&(stats.pathway==path)];pv=float(k.iloc[0].paired_p) if len(k) else np.nan
    ax.set_title(f'{path.replace("_"," ").title()}   |   paired p = {fmt(pv)} (exploratory)',fontsize=10)
    ax.set_ylabel('Centered\nmodule score',fontsize=9)
   else:
    ax.set_title('');ax.set_ylabel('');ax.set_yticklabels([])
   if i==0 and with_text:ax.legend(frameon=False,ncol=2,loc='upper right',fontsize=9)
  axs[-1,0].set_xlabel('AnchorPhase inferred phase (h)' if with_text else '')
  if with_text:fig.suptitle('Figure 4C | Inferred-phase pathway associations (95% patient bootstrap CI)',fontsize=12)
  fig.tight_layout();save(fig,out,'FIG4C_REPRESENTATIVE_PATHWAYS_'+('WITH_TEXT' if with_text else 'NO_TEXT'))
 return selected

def fig5b(out,atlas,drug,max_total=12):
 cols={'Direct / investigational':'#bf704b','Direct / multi-PLA2':'#bf704b','Direct / oncology pathway':'#bf704b',
       'Exposure / resistance':'#447ca8','Surface-targeted delivery':'#3e948c','Non-oncology physiological':'#9993b5'}
 groups=['Direct / oncology pathway','Direct / investigational','Direct / multi-PLA2','Exposure / resistance','Surface-targeted delivery','Non-oncology physiological']
 drug=drug.copy();drug['drug_name']=drug.drug_name.astype(str).str.upper()
 drug['pref_rank']=drug.apply(lambda r:0 if r.gene in REFERENCE and r.drug_name==REFERENCE[r.gene][0] else 1,axis=1)
 representative=drug.sort_values(['gene','pref_rank','drug_name']).drop_duplicates('gene')[['gene','drug_name']]
 merged=atlas.merge(representative,on='gene',how='left',suffixes=('','_db'))
 merged['drug_name']=merged.drug_name.fillna('UNMAPPED')
 merged=merged[merged.gene.isin(REFERENCE)].copy()
 merged['drug_display']=merged.apply(lambda r: REFERENCE[r.gene][0] if REFERENCE[r.gene][0] in set(drug.loc[drug.gene==r.gene,'drug_name'].astype(str).str.upper()) else str(r.drug_name),axis=1)
 merged['category']=merged.gene.map({k:v[1] for k,v in REFERENCE.items()})
 merged['evidence']=pd.to_numeric(merged.paired_delta_amp,errors='coerce')
 merged=merged.dropna(subset=['evidence']).sort_values('evidence',ascending=False).head(max_total)
 if merged.empty:raise ValueError('5B no matched target evidence')
 merged=merged.sort_values(['category','evidence'],ascending=[True,False]);n=len(merged)
 fig,ax=plt.subplots(figsize=(11,max(5,.55*n+1.5)));y=np.arange(n)
 for j,(_,r) in enumerate(merged.iterrows()):
  col=cols.get(r.category,'#999');ax.hlines(j,0,r.evidence,color=col,lw=2);ax.scatter(r.evidence,j,s=90,color=col,zorder=3)
 ax.set_yticks(y,[f'{r.gene}   ·   {r.drug_display}' for r in merged.itertuples()],fontsize=10);ax.invert_yaxis()
 ax.set_xlabel('Paired first-harmonic contrast (RNA units; not efficacy)')
 ax.set_title('Figure 5B | Pharmacological mechanism groups',fontsize=14)
 for c in merged.category.drop_duplicates():
  loc=np.where(merged.category.to_numpy()==c)[0];ax.axhspan(loc.min()-.48,loc.max()+.48,color=cols.get(c,'#999'),alpha=.05)
 from matplotlib.lines import Line2D
 unique=merged.category.drop_duplicates().tolist()
 handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=cols.get(g,'#999'),markersize=8,label=g) for g in unique]
 ax.legend(handles=handles,loc='lower right',fontsize=8,frameon=False)
 ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True);ax.spines[['right','top']].set_visible(False)
 fig.tight_layout();save(fig,out,'FIG5B_MECHANISM_LOLLIPOP')
 return merged

def fig5c(out,delta,genes):
 h=np.linspace(0,24,97);co=np.cos(2*np.pi*h/24);si=np.sin(2*np.pi*h/24)
 req=['beta_cos_Normal','beta_sin_Normal','beta_cos_Tumor','beta_sin_Tumor','mean_logcp10k_Normal','mean_logcp10k_Tumor']
 if not set(req).issubset(delta):raise ValueError('5C missing columns: '+str(set(req)-set(delta)))
 selected=[g for g in genes if g in set(delta.gene)]
 for mode in ['WITH_TEXT','NO_TEXT']:
  fig,axs=plt.subplots(len(selected),3,figsize=(13,2.65*len(selected)),squeeze=False,sharex=True)
  for i,g in enumerate(selected):
   d=delta[delta.gene==g].dropna(subset=req).drop_duplicates('patient_id')
   n=len(d)
   if n<2:continue
   vn=d[req].to_numpy(float)
   normal=vn[:,4,None]+vn[:,0,None]*co+vn[:,1,None]*si
   tumor=vn[:,5,None]+vn[:,2,None]*co+vn[:,3,None]*si
   full=tumor-normal;harm=full-full.mean(axis=1,keepdims=True)
   for j,items in [(0,[(normal,N,'Normal'),(tumor,T,'Tumor')]),(1,[(full,D,'Tumor - Normal')]),(2,[(harm,D,'Harmonic contrast')])]:
    ax=axs[i,j]
    for arr,color,label in items:
     mu=arr.mean(axis=0);se=arr.std(axis=0,ddof=1)/np.sqrt(n);ax.plot(h,mu,lw=1.9,color=color,label=label);ax.fill_between(h,mu-1.96*se,mu+1.96*se,color=color,alpha=.18)
    ax.set_xlim(0,24);ax.set_xticks([0,6,12,18,24]);ax.axhline(0,color='#888',lw=.5)
    if mode=='WITH_TEXT':
     if j==0:ax.set_ylabel(f'{g}  (n={n})',fontsize=10)
     if i==0:ax.set_title(['Full fitted RNA','Tumor - Normal','Harmonic-only difference'][j],fontsize=10)
    else:ax.set_title('');ax.set_ylabel('');ax.set_yticklabels([])
   if i==0 and mode=='WITH_TEXT':axs[i,0].legend(loc='best',frameon=False,fontsize=8)
  if mode=='WITH_TEXT':
   for ax in axs[-1]:ax.set_xlabel('AnchorPhase inferred phase (h)',fontsize=9)
   fig.suptitle('Figure 5C | Patient-equal RNA expression proxies, NOT drug response',fontsize=13)
  fig.tight_layout();save(fig,out,'FIG5C_SELECTED_TARGET_CURVES_'+mode)

def fig5d(root,out,delta):
 # use true LOO projections from V5 if available, to avoid leakage from full-cohort direction
 loo=load(root/'FIG45_ALL125_DRUG_SUBTYPE_v5/02_ALL125_PATIENT_LOO.csv.gz',False)
 if loo is not None and {'gene','patient_id','loo_projection'}.issubset(loo):
  d=loo[loo.gene.isin(FOCUS)][['gene','patient_id','loo_projection']].copy();d=d.rename(columns={'loo_projection':'projection'})
 else:
  records=[]
  for g,sub in delta[delta.gene.isin(FOCUS)].groupby('gene'):
   a=sub.dropna(subset=['dc','ds']).drop_duplicates('patient_id');v=a[['dc','ds']].to_numpy(float)
   if len(v)<3:continue
   for i,pid in enumerate(a.patient_id):
    ref=(v.sum(0)-v[i])/(len(v)-1);norm=np.linalg.norm(ref);val=float(v[i]@ref/norm) if norm>1e-12 else np.nan
    records.append(dict(gene=g,patient_id=pid,projection=val))
  d=pd.DataFrame(records)
 if d.empty:raise ValueError('5D no LOO projections')
 order=[g for g in FOCUS if g in set(d.gene)];fig,ax=plt.subplots(figsize=(10,5.1));rng=np.random.default_rng(26)
 xmax=max(.3,float(d.projection.max(skipna=True))*1.25)
 for j,g in enumerate(order):
  x=d[d.gene==g].projection.dropna().to_numpy(float);y=j+rng.uniform(-.13,.13,len(x))
  ax.scatter(x,y,s=52,color=N,alpha=.68,edgecolor='white',linewidth=.5)
  if len(x):ax.scatter([x.mean()],[j],marker='D',s=120,c=T,zorder=5,edgecolor='white',lw=.5)
  ax.text(xmax*.985,j,f'{sum(x>0)}/{len(x)}',ha='right',va='center',fontsize=10,color='#555')
 ax.axvline(0,color='gray',lw=.9);ax.set_xlim(min(-.1,float(d.projection.min(skipna=True))*1.2),xmax)
 ax.set_yticks(range(len(order)),order);ax.invert_yaxis();ax.set_xlabel('Held-out patient projection onto OTHER patients\' mean (LOO)')
 ax.set_title('Figure 5D | Leave-one-patient-out directional agreement',fontsize=13)
 ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.spines[['right','top']].set_visible(False)
 fig.tight_layout();save(fig,out,'FIG5D_PATIENT_LOO_COMPACT')

def fig5e(out, selected, allpairs, max_per_target=2):
 """Verified-edge bipartite network, at most 2 molecules per focus target.
 Only edges actually present in ChEMBL match table are drawn. No fake links.
 """
 from matplotlib.patches import FancyBboxPatch, PathPatch
 from matplotlib.path import Path as MP
 import textwrap
 colors={'Exposure / resistance':'#47789f','Direct / investigational':'#c16e4d',
         'Direct / multi-PLA2':'#c79656','Direct / oncology pathway':'#8870aa',
         'Non-oncology physiological':'#71a09a','Surface-targeted delivery':'#5ca79e'}
 pairs=allpairs.copy()
 if not {'gene','drug_name'}.issubset(pairs):raise ValueError('Missing genuine ChEMBL gene/drug_name links')
 pairs['gene']=pairs.gene.astype(str).str.upper().str.strip()
 pairs['drug_name']=pairs.drug_name.astype(str).str.upper().str.strip()
 pairs=pairs[pairs.gene.isin(FOCUS)&~pairs.drug_name.isin(['','NAN','NONE','UNKNOWN'])].copy()
 pairs=pairs.drop_duplicates(['gene','drug_name'])
 # An explicitly preselected representative ranks first ONLY if it exists in observed edges
 records=[];counts=[]
 for g in FOCUS:
  available=pairs.loc[pairs.gene==g,'drug_name'].unique().tolist()
  if not available:continue
  key=REFERENCE[g][0].upper()
  chosen=sorted(available,key=lambda x:(x!=key,len(x),x))[:max_per_target]
  counts.append(dict(gene=g,total_observed=len(available),displayed=len(chosen),collapsed=len(available)-len(chosen)))
  for d in chosen:
   records.append(dict(gene=g,drug_name=d,mechanism_class=REFERENCE[g][1],
                       preferred_reference_present=key in available,source='Observed ChEMBL match'))
 edge=pd.DataFrame(records)
 if edge.empty:raise ValueError('No observed focus target ChEMBL edges')
 edge.to_csv(out/'FIG5E_NETWORK_EDGES_AUDIT.csv',index=False)
 pd.DataFrame(counts).to_csv(out/'FIG5E_COLLAPSED_COUNTS.csv',index=False)
 targets=[g for g in FOCUS if g in set(edge.gene)]
 # group drugs by linked target, preserving readability; many-to-many shared drugs get ONE node
 unique_drugs=[]
 for g in targets:
  for d in edge.loc[edge.gene==g,'drug_name']:
   if d not in unique_drugs:unique_drugs.append(d)
 n=max(len(unique_drugs),len(targets))
 fig,ax=plt.subplots(figsize=(14,max(6.5,n*.68+2.5)))
 ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
 # Use broad vertical range, fixed minimum spacing, no molecule labels crossing
 yt={g:y for g,y in zip(targets,np.linspace(.88,.12,len(targets)))}
 yd={d:y for d,y in zip(unique_drugs,np.linspace(.88,.12,len(unique_drugs)))}
 xL,xR=.22,.73
 ax.add_patch(FancyBboxPatch((.04,.065),.23,.855,boxstyle='round,pad=.008',facecolor='#f4f7fa',edgecolor='#e0e6ed',lw=.8))
 ax.add_patch(FancyBboxPatch((.69,.065),.29,.855,boxstyle='round,pad=.008',facecolor='#f7f8fa',edgecolor='#e0e6ed',lw=.8))
 for r in edge.itertuples():
  yy0,yy1=yt[r.gene],yd[r.drug_name]
  verts=[(xL,yy0),(.40,yy0),(.56,yy1),(xR,yy1)]
  path=MP(verts,[MP.MOVETO,MP.CURVE4,MP.CURVE4,MP.CURVE4])
  ax.add_patch(PathPatch(path,facecolor='none',edgecolor=colors[r.mechanism_class],lw=2,alpha=.51,zorder=1))
 summary={r['gene']:r for r in counts}
 for g,y in yt.items():
  c=colors[REFERENCE[g][1]]
  ax.scatter([xL],[y],s=230,color=c,edgecolors='white',lw=1.2,zorder=4)
  ax.text(.192,y+.012,g,va='center',ha='right',fontsize=12,fontweight='bold')
  x=summary[g]
  ax.text(.192,y-.020,f"{x['displayed']} shown / {x['total_observed']} observed",va='center',ha='right',fontsize=8.5,color='#687684')
 for d,y in yd.items():
  involved=edge.loc[edge.drug_name==d,'gene'].tolist()
  col=colors[REFERENCE[involved[0]][1]] if involved else '#596c80'
  ax.scatter([xR],[y],s=160,color=col,edgecolors='white',lw=1.0,zorder=4)
  label=(d[:28]+'…') if len(d)>29 else d
  ax.text(.754,y,label,va='center',ha='left',fontsize=10)
 ax.text(.22,.956,'TARGET',fontsize=12,fontweight='bold',ha='center')
 ax.text(.79,.956,'OBSERVED MOLECULE',fontsize=12,fontweight='bold',ha='center')
 from matplotlib.lines import Line2D
 used=list(dict.fromkeys(REFERENCE[g][1] for g in targets))
 handles=[Line2D([0],[0],lw=3,color=colors[u],label=u) for u in used]
 fig.legend(handles=handles,ncol=min(3,len(handles)),loc='lower center',bbox_to_anchor=(.5,.01),frameon=False,fontsize=9)
 fig.suptitle('Figure 5E | Auditable target–molecule network',fontsize=15,y=.995)
 fig.text(.5,.005,'Only observed ChEMBL matches; up to two displayed per target; RNA association ≠ therapeutic efficacy.',ha='center',fontsize=8)
 save(fig,out,'FIG5E_GENUINE_BIPARTITE_NETWORK')
 return counts

def fig4d(root,out):
 """ORA summary: show two conditions separately; avoid false claim differential enrichment."""
 d=load(root/'fig4_DETERMINISTIC_COSINOR_v1/08_Combined_ORA.csv')
 req={'collection','pathway','condition','q_collection','fold_enrichment'}
 if req-set(d):raise ValueError(f'FIG4D missing columns {req-set(d)}')
 d=d.copy();d['q_collection']=pd.to_numeric(d.q_collection,errors='coerce');d['fold_enrichment']=pd.to_numeric(d.fold_enrichment,errors='coerce')
 fig,axs=plt.subplots(1,2,figsize=(13,6),layout='constrained')
 for ax,cond in zip(axs,['Normal','Tumor']):
  sub=d[(d.condition==cond)&(d.q_collection<.05)&(d.fold_enrichment>0)].sort_values('q_collection').drop_duplicates(['collection','pathway']).head(12)
  if sub.empty:ax.text(.5,.5,'No qualified pathways',transform=ax.transAxes,ha='center');continue
  sub=sub.sort_values('fold_enrichment')
  colors=[N if str(c).lower().startswith('hallmark') else GREEN for c in sub.collection]
  ax.scatter(sub.fold_enrichment,np.arange(len(sub)),s=70,c=colors)
  ax.set_yticks(np.arange(len(sub)),[x.replace('_',' ').title()[:46] for x in sub.pathway],fontsize=8)
  ax.set_xlabel('ORA fold enrichment (not differential)');ax.set_title(cond)
  ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True)
 fig.suptitle('Figure 4D | Within-group phase-associated gene enrichment',fontsize=14)
 save(fig,out,'FIG4D_FUNCTIONAL_ENRICHMENT')

def fig4e(root,out,atlas):
 """Whole-target landscape; focus targets always marked as illustrative, not validated."""
 must={'gene','tumor_amp','paired_delta_amp'}
 # adapt to actual V2 columns using conservative exact alternatives
 xcol=next((x for x in ['tumor_amplitude','tumor_amp','amplitude_Tumor','amp_Tumor'] if x in atlas),None)
 ycol=next((x for x in ['paired_delta_amp','paired_harmonic_contrast','delta_amp'] if x in atlas),None)
 if xcol is None or ycol is None:raise ValueError('Figure4E needs tumor amplitude and paired contrast columns; found '+str(list(atlas.columns)))
 d=atlas[['gene',xcol,ycol]].copy()
 for x in (xcol,ycol):d[x]=pd.to_numeric(d[x],errors='coerce')
 d=d.dropna().drop_duplicates('gene');focus=d[d.gene.isin(FOCUS)]
 fig,ax=plt.subplots(figsize=(9,6));ax.scatter(d[xcol],d[ycol],s=24,color='#b2bcc5',alpha=.6,label='Mechanism-linked target')
 ax.scatter(focus[xcol],focus[ycol],s=92,color=T,edgecolors='white',zorder=5,label='Illustrative targets')
 for r in focus.itertuples(index=False):ax.annotate(r.gene,(getattr(r,xcol),getattr(r,ycol)),xytext=(5,5),textcoords='offset points',fontsize=9)
 ax.set_xlabel('Tumor first-harmonic amplitude (logCP10K)');ax.set_ylabel('Paired harmonic contrast (logCP10K)')
 ax.set_title('Figure 4E | Mechanism-linked expression landscape');ax.legend(frameon=False)
 ax.spines[['top','right']].set_visible(False);fig.tight_layout();save(fig,out,'FIG4E_TARGET_LANDSCAPE')

def fig4f_and_fig5f(root,out,delta):
 """Patient bootstrap uncertainty for focus genes, without claiming null-calibrated effects."""
 rng=np.random.default_rng(20261026);rows=[]
 for g in FOCUS:
  s=delta[(delta.gene==g)].dropna(subset=['dc','ds']).drop_duplicates('patient_id')
  if len(s)<3:continue
  v=s[['dc','ds']].to_numpy(float);mean=v.mean(0);magnitude=float(np.linalg.norm(mean))
  ids=rng.integers(len(v),size=(3000,len(v)));boot=np.linalg.norm(v[ids].mean(axis=1),axis=1)
  ci=np.quantile(boot,[.025,.975]);amp1=np.hypot(s.beta_cos_Normal,s.beta_sin_Normal).to_numpy(float)
  amp2=np.hypot(s.beta_cos_Tumor,s.beta_sin_Tumor).to_numpy(float)
  change=amp2-amp1;change_ci=np.quantile(change[ids].mean(axis=1),[.025,.975])
  # true held-out directional prediction
  lo=[]
  for i in range(len(v)):
   ref=np.mean(np.delete(v,i,axis=0),axis=0);nm=np.linalg.norm(ref)
   if nm>1e-10:lo.append(v[i]@ref/nm)
  rows.append(dict(gene=g,n=len(v),contrast=magnitude,contrast_lo=ci[0],contrast_hi=ci[1],
                   signed_change=change.mean(),signed_lo=change_ci[0],signed_hi=change_ci[1],
                   loo_positive=sum(q>0 for q in lo),loo_evaluable=len(lo)))
 d=pd.DataFrame(rows)
 if d.empty:raise ValueError('No paired patients for uncertainty plots')
 d.to_csv(out/'FIG45_FOCUS_GENE_PATIENT_BOOTSTRAP.csv',index=False)
 fig,ax=plt.subplots(figsize=(9,4.5));yy=np.arange(len(d))
 ax.errorbar(d.signed_change,yy,xerr=[d.signed_change-d.signed_lo,d.signed_hi-d.signed_change],fmt='o',color=T,capsize=3)
 ax.set_yticks(yy,d.gene);ax.invert_yaxis();ax.axvline(0,color='gray',lw=.8)
 ax.set_xlabel('Patient-mean Tumor−Normal harmonic amplitude (95% bootstrap CI)')
 ax.set_title('Figure 4F | Amplitude changes in illustrative targets');fig.tight_layout();save(fig,out,'FIG4F_PAIRED_AMPLITUDE_CI')
 fig,axs=plt.subplots(1,3,figsize=(15,4.8),sharey=True,layout='constrained')
 for ax,typ,lo,hi,col,lab in [(axs[0],'contrast','contrast_lo','contrast_hi',N,'Paired harmonic contrast'),
                              (axs[1],'signed_change','signed_lo','signed_hi',T,'Tumor−Normal amplitude')]:
  ax.errorbar(d[typ],yy,xerr=[d[typ]-d[lo],d[hi]-d[typ]],fmt='o',color=col,capsize=3)
  ax.set_xlabel(lab+'\n95% patient bootstrap CI');ax.axvline(0,color='gray',lw=.7)
 axs[2].scatter(d.loo_positive/d.loo_evaluable,yy,c=GREEN,s=75)
 for y,r in enumerate(d.itertuples()):axs[2].text(min(1.04,r.loo_positive/r.loo_evaluable+.015),y,f'{r.loo_positive}/{r.loo_evaluable}',va='center',fontsize=9)
 axs[2].set_xlim(0,1.16);axs[2].axvline(.5,ls='--',color='gray',lw=.8);axs[2].set_xlabel('Held-out patient directional agreement')
 axs[0].set_yticks(yy,d.gene);axs[0].invert_yaxis();fig.suptitle('Figure 5F | RNA association, uncertainty and held-out consistency')
 save(fig,out,'FIG5F_RNA_EVIDENCE_WITH_UNCERTAINTY')

def fig5a(root,out,atlas):
 """Counts from existing source, no arbitrary effect-score threshold."""
 n=atlas.gene.nunique()
 p=root/'FIG45_ALL125_DRUG_SUBTYPE_v5/01_ALL125_PATIENT_SIGNFLIP_BH.csv'
 q=load(p,False)
 vals=[n]
 labels=['Mechanism-linked genes']
 if q is not None and 'gene' in q:
  vals.append(q.gene.nunique());labels.append('Patient-tested candidate genes')
  qc=next((c for c in ['BH_q_all_eligible','BH_q','q_BH'] if c in q),None)
  if qc:
   vals.append(int((pd.to_numeric(q[qc],errors='coerce')<.05).sum()));labels.append('Patient-level BH q < 0.05')
 fig,ax=plt.subplots(figsize=(8,3.7));yy=np.arange(len(vals));ax.barh(yy,vals,color=[T,N,GREEN][:len(vals)],height=.5)
 for i,x in enumerate(vals):ax.text(x+max(vals)*.015,i,str(x),va='center')
 ax.set_yticks(yy,labels);ax.invert_yaxis();ax.set_xlim(0,max(vals)*1.2);ax.set_xlabel('Unique genes')
 ax.set_title('Figure 5A | Auditable candidate filtering');ax.spines[['top','right']].set_visible(False)
 fig.tight_layout();save(fig,out,'FIG5A_CANDIDATE_COUNTS')

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='16_FIG4_FIG5_APPLICATIONS/GSE166555');p.add_argument('--output');p.add_argument('--top-pathways',type=int,default=25);p.add_argument('--fig5c-genes',default=','.join(FIG5C));p.add_argument('--max-drugs-per-target',type=int,default=2);p.add_argument('--strict',action='store_true',help='stop if any requested panel cannot be generated');args=p.parse_args()
 root=Path(args.root);out=Path(args.output) if args.output else root/'FIG45_PUBLICATION_REPLOT_v7';out.mkdir(parents=True,exist_ok=True)
 report={'source_root':str(root),'output':str(out),'panels':{},'warnings':[]}
 def run(name,fun):
  try:r=fun();report['panels'][name]='PASS';return r
  except Exception as e:
   report['panels'][name]='FAILED: '+str(e);report['warnings'].append(f'{name}: {e}');print('WARN',name,str(e),flush=True)
   if args.strict:raise
   return None
 run('FIG4A',lambda:fig4a(root,out))
 run('FIG4B',lambda:fig4b(root,out,args.top_pathways))
 run('FIG4C',lambda:fig4c(root,out))
 run('FIG4D',lambda:fig4d(root,out))
 v2=root/'FIG45_MECHANISM_PHASE_v2'
 atlas=run('READ_ATLAS',lambda:load(v2/'04_FIG4_DRUGGABLE_RHYTHMIC_TARGET_ATLAS.csv.gz'))
 delta=run('READ_PAIRED',lambda:load(v2/'03_PATIENT_PAIRED_DELTAS.csv.gz'))
 pairs=run('READ_DRUG_PAIRS',lambda:load(v2/'06_FIG5_ALL_MECHANISM_QUALIFIED_MATCHES.csv.gz'))
 if atlas is not None and pairs is not None:
  atlas.gene=atlas.gene.astype(str).str.upper();pairs.gene=pairs.gene.astype(str).str.upper()
  run('FIG4E',lambda:fig4e(root,out,atlas))
  run('FIG5A',lambda:fig5a(root,out,atlas))
  run('FIG5B',lambda:fig5b(out,atlas,pairs))
  run('FIG5E',lambda:fig5e(out,atlas,pairs,args.max_drugs_per_target))
 if delta is not None:
  delta.gene=delta.gene.astype(str).str.upper()
  run('FIG5C',lambda:fig5c(out,delta,[g.strip().upper() for g in args.fig5c_genes.split(',') if g.strip()]))
  run('FIG5D',lambda:fig5d(root,out,delta))
  run('FIG4F_FIG5F',lambda:fig4f_and_fig5f(root,out,delta))
 (out/'00_REPLOT_MANIFEST.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
 print(json.dumps(report['panels'],indent=2,ensure_ascii=False),flush=True)
 if report['warnings']:sys.exit(2)

if __name__=='__main__':main()
