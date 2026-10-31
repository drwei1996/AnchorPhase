#!/usr/bin/env python3
"""GSE166555: convert sample-wise gene x cell gzipped UMI TSV in RAW.tar to epithelial-only sparse AnnData.
No phase inference or rhythmicity statistics performed here.
"""
from __future__ import annotations
import argparse, gzip, json, re, tarfile, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import anndata as ad

REQUIRED = ['cell_id','sample_id','case_id','sample_origin','main_cell_type','cell_type_epi_simple','nCount_RNA','nFeature_RNA']

def run(args):
    datadir=Path(args.data_dir).expanduser().resolve()
    tarpath=datadir/'GSE166555_RAW.tar'
    metapath=datadir/'GSE166555_meta_data.tsv.gz'
    if not tarpath.exists() or not metapath.exists():
        raise FileNotFoundError(f'Required files missing from {datadir}')
    out=Path(args.output).expanduser().resolve() if args.output else datadir.parent/'anchorphase_project'/'data'/'count_matrix.h5ad'
    out.parent.mkdir(parents=True,exist_ok=True)
    print('Reading metadata...',flush=True)
    meta=pd.read_csv(metapath,sep='\t',low_memory=False, usecols=REQUIRED)
    if meta['cell_id'].duplicated().any():
        raise ValueError('Duplicate cell_id in metadata')
    meta=meta.loc[meta['main_cell_type'].eq('Epithelial')].copy()
    meta['cell_id']=meta['cell_id'].astype(str)
    meta['sample_id']=meta['sample_id'].astype(str)
    meta['case_id']=meta['case_id'].astype(str)
    meta=meta.set_index('cell_id',drop=False)
    print('Epithelial cells by condition:', meta['sample_origin'].value_counts().to_dict(),flush=True)
    expected=set(meta.index)
    blocks=[]; names_blocks=[]; gene_ref=None; seen=set(); audits=[]
    with tarfile.open(tarpath,'r') as tf:
        members=[m for m in tf.getmembers() if m.isfile() and m.name.endswith('.tsv.gz')]
        if len(members)!=25:
            print(f'WARNING: detected {len(members)} .tsv.gz files (expected 25)',flush=True)
        for num,m in enumerate(members,1):
            match=re.search(r'_(p\d+[nt](?:\d+)?)\.tsv\.gz$',Path(m.name).name)
            if not match:
                raise ValueError(f'Cannot determine sample id from archive member: {m.name}')
            sample=match.group(1)
            group=meta.loc[meta['sample_id'].eq(sample)]
            if group.empty:
                raise ValueError(f'No epithelial metadata for sample {sample}; inspect mapping')
            with gzip.GzipFile(fileobj=tf.extractfile(m),mode='rb') as gz:
                header=gz.readline().decode('utf-8-sig').rstrip('\r\n').split('\t')
            if header[0]!='gene':
                raise ValueError(f'{sample}: first header field is {header[0]!r}, expected gene')
            all_cells=header[1:]
            if len(all_cells)!=len(set(all_cells)):
                raise ValueError(f'{sample}: duplicate cell identifiers in matrix')
            if any(not x.startswith(sample+':') for x in all_cells):
                raise ValueError(f'{sample}: barcode prefix mismatch')
            selected=[x for x in all_cells if x in group.index]
            missing=set(group.index).difference(selected)
            if missing:
                raise ValueError(f'{sample}: {len(missing)} epithelial metadata cells absent from matrix; examples {list(missing)[:4]}')
            if len(selected)!=len(group):
                raise ValueError(f'{sample}: unexpected count mismatch')
            if seen.intersection(selected):
                raise ValueError(f'{sample}: duplicate cell across sample files')
            seen.update(selected)
            # usecols names ensures exact barcode matching; each chunk is gene x selected-cell matrix
            colidx={v:i for i,v in enumerate(selected)}
            chunks=[]; genes=[]; nzparts_r=[]; nzparts_c=[]; nzparts_v=[]; gi=0
            print(f'[{num:02d}/{len(members)}] {sample}: {len(selected)} epithelial / {len(all_cells)} total cells',flush=True)
            # Open a fresh member stream; avoid materializing compressed archive on disk
            with gzip.GzipFile(fileobj=tf.extractfile(m),mode='rb') as gz:
                reader=pd.read_csv(gz,sep='\t',chunksize=args.chunk_genes,usecols=['gene']+selected,
                                   dtype={c:'int32' for c in selected},low_memory=False)
                for chunk in reader:
                    gs=chunk['gene'].astype(str).tolist()
                    if gene_ref is not None:
                        ref=gene_ref[gi:gi+len(gs)]
                        if gs!=ref:
                            raise ValueError(f'{sample}: gene order/names differ from first sample near row {gi}. '
                                             'Stop rather than silently corrupt matrix.')
                    else:
                        genes.extend(gs)
                    v=chunk[selected].to_numpy(dtype=np.int32,copy=False)
                    if (v<0).any():
                        raise ValueError(f'{sample}: negative raw count found')
                    rr,cc=np.nonzero(v)
                    if len(rr):
                        nzparts_r.append(cc.astype(np.int32,copy=False)) # cells rows
                        nzparts_c.append((rr+gi).astype(np.int32,copy=False)) # genes columns
                        nzparts_v.append(v[rr,cc].astype(np.int32,copy=False))
                    gi+=len(gs)
            if gene_ref is None:
                gene_ref=genes
                if len(gene_ref)!=len(set(gene_ref)):
                    dup=pd.Series(gene_ref).value_counts()
                    raise ValueError('Duplicate gene symbols in TSV: '+str(dup[dup>1].head(10).to_dict()))
                print(f'Gene reference: {len(gene_ref):,} unique genes',flush=True)
            if gi!=len(gene_ref):
                raise ValueError(f'{sample}: gene number {gi} differs from {len(gene_ref)}')
            if nzparts_r:
                rr=np.concatenate(nzparts_r);cc=np.concatenate(nzparts_c);vv=np.concatenate(nzparts_v)
                mat=sparse.coo_matrix((vv,(rr,cc)),shape=(len(selected),len(gene_ref)),dtype=np.int32).tocsr()
            else:
                mat=sparse.csr_matrix((len(selected),len(gene_ref)),dtype=np.int32)
            del nzparts_r,nzparts_c,nzparts_v
            if np.any(np.asarray(mat.sum(axis=1)).ravel()==0):
                print(f'WARNING {sample}: cells with zero reconstructed counts detected',flush=True)
            audit=pd.DataFrame({'cell_id':selected,'sample_id':sample,
                                'umi_reconstructed':np.asarray(mat.sum(axis=1)).ravel(),
                                'genes_reconstructed':np.diff(mat.indptr)})
            audit['nCount_RNA_meta']=pd.to_numeric(meta.loc[selected,'nCount_RNA'],errors='coerce').to_numpy()
            audit['nFeature_RNA_meta']=pd.to_numeric(meta.loc[selected,'nFeature_RNA'],errors='coerce').to_numpy()
            audits.append(audit)
            blocks.append(mat);names_blocks.extend(selected)
    if seen!=expected:
        raise ValueError(f'Missing epithelial cells after parsing: {len(expected-seen)}')
    X=sparse.vstack(blocks,format='csr')
    if X.shape!=(len(meta),len(gene_ref)):
        raise ValueError(f'Unexpected final matrix shape {X.shape}')
    obs=meta.loc[names_blocks].copy()
    obs['condition']=obs['sample_origin'].astype(str)
    obs['lineage']='Epithelial'
    obs['blind_sample_id']=obs['sample_id'].astype(str)
    obs['patient_id']=obs['case_id'].astype(str)
    obs['epithelial_subtype']=obs['cell_type_epi_simple'].astype(str)
    var=pd.DataFrame(index=pd.Index(gene_ref,name='gene'))
    a=ad.AnnData(X=X,obs=obs,var=var)
    a.uns['provenance']={'accession':'GSE166555','raw_archive':'GSE166555_RAW.tar',
                        'metadata':'GSE166555_meta_data.tsv.gz',
                        'selection':'main_cell_type == Epithelial',
                        'matrix_orientation':'cells x genes','expression':'raw UMI counts',
                        'condition':'sample_origin','patient':'case_id',
                        'note':'Tumor denotes tissue origin, not independent CNV-validated malignancy'}
    audit=pd.concat(audits,ignore_index=True)
    audit_path=out.parent/'GSE166555_RECONSTRUCTION_AUDIT.csv.gz'
    audit.to_csv(audit_path,index=False,compression='gzip')
    summary={
        'n_cells':int(a.n_obs),'n_genes':int(a.n_vars),'nnz':int(a.X.nnz),
        'conditions':a.obs['condition'].value_counts().to_dict(),
        'patients_by_condition':a.obs.groupby('condition')['patient_id'].nunique().to_dict(),
        'umi_median_reconstructed':audit.groupby(audit['cell_id'].map(obs['condition']))['umi_reconstructed'].median().to_dict(),
        'cell_count_discrepancy_fraction':float(np.mean(audit['umi_reconstructed'].to_numpy()!=audit['nCount_RNA_meta'].to_numpy())),
        'gene_reference_length':len(gene_ref),
    }
    (out.parent/'GSE166555_BUILD_SUMMARY.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print('Writing',out,flush=True)
    a.write_h5ad(out,compression='gzip')
    print('DONE',json.dumps(summary,indent=2,ensure_ascii=False),flush=True)
    print('Audit:',audit_path,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',default='16_FIG4_FIG5_APPLICATIONS/GSE166555/data')
    ap.add_argument('--output',default=None)
    ap.add_argument('--chunk-genes',type=int,default=128)
    run(ap.parse_args())
