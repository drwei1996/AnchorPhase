"""SUPP DISCOVERY 01 — Step 1: genome-wide per-patient per-condition harmonic fits (GSE166555).

Replicates the EXACT fitting convention of FIG45_V2_REBUILD.fit_patient_harmonics:
  Y = log1p(1e4 * counts / lib);  X = [1, log1p(lib), subtype dummies(drop_first), cos(2*pi*phase/24), sin(...)]
  fitted separately within each paired patient x condition block (>=100 cells both conditions).
Extended from 125 mechanism genes to ALL genes in the H5AD.

Read-only w.r.t. frozen inputs. Outputs a new table only.
"""
import json
import time

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp

import os as _os  # release patch: path de-hardcoded (see docs/port_notes.md)
ROOT = _os.environ.get('ANCHORPHASE_PROJECT_ROOT',
                        _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))
G = ROOT + r'\16_FIG4_FIG5_APPLICATIONS\GSE166555'
OUT = ROOT + r'\ANCHORPHASE_MANUSCRIPT_COMPLETION_20261028\02_NEW_BIOLOGICAL_SUPPLEMENTARY_FIGURES\SUPP_DISCOVERY_01\DATA'
import os
os.makedirs(OUT, exist_ok=True)

import anndata as ad

t0 = time.time()
adata = ad.read_h5ad(G + r'\anchorphase_project\data\count_matrix.h5ad')
X = adata.X
if not sp.issparse(X):
    X = sp.csr_matrix(X)
X = X.tocsc()
pred = pd.read_csv(G + r'\anchorphase_project\data\PREDICTIONS_BLINDED.csv.gz')
assert pred.cell_id.is_unique and adata.obs_names.is_unique
pos = pd.Index(pred.cell_id.astype(str)).get_indexer(adata.obs_names.astype(str))
assert (pos >= 0).all()
phase = pred.iloc[pos]['phase_mean_h'].to_numpy(float)
assert np.isfinite(phase).all() and ((phase >= 0) & (phase <= 24)).all()

obs = adata.obs
patient = obs['patient_id'].astype(str).to_numpy()
condition = obs['condition'].astype(str).to_numpy()
subtype = obs['epithelial_subtype'].astype(object).fillna('Unknown').astype(str).to_numpy()

lib = X.sum(axis=1).A.ravel().astype(np.float64)
assert np.all(lib > 0)

ct = pd.crosstab(patient, condition)
paired = ct.index[(ct.get('Normal', 0) >= 100) & (ct.get('Tumor', 0) >= 100)].astype(str).tolist()
print('paired patients:', len(paired), paired, flush=True)

MIN_CELLS = 100
blocks = []
for p in paired:
    for cond in ('Normal', 'Tumor'):
        ix = np.where((patient == p) & (condition == cond))[0]
        if len(ix) < MIN_CELLS:
            continue
        Z = np.column_stack([np.ones(len(ix)), np.log1p(lib[ix]),
                             pd.get_dummies(subtype[ix], drop_first=True, dtype=float).to_numpy()])
        Q, R, piv = la.qr(Z, mode='economic', pivoting=True)
        rk = int(np.sum(abs(np.diag(R)) > 1e-8 * max(1, abs(R[0, 0]))))
        Z = Z[:, sorted(piv[:rk])]
        theta = 2 * np.pi * phase[ix] / 24
        Xd = np.column_stack((Z, np.cos(theta), np.sin(theta)))
        if np.linalg.matrix_rank(Xd) < Xd.shape[1]:
            continue
        pinv = np.linalg.pinv(Xd, rcond=1e-9)
        blocks.append((p, cond, ix, pinv))
print('fit blocks:', len(blocks), flush=True)

genes = adata.var_names.astype(str).tolist()
gix_all = np.arange(len(genes))
records = np.empty((len(genes) * len(blocks),), dtype=object)  # fill as float array instead
# Use plain float arrays for speed
n_rows = len(genes) * len(blocks)
out_gene = np.empty(n_rows, dtype=object)
out_pat = np.empty(n_rows, dtype=object)
out_cond = np.empty(n_rows, dtype=object)
out_bcos = np.empty(n_rows, dtype=np.float32)
out_bsin = np.empty(n_rows, dtype=np.float32)
out_ncell = np.empty(n_rows, dtype=np.int32)
out_mean = np.empty(n_rows, dtype=np.float32)
out_det = np.empty(n_rows, dtype=np.float32)

row = 0
BATCH = 200
for start in range(0, len(genes), BATCH):
    stop = min(len(genes), start + BATCH)
    idx = gix_all[start:stop]
    sub = X[:, idx]
    arr = sub.toarray().astype(np.float64)  # n_cells x batch
    counts_pos = (arr > 0)
    Y = np.log1p(1e4 * arr / np.maximum(lib[:, None], 1))
    for (p, cond, ix, pinv) in blocks:
        coef = pinv @ Y[ix, :]
        det = counts_pos[ix, :].mean(axis=0)
        mean_y = Y[ix, :].mean(axis=0)
        nb = stop - start
        out_gene[row:row + nb] = genes[start:stop]
        out_pat[row:row + nb] = p
        out_cond[row:row + nb] = cond
        out_bcos[row:row + nb] = coef[-2, :]
        out_bsin[row:row + nb] = coef[-1, :]
        out_ncell[row:row + nb] = len(ix)
        out_mean[row:row + nb] = mean_y
        out_det[row:row + nb] = det
        row += nb
    if (start // BATCH) % 10 == 0:
        print(f'genes {stop}/{len(genes)} elapsed {time.time()-t0:.0f}s', flush=True)

fit = pd.DataFrame({
    'gene': out_gene[:row], 'patient_id': out_pat[:row], 'condition': out_cond[:row],
    'beta_cos': out_bcos[:row], 'beta_sin': out_bsin[:row], 'n_cells': out_ncell[:row],
    'mean_logcp10k': out_mean[:row], 'detect_frac': out_det[:row]})
fit.to_csv(OUT + r'\01_GENOMEWIDE_PATIENT_GENE_HARMONICS.csv.gz', index=False, compression={'method': 'gzip', 'compresslevel': 6})

manifest = {
    'dataset': 'GSE166555', 'n_cells': int(adata.n_obs), 'n_genes': int(adata.n_vars),
    'paired_patients': paired, 'n_blocks': len(blocks), 'min_cells_per_block': MIN_CELLS,
    'model': 'within patient x condition OLS: log1p(CP10K) ~ 1 + log1p(lib) + subtype dummies + cos + sin (identical to FIG45_V2_REBUILD convention)',
    'phase_source': 'frozen AnchorPhase B0 phase_mean_h (PREDICTIONS_BLINDED.csv.gz, unmodified)',
    'runtime_note': 'exploratory remodeling analysis; inferred phase, not sampling time',
    'elapsed_s': round(time.time() - t0, 1),
}
with open(OUT + r'\00_FIT_MANIFEST.json', 'w') as fh:
    json.dump(manifest, fh, indent=2)
print('DONE', manifest['elapsed_s'], 'rows:', row, flush=True)
