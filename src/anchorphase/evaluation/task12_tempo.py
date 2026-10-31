"""Task12 Tempo comparator on external cohorts: official tempo-main
implementation, official aorta priors (same frozen asset as all
development comparisons), config identical to frozen 02D_FIX2
(grid=12, 3/3, vi=3000, disp bin 50, cell batch 256, seed 20261004).
Unsupervised - never reads labels. Blind predictions + freeze."""
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TP = ROOT / "external_repos/tempo-main"
RR = ROOT / "results/single_cell_phase_rebuild"
T12 = RR / "12_KAT_V2_EXTERNAL_VALIDATION_20261008"
H5 = {"GSE145197": ROOT / "data/processed/GSE145197/"
                     "gse145197_sparse.h5ad",
      "GSE223109": ROOT / "data/processed/GSE223109/"
                     "gse223109_sparse.h5ad"}
SEED = 20261004
CELL_BATCH = 256
GRID, EST_C, EST_G, VI, DISP = 12, 3, 3, 3000, 50


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main(gse):
    out = T12 / gse / "07_COMPARATORS"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "PREDICTION_FREEZE.json").exists():
        print("already frozen"); return
    import anndata as ad
    import scipy.sparse as sp
    A = ad.read_h5ad(H5[gse])
    X = (A.X if sp.issparse(A.X)
         else sp.csr_matrix(A.X)).tocsr().astype(np.float32)
    genes = [str(g) for g in A.var.gene_name]
    # collapse duplicated symbols by summing raw UMI (data plumbing;
    # GSE145197 has exactly one duplicated symbol)
    if len(set(genes)) != len(genes):
        df = pd.DataFrame(genes, columns=["g"])
        first = ~df.duplicated()
        uniq_idx = np.where(first)[0]
        agg = pd.DataFrame({"col": np.arange(len(genes))}, index=genes)             .groupby(level=0)["col"].apply(list)
        Xd = sp.csr_matrix(X[:, uniq_idx].astype(np.float64))
        dupmap = {g: cols for g, cols in agg.items()
                  if isinstance(cols, list) and len(cols) > 1}
        for g, cols in dupmap.items():
            dst = int(np.where(np.array(genes)[uniq_idx] == g)[0][0])
            Xd[:, dst] = X[:, cols].sum(axis=1)
        X = Xd.tocsr().astype(np.float32)
        genes = [genes[i] for i in uniq_idx]
    # map lowercase symbols to the canonical case used by the
    # frozen prior/core-clock assets (pure plumbing; required for
    # GSE145197 whose symbols are all-lowercase)
    prior_syms = [ln.split(",")[0] for ln in
                  (TP / "data/clock_aorta_acrophase_prior.csv")
                  .read_text().splitlines()[1:] if ln.strip()]
    core_syms = [ln.strip() for ln in
                 (TP / "data/core_clock_genes.txt")
                 .read_text().splitlines() if ln.strip()]
    canon = {s_.lower(): s_ for s_ in set(prior_syms + core_syms)}
    genes = [canon.get(g, g) for g in genes]
    adata = ad.AnnData(X=X, obs=pd.DataFrame(
        index=[f"cell_{i}" for i in range(A.n_obs)]),
        var=pd.DataFrame(index=genes))
    prior_out = out / "clock_aorta_acrophase_prior.csv"
    core_out = out / "core_clock_genes.txt"
    shutil.copy2(TP / "data/clock_aorta_acrophase_prior.csv",
                 prior_out)
    shutil.copy2(TP / "data/core_clock_genes.txt", core_out)
    sys.path.insert(0, str(TP))
    import torch
    torch.set_num_threads(4)
    from tempo import clock_gene_posterior
    orig = (clock_gene_posterior.ClockGenePosterior
            .compute_cell_phase_posterior_likelihood)

    def batched(self, gene_X, log_L, d, num_gene_samples=100,
                **kw):
        n = gene_X.shape[0]
        if n <= CELL_BATCH:
            return orig(self, gene_X, log_L, d,
                        num_gene_samples=num_gene_samples, **kw)
        return torch.cat([
            orig(self, gene_X[s:s + CELL_BATCH],
                 log_L[s:s + CELL_BATCH], d,
                 num_gene_samples=num_gene_samples, **kw)
            for s in range(0, n, CELL_BATCH)], dim=0)

    (clock_gene_posterior.ClockGenePosterior
     .compute_cell_phase_posterior_likelihood) = batched
    from tempo import unsupervised_alg
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    folder = out / "tempo_out"
    folder.mkdir()
    t0 = time.perf_counter()
    try:
        unsupervised_alg.run(
            adata=adata, folder_out=str(folder),
            gene_acrophase_prior_path=str(prior_out),
            core_clock_gene_path=str(core_out),
            reference_gene="Arntl", min_gene_prop=1e-5,
            num_phase_grid_points=GRID, vi_batch_size=VI,
            num_phase_est_cell_samples=EST_C,
            num_phase_est_gene_samples=EST_G,
            mean_disp_max_num_genes_per_bin=DISP)
    except Exception as e:
        import traceback
        (out / "TEMPO_ERROR.json").write_text(json.dumps(
            {"elapsed": round(time.perf_counter() - t0),
             "error": str(e),
             "traceback": traceback.format_exc()}, indent=2))
        raise
    cp = folder / "tempo_results" / "0" / \
        "cell_phase_estimation" / "cell_posterior.tsv"
    if not cp.exists():
        m = sorted(folder.glob(
            "*/cell_phase_estimation/cell_posterior.tsv"))
        assert len(m) == 1, m
        cp = m[0]
    cell_post = pd.read_csv(cp, sep="\t", index_col="barcode")
    grid = np.linspace(0, 2 * np.pi, cell_post.shape[1])
    probs = cell_post.values
    tot = probs.sum(axis=1)
    map_h = np.mod(grid[probs.argmax(axis=1)] * 24 / (2 * np.pi),
                   24)
    mr = np.mod((probs * grid[None, :]).sum(axis=1)
                / np.maximum(tot, 1e-12), 2 * np.pi)
    var = (probs * (np.mod(grid[None, :] - mr[:, None] + np.pi,
                           2 * np.pi) - np.pi) ** 2).sum(1) \
        / np.maximum(tot, 1e-12)
    cstd = np.sqrt(np.maximum(var, 0)) * 24 / (2 * np.pi)
    obs = pd.read_csv(T12 / gse / "00_DATA_AUDIT" /
                      "BLIND_MAP_PRIVATE.csv")
    # blind cell ids in pipeline order
    blind_ids = None
    # recover blind ids via original sample order
    A2 = ad.read_h5ad(H5[gse])
    sids = sorted(set(A2.obs.sample_id.astype(str)))
    bmap = {s: f"BLIND_{i:04d}" for i, s in enumerate(sids)}
    pred = pd.DataFrame({
        "cell_id": [f"{gse}_{i}" for i in range(A2.n_obs)],
        "blind_sample_id": [bmap[s] for s in
                            A2.obs.sample_id.astype(str)],
        "pred_phase_MAP_h": map_h,
        "posterior_cSTD_h": cstd})
    p = out / "tempo_predictions_blinded.csv.gz"
    pred.to_csv(p, index=False, compression="gzip")
    (out / "PREDICTION_FREEZE.json").write_text(json.dumps({
        "method": "Tempo official, frozen FIX2 config, aorta prior",
        "n_cells": len(pred), "seed": SEED,
        "prediction_sha256": sha(p),
        "runtime_sec": round(time.perf_counter() - t0),
        "TRUE_ZT_READ": False}, indent=2))
    print(f"[{gse}] Tempo frozen "
          f"({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1])
