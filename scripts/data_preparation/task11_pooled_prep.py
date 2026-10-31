"""Task 11 real-data prep: pooled heterogeneous GSE206583 aorta.

Builds from GEO supplementary mtx (fibroblast/endothelial/macrophage
downloaded to data/external/GSE206583/raw_task11/) plus the FROZEN 02B
SMC set, using ORIGINAL annotations only (cell_type from file, sample
= cell_type x zt file; the SMC side keeps the frozen cell order and
blind ids).

Outputs (under T11/11_REALDATA/00_POOLED_FREEZE/):
  POOLED_COUNTS.npz            csr (N,17) anchored genes, frozen order
  POOLED_SIZE_FACTOR.npy       whole-transcriptome median-1
  POOLED_OBS_BLINDED.csv       cell_id, lineage, blind_sample_id
  POOLED_BLINDED_MAP_PRIVATE.csv  blind_sample_id -> hidden_ZT
  POOLED_GENES.csv             17 genes, order = frozen anchored order
  POOLED_PCA_NONCLOCK.npz      z (N,8) PCA of non-clock genes
  POOLED_FREEZE.json           hashes + stats
No ZT used in any fitting; map is private-side only.
"""
import csv
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/GSE206583/raw_task11"
RR = ROOT / "results/single_cell_phase_rebuild"
SMC = RR / "02B_BASELINE_REPRODUCTION_STRICT"
T11 = RR / "11_KAT_V2_MODEL_FAMILY_DEVELOPMENT_20261007"
OUT = T11 / "11_REALDATA/00_POOLED_FREEZE"

CELL_TYPES = ["fibroblast", "endothelial", "macrophage"]
ZTS = ["zt0", "zt6", "zt12", "zt18"]
GENES17 = ["Arntl", "Bhlhe41", "Clock", "Cry1", "Cry2", "Dbp",
           "Fbxo21", "Hlf", "Nfil3", "Npas2", "Nr1d1", "Nr1d2",
           "Per1", "Per2", "Per3", "Rorc", "Tef"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_geo(ct, zt):
    """Return (X csr cells x genes, gene_symbols, barcodes)."""
    pfx = str(RAW / f"GSE206583_{ct}_{zt}")
    import scipy.io
    M = scipy.io.mmread(pfx + "_matrix.mtx.gz").tocsr()
    with gzip.open(pfx + "_features.tsv.gz", "rt") as f:
        genes = [ln.split("\t")[1] for ln in f.read().splitlines()]
    with gzip.open(pfx + "_barcodes.tsv.gz", "rt") as f:
        bcs = f.read().splitlines()
    # GEO mtx convention here: rows = genes, cols = cells
    if M.shape[0] == len(genes) and M.shape[1] == len(bcs):
        M = M.T.tocsr()
    elif not (M.shape[0] == len(bcs) and M.shape[1] == len(genes)):
        raise RuntimeError(f"shape mismatch {ct} {zt}: {M.shape} vs "
                           f"{len(bcs)}x{len(genes)}")
    return M.astype(np.float64), np.array(genes), np.array(bcs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # ---- frozen SMC side ----
    smc_X = sp.load_npz(SMC / "SMC_counts.npz").tocsr()
    smc_var = pd.read_csv(SMC / "SMC_var.csv")
    smc_names = list(smc_var.gene_id.values)
    smc_obs = pd.read_csv(SMC / "01_SMC_CELL_SELECTION_AUDIT.csv",
                          usecols=["cell_id", "blind_sample_id"])
    smc_ids = list(smc_obs.cell_id)
    assert smc_X.shape[0] == len(smc_ids)

    Xs, gene_lists, metas = [], [], []
    # SMC first (frozen order, lineage smc)
    Xs.append(smc_X)
    gene_lists.append(smc_names)
    metas.append(pd.DataFrame({
        "cell_id": smc_ids, "lineage": "smc",
        "blind_sample_id": list(smc_obs.blind_sample_id),
        "source": "frozen_02B"}))
    next_sample = 100
    priv_rows = []
    smc_priv = pd.read_csv(SMC / "03_BLINDED_SAMPLE_ID_MAP_PRIVATE.csv")
    for _, r in smc_priv.iterrows():
        priv_rows.append((r.blind_sample_id, float(r.hidden_ZT),
                          "smc", len(smc_ids)))

    common = set(smc_names)
    for ct in CELL_TYPES:
        for zt in ZTS:
            X, genes, bcs = load_geo(ct, zt)
            common &= set(genes.tolist())
            zt_h = float(zt.replace("zt", ""))
            bid = f"POOLED_SAMPLE_{next_sample:04d}"
            next_sample += 1
            Xs.append(X)
            gene_lists.append(genes)
            metas.append(pd.DataFrame({
                "cell_id": [f"{ct}_{zt}_{b}" for b in bcs],
                "lineage": ct, "blind_sample_id": bid,
                "source": f"geo_{ct}_{zt}"}))
            priv_rows.append((bid, zt_h, ct, X.shape[0]))
            log(f"loaded {ct} {zt}: {X.shape[0]} cells "
                f"{X.shape[1]} genes")
    common = sorted(common)
    for g in GENES17:
        assert g in common, g
    log(f"common gene space: {len(common)}; anchored 17 present")

    def align(X, names):
        pos = {g: i for i, g in enumerate(names)}
        idx = [pos[g] for g in common]
        return X[:, idx]

    aligned = [align(X, names) for X, names in zip(Xs, gene_lists)]
    Xp = sp.vstack(aligned).tocsr()
    obs = pd.concat(metas, ignore_index=True)
    log(f"pooled matrix: {Xp.shape}, lineages: "
        f"{obs.lineage.value_counts().to_dict()}")

    # anchored 17 in frozen order + size factors
    cpos = {g: i for i, g in enumerate(common)}
    idx17 = [cpos[g] for g in GENES17]
    X17 = Xp[:, idx17].tocsr()
    tot = np.asarray(Xp.sum(axis=1)).ravel().astype(np.float64)
    if (tot <= 0).any():
        raise RuntimeError(f"STOP_INVALID cells total UMI<=0: "
                           f"{int((tot <= 0).sum())}")
    sf = tot / float(np.median(tot))

    # SMC-block cross-check vs frozen 10A artifacts (first 18863 rows)
    sf10a = np.load(RR / ("10A_KAT_V2_B0_LIBRARY_OFFSET_CORRECTION_"
                          "AND_FAIRNESS_20261007/"
                          "02_CORRECTED_SIZE_FACTOR_FREEZE/"
                          "SIZE_FACTOR_TOTAL_UMI_MEDIAN1.npy"))
    med_smc = float(np.median(tot[:18863]))
    sf_smc = tot[:18863] / med_smc
    rel = np.abs(sf_smc - sf10a) / sf10a
    log(f"SMC size-factor cross-check vs frozen 10A: max rel diff "
        f"{rel.max():.2e} (median-normalized over SMC block)")

    sp.save_npz(OUT / "POOLED_COUNTS.npz", X17)
    np.save(OUT / "POOLED_SIZE_FACTOR.npy", sf)
    np.save(OUT / "POOLED_TOTAL_UMI.npy", tot)
    obs.to_csv(OUT / "POOLED_OBS_BLINDED.csv", index=False)
    pd.DataFrame({"gene": GENES17}).to_csv(
        OUT / "POOLED_GENES.csv", index=False)
    with open(OUT / "POOLED_BLINDED_MAP_PRIVATE.csv", "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["blind_sample_id", "hidden_ZT", "lineage",
                    "n_cells"])
        w.writerows(priv_rows)

    # ---- non-clock PCA state (B2) ----
    clock_pos = {g for g in GENES17}
    nonclock = [g for g in common if g not in clock_pos]
    nc_idx = [cpos[g] for g in nonclock]
    Xnc = Xp[:, nc_idx].astype(np.float64)
    log_norm = Xnc.multiply(1.0 / np.maximum(tot[:, None], 1.0)) \
        .multiply(1e4).tocsr()
    Xln = log_norm.tocsr()
    Xln.data = np.log1p(Xln.data)
    from sklearn.decomposition import PCA
    Xd = Xln.astype(np.float32).toarray()
    pca = PCA(n_components=8, random_state=20261007,
              svd_solver="randomized")
    z = pca.fit_transform(Xd)
    np.savez_compressed(OUT / "POOLED_PCA_NONCLOCK.npz", z=z,
                        explained=pca.explained_variance_ratio_)
    log(f"non-clock PCA: {Xln.shape[1]} genes -> 8 PCs, evr "
        f"{pca.explained_variance_ratio_[:3].round(3).tolist()}")

    freeze = {
        "n_cells": int(Xp.shape[0]),
        "n_common_genes": len(common),
        "lineage_counts": obs.lineage.value_counts().to_dict(),
        "n_samples": len(priv_rows),
        "pooled_counts17_sha256": sha(OUT / "POOLED_COUNTS.npz"),
        "size_factor_sha256": sha(OUT / "POOLED_SIZE_FACTOR.npy"),
        "obs_sha256": sha(OUT / "POOLED_OBS_BLINDED.csv"),
        "pca_sha256": sha(OUT / "POOLED_PCA_NONCLOCK.npz"),
        "smc_block_crosscheck_max_rel": float(rel.max()),
        "no_true_ZT_used_for_features": True,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    (OUT / "POOLED_FREEZE.json").write_text(json.dumps(freeze,
                                                       indent=2))
    log("pooled freeze complete")


if __name__ == "__main__":
    main()
