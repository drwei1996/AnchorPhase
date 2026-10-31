"""10A stage 09: TEMPO_SAME17 — official Tempo restricted to the SAME
18,863 SMC cells and the SAME 17 anchored genes, official priors, no ZT,
identical config to the frozen 02D_FIX2 Tempo run (grid=12, 3/3, vi
3000, disp bin 50, cell batch 256, seed 20261004). Blind: never reads
the private ZT map. Writes only under 10A/09_COMPARATOR_RECOVERY/
TEMPO_SAME17/."""
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
SMC_DIR = RR / "02B_BASELINE_REPRODUCTION_STRICT"
A = RR / ("10A_KAT_V2_B0_LIBRARY_OFFSET_CORRECTION_"
          "AND_FAIRNESS_20261007")
OUT = A / "09_COMPARATOR_RECOVERY/TEMPO_SAME17"

SEED = 20261004
CELL_BATCH_SIZE = 256
NUM_PHASE_GRID_POINTS = 12
NUM_PHASE_EST_CELL_SAMPLES = 3
NUM_PHASE_EST_GENE_SAMPLES = 3
VI_BATCH_SIZE = 3000
MEAN_DISP_MAX_NUM_GENES_PER_BIN = 50

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


def install_memory_patch():
    import torch
    from tempo import clock_gene_posterior
    orig = (clock_gene_posterior.ClockGenePosterior
            .compute_cell_phase_posterior_likelihood)

    def batched(self, gene_X, log_L, prior_theta_euclid_dist,
                num_gene_samples=100, **kw):
        n = gene_X.shape[0]
        if n <= CELL_BATCH_SIZE:
            return orig(self, gene_X, log_L,
                        prior_theta_euclid_dist,
                        num_gene_samples=num_gene_samples, **kw)
        return torch.cat([
            orig(self, gene_X[s:s + CELL_BATCH_SIZE],
                 log_L[s:s + CELL_BATCH_SIZE],
                 prior_theta_euclid_dist,
                 num_gene_samples=num_gene_samples, **kw)
            for s in range(0, n, CELL_BATCH_SIZE)], dim=0)

    (clock_gene_posterior.ClockGenePosterior
     .compute_cell_phase_posterior_likelihood) = batched
    log("memory patch installed")


def main():
    if (OUT / "PREDICTION_FREEZE.json").exists():
        raise RuntimeError("STOP_OUTPUT_COLLISION")
    OUT.mkdir(parents=True, exist_ok=True)
    import scipy.sparse as sp
    import anndata as ad
    X = sp.load_npz(SMC_DIR / "SMC_counts.npz").tocsr()
    var = pd.read_csv(SMC_DIR / "SMC_var.csv")
    obs = pd.read_csv(SMC_DIR / "01_SMC_CELL_SELECTION_AUDIT.csv",
                      usecols=["cell_id"])
    names = list(var.gene_id.values)
    idx = [names.index(g) for g in GENES17]
    X17 = X[:, idx].astype(np.float32)
    log(f"SAME17 matrix: {X17.shape} from {X.shape}")
    adata = ad.AnnData(X=X17,
                       obs=pd.DataFrame(index=obs.cell_id.values),
                       var=pd.DataFrame(index=GENES17))
    prior_out = OUT / "clock_aorta_acrophase_prior.csv"
    core_out = OUT / "core_clock_genes_17.txt"
    shutil.copy2(TP / "data/clock_aorta_acrophase_prior.csv", prior_out)
    core_out.write_text("\n".join(GENES17) + "\n")
    sys.path.insert(0, str(TP))
    import torch
    install_memory_patch()
    from tempo import unsupervised_alg
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    folder_out = OUT / "tempo_same17_out"
    folder_out.mkdir()
    t0 = time.perf_counter()
    try:
        unsupervised_alg.run(
            adata=adata, folder_out=str(folder_out),
            gene_acrophase_prior_path=str(prior_out),
            core_clock_gene_path=str(core_out),
            reference_gene="Arntl", min_gene_prop=1e-5,
            num_phase_grid_points=NUM_PHASE_GRID_POINTS,
            vi_batch_size=VI_BATCH_SIZE,
            num_phase_est_cell_samples=NUM_PHASE_EST_CELL_SAMPLES,
            num_phase_est_gene_samples=NUM_PHASE_EST_GENE_SAMPLES,
            mean_disp_max_num_genes_per_bin=MEAN_DISP_MAX_NUM_GENES_PER_BIN,
        )
    except Exception as e:
        import traceback
        (OUT / "TEMPO_ERROR.json").write_text(json.dumps(
            {"elapsed_sec": round(time.perf_counter() - t0),
             "exception": str(e),
             "traceback": traceback.format_exc()}, indent=2))
        raise
    runtime = time.perf_counter() - t0
    log(f"Tempo SAME17 completed in {runtime:.0f}s")
    cp = folder_out / "tempo_results" / "0" / \
        "cell_phase_estimation" / "cell_posterior.tsv"
    if not cp.exists():
        matches = sorted(folder_out.glob(
            "*/cell_phase_estimation/cell_posterior.tsv"))
        assert len(matches) == 1, matches
        cp = matches[0]
    cell_post = pd.read_csv(cp, sep="\t", index_col="barcode")
    grid = np.linspace(0, 2 * np.pi, cell_post.shape[1])
    probs = cell_post.values
    total = probs.sum(axis=1)
    map_h = np.mod(grid[probs.argmax(axis=1)] * 24 / (2 * np.pi), 24)
    mean_r = np.mod((probs * grid[None, :]).sum(axis=1)
                    / np.maximum(total, 1e-12), 2 * np.pi)
    mean_h = np.mod(mean_r * 24 / (2 * np.pi), 24)
    var = (probs * (np.mod(grid[None, :] - mean_r[:, None] + np.pi,
                           2 * np.pi) - np.pi) ** 2).sum(axis=1) \
        / np.maximum(total, 1e-12)
    cstd = np.sqrt(np.maximum(var, 0)) * 24 / (2 * np.pi)
    pred = pd.DataFrame({"cell_id": cell_post.index.values,
                         "pred_phase_MAP_h": map_h,
                         "pred_phase_circular_mean_h": mean_h,
                         "posterior_cSTD_h": cstd})
    pred_path = OUT / "tempo_same17_predictions_blinded.csv.gz"
    pred.to_csv(pred_path, index=False, compression="gzip")
    (OUT / "PREDICTION_FREEZE.json").write_text(json.dumps(
        {"method": "TEMPO_SAME17", "genes": GENES17,
         "n_cells": len(pred),
         "prediction_sha256": sha(pred_path),
         "cell_posterior_sha256": sha(cp),
         "prior_sha256": sha(prior_out),
         "core17_sha256": sha(core_out),
         "config": "identical to frozen 02D_FIX2 (grid=12, 3/3, "
                   "vi=3000, disp_bin=50, cell_batch=256)",
         "seed": SEED, "runtime_sec": round(runtime),
         "TRUE_ZT_READ": False,
         "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    log(f"TEMPO_SAME17 frozen: {len(pred)} cells, "
        f"{sha(pred_path)[:16]}...")


if __name__ == "__main__":
    main()
