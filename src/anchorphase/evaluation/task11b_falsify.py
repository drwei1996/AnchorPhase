"""Task11B falsification F1 (permuted-theta) / F3 (wrong-anchor).
Re-runs the full cross-fitted CPI SCORING pipeline with permuted or
rotated (theta, q0) reference. Blind (no ZT).

Usage: task11b_falsify.py perm SEED | rot HOURS
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task11b_cpi import (SMC, T11B, KF, fit_genes_3col,
                         nb_loglik_grid, entropy_reduction, circ_mean_h,
                         log)
from task11b_run_smc import (load_smc, fold_gene_fits, score_cpi,
                             SCORE_CHUNK)

RR = T11B.parents[0]


def run(mode, arg):
    X, genes, obs, sf, q0, theta_hat = load_smc()
    log_s = np.log(sf)
    rng = np.random.default_rng(20261007 + int(arg) * 1000
                                + (0 if mode == "perm" else 500))
    if mode == "perm":
        p = rng.permutation(len(theta_hat))
        theta_use, q0_use = theta_hat[p], q0[p]
        tag = f"perm{arg}"
    else:
        h = float(arg)
        # WRONG-ANCHOR control: rotate ONLY the gene-fit target; the
        # B0 posterior q0 stays at the true anchor. A real conditional
        # information selector must degrade. (v1 rotated q0 too, which
        # rigidly remapped the problem and gave identical CPI - kept
        # as symmetry check in ROTSYM files.)
        theta_use = np.mod(theta_hat + h * 2 * np.pi / 24, 2 * np.pi)
        q0_use = q0
        tag = f"rotfix{arg}"
    folds = obs.blind_sample_id.astype("category").cat.codes
    params = fold_gene_fits(X, genes, log_s, theta_use, folds)
    cpi, per_fold, n_by = score_cpi(X, log_s, q0_use, folds, params)
    top = np.argsort(-cpi)[:40]
    out = {"mode": mode, "arg": arg,
           "mean_cpi": float(cpi.mean()),
           "max_cpi": float(cpi.max()),
           "top40_mean_cpi": float(cpi[top].mean()),
           "n_genes_cpi_gt_0.01": int((cpi > 0.01).sum())}
    d = T11B / "08_FALSIFICATION"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{tag}_CPI.json").write_text(json.dumps(out, indent=2))
    log(f"{tag}: mean={out['mean_cpi']:.5f} "
        f"top40={out['top40_mean_cpi']:.4f}")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
