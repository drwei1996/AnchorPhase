"""Task11B ablations that reuse the cached fold params/CPI (no
refits): A4 (lambda=0), A5 (beta=0), A6 (alpha=1), C2 perturbation
stability (1e-4 on gene params). Blind."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task11b_cpi import T11B, THETA_H, nb_loglik_grid, \
    joint_posterior, log, sha
from task11b_run_smc import load_smc, greedy_panel, \
    make_resid_corr_fn, fisher_info, K_PANEL

AB = T11B / "09_ABLATIONS"


def cached():
    from task11b_utils import load_fold_cache
    loaded, reason = load_fold_cache(
        T11B / "03_CPI_METHOD/SMC_FOLD_CACHE_v1.npz")
    assert loaded is not None, f"cache load failed: {reason}"
    return (loaded["params"], loaded["cpi"], loaded["per_fold"],
            None, loaded["fold_ids"])


def main():
    X, genes, obs, sf, q0, theta_hat = load_smc()
    log_s = np.log(sf)
    folds = obs.blind_sample_id.astype("category").cat.codes
    params, cpi_all, per_fold, _, ks = cached()
    n_by_k = {k: float((folds.values == k).sum()) for k in ks}

    def run_variant(tag, lam, beta, alpha_override=None,
                    perturb=None):
        outdir = AB / tag
        outdir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(20261007)
        frames = []
        from task11b_utils import assemble_predictions
        for k in ks:
            te_mask = folds.values == k
            te = np.where(te_mask)[0]
            other = [kk for kk in ks if kk != k]
            n_o = np.array([n_by_k[kk] for kk in other])
            cpi_lfo = (np.stack([per_fold[kk] for kk in other])
                       * n_o[:, None]).sum(0) / n_o.sum()
            mF = np.stack([params[kk][0] for kk in other]).mean(0)
            aF = np.stack([params[kk][1] for kk in other]).mean(0)
            bF = np.stack([params[kk][2] for kk in other]).mean(0)
            alF = np.stack([params[kk][3] for kk in other]).mean(0)
            _, fbins = fisher_info(mF, aF, bF, alF)
            redfn = make_resid_corr_fn(X, log_s, theta_hat,
                                       params[k])
            sel = greedy_panel(cpi_lfo, redfn, fbins, K_PANEL,
                               lam=lam, beta=beta)
            m, a, b, al = params[k]
            if perturb is not None:
                m = m + rng.normal(0, perturb, m.shape)
                a = a + rng.normal(0, perturb, a.shape)
                b = b + rng.normal(0, perturb, b.shape)
            Xte = X[te_mask][:, sel].toarray().astype(np.float64)
            LL = nb_loglik_grid(Xte, log_s[te], m[sel], a[sel],
                                b[sel], al[sel]).sum(1)
            alpha = alpha_override if alpha_override is not None \
                else json.loads((T11B / "05_C2_SAFE_CPI/"
                                 "MODEL_CONFIG.json")
                                .read_text())["alpha"]
            q, mh, R, H, ci = joint_posterior(q0[te_mask], LL,
                                              alpha=alpha)
            frames.append(assemble_predictions(
                te, q, mh, R, H, ci, obs.cell_id.values,
                obs.blind_sample_id.values, theta_grid_h=THETA_H))
            log(f"  {tag} fold {k} done (alpha={alpha})")
        df = pd.concat(frames, ignore_index=True)
        p = outdir / "PREDICTIONS_BLINDED.csv.gz"
        df.to_csv(p, index=False, compression="gzip")
        (outdir / "PREDICTION_SHA256.txt").write_text(sha(p) + "\n")
        (outdir / "MODEL_CONFIG.json").write_text(json.dumps(
            {"variant": tag, "lambda": lam, "beta": beta,
             "alpha_override": alpha_override,
             "perturb": perturb, "TRUE_ZT_READ": False}, indent=2))

    run_variant("A4_C2_no_redundancy", 0.0, 0.5)
    run_variant("A5_C2_no_coverage", 0.5, 0.0)
    run_variant("A6_C2_alpha1", 0.5, 0.5, alpha_override=1.0)
    run_variant("C2_PERTURB1e-4", 0.5, 0.5, perturb=1e-4)
    log("ablations A4/A5/A6 + perturbation frozen")


if __name__ == "__main__":
    main()
