"""Task11B pooled C3 pipeline: SAFE-CPI-LS on pooled heterogeneous
aorta (lineage-adaptive INFO readout, state-conditioned, shared
theta). 5-fold sample-blocked. Blind (no ZT).

Variants (preregistered ablations):
  full   = C3 SAFE-CPI-LS   (A11)
  lin    = lineage only      (A9)
  state  = state only        (A10 - for contrast; primary C3 = full)
  pooled = neither           (A8 pooled INFO head)
Usage: task11b_run_pooled.py [full|lin|state|pooled]
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
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task11b_cpi import (RR, T11, T11B, KF, THETA, THETA_H,
                         fit_genes_LS, eff_params, nb_loglik_grid,
                         entropy_reduction, joint_posterior,
                         alpha_gate, circ_mean_h, log, sha, K_PANEL)
from task11b_run_smc import (fisher_info, greedy_panel,
                             make_resid_corr_fn, SCORE_CHUNK,
                             FIT_CHUNK)

CORE17 = ["Arntl", "Bhlhe41", "Clock", "Cry1", "Cry2", "Dbp",
          "Fbxo21", "Hlf", "Nfil3", "Npas2", "Nr1d1", "Nr1d2",
          "Per1", "Per2", "Per3", "Rorc", "Tef"]
POOL = T11 / "11_REALDATA/00_POOLED_FREEZE"
OUTD = {"full": T11B / "06_C3_SAFE_CPI_LS",
        "lin": T11B / "09_ABLATIONS/A9_C3_LIN",
        "state": T11B / "09_ABLATIONS/A10_C3_STATE",
        "pooled": T11B / "09_ABLATIONS/A8_C3_POOLED"}


def load_pooled():
    X17 = sp.load_npz(POOL / "POOLED_COUNTS.npz")   # anchored 17 only
    obs = pd.read_csv(POOL / "POOLED_OBS_BLINDED.csv")
    sf = np.load(POOL / "POOLED_SIZE_FACTOR.npy")
    zp = np.load(POOL / "POOLED_PCA_NONCLOCK.npz")["z"][:, :4]
    q0 = np.load(T11B / ("01_B0_REFERENCE/"
                         "B0_POOLED_PHASE_POSTERIOR.npz"))["posterior"]
    theta_hat = np.mod(circ_mean_h(q0) * 2 * np.pi / 24, 2 * np.pi)
    # candidate gene filter on POOLED statistics: need the pooled
    # full matrix - rebuild from 02B SMC + GEO blocks like prep did
    # (cheap: reuse POOLED freeze artifacts only store 17 genes, so
    # reload raw sources)
    var = pd.read_csv(RR / ("02B_BASELINE_REPRODUCTION_STRICT/"
                            "SMC_var.csv"))
    names = np.array(var.gene_id.values)
    Xsmc = sp.load_npz(RR / ("02B_BASELINE_REPRODUCTION_STRICT/"
                             "SMC_counts.npz")).tocsr()
    blocks = [(Xsmc, names)]
    RAW = ROOT_RAW = (Path(__file__).resolve().parents[1]
                      / "data/external/GSE206583/raw_task11")
    import gzip
    import scipy.io
    for ct in ["fibroblast", "endothelial", "macrophage"]:
        for zt in ["zt0", "zt6", "zt12", "zt18"]:
            pfx = str(RAW / f"GSE206583_{ct}_{zt}")
            M0 = scipy.io.mmread(pfx + "_matrix.mtx.gz").tocsr()
            with gzip.open(pfx + "_features.tsv.gz", "rt") as f:
                g = np.array([ln.split("\t")[1]
                              for ln in f.read().splitlines()])
            with gzip.open(pfx + "_barcodes.tsv.gz", "rt") as f:
                nb = len(f.read().splitlines())
            # orient to cells x genes regardless of storage order
            M = M0.T.tocsr() if M0.shape[0] == len(g) \
                and M0.shape[1] == nb else M0.tocsr()
            assert M.shape == (nb, len(g)), M.shape
            blocks.append((M.astype(np.float64), g))
    common = set(names)
    for _, g in blocks[1:]:
        common &= set(g.tolist())
    common = sorted(common)
    pos0 = {g: i for i, g in enumerate(names)}
    aligned = [blocks[0][0][:, [pos0[g] for g in common]]]
    for M, g in blocks[1:]:
        p = {x: i for i, x in enumerate(g)}
        aligned.append(M[:, [p[x] for x in common]])
    Xp = sp.vstack(aligned).tocsr()
    cpos = {g: i for i, g in enumerate(common)}
    y_sum = np.asarray(Xp.sum(axis=0)).ravel()
    det = np.asarray((Xp > 0).sum(axis=0)).ravel()
    keep = ((y_sum / Xp.shape[0] >= 0.1)
            & (det / Xp.shape[0] >= 0.01)
            & np.array([g not in CORE17 for g in common]))
    cidx = [cpos[g] for i, g in enumerate(common) if keep[i]]
    gnames = [g for i, g in enumerate(common) if keep[i]]
    Xc = Xp[:, cidx].tocsc()
    # pooled 5-fold from manifest
    man = pd.read_csv(KF / "POOLED_5FOLD_MANIFEST.csv")
    assert list(man.cell_id) == list(obs.cell_id)
    lin = pd.factorize(obs.lineage)[0]
    return Xc, np.array(gnames), obs, sf, q0, theta_hat, zp, lin, \
        man.fold.values


def fit_folds_LS(X, log_s, theta, lin, z, folds, use_lin, use_state):
    G = X.shape[1]
    params = {}
    for k in sorted(set(folds)):
        tr = folds != k
        beta_all, alpha_all, names_all = [], [], None
        for s0 in range(0, G, 256):
            s1 = min(s0 + 256, G)
            beta, alpha, names = fit_genes_LS(
                X[tr][:, s0:s1].toarray().astype(np.float64),
                log_s[tr], theta[tr], lin[tr], z[tr],
                use_lin=use_lin, use_state=use_state)
            beta_all.append(beta); alpha_all.append(alpha)
            names_all = names
        params[k] = (np.vstack(beta_all),
                     np.concatenate(alpha_all), names_all)
        log(f"  fold {k}: fitted ({tr.sum()} cells)")
    return params


def eff_arrays(beta, names, lin, z, n):
    """Per-cell (m,a,b) for all genes: returns arrays (G, n)?"""
    G = beta.shape[0]
    m = np.tile(beta[:, 0][:, None], (1, n))
    a = np.tile(beta[:, 1][:, None], (1, n))
    b = np.tile(beta[:, 2][:, None], (1, n))
    nm = dict(enumerate(names))
    for j in range(3, len(names)):
        nmj = names[j]
        if nmj.startswith("dM"):
            l = int(nmj[2:])
            m[:, lin == l] += beta[:, j][:, None]
        elif nmj.startswith("dAc"):
            l = int(nmj[3:])
            a[:, lin == l] += beta[:, j][:, None]
        elif nmj.startswith("dAs"):
            l = int(nmj[3:])
            b[:, lin == l] += beta[:, j][:, None]
        elif nmj.startswith("z"):
            d = int(nmj[1:])
            m += beta[:, j][:, None] * z[:, d][None, :]
    return m, a, b   # (G, n) each


def make_resid_corr_pooled(X, log_s, theta, lin, z, params_fold):
    """NB residuals under the fold's LS model (per-cell effective
    params), standardized float32; returns corr fn for greedy."""
    beta, alpha, names = params_fold
    G = beta.shape[0]
    n = X.shape[0]
    R = np.empty((n, G), dtype=np.float32)
    for s0 in range(0, G, 256):
        s1 = min(s0 + 256, G)
        m, a, b = eff_arrays(beta[s0:s1], names, lin, z, n)
        eta = log_s[:, None] + m.T + np.cos(theta)[:, None] * a.T \
            + np.sin(theta)[:, None] * b.T
        mu = np.exp(np.clip(eta, -20, 20)).astype(np.float32)
        R[:, s0:s1] = X[:, s0:s1].toarray().astype(np.float32) - mu
    R /= np.maximum(R.std(axis=0), 1e-6).astype(np.float32)

    def fn(sel, cands):
        if not sel:
            return np.zeros((0, len(cands)))
        C = R[:, cands].T @ R[:, sel]
        return np.asarray(C.T) / n
    return fn


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "full"
    use_lin = variant in ("full", "lin")
    use_state = variant in ("full", "state")
    t0 = time.perf_counter()
    X, genes, obs, sf, q0, theta_hat, zp, lin, folds = load_pooled()
    log_s = np.log(sf)
    ks = sorted(set(folds))
    n_by = np.array([(folds == k).sum() for k in ks], float)
    log(f"[{variant}] pooled data {X.shape}, folds "
        f"{np.bincount(folds).tolist()}")
    from task11b_utils import (load_fold_cache, save_fold_cache,
                               file_sha)
    code_hash = file_sha(Path(__file__).resolve())
    input_hash = file_sha(POOL / "POOLED_COUNTS.npz")
    cache = T11B / "03_CPI_METHOD" / \
        f"POOL_FOLD_CACHE_v1_{variant}.npz"

    # ---------- incremental stage cache ----------
    state = {"params": {}, "cpi": None, "per_fold": None,
             "panels": None, "LL": {}, "alpha": None,
             "names": None}
    loaded, reason = (load_fold_cache(cache) if cache.exists()
                      else (None, "absent"))
    if loaded is not None:
        assert list(loaded["gene_names"]) == list(genes)
        state["params"] = {k: (loaded["params"][k][0],
                               loaded["params"][k][1],
                               [str(x) for x in
                                loaded["extra"]["names"]])
                           for k in loaded["fold_ids"]}
        state["cpi"] = loaded["cpi"]
        state["per_fold"] = loaded["per_fold"]
        state["names"] = list(loaded["extra"]["names"])
        ex = loaded["extra"]
        if "panels" in ex:
            state["panels"] = [list(pnl) for pnl in ex["panels"]]
        if "alpha" in ex:
            state["alpha"] = float(ex["alpha"])
        for k in loaded["fold_ids"]:
            if f"LL{k}" in ex:
                state["LL"][k] = ex[f"LL{k}"]
        log(f"pooled cache loaded: {reason}")

    def save_state():
        extra = {"names": np.array(
            state["names"] if state["names"] else
            params_first_names(), dtype=object)}
        if state["panels"] is not None:
            extra["panels"] = np.array(state["panels"],
                                       dtype=object)
        if state["alpha"] is not None:
            extra["alpha"] = np.array(state["alpha"])
        for k, v in state["LL"].items():
            extra[f"LL{k}"] = v
        save_fold_cache(
            cache, fold_ids=sorted(state["params"]),
            train_sample_ids=[sorted(obs.blind_sample_id[
                folds != k].unique()) for k in ks],
            test_sample_ids=[sorted(obs.blind_sample_id[
                folds == k].unique()) for k in ks],
            gene_names=list(genes),
            cell_count_train=[int((folds != k).sum()) for k in ks],
            cell_count_test=[int((folds == k).sum()) for k in ks],
            params_by_fold={k: (state["params"][k][0],
                                state["params"][k][1])
                            for k in state["params"]},
            cpi=(state["cpi"] if state["cpi"] is not None
                 else np.zeros(len(genes))),
            per_fold=(state["per_fold"] if state["per_fold"]
                      is not None else np.zeros((len(ks),
                                                 len(genes)))),
            code_hash=code_hash, input_hash=input_hash,
            extra=extra)

    def params_first_names():
        return ["m", "a", "b"]

    # ---------- stage 1: fold fits (cache after EACH fold) ----------
    for k in ks:
        if k in state["params"]:
            continue
        tr = folds != k
        betas, alphas, names_ref = [], [], None
        G = X.shape[1]
        for s0 in range(0, G, 256):
            s1 = min(s0 + 256, G)
            beta, alpha, names = fit_genes_LS(
                X[tr][:, s0:s1].toarray().astype(np.float64),
                log_s[tr], theta_hat[tr], lin[tr], zp[tr],
                use_lin=use_lin, use_state=use_state)
            betas.append(beta); alphas.append(alpha)
            names_ref = names
        state["params"][k] = (np.vstack(betas),
                              np.concatenate(alphas), names_ref)
        state["names"] = names_ref
        save_state()
        log(f"  fold {k}: fitted ({tr.sum()} cells) [cached]")

    # ---------- stage 2: CPI scoring ----------
    if state["cpi"] is None or not np.any(state["cpi"]):
        G = X.shape[1]
        fold_mean = {}
        for k in ks:
            te = folds == k
            beta, alpha, names = state["params"][k]
            n_te = int(te.sum())
            igs = np.empty(G)
            for s0 in range(0, G, 32):
                s1 = min(s0 + 32, G)
                Xte = X[te][:, s0:s1].toarray().astype(np.float64)
                m, a, b = eff_arrays(beta[s0:s1], names, lin[te],
                                     zp[te], n_te)
                LL = nb_loglik_grid(Xte, log_s[te], m.T, a.T, b.T,
                                    alpha[s0:s1])
                igs[s0:s1] = entropy_reduction(q0[te], LL)
            fold_mean[k] = igs
        per_fold = np.stack([fold_mean[k] for k in ks])
        state["per_fold"] = per_fold
        state["cpi"] = (per_fold * n_by[:, None]).sum(0) / n_by.sum()
        save_state()
        log("  CPI scored [cached]")
    per_fold, cpi = state["per_fold"], state["cpi"]

    # ---------- stage 3: panels + joint LL (cache after EACH) ------
    if state["panels"] is None:
        panels = []
        for k in ks:
            other = [kk for kk in ks if kk != k]
            n_o = np.array([n_by[kk] for kk in other])
            cpi_lfo = (np.stack([per_fold[kk] for kk in other])
                       * n_o[:, None]).sum(0) / n_o.sum()
            beta, alpha, names = state["params"][k]
            _, fish_bins = fisher_info(beta[:, 0], beta[:, 1],
                                       beta[:, 2], alpha)
            redfn = make_resid_corr_pooled(X, log_s, theta_hat,
                                           lin, zp,
                                           state["params"][k])
            sel = greedy_panel(cpi_lfo, redfn, fish_bins, K_PANEL)
            panels.append([str(genes[i]) for i in sel])
            te = folds == k
            beta_s, alpha_s = beta[sel], alpha[sel]
            n_te = int(te.sum())
            LL = np.zeros((n_te, 48))
            m, a, b = eff_arrays(beta_s, names, lin[te], zp[te],
                                 n_te)
            for gi in range(len(sel)):
                LL += nb_loglik_grid(
                    X[te][:, sel[gi:gi + 1]].toarray()
                    .astype(np.float64), log_s[te],
                    m[gi][:, None], a[gi][:, None],
                    b[gi][:, None], alpha_s[gi:gi + 1])[:, 0, :]
            state["LL"][k] = LL
            state["panels"] = panels
            save_state()
            log(f"  fold {k}: panel + joint LL done [cached]")
    panels = state["panels"]
    LL_store = state["LL"]

    # ---------- stage 4: alpha gate + predictions ----------
    q0f = [q0[folds == k] for k in ks]
    alpha, prof = alpha_gate(q0f, [LL_store[k] for k in ks])
    state["alpha"] = alpha
    save_state()
    log(f"[{variant}] alpha = {alpha} [cached]")
    outdir = OUTD[variant]
    outdir.mkdir(parents=True, exist_ok=True)
    from task11b_utils import assemble_predictions
    ids, bs, lins = obs.cell_id.values, obs.blind_sample_id.values, \
        obs.lineage.values
    frames = []
    for k in ks:
        te = np.where(folds == k)[0]
        q, mh, R, H, ci = joint_posterior(q0[folds == k],
                                          LL_store[k],
                                          alpha=alpha)
        frames.append(assemble_predictions(
            te, q, mh, R, H, ci, ids, bs, lineage_global=lins,
            theta_grid_h=THETA_H))
    df = pd.concat(frames, ignore_index=True)
    p = outdir / "PREDICTIONS_BLINDED.csv.gz"
    df.to_csv(p, index=False, compression="gzip")
    (outdir / "PREDICTION_SHA256.txt").write_text(sha(p) + "\n")
    np.save(outdir / "CPI_scores.npy", cpi)
    (outdir / "MODEL_CONFIG.json").write_text(json.dumps({
        "model": f"C3_{variant}", "alpha": alpha, "K": K_PANEL,
        "crossfit": "pooled 5-fold sample-blocked (anchored cache)",
        "panels": panels, "alpha_profile": prof,
        "TRUE_ZT_READ": False}, indent=2, default=float))
    log(f"[{variant}] frozen ({len(df)} cells, "
        f"{(time.perf_counter()-t0)/60:.0f} min)")


if __name__ == "__main__":
    main()
