"""Task11B SMC pipeline: cross-fitted CPI scoring + C1 (top-K CPI)
+ C2 (SAFE panel + alpha gate). Blind: no ZT anywhere.

Outputs under 11B/: 03_CPI_METHOD, 04_C1_CPI, 05_C2_SAFE_CPI,
07_SAMPLE_BLOCK_CROSSFIT artifacts, TABLES/T2.
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
from task11b_cpi import (RR, SMC, T11, T11B, KF, THETA, THETA_H,
                         K_GRID, ALPHA_GRID, fit_genes_3col,
                         nb_loglik_grid, entropy_reduction,
                         joint_posterior, alpha_gate, entropy,
                         circ_mean_h, log, sha, K_PANEL, LAMBDA_RED,
                         BETA_COV, GAMMA_ALPHA)

A10A = RR / ("10A_KAT_V2_B0_LIBRARY_OFFSET_CORRECTION_"
             "AND_FAIRNESS_20261007")
OUT_CPI = T11B / "03_CPI_METHOD"
FIT_CHUNK, SCORE_CHUNK = 512, 64


def load_smc():
    X = sp.load_npz(SMC / "SMC_counts.npz").tocsr()
    var = pd.read_csv(SMC / "SMC_var.csv")
    names = np.array(var.gene_id.values)
    cand = pd.read_csv(KF / "CANDIDATE_GENES.csv")
    pos = {g: i for i, g in enumerate(names)}
    cidx = np.array([pos[g] for g in cand.gene])
    obs = pd.read_csv(SMC / "01_SMC_CELL_SELECTION_AUDIT.csv",
                      usecols=["cell_id", "blind_sample_id"])
    sf = np.load(A10A / ("02_CORRECTED_SIZE_FACTOR_FREEZE/"
                         "SIZE_FACTOR_TOTAL_UMI_MEDIAN1.npy"))
    posts = np.load(A10A / "05_B0_CORRECTED_RUN1/PHASE_POSTERIOR.npz")
    q0 = posts["posterior"]
    theta_hat = np.mod(circ_mean_h(q0) * 2 * np.pi / 24, 2 * np.pi)
    return X[:, cidx].tocsc(), cand.gene.values, obs, sf, q0, \
        theta_hat


def fold_gene_fits(X, genes, log_s, theta, folds, fit_chunk=FIT_CHUNK,
                   perturb=None):
    """Fit 3-col NB models per fold on train cells. Returns
    params[fold] = (m,a,b,alpha) arrays (G,)."""
    G = X.shape[1]
    params = {}
    rng = np.random.default_rng(20261007)
    for k in sorted(folds.unique()):
        tr = folds.values != k
        Xtr = X[tr].toarray().astype(np.float64)
        ls, th = log_s[tr], theta[tr]
        m = np.empty(G); a = np.empty(G); b = np.empty(G)
        al = np.empty(G)
        for s0 in range(0, G, fit_chunk):
            s1 = min(s0 + fit_chunk, G)
            mm, aa, bb, aaal = fit_genes_3col(Xtr[:, s0:s1], ls, th)
            if perturb is not None:
                mm = mm + rng.normal(0, perturb, mm.shape)
                aa = aa + rng.normal(0, perturb, aa.shape)
                bb = bb + rng.normal(0, perturb, bb.shape)
            m[s0:s1], a[s0:s1], b[s0:s1], al[s0:s1] = mm, aa, bb, aaal
        params[k] = (m, a, b, al)
        del Xtr
        log(f"  fold {k}: {tr.sum()} train cells, {G} genes fitted")
    return params


def score_cpi(X, log_s, q0, folds, params, score_chunk=SCORE_CHUNK):
    """Per-fold mean entropy reduction per gene -> cell-weighted CPI.
    Returns cpi (G,), per_fold (F,G) contributions (mean per fold)."""
    G = X.shape[1]
    fold_mean = {}
    for k in sorted(folds.unique()):
        te = folds.values == k
        if te.sum() == 0:
            fold_mean[k] = np.zeros(G)
            continue
        Xte = X[te].toarray().astype(np.float64)
        m, a, b, al = params[k]
        igs = np.empty(G)
        for s0 in range(0, G, score_chunk):
            s1 = min(s0 + score_chunk, G)
            LL = nb_loglik_grid(Xte[:, s0:s1], log_s[te],
                                m[s0:s1], a[s0:s1], b[s0:s1],
                                al[s0:s1])
            igs[s0:s1] = entropy_reduction(q0[te], LL)
        fold_mean[k] = igs
        del Xte
    ks = sorted(fold_mean)
    n_by = np.array([(folds.values == k).sum() for k in ks], float)
    per_fold = np.stack([fold_mean[k] for k in ks])
    cpi = (per_fold * n_by[:, None]).sum(0) / n_by.sum()
    return cpi, per_fold, dict(zip(ks, n_by))


def fisher_info(m, a, b, alpha):
    """Integrated + 6-bin Fisher mass at median size factor."""
    s_bar = 1.0
    eta = s_bar * np.exp(np.clip(
        m[:, None] + a[:, None] * np.cos(THETA)[None, :]
        + b[:, None] * np.sin(THETA)[None, :], -20, 20))
    dm = eta * (-a[:, None] * np.sin(THETA)[None, :]
                + b[:, None] * np.cos(THETA)[None, :])
    I = dm ** 2 / (eta + alpha[:, None] * eta ** 2)
    tot = I.sum(1)
    bins = I.reshape(-1, 6, 8).sum(2)
    return tot, bins


def greedy_panel(cpi, red_pairs_fn, fisher_bins, K,
                 lam=LAMBDA_RED, beta=BETA_COV):
    """Greedy maximize J = sumCPI/Z - lam*meanR + beta*cov.
    curve_corr matrix precomputed (G,G) is too big; use closed-form
    corr via (a,b): corr = (a_g a_h + b_g b_h)/(A_g A_h). red_pairs_fn
    supplies residual corr lazily (selected x candidates)."""
    G = len(cpi)
    Z = np.sort(cpi)[::-1][:K].sum()
    fb = fisher_bins / np.maximum(fisher_bins.sum(1, keepdims=True),
                                  1e-12)
    sel = []
    sel_mask = np.zeros(G, bool)
    H6 = lambda p: -(p * np.log(np.maximum(p, 1e-12))).sum()
    cur_cpi = 0.0
    while len(sel) < K:
        best_j, best_J = None, -np.inf
        # vectorized marginal over all candidates
        cand_idx = np.where(~sel_mask)[0]
        add = cpi[cand_idx]
        # coverage marginal: recompute H6 with each candidate (loop
        # only over shortlist to stay fast: top 400 by marginal)
        marg = add / Z
        short = cand_idx[np.argsort(-marg)[:400]]
        resid = red_pairs_fn(sel, short) if sel else np.zeros(
            (0, len(short)))
        for si, g in enumerate(short):
            S = sel + [g]
            fbS = fb[S].sum(0)
            p = fbS / fbS.sum()
            cov = 1 - H6(p) / np.log(6)
            if sel:
                rc = resid[:, si]
                n_pairs = len(sel)
                meanR = (rc.sum() + 0.0) / n_pairs
            else:
                meanR = 0.0
            J = (cur_cpi + cpi[g]) / Z - lam * meanR + beta * cov
            if np.isfinite(J) and J > best_J:
                best_J, best_j = J, g
        assert best_j is not None, "greedy: all J non-finite"
        sel.append(int(best_j))
        sel_mask[best_j] = True
        cur_cpi += cpi[best_j]
    return sel


def make_resid_corr_fn(X, log_s, theta, params_fold):
    """NB-residual correlation between selected genes and shortlist
    candidates. Residuals under the fold's fitted mu(theta_hat);
    standardized dense float32 (~0.7 GB) computed once per fold."""
    m, a, b, al = params_fold
    # log_s is ALREADY the log size factor - no further log
    eta = log_s[:, None] + m[None, :] \
        + np.cos(theta)[:, None] * a[None, :] \
        + np.sin(theta)[:, None] * b[None, :]
    mu = np.exp(np.clip(eta, -20, 20))
    R = (X.toarray().astype(np.float32) - mu.astype(np.float32))
    sd = R.std(axis=0)
    R /= np.maximum(sd, 1e-6).astype(np.float32)

    def fn(sel, cands):
        if not sel:
            return np.zeros((0, len(cands)))
        C = R[:, cands].T @ R[:, sel]        # (len(cands), len(sel))
        return np.asarray(C.T) / R.shape[0]  # (len(sel), len(cands))
    return fn


def main():
    t00 = time.perf_counter()
    X, genes, obs, sf, q0, theta_hat = load_smc()
    log_s = np.log(sf)
    folds = obs.blind_sample_id.astype("category").cat.codes
    log(f"data: {X.shape} candidates, {len(obs)} cells, 5 LOSO folds")

    from task11b_utils import (load_fold_cache, save_fold_cache,
                               file_sha)
    cache = OUT_CPI / "SMC_FOLD_CACHE_v1.npz"
    legacy = OUT_CPI / "SMC_PIPELINE_CACHE.npz"
    loaded, reason = (load_fold_cache(cache) if cache.exists()
                      else (None, "no v1 cache"))
    if loaded is None and legacy.exists():
        log(f"v1 cache absent ({reason}); migrating legacy cache "
            f"(fits are VALID_CACHE_ONLY per validity table)")
        z = np.load(legacy, allow_pickle=True)
        import re
        fold_ids = sorted({int(m.group(1)) for kk in z.files
                           if (m := re.match(r"^p(\d+)_", str(kk)))})
        params = {k: tuple(z[f"p{k}_{i}"] for i in range(4))
                  for k in fold_ids}
        cpi, per_fold = z["cpi"], z["per_fold"]
        n_by = {k: float((folds.values == k).sum())
                for k in fold_ids}
        save_fold_cache(
            cache, fold_ids=fold_ids,
            train_sample_ids=[sorted(obs.blind_sample_id[
                folds.values != k].unique()) for k in fold_ids],
            test_sample_ids=[sorted(obs.blind_sample_id[
                folds.values == k].unique()) for k in fold_ids],
            gene_names=list(genes),
            cell_count_train=[int(sum(n_by[k2] for k2 in fold_ids
                                      if k2 != k)) for k in fold_ids],
            cell_count_test=[int(n_by[k]) for k in fold_ids],
            params_by_fold=params, cpi=cpi, per_fold=per_fold,
            code_hash=file_sha(Path(__file__).resolve()),
            input_hash=file_sha(
                RR / ("02B_BASELINE_REPRODUCTION_STRICT/"
                      "SMC_counts.npz")))
        loaded, reason = load_fold_cache(cache)
    if loaded is not None:
        params = loaded["params"]
        cpi, per_fold = loaded["cpi"], loaded["per_fold"]
        n_by = {k: float(v) for k, v in zip(loaded["fold_ids"],
                                            loaded["cell_count_test"])}
        assert list(loaded["gene_names"]) == list(genes)
        log(f"anchored cache loaded: {reason}")
    else:
        params = fold_gene_fits(X, genes, log_s, theta_hat, folds)
        cpi, per_fold, n_by = score_cpi(X, log_s, q0, folds, params)
        OUT_CPI.mkdir(parents=True, exist_ok=True)
        save_fold_cache(
            cache, fold_ids=sorted(params),
            train_sample_ids=[sorted(obs.blind_sample_id[
                folds.values != k].unique()) for k in sorted(params)],
            test_sample_ids=[sorted(obs.blind_sample_id[
                folds.values == k].unique()) for k in sorted(params)],
            gene_names=list(genes),
            cell_count_train=[int(sum((folds.values != k2).sum()
                                      for k2 in sorted(params)
                                      if k2 != k))
                              for k in sorted(params)],
            cell_count_test=[int((folds.values == k).sum())
                             for k in sorted(params)],
            params_by_fold=params, cpi=cpi, per_fold=per_fold,
            code_hash=file_sha(Path(__file__).resolve()),
            input_hash=file_sha(
                RR / ("02B_BASELINE_REPRODUCTION_STRICT/"
                      "SMC_counts.npz")))
        log("anchored cache written")

    # deployment (full-data) fit for diagnostics + same-data check
    mF, aF, bF, alF = [np.stack([params[k][i] for k in
                                 sorted(params)]).mean(0)
                       for i in range(4)]
    fish_tot, fish_bins = fisher_info(mF, aF, bF, alF)
    A = np.sqrt(aF ** 2 + bF ** 2)
    phi = np.mod(np.arctan2(bF, aF), 2 * np.pi) * 24 / (2 * np.pi)
    scores = pd.DataFrame({
        "gene": genes, "CPI": cpi, "amplitude": A, "acrophase_h": phi,
        "alpha_disp": alF, "fisher_total": fish_tot,
        **{f"fisher_bin{b}": fish_bins[:, b] for b in range(6)}})
    cand = pd.read_csv(KF / "CANDIDATE_GENES.csv")
    scores = scores.merge(cand, on="gene")
    scores["dropout"] = 1 - scores.prevalence
    OUT_CPI.mkdir(parents=True, exist_ok=True)
    scores.to_csv(T11B / "TABLES/T2_CPI_GENE_SCORES.csv", index=False)

    # ---------- fold-wise leave-fold-out CPI + panels ----------
    ks = sorted(params)
    panels, preds = {}, {}
    LL_store = {}     # (model, fold) -> joint (n_test, K)
    for model in ("C1", "C2"):
        LL_store[model] = {}
    for k in ks:
        te = folds.values == k
        # leave-fold-out CPI for this fold's selection
        other = [kk for kk in ks if kk != k]
        n_by_o = np.array([n_by[kk] for kk in other])
        cpi_lfo = (np.stack([per_fold[kk] for kk in other])
                   * n_by_o[:, None]).sum(0) / n_by_o.sum()
        # C1 panel
        topK = np.argsort(-cpi_lfo)[:K_PANEL]
        m, a, b, al = params[k]
        Xte = X[te].toarray().astype(np.float64)
        LL1 = nb_loglik_grid(Xte[:, topK], log_s[te], m[topK],
                             a[topK], b[topK], al[topK]).sum(1)
        LL_store["C1"][k] = LL1
        panels.setdefault("C1", []).append(
            [str(genes[i]) for i in topK])
        # C2 panel: greedy on leave-fold-out CPI with deployment
        # redundancy/coverage diagnostics
        fish_totF, fish_binsF = fisher_info(
            *[np.stack([params[kk][i] for kk in other]).mean(0)
              for i in range(4)])
        redfn = make_resid_corr_fn(X, log_s, theta_hat, params[k])
        sel = greedy_panel(cpi_lfo, redfn, fish_binsF, K_PANEL)
        LL2 = nb_loglik_grid(Xte[:, sel], log_s[te],
                             m[sel], a[sel], b[sel], al[sel]).sum(1)
        LL_store["C2"][k] = LL2
        panels.setdefault("C2", []).append(
            [str(genes[i]) for i in sel])
        del Xte
        log(f"  fold {k}: C1/C2 panels selected")
    with open(OUT_CPI / "PANELS_PER_FOLD.json", "w") as f:
        json.dump(panels, f)

    # ---------- alpha gate (cross-fitted profile) ----------
    q0f = [q0[folds.values == k] for k in ks]
    LLf = {m_: [LL_store[m_][k] for k in ks] for m_ in LL_store}
    alpha1, prof1 = alpha_gate(q0f, LLf["C1"])
    alpha2, prof2 = alpha_gate(q0f, LLf["C2"])
    log(f"alpha gates: C1={alpha1}, C2={alpha2}")

    # ---------- predictions ----------
    from task11b_utils import assemble_predictions
    obs_ids = obs.cell_id.values
    bsids = obs.blind_sample_id.values
    def write_preds(model, alpha):
        outdir = {"C1": T11B / "04_C1_CPI",
                  "C2": T11B / "05_C2_SAFE_CPI"}[model]
        outdir.mkdir(parents=True, exist_ok=True)
        frames = []
        for k in ks:
            te = np.where(folds.values == k)[0]
            q, mh, R, H, ci = joint_posterior(q0[te],
                                              LL_store[model][k],
                                              alpha=alpha)
            frames.append(assemble_predictions(
                te, q, mh, R, H, ci, obs_ids, bsids,
                theta_grid_h=THETA_H))
        df = pd.concat(frames, ignore_index=True)
        p = outdir / "PREDICTIONS_BLINDED.csv.gz"
        df.to_csv(p, index=False, compression="gzip")
        (outdir / "PREDICTION_SHA256.txt").write_text(sha(p) + "\n")
        (outdir / "MODEL_CONFIG.json").write_text(json.dumps({
            "model": model, "alpha": alpha,
            "K_panel": K_PANEL, "lambda": LAMBDA_RED,
            "beta": BETA_COV, "gamma_alpha": GAMMA_ALPHA,
            "selection": "top-K CPI" if model == "C1" else
            "greedy SAFE objective",
            "crossfit": "SMC LOSO 5-fold",
            "alpha_profile": prof2 if model == "C2"
            else prof1,
            "TRUE_ZT_READ": False}, indent=2))
        log(f"  {model} blind predictions frozen "
            f"(alpha={alpha}, {len(df)} cells)")
    write_preds("C1", 1.0)
    write_preds("C2", alpha2)

    # ---------- F4 same-data optimism ----------
    ig_in = np.empty(len(genes))
    Xa = X.toarray().astype(np.float64)
    for s0 in range(0, len(genes), SCORE_CHUNK):
        s1 = min(s0 + SCORE_CHUNK, len(genes))
        LL = nb_loglik_grid(Xa[:, s0:s1], log_s, mF[s0:s1],
                            aF[s0:s1], bF[s0:s1], alF[s0:s1])
        ig_in[s0:s1] = entropy_reduction(q0, LL)
    dep_panel = panels["C2"][0]  # fold-0 panel as representative
    idx = [list(genes).index(g) for g in dep_panel]
    f4 = {"in_sample_mean_IG_panel": float(ig_in[idx].mean()),
          "crossfit_mean_IG_panel": float(cpi[idx].mean()),
          "optimism_ratio": float(ig_in[idx].mean()
                                  / max(cpi[idx].mean(), 1e-12)),
          "genome_in_sample_vs_crossfit_mean": float(
              ig_in.mean() / max(cpi.mean(), 1e-12))}
    (T11B / "08_FALSIFICATION/F4_SAME_VS_CROSSFIT.json").write_text(
        json.dumps(f4, indent=2))
    log(f"F4 optimism: in-sample {f4['in_sample_mean_IG_panel']:.4f}"
        f" vs cross-fit {f4['crossfit_mean_IG_panel']:.4f} nats")
    log(f"SMC pipeline complete in "
        f"{(time.perf_counter()-t00)/60:.1f} min")


if __name__ == "__main__":
    main()
