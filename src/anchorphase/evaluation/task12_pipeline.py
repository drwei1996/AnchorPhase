"""TASK 12 — FROZEN external validation pipeline (identical for both
holdouts). Stages: audit | b0 | cpi | safe | freeze | falsify | eval.
Blind discipline: modeling stages never read ZT; the private map is
written once at audit time (private-side, hashed) and only the eval
stage joins it, after BLIND_FREEZE_COMPLETE=TRUE.

Frozen components (unchanged from Task10A/11B):
- B0: run_katv2_b0_corrected_offset.KATv2AnchoredNBCorrected (EM +
  L-BFGS, CORE17, aorta priors, 48-grid, float64)
- CPI: task11b_cpi fit_genes_3col + score_cpi (entropy reduction)
- panel: greedy SAFE objective (K=40, lambda=0.5, beta=0.5)
- gate: trust-region alpha v2 (<=0.75h, R non-decreasing)
- candidate filter: mean>=0.1 & detected>=1% & non-core
- folds: LOSO if <=8 samples else 5-fold round-robin (sorted ids)
- assembler/cache: task11b_utils (stabilized)
"""
import hashlib
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
from task11b_cpi import (THETA_H, fit_genes_3col, nb_loglik_grid,
                         entropy_reduction, joint_posterior,
                         alpha_gate, circ_mean_h)
from task11b_utils import (assemble_predictions, save_fold_cache,
                           load_fold_cache)
from task11b_run_smc import score_cpi, greedy_panel, fisher_info

ROOT = Path(__file__).resolve().parents[1]
RR = ROOT / "results/single_cell_phase_rebuild"
T12 = RR / "12_KAT_V2_EXTERNAL_VALIDATION_20261008"
A10A = RR / ("10A_KAT_V2_B0_LIBRARY_OFFSET_CORRECTION_"
             "AND_FAIRNESS_20261007")
FREEZE10 = (RR / "10_KAT_V2_FOUNDATION_SINGLECELL_BENCHMARK_20261006"
            / "02_DATA_AND_PRIOR_FREEZE")
CORE17 = ["Arntl", "Bhlhe41", "Clock", "Cry1", "Cry2", "Dbp",
          "Fbxo21", "Hlf", "Nfil3", "Npas2", "Nr1d1", "Nr1d2",
          "Per1", "Per2", "Per3", "Rorc", "Tef"]
H5 = {"GSE145197": ROOT / ("data/processed/GSE145197/"
                           "gse145197_sparse.h5ad"),
      "GSE223109": ROOT / ("data/processed/GSE223109/"
                           "gse223109_sparse.h5ad")}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def D(gse):
    return T12 / gse


def load_cohort(gse):
    """Blind model-facing data: counts (N x G raw UMI, canonical
    gene order), gene list, cell ids, BLIND sample ids, size
    factors. Never touches ZT."""
    import anndata as ad
    A = ad.read_h5ad(H5[gse])
    X = A.X if sp.issparse(A.X) else sp.csr_matrix(A.X)
    X = X.tocsc().astype(np.float64)
    X.data = np.round(X.data)          # enforce raw UMI integers
    genes_lower = {str(g).lower(): i for i, g in
                   enumerate(A.var.gene_name)}
    c17 = [genes_lower[g.lower()] for g in CORE17]
    tot = np.asarray(X.sum(axis=1)).ravel()
    assert (tot > 0).all(), "cells with total UMI 0"
    sf = tot / float(np.median(tot))
    obs = pd.DataFrame({
        "cell_id": [f"{gse}_{i}" for i in range(A.n_obs)],
        "orig_sample_id": A.obs.sample_id.astype(str).values,
        "barcode": A.obs.barcode.astype(str).values
        if "barcode" in A.obs.columns else "NA"})
    sids = sorted(obs.orig_sample_id.unique())
    bmap = {s: f"BLIND_{i:04d}" for i, s in enumerate(sids)}
    obs["blind_sample_id"] = obs.orig_sample_id.map(bmap)
    return X, c17, tot, sf, obs, A


def folds_for(obs):
    n_s = obs.blind_sample_id.nunique()
    if n_s <= 8:
        f = pd.Series(obs.blind_sample_id).astype(
            "category").cat.codes
        scheme = "LOSO"
    else:
        sids = sorted(obs.blind_sample_id.unique())
        bmap = {s: i % 5 for i, s in enumerate(sids)}
        f = obs.blind_sample_id.map(bmap)
        scheme = "5-fold round-robin"
    return f, scheme


def candidates_for(X, gene_names):
    y_sum = np.asarray(X.sum(axis=0)).ravel()
    det = np.asarray((X > 0).sum(axis=0)).ravel()
    keep = ((y_sum / X.shape[0] >= 0.1)
            & (det / X.shape[0] >= 0.01)
            & np.array([g.lower() not in {c.lower()
                                          for c in CORE17}
                        for g in gene_names]))
    return keep, y_sum / X.shape[0], det / X.shape[0]


# ============================================================ audit
def audit(gse):
    out = D(gse) / "00_DATA_AUDIT"
    out.mkdir(parents=True, exist_ok=True)
    X, c17, tot, sf, obs, A = load_cohort(gse)
    f, scheme = folds_for(obs)
    gene_names = [str(g) for g in A.var.gene_name]
    keep, mean_c, prev = candidates_for(X, gene_names)
    # private map (private-side only; hashed; never used before eval)
    zt = A.obs.ZT
    priv = pd.DataFrame({
        "blind_sample_id": obs.blind_sample_id.values,
        "orig_sample_id": obs.orig_sample_id.values,
        "hidden_ZT": pd.to_numeric(zt, errors="coerce"
                                   ).values})
    priv = priv.drop_duplicates("blind_sample_id")
    priv.to_csv(out / "BLIND_MAP_PRIVATE.csv", index=False)
    n_cells = X.shape[0]
    audit = {
        "dataset": gse, "n_cells": int(n_cells),
        "n_genes": int(X.shape[1]),
        "species": str(A.obs.species.iloc[0]),
        "tissue": str(A.obs.tissue.iloc[0]),
        "cell_type": str(A.obs.cell_type.iloc[0]),
        "n_samples": int(obs.blind_sample_id.nunique()),
        "fold_scheme": scheme,
        "X_format": "sparse raw UMI (rounded, integer-verified)",
        "core17": "17/17 present (case-insensitive mapping for "
                  "GSE145197)" if gse == "GSE145197"
            else "17/17 native",
        "total_umi": {"min": float(tot.min()),
                      "median": float(np.median(tot)),
                      "max": float(tot.max()),
                      "zeros": int((tot == 0).sum())},
        "n_candidate_genes": int(keep.sum()),
        "time_label": {"column": "ZT", "available": True,
                       "n_levels_private_side": int(
                           priv.hidden_ZT.nunique()),
                       "note": "values NOT read by modeling stages"},
        "private_map_sha256": sha(out / "BLIND_MAP_PRIVATE.csv"),
        "audit_only_no_performance_inspection": True,
    }
    (out / "DATA_AUDIT.json").write_text(json.dumps(audit,
                                                    indent=2))
    log(f"[{gse}] audit: {n_cells} cells, "
        f"{obs.blind_sample_id.nunique()} samples, "
        f"{keep.sum()} candidates, {scheme}")


# ============================================================ B0
def b0(gse):
    out = D(gse) / "01_B0_BLINDED"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "PREDICTION_SHA256.txt").exists():
        log("B0 already frozen"); return
    X, c17, tot, sf, obs, A = load_cohort(gse)
    X17 = np.asarray(X[:, c17].todense()).astype(np.float64)
    prior = json.loads((FREEZE10 / "ACROPHASE_PRIOR_FREEZE.json")
                       .read_text())
    phi = np.array([prior["priors"][g]["loc_rad"] for g in CORE17])
    from run_katv2_b0_corrected_offset import \
        KATv2AnchoredNBCorrected
    mod = KATv2AnchoredNBCorrected(X17, sf, phi, CORE17)
    t0 = time.perf_counter()
    _, hist = mod.fit()
    mean_h, map_h, R, v, ent, ci, q0 = mod.predict()
    runtime = time.perf_counter() - t0
    frames = assemble_predictions(
        np.arange(len(obs)), q0, np.asarray(mean_h),
        np.asarray(R), np.asarray(ent), np.asarray(ci),
        obs.cell_id.values, obs.blind_sample_id.values,
        theta_grid_h=THETA_H)
    p = out / "PREDICTIONS_BLINDED.csv.gz"
    frames.to_csv(p, index=False, compression="gzip")
    (out / "PREDICTION_SHA256.txt").write_text(sha(p) + "\n")
    np.savez_compressed(out / "PHASE_POSTERIOR.npz", posterior=q0,
                        theta_h=THETA_H)
    gp = pd.DataFrame({"gene": CORE17,
                       "m": mod.m.detach().numpy(),
                       "A": __import__("torch").nn.functional.softplus(
                           mod.a).detach().numpy(),
                       "log_alpha": mod.la.detach().numpy(),
                       "phi_fixed": phi})
    gp.to_csv(out / "GENE_PARAMETERS.csv", index=False)
    (out / "MODEL_FREEZE.json").write_text(json.dumps({
        "model": "B0_corrected_frozen_engine", "n_cells": len(obs),
        "em_cycles": len(hist),
        "runtime_sec": round(runtime),
        "engine_sha256": sha(ROOT / ("scripts/"
                                     "run_katv2_b0_corrected_"
                                     "offset.py")),
        "priors": "frozen aorta acrophase (Task10A)",
        "TRUE_ZT_READ": False}, indent=2))
    log(f"[{gse}] B0 frozen ({len(hist)} cycles, "
        f"{runtime:.0f}s)")


# ============================================================ CPI+C1/C2
def cpi(gse):
    out2 = D(gse) / "02_CPI_BLINDED"
    out3 = D(gse) / "03_SAFE_CPI_BLINDED"
    out2.mkdir(parents=True, exist_ok=True)
    out3.mkdir(parents=True, exist_ok=True)
    X, c17, tot, sf, obs, A = load_cohort(gse)
    gene_names = np.array([str(g) for g in A.var.gene_name])
    keep, mean_c, prev = candidates_for(X, gene_names)
    cand_idx = np.where(keep)[0]
    genes = gene_names[cand_idx]
    Xc = X[:, cand_idx].tocsc()
    log_s = np.log(sf)
    q0 = np.load(D(gse) / "01_B0_BLINDED/PHASE_POSTERIOR.npz"
                 )["posterior"]
    theta_hat = np.mod(circ_mean_h(q0) * 2 * np.pi / 24, 2 * np.pi)
    folds, scheme = folds_for(obs)
    obs.assign(fold=folds.values)[["cell_id", "blind_sample_id",
                                    "fold"]].to_csv(
        out2 / "FOLD_MANIFEST.csv", index=False)
    cache = out2 / "FOLD_CACHE_v1.npz"
    loaded, reason = (load_fold_cache(cache) if cache.exists()
                      else (None, "absent"))
    params = {}
    if loaded is not None:
        params = loaded["params"]
        log(f"cache loaded ({reason}); folds present: "
            f"{sorted(params)}")
    if loaded is not None and np.any(loaded["cpi"]):
        cpi, per_fold = loaded["cpi"], loaded["per_fold"]
    else:
        ks_all = sorted(set(folds.values))
        for k in [k for k in ks_all if k not in params]:
            tr = folds.values != k
            m = np.empty(len(genes)); a = np.empty(len(genes))
            b = np.empty(len(genes)); al = np.empty(len(genes))
            for s0 in range(0, len(genes), 512):
                s1 = min(s0 + 512, len(genes))
                mm, aa, bb, aaal = fit_genes_3col(
                    Xc[tr][:, s0:s1].toarray().astype(np.float64),
                    log_s[tr], theta_hat[tr])
                m[s0:s1], a[s0:s1], b[s0:s1], al[s0:s1] = \
                    mm, aa, bb, aaal
            params[k] = (m, a, b, al)
            ks_done = sorted(params)
            save_fold_cache(cache, fold_ids=ks_done,
                train_sample_ids=[sorted(obs.blind_sample_id[
                    folds != kk].unique()) for kk in ks_done],
                test_sample_ids=[sorted(obs.blind_sample_id[
                    folds == kk].unique()) for kk in ks_done],
                gene_names=list(genes),
                cell_count_train=[int((folds != kk).sum())
                                  for kk in ks_done],
                cell_count_test=[int((folds == kk).sum())
                                 for kk in ks_done],
                params_by_fold=params,
                cpi=np.zeros(len(genes)),
                per_fold=np.zeros((len(set(folds)), len(genes))),
                code_hash=sha(Path(__file__).resolve()),
                input_hash=sha(H5[gse]))
            log(f"  fold {k}: fitted [cached]")
        cpi, per_fold, n_by = score_cpi(Xc, log_s, q0, folds,
                                        params)
        ks = sorted(params)
        save_fold_cache(cache, fold_ids=ks,
            train_sample_ids=[sorted(obs.blind_sample_id[
                folds != kk].unique()) for kk in ks],
            test_sample_ids=[sorted(obs.blind_sample_id[
                folds == kk].unique()) for kk in ks],
            gene_names=list(genes),
            cell_count_train=[int((folds != kk).sum()) for kk in ks],
            cell_count_test=[int((folds == kk).sum()) for kk in ks],
            params_by_fold=params, cpi=cpi, per_fold=per_fold,
            code_hash=sha(Path(__file__).resolve()),
            input_hash=sha(H5[gse]))
        log("CPI scored [cached]")
    # scores table
    A_, B_ = np.sqrt(params[0][1] ** 2 + params[0][2] ** 2), None
    phi_hat = np.mod(np.arctan2(params[0][2], params[0][1]),
                     2 * np.pi) * 24 / (2 * np.pi)
    scores = pd.DataFrame({"gene": genes, "CPI": cpi,
                           "amplitude": A_, "acrophase_h": phi_hat,
                           "mean_counts": mean_c[cand_idx],
                           "prevalence": prev[cand_idx],
                           "dropout": 1 - prev[cand_idx],
                           "alpha_disp": params[0][3]})
    scores.to_csv(out2 / "CPI_SCORES.csv", index=False)
    ks = sorted(params)
    n_by = {k: float((folds.values == k).sum()) for k in ks}
    # C1 + C2 per fold
    panels = {"C1": [], "C2": []}
    LL_store = {"C1": {}, "C2": {}}
    for k in ks:
        te_mask = folds.values == k
        other = [kk for kk in ks if kk != k]
        n_o = np.array([n_by[kk] for kk in other])
        cpi_lfo = (np.stack([per_fold[kk] for kk in other])
                   * n_o[:, None]).sum(0) / n_o.sum()
        m, a, b, al = params[k]
        topK = np.argsort(-cpi_lfo)[:40]
        Xte = Xc[te_mask][:, topK].toarray().astype(np.float64)
        LL_store["C1"][k] = nb_loglik_grid(
            Xte, log_s[te_mask], m[topK], a[topK], b[topK],
            al[topK]).sum(1)
        panels["C1"].append([str(genes[i]) for i in topK])
        mF = np.stack([params[kk][0] for kk in other]).mean(0)
        aF = np.stack([params[kk][1] for kk in other]).mean(0)
        bF = np.stack([params[kk][2] for kk in other]).mean(0)
        alF = np.stack([params[kk][3] for kk in other]).mean(0)
        _, fbins = fisher_info(mF, aF, bF, alF)
        from task11b_run_smc import make_resid_corr_fn
        redfn = make_resid_corr_fn(Xc, log_s, theta_hat,
                                   params[k])
        sel = greedy_panel(cpi_lfo, redfn, fbins, 40)
        Xte2 = Xc[te_mask][:, sel].toarray().astype(np.float64)
        LL_store["C2"][k] = nb_loglik_grid(
            Xte2, log_s[te_mask], m[sel], a[sel], b[sel],
            al[sel]).sum(1)
        panels["C2"].append([str(genes[i]) for i in sel])
        log(f"  fold {k}: C1/C2 panels")
    q0f = [q0[folds.values == k] for k in ks]
    alpha2, prof2 = alpha_gate(q0f, [LL_store["C2"][k]
                                     for k in ks])
    log(f"alpha gate: {alpha2}")
    (out3 / "ALPHA_GATE_PROFILE.csv").write_text(
        pd.DataFrame(prof2).to_csv(index=False))
    for model, outdir, alpha in (("C1", out2, 1.0),
                                 ("C2", out3, alpha2)):
        frames = []
        for k in ks:
            te = np.where(folds.values == k)[0]
            q, mh, R, H, ci = joint_posterior(
                q0[folds.values == k], LL_store[model][k],
                alpha=alpha)
            frames.append(assemble_predictions(
                te, q, mh, R, H, ci, obs.cell_id.values,
                obs.blind_sample_id.values, theta_grid_h=THETA_H))
        df = pd.concat(frames, ignore_index=True)
        p = outdir / "PREDICTIONS_BLINDED.csv.gz"
        df.to_csv(p, index=False, compression="gzip")
        (outdir / "PREDICTION_SHA256.txt").write_text(sha(p)
                                                      + "\n")
        (outdir / "MODEL_CONFIG.json").write_text(json.dumps({
            "model": model, "alpha": alpha, "K": 40,
            "folds": scheme, "TRUE_ZT_READ": False}, indent=2))
        log(f"  {model} blind predictions frozen "
            f"(alpha={alpha})")
    pd.DataFrame({"fold": [k for k in ks for _ in panels["C2"][k]],
                  "gene": [g for pnl in panels["C2"]
                           for g in pnl]}).to_csv(
        out3 / "SELECTED_PANEL.csv", index=False)
    with open(out2 / "PANELS_PER_FOLD.json", "w") as f:
        json.dump(panels, f)


# ============================================================ freeze
def freeze(gse):
    out = D(gse) / "04_PREDICTION_FREEZE"
    out.mkdir(parents=True, exist_ok=True)
    entries = {}
    for stage, d in (("B0", "01_B0_BLINDED"),
                     ("C1", "02_CPI_BLINDED"),
                     ("C2", "03_SAFE_CPI_BLINDED")):
        p = D(gse) / d / "PREDICTIONS_BLINDED.csv.gz"
        entries[stage] = {"file": str(p.relative_to(ROOT)),
                          "sha256": sha(p),
                          "n_cells": len(pd.read_csv(p))}
    code_rows = ["file,sha256"]
    for f_ in ["task12_pipeline.py", "task11b_cpi.py",
               "task11b_utils.py", "task11b_run_smc.py",
               "run_katv2_b0_corrected_offset.py"]:
        code_rows.append(f"scripts/{f_},{sha(ROOT / 'scripts' / f_)}")
    (out / "CODE_HASHES.csv").write_text("\n".join(code_rows)
                                         + "\n")
    (out / "INPUT_HASHES.csv").write_text(
        f"input,sha256\n{H5[gse].name},{sha(H5[gse])}\n")
    for src, dst in [("02_CPI_BLINDED/CPI_SCORES.csv",
                      "CPI_SCORES.csv"),
                     ("03_SAFE_CPI_BLINDED/SELECTED_PANEL.csv",
                      "SELECTED_PANEL.csv"),
                     ("03_SAFE_CPI_BLINDED/ALPHA_GATE_PROFILE.csv",
                      "ALPHA_GATE_PROFILE.csv")]:
        (out / dst).write_bytes((D(gse) / src).read_bytes())
    man = {"dataset": gse, "predictions": entries,
           "BLIND_FREEZE_COMPLETE": True,
           "TRUE_ZT_READ_BY_PIPELINE": False,
           "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "BLIND_FREEZE_MANIFEST.json").write_text(json.dumps(
        man, indent=2))
    log(f"[{gse}] BLIND_FREEZE_COMPLETE = TRUE")


# ============================================================ eval
def eval_ds(gse):
    assert json.loads((D(gse) / "04_PREDICTION_FREEZE/"
                      "BLIND_FREEZE_MANIFEST.json").read_text()
                      )["BLIND_FREEZE_COMPLETE"]
    priv = pd.read_csv(D(gse) / "00_DATA_AUDIT/"
                       "BLIND_MAP_PRIVATE.csv")
    out = D(gse) / "06_EVALUATION"
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    per_sample = {}
    for name, d in (("B0", "01_B0_BLINDED"),
                    ("C1", "02_CPI_BLINDED"),
                    ("C2", "03_SAFE_CPI_BLINDED")):
        df = pd.read_csv(D(gse) / d / "PREDICTIONS_BLINDED.csv.gz")
        m = df.merge(priv, on="blind_sample_id",
                     validate="many_to_one")
        err = d24h(m.phase_mean_h, m.hidden_ZT)
        rows = []
        for sid, g in m.groupby("blind_sample_id"):
            z = np.exp(1j * 2 * np.pi * g.phase_mean_h / 24)
            sp_ = float(np.mod(np.angle(z.mean()) * 24
                               / (2 * np.pi), 24))
            rows.append((sid, float(g.hidden_ZT.iloc[0]), sp_,
                         len(g), float(np.median(
                             d24h(g.phase_mean_h,
                                  g.hidden_ZT)))))
        sdf = pd.DataFrame(rows, columns=["blind_sample_id",
                                          "true_ZT", "sp_phase",
                                          "n_cells", "cell_AE"])
        sdf["sample_AE"] = d24h(sdf.sp_phase, sdf.true_ZT)
        per_sample[name] = sdf
        dd = np.arange(0, 24, 0.01)
        rot_c = np.array([np.median(d24h(m.phase_mean_h + x,
                                         m.hidden_ZT))
                          for x in dd])
        rot_s = np.array([np.median(d24h(sdf.sp_phase + x,
                                         sdf.true_ZT))
                          for x in dd])
        refl = np.array([np.median(d24h(-sdf.sp_phase + x,
                                        sdf.true_ZT))
                         for x in dd])
        cerr = d24h(sdf.sp_phase, sdf.true_ZT)
        from scipy.stats import spearmanr
        cal = float(spearmanr(m.posterior_R, err).statistic)
        qr = pd.qcut(m.posterior_R, 5, duplicates="drop")
        cal_rows = []
        for cat, idx in m.groupby(qr, observed=True).groups.items():
            ii = list(idx)
            cal_rows.append({"model": name, "R_quintile": str(cat),
                             "median_AE": float(np.median(err[ii])),
                             "mean_R": float(
                                 m.posterior_R[ii].mean())})
        results[name] = {
            "cell_median_AE": float(np.median(err)),
            "cell_mean_AE": float(err.mean()),
            "sample_median_AE": float(np.median(cerr)),
            "within_2h": float((err <= 2).mean()),
            "within_4h": float((err <= 4).mean()),
            "anchored_cell_median_AE": float(rot_c.min()),
            "oracle_rotation_h": float(dd[np.argmin(rot_s)]),
            "oracle_sample_median_AE": float(rot_s.min()),
            "anchor_penalty_h": float(np.median(cerr)
                                      - rot_s.min()),
            "reflected_sample_median_AE": float(refl.min()),
            "reflection_gain_h": float(np.median(cerr)
                                       - refl.min()),
            "calib_spearman": cal,
            "calib_rows": cal_rows}
        log(f"[{gse}] {name}: cell={results[name]['cell_median_AE']:.3f} "
            f"sample={results[name]['sample_median_AE']:.3f} "
            f"oracle_rot={results[name]['oracle_rotation_h']:.2f}")
    (out / "EVALUATION.json").write_text(json.dumps(results,
                                                    indent=2))
    all_ps = []
    for name, sdf in per_sample.items():
        sdf = sdf.copy(); sdf["model"] = name
        all_ps.append(sdf)
    pd.concat(all_ps).to_csv(out / "PER_SAMPLE_RESULTS.csv",
                             index=False)
    # falsification-lite summaries if present
    fdir = D(gse) / "08_DIAGNOSTICS"
    fals = {}
    if fdir.exists():
        for f_ in sorted(fdir.glob("perm*_CPI.json")):
            fals[f_.stem] = json.loads(f_.read_text())
        for f_ in sorted(fdir.glob("rot8_CPI.json")):
            fals[f_.stem] = json.loads(f_.read_text())
        for f_ in sorted(fdir.glob("MATCHED_*.json")):
            fals[f_.stem] = json.loads(f_.read_text())
    (out / "FALSIFICATION_LITE.json").write_text(json.dumps(
        fals, indent=2, default=float))


def d24h(a, b):
    d = np.abs(np.mod(a, 24.0) - np.mod(b, 24.0))
    return np.minimum(d, 24.0 - d)


if __name__ == "__main__":
    cmd, gse = sys.argv[1], sys.argv[2]
    {"audit": audit, "b0": b0, "cpi": cpi, "freeze": freeze,
     "eval": eval_ds}[cmd](gse)
