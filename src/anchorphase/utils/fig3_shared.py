"""Shared data loading + curve plotting for fig3a / fig3b panels.

Loads frozen external-validation artifacts (never modifies them):
  - data/processed/<DS>/<ds>_sparse.h5ad   raw counts + ZT labels
  - <T12>/<DS>/01_B0_BLINDED/PREDICTIONS_BLINDED.csv.gz  B0 phases
  - <T12>/TABLES/EXTERNAL_SELECTED_GENES.csv  fitted acrophase/amplitude
Computes per-sample mean log-normalized expression and returns arrays
for plotting expression vs true ZT and vs inferred B0 phase.
"""
import os
import numpy as np
import pandas as pd
import anndata as ad
from fig_style import ROOT, T12


def load_cohort(ds):
    """Returns dict with genes x cells counts slice + phases + ZT."""
    a = ad.read_h5ad(os.path.join(ROOT, "data", "processed", ds,
                                  f"{ds.lower()}_sparse.h5ad"))
    pred = pd.read_csv(os.path.join(T12, ds, "01_B0_BLINDED",
                                    "PREDICTIONS_BLINDED.csv.gz"))
    assert len(pred) == a.n_obs, "positional join broken"
    L = np.asarray(a.X.sum(axis=1)).ravel()
    s = L / np.median(L)
    zt = a.obs["ZT"].astype(float).values
    zt24 = zt % 24.0
    return {"adata": a, "s": s, "zt": zt24, "phase": pred["phase_mean_h"].values,
            "sample": a.obs["sample_id"].astype(str).values}


def gene_expr(coh, gene):
    """log-normalized expression vector for one gene (positional)."""
    a = coh["adata"]
    hit = [v for v in a.var_names if v.lower() == gene.lower()]
    assert hit, f"gene {gene} not found"
    col = a.var_names.get_loc(hit[0])
    x = np.asarray(a.X[:, col].todense()).ravel()
    return np.log1p(x / coh["s"])


def sample_means(coh, y):
    """per-sample (ZT, mean, sd, n) sorted by ZT."""
    df = pd.DataFrame({"zt": coh["zt"], "y": y, "smp": coh["sample"]})
    g = df.groupby("smp").agg(zt=("zt", "first"), mean=("y", "mean"),
                              sd=("y", "std"), n=("y", "size"))
    return g.sort_values("zt")


def fit_cosine(t, y, period=24.0):
    """Least-squares single-cosine fit; returns (offset, amp, acrophase_h)."""
    w = 2 * np.pi / period
    D = np.column_stack([np.ones_like(t, dtype=float),
                         np.cos(w * t), np.sin(w * t)])
    coef, *_ = np.linalg.lstsq(D, y, rcond=None)
    off, a, b = coef
    amp = np.hypot(a, b)
    ph = (np.arctan2(b, a) * period / (2 * np.pi)) % period
    return off, amp, ph


def gene_acrophase(ds, gene):
    """Frozen NB acrophase (h) for a selected gene (mean over folds)."""
    df = pd.read_csv(os.path.join(T12, "TABLES",
                                  "EXTERNAL_SELECTED_GENES.csv"))
    df = df[(df.dataset == ds) & (df.gene.str.lower() == gene.lower())]
    if len(df) == 0:
        return None
    return float(df.acrophase_h.mean())


def circ_ae(pred_h, zt_h):
    d = np.abs((pred_h - zt_h + 12.0) % 24.0 - 12.0)
    return d
