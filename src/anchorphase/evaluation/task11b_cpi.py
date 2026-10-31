"""Task11B CPI library: gene NB periodic fits (chunked IRLS),
cross-fitted conditional phase information (posterior entropy
reduction), non-redundant panel selection, safe alpha gate.

All formulas frozen in 00_PREREGISTRATION/PREREGISTRATION.md.
Single-threaded (CPU quota); parallelism at process level.
"""
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
RR = ROOT / "results/single_cell_phase_rebuild"
SMC = RR / "02B_BASELINE_REPRODUCTION_STRICT"
T11 = RR / "11_KAT_V2_MODEL_FAMILY_DEVELOPMENT_20261007"
T11B = RR / "11B_KAT_V2_SAFE_PHASE_INFORMATION_20261007"
KF = T11B / "07_SAMPLE_BLOCK_CROSSFIT"

K_GRID = 48
THETA = np.linspace(0, 2 * np.pi, K_GRID, endpoint=False)
THETA_H = THETA * 24 / (2 * np.pi)
GAMMA_ALPHA = 2.0
LAMBDA_RED, BETA_COV = 0.5, 0.5
K_PANEL = 40
ALPHA_GRID = np.round(np.arange(0, 1.0001, 0.05), 2)
CHUNK = 512
RIDGE_AB = 1e-2
SHRINK_LS = 8.0        # lineage/state shrinkage weight (prereg)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def circ_mean_h(q, theta_h=THETA_H):
    z = (q * np.exp(1j * 2 * np.pi * theta_h / 24)).sum(axis=1)
    return np.mod(np.angle(z) * 24 / (2 * np.pi), 24.0)


def entropy(q):
    return -(q * np.log(np.maximum(q, 1e-300))).sum(axis=1)


# =================================================== gene NB fits
def fit_genes_3col(Xc, log_s, theta):
    """Fit log mu = log_s + m + a cos t + b sin t per gene (chunk).
    Xc: dense (n, g). Returns m, a, b, alpha arrays (g,)."""
    n, g = Xc.shape
    c, s = np.cos(theta), np.sin(theta)
    tot = Xc.sum(axis=0)
    m = np.log((tot + 0.5) / np.exp(log_s).sum())
    a = np.zeros(g); b = np.zeros(g)
    mu_bar = np.maximum((tot + 0.5) / n, 1e-8)
    var = Xc.var(axis=0)
    alpha = np.clip((var - mu_bar) / mu_bar ** 2, 1e-3, 1e3)
    D = np.column_stack([np.ones(n), c, s])
    for it in range(15):
        eta = log_s[:, None] + m[None, :] + c[:, None] * a[None, :] \
            + s[:, None] * b[None, :]
        mu = np.exp(np.clip(eta, -20, 20))
        w = mu / (1.0 + alpha[None, :] * mu)
        z = eta + (Xc - mu) / np.maximum(mu, 1e-10)
        # weighted normal equations, per gene 3x3
        A = np.empty((g, 3, 3))
        r = np.empty((g, 3))
        for d in range(3):
            Dd = D[:, d]
            wz = w * z                   # (n,g)
            r[:, d] = (wz * Dd[:, None]).sum(axis=0)
            for e in range(d, 3):
                De = D[:, e]
                A[:, d, e] = A[:, e, d] = (w * (Dd * De)[:, None]) \
                    .sum(axis=0)
        A[:, 1, 1] += RIDGE_AB * n
        A[:, 2, 2] += RIDGE_AB * n
        try:
            sol = np.linalg.solve(A, r[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(A, r[:, :, None])[0][:, :, 0]
        m, a, b = sol[:, 0], sol[:, 1], sol[:, 2]
        mu = np.exp(np.clip(log_s[:, None] + m[None, :]
                            + c[:, None] * a[None, :]
                            + s[:, None] * b[None, :], -20, 20))
        var = ((Xc - mu) ** 2).mean(axis=0)
        mubar = mu.mean(axis=0)
        alpha = np.clip((var - mubar) / np.maximum(mubar, 1e-8) ** 2,
                        1e-3, 1e3)
    return m, a, b, alpha


def fit_genes_LS(Xc, log_s, theta, lin_idx, zcov, use_lin=True,
                 use_state=True, shrink=SHRINK_LS):
    """Pooled C3 gene fit: 3 base cols + lineage deltas (dM per
    lineage, dA harmonic deltas) + state betas, ridge-shrunk toward
    pooled. lin_idx: (n,) int 0..L-1; zcov: (n, dz)."""
    n, g = Xc.shape
    c, s = np.cos(theta), np.sin(theta)
    L = int(lin_idx.max()) + 1
    dz = zcov.shape[1] if zcov is not None else 0
    cols = [np.ones(n), c, s]
    names = ["m", "a", "b"]
    if use_lin:
        for l in range(1, L):
            e = (lin_idx == l).astype(float); cols.append(e)
            names.append(f"dM{l}")
        for l in range(1, L):
            e = (lin_idx == l).astype(float)
            cols.append(e * c); names.append(f"dAc{l}")
            cols.append(e * s); names.append(f"dAs{l}")
    if use_state and dz:
        for d in range(dz):
            cols.append(zcov[:, d]); names.append(f"z{d}")
    D = np.column_stack(cols)
    p = D.shape[1]
    pen = np.zeros(p)
    for i, nm in enumerate(names):
        if nm not in ("m", "a", "b"):
            pen[i] = shrink * n
    tot = Xc.sum(axis=0)
    beta = np.zeros((g, p))
    beta[:, 0] = np.log((tot + 0.5) / np.exp(log_s).sum())
    mu_bar = np.maximum((tot + 0.5) / n, 1e-8)
    alpha = np.clip((Xc.var(axis=0) - mu_bar) / mu_bar ** 2,
                    1e-3, 1e3)
    pairs = [(d, e) for d in range(p) for e in range(d, p)]
    Dcols = [D[:, d].copy() for d in range(p)]
    for it in range(15):
        eta = log_s[:, None] + D @ beta.T
        mu = np.exp(np.clip(eta, -20, 20))
        w = mu / (1.0 + alpha[None, :] * mu)
        z = eta + (Xc - mu) / np.maximum(mu, 1e-10)
        A = np.empty((g, p, p))
        r = np.empty((g, p))
        for (d, e) in pairs:
            de = Dcols[d] * Dcols[e]
            A[:, d, e] = (w * de[:, None]).sum(axis=0)
            if e != d:
                A[:, e, d] = A[:, d, e]
        for d in range(p):
            r[:, d] = (w * z * Dcols[d][:, None]).sum(axis=0)
        A[:, np.arange(p), np.arange(p)] += pen[None, :]
        try:
            beta = np.linalg.solve(A, r[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(A, r[:, :, None])[0][:, :, 0]
        mu = np.exp(np.clip(log_s[:, None] + D @ beta.T, -20, 20))
        var = ((Xc - mu) ** 2).mean(axis=0)
        mubar = mu.mean(axis=0)
        alpha = np.clip((var - mubar) / np.maximum(mubar, 1e-8) ** 2,
                        1e-3, 1e3)
    return beta, alpha, names


def eff_params(beta, names, lin_idx=None, zcov=None):
    """Fold lineage deltas and state terms into per-cell effective
    (m, a, b): eta = log_s + m + c(t)(a) + s(t)(b) holds per cell."""
    n = len(lin_idx) if lin_idx is not None else None
    b0 = dict(zip(names, beta))
    m = np.full(n, b0["m"]) if n is not None else b0["m"]
    a = np.full(n, b0["a"]) if n is not None else b0["a"]
    bb = np.full(n, b0["b"]) if n is not None else b0["b"]
    if lin_idx is not None:
        for l in range(1, int(lin_idx.max()) + 1):
            e = lin_idx == l
            if f"dM{l}" in b0:
                m[e] += b0[f"dM{l}"]
            if f"dAc{l}" in b0:
                a[e] += b0[f"dAc{l}"]
            if f"dAs{l}" in b0:
                bb[e] += b0[f"dAs{l}"]
    if zcov is not None:
        for d in range(zcov.shape[1]):
            if f"z{d}" in b0:
                m = m + b0[f"z{d}"] * zcov[:, d]
    return m, a, bb


# =================================================== likelihood grid
def nb_loglik_grid(Xc, log_s, m, a, b, alpha, theta=THETA):
    """log NB pmf on the phase grid. m/a/b either (g,) gene scalars
    or (n,g) per-cell effective params. Returns (n, g, K)."""
    from scipy.special import gammaln
    n, g = Xc.shape
    K = len(theta)
    c = np.cos(theta); s = np.sin(theta)
    r = 1.0 / alpha
    out = np.empty((n, g, K))
    m = np.broadcast_to(m, (n, g)) if np.ndim(m) == 1 else m
    a = np.broadcast_to(a, (n, g)) if np.ndim(a) == 1 else a
    b = np.broadcast_to(b, (n, g)) if np.ndim(b) == 1 else b
    for gi in range(g):
        eta = log_s[:, None] + m[:, gi:gi + 1] \
            + c[None, :] * a[:, gi:gi + 1] \
            + s[None, :] * b[:, gi:gi + 1]        # (n,K)
        mu = np.exp(np.clip(eta, -20, 20))
        x = Xc[:, gi][:, None]
        out[:, gi, :] = (gammaln(x + r[gi]) - gammaln(r[gi])
                         - gammaln(x + 1)
                         + r[gi] * np.log(r[gi])
                         + x * np.clip(np.log(mu), -40, 40)
                         - (r[gi] + x) * np.log(r[gi] + mu))
    return out


# =================================================== scoring
def entropy_reduction(q0, logL):
    """q0: (n,K); logL: (n,g,K). Returns mean_i [H(q0)-H(qg)] (g,)
    and updated joint logliks helper."""
    n, g, K = logL.shape
    H0 = entropy(q0)
    igs = np.empty(g)
    for gi in range(g):
        lj = np.log(np.maximum(q0, 1e-300)) + logL[:, gi, :]
        lse = np.log(np.exp(lj - lj.max(axis=1, keepdims=True))
                     .sum(axis=1, keepdims=True)) \
            + lj.max(axis=1, keepdims=True)
        qg = np.exp(lj - lse)
        igs[gi] = np.mean(H0 - entropy(qg))
    return igs


def joint_posterior(q0, LL, alpha=1.0):
    """q0 (n,K); LL (n,K) joint info-gene loglik. Return posterior,
    mean_h, R, entropy, ci95 width."""
    lj = np.log(np.maximum(q0, 1e-300)) + alpha * LL
    lse = np.log(np.exp(lj - lj.max(axis=1, keepdims=True))
                 .sum(axis=1, keepdims=True)) + lj.max(axis=1,
                                                      keepdims=True)
    q = np.exp(lj - lse)
    mean_h = circ_mean_h(q)
    z = (q * np.exp(1j * 2 * np.pi * THETA_H / 24)).sum(axis=1)
    R = np.abs(z)
    H = entropy(q)
    ci = np.empty(len(q))
    for i in range(len(q)):
        o = np.argsort(-q[i])
        cum = np.cumsum(q[i][o])
        ci[i] = (int(np.searchsorted(cum, 0.95)) + 1) * 0.5
    return q, mean_h, R, H, ci


def alpha_gate(q0_by_fold, LL_by_fold, grid=ALPHA_GRID,
               gamma=GAMMA_ALPHA):
    """SAFE gate v2 (trust-region; prereg amendment 2026-10-07,
    see FAILED_ATTEMPTS/ALPHA_GATE_V1_BROKEN). alpha* = largest a
    on the grid with (i) median circular |phase_a - phase_B0| <=
    0.75 h and (ii) mean posterior R >= mean R of q0. v1's tempered
    marginal objective was convex in a and structurally pinned to
    alpha=0; kept for the record as v1_profile."""
    from task11b_cpi import circ_mean_h as _cmh

    def d24(x, y):
        dd = np.abs(np.mod(x, 24) - np.mod(y, 24))
        return np.minimum(dd, 24 - dd)
    base_ph, base_R = [], []
    for q0 in q0_by_fold:
        base_ph.append(_cmh(q0))
        z = (q0 * np.exp(1j * 2 * np.pi * THETA_H / 24)).sum(axis=1)
        base_R.append(float(np.abs(z).mean()))
    base_ph = np.concatenate(base_ph)
    base_R_mean = float(np.mean(base_R))
    shift_prof, R_prof = [], []
    for al in grid:
        phs, Rs = [], []
        for q0, LL in zip(q0_by_fold, LL_by_fold):
            lj = np.log(np.maximum(q0, 1e-300)) + al * LL
            lse = np.log(np.exp(lj - lj.max(axis=1, keepdims=True))
                         .sum(axis=1, keepdims=True)) \
                + lj.max(axis=1, keepdims=True)
            q = np.exp(lj - lse)
            phs.append(_cmh(q))
            z = (q * np.exp(1j * 2 * np.pi * THETA_H / 24)) \
                .sum(axis=1)
            Rs.append(float(np.abs(z).mean()))
        shift_prof.append(float(np.median(
            d24(np.concatenate(phs), base_ph))))
        R_prof.append(float(np.mean(Rs)))
    ok = [(s <= 0.75) and (r >= base_R_mean)
          for s, r in zip(shift_prof, R_prof)]
    alpha = 0.0
    for al, o in zip(grid, ok):
        if o:
            alpha = float(al)     # largest passing (grid ascending)
    return alpha, {"shift_h": shift_prof, "mean_R": R_prof,
                   "base_R": base_R_mean}
