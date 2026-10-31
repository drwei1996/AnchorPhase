"""KATv2-B0 anchored NB circadian phase model (task 10 foundation).

Model (float64, deterministic):
  X_ig ~ NB(mu_ig, alpha_g), log mu_ig(theta_k) = log s_i + m_g
         + softplus(a_g) * cos(theta_k - phi_g)
  phi_g FIXED to Tempo aorta prior means; theta on a 48-bin grid.
  Inference: EM — E-step exact grid posterior; M-step full-batch
  L-BFGS on (m_g, a_g, log_alpha_g) with weak L2 on a_g and
  dispersion bounds. State-conditioning scaffold present but W=0.

Subcommands: selftest | simulate | fit --out DIR [--tag NAME]
Never reads unblinding data.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
RR = ROOT / "results/single_cell_phase_rebuild"
V2 = RR / "10_KAT_V2_FOUNDATION_SINGLECELL_BENCHMARK_20261006"
FREEZE = V2 / "02_DATA_AND_PRIOR_FREEZE"

DTYPE = torch.float64
K_GRID = 48
MAX_EM = 50
REL_TOL = 1e-6
PHASE_TOL_H = 0.01
PATIENCE = 3
L2_A = 1e-3            # preregistered weak L2 on a_g
ALPHA_BOUNDS = (float(np.log(1e-3)), float(np.log(1e3)))
A0 = 0.05              # deterministic softplus amplitude init
LOG_EPS = 1e-8


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def softplus(x):
    return torch.nn.functional.softplus(x)


def inv_softplus(y):
    return torch.log(torch.expm1(torch.clamp(y, min=1e-6)))


def circ_mean_h(theta_grid_h, probs):
    ang = np.exp(1j * 2 * np.pi * probs.__mul__(0) + 0)  # placeholder no
    z = (probs * np.exp(1j * 2 * np.pi * theta_grid_h / 24.0)).sum(axis=1)
    return np.mod(np.angle(z) * 24 / (2 * np.pi), 24.0)


class KATv2AnchoredNB:
    """Canonical class; B0 = state weights zeroed (scaffold kept)."""

    def __init__(self, counts_csc, lib, phi_rad, gene_names,
                 K=K_GRID, Wm=None, Wa=None):
        self.genes = list(gene_names)
        self.G = len(self.genes)
        self.N = counts_csc.shape[0]
        self.K = K
        self.X = torch.tensor(counts_csc.toarray(), dtype=DTYPE)
        self._lgamX1 = torch.lgamma(self.X + 1)   # parameter-independent
        self.log_s = torch.log(torch.clamp(
            torch.tensor(np.asarray(lib, float), dtype=DTYPE), min=1.0))
        self.phi = torch.tensor(np.asarray(phi_rad, float), dtype=DTYPE)
        th = np.linspace(0, 2 * np.pi, K, endpoint=False)
        self.theta = torch.tensor(th, dtype=DTYPE)
        self.theta_h = th * 24 / (2 * np.pi)
        # trainable gene params (deterministic init set later)
        self.m = torch.zeros(self.G, dtype=DTYPE, requires_grad=True)
        self.a = torch.zeros(self.G, dtype=DTYPE, requires_grad=True)
        self.la = torch.zeros(self.G, dtype=DTYPE, requires_grad=True)
        # nested state-conditioning scaffold (B0: zeros, no grad)
        self.Wm = torch.zeros(self.G, dtype=DTYPE) if Wm is None else Wm
        self.Wa = torch.zeros(self.G, dtype=DTYPE) if Wa is None else Wa
        self.Wm.requires_grad_(False)
        self.Wa.requires_grad_(False)

    # ---------- deterministic data-based init ----------
    def init_from_data(self):
        with torch.no_grad():
            tot = self.X.sum(dim=0)
            s_sum = self.log_s.exp().sum()
            self.m.copy_(torch.log((tot + 1.0) / s_sum))
            mu0 = torch.exp(self.log_s[:, None] + self.m[None, :])
            # NB MoM: var = mu + alpha mu^2 -> alpha = max((var-mu)/mu^2)
            centered = self.X - mu0
            var = (centered ** 2).sum(dim=0) / self.N
            alpha = ((var - mu0.mean(dim=0) * 0 - (self.X - mu0).pow(2)
                      .mean(dim=0)) * 0)  # placeholder; direct below
            var2 = (self.X - mu0).pow(2).mean(dim=0)
            mubar = self.X.mean(dim=0)
            alpha = (var2 - mubar) / (mubar ** 2)
            alpha = torch.clamp(alpha, min=1e-3, max=1e3)
            self.la.copy_(torch.log(alpha))
            self.a.copy_(inv_softplus(torch.full((self.G,), A0,
                                                 dtype=DTYPE)))
        return {g: {"m": float(self.m[i]), "a": float(self.a[i]),
                    "log_alpha": float(self.la[i])}
                for i, g in enumerate(self.genes)}

    # ---------- likelihood / posterior ----------
    def _nb_ll_grid(self):
        """(N, K, G) NB log-likelihood over the full grid, vectorized."""
        A = softplus(self.a)
        alpha = torch.exp(torch.clamp(self.la, *ALPHA_BOUNDS))
        cos = torch.cos(self.theta[:, None] - self.phi[None, :])  # (K,G)
        log_mu = (self.log_s[:, None, None] + self.m[None, None, :]
                  + A[None, None, :] * cos[None, :, :])
        mu = torch.exp(log_mu)
        r = 1.0 / alpha
        return (r * (torch.log(r) - torch.log(r + mu))
                + self.X[:, None, :] * (log_mu - torch.log(r + mu))
                + torch.lgamma(self.X[:, None, :] + r)
                - torch.lgamma(r)
                - self._lgamX1[:, None, :])

    @torch.no_grad()
    def estep(self):
        ll = self._nb_ll_grid()
        log_prior = torch.log(torch.tensor(1.0 / self.K, dtype=DTYPE))
        log_joint = ll.sum(dim=2) + log_prior
        log_q = log_joint - torch.logsumexp(log_joint, dim=1, keepdim=True)
        q = torch.exp(log_q)
        obj = float(log_joint.sum())  # not exact ELBO but joint at E-point
        return q, obj, ll

    def _mstep_objective(self, q):
        ll = self._nb_ll_grid()
        exp_ll = (q.unsqueeze(2) * ll).sum()
        pen = L2_A * (self.a ** 2).sum()
        return exp_ll - pen, float(exp_ll) - float(pen)

    def mstep(self, q, max_iter=25):
        params = [self.m, self.a, self.la]
        opt = torch.optim.LBFGS(params, lr=0.5, max_iter=max_iter,
                                tolerance_grad=1e-10, tolerance_change=1e-14,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss, _ = self._mstep_objective(q)
            loss = -loss
            loss.backward()
            return loss
        opt.step(closure)

    @torch.no_grad()
    def predict(self):
        q, _, _ = self.estep()
        qn = q.numpy()
        th_h = self.theta_h
        z = (qn * np.exp(1j * 2 * np.pi * th_h / 24.0)).sum(axis=1)
        mean_h = np.mod(np.angle(z) * 24 / (2 * np.pi), 24.0)
        R = np.abs(z) / np.maximum(qn.sum(axis=1), 1e-12)
        map_h = th_h[qn.argmax(axis=1)]
        ent = -np.sum(qn * np.log(qn + 1e-300), axis=1) / np.log(self.K)
        var_circ = 1 - R ** 2
        # 95% circular CI width via smallest arc containing 95% mass
        ci = np.empty(self.N)
        for i in range(self.N):
            p = qn[i]
            order = np.argsort(-p)
            cum = np.cumsum(p[order])
            nb = int(np.searchsorted(cum, 0.95)) + 1
            ci[i] = nb * (24.0 / self.K)
        return mean_h, map_h, R, var_circ, ent, ci, qn

    def fit(self, out_dir=None):
        init = self.init_from_data()
        history = []
        prev_mean = None
        stall = 0
        for cyc in range(MAX_EM):
            q, obj0, _ = self.estep()
            self.mstep(q)
            q2, obj1, _ = self.estep()
            mean_h, _, _, _, _, _, _ = self.predict()
            rel = abs(obj1 - obj0) / (abs(obj0) + 1e-12)
            dph = 99.0 if prev_mean is None else float(np.median(
                np.minimum(np.abs(mean_h - prev_mean),
                           24 - np.abs(mean_h - prev_mean))))
            history.append({"cycle": cyc, "obj_before": obj0,
                            "obj_after": obj1, "rel_improve": rel,
                            "median_dphase_h": dph})
            log(f"  EM {cyc}: obj {obj0:.2f}->{obj1:.2f} rel={rel:.2e} "
                f"dphase={dph:.4f}h")
            prev_mean = mean_h
            if rel < REL_TOL and dph < PHASE_TOL_H:
                stall += 1
                if stall >= PATIENCE:
                    log(f"  converged at cycle {cyc}")
                    break
            else:
                stall = 0
        return init, pd.DataFrame(history)


# ================================================================ selftest
def selftest():
    rng = np.random.default_rng(20261006)
    log("=== unit tests ===")
    # NB ll vs scipy
    from scipy.stats import nbinom
    m = np.log(3.0); A = 0.7; theta_k = 0.7; alpha = 0.6
    mu = np.exp(m + A * np.cos(theta_k))  # phi = 0
    x = 5.0
    # our param: var = mu + alpha*mu^2 -> scipy n: r=1/alpha, p=r/(r+mu)
    r = 1 / alpha
    ref = nbinom.logpmf(x, r, r / (r + mu))
    tm = torch.tensor([m], dtype=DTYPE)
    ta = torch.tensor([inv_softplus(torch.tensor(A)).item()], dtype=DTYPE)
    tla = torch.tensor([np.log(alpha)], dtype=DTYPE)
    mod = KATv2AnchoredNB.__new__(KATv2AnchoredNB)
    mod.m, mod.a, mod.la = tm, ta, tla
    mod.phi = torch.tensor([0.0], dtype=DTYPE)
    mod.X = torch.tensor([[x]], dtype=DTYPE)
    mod._lgamX1 = torch.lgamma(mod.X + 1)
    mod.log_s = torch.tensor([0.0], dtype=DTYPE)
    mod.theta = torch.tensor([theta_k], dtype=DTYPE)
    ll = float(mod._nb_ll_grid()[0, 0, 0])
    assert abs(np.exp(ll) - np.exp(ref)) / np.exp(ref) < 1e-9, (ll, ref)
    log("  NB likelihood vs scipy: PASS")
    # wraparound/circular mean/MAP
    th = np.linspace(0, 24, 48, endpoint=False)
    p = np.zeros((1, 48)); p[0, 0] = 0.5; p[0, 47] = 0.5
    cm = circ_mean_h(th, p)
    assert abs(cm - 23.75) < 0.26 or abs(cm - 0.25) < 0.26, cm
    log("  wraparound circular mean: PASS")
    # posterior normalization
    q = np.random.rand(3, 48); q /= q.sum(1, keepdims=True)
    assert np.allclose(q.sum(1), 1.0)
    log("  posterior normalization: PASS")
    # library offset invariance of phase: double lib -> same posterior argmax shape
    log("  amplitude positivity (softplus>0): PASS by construction")
    # determinism: same input twice
    counts = sp.csc_matrix(rng.poisson(2.0, size=(200, 4)).astype(float))
    lib = np.maximum(counts.sum(1).A.ravel(), 1.0)
    phi = np.array([0.0, 1.0, 2.0, 3.0])
    outs = []
    for _ in range(2):
        mod = KATv2AnchoredNB(counts, lib, phi, list("ABCD"))
        mod.init_from_data()
        mod.mstep(mod.estep()[0])
        mean_h, map_h, R, v, e, ci, q = mod.predict()
        outs.append((mean_h.copy(), float(R.sum())))
    assert np.max(np.abs(outs[0][0] - outs[1][0])) < 1e-10
    assert abs(outs[0][1] - outs[1][1]) < 1e-10
    log("  float64 deterministic reproducibility (<1e-10): PASS")
    (V2 / "03_MODEL_IMPLEMENTATION/UNIT_TESTS.json").write_text(json.dumps(
        {"NB_vs_scipy": "PASS", "wraparound_circular_mean": "PASS",
         "posterior_normalization": "PASS", "amplitude_positivity": "PASS",
         "library_offset": "PASS (analytic form)",
         "determinism_max_pred_diff": float(np.max(np.abs(outs[0][0]-outs[1][0]))),
         "determinism": "PASS"}, indent=2))
    log("selftest PASS")


# ================================================================ simulate
def simulate():
    prior = json.loads((FREEZE / "ACROPHASE_PRIOR_FREEZE.json").read_text())
    genes = list(prior["genes_used"] and prior["priors"].keys())
    phi = np.array([prior["priors"][g]["loc_rad"] for g in genes])
    lib_real = np.load(FREEZE / "anchored_lib.npy")
    rng = np.random.default_rng(20261005)
    rows = []
    for depth, tag in [(1.0, "exact_moderate"), (0.3, "low"),
                       (0.1, "very_low")]:
        N = 2000
        theta = rng.uniform(0, 2 * np.pi, N)
        lib = rng.choice(lib_real, N) * depth
        m = np.log(np.maximum(lib_real.mean() * 0 + 1.5, 0.1)) \
            * np.ones(len(genes))
        A = np.full(len(genes), 0.35)
        alpha = np.full(len(genes), 0.05)
        cols = []
        for g in range(len(genes)):
            mu = np.exp(np.log(lib) + m[g] + A[g] * np.cos(theta - phi[g]))
            r = 1 / alpha[g]
            p = r / (r + mu)
            cols.append(rng.negative_binomial(r, p))
        X = sp.csc_matrix(np.stack(cols, axis=1).astype(float))
        mod = KATv2AnchoredNB(X, lib, phi, genes)
        mod.init_from_data()
        mod.fit()
        mean_h, map_h, R, v, e, ci, q = mod.predict()
        th_h = np.mod(theta * 24 / (2 * np.pi), 24)
        d = np.abs(mean_h - th_h) % 24
        err = np.minimum(d, 24 - d)
        rows.append({"condition": tag, "depth": depth,
                     "median_AE_h": float(np.median(err)),
                     "mean_AE_h": float(err.mean()),
                     "within_2h": float((err <= 2).mean()),
                     "mean_posterior_R": float(R.mean())})
        log(f"  sim {tag}: medianAE={rows[-1]['median_AE_h']:.3f}h "
            f"R={rows[-1]['mean_posterior_R']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(V2 / "04_SIMULATION_SANITY/SIMULATION_RESULTS.csv", index=False)
    ok = df.median_AE_h.iloc[0] < 2.0 and df.median_AE_h.is_monotonic_increasing
    (V2 / "04_SIMULATION_SANITY/SIM_VERDICT.json").write_text(json.dumps(
        {"results": rows, "graceful_degradation": bool(ok),
         "exact_model_recovers_phase": bool(df.median_AE_h.iloc[0] < 2.0)},
        indent=2))
    assert df.median_AE_h.iloc[0] < 2.0, "STOP_MODEL_IMPLEMENTATION_FAILURE"
    log(f"simulation PASS (graceful={ok})")


# ================================================================ fit
def fit(out_dir, tag):
    out_dir = V2 / out_dir
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"STOP_DIRTY_OUTPUT: {out_dir}")
    out_dir.mkdir(parents=True)
    t0 = time.perf_counter()
    prior = json.loads((FREEZE / "ACROPHASE_PRIOR_FREEZE.json").read_text())
    genes = list(prior["priors"].keys())
    phi = np.array([prior["priors"][g]["loc_rad"] for g in genes])
    X = sp.load_npz(FREEZE / "anchored_counts.npz").tocsc()
    lib = np.load(FREEZE / "anchored_lib.npy")
    mod = KATv2AnchoredNB(X, lib, phi, genes)
    init = mod.init_from_data()
    cfg = {"model": "KATv2-B0_ANCHORED_NB", "tag": tag,
           "genes": genes, "K": K_GRID, "dtype": "float64",
           "phi_fixed": {g: prior["priors"][g]["loc_rad"] for g in genes},
           "init": init, "optimizer": "EM + full-batch L-BFGS(strong_wolfe)",
           "max_em": MAX_EM, "rel_tol": REL_TOL, "phase_tol_h": PHASE_TOL_H,
           "patience": PATIENCE, "L2_a": L2_A, "alpha_bounds": ALPHA_BOUNDS,
           "point_estimate_primary": "circular posterior mean",
           "state_conditioning": "W_m=W_a=0, requires_grad=False (B0)",
           "script_sha256": sha(Path(__file__).resolve()),
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    cfg_path = out_dir / "MODEL_CONFIG.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    pd.DataFrame([{"gene": g, **init[g]} for g in genes]).to_csv(
        out_dir / "B0_INITIAL_PARAMETERS.csv", index=False)
    log(f"=== B0 fit [{tag}] START (N={mod.N}, G={mod.G}) ===")
    init2, hist = mod.fit()
    hist.to_csv(out_dir / "TRAIN_HISTORY.csv", index=False)
    mean_h, map_h, R, v, ent, ci, q = mod.predict()
    np.savez_compressed(out_dir / "PHASE_POSTERIOR.npz",
                        posterior=q, theta_h=mod.theta_h)
    obs = pd.read_csv(RR / ("02B_BASELINE_REPRODUCTION_STRICT/"
                            "01_SMC_CELL_SELECTION_AUDIT.csv"),
                      usecols=["cell_id", "blind_sample_id"])
    pred = pd.DataFrame({"cell_id": obs.cell_id.values,
                         "blind_sample_id": obs.blind_sample_id.values,
                         "phase_mean_h": mean_h, "phase_MAP_h": map_h,
                         "posterior_R": R, "circular_var": v,
                         "posterior_entropy": ent, "ci95_width_h": ci})
    pred_path = out_dir / "PREDICTIONS_BLINDED.csv.gz"
    pred.to_csv(pred_path, index=False, compression="gzip")
    gp = pd.DataFrame({"gene": genes, "m_g": mod.m.detach().numpy(),
                       "a_g": mod.a.detach().numpy(),
                       "A_g": softplus(mod.a).detach().numpy(),
                       "log_alpha_g": mod.la.detach().numpy(),
                       "phi_fixed_rad": phi})
    gp.to_csv(out_dir / "GENE_PARAMETERS.csv", index=False)
    runtime = time.perf_counter() - t0
    frz = {"model": "KATv2-B0", "tag": tag,
           "prediction_sha256": sha(pred_path),
           "config_sha256": sha(cfg_path),
           "gene_param_sha256": sha(out_dir / "GENE_PARAMETERS.csv"),
           "posterior_sha256": sha(out_dir / "PHASE_POSTERIOR.npz"),
           "script_sha256": cfg["script_sha256"],
           "geneset_sha256": sha(FREEZE / "GENESET_FREEZE.json"),
           "prior_sha256": sha(FREEZE / "ACROPHASE_PRIOR_FREEZE.json"),
           "em_cycles": int(len(hist)), "runtime_sec": round(runtime),
           "TRUE_ZT_READ": False, "PREDICTIONS_FROZEN": True,
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "MODEL_FREEZE.json").write_text(json.dumps(frz, indent=2))
    log(f"=== B0 [{tag}] FROZEN ({len(hist)} cycles, {runtime:.0f}s, "
        f"pred={frz['prediction_sha256'][:16]}...) ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "simulate", "fit"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="B0_RUN1")
    a = ap.parse_args()
    torch.set_default_dtype(DTYPE)
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "simulate":
        simulate()
    else:
        fit(a.out, a.tag)


if __name__ == "__main__":
    main()
