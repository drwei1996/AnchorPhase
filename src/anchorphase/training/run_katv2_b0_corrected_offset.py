"""KATv2-B0 with CORRECTED library offset (task 10A, memory-lean v2).

Derived from scripts/run_katv2_b0.py (task 10, untouched). Substantive
model change is EXACTLY ONE: size-factor offset replaces the clock17
subtotal. Everything else (17 genes, phi, NB, softplus amplitude, 48
grid, EM + full-batch L-BFGS float64 strong_wolfe max_iter=25, L2 1e-3,
alpha bounds, init METHOD, posterior mean primary) unchanged.

Memory adaptations for the 2GB cgroup (base services occupy ~1.5GB):
- no pandas/scipy imports; npz read via zipfile streaming; csv module
- M-step accumulates gradients chunk-by-chunk (per-grid-point backward),
  mathematically identical to summing the full objective
Threading pinned to 1 (env + torch) per task reproducibility protocol.

Subcommands: selftest | simulate | fit --out DIR --tag NAME [--perturb X]
Never reads unblinding data.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import time
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

ROOT = Path(__file__).resolve().parents[1]
RR = ROOT / "results/single_cell_phase_rebuild"
A = RR / "10A_KAT_V2_B0_LIBRARY_OFFSET_CORRECTION_AND_FAIRNESS_20261007"
T10 = RR / "10_KAT_V2_FOUNDATION_SINGLECELL_BENCHMARK_20261006"
FREEZE = T10 / "02_DATA_AND_PRIOR_FREEZE"
SIZE_FACTOR = A / "02_CORRECTED_SIZE_FACTOR_FREEZE/SIZE_FACTOR_TOTAL_UMI_MEDIAN1.npy"

DTYPE = torch.float64
K_GRID = 48
MAX_EM = 50
REL_TOL = 1e-6
PHASE_TOL_H = 0.01
PATIENCE = 3
L2_A = 1e-3
ALPHA_BOUNDS = (float(np.log(1e-3)), float(np.log(1e3)))
A0 = 0.05
PERTURB_SEED = 20261007


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def softplus(x):
    return torch.nn.functional.softplus(x)


def inv_softplus(y):
    return torch.log(torch.expm1(torch.clamp(y, min=1e-6)))


def load_dense_counts_npz(npz_path):
    """Stream a scipy csr .npz (already subset to the anchored 17
    genes) into a dense (N, G) float64 array without scipy."""
    with zipfile.ZipFile(npz_path) as zf:
        def member(name):
            f = zf.open(name)
            version = np.lib.format.read_magic(f)
            hdr = (np.lib.format.read_array_header_1_0(f) if version
                   == (1, 0) else np.lib.format.read_array_header_2_0(f))
            shape, dtype = hdr[0], hdr[2]
            n = int(np.prod(shape))
            return np.frombuffer(f.read(n * dtype.itemsize),
                                 dtype=dtype), shape
        data, _ = member("data.npy")
        indices, _ = member("indices.npy")
        indptr, _ = member("indptr.npy")
        shape, _ = member("shape.npy")
        N, G = int(shape[0]), int(shape[1])
        X = np.zeros((N, G), dtype=np.float64)
        for i in range(N):
            a, b = int(indptr[i]), int(indptr[i + 1])
            if a != b:
                X[i, indices[a:b].astype(np.int64)] = data[a:b]
        return X


def read_cell_ids():
    with open(RR / ("02B_BASELINE_REPRODUCTION_STRICT/"
                    "01_SMC_CELL_SELECTION_AUDIT.csv")) as f:
        r = csv.DictReader(f)
        ids, bsids = [], []
        for row in r:
            ids.append(row["cell_id"])
            bsids.append(row["blind_sample_id"])
    return ids, bsids


class KATv2AnchoredNBCorrected:
    """Identical math to task-10 model; offset = whole-transcriptome
    size factor; per-grid-point (chunked) evaluation."""

    def __init__(self, X_dense, size_factor, phi_rad, gene_names,
                 K=K_GRID):
        self.genes = list(gene_names)
        self.G = len(self.genes)
        self.N = X_dense.shape[0]
        self.K = K
        self.X = torch.tensor(X_dense, dtype=DTYPE)
        self._lgamX1 = torch.lgamma(self.X + 1)
        sf = torch.tensor(np.asarray(size_factor, float), dtype=DTYPE)
        assert bool(torch.all(sf > 0)) and bool(torch.isfinite(sf).all())
        self.log_s = torch.log(sf)
        self.phi = torch.tensor(np.asarray(phi_rad, float), dtype=DTYPE)
        th = np.linspace(0, 2 * np.pi, K, endpoint=False)
        self.theta = torch.tensor(th, dtype=DTYPE)
        self.theta_h = th * 24 / (2 * np.pi)
        self.m = torch.zeros(self.G, dtype=DTYPE, requires_grad=True)
        self.a = torch.zeros(self.G, dtype=DTYPE, requires_grad=True)
        self.la = torch.zeros(self.G, dtype=DTYPE, requires_grad=True)

    def _nb_ll_chunk(self, theta_k):
        Am = softplus(self.a)
        alpha = torch.exp(torch.clamp(self.la, *ALPHA_BOUNDS))
        log_mu = (self.log_s[:, None] + self.m[None, :]
                  + Am[None, :] * torch.cos(theta_k - self.phi)[None, :])
        mu = torch.exp(log_mu)
        r = 1.0 / alpha
        return (r * (torch.log(r) - torch.log(r + mu))
                + self.X * (log_mu - torch.log(r + mu))
                + torch.lgamma(self.X + r) - torch.lgamma(r)
                - self._lgamX1)

    @torch.no_grad()
    def estep(self):
        parts = [self._nb_ll_chunk(self.theta[k]).sum(dim=1)
                 for k in range(self.K)]
        log_joint = torch.stack(parts, dim=1)
        log_joint = log_joint + math.log(1.0 / self.K)
        log_q = log_joint - torch.logsumexp(log_joint, dim=1, keepdim=True)
        return torch.exp(log_q), float(log_joint.sum())

    def mstep(self, q, max_iter=25):
        opt = torch.optim.LBFGS([self.m, self.a, self.la], lr=0.5,
                                max_iter=max_iter,
                                tolerance_grad=1e-10,
                                tolerance_change=1e-14,
                                line_search_fn="strong_wolfe")

        def closure():
            # backwards and returns the SAME negated objective
            # (task-10 pattern); sign consistency is mandatory for
            # strong_wolfe - a flipped gradient sign freezes LBFGS
            opt.zero_grad()
            loss_total = 0.0
            for k in range(self.K):
                term = (q[:, k:k + 1]
                        * self._nb_ll_chunk(self.theta[k])).sum()
                pen = (L2_A * (self.a ** 2).sum()) / self.K
                chunk_loss = pen - term
                chunk_loss.backward()
                loss_total += float(chunk_loss.detach())
            return torch.tensor(loss_total, dtype=DTYPE)
        opt.step(closure)

    @torch.no_grad()
    def predict(self):
        q, _ = self.estep()
        qn = q.numpy()
        th_h = self.theta_h
        z = (qn * np.exp(1j * 2 * np.pi * th_h / 24.0)).sum(axis=1)
        mean_h = np.mod(np.angle(z) * 24 / (2 * np.pi), 24.0)
        R = np.abs(z) / np.maximum(qn.sum(axis=1), 1e-12)
        map_h = th_h[qn.argmax(axis=1)]
        ent = -np.sum(qn * np.log(qn + 1e-300), axis=1) / np.log(self.K)
        ci = np.empty(self.N)
        for i in range(self.N):
            p = qn[i]
            order = np.argsort(-p)
            cum = np.cumsum(p[order])
            nb = int(np.searchsorted(cum, 0.95)) + 1
            ci[i] = nb * (24.0 / self.K)
        return mean_h, map_h, R, 1 - R ** 2, ent, ci, qn

    def init_from_data(self):
        with torch.no_grad():
            tot = self.X.sum(dim=0)
            s_sum = self.log_s.exp().sum()
            self.m.copy_(torch.log((tot + 1.0) / s_sum))
            mu0 = torch.exp(self.log_s[:, None] + self.m[None, :])
            var2 = (self.X - mu0).pow(2).mean(dim=0)
            mubar = self.X.mean(dim=0)
            alpha = torch.clamp((var2 - mubar) / (mubar ** 2),
                                min=1e-3, max=1e3)
            self.la.copy_(torch.log(alpha))
            self.a.copy_(inv_softplus(torch.full((self.G,), A0,
                                                 dtype=DTYPE)))
        return {g: {"m": float(self.m[i].detach()),
                    "a": float(self.a[i].detach()),
                    "log_alpha": float(self.la[i].detach())}
                for i, g in enumerate(self.genes)}

    def fit(self, perturb=None):
        init = self.init_from_data()
        if perturb is not None:
            g = torch.Generator().manual_seed(PERTURB_SEED)
            with torch.no_grad():
                for p in (self.m, self.a, self.la):
                    delta = torch.randn(p.shape, generator=g,
                                        dtype=DTYPE)
                    p.add_(delta / delta.norm() * perturb)
        history = []
        prev_mean = None
        stall = 0
        for cyc in range(MAX_EM):
            q, obj0 = self.estep()
            self.mstep(q)
            _, obj1 = self.estep()
            mean_h = self.predict()[0]
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
        return init, history


def write_csv(path, header, rows, compress=False):
    op = (gzip.open if compress else open)
    mode = "wt" if compress else "w"
    with op(path, mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ================================================================ selftest
def selftest():
    log("=== unit tests (corrected offset, memory-lean v2) ===")
    # 1) NB vs closed-form reference (math.lgamma; no scipy)
    m0 = math.log(3.0); A_t = 0.7; th = 0.7; alpha = 0.6; sf = 2.5
    x = 5.0
    mu = math.exp(math.log(sf) + m0 + A_t * math.cos(th))
    r = 1 / alpha
    ref = (math.lgamma(x + r) - math.lgamma(r) - math.lgamma(x + 1)
           + r * math.log(r) + x * math.log(mu)
           - (r + x) * math.log(r + mu))
    mod = KATv2AnchoredNBCorrected.__new__(KATv2AnchoredNBCorrected)
    mod.m = torch.tensor([m0], dtype=DTYPE)
    mod.a = torch.tensor([inv_softplus(torch.tensor(A_t)).item()],
                         dtype=DTYPE)
    mod.la = torch.tensor([math.log(alpha)], dtype=DTYPE)
    mod.phi = torch.tensor([0.0], dtype=DTYPE)
    mod.X = torch.tensor([[x]], dtype=DTYPE)
    mod._lgamX1 = torch.lgamma(mod.X + 1)
    mod.log_s = torch.tensor([math.log(sf)], dtype=DTYPE)
    mod.theta = torch.tensor([th], dtype=DTYPE)
    ll = float(mod._nb_ll_chunk(mod.theta[0])[0, 0])
    assert abs(math.exp(ll) - math.exp(ref)) / math.exp(ref) < 1e-9
    log("  NB likelihood vs closed-form (size-factor offset): PASS")
    # 2) chunked torch grid vs closed-form numpy at 5 grid points
    rng = np.random.default_rng(7)
    Xd = rng.poisson(2.0, size=(150, 5)).astype(np.float64)
    sfv = rng.uniform(1.0, 3.0, 150)
    phi = rng.uniform(0, 6, 5)
    mc = KATv2AnchoredNBCorrected(Xd, sfv, phi, list("ABCDE"))
    mc.init_from_data()
    m_ = mc.m.detach().numpy()
    a_ = softplus(mc.a).detach().numpy()
    la_ = np.clip(mc.la.detach().numpy(), *ALPHA_BOUNDS)
    dmax = 0.0
    for k in (0, 9, 19, 33, 47):
        thk = float(mc.theta[k])
        tll = mc._nb_ll_chunk(mc.theta[k]).detach().numpy()
        for gi in range(5):
            lg = (math.log(sfv[0]) if False else
                  np.log(sfv) + m_[gi] + a_[gi] * np.cos(thk - phi[gi]))
            mu_ = np.exp(lg)
            r_ = 1 / math.exp(la_[gi])
            nlg = (np.fromiter((math.lgamma(v + r_) for v in Xd[:, gi]),
                               float, Xd.shape[0])
                   - math.lgamma(r_)
                   - np.fromiter((math.lgamma(v + 1) for v in Xd[:, gi]),
                                 float, Xd.shape[0])
                   + r_ * math.log(r_) + Xd[:, gi] * lg
                   - (r_ + Xd[:, gi]) * np.log(r_ + mu_))
            dmax = max(dmax, float(np.abs(tll[:, gi] - nlg).max()))
    assert dmax < 1e-9, dmax
    log(f"  chunked grid vs closed-form: max {dmax:.2e} PASS")
    # 3) offset assertion
    sf_frozen = np.load(SIZE_FACTOR)
    mod2 = KATv2AnchoredNBCorrected(np.zeros((3, 2)), sf_frozen[:3],
                                    np.zeros(2), ["A", "B"])
    assert torch.allclose(mod2.log_s,
                          torch.log(torch.tensor(sf_frozen[:3],
                                                 dtype=DTYPE)))
    c17 = np.load(A / "01_OFFSET_FORENSIC/clock17_sum.npy")[:3]
    assert not torch.allclose(
        mod2.log_s, torch.log(torch.clamp(torch.tensor(c17, dtype=DTYPE),
                                          min=1.0)))
    log("  offset == log(SIZE_FACTOR_TOTAL_UMI_MEDIAN1): PASS")
    # 4) wraparound circular mean
    thg = np.linspace(0, 24, 48, endpoint=False)
    p = np.zeros((1, 48)); p[0, 0] = 0.5; p[0, 47] = 0.5
    zc = (p * np.exp(1j * 2 * np.pi * thg / 24)).sum(1)
    cm = np.mod(np.angle(zc) * 24 / (2 * np.pi), 24)
    assert abs(cm[0] - 23.75) < 0.26 or abs(cm[0] - 0.25) < 0.26
    log("  wraparound circular mean: PASS")
    # 5) determinism
    outs = []
    for _ in range(2):
        mm = KATv2AnchoredNBCorrected(Xd, sfv, phi, list("ABCDE"))
        mm.init_from_data()
        mm.mstep(mm.estep()[0])
        outs.append(mm.predict()[0].copy())
    assert np.max(np.abs(outs[0] - outs[1])) < 1e-10
    log("  determinism <1e-10: PASS")
    # 6) mstep must actually improve the q-weighted objective
    #    (regression test for the gradient-sign bug that froze LBFGS)
    mm = KATv2AnchoredNBCorrected(Xd, sfv, phi, list("ABCDE"))
    mm.init_from_data()
    q0, _ = mm.estep()
    def _obj():
        with torch.no_grad():
            tot = 0.0
            for k in range(mm.K):
                tot += float((q0[:, k:k + 1]
                              * mm._nb_ll_chunk(mm.theta[k])).sum()) \
                       - float(L2_A * (mm.a ** 2).sum()) / mm.K
            return tot
    o0 = _obj()
    mm.mstep(q0)
    o1 = _obj()
    assert o1 > o0 + 1e-6, f"mstep failed to improve objective {o0}->{o1}"
    log(f"  mstep improves objective: {o0:.4f}->{o1:.4f} PASS")
    (A / "03_MODEL_FREEZE/UNIT_TESTS.json").write_text(json.dumps({
        "NB_vs_closed_form": "PASS",
        "chunked_vs_closed_form_max": dmax,
        "offset_is_size_factor": "PASS",
        "wraparound": "PASS", "determinism": "PASS",
        "mstep_improves_objective": {"before": o0, "after": o1,
                                     "PASS": True},
        "memory_note": "memory-lean v2 (no pandas/scipy; per-chunk "
                       "backward) under 2GB cgroup; mstep closure sign "
                       "fixed 2026-10-07 (see FAILED_ATTEMPTS/"
                       "MSTEP_SIGN_BUG_20261007)"}, indent=2))
    log("selftest PASS")


# ================================================================ simulate
def simulate():
    prior = json.loads((FREEZE / "ACROPHASE_PRIOR_FREEZE.json").read_text())
    genes = list(prior["priors"].keys())
    phi = np.array([prior["priors"][g]["loc_rad"] for g in genes])
    sf_real = np.load(SIZE_FACTOR)
    rng = np.random.default_rng(20261005)
    rows = []
    for depth, tag in [(1.0, "exact_moderate"), (0.3, "low"),
                       (0.1, "very_low")]:
        N = 2000
        theta = rng.uniform(0, 2 * np.pi, N)
        sf = rng.choice(sf_real, N) * depth
        # match task10 sim median mu on the new offset scale:
        # task10 used lib(median 8) * 1.5 => median mu 12; here
        # sf(median 1) * exp(m) => m = log(12)
        m = np.full(len(genes), np.log(12.0))
        Amp = np.full(len(genes), 0.35)
        alpha = np.full(len(genes), 0.05)
        cols = []
        for g in range(len(genes)):
            mu = np.exp(np.log(sf) + m[g]
                        + Amp[g] * np.cos(theta - phi[g]))
            r = 1 / alpha[g]
            cols.append(rng.negative_binomial(r, r / (r + mu)))
        X = np.stack(cols, axis=1).astype(float)
        mod = KATv2AnchoredNBCorrected(X, sf, phi, genes)
        mod.init_from_data()
        mod.fit()
        mean_h, _, R, _, _, _, _ = mod.predict()
        th_h = np.mod(theta * 24 / (2 * np.pi), 24)
        d = np.abs(mean_h - th_h) % 24
        err = np.minimum(d, 24 - d)
        rows.append({"condition": tag, "depth": depth,
                     "median_AE_h": round(float(np.median(err)), 4),
                     "mean_AE_h": round(float(err.mean()), 4),
                     "within_2h": round(float((err <= 2).mean()), 4),
                     "mean_posterior_R": round(float(R.mean()), 4)})
        log(f"  sim {tag}: medianAE={rows[-1]['median_AE_h']}h "
            f"R={rows[-1]['mean_posterior_R']}")
        del mod
    write_csv(A / "04_SYNTHETIC_CHECK/SIMULATION_RESULTS.csv",
              ["condition", "depth", "median_AE_h", "mean_AE_h",
               "within_2h", "mean_posterior_R"],
              [[r["condition"], r["depth"], r["median_AE_h"],
                r["mean_AE_h"], r["within_2h"], r["mean_posterior_R"]]
               for r in rows])
    med0 = rows[0]["median_AE_h"]
    ok = med0 < 2.0 and rows[0]["median_AE_h"] <= rows[1]["median_AE_h"] \
        <= rows[2]["median_AE_h"]
    (A / "04_SYNTHETIC_CHECK/SIM_VERDICT.json").write_text(json.dumps(
        {"results": rows, "graceful_degradation": bool(ok),
         "exact_model_recovers_phase": bool(med0 < 2.0)}, indent=2))
    assert med0 < 2.0, "STOP_CORRECTED_MODEL_IMPLEMENTATION_FAILURE"
    log(f"simulation PASS (graceful={ok})")


# ================================================================ fit
def fit(out_dir, tag, perturb=None):
    out_dir = A / out_dir
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"STOP_DIRTY_OUTPUT: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    prior = json.loads((FREEZE / "ACROPHASE_PRIOR_FREEZE.json").read_text())
    genes = list(prior["priors"].keys())
    phi = np.array([prior["priors"][g]["loc_rad"] for g in genes])
    X = load_dense_counts_npz(FREEZE / "anchored_counts.npz")
    sf = np.load(SIZE_FACTOR)
    mod = KATv2AnchoredNBCorrected(X, sf, phi, genes)
    del X
    init = mod.init_from_data()
    cfg = {"model": "KATv2-B0_ANCHORED_NB_CORRECTED_OFFSET", "tag": tag,
           "genes": genes, "K": K_GRID, "dtype": "float64",
           "threads": 1, "implementation": "memory-lean v2",
           "size_factor": "TOTAL_UMI_ALL_GENES / median (frozen 10A)",
           "size_factor_sha256": sha(SIZE_FACTOR),
           "phi_fixed": {g: prior["priors"][g]["loc_rad"] for g in genes},
           "init": init,
           "m_init_note": "m_g = log((gene total+1)/sum(size_factor)); "
                          "median(sf)=1 so mesors live on the size-factor "
                          "scale, coherent with the corrected offset",
           "optimizer": "EM + full-batch L-BFGS(strong_wolfe, max_iter=25)",
           "max_em": MAX_EM, "rel_tol": REL_TOL,
           "phase_tol_h": PHASE_TOL_H,
           "patience": PATIENCE, "L2_a": L2_A,
           "alpha_bounds": ALPHA_BOUNDS,
           "point_estimate_primary": "circular posterior mean",
           "perturbation": perturb,
           "script_sha256": sha(Path(__file__).resolve()),
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    cfg_path = out_dir / "MODEL_CONFIG.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    write_csv(out_dir / "INITIAL_PARAMETERS.csv",
              ["gene", "m", "a", "log_alpha"],
              [[g, init[g]["m"], init[g]["a"], init[g]["log_alpha"]]
               for g in genes])
    log(f"=== corrected B0 fit [{tag}] START (N={mod.N}, G={mod.G}, "
        f"perturb={perturb}) ===")
    _, hist = mod.fit(perturb=perturb)
    write_csv(out_dir / "TRAIN_HISTORY.csv",
              ["cycle", "obj_before", "obj_after", "rel_improve",
               "median_dphase_h"],
              [[h["cycle"], h["obj_before"], h["obj_after"],
                h["rel_improve"], h["median_dphase_h"]] for h in hist])
    mean_h, map_h, R, v, ent, ci, q = mod.predict()
    np.savez_compressed(out_dir / "PHASE_POSTERIOR.npz",
                        posterior=q, theta_h=mod.theta_h)
    ids, bsids = read_cell_ids()
    write_csv(out_dir / "PREDICTIONS_BLINDED.csv.gz",
              ["cell_id", "blind_sample_id", "phase_mean_h",
               "phase_MAP_h", "posterior_R", "circular_var",
               "posterior_entropy", "ci95_width_h"],
              zip(ids, bsids, mean_h, map_h, R, v, ent, ci),
              compress=True)
    write_csv(out_dir / "GENE_PARAMETERS.csv",
              ["gene", "m_g", "a_g", "A_g", "log_alpha_g",
               "phi_fixed_rad"],
              [[g, float(mod.m[i].detach()), float(mod.a[i].detach()),
                float(softplus(mod.a[i]).detach()),
                float(mod.la[i].detach()), float(phi[i])]
               for i, g in enumerate(genes)])
    runtime = time.perf_counter() - t0
    frz = {"model": "KATv2-B0-corrected", "tag": tag,
           "perturbation": perturb,
           "prediction_sha256": sha(out_dir / "PREDICTIONS_BLINDED.csv.gz"),
           "config_sha256": sha(cfg_path),
           "gene_param_sha256": sha(out_dir / "GENE_PARAMETERS.csv"),
           "posterior_sha256": sha(out_dir / "PHASE_POSTERIOR.npz"),
           "script_sha256": cfg["script_sha256"],
           "geneset_sha256": sha(FREEZE / "GENESET_FREEZE.json"),
           "size_factor_sha256": sha(SIZE_FACTOR),
           "phi_prior_sha256": sha(FREEZE / "ACROPHASE_PRIOR_FREEZE.json"),
           "em_cycles": int(len(hist)), "runtime_sec": round(runtime),
           "TRUE_ZT_READ": False, "PREDICTIONS_FROZEN": True,
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "MODEL_FREEZE.json").write_text(json.dumps(frz, indent=2))
    log(f"=== [{tag}] FROZEN ({len(hist)} cycles, {runtime:.0f}s, "
        f"pred={frz['prediction_sha256'][:16]}...) ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "simulate", "fit"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="RUN1")
    ap.add_argument("--perturb", type=float, default=None)
    a = ap.parse_args()
    torch.set_default_dtype(DTYPE)
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "simulate":
        simulate()
    else:
        fit(a.out, a.tag, a.perturb)


if __name__ == "__main__":
    main()
