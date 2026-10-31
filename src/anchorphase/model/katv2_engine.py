"""KATv2 canonical engine: ONE model family B0..B7 via component flags.

All models share: NB likelihood on raw UMI, corrected whole-transcriptome
median-1 size-factor offset, fixed external phi_g (anchored genes),
48-bin theta grid, exact grid E-step, full-batch L-BFGS M-step
(strong_wolfe, float64, chunked closure over K), softplus amplitudes.

Components (flags; all reduce EXACTLY to B0 at zero limits):
  b1_lineage  uM[lin,arm], uA[lin,arm]     hierarchical N(0,0.25) shrinkage
  b1_free     per-lineage free m',A'       overparam control (no arms)
  b2_state    hM=z@Wm, hA=z@Wa (6 arms)    strong N(0,0.25) on W
  b3_wave     A2_g cos(2t - 2 phi_g)       strong N(0,0.05) on a2
  b4_strength c_i=exp(raw-mean(raw))       weak N(0,0.5) on raw
  b5_sample   b=e[s]*f[g] low-rank 1       strong N(0,0.25) on e,f
  b6 info genes carried via gene set extension (phi frozen externally)

Six-arm order: EBOX_POS, EBOX_NEG, RORE_ACT, RORE_REP, DBOX_ACT, DBOX_REP
"""
import math
import time

import numpy as np
import torch

# CPU quota on this instance is 15 cores; unpinned torch OMP threads
# (default = nproc) thrash the quota and stall every fit. Pin to 1
# thread per process; parallelism comes from separate processes.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

DTYPE = torch.float64
K_GRID = 48
ARMS = ["EBOX_POS", "EBOX_NEG", "RORE_ACT", "RORE_REP",
        "DBOX_ACT", "DBOX_REP"]

# preregistered penalty weights w = 1/(2 sigma^2)
W_U = 1.0 / (2 * 0.25 ** 2)     # B1 uM/uA, B2 W, B5 e/f
W_A2 = 1.0 / (2 * 0.05 ** 2)    # B3 second harmonic
W_C = 1.0 / (2 * 0.5 ** 2)      # B4 log c (weak)
W_A6 = 1.0 / (2 * 0.05 ** 2)    # B6 information-gene amplitude
L2_A = 1e-3                      # B0 legacy weak L2 on a_g
ALPHA_BOUNDS = (float(np.log(1e-3)), float(np.log(1e3)))
A0 = 0.05


def softplus(x):
    return torch.nn.functional.softplus(x)


def inv_softplus(y):
    return torch.log(torch.expm1(torch.clamp(y, min=1e-6)))


def circdist_h(a, b):
    d = np.abs(np.mod(a, 24.0) - np.mod(b, 24.0))
    return np.minimum(d, 24.0 - d)


class KATv2Family:
    """Canonical engine. X (N,G) float tensor, sf (N,), phi (G,),
    genes list, arm_of_gene (G,) int or None."""

    def __init__(self, X, sf, phi, genes, arm_of_gene=None,
                 lineage=None, n_lineage=1, sample=None, n_sample=1,
                 z=None, phi2_mode="2phi",
                 b1_lineage=False, b1_free=False, b2_state=False,
                 b3_wave=False, b4_strength=False, b5_sample=False,
                 K=K_GRID, device="cpu", info_gene_mask=None):
        self.genes = list(genes)
        self.G = len(self.genes)
        self.N = X.shape[0]
        self.K = K
        self.device = device
        self.flags = dict(b1_lineage=b1_lineage, b1_free=b1_free,
                          b2_state=b2_state, b3_wave=b3_wave,
                          b4_strength=b4_strength, b5_sample=b5_sample)
        self.X = torch.as_tensor(X, dtype=DTYPE, device=device)
        self._lgamX1 = torch.lgamma(self.X + 1)
        sf = torch.as_tensor(np.asarray(sf, float), dtype=DTYPE,
                             device=device)
        assert bool((sf > 0).all())
        self.log_s = torch.log(sf)
        self.phi = torch.as_tensor(np.asarray(phi, float), dtype=DTYPE,
                                   device=device)
        self.phi2 = 2.0 * self.phi if phi2_mode == "2phi" else \
            torch.as_tensor(np.asarray(phi, float) * 0, dtype=DTYPE)
        th = np.linspace(0, 2 * np.pi, K, endpoint=False)
        self.theta = torch.as_tensor(th, dtype=DTYPE, device=device)
        self.theta_h = th * 24 / (2 * np.pi)
        # gene params
        self.m = torch.zeros(self.G, dtype=DTYPE, device=device,
                             requires_grad=True)
        self.a = torch.zeros(self.G, dtype=DTYPE, device=device,
                             requires_grad=True)
        self.la = torch.zeros(self.G, dtype=DTYPE, device=device,
                              requires_grad=True)
        self.info_mask = (torch.as_tensor(info_gene_mask, dtype=DTYPE,
                                          device=device)
                          if info_gene_mask is not None else None)
        # components
        self.arm = (torch.as_tensor(arm_of_gene, dtype=torch.long,
                                    device=device)
                    if arm_of_gene is not None else None)
        if lineage is None:
            lineage = np.zeros(self.N, dtype=np.int64)
        self.lin = torch.as_tensor(lineage, dtype=torch.long,
                                   device=device)
        self.L = int(n_lineage)
        if sample is None:
            sample = np.zeros(self.N, dtype=np.int64)
        self.smp = torch.as_tensor(sample, dtype=torch.long,
                                   device=device)
        self.S = int(n_sample)
        self.z = (torch.as_tensor(z, dtype=DTYPE, device=device)
                  if z is not None else None)
        D = 0 if self.z is None else self.z.shape[1]
        Z = lambda *s: torch.zeros(*s, dtype=DTYPE, device=device,
                                   requires_grad=True)
        self.uM = Z(self.L, 6) if b1_lineage else None
        self.uA = Z(self.L, 6) if b1_lineage else None
        self.mL = Z(self.L, self.G) if b1_free else None
        self.aL = Z(self.L, self.G) if b1_free else None
        self.Wm = Z(D, 6) if (b2_state and D > 0) else None
        self.Wa = Z(D, 6) if (b2_state and D > 0) else None
        self.a2 = Z(self.G) if b3_wave else None
        self.raw_c = (Z(self.N) if b4_strength else None)
        self.e5 = Z(self.S) if b5_sample else None
        self.f5 = Z(self.G) if b5_sample else None

    # ------------------------------------------------------------ core eta
    def _mod_m(self):
        mm = self.m[None, :]
        if self.flags["b1_lineage"]:
            mm = mm + self.uM[self.lin][:, self.arm]
        if self.flags["b1_free"]:
            mm = mm + self.mL[self.lin]
        if self.flags["b2_state"]:
            mm = mm + (self.z @ self.Wm)[:, self.arm]
        if self.flags["b5_sample"]:
            mm = mm + self.e5[self.smp][:, None] * self.f5[None, :]
        return mm

    def _amp(self):
        A = softplus(self.a)[None, :]
        log_mod = None
        if self.flags["b1_lineage"]:
            m = self.uA[self.lin][:, self.arm]
            log_mod = m if log_mod is None else log_mod + m
        if self.flags["b1_free"]:
            m = self.aL[self.lin]
            log_mod = m if log_mod is None else log_mod + m
        if self.flags["b2_state"]:
            m = (self.z @ self.Wa)[:, self.arm]
            log_mod = m if log_mod is None else log_mod + m
        if log_mod is not None:
            # combined log-amplitude modifiers can run away when
            # several components stack (e.g. B1+B2); clamp to a
            # generous biological range to keep exp() finite
            A = A * torch.exp(torch.clamp(log_mod, -3.0, 3.0))
        if self.flags["b4_strength"]:
            c = torch.exp(torch.clamp(self.raw_c - self.raw_c.mean(),
                                      -3.0, 3.0))
            A = A * c[:, None]
        return A

    def _nb_ll_chunk(self, theta_k):
        alpha = torch.exp(torch.clamp(self.la, *ALPHA_BOUNDS))
        cos1 = torch.cos(theta_k - self.phi)[None, :]
        eta = self._mod_m() + self._amp() * cos1
        if self.flags["b3_wave"]:
            # a2 IS the amplitude, clamped >= 0: exact zero limit at
            # a2=0 (softplus(0)=ln2 would NOT reduce to B0)
            eta = eta + torch.clamp(self.a2, 0.0, 2.0)[None, :] * \
                torch.cos(2 * theta_k - self.phi2)[None, :]
        log_mu = self.log_s[:, None] + torch.clamp(eta, -30.0, 30.0)
        mu = torch.exp(log_mu)
        r = 1.0 / alpha
        return (r * (torch.log(r) - torch.log(r + mu))
                + self.X * (log_mu - torch.log(r + mu))
                + torch.lgamma(self.X + r) - torch.lgamma(r)
                - self._lgamX1)

    # ------------------------------------------------------------ penalty
    def _penalty(self):
        pen = L2_A * (self.a ** 2).sum()
        if self.info_mask is not None:
            pen = pen + W_A6 * (softplus(self.a) ** 2
                                * self.info_mask).sum()
        if self.uM is not None:
            pen = pen + W_U * ((self.uM ** 2).sum()
                               + (self.uA ** 2).sum())
        if self.mL is not None:
            pen = pen + W_U * ((self.mL ** 2).sum()
                               + (self.aL ** 2).sum())
        if self.Wm is not None:
            pen = pen + W_U * ((self.Wm ** 2).sum()
                               + (self.Wa ** 2).sum())
        if self.a2 is not None:
            pen = pen + W_A2 * (self.a2 ** 2).sum()
        if self.raw_c is not None:
            pen = pen + W_C * (self.raw_c ** 2).sum()
        if self.e5 is not None:
            pen = pen + W_U * ((self.e5 ** 2).sum()
                               + (self.f5 ** 2).sum())
        return pen

    # ------------------------------------------------------------ EM
    @torch.no_grad()
    def estep(self):
        parts = [self._nb_ll_chunk(self.theta[k]).sum(dim=1)
                 for k in range(self.K)]
        log_joint = torch.stack(parts, dim=1) + math.log(1.0 / self.K)
        log_q = log_joint - torch.logsumexp(log_joint, dim=1,
                                            keepdim=True)
        return torch.exp(log_q), float(log_joint.sum())

    def mstep(self, q, max_iter=25, lr=0.5):
        params = [p for p in (self.m, self.a, self.la, self.uM,
                              self.uA, self.mL, self.aL, self.Wm,
                              self.Wa, self.a2, self.raw_c, self.e5,
                              self.f5) if p is not None]
        opt = torch.optim.LBFGS(params, lr=lr, max_iter=max_iter,
                                tolerance_grad=1e-10,
                                tolerance_change=1e-14,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss_total = 0.0
            for k in range(self.K):
                term = (q[:, k:k + 1]
                        * self._nb_ll_chunk(self.theta[k])).sum()
                chunk_loss = (self._penalty() / self.K) - term
                chunk_loss.backward()
                loss_total += float(chunk_loss.detach())
            return torch.tensor(loss_total, dtype=DTYPE)
        opt.step(closure)

    @torch.no_grad()
    def objective(self, q=None):
        if q is None:
            q, _ = self.estep()
        tot = sum(float((q[:, k:k + 1]
                         * self._nb_ll_chunk(self.theta[k])).sum())
                  for k in range(self.K))
        return tot - float(self._penalty())

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
            for p in (self.uM, self.uA, self.mL, self.aL, self.Wm,
                      self.Wa, self.a2, self.raw_c, self.e5, self.f5):
                if p is not None:
                    p.zero_()
        return {g: {"m": float(self.m[i].detach()),
                    "a": float(self.a[i].detach()),
                    "log_alpha": float(self.la[i].detach())}
                for i, g in enumerate(self.genes)}

    def fit(self, max_em=50, rel_tol=1e-6, phase_tol_h=0.01,
            patience=3, verbose=False, log_fn=print, init=True,
            perturb=None, perturb_seed=20261007):
        init_params = self.init_from_data() if init else None
        if perturb is not None:
            g = torch.Generator().manual_seed(perturb_seed)
            with torch.no_grad():
                for p in [self.m, self.a, self.la, self.uM, self.uA,
                          self.mL, self.aL, self.Wm, self.Wa, self.a2,
                          self.raw_c, self.e5, self.f5]:
                    if p is None:
                        continue
                    delta = torch.randn(p.shape, generator=g,
                                        dtype=DTYPE)
                    p.add_(delta / delta.norm() * perturb)
        history, prev_mean, stall = [], None, 0
        for cyc in range(max_em):
            q, obj0 = self.estep()
            self.mstep(q)
            mean_h = self.predict()["phase_mean_h"]
            _, obj1 = self.estep()
            rel = abs(obj1 - obj0) / (abs(obj0) + 1e-12)
            dph = 99.0 if prev_mean is None else float(np.median(
                circdist_h(mean_h, prev_mean)))
            history.append({"cycle": cyc, "obj_before": obj0,
                            "obj_after": obj1, "rel_improve": rel,
                            "median_dphase_h": dph})
            if verbose:
                log_fn(f"    EM {cyc}: obj {obj0:.1f}->{obj1:.1f} "
                       f"rel={rel:.1e} dph={dph:.4f}")
            prev_mean = mean_h
            if rel < rel_tol and dph < phase_tol_h:
                stall += 1
                if stall >= patience:
                    break
            else:
                stall = 0
        return init, history

    # ------------------------------------------------------------ predict
    @torch.no_grad()
    def predict(self):
        q, _ = self.estep()
        qn = q.cpu().numpy()
        th_h = self.theta_h
        z = (qn * np.exp(1j * 2 * np.pi * th_h / 24.0)).sum(axis=1)
        mean_h = np.mod(np.angle(z) * 24 / (2 * np.pi), 24.0)
        R = np.abs(z) / np.maximum(qn.sum(axis=1), 1e-12)
        map_h = th_h[qn.argmax(axis=1)]
        ent = -np.sum(qn * np.log(qn + 1e-300), axis=1) / np.log(self.K)
        out = {"phase_mean_h": mean_h, "phase_MAP_h": map_h,
               "posterior_R": R, "circular_var": 1 - R ** 2,
               "posterior_entropy": ent, "posterior": qn}
        if self.raw_c is not None:
            c = torch.exp(self.raw_c - self.raw_c.mean())
            out["clock_strength_c"] = c.cpu().numpy()
        return out

    # ------------------------------------------------------------ extras
    def component_summaries(self):
        s = {}
        if self.uM is not None:
            s["uM_max_abs"] = float(self.uM.detach().abs().max())
            s["uA_max_abs"] = float(self.uA.detach().abs().max())
            s["uM_matrix"] = self.uM.detach().cpu().numpy().tolist()
            s["uA_matrix"] = self.uA.detach().cpu().numpy().tolist()
        if self.mL is not None:
            s["mL_max_abs"] = float(self.mL.detach().abs().max())
        if self.Wm is not None:
            s["W_max_abs"] = float(max(self.Wm.detach().abs().max(), self.Wa.detach().abs().max()))
        if self.a2 is not None:
            A2 = torch.clamp(self.a2, 0.0, 2.0).detach()
            s["A2_p95"] = float(A2.quantile(0.95))
            s["A2_mean"] = float(A2.mean())
        if self.raw_c is not None:
            lc = (self.raw_c - self.raw_c.mean()).detach().cpu().numpy()
            s["log_c_sd"] = float(np.std(lc))
            s["log_c_p95"] = float(np.percentile(np.abs(lc), 95))
        if self.e5 is not None:
            s["e5_max_abs"] = float(self.e5.detach().abs().max())
        return s
