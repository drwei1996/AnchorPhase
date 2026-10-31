"""Task11 01_COMMON_ENGINE unit tests: exact B0 reduction, NB closed
form, per-component mstep improvement, B4 centering."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from katv2_engine import (KATv2Family, ARMS, softplus, inv_softplus,
                          circdist_h)

T11 = Path(__file__).resolve().parents[1] / (
    "results/single_cell_phase_rebuild/"
    "11_KAT_V2_MODEL_FAMILY_DEVELOPMENT_20261007")
OUT = T11 / "01_COMMON_ENGINE"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    results = {}
    rng = np.random.default_rng(20261007)
    N, G = 400, 6
    sf = rng.uniform(0.5, 2.0, N)
    phi = rng.uniform(0, 2 * np.pi, G)
    theta_t = rng.uniform(0, 2 * np.pi, N)
    X = rng.poisson(np.exp(np.log(sf)[:, None] + np.log(3.0)
                           + 0.3 * np.cos(theta_t[:, None]
                                          - phi[None, :])))
    arm = rng.integers(0, 6, G)
    lin = rng.integers(0, 3, N)
    z = rng.normal(size=(N, 4))
    smp = rng.integers(0, 5, N)

    # 1) NB closed form
    mod = KATv2Family(X, sf, phi, list("ABCDEF"))
    import math
    with torch.no_grad():
        mod.m[0] = 1.2
        mod.a[0] = inv_softplus(torch.tensor(0.4, dtype=torch.float64))
        mod.la[0] = np.log(0.3)
        A_ref = float(softplus(mod.a[0]))
        ll = float(mod._nb_ll_chunk(mod.theta[7])[0, 0])
        x = float(X[0, 0]); k7 = float(mod.theta[7])
        mu = math.exp(math.log(sf[0]) + 1.2 + A_ref
                      * math.cos(k7 - phi[0]))
        r = 1 / 0.3
        ref = (math.lgamma(x + r) - math.lgamma(r) - math.lgamma(x + 1)
               + r * math.log(r) + x * math.log(mu)
               - (r + x) * math.log(r + mu))
    assert abs(ll - ref) < 1e-9
    results["nb_closed_form"] = "PASS"
    log("NB closed form: PASS")

    # 2) exact B0 reduction with all components at zero (frozen)
    mk = lambda **kw: KATv2Family(X, sf, phi, list("ABCDEF"),
                                  arm_of_gene=arm, lineage=lin,
                                  n_lineage=3, sample=smp, n_sample=5,
                                  z=z, **kw)
    b0a, b0b = mk(), mk(b1_lineage=True, b2_state=True, b3_wave=True,
                        b4_strength=True, b5_sample=True)
    b0a.init_from_data()
    b0b.init_from_data()
    for p in (b0b.uM, b0b.uA, b0b.Wm, b0b.Wa, b0b.a2, b0b.raw_c,
              b0b.e5, b0b.f5):
        p.requires_grad_(False)
    qa, oa = b0a.estep()
    qb, ob = b0b.estep()
    assert abs(oa - ob) < 1e-9 and float((qa - qb).abs().max()) < 1e-12
    assert abs(b0a.objective(qa) - b0b.objective(qb)) < 1e-9
    results["b0_exact_reduction_frozen_zero"] = "PASS"
    log("B0 exact reduction (components frozen at 0): PASS")

    # 3) per-component mstep improvement (components trainable)
    for flag in [{}, {"b1_lineage": True}, {"b1_free": True},
                 {"b2_state": True}, {"b3_wave": True},
                 {"b4_strength": True}, {"b5_sample": True}]:
        m = mk(**flag)
        m.init_from_data()
        q, _ = m.estep()
        o0 = m.objective(q)
        m.mstep(q)
        o1 = m.objective(q)
        assert o1 > o0 + 1e-6, (flag, o0, o1)
        results[f"mstep_improves[{','.join(flag) or 'b0'}"] = \
            f"{o0:.2f}->{o1:.2f} PASS"
        log(f"mstep improves ({','.join(flag) or 'B0'}): PASS")

    # 4) B4 centering + positive strength
    m = mk(b4_strength=True)
    m.init_from_data()
    m.fit(max_em=8, verbose=False)
    c = m.predict()["clock_strength_c"]
    assert (c > 0).all()
    assert abs(np.mean(np.log(c))) < 1e-8
    results["b4_centering_mean_log_c"] = "PASS"
    log("B4 c>0 and mean(log c)=0: PASS")

    # 5) B1 null-world shrinkage sanity (S0-like data, 3 EM cycles)
    m = mk(b1_lineage=True)
    m.init_from_data()
    m.fit(max_em=6, verbose=False)
    s = m.component_summaries()
    results["b1_nullworld_uM_max_abs"] = s["uM_max_abs"]
    results["b1_nullworld_uA_max_abs"] = s["uA_max_abs"]
    log(f"B1 null-world uM max {s['uM_max_abs']:.3f}, "
        f"uA max {s['uA_max_abs']:.3f} (info only)")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ENGINE_UNIT_TESTS.json").write_text(
        json.dumps(results, indent=2))
    log("ALL ENGINE TESTS PASS")


if __name__ == "__main__":
    main()
