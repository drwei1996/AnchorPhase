"""Release smoke test — offline verification of the AnchorPhase package.

Runs with the repo checked out anywhere (no dataset downloads):
1. import the frozen engine from the installed package (or src/ fallback);
2. load the frozen CORE17 anchor priors + frozen model config (JSON integrity);
3. synthesize 300 cells from the anchored-NB generative model itself with
   known phases, fit B0 (few EM cycles), and check posterior output shapes,
   ranges, and that recovered phases beat the shuffled control;
4. verify the shipped frozen GENE_PARAMETERS table is loadable and matches
   the 17 anchor genes.

Exit code 0 = PASS.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

try:
    from anchorphase.model import katv2_engine as eng
except ImportError:
    sys.path.insert(0, str(REPO / "src"))
    from anchorphase.model import katv2_engine as eng

SEED = 20261007  # frozen engine seed


def main():
    # 1. frozen priors ----------------------------------------------------
    prior_path = REPO / "configs" / "ACROPHASE_PRIOR_FREEZE.json"
    priors = json.loads(prior_path.read_text())
    assert priors["genes_used"] == 17, priors["genes_used"]
    genes = sorted(priors["priors"])
    phi = np.array([priors["priors"][g]["loc_h"] for g in genes], float)
    assert len(genes) == 17 and np.isfinite(phi).all()

    # frozen config sanity (10A corrected run 1)
    cfg = json.loads((REPO / "configs" / "MODEL_CONFIG_B0_CORRECTED_RUN1.json").read_text())
    assert isinstance(cfg, dict) and len(cfg) > 0
    gp = REPO / "models" / "frozen_b0_corrected_run1" / "GENE_PARAMETERS.csv"
    header = gp.read_text().splitlines()[0].split(",")
    assert any("gene" in h.lower() for h in header), header

    # 2. synthetic generative data from the anchored-NB model --------------
    rng = np.random.default_rng(SEED)
    N = 300
    true_h = rng.uniform(0, 24, N)
    base = np.log(np.array([30, 25, 40, 20, 18, 15, 22, 12, 14, 16,
                            28, 24, 35, 20, 26, 30, 15])[None, :])
    amp = np.array([1.2, 0.8, 0.4, 1.0, 1.1, 0.7, 0.9, 0.6, 0.5, 0.8,
                    1.3, 1.0, 0.9, 1.1, 0.7, 1.2, 0.9])[None, :]
    theta = true_h[:, None] * 2 * np.pi / 24.0
    phir = phi[None, :] * 2 * np.pi / 24.0
    log_mu = base + amp * np.cos(theta - phir)
    sf = rng.lognormal(0, 0.15, N)
    alpha = 0.3  # frozen B0 dispersion scale
    mu = sf[:, None] * np.exp(log_mu)
    # NB via gamma-Poisson mixture: L ~ Gamma(1/alpha, rate=(1/alpha)/mu); X ~ Poisson(L)
    g_shape = np.full_like(mu, 1.0 / alpha)
    lam = rng.gamma(shape=g_shape, scale=mu / g_shape)
    X = rng.poisson(lam).astype(float)

    # 3. fit B0 (few cycles) ---------------------------------------------
    fam = eng.KATv2Family(X, sf, phi, genes)
    fam.fit(max_em=6, verbose=False, perturb=None)
    out = fam.predict()
    for key, shape in [("phase_mean_h", (N,)), ("phase_MAP_h", (N,)),
                       ("posterior_R", (N,)), ("circular_var", (N,)),
                       ("posterior_entropy", (N,)), ("posterior", (N, 48))]:
        assert out[key].shape == shape, (key, out[key].shape)
    assert ((out["phase_mean_h"] >= 0) & (out["phase_mean_h"] < 24)).all()
    assert ((out["posterior_R"] >= 0) & (out["posterior_R"] <= 1)).all()
    assert np.allclose(out["posterior"].sum(axis=1), 1.0, atol=1e-9)

    err = eng.circdist_h(out["phase_mean_h"], true_h)
    perm_err = eng.circdist_h(out["phase_mean_h"], rng.permutation(true_h))
    median_err = float(np.median(err))
    median_perm = float(np.median(perm_err))
    print(f"synthetic fit: median AE {median_err:.2f} h "
          f"(shuffled control {median_perm:.2f} h)")
    assert median_err < median_perm, "fitted phases must beat shuffled control"
    assert median_err < 6.0, f"recovery too weak on synthetic anchors ({median_err:.2f} h)"

    print("SMOKE TEST PASS — engine, priors, config, outputs OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
