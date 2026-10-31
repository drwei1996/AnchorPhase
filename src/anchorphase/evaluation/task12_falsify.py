"""Task12 external falsification-lite (confirmatory, blind):
A) 5 permuted-theta CPI controls
B) 5 expression/prevalence-matched random panels (blind preds)
C) wrong-anchor +8h CPI control
Usage: task12_falsify.py perm N | rot8 | matched
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task11b_cpi import (fit_genes_3col, nb_loglik_grid,
                         joint_posterior, circ_mean_h, THETA_H)
from task11b_utils import assemble_predictions
from task12_pipeline import (load_cohort, folds_for,
                             candidates_for, D, CORE17)

RR = Path(__file__).resolve().parents[1] / \
    "results/single_cell_phase_rebuild"


def prep(gse):
    X, c17, tot, sf, obs, A = load_cohort(gse)
    gene_names = np.array([str(g) for g in A.var.gene_name])
    keep, mean_c, prev = candidates_for(X, gene_names)
    cand = np.where(keep)[0]
    Xc = X[:, cand].tocsc()
    genes = gene_names[cand]
    q0 = np.load(D(gse) / "01_B0_BLINDED/PHASE_POSTERIOR.npz"
                 )["posterior"]
    theta = np.mod(circ_mean_h(q0) * 2 * np.pi / 24, 2 * np.pi)
    folds, _ = folds_for(obs)
    return Xc, genes, obs, np.log(sf), q0, theta, folds, \
        mean_c[cand], prev[cand]


def score_cpi_custom(Xc, log_s, q0, theta, folds):
    from task11b_run_smc import score_cpi
    params = {}
    for k in sorted(set(folds.values)):
        tr = folds.values != k
        m = np.empty(Xc.shape[1]); a = np.empty(Xc.shape[1])
        b = np.empty(Xc.shape[1]); al = np.empty(Xc.shape[1])
        for s0 in range(0, Xc.shape[1], 512):
            s1 = min(s0 + 512, Xc.shape[1])
            mm, aa, bb, aaal = fit_genes_3col(
                Xc[tr][:, s0:s1].toarray().astype(np.float64),
                log_s[tr], theta[tr])
            m[s0:s1], a[s0:s1], b[s0:s1], al[s0:s1] = mm, aa, bb, aaal
        params[k] = (m, a, b, al)
    cpi, per_fold, _ = score_cpi(Xc, log_s, q0, folds, params)
    return cpi, params


def perm(gse, seed):
    Xc, genes, obs, log_s, q0, theta, folds, _, _ = prep(gse)
    rng = np.random.default_rng(20261008 + int(seed) * 100)
    p = rng.permutation(len(theta))
    cpi, _ = score_cpi_custom(Xc, log_s, q0[p], theta[p], folds)
    out = D(gse) / "08_DIAGNOSTICS"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"perm{seed}_CPI.json").write_text(json.dumps({
        "top40_mean_cpi": float(np.sort(cpi)[::-1][:40].mean()),
        "mean_cpi": float(cpi.mean())}, indent=2))
    print(f"perm{seed}: top40="
          f"{np.sort(cpi)[::-1][:40].mean():.5f}")


def rot8(gse):
    Xc, genes, obs, log_s, q0, theta, folds, _, _ = prep(gse)
    theta_r = np.mod(theta + 8 * 2 * np.pi / 24, 2 * np.pi)
    cpi, _ = score_cpi_custom(Xc, log_s, q0, theta_r, folds)
    out = D(gse) / "08_DIAGNOSTICS"
    out.mkdir(parents=True, exist_ok=True)
    (out / "rot8_CPI.json").write_text(json.dumps({
        "top40_mean_cpi": float(np.sort(cpi)[::-1][:40].mean()),
        "mean_cpi": float(cpi.mean())}, indent=2))
    print(f"rot8: top40={np.sort(cpi)[::-1][:40].mean():.5f}")


def matched(gse):
    Xc, genes, obs, log_s, q0, theta, folds, mean_c, prev = \
        prep(gse)
    panels = json.loads((D(gse) / "02_CPI_BLINDED/"
                         "PANELS_PER_FOLD.json").read_text())
    c2_panel = panels["C2"][0]
    gpos = {g: i for i, g in enumerate(genes)}
    panel_idx = [gpos[g] for g in c2_panel]
    strat = (pd.qcut(pd.Series(mean_c), 4, labels=False)
             .values * 16
             + pd.qcut(pd.Series(prev), 4, labels=False)
             .values * 4)
    # load fold params from cache
    from task11b_utils import load_fold_cache
    loaded, reason = load_fold_cache(
        D(gse) / "02_CPI_BLINDED/FOLD_CACHE_v1.npz")
    assert loaded is not None, reason
    ks = loaded["fold_ids"]
    params = loaded["params"]
    panel_strata = strat[panel_idx]
    out = D(gse) / "08_DIAGNOSTICS"
    for rep in range(5):
        rng = np.random.default_rng(20261008 + 500 + rep)
        sel = []
        for st, cnt in zip(*np.unique(panel_strata,
                                      return_counts=True)):
            pool = np.where(strat == st)[0]
            sel.extend(rng.choice(pool, size=cnt,
                                  replace=False).tolist())
        frames = []
        for k in ks:
            te_mask = folds.values == k
            te = np.where(te_mask)[0]
            m, a, b, al = params[k]
            Xte = Xc[te_mask][:, sel].toarray().astype(np.float64)
            LL = nb_loglik_grid(Xte, log_s[te_mask], m[sel],
                                a[sel], b[sel], al[sel]).sum(1)
            q, mh, R, H, ci = joint_posterior(q0[te_mask], LL,
                                              alpha=1.0)
            frames.append(assemble_predictions(
                te, q, mh, R, H, ci, obs.cell_id.values,
                obs.blind_sample_id.values, theta_grid_h=THETA_H))
        df = pd.concat(frames, ignore_index=True)
        p = out / f"MATCHED_PANEL_{rep}_PRED.csv.gz"
        df.to_csv(p, index=False, compression="gzip")
        (out / f"MATCHED_PANEL_{rep}.json").write_text(
            json.dumps({"rep": rep,
                        "genes": [str(genes[i]) for i in sel]},
                       indent=2))
        print(f"matched panel {rep} frozen")


if __name__ == "__main__":
    cmd = sys.argv[1]
    gse = sys.argv[2]
    {"perm": lambda: perm(gse, sys.argv[3]),
     "rot8": lambda: rot8(gse),
     "matched": lambda: matched(gse)}[cmd]()
