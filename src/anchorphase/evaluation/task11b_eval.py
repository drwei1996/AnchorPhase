"""Task11B evaluation + model selection + final freeze.
Reads ZT ONLY after all blind prediction freezes verify.
Outputs: TABLES T1, T5-T10; FIGURES F1-F14; 11_FINAL_FREEZE;
TASK11B_FINAL_REPORT.txt."""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RR = ROOT / "results/single_cell_phase_rebuild"
SMC = RR / "02B_BASELINE_REPRODUCTION_STRICT"
T11 = RR / "11_KAT_V2_MODEL_FAMILY_DEVELOPMENT_20261007"
T11B = RR / "11B_KAT_V2_SAFE_PHASE_INFORMATION_20261007"
PRIV = SMC / "03_BLINDED_SAMPLE_ID_MAP_PRIVATE.csv"
POOLPRIV = T11 / ("11_REALDATA/00_POOLED_FREEZE/"
                  "POOLED_BLINDED_MAP_PRIVATE.csv")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def d24(a, b):
    d = np.abs(np.mod(a, 24.0) - np.mod(b, 24.0))
    return np.minimum(d, 24.0 - d)


SMC_MODELS = {
    "B0": "01_B0_REFERENCE", "B6": "02_B6_REFERENCE",
    "C1_CPI": "04_C1_CPI", "C2_SAFE_CPI": "05_C2_SAFE_CPI",
    "A4_no_red": "09_ABLATIONS/A4_C2_no_redundancy",
    "A5_no_cov": "09_ABLATIONS/A5_C2_no_coverage",
    "A6_alpha1": "09_ABLATIONS/A6_C2_alpha1",
    "C2_perturb": "09_ABLATIONS/C2_PERTURB1e-4"}
POOLED_MODELS = {
    "C3_full": "06_C3_SAFE_CPI_LS",
    "C3_pooled": "09_ABLATIONS/A8_C3_POOLED",
    "C3_lineage": "09_ABLATIONS/A9_C3_LIN",
    "C3_state": "09_ABLATIONS/A10_C3_STATE"}


def gate():
    entries = {}
    for name, d in {**SMC_MODELS,
                    **POOLED_MODELS}.items():
        p = T11B / d / "PREDICTIONS_BLINDED.csv.gz"
        if not p.exists():
            entries[name] = "MISSING (ran later or skipped by " \
                            "futility rule)"
            continue
        h = (T11B / d / "PREDICTION_SHA256.txt").read_text().split()[
            0]
        import hashlib
        hh = hashlib.sha256(p.read_bytes()).hexdigest()
        assert hh == h, f"hash mismatch {name}"
        entries[name] = h
    return entries


def eval_smc(df, phase_col="phase_mean_h"):
    priv = pd.read_csv(PRIV, usecols=["blind_sample_id", "hidden_ZT"])
    m = df.merge(priv, on="blind_sample_id", validate="many_to_one")
    err = d24(m[phase_col], m.hidden_ZT)
    rows = []
    for _, g in m.groupby("blind_sample_id"):
        z = np.exp(1j * 2 * np.pi * g[phase_col].to_numpy() / 24)
        sp = float(np.mod(np.angle(z.mean()) * 24 / (2 * np.pi), 24))
        rows.append((float(g.hidden_ZT.iloc[0]), sp))
    sdf = pd.DataFrame(rows, columns=["true_ZT", "sp"])
    cerr = d24(sdf.sp, sdf.true_ZT)
    dd = np.arange(0, 24, 0.01)
    rot_s = np.array([np.median(d24(sdf.sp + x, sdf.true_ZT))
                      for x in dd])
    rot_c = np.array([np.median(d24(m[phase_col] + x,
                                    m.hidden_ZT)) for x in dd])
    ks = int(np.argmin(rot_s))
    # reflection test: does mapping phase -> -phase + c materially
    # improve sample AE? (a correctly anchored model should NOT
    # need reflection)
    dd = np.arange(0, 24, 0.01)
    refl = np.array([np.median(d24(-sdf.sp + x, sdf.true_ZT))
                     for x in dd])
    refl_best = float(refl.min())
    reflection_gain = float(np.median(cerr) - refl_best)
    reflection_required = bool(reflection_gain > 0.5)
    # circular CI coverage proxy: true phase within half the
    # reported smallest-arc 95% CI width of the predicted mean
    ci_cov = float((err <= df["ci95_width_h"].values / 2).mean()) \
        if "ci95_width_h" in df else np.nan
    from scipy.stats import spearmanr
    out = {"cell_median_AE": float(np.median(err)),
           "cell_mean_AE": float(err.mean()),
           "within_2h": float((err <= 2).mean()),
           "within_4h": float((err <= 4).mean()),
           "sample_median_AE": float(np.median(cerr)),
           "sample_mean_AE": float(cerr.mean()),
           "anchored_cell_median_AE": float(rot_c[int(
               np.argmin(rot_c))]),
           "oracle_rotation_h": float(dd[ks]),
           "oracle_sample_median_AE": float(rot_s[ks]),
           "anchor_penalty_h": float(np.median(cerr) - rot_s[ks]),
           "calib_spearman": float(spearmanr(
               m.posterior_R, err).statistic) if "posterior_R"
           in m else np.nan,
           "reflected_sample_median_AE": refl_best,
           "reflection_gain_h": reflection_gain,
           "reflection_required": reflection_required,
           "ci95_coverage_proxy": ci_cov}
    return out, m, sdf


def main():
    entries = gate()
    log(f"gate verified: {len(entries)} models")
    log("=== reading ZT (post-freeze) ===")

    results = {}
    merged = {}
    for name, d in SMC_MODELS.items():
        p = T11B / d / "PREDICTIONS_BLINDED.csv.gz"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        r, m, sdf = eval_smc(df)
        r["scope"] = "smc"
        results[name] = r
        merged[name] = m
    # pooled models
    privp = pd.read_csv(POOLPRIV,
                        usecols=["blind_sample_id", "hidden_ZT"])
    for name, d in POOLED_MODELS.items():
        p = T11B / d / "PREDICTIONS_BLINDED.csv.gz"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        m = df.merge(privp, on="blind_sample_id",
                     validate="many_to_one")
        err = d24(m.phase_mean_h, m.hidden_ZT)
        r = {"cell_median_AE": float(np.median(err)),
             "cell_mean_AE": float(err.mean()),
             "within_2h": float((err <= 2).mean()),
             "within_4h": float((err <= 4).mean()),
             "scope": "pooled"}
        for lin in ("smc", "fibroblast", "endothelial",
                    "macrophage"):
            e = d24(m[m.lineage == lin].phase_mean_h,
                    m[m.lineage == lin].hidden_ZT)
            r[f"AE_{lin}"] = float(np.median(e)) if len(e) else \
                np.nan
        results[name] = r
    # B0 pooled reference
    b0p = pd.read_csv(T11 / "11_REALDATA/B0_POOLED/"
                      "PREDICTIONS_BLINDED.csv.gz")
    m = b0p.merge(privp, on="blind_sample_id",
                  validate="many_to_one")
    err = d24(m.phase_mean_h, m.hidden_ZT)
    r = {"cell_median_AE": float(np.median(err)),
         "cell_mean_AE": float(err.mean()), "scope": "pooled_B0_ref"}
    for lin in ("smc", "fibroblast", "endothelial", "macrophage"):
        e = d24(m[m.lineage == lin].phase_mean_h,
                m[m.lineage == lin].hidden_ZT)
        r[f"AE_{lin}"] = float(np.median(e)) if len(e) else np.nan
    results["B0_POOLED_ref"] = r

    resdf = pd.DataFrame(results).T
    resdf.to_csv(T11B / "TABLES/T1_MODEL_PERFORMANCE.csv")

    # ---------- T3 selected panel + T4 fold manifest ----------
    panels = json.loads((T11B / "03_CPI_METHOD/"
                         "PANELS_PER_FOLD.json").read_text())
    scores = pd.read_csv(T11B / "TABLES/T2_CPI_GENE_SCORES.csv")
    rows3 = []
    for model in panels:
        for k, pnl in enumerate(panels[model]):
            for rank, g in enumerate(pnl):
                cpi = float(scores.loc[scores.gene == g, "CPI"]
                            .iloc[0]) if (scores.gene == g).any()                     else np.nan
                rows3.append({"model": model, "fold": k,
                              "rank": rank, "gene": g, "CPI": cpi})
    pd.DataFrame(rows3).to_csv(T11B / "TABLES/T3_SELECTED_PANEL.csv",
                               index=False)
    import hashlib
    t4 = pd.read_csv(T11B / ("07_SAMPLE_BLOCK_CROSSFIT/"
                             "SMC_LOSO_MANIFEST.csv"))
    t4["fold_test"] = t4.fold
    t4.to_csv(T11B / "TABLES/T4_SAMPLE_BLOCK_CROSSFIT.csv",
              index=False)
    # overlap across folds (selection stability)
    ov = []
    for model in panels:
        sets = [set(p) for p in panels[model]]
        import itertools
        for i, j in itertools.combinations(range(len(sets)), 2):
            ov.append({"model": model, "fold_i": i, "fold_j": j,
                       "overlap": len(sets[i] & sets[j])})
    pd.DataFrame(ov).to_csv(
        T11B / "TABLES/T4_PANEL_OVERLAP.csv", index=False)

    # ---------- T7 lineage ----------
    t7 = {n: {k: v for k, v in results[n].items()
              if k.startswith("AE_") or k in ("cell_median_AE",)}
          for n in results if any(k.startswith("AE_")
                                  for k in results[n])}
    pd.DataFrame(t7).T.to_csv(T11B / "TABLES/T7_LINEAGE_RESULTS.csv")
    print(resdf[["cell_median_AE", "cell_mean_AE",
                 "sample_median_AE"]].round(3).to_string()
          if "sample_median_AE" in resdf else resdf.round(3))

    # ---------- F2 evaluation ----------
    f2_res = []
    for f in sorted((T11B / "08_FALSIFICATION").glob(
            "F2_MATCHED_PANEL_*_PRED.csv.gz")):
        df = pd.read_csv(f)
        r, _, _ = eval_smc(df)
        f2_res.append(r["cell_median_AE"])
    if f2_res:
        (T11B / "TABLES/T5_FALSIFICATION.csv").parent.mkdir(
            exist_ok=True)
    # ---------- T5 falsification summary ----------
    perms = sorted((T11B / "08_FALSIFICATION").glob("perm*_CPI.json"))
    perm_stats = [json.loads(f.read_text()) for f in perms]
    rotfix = sorted((T11B / "08_FALSIFICATION").glob(
        "rotfix*_CPI.json"))
    rot_stats = [json.loads(f.read_text()) for f in rotfix]
    rotsym = sorted((T11B / "08_FALSIFICATION").glob(
        "rotsym*_CPI.json"))
    real_cpi = pd.read_csv(T11B / "TABLES/T2_CPI_GENE_SCORES.csv")
    f4 = json.loads((T11B / "08_FALSIFICATION/"
                     "F4_SAME_VS_CROSSFIT.json").read_text())
    t5 = {
        "real_CPI": {"mean": float(real_cpi.CPI.mean()),
                     "top40_mean": float(
                         real_cpi.CPI.nlargest(40).mean()),
                     "max": float(real_cpi.CPI.max()),
                     "n_gt_0.01": int((real_cpi.CPI > 0.01).sum())},
        "F1_permuted": {"n": len(perm_stats),
                        "mean_of_means": float(np.mean(
                            [p["mean_cpi"] for p in perm_stats])),
                        "max_of_top40s": float(np.max(
                            [p["top40_mean_cpi"] for p in
                             perm_stats]))},
        "F3_wrong_anchor": {f["arg"]: f for f in rot_stats},
        "F3_rotation_symmetry_check": [json.loads(f.read_text())
                                       for f in rotsym][:1],
        "F2_matched_panels": {"n": len(f2_res),
                              "AE_mean": float(np.mean(f2_res)),
                              "AE_min": float(np.min(f2_res))},
        "F4_same_vs_crossfit": f4}
    if perm_stats:
        pm = [p["top40_mean_cpi"] for p in perm_stats]
        t5["F1_verdict"] = "PASS" if float(
            real_cpi.CPI.nlargest(40).mean()) > 10 * max(pm) \
            else "FAIL"
    if rot_stats:
        rr_ = float(real_cpi.CPI.nlargest(40).mean())
        t5["F3_verdict"] = "PASS" if all(
            rr_ > 2 * f["top40_mean_cpi"] for f in rot_stats) \
            else "CHECK"
    if f2_res:
        c1_ae = results.get("C1_CPI", {}).get("cell_median_AE",
                                              np.nan)
        t5["F2_verdict"] = "PASS" if np.min(f2_res) > c1_ae \
            else "FAIL"
    pd.DataFrame([t5]).to_csv(T11B / "TABLES/T5_FALSIFICATION.csv",
                              index=False)
    (T11B / "TABLES/T5_FALSIFICATION.json").write_text(
        json.dumps(t5, indent=2, default=float))

    # ---------- T8 calibration ----------
    cal_rows = []
    for name in ("B0", "B6", "C1_CPI", "C2_SAFE_CPI"):
        if name not in merged:
            continue
        m = merged[name]
        err = d24(m.phase_mean_h, m.hidden_ZT)
        q = pd.qcut(m.posterior_R, 5, duplicates="drop")
        for cat, idx in m.groupby(q, observed=True).groups.items():
            ii = list(idx)
            cal_rows.append({
                "model": name, "R_quintile": str(cat),
                "median_AE": float(np.median(err[ii])),
                "mean_R": float(m.posterior_R[ii].mean()),
                "mean_entropy": float(m["entropy_nats" if
                                        "entropy_nats" in m
                                        else "posterior_entropy"]
                                      [ii].mean())
                if ("entropy_nats" in m or
                    "posterior_entropy" in m) else np.nan,
                "mean_ci95": float(m.ci95_width_h[ii].mean())
                if "ci95_width_h" in m else np.nan})
        cal_rows.append({"model": name, "R_quintile": "ALL",
                         "median_AE": float(np.median(err)),
                         "mean_R": float(m.posterior_R.mean()),
                         "mean_entropy": np.nan, "mean_ci95": np.nan,
                         "spearman_R_vs_AE": float(
                             pd.Series(err).corr(
                                 m.posterior_R,
                                 method="spearman"))})
    pd.DataFrame(cal_rows).to_csv(
        T11B / "TABLES/T8_UNCERTAINTY_CALIBRATION.csv", index=False)

    # ---------- T9 anchor ----------
    t9 = {n: {k: results[n].get(k) for k in
              ("cell_median_AE", "oracle_rotation_h",
               "oracle_sample_median_AE", "anchor_penalty_h")}
          for n in results if "oracle_rotation_h" in results[n]}
    pd.DataFrame(t9).T.to_csv(
        T11B / "TABLES/T9_ANCHOR_ORACLE_DECOMPOSITION.csv")

    # ---------- model selection (frozen rules) ----------
    b0 = results["B0"]["cell_median_AE"]
    b6 = results["B6"]["cell_median_AE"]
    sel_rows = []
    for name in ("C1_CPI", "C2_SAFE_CPI", "C3_full", "C3_lineage"):
        if name not in results:
            sel_rows.append({"model": name,
                             "decision": "NOT_RUN (futility)"})
            continue
        ae = results[name]["cell_median_AE"]
        step1 = ae < b0 - 0.05
        step5 = ae <= 1.710
        stab = "C2_perturb" in results and abs(
            results["C2_perturb"]["cell_median_AE"]
            - results["C2_SAFE_CPI"]["cell_median_AE"]) < 0.25
        if not step1:
            dec = "REJECT (not better than B0)"
        elif not (t5.get("F1_verdict") == "PASS"
                  and t5.get("F2_verdict") == "PASS"):
            dec = "REJECT (falsification failed)"
        elif not stab and name == "C2_SAFE_CPI":
            dec = "DOWNGRADE (stability)"
        elif step5:
            dec = "STRONG CANDIDATE"
        elif ae <= 1.800:
            dec = "ACCEPTABLE"
        else:
            dec = "WEAK"
        sel_rows.append({"model": name, "cell_AE": ae,
                         "keeps_80pct_B6_gain": step5,
                         "decision": dec})
    pd.DataFrame(sel_rows).to_csv(
        T11B / "TABLES/T10_MODEL_SELECTION_DECISION.csv",
        index=False)
    print(pd.DataFrame(sel_rows).to_string(index=False))
    json.dump({k: {kk: float(vv) if isinstance(vv, (int, float))
                   else vv for kk, vv in v.items()}
               for k, v in results.items()},
              open(T11B / "TABLES/T1_MODEL_PERFORMANCE.json", "w"),
              indent=2, default=str)
    log("evaluation complete; final freeze + report in "
        "task11b_freeze.py")


if __name__ == "__main__":
    main()
