"""Task 10A stages 10-13: freeze gate -> unblind -> metrics -> paired
sample fairness -> final status. Reads the private ZT map ONLY after
the all-methods gate passes."""
import gzip
import hashlib
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
A = RR / ("10A_KAT_V2_B0_LIBRARY_OFFSET_CORRECTION_"
          "AND_FAIRNESS_20261007")
SMC = RR / "02B_BASELINE_REPRODUCTION_STRICT"
PRIV = SMC / "03_BLINDED_SAMPLE_ID_MAP_PRIVATE.csv"
TEMPO_PRED = A / ("09_COMPARATOR_RECOVERY/TEMPO_FROZEN_REFERENCE/"
                  "tempo_predictions_blinded.csv.gz")
TEMPO17_PRED = A / ("09_COMPARATOR_RECOVERY/TEMPO_SAME17/"
                    "tempo_same17_predictions_blinded.csv.gz")
RUNS = {"RUN1": "05_B0_CORRECTED_RUN1", "RUN2": "06_B0_CORRECTED_RUN2",
        "RUN3": "07_B0_CORRECTED_RUN3"}
PERTURBS = ["1e-6", "1e-4", "1e-2"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def d24(a, b):
    d = np.abs(np.mod(np.asarray(a, float), 24.0)
               - np.mod(np.asarray(b, float), 24.0))
    return np.minimum(d, 24.0 - d)


def gate():
    entries = {}
    for tag, d in RUNS.items():
        fz = json.loads((A / d / "MODEL_FREEZE.json").read_text())
        assert sha(A / d / "PREDICTIONS_BLINDED.csv.gz") \
            == fz["prediction_sha256"], tag
        entries[f"B0_corrected[{tag}]"] = fz
    for p in PERTURBS:
        d = A / f"08_STABILITY/PERTURB_{p}"
        fz = json.loads((d / "MODEL_FREEZE.json").read_text())
        assert sha(d / "PREDICTIONS_BLINDED.csv.gz") \
            == fz["prediction_sha256"], p
        entries[f"B0_perturb[{p}]"] = fz
    tmeta = json.loads((A / "09_COMPARATOR_RECOVERY/TEMPO_FROZEN_"
                        "REFERENCE/TEMPO_FROZEN_METADATA.json")
                       .read_text())
    assert sha(TEMPO_PRED) == tmeta["prediction_sha256"]
    entries["Tempo(frozen)"] = tmeta
    if TEMPO17_PRED.exists():
        fz = json.loads((A / "09_COMPARATOR_RECOVERY/TEMPO_SAME17/"
                         "PREDICTION_FREEZE.json").read_text())
        assert sha(TEMPO17_PRED) == fz["prediction_sha256"]
        entries["TEMPO_SAME17"] = fz
    else:
        entries["TEMPO_SAME17"] = {"status": "NOT_RUN_TECHNICAL"}
    out = {"ALL_METHOD_PREDICTIONS_FROZEN": True,
           "methods": {k: {kk: vv for kk, vv in v.items()
                           if "sha256" in kk or kk == "status"}
                       for k, v in entries.items()},
           "verified_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (A / "10_ALL_METHOD_FREEZE_GATE"
     / "ALL_METHOD_PREDICTIONS_FROZEN.json").write_text(
        json.dumps(out, indent=2))
    log(f"gate PASS ({len(entries)} entries)")


def eval_method(pred_df, phase_col, priv):
    m = pred_df.merge(priv, on="blind_sample_id", how="left",
                      validate="many_to_one")
    if m.hidden_ZT.isna().any():
        raise RuntimeError("join failure")
    ph, zt = m[phase_col].to_numpy(), m.hidden_ZT.to_numpy()
    err = d24(ph, zt)
    rows = []
    for _, g in m.groupby("blind_sample_id", sort=True):
        z = np.exp(1j * 2 * np.pi * g[phase_col].to_numpy() / 24)
        sp = float(np.mod(np.angle(z.mean()) * 24 / (2 * np.pi), 24))
        rows.append((float(g.hidden_ZT.iloc[0]), sp,
                     len(g)))
    sdf = pd.DataFrame(rows, columns=["true_ZT", "sp", "n_cells"])
    cerr = d24(sdf.sp, sdf.true_ZT)
    deltas = np.arange(0, 24, 0.01)
    rot_cell = np.array([np.median(d24(ph + dd, zt)) for dd in deltas])
    rot_samp = np.array([np.median(d24(sdf.sp + dd, sdf.true_ZT))
                         for dd in deltas])
    kc, ks = int(np.argmin(rot_cell)), int(np.argmin(rot_samp))
    zc = np.exp(1j * 2 * np.pi * ph / 24).mean()
    zs = np.exp(1j * 2 * np.pi * sdf.sp / 24).mean()
    return {"cell_median_AE": float(np.median(err)),
            "cell_mean_AE": float(err.mean()),
            "sample_median_AE": float(np.median(cerr)),
            "sample_mean_AE": float(cerr.mean()),
            "within_1h": float((err <= 1).mean()),
            "within_2h": float((err <= 2).mean()),
            "within_4h": float((err <= 4).mean()),
            "PLV_direct": float(abs(zs)),
            "PLV_reflected": float(abs(np.exp(
                1j * 2 * np.pi * (sdf.sp + sdf.true_ZT) / 24).mean())),
            "cell_R": float(abs(zc)),
            "shifted_cell_median_AE": float(rot_cell[kc]),
            "shifted_sample_median_AE": float(rot_samp[ks]),
            "oracle_rotation_h": float(deltas[ks]),
            "anchoring_penalty_h": float(np.median(cerr)
                                         - rot_samp[ks])}, m, sdf


def main():
    if "--gate-only" in sys.argv:
        gate()
        return
    gate()
    log("=== post-freeze evaluation START (reading ZT) ===")
    priv = pd.read_csv(PRIV, usecols=["blind_sample_id", "hidden_ZT"])
    results, sdfs = {}, {}

    for tag, d in RUNS.items():
        pdf = pd.read_csv(A / d / "PREDICTIONS_BLINDED.csv.gz")
        r, m, sdf = eval_method(pdf, "phase_mean_h", priv)
        results[f"B0_corrected[{tag}]"] = r
        sdfs[f"B0_corrected[{tag}]"] = sdf
    tp = pd.read_csv(TEMPO_PRED)
    r, m, sdf = eval_method(tp, "pred_phase_MAP_h", priv)
    results["Tempo(frozen)"] = r
    sdfs["Tempo(frozen)"] = sdf
    tempo_cell = r["cell_median_AE"]
    if TEMPO17_PRED.exists():
        t17 = pd.read_csv(TEMPO17_PRED)
        r, m, sdf = eval_method(t17, "pred_phase_MAP_h", priv)
        results["TEMPO_SAME17"] = r
        sdfs["TEMPO_SAME17"] = sdf

    resdf = pd.DataFrame(results).T
    resdf.to_csv(A / "11_POST_FREEZE_EVALUATION/ALL_METHOD_METRICS.csv")

    # ---- cross-run stability ----
    b0 = {t: pd.read_csv(A / d / "PREDICTIONS_BLINDED.csv.gz")
          .sort_values("cell_id").reset_index(drop=True)
          for t, d in RUNS.items()}
    base = b0["RUN1"]
    rows = []
    for t in ("RUN2", "RUN3"):
        dd = d24(base.phase_mean_h, b0[t].phase_mean_h)
        rows.append({"comparison": f"RUN1_vs_{t}",
                     "median_h": float(np.median(dd)),
                     "max_h": float(dd.max())})
    cross_run_identical = bool(
        base.phase_mean_h.equals(b0["RUN2"].phase_mean_h)
        and base.phase_mean_h.equals(b0["RUN3"].phase_mean_h))
    # ---- perturbation stability ----
    for p in PERTURBS:
        pp = pd.read_csv(A / f"08_STABILITY/PERTURB_{p}"
                         / "PREDICTIONS_BLINDED.csv.gz") \
            .sort_values("cell_id").reset_index(drop=True)
        dd = d24(base.phase_mean_h, pp.phase_mean_h)
        rows.append({"comparison": f"RUN1_vs_PERTURB_{p}",
                     "median_h": float(np.median(dd)),
                     "max_h": float(dd.max()),
                     "p95_h": float(np.percentile(dd, 95))})
    sdfstab = pd.DataFrame(rows)
    sdfstab.to_csv(A / "08_STABILITY/PERTURBATION_STABILITY.csv",
                   index=False)
    stable = bool(cross_run_identical
                  and sdfstab.loc[sdfstab.comparison.str.contains(
                      "PERTURB"), "median_h"].max() <= 1.0)
    log(f"stability: cross-run identical={cross_run_identical}; "
        f"perturb medians "
        f"{sdfstab.loc[3:, 'median_h'].tolist()}")

    # ---- uncertainty calibration ----
    err1 = d24(base.phase_mean_h,
               base.merge(priv, on="blind_sample_id")
               .hidden_ZT)
    cal_rows = []
    q = pd.qcut(base.posterior_R, 5, duplicates="drop")
    for cat, idx in base.groupby(q, observed=True).groups.items():
        ii = list(idx)
        cal_rows.append({"R_quintile": str(cat),
                         "median_AE": float(np.median(err1[ii])),
                         "mean_R": float(base.posterior_R[ii].mean()),
                         "mean_entropy": float(
                             base.posterior_entropy[ii].mean()),
                         "mean_ci95_width": float(
                             base.ci95_width_h[ii].mean())})
    caldf = pd.DataFrame(cal_rows)
    caldf.to_csv(A / "11_POST_FREEZE_EVALUATION/"
                 "UNCERTAINTY_CALIBRATION.csv", index=False)
    calib_spearman = float(pd.Series(
        err1).corr(base.posterior_R, method="spearman"))

    # ---- 12: paired sample-level fairness B0 vs Tempo ----
    b0sdf = sdfs["B0_corrected[RUN1]"].copy()
    tsdf = sdfs["Tempo(frozen)"].copy()
    assert np.allclose(b0sdf.true_ZT.values, tsdf.true_ZT.values)
    paired = pd.DataFrame({
        "true_ZT": b0sdf.true_ZT.values,
        "n_cells": b0sdf.n_cells.values,
        "sp_B0": b0sdf.sp.values, "sp_Tempo": tsdf.sp.values})
    paired["AE_B0"] = d24(paired.sp_B0, paired.true_ZT)
    paired["AE_Tempo"] = d24(paired.sp_Tempo, paired.true_ZT)
    paired["delta_j"] = paired.AE_B0 - paired.AE_Tempo
    paired.to_csv(A / "12_PAIRED_SAMPLE_COMPARISON/"
                  "PAIRED_SAMPLE_TABLE.csv", index=False)
    rng = np.random.default_rng(20261007)
    deltas = paired.delta_j.to_numpy()
    n = len(deltas)
    boot = np.array([np.median(deltas[rng.integers(0, n, n)])
                     for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    sf = np.array([np.median(deltas * rng.choice([-1.0, 1.0], n))
                   for _ in range(10000)])
    p_sign = float((np.abs(sf) >= abs(np.median(deltas))).mean())
    fairness = {
        "n_samples": n,
        "median_delta_B0_minus_Tempo": float(np.median(deltas)),
        "mean_delta": float(deltas.mean()),
        "bootstrap95_ci": [float(lo), float(hi)],
        "signflip_permutation_p": p_sign,
        "note": "sign-flip granularity 1/2^%d; n small" % n,
        "resampling_unit": "whole biological samples",
    }
    (A / "12_PAIRED_SAMPLE_COMPARISON"
     / "PAIRED_SAMPLE_FAIRNESS.json").write_text(
        json.dumps(fairness, indent=2))
    log(f"paired fairness: median delta "
        f"{fairness['median_delta_B0_minus_Tempo']:.3f}h "
        f"CI [{lo:.3f},{hi:.3f}] p={p_sign:.4f}")

    # ---- 13: final status ----
    b0_cell = results["B0_corrected[RUN1]"]["cell_median_AE"]
    gap = b0_cell - tempo_cell
    if not stable or b0_cell >= 4.0:
        verdict = "V2_FOUNDATION_CORRECTED_FAILED"
    elif b0_cell <= tempo_cell + 0.50:
        verdict = "V2_FOUNDATION_CORRECTED_COMPETITIVE"
    else:
        verdict = "V2_FOUNDATION_CORRECTED_VIABLE"
    verdict_json = {
        "verdict": verdict,
        "b0_cell_raw_median_AE": b0_cell,
        "b0_sample_raw_median_AE":
            results["B0_corrected[RUN1]"]["sample_median_AE"],
        "tempo_cell_raw_median_AE": tempo_cell,
        "gap_vs_tempo_h": gap,
        "stable": stable,
        "cross_run_bit_identical": cross_run_identical,
        "task10_uncorrected_cell_AE": 2.409,
        "calibration_spearman_R_vs_AE": calib_spearman,
        "all_results": {k: {kk: vv for kk, vv in v.items()
                            if isinstance(vv, (int, float, bool))}
                        for k, v in results.items()},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    (A / "13_FAIRNESS_AUDIT/10A_FINAL_STATUS.json").write_text(
        json.dumps(verdict_json, indent=2))
    log(f"VERDICT: {verdict} (B0 cell {b0_cell:.3f} vs Tempo "
        f"{tempo_cell:.3f}, gap {gap:+.3f}h)")

    # ---- figures + report ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    nm = [k for k in results]
    cm = [results[k]["cell_median_AE"] for k in nm]
    sm = [results[k]["sample_median_AE"] for k in nm]
    x = np.arange(len(nm))
    axes[0].bar(x - 0.18, cm, 0.36, label="cell median AE")
    axes[0].bar(x + 0.18, sm, 0.36, label="sample median AE")
    axes[0].set_xticks(x); axes[0].set_xticklabels(nm, rotation=45,
                                                   ha="right")
    axes[0].set_ylabel("raw AE (h)"); axes[0].legend()
    axes[0].set_title("10A raw absolute error")
    axes[1].plot(caldf.mean_R, caldf.median_AE, "o-")
    axes[1].set_xlabel("posterior R quintile mean")
    axes[1].set_ylabel("median AE (h)")
    axes[1].set_title("uncertainty calibration")
    pp = [r for r in rows if "PERTURB" in r["comparison"]]
    axes[2].bar([r["comparison"].split("_vs_")[1] for r in pp],
                [r["median_h"] for r in pp])
    axes[2].set_ylabel("median prediction distance (h)")
    axes[2].set_title("initialization perturbation stability")
    fig.tight_layout()
    fig.savefig(A / "FIGURES/10A_SUMMARY.png", dpi=150)

    rep = [
        "TASK 10A FINAL REPORT — corrected KATv2-B0",
        "=" * 50,
        f"verdict: {verdict}",
        f"B0 corrected cell raw median AE: {b0_cell:.3f} h",
        f"B0 corrected sample raw median AE: "
        f"{results['B0_corrected[RUN1]']['sample_median_AE']:.3f} h",
        f"Tempo frozen cell raw median AE: {tempo_cell:.3f} h",
        f"gap (B0 - Tempo): {gap:+.3f} h",
        f"cross-run bit-identical: {cross_run_identical}",
        f"calibration Spearman(R, AE): {calib_spearman:.3f}",
        "",
        "paired sample fairness (B0 - Tempo):",
        f"  median delta {fairness['median_delta_B0_minus_Tempo']:+.3f} h"
        f", 95% CI [{lo:.3f}, {hi:.3f}], sign-flip p={p_sign:.4f}",
        "",
        "all methods:",
        resdf[["cell_median_AE", "sample_median_AE", "within_2h",
               "within_4h", "PLV_direct"]].to_string(),
    ]
    (A / "13_FAIRNESS_AUDIT/10A_FINAL_REPORT.txt").write_text(
        "\n".join(rep))
    log("10A evaluation complete")


if __name__ == "__main__":
    main()
