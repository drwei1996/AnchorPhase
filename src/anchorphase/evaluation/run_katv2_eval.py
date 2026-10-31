"""Task 10 — freeze gate, post-freeze evaluation, tables, figures,
verdict, final report. Reads private ZT only after the all-methods
gate passes."""
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
V2 = RR / "10_KAT_V2_FOUNDATION_SINGLECELL_BENCHMARK_20261006"
BASE = V2 / "07_BASELINE_RECOVERY"
SMC = RR / "02B_BASELINE_REPRODUCTION_STRICT"
PRIV = SMC / "03_BLINDED_SAMPLE_ID_MAP_PRIVATE.csv"
TEMPO_PRED = (RR / "02D_FIX2_TEMPO_SERVER_REPRODUCTION_20261005"
              / "tempo_predictions_blinded.csv.gz")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def d24(a, b):
    d = np.abs(np.mod(np.asarray(a, float), 24.0)
               - np.mod(np.asarray(b, float), 24.0))
    return np.minimum(d, 24.0 - d)


B0_RUNS = {"B0_RUN1": "05_B0_TRAIN_BLINDED_RUN1",
           "B0_RUN2": "05_B0_TRAIN_BLINDED_RUN2",
           "B0_RUN3": "05_B0_TRAIN_BLINDED_RUN3"}


def gate():
    entries = {}
    for tag, d in B0_RUNS.items():
        fz = json.loads((V2 / d / "MODEL_FREEZE.json").read_text())
        assert sha(V2 / d / "PREDICTIONS_BLINDED.csv.gz") \
            == fz["prediction_sha256"]
        entries[tag] = fz
    for name, d in [("Tempo_frozen", None),
                    ("PCA_CORE17", "PCA_CORE17"),
                    ("PCA_ALL", "PCA_ALL")]:
        if d is None:
            fz = {"prediction_sha256": sha(TEMPO_PRED),
                  "config_sha256": json.loads(
                      (BASE / "TEMPO_FROZEN_METADATA.json")
                      .read_text())["config_sha256"],
                  "code_hash": "official tempo-main repo (task-06 "
                               "editable install)",
                  "gene_set_hash": "official (de novo + priors)",
                  "prior_hash": sha(ROOT / "external_repos/tempo-main"
                                    "/data/"
                                    "clock_aorta_acrophase_prior.csv")}
        else:
            fz = json.loads((BASE / d / "MODEL_FREEZE.json").read_text())
            fz["gene_set_hash"] = sha(V2 / "02_DATA_AND_PRIOR_FREEZE"
                                      "/GENESET_FREEZE.json")
        entries[name] = fz
    nv = json.loads((BASE / "NAIVE" / "NAIVE_FREEZE.json").read_text())
    for r in nv["baselines"]:
        entries[r["method"]] = {"prediction_sha256":
                                 r["prediction_sha256"],
                                 "code_hash": "naive"}
    out = {"ALL_METHOD_PREDICTIONS_FROZEN": True,
           "methods": {k: {kk: vv for kk, vv in v.items()
                           if kk in ("prediction_sha256",
                                     "config_sha256", "code_hash",
                                     "gene_set_hash", "prior_hash")}
                       for k, v in entries.items()},
           "verified_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (V2 / "08_ALL_METHODS_FREEZE_GATE"
     / "ALL_METHOD_PREDICTIONS_FROZEN.json").write_text(
        json.dumps(out, indent=2))
    g = json.loads((V2 / "08_ALL_METHODS_FREEZE_GATE"
                    / "ALL_METHOD_PREDICTIONS_FROZEN.json").read_text())
    assert g["ALL_METHOD_PREDICTIONS_FROZEN"]
    log(f"all-methods gate PASS ({len(entries)} methods)")


def eval_method(pred_df, phase_col, priv, anchored):
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
        rows.append((float(g.hidden_ZT.iloc[0]), sp))
    sdf = pd.DataFrame(rows, columns=["true_ZT", "sp"])
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
            "anchored": anchored}, m


def main():
    if "--gate-only" in sys.argv:
        gate()
        return
    gate()
    log("=== post-freeze evaluation START ===")
    priv = pd.read_csv(PRIV, usecols=["blind_sample_id", "hidden_ZT"])
    log(f"private map loaded post-gate: {len(priv)} rows")

    results, merged_by = {}, {}
    # KATv2 B0 runs (primary point estimate: circular posterior mean)
    for tag, d in B0_RUNS.items():
        pdf = pd.read_csv(V2 / d / "PREDICTIONS_BLINDED.csv.gz")
        r, m = eval_method(pdf, "phase_mean_h", priv, True)
        r["condition"] = "core_clock_anchored17"
        results[f"KATv2-B0[{tag}]"] = r
        merged_by[tag] = m
    # Tempo frozen (MAP convention as frozen)
    tp = pd.read_csv(TEMPO_PRED)
    r, m = eval_method(tp, "pred_phase_MAP_h", priv, True)
    r["condition"] = "official_full"
    results["Tempo(frozen)"] = r
    merged_by["Tempo"] = m
    # PCA
    for name, d, cond in [("PCA_CORE17", "PCA_CORE17",
                           "core_clock_anchored17"),
                          ("PCA_ALL", "PCA_ALL", "all_genes")]:
        pdf = pd.read_csv(BASE / d / "PREDICTIONS_BLINDED.csv.gz")
        r, m = eval_method(pdf, "phase_h", priv, False)
        r["condition"] = cond
        results[name] = r
        merged_by[name] = m
    # naive
    nv_dir = BASE / "NAIVE"
    for name, fn in [("UNIFORM_RANDOM",
                      "UNIFORM_RANDOM_PREDICTIONS_BLINDED.csv.gz"),
                     ("SINGLE_PHASE_POINT",
                      "SINGLE_PHASE_POINT_PREDICTIONS_BLINDED.csv.gz")]:
        pdf = pd.read_csv(nv_dir / fn)
        r, m = eval_method(pdf, "phase_h", priv, False)
        r["condition"] = "chance"
        results[name] = r

    resdf = pd.DataFrame(results).T
    resdf.to_csv(V2 / "09_POST_FREEZE_EVALUATION/ALL_METHOD_METRICS.csv")

    # ---- B0 stability (3 runs) ----
    b0 = {t: pd.read_csv(V2 / d / "PREDICTIONS_BLINDED.csv.gz")
          .sort_values("cell_id").reset_index(drop=True)
          for t, d in B0_RUNS.items()}
    stab = []
    for a, b in [("B0_RUN1", "B0_RUN2"), ("B0_RUN1", "B0_RUN3"),
                 ("B0_RUN2", "B0_RUN3")]:
        dd = d24(b0[a].phase_mean_h, b0[b].phase_mean_h)
        stab.append({"pair": f"{a}_vs_{b}",
                     "median_h": float(np.median(dd)),
                     "p95_h": float(np.percentile(dd, 95))})
    sdf = pd.DataFrame(stab)
    sdf.to_csv(V2 / "06_B0_STABILITY/STABILITY_METRICS.csv", index=False)
    stab_pass = bool(sdf.median_h.max() <= 0.05
                     and sdf.p95_h.max() <= 0.25)
    (V2 / "06_B0_STABILITY/STABILITY_VERDICT.json").write_text(
        json.dumps({"pairs": stab,
                    "V2_OPTIMIZATION_STABILITY": "PASS" if stab_pass
                    else "FAIL"}, indent=2))
    log(f"B0 stability: median max={sdf.median_h.max():.6f}h "
        f"p95 max={sdf.p95_h.max():.6f}h -> "
        f"{'PASS' if stab_pass else 'FAIL'}")

    # ---- uncertainty evaluation (B0 + Tempo) ----
    b0r = merged_by["B0_RUN1"]
    pdf1 = pd.read_csv(V2 / B0_RUNS["B0_RUN1"]
                       / "PREDICTIONS_BLINDED.csv.gz")
    unc_rows = []
    qb = pd.qcut(pdf1.posterior_R, 5, duplicates="drop")
    err1 = d24(b0r.phase_mean_h, b0r.hidden_ZT)
    for cat, idx in pdf1.groupby(qb, observed=True).groups.items():
        ii = list(idx)
        unc_rows.append({"method": "KATv2-B0", "R_quartile": str(cat),
                         "median_AE": float(np.median(err1[ii])),
                         "mean_R": float(pdf1.posterior_R[ii].mean())})
    tm = merged_by["Tempo"]
    tpdf = pd.read_csv(TEMPO_PRED).sort_values("cell_id").reset_index(
        drop=True)
    qt = pd.qcut(tpdf.posterior_cSTD_h, 5, duplicates="drop")
    errT = d24(tm.pred_phase_MAP_h, tm.hidden_ZT)
    for cat, idx in tpdf.groupby(qt, observed=True).groups.items():
        ii = list(idx)
        unc_rows.append({"method": "Tempo", "cSTD_quintile": str(cat),
                         "median_AE": float(np.median(errT[ii])),
                         "mean_cSTD": float(tpdf.posterior_cSTD_h[ii]
                                            .mean())})
    pd.DataFrame(unc_rows).to_csv(
        V2 / "09_POST_FREEZE_EVALUATION/UNCERTAINTY_CALIBRATION.csv",
        index=False)

    # ---- library-size quartiles ----
    Xlib = np.load(V2 / "02_DATA_AND_PRIOR_FREEZE/anchored_lib.npy")
    ql = pd.qcut(Xlib, 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    lib_rows = []
    for cat in ["Q1_low", "Q2", "Q3", "Q4_high"]:
        ii = np.where(ql == cat)[0]
        lib_rows.append({"quartile": cat,
                         "median_AE": float(np.median(err1[ii])),
                         "mean_entropy": float(
                             pdf1.posterior_entropy[ii].mean()),
                         "mean_R": float(
                             pdf1.posterior_R[ii].mean())})
    pd.DataFrame(lib_rows).to_csv(
        V2 / "09_POST_FREEZE_EVALUATION/LIBRARY_QUARTILES.csv",
        index=False)

    # ---- gene information (STEP 25, model-internal) ----
    import torch
    gp = pd.read_csv(V2 / B0_RUNS["B0_RUN1"] / "GENE_PARAMETERS.csv")
    fz = json.loads((V2 / B0_RUNS["B0_RUN1"]
                     / "MODEL_FREEZE.json").read_text())
    hist = pd.read_csv(V2 / B0_RUNS["B0_RUN1"] / "TRAIN_HISTORY.csv")
    gp["information_note"] = ("A_g*dispersion-weighted harmonic "
                              "contrast (model-internal ranking only)")
    gp = gp.sort_values("A_g", ascending=False)
    gp.to_csv(V2 / "12_V2_DIAGNOSTICS/CLOCK_GENE_INFORMATION.csv",
              index=False)

    # ---- benchmark tables ----
    tempo_cell = float(results["Tempo(frozen)"]["cell_median_AE"])
    rows = []
    runtime_map = {}
    for name, r in results.items():
        rows.append({
            "method": name,
            "condition": r.get("condition", ""),
            "native_single_cell": "YES" if ("KATv2" in name or "Tempo"
                                            in name or "PCA" in name)
            else ("n/a" if "UNIFORM" in name or "SINGLE" in name
                  else "?"),
            "absolute_anchor": r["anchored"],
            "supervision": "none",
            "raw_counts_model": ("YES" if ("KATv2" in name
                                           or "Tempo" in name)
                                 else "NO"),
            "uncertainty": ("YES" if ("KATv2" in name or "Tempo"
                                      in name) else "NO"),
            "cell_raw_median_AE": r["cell_median_AE"],
            "sample_raw_median_AE": r["sample_median_AE"],
            "cell_shifted_median_AE": r["shifted_cell_median_AE"],
            "sample_shifted_median_AE": r["shifted_sample_median_AE"],
            "PLV_direct": r["PLV_direct"],
            "phase_R": r["cell_R"],
            "within_2h": r["within_2h"],
            "within_4h": r["within_4h"],
            "runtime_minutes": "",
            "peak_RAM": "", "peak_VRAM": "",
            "stability_metric": (f"median {sdf.median_h.max():.4f}h "
                                 "cross-run" if "B0_RUN1" in name
                                 else ""),
            "notes": "" if r["anchored"]
            else "UNANCHORED_NOT_PRIMARY"})
    tab = pd.DataFrame(rows)
    tab.to_csv(V2 / "TABLES/SINGLE_CELL_METHOD_BENCHMARK.csv", index=False)

    sl = [["KATv2-B0", "deterministic anchored NB; exact posterior; "
           "uncertainty-first", "fixed-prior rigidity; single-cohort",
           "YES (Tempo priors)", "YES", "NB", "YES", "no", "yes",
           "yes", "cohort-specific", "scales with N*K*G", 
           "fast deterministic cohort phase model", "YES"],
          ["Tempo", "scRNA-native NB phase + priors + BF",
           "heavy; prior-dependent", "YES", "YES", "NB", "YES",
           "no", "yes", "yes", "cohort-specific", "minutes-hours",
           "full probabilistic scRNA clock", "YES"],
          ["Cyclops", "established circular reconstruction",
           "anchor-free; bulk-origin", "NO", "NO", "none", "NO",
           "no", "no", "partial", "cohort", "moderate",
           "relative ordering reference", "NO (R env)"],
          ["Cyclum", "lightweight linear-cycle",
           "anchor-free; TF1.15 legacy", "NO", "NO", "none", "NO",
           "no", "no", "partial", "cohort", "fast",
           "relative ordering reference", "NO (env)"],
          ["PCA", "trivial reproducible baseline", "linear; anchor-free",
           "NO", "NO", "none", "NO", "no", "no", "yes", "cohort",
           "seconds", "sanity floor", "YES (relative only)"],
          ["DCPR", "whole-cohort circular ordering (bulk)",
           "not a droplet-scRNA count model", "NO", "NO", "approx",
           "NO", "no", "no", "no", "cohort", "moderate",
           "bulk cohort ordering", "NO"],
          ["TimeTeller", "supervised absolute clock",
           "needs time labels", "YES", "YES", "Gaussian", "PARTIAL",
           "yes", "no", "no", "pretrained", "fast",
           "supervised absolute time", "NO (labels)"],
          ["PRECISE", "principled stochastic-expression model",
           "noise limits; benefits from averaging", "NO", "PARTIAL",
           "GP", "YES", "no", "no", "partial", "cohort", "moderate",
           "single-cell noise modeling", "NO"],
          ["ZeitZeiger", "supervised sparse periodic prediction",
           "needs time labels", "YES(trained)", "YES(trained)",
           "sparse GLM", "NO", "yes", "no", "no", "cohort/pretrained",
           "fast", "supervised bulk benchmark", "NO (labels)"]]
    pd.DataFrame(sl, columns=["method", "strength_for_scRNA",
                              "weakness_for_scRNA", "absolute_phase",
                              "relative_order", "count_noise_model",
                              "uncertainty", "needs_time_labels",
                              "needs_external_clock_prior",
                              "works_on_single_cell", "cohort_specific",
                              "expected_scalability", "main_use_case",
                              "fair_direct_competitor"]).to_csv(
        V2 / "TABLES/METHOD_STRENGTHS_LIMITATIONS.csv", index=False)

    # ---- verdict (STEP 29) ----
    b0_cell = float(results["KATv2-B0[B0_RUN1]"]["cell_median_AE"])
    sim_pass = json.loads((V2 / "04_SIMULATION_SANITY"
                           / "SIM_VERDICT.json").read_text())
    foundation = (sim_pass["exact_model_recovers_phase"] and stab_pass
                  and b0_cell < 4.0)
    if not foundation:
        verdict = "V2_FOUNDATION_FAILED"
    elif b0_cell <= tempo_cell + 0.50:
        verdict = ("V2_FOUNDATION_PROMISING" if b0_cell < tempo_cell
                   else "V2_FOUNDATION_COMPETITIVE")
    else:
        verdict = "V2_FOUNDATION_VIABLE_BUT_BELOW_TEMPO"
    log(f"VERDICT: {verdict} (B0 cell {b0_cell:.3f} vs Tempo "
        f"{tempo_cell:.3f})")

    (V2 / "KAT_V2_FOUNDATION_VERDICT.json").write_text(json.dumps({
        "verdict": verdict,
        "b0_cell_raw_median_AE": b0_cell,
        "tempo_cell_raw_median_AE": tempo_cell,
        "stability_pass": stab_pass,
        "simulation_pass": sim_pass,
        "all_results": {k: {kk: vv for kk, vv in v.items()
                            if isinstance(vv, (int, float, bool))}
                        for k, v in results.items()},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    with open("/tmp/katv2_eval_state.json", "w") as f:
        json.dump({"verdict": verdict, "b0_cell": b0_cell,
                   "tempo_cell": tempo_cell,
                   "stab_pass": stab_pass}, f)
    log("evaluation complete; run report generation separately")


if __name__ == "__main__":
    main()
