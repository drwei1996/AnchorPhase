"""Task12 cross-dataset summary + final verdict.
Run after BOTH holdouts are evaluated (B0/C1/C2 [+Tempo])."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RR = ROOT / "results/single_cell_phase_rebuild"
T12 = RR / "12_KAT_V2_EXTERNAL_VALIDATION_20261008"
OUT = T12 / "TABLES"


def d24(a, b):
    d = np.abs(np.mod(a, 24) - np.mod(b, 24))
    return np.minimum(d, 24 - d)


def tempo_eval(gse):
    p = T12 / gse / "07_COMPARATORS/tempo_predictions_blinded.csv.gz"
    if not p.exists():
        return None
    priv = pd.read_csv(T12 / gse / "00_DATA_AUDIT/"
                       "BLIND_MAP_PRIVATE.csv")
    df = pd.read_csv(p)
    m = df.merge(priv, on="blind_sample_id")
    err = d24(m.pred_phase_MAP_h, m.hidden_ZT)
    rows = []
    for _, g in m.groupby("blind_sample_id"):
        z = np.exp(1j * 2 * np.pi * g.pred_phase_MAP_h / 24)
        sp = float(np.mod(np.angle(z.mean()) * 24 / (2 * np.pi),
                          24))
        rows.append((float(g.hidden_ZT.iloc[0]), sp))
    sdf = pd.DataFrame(rows, columns=["t", "sp"])
    cerr = d24(sdf.sp, sdf.t)
    return {"cell_median_AE": float(np.median(err)),
            "cell_mean_AE": float(err.mean()),
            "sample_median_AE": float(np.median(cerr)),
            "within_2h": float((err <= 2).mean()),
            "within_4h": float((err <= 4).mean())}


def main():
    OUT.mkdir(exist_ok=True)
    rows, per_s, sel_genes, alpha_rows, cal_rows, anchor_rows = \
        [], [], [], [], [], []
    pred_hashes = ["dataset,model,file,sha256"]
    gains = {}
    for gse in ("GSE145197", "GSE223109"):
        ev = json.loads((T12 / gse / "06_EVALUATION/"
                         "EVALUATION.json").read_text())
        audit = json.loads((T12 / gse / "00_DATA_AUDIT/"
                            "DATA_AUDIT.json").read_text())
        tp = tempo_eval(gse)
        alpha = json.loads((T12 / gse / "03_SAFE_CPI_BLINDED/"
                            "MODEL_CONFIG.json").read_text())["alpha"]
        b0ae = ev["B0"]["cell_median_AE"]
        c2ae = ev["C2"]["cell_median_AE"]
        gains[gse] = {"dC2_B0": b0ae - c2ae,
                      "dC2_Tempo": (tp["cell_median_AE"] - c2ae)
                      if tp else None}
        row = {"Dataset": gse, "N_cells": audit["n_cells"],
               "N_samples": audit["n_samples"],
               "B0_median_AE": round(b0ae, 3),
               "C1_median_AE": round(
                   ev["C1"]["cell_median_AE"], 3),
               "C2_median_AE": round(c2ae, 3),
               "Tempo_median_AE": round(tp["cell_median_AE"], 3)
               if tp else "NOT_RUN",
               "C2_minus_B0_gain": round(b0ae - c2ae, 3),
               "C2_vs_Tempo_gain": round(
                   tp["cell_median_AE"] - c2ae, 3) if tp else None,
               "C2_alpha": alpha,
               "oracle_rotation_h": round(
                   ev["C2"]["oracle_rotation_h"], 2),
               "Spearman_R_vs_AE": round(
                   ev["C2"]["calib_spearman"], 3),
               "tissue": audit["tissue"]}
        # validation status per dataset
        ok = (b0ae - c2ae) >= 0.20
        row["improves_B0_ge_0.20"] = ok
        rows.append(row)
        # per-sample
        ps = pd.read_csv(T12 / gse / "06_EVALUATION/"
                         "PER_SAMPLE_RESULTS.csv")
        ps["dataset"] = gse
        per_s.append(ps)
        # selected genes + dev overlap
        panel = pd.read_csv(T12 / gse / "04_PREDICTION_FREEZE/"
                            "SELECTED_PANEL.csv")
        panel = panel[panel.fold == 0]
        sc = pd.read_csv(T12 / gse / "04_PREDICTION_FREEZE/"
                         "CPI_SCORES.csv").set_index("gene")
        dev = set(pd.read_csv(
            RR / ("11B_KAT_V2_SAFE_PHASE_INFORMATION_20261007/"
                  "04_C1_CPI/PREDICTIONS_BLINDED.csv.gz"),
            compression="gzip", usecols=[]).columns) if False \
            else set(json.loads((RR / "11B_KAT_V2_SAFE_"
                                 "PHASE_INFORMATION_20261007/"
                                 "03_CPI_METHOD/PANELS_PER_FOLD"
                                 ".json").read_text())["C1"][0])
        panel["CPI"] = [sc.loc[g, "CPI"] for g in panel.gene]
        panel["amplitude"] = [sc.loc[g, "amplitude"]
                              for g in panel.gene]
        panel["acrophase_h"] = [sc.loc[g, "acrophase_h"]
                                for g in panel.gene]
        panel["mean_counts"] = [sc.loc[g, "mean_counts"]
                                for g in panel.gene]
        panel["prevalence"] = [sc.loc[g, "prevalence"]
                               for g in panel.gene]
        panel["dropout"] = [1 - sc.loc[g, "prevalence"]
                            for g in panel.gene]
        panel["dev_panel_overlap"] = [g in dev for g in
                                      panel.gene]
        panel["dataset"] = gse
        sel_genes.append(panel)
        alpha_rows.append({
            "dataset": gse, "alpha": alpha,
            "profile": json.loads((T12 / gse /
                                   "03_SAFE_CPI_BLINDED/"
                                   "MODEL_CONFIG.json")
                                  .read_text())})
        for model in ("B0", "C1", "C2"):
            cal = pd.DataFrame(ev[model]["calib_rows"])
            cal["dataset"] = gse
            cal_rows.append(cal)
            anchor_rows.append({
                "dataset": gse, "model": model,
                "anchored_cell_AE": ev[model][
                    "anchored_cell_median_AE"],
                "oracle_rotation_h": ev[model][
                    "oracle_rotation_h"],
                "anchor_penalty_h": ev[model][
                    "anchor_penalty_h"],
                "reflection_gain_h": ev[model][
                    "reflection_gain_h"]})
        for model, d in (("B0", "01_B0_BLINDED"),
                         ("C1", "02_CPI_BLINDED"),
                         ("C2", "03_SAFE_CPI_BLINDED")):
            p = T12 / gse / d / "PREDICTIONS_BLINDED.csv.gz"
            pred_hashes.append(
                f"{gse},{model},{p.name},"
                f"{hashlib.sha256(p.read_bytes()).hexdigest()}")
        tp_ = T12 / gse / ("07_COMPARATORS/"
                           "tempo_predictions_blinded.csv.gz")
        if tp_.exists():
            pred_hashes.append(
                f"{gse},Tempo,{tp_.name},"
                f"{hashlib.sha256(tp_.read_bytes()).hexdigest()}")
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "EXTERNAL_VALIDATION_TABLE.csv", index=False)
    pd.concat(per_s).to_csv(OUT / "PER_SAMPLE_RESULTS.csv",
                            index=False)
    pd.concat(sel_genes).to_csv(
        OUT / "EXTERNAL_SELECTED_GENES.csv", index=False)
    pd.DataFrame(alpha_rows).drop(columns=["profile"]).to_csv(
        OUT / "EXTERNAL_ALPHA_GATE_RESULTS.csv", index=False)
    pd.concat(cal_rows).to_csv(OUT / "EXTERNAL_CALIBRATION.csv",
                               index=False)
    pd.DataFrame(anchor_rows).to_csv(
        OUT / "EXTERNAL_ANCHOR_ORACLE.csv", index=False)
    (OUT / "EXTERNAL_PREDICTION_HASHES.csv").write_text(
        "\n".join(pred_hashes) + "\n")
    code_rows = ["file,sha256"]
    for f_ in sorted((ROOT / "scripts").glob("task12*.py")):
        code_rows.append(f"scripts/{f_.name},"
                         f"{hashlib.sha256(f_.read_bytes()).hexdigest()}")
    (OUT / "EXTERNAL_CODE_HASHES.csv").write_text(
        "\n".join(code_rows) + "\n")

    # falsification summary
    fals = []
    for gse in ("GSE145197", "GSE223109"):
        d = T12 / gse / "08_DIAGNOSTICS"
        perms = [json.loads(f.read_text())["top40_mean_cpi"]
                 for f in sorted(d.glob("perm*_CPI.json"))]
        rot = json.loads((d / "rot8_CPI.json").read_text())[
            "top40_mean_cpi"] if (d / "rot8_CPI.json"
                                  ).exists() else None
        sc = pd.read_csv(T12 / gse / "04_PREDICTION_FREEZE/"
                         "CPI_SCORES.csv")
        real = sc.CPI.nlargest(40).mean()
        matched = []
        for rep in range(5):
            f = d / f"MATCHED_PANEL_{rep}_PRED.csv.gz"
            if f.exists():
                priv = pd.read_csv(
                    T12 / gse / "00_DATA_AUDIT/"
                    "BLIND_MAP_PRIVATE.csv")
                m = pd.read_csv(f).merge(priv,
                                         on="blind_sample_id")
                matched.append(float(np.median(d24(
                    m.phase_mean_h, m.hidden_ZT))))
        ev = json.loads((T12 / gse / "06_EVALUATION/"
                         "EVALUATION.json").read_text())
        fals.append({"dataset": gse, "real_top40_CPI": real,
                     "perm_max_top40": max(perms) if perms else None,
                     "perm_sep_x": round(real / max(perms), 1)
                     if perms else None,
                     "rot8_top40": rot,
                     "matched_min_AE": min(matched) if matched
                     else None,
                     "C1_AE": ev["C1"]["cell_median_AE"],
                     "F1_pass": max(perms) < real / 10 if perms
                     else None,
                     "F2_pass": (min(matched) > ev["C1"]
                                 ["cell_median_AE"]) if matched
                     else None})
    pd.DataFrame(fals).to_csv(OUT / "EXTERNAL_FALSIFICATION.csv",
                              index=False)

    # ---- verdict ----
    g1, g2 = gains.get("GSE145197", {}), gains.get(
        "GSE223109", {})
    both_improve = (g1.get("dC2_B0", -1) >= 0.20
                    and g2.get("dC2_B0", -1) >= 0.20)
    both_beat_tempo = (
        g1.get("dC2_Tempo") is not None
        and g2.get("dC2_Tempo") is not None
        and g1["dC2_Tempo"] > 0 and g2["dC2_Tempo"] > 0)
    if both_improve and min(g1["dC2_B0"],
                            g2["dC2_B0"]) >= 0.30 \
            and both_beat_tempo:
        validation = "VERY_STRONG"
    elif both_improve and min(g1["dC2_B0"], g2["dC2_B0"]) >= 0.30:
        validation = "STRONG"
    elif both_improve:
        validation = "STRONG"
    elif ((g1.get("dC2_B0", -1) >= 0.20)
          ^ (g2.get("dC2_B0", -1) >= 0.20)):
        validation = "PARTIAL"
    else:
        validation = "FAILED"
    # anchoring/calibration pass across datasets
    anchor_pass = all(
        abs(json.loads((T12 / g / "06_EVALUATION/"
                        "EVALUATION.json").read_text()
                       )["C2"]["oracle_rotation_h"]) <= 3.0
        for g in ("GSE145197", "GSE223109"))
    calib_pass = all(
        json.loads((T12 / g / "06_EVALUATION/"
                    "EVALUATION.json").read_text()
                   )["C2"]["calib_spearman"] < -0.10
        for g in ("GSE145197", "GSE223109"))
    status = {}
    for gse in ("GSE145197", "GSE223109"):
        ev = json.loads((T12 / gse / "06_EVALUATION/"
                         "EVALUATION.json").read_text())
        g_ = ev["B0"]["cell_median_AE"] \
            - ev["C2"]["cell_median_AE"]
        status[gse] = ("IMPROVED" if g_ >= 0.20 else
                       "NEUTRAL" if g_ > -0.10 else "DEGRADED")
    verdict = {
        "TASK12_COMPLETE": True,
        "MODEL": "C2_SAFE_CPI",
        "GSE145197_STATUS": status.get("GSE145197"),
        "GSE223109_STATUS": status.get("GSE223109"),
        "BOTH_HOLDOUTS_IMPROVE_B0": bool(both_improve),
        "BOTH_HOLDOUTS_BEAT_TEMPO": bool(both_beat_tempo)
        if both_beat_tempo is not None else None,
        "ANCHORING_PASS": bool(anchor_pass),
        "CALIBRATION_PASS": bool(calib_pass),
        "EXTERNAL_VALIDATION": validation,
        "FINAL_KATV2_STATUS":
            "KATv2 = corrected anchored B0 + SAFE-CPI information "
            "head; externally "
            + validation.lower().replace("_", " "),
        "gains": {k: {kk: round(vv, 3) for kk, vv in v.items()
                      if vv is not None} for k, v in
                  gains.items()},
    }
    (T12 / "TASK12_FINAL_VERDICT.json").write_text(json.dumps(
        verdict, indent=2))
    print(tab.to_string(index=False))
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
