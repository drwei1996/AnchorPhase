"""Task 10 comparators: Tempo frozen recovery, PCA circular baseline
(core + all-gene), naive baselines. All blinded; no ZT."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parents[1]
RR = ROOT / "results/single_cell_phase_rebuild"
V2 = RR / "10_KAT_V2_FOUNDATION_SINGLECELL_BENCHMARK_20261006"
BASE = V2 / "07_BASELINE_RECOVERY"
SMC = RR / "02B_BASELINE_REPRODUCTION_STRICT"
TEMPO = RR / "02D_FIX2_TEMPO_SERVER_REPRODUCTION_20261005"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tempo_recovery():
    pred_path = TEMPO / "tempo_predictions_blinded.csv.gz"
    fz = json.loads((TEMPO / "PREDICTION_FREEZE.json").read_text())
    assert sha(pred_path) == fz["prediction_sha256"]
    df = pd.read_csv(pred_path)
    meta = {"method": "Tempo (frozen task-06 FIX2 reproduction)",
            "prediction_file": str(pred_path.relative_to(ROOT)),
            "prediction_sha256": fz["prediction_sha256"],
            "config_sha256": fz["config_sha256"],
            "config": "official-like: grid=12, phase_est=3/3, vi=3000, "
                      "mean_disp_bin=50, cell_batch=256 (memory patch)",
            "n_cells": int(len(df)),
            "uncertainty": "posterior_cSTD_h available in frozen file",
            "runtime_sec": fz.get("tempo_runtime_sec"),
            "core_clock_only_result_exists": False,
            "recovered_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (BASE / "TEMPO_FROZEN_METADATA.json").write_text(json.dumps(meta,
                                                                 indent=2))
    print("Tempo frozen recovery OK:",
          fz["prediction_sha256"][:16])


def pca_baseline(condition, gene_idx, tag):
    out = BASE / f"PCA_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    X = sp.load_npz(SMC / "SMC_counts.npz").tocsr()
    Xs = X[:, gene_idx].astype(np.float64)
    lib = np.asarray(X.sum(axis=1)).ravel()
    # Tempo-style preprocessing: log10 of cell-normalized proportions
    row = np.asarray(Xs.sum(axis=1)).ravel()
    Xn = Xs.multiply((1e4 / np.maximum(row, 1.0))[:, None])
    Xn = Xn.log1p() if hasattr(Xn, "log1p") else Xn
    svd = TruncatedSVD(n_components=2, random_state=0,
                       algorithm="arpack")
    coords = svd.fit_transform(Xn)
    phase = np.mod(np.arctan2(coords[:, 1], coords[:, 0])
                   * 24 / (2 * np.pi), 24.0)
    obs = pd.read_csv(SMC / "01_SMC_CELL_SELECTION_AUDIT.csv",
                      usecols=["cell_id", "blind_sample_id"])
    pred = pd.DataFrame({"cell_id": obs.cell_id.values,
                         "blind_sample_id": obs.blind_sample_id.values,
                         "phase_h": phase,
                         "pc1": coords[:, 0], "pc2": coords[:, 1]})
    pp = out / "PREDICTIONS_BLINDED.csv.gz"
    pred.to_csv(pp, index=False, compression="gzip")
    cfg = {"method": "PCA circular baseline (Tempo-style)",
           "condition": condition, "n_genes": int(len(gene_idx)),
           "preprocessing": "log1p(1e4-normalized counts over input genes)",
           "phase": "atan2(PC2, PC1) mapped to [0,24)",
           "svd": "sklearn TruncatedSVD(arpack, random_state=0)",
           "anchored": False, "uncertainty": False,
           "runtime_sec": round(time.perf_counter() - t0)}
    cp = out / "MODEL_CONFIG.json"
    cp.write_text(json.dumps(cfg, indent=2))
    frz = {"prediction_sha256": sha(pp), "config_sha256": sha(cp),
           "script_sha256": sha(Path(__file__).resolve()),
           "anchored": False,
           "PREDICTIONS_FROZEN": True, "TRUE_ZT_READ": False,
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "MODEL_FREEZE.json").write_text(json.dumps(frz, indent=2))
    print(f"PCA[{tag}] frozen ({cfg['runtime_sec']}s)")


def naive():
    out = BASE / "NAIVE"
    out.mkdir(parents=True, exist_ok=True)
    obs = pd.read_csv(SMC / "01_SMC_CELL_SELECTION_AUDIT.csv",
                      usecols=["cell_id", "blind_sample_id"])
    rng = np.random.default_rng(20261005)
    rows = []
    for name, ph in [
            ("UNIFORM_RANDOM", rng.uniform(0, 24, len(obs))),
            ("SINGLE_PHASE_POINT", np.full(len(obs), 12.0))]:
        pred = pd.DataFrame({"cell_id": obs.cell_id.values,
                             "blind_sample_id": obs.blind_sample_id.values,
                             "phase_h": ph})
        pp = out / f"{name}_PREDICTIONS_BLINDED.csv.gz"
        pred.to_csv(pp, index=False, compression="gzip")
        rows.append({"method": name, "prediction_sha256": sha(pp),
                     "seed_or_constant": "20261005 / 12.0h"})
    (out / "NAIVE_FREEZE.json").write_text(json.dumps(
        {"baselines": rows, "PREDICTIONS_FROZEN": True,
         "TRUE_ZT_READ": False,
         "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    print("naive baselines frozen")


def main():
    tempo_recovery()
    var = pd.read_csv(SMC / "SMC_var.csv")
    freeze = json.loads((V2 / "02_DATA_AND_PRIOR_FREEZE"
                         "/GENESET_FREEZE.json").read_text())
    core = freeze["GENESET_A_TEMPO_CORE_CLOCK"]["anchored_genes"]
    gi = {g: i for i, g in enumerate(var.gene_id)}
    pca_baseline("core_clock_anchored17", [gi[g] for g in core],
                 "CORE17")
    pca_baseline("all_genes", list(range(len(var))), "ALL")
    naive()
    print("comparators complete")


if __name__ == "__main__":
    main()
