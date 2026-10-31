"""Task11B P0: fold manifests (SMC LOSO 5 folds; pooled 5-fold
sample-blocked) + candidate gene filter, all hashed BEFORE any ZT
access."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
RR = ROOT / "results/single_cell_phase_rebuild"
SMC = RR / "02B_BASELINE_REPRODUCTION_STRICT"
T11 = RR / "11_KAT_V2_MODEL_FAMILY_DEVELOPMENT_20261007"
T11B = RR / "11B_KAT_V2_SAFE_PHASE_INFORMATION_20261007"
OUT = T11B / "07_SAMPLE_BLOCK_CROSSFIT"
CORE17 = ["Arntl", "Bhlhe41", "Clock", "Cry1", "Cry2", "Dbp",
          "Fbxo21", "Hlf", "Nfil3", "Npas2", "Nr1d1", "Nr1d2",
          "Per1", "Per2", "Per3", "Rorc", "Tef"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # ---- SMC LOSO ----
    obs = pd.read_csv(SMC / "01_SMC_CELL_SELECTION_AUDIT.csv",
                      usecols=["cell_id", "blind_sample_id"])
    samples = sorted(obs.blind_sample_id.unique())
    smc = obs.copy()
    smc["lineage"] = "smc"
    smc["fold"] = smc.blind_sample_id.map(
        {s: i for i, s in enumerate(samples)})
    # ---- pooled 5-fold sample-blocked ----
    pobs = pd.read_csv(T11 / "11_REALDATA/00_POOLED_FREEZE/"
                       "POOLED_OBS_BLINDED.csv")
    psamples = sorted(pobs.blind_sample_id.unique())
    # round-robin over sorted sample ids so blocks span lineages/ZTs
    block = {s: i % 5 for i, s in enumerate(psamples)}
    pooled = pobs.copy()
    pooled["fold"] = pooled.blind_sample_id.map(block)

    for name, df in (("SMC_LOSO", smc), ("POOLED_5FOLD", pooled)):
        df["selector_train"] = "TRUE"   # filled per-fold at runtime
        df[["cell_id", "blind_sample_id", "lineage", "fold"]] \
            .to_csv(OUT / f"{name}_MANIFEST.csv", index=False)
        (OUT / f"{name}_MANIFEST.sha256").write_text(
            sha(OUT / f"{name}_MANIFEST.csv") + "\n")
    folds = {"SMC_LOSO": {"n_folds": 5, "design": "leave-one-"
                          "sample-out over 5 frozen SMC samples",
                          "manifest_sha256": sha(
                              OUT / "SMC_LOSO_MANIFEST.csv")},
             "POOLED_5FOLD": {"n_folds": 5,
                              "design": "sample-blocked 5-fold over "
                              "17 pooled samples, round-robin in "
                              "sorted blind id order",
                              "manifest_sha256": sha(
                                  OUT / "POOLED_5FOLD_MANIFEST.csv")}}
    (OUT / "FOLD_FREEZE.json").write_text(json.dumps(
        {"folds": folds, "hashed_at":
         time.strftime("%Y-%m-%d %H:%M:%S"),
         "no_true_ZT_read": True}, indent=2))

    # ---- candidate gene filter ----
    X = sp.load_npz(SMC / "SMC_counts.npz").tocsr()
    var = pd.read_csv(SMC / "SMC_var.csv")
    names = np.array(var.gene_id.values)
    y_sum = np.asarray(X.sum(axis=0)).ravel()
    det = np.asarray((X > 0).sum(axis=0)).ravel()
    keep = ((y_sum / X.shape[0] >= 0.1) & (det / X.shape[0] >= 0.01)
            & np.array([g not in CORE17 for g in names]))
    cand = pd.DataFrame({"gene": names[keep],
                         "mean_counts": y_sum[keep] / X.shape[0],
                         "prevalence": det[keep] / X.shape[0]})
    cand.to_csv(OUT / "CANDIDATE_GENES.csv", index=False)
    (OUT / "CANDIDATE_GENES.sha256").write_text(
        sha(OUT / "CANDIDATE_GENES.csv") + "\n")
    print(f"SMC folds: {len(samples)} samples; pooled: "
          f"{len(psamples)} samples / 5 blocks")
    print(f"candidate genes: {keep.sum()} of {len(names)} "
          f"(mean>=0.1 & detected>=1% & non-core)")
    print(json.dumps(folds, indent=1))


if __name__ == "__main__":
    main()
