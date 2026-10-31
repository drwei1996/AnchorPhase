# models/ — frozen model artifacts

AnchorPhase freezes model state as explicit parameter tables + JSON (no opaque checkpoints):

| Directory | What it is |
|---|---|
| `frozen_b0_corrected_run1/` | The paper's deployed model: KATv2-B0 with whole-transcriptome size-factor offset (task 10A run 1). `GENE_PARAMETERS.csv` = per-gene m/beta/A/log_alpha (phi fixed externally); `MODEL_CONFIG.json` = frozen hyperparameters incl. 48-grid; `MODEL_FREEZE.json` = SHA-256 integrity set; `TRAIN_HISTORY.csv` = deterministic full-batch L-BFGS trace. |
| `b0_reference_pooled/` | Pooled reference refit (task 11B) used as the B0 baseline in the model-family comparison; includes its prediction hash. |
| `safe_cpi_final_freeze/` | The falsified-away SAFE-CPI (C2) final freeze. Kept deliberately for transparency: C2 won development but failed external falsification F2 (GSE223109), so B0 was retained (CASE C). |

To instantiate the deployed model: load `frozen_b0_corrected_run1/GENE_PARAMETERS.csv` + `configs/ACROPHASE_PRIOR_FREEZE.json` into `anchorphase.model.katv2_engine` (see `tests/test_smoke_release.py` for the API pattern). Seed constant: 20261007.
