# AnchorPhase

**AnchorPhase: anchored negative-binomial inference of cellular and sample-level transcriptional phase from sparse single-cell RNA-seq data**

This repository is the complete code and data release for the AnchorPhase model-development paper (Methods & Resources article). It contains the frozen model implementation (KATv2-B0), the external-validation and falsification pipeline, the colorectal-cancer application (GSE166555), all figure-reproduction scripts, and the per-panel Source Data tables.

> **Companion paper.** A second paper, *"Lineage-associated organization of clock gene expression across human tissues"* (Tabula Sapiens circadian architecture), uses AnchorPhase-inferred phases as an external tool and ships in its own repository. The two releases are strictly separated; see §15.

---

## 1. What AnchorPhase is

AnchorPhase (internal development name **KATv2-B0, corrected-offset variant, task 10A**) infers a per-cell circadian phase posterior from sparse raw UMI counts of 17 anchored clock genes:

- Negative-binomial observation model on **raw UMI**: `x_ig ~ NB(mu_ig, kappa_g)` with `log mu_ig = o_i + beta_g + a_g * cos(theta_i - varphi_g)`;
- whole-transcriptome **median-1 size-factor offset** `o_i = log(L_i / median(L))` computed over all genes (not only anchors);
- **external acrophase priors** `varphi_g` for the CORE17 anchor genes — frozen from an external resource, never trained (see `configs/ACROPHASE_PRIOR_FREEZE.json`, SHA-256 recorded);
- **48-point exact-grid posterior** over theta (exact E-step via logsumexp; no sampling);
- full-batch L-BFGS M-step (strong Wolfe, float64);
- posterior outputs: `phase_mean_h`, `phase_MAP_h`, posterior resultant length `R`, circular variance, posterior entropy.

The paper's headline validation result is a **falsification-gated deployment decision**: the higher-capacity SAFE-CPI variant (C2) passed internal development gates but **failed the external falsification gate F2 on GSE223109** (blood-contamination capture), so the anchored B0 model was retained (CASE C). This honest negative result is part of the paper, and all supporting tables are included.

## 2. Paper this repository supports

| Item | Value |
|---|---|
| Title | AnchorPhase: anchored negative-binomial inference of cellular and sample-level transcriptional phase from sparse single-cell RNA-seq data |
| Type | Methods & Resources article |
| Figures | 5 main (24 panels: 1A–D, 2A–D, 3A–D, 4A–E, 5A–G) + supplementary S1–S5 |
| Frozen model | KATv2-B0 corrected offset (task 10A, frozen 2026-10-07/08) |
| DOI / archive | `ZENODO-DOI-PLACEHOLDER` (fill after upload; see UPLOAD_INSTRUCTIONS.md in the release audit) |

## 3. Input data and access

No raw count matrices are redistributed. All four datasets are public GEO series:

| Role | Accession | Cells | Notes |
|---|---|---|---|
| Development | GSE206583 (mouse aorta SMC) | 22,573 | Tempo's dataset; raw 10x |
| External validation 1 | GSE145197 (mouse liver hepatocytes) | 18,378 | 10 samples, ZT-labelled |
| External validation 2 | GSE223109 (mouse dorsal dermis) | 23,990 | 18 samples, ZT-labelled |
| Application | **GSE166555 (human colorectal cancer; Uhlitz et al. 2021)** | 31,247 | epithelial compartment; 11 matched Normal/Tumor patients |

Exact download commands, build steps for each `.h5ad`, and versions are in [`docs/data_sources.md`](docs/data_sources.md).

**Important historical note.** Several legacy scripts retain the filename prefix `AnchorPhase_HCC_*`. GSE166555 is colorectal cancer, not hepatocellular carcinoma; the disease label was corrected during manuscript finalization (verified independently against the original publication, PMID 34409732). The frozen analysis code itself is unaffected; only legacy filenames carry the outdated label. No `AnchorPhase_HCC_*` script is part of the final figure pipeline shipped here.

## 4. Installation

```bash
# conda (reproduces the frozen server environment, python 3.11.17 / torch 2.14.1+cpu)
conda env create -f environment.yml
conda activate anchorphase

# or pip
python -m pip install -r requirements.txt   # minimal pins
python -m pip install -e .                  # install the anchorphase package
```

CPU-only. A single fit of ~20k cells x 17 genes takes minutes on one core; the engine pins `torch.set_num_threads(1)` by design.

## 5. Minimal running example

```bash
python tests/test_smoke_release.py
```

This runs a synthetic 400-cell inference with the frozen engine (import from `anchorphase.model.katv2_engine`), checks the posterior output shapes/ranges, and loads the frozen prior/config JSONs — all offline, no dataset download needed.

To load the engine directly:

```python
from anchorphase.model import katv2_engine as eng
import json, numpy as np
priors = json.load(open("configs/ACROPHASE_PRIOR_FREEZE.json"))  # 17 anchored genes, fixed phi_g
# engine API: see katv2_engine.py (class KATv2AnchoredNB / fit / predict)
```

## 6. Full analysis workflow

| Stage | Entry point | Frozen outputs shipped |
|---|---|---|
| Data preparation (pooled dev set) | `scripts/data_preparation/task11_pooled_prep.py` | — |
| Model training (frozen B0 corrected) | `src/anchorphase/training/run_katv2_b0_corrected_offset.py` | `models/frozen_b0_corrected_run1/` (parameters, config, freeze hashes, train history) |
| Development benchmark / model family | `analyses/model_benchmarks/` tables | `ST04_MODEL_PERFORMANCE_DEVELOPMENT.csv`, `TABLE_ROBUSTNESS.csv`, `CLAIM_MATRIX.csv` |
| CPI / SAFE-CPI comparators | `src/anchorphase/evaluation/task11b_*.py` | frozen verdicts in `models/safe_cpi_final_freeze/` |
| External validation (blinded, Task 12) | `src/anchorphase/evaluation/task12_pipeline.py` | `analyses/external_validation/` (9 tables + final verdict) |
| Figures 1–3 | `scripts/figure_reproduction/make_fig*_v2.py`, `make_fig{2,3}_final_panels.py` | `docs/panel_previews/` |
| GSE166555 application (Figs 4–5, S1–S3) | `scripts/colorectal_cancer_GSE166555/` (bash drivers + python) | `analyses/colorectal_cancer_GSE166555/` |
| Supplementary S4–S5 (discovery) | `scripts/supplementary_discovery/RUN_ALL_SUPPLEMENTARY_DISCOVERY.bat` | `source_data/SupplementaryFigure{4,5}/` |

The frozen figure scripts were executed from the analysis package root, where each resolves `ROOT = Path(__file__).resolve().parent.parent` and reads frozen artifacts under `ROOT/results/...` and `ROOT/data/...`. In this repository the same relative layout does not exist by default (raw data is not redistributed), so re-rendering figures requires first downloading the datasets and materializing `data/processed/<GSE>/...` as described in `docs/reproduction_guide.md`. All **numbers** shown on every panel are independently verifiable from the shipped Source Data tables without any rerun.

## 7. Figure ↔ script ↔ data map

Full mapping (panel by panel, 24 main + 3 supplementary figure rows): [`docs/figure_code_mapping.csv`](docs/figure_code_mapping.csv). Summary:

| Figure | Panels | Producing script(s) | Source data shipped |
|---|---|---|---|
| Fig 1 | 1A–1D (schematics) | `scripts/figure_reproduction/make_fig1{a,b,c,d}_*_v2.py` | manual schematics (formulas frozen in scripts) |
| Fig 2 | 2A–2D | `make_fig2_final_panels.py` | `source_data/Figure2/ST03*` |
| Fig 3 | 3A–3D | `make_fig3_final_panels.py` | `source_data/Figure3/ST05*` |
| Fig 4 | 4A–4E | `scripts/colorectal_cancer_GSE166555/GSE166555_FIG4_DETERMINISTIC_COSINOR_V1.py`, `FIG4A_alluvial_final.py`, `FIG4D_GSE166555_UNIQUE_RHYTHMIC_ORA.py`, `FIG4D_TumorOnly_Top4_GeneHeatmap.py` | `source_data/Figure4/`, `analyses/colorectal_cancer_GSE166555/` |
| Fig 5 | 5A–5G | `FIG45_V7_COMPLETE_REPLOT.py`, `FIG5E_NETWORK_MATCH_FIG5B.py` | `source_data/Figure5/` |
| Supp S1 | clock-output curves | `GSE166555_FIG4_DETERMINISTIC_COSINOR_V1.py` (12-gene subset of the 17x2 parent) | `analyses/colorectal_cancer_GSE166555/04_CORE17_curves_long.csv` |
| Supp S2 | all-pathway paired harmonic | `GSE166555_ALL_PATHWAYS_NOMINAL_P_ONLY.py` | `source_data/SupplementaryFigure2/` |
| Supp S3 | top-4 tumor-only curves | `FIG4D_TumorOnly_Top4.py` | (frozen CSVs in the analysis package; ST tables cover statistics) |
| Supp S4 | gene-level remodeling | `scripts/supplementary_discovery/supp_discovery_01_*.py` | `source_data/SupplementaryFigure4/` |
| Supp S5 | program coordination | `supp_discovery_02_figure.py` | `source_data/SupplementaryFigure5/` |

## 8. Source Data location

`source_data/Figure1..Figure5/` and `source_data/SupplementaryFigure{1..5}/` — comma-separated tables with headers and units, keyed to panels. The full 20-table supplementary-table set (ST02–ST10C) is under `source_data/` in its original numbering; `docs/figure_code_mapping.csv` states which ST table backs which panel.

## 9. Frozen versions

- Model: `KATv2-B0-corrected-offset` (task 10A, runs 1–3; run 1 parameters shipped in `models/frozen_b0_corrected_run1/`, with SHA-256 freeze hashes).
- Anchor priors: `configs/ACROPHASE_PRIOR_FREEZE.json` (17 genes, mouse symbols; SHA-256 recorded inside).
- Reference pooled refit: `models/b0_reference_pooled/` (task 11B).
- SAFE-CPI final freeze (the falsified-away variant, kept for transparency): `models/safe_cpi_final_freeze/`.
- Per-cohort external predictions: `data/derived/GSE{145197,223109}_B0_PREDICTIONS_BLINDED.csv.gz` (+ freeze hashes).

## 10. Random seeds and statistics

- Engine perturbation seed: `20261007` (constant in code).
- GSE166555 bootstrap replicates: 2,000, seed `20261023`; sign-flip Monte Carlo: 10,000.
- EM/L-BFGS fits are deterministic (full-batch).
- Blinding protocol: external-cohort predictions frozen and SHA-256-hashed **before** ZT unblinding (`analyses/external_validation/EXTERNAL_PREDICTION_HASHES.csv`).
- Multiple testing: BH within condition across eligible genes / across both GMT collections and all gene sets (see `docs/reproduction_guide.md` for the exact family definitions per panel).

## 11. Compute requirements

- CPU only; 1 thread per fit (pinned in the engine). 16 GB RAM comfortable; genome-wide GSE166555 fits peak at ~2–3 GB.
- Full development benchmark family (tasks 03–11B) ≈ 1–2 core-days; external validation (Task 12) ≈ 4–8 core-hours; GSE166555 Fig 4–5 chain ≈ 1–2 core-days; S4/S5 discovery ≈ 1–2 minutes.

## 12. Reproduction limits and open items

- Raw matrices are not redistributed; the GSE166555 epithelial `.h5ad` build step (`build_GSE166555_epithelial_h5ad.py`) requires a ~5 GB download.
- The figure-reproduction scripts are byte-identical to the frozen run but expect the original package-root layout; re-rendering requires materializing `data/` and selected `results/` paths first (`docs/reproduction_guide.md` §C).
- Figure 1 panels 1A–1D are manual schematics generated by their scripts with frozen constants (flagged `MANUAL_SCHEMATIC` in the provenance register); they re-render but are not data-driven.
- Zenodo DOI: placeholder until upload (`ZENODO-DOI-PLACEHOLDER`).

## 13. Citation

```bibtex
@software{anchorphase2026,
  title  = {AnchorPhase: anchored negative-binomial inference of cellular and
            sample-level transcriptional phase from sparse single-cell RNA-seq data},
  author = {AnchorPhase authors},
  year   = {2026},
  version = {1.0.0},
  doi    = {ZENODO-DOI-PLACEHOLDER},
  url    = {https://github.com/PLACEHOLDER-ORG/AnchorPhase}
}
```

## 14. License

Code: MIT (`LICENSE`). Derived statistical tables: CC BY 4.0. Third-party data: original submitters' terms (not redistributed); see `LICENSE` scope note and `docs/data_sources.md`.

## 15. Relationship to the Tabula Sapiens clock-architecture paper

The companion paper *"Lineage-associated organization of clock gene expression across human tissues"* uses AnchorPhase(KATv2-B0) inferred phases (60-cohort phase atlas) as an external tool for its Figure 6 functional-module analyses. That paper's repository declares AnchorPhase as an external dependency pinned to this release; it does **not** duplicate the model-development benchmark system, and this repository does not contain any Tabula-analysis code. The two cite each other by versioned reference.
