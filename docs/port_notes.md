# Port notes — what changed between the frozen analysis tree and this repository

Principle: **byte-exact copies** of all frozen code and frozen results. The only additions are packaging files (`pyproject.toml`, `src/anchorphase/__init__.py`, docs, tests). No frozen script was edited, no result regenerated, no parameter touched.

## File moves (original location -> release location)

| Frozen source (under katv2/) | Release location |
|---|---|
| `00_SERVER_FINAL_ARCHIVE/KATv2_PROJECT_FINAL_20261008_FULL/sckat_server_package/scripts/katv2_engine.py` | `src/anchorphase/model/katv2_engine.py` |
| `.../scripts/run_katv2_b0*.py` | `src/anchorphase/training/` |
| `.../scripts/run_katv2_{eval,report,comparators}.py`, `.../scripts/task11b_*.py`, `.../scripts/task12_*.py` | `src/anchorphase/evaluation/` |
| `.../scripts/fig_style_v2.py`, `fig3_shared.py` | `src/anchorphase/utils/` |
| `.../scripts/make_fig*_v2.py`, `make_fig{2,3}_final_panels.py`, `render_all_panels_v2.py` | `scripts/figure_reproduction/` |
| katv2 root `GSE166555_*.py`, `FIG4*.py`, `FIG45_V7_COMPLETE_REPLOT.py`, `FIG5*.py`, `build_GSE166555_epithelial_h5ad.py`, `run_GSE166555_*.sh` | `scripts/colorectal_cancer_GSE166555/` |
| `ANCHORPHASE_MANUSCRIPT_COMPLETION_20261028/02_NEW_BIOLOGICAL_SUPPLEMENTARY_FIGURES/CODE/supp_discovery_*.py` + runners | `scripts/supplementary_discovery/` |
| `.../07_SUPPLEMENTARY_TABLES/ST*.csv|json` | `source_data/` (per-figure folders) |
| `15_KATV2_MANUSCRIPT_FREEZE_20261009_v2/04_FINAL_REPORTS/TABLE_*.csv` + reports | `analyses/model_benchmarks/` |
| `.../12_KAT_V2_EXTERNAL_VALIDATION_20261008/TABLES/*` + verdict | `analyses/external_validation/` |
| `.../10_.../02_DATA_AND_PRIOR_FREEZE/{ACROPHASE_PRIOR_FREEZE,GENESET_FREEZE,PHASE_CONVENTION}.json` | `configs/` |
| `.../10A_.../05_B0_CORRECTED_RUN1/*` | `models/frozen_b0_corrected_run1/` (+2 copies in `configs/`) |
| `.../11B_.../01_B0_REFERENCE/{GENE_PARAMETERS.csv,PREDICTIONS_BLINDED.csv.gz,PREDICTION_SHA256.txt}` | `models/b0_reference_pooled/` |
| `.../11B_.../11_FINAL_FREEZE/FINAL_{MODEL_PARAMETERS,MODEL_SPEC,FALSIFICATION_SUMMARY}.json` | `models/safe_cpi_final_freeze/` |
| `16_FIG4_FIG5_APPLICATIONS/GSE166555/**` frozen result CSVs/JSON | `analyses/colorectal_cancer_GSE166555/` |
| `.../02_NEW_BIOLOGICAL_SUPPLEMENTARY_FIGURES/SUPP_DISCOVERY_0{1,2}/DATA/*.csv` | `source_data/SupplementaryFigure{4,5}/` |
| `.../environment/sckat_conda_env.yml` | `environment.yml` |
| final panel PNGs (freeze manifest 00_FIGURE_FREEZE_MANIFEST.csv) | `docs/panel_previews/` (reference copies) |

## Path behaviour of the shipped scripts (unchanged)

- `src/anchorphase/**` scripts keep their original relative-path resolution. They were executed with the server package as CWD/package root; see `docs/reproduction_guide.md` §C for how to lay out `<workdir>` to re-run them.
- No absolute local user paths remain in shipped code: frozen scripts were already ROOT-relative, and the few hardcoded analysis-root constants found in copied scripts were replaced by environment variables (`TABULA_PROJECT_ROOT` / `ANCHORPHASE_PROJECT_ROOT`) with repository-relative defaults. Every such edit is listed in `RELEASE_AUDIT/` (patch log); no algorithm or parameter was touched.
- Known intentional exclusions: `PHASE_POSTERIOR.npz` arrays (size), raw/processed h5ad matrices (redistribution), MSigDB GMTs (license), `external_repos/` comparator sources (scRITMO/Tempo/Cyclum/s-vae have their own upstream repositories; `environment.yml` pins the versions used).

## Historical-name disclosure

Legacy exploratory scripts named `AnchorPhase_HCC_*` exist in the research tree. They predate the dataset-identity correction (GSE166555 = colorectal cancer, not HCC) and are **not** part of the frozen figure pipeline; none are shipped. The shipped GSE166555 pipeline files never encoded the wrong disease label in their logic — only the legacy filenames did.
