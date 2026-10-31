# Reproduction guide — AnchorPhase release

This guide states exactly what was run, in what order, with what seeds, and what you must download to re-run each stage. Nothing in this repository regenerates frozen results: all shipped tables are the frozen outputs themselves.

## A. Environment

- Frozen server environment: `environment.yml` (conda, python 3.11.17, torch 2.14.1+cpu); full pip freeze in `docs/pip_freeze_server.txt`.
- Windows analysis environment used for the S4/S5 discovery figures: python 3.14.5, numpy 2.5.0, scipy 1.18.0, pandas 2.3.3, matplotlib 3.11.0 (no torch needed for those figures).
- R: MetaCycle used only in a legacy exploratory branch; the frozen figure pipeline is pure Python.

## B. Offline verification (no downloads)

```bash
pip install -e .
python tests/test_smoke_release.py     # engine import + synthetic 400-cell inference + config loads
python -m pytest tests/test_katv2_engine.py -q   # original frozen engine unit tests
```

## C. Stage-by-stage reproduction

### C1. Datasets (download required)

See `docs/data_sources.md` for GEO accessions, exact files, and the build commands for:
- `data/processed/GSE206583/...` (development; pooled prep: `scripts/data_preparation/task11_pooled_prep.py`)
- `data/processed/GSE145197/gse145197_sparse.h5ad` (18,378 x 14,812)
- `data/processed/GSE223109/gse223109_sparse.h5ad` (23,990 x 32,285)
- GSE166555 epithelial `count_matrix.h5ad` (31,247 x 21,854) via `scripts/colorectal_cancer_GSE166555/build_GSE166555_epithelial_h5ad.py`

The frozen scripts resolve their inputs relative to the analysis-package root (`ROOT = Path(__file__).resolve().parent.parent` for `scripts/*` files). To re-render Figures 1–3 with the shipped scripts, reproduce the original layout:

```
<workdir>/
├── scripts/make_fig2_final_panels.py   (copy from scripts/figure_reproduction/)
├── results/single_cell_phase_rebuild/12_KAT_V2_EXTERNAL_VALIDATION_20261008/TABLES/...
├── data/processed/GSE145197/gse145197_sparse.h5ad
└── ...
```

The frozen tables under `analyses/external_validation/` are exactly the files `make_fig2_final_panels.py` and `make_fig3_final_panels.py` read (place them at the `results/.../TABLES/` path). Blinded per-cell predictions are in `data/derived/`.

### C2. Training the frozen model from scratch

```bash
python src/anchorphase/training/run_katv2_b0_corrected_offset.py   # task 10A trainer
```

Deterministic full-batch fit; seed 20261007 (in code). Frozen artifacts of the shipped run: `models/frozen_b0_corrected_run1/` (parameters + config + freeze hashes + train history). A refit reproduces the parameters to numerical tolerance but not the file timestamps/hashes.

### C3. External validation (Task 12, blinded)

```bash
python src/anchorphase/evaluation/task12_pipeline.py --stage audit   # then b0, cpi, safe, freeze, falsify, eval
```

Order matters: predictions were frozen and hashed before the ZT blind map was opened (`analyses/external_validation/EXTERNAL_PREDICTION_HASHES.csv`). Final verdict: `TASK12_FINAL_VERDICT.json` — external validation FAILED for universal C2 deployment; anchored B0 retained (CASE C).

### C4. GSE166555 colorectal-cancer application (Figures 4–5, S1–S3)

```bash
cd scripts/colorectal_cancer_GSE166555
bash run_GSE166555_FIG4_FIG5_FROM_SCRATCH.sh   # end-to-end; individual stages below
# stage-by-stage (frozen versions):
python build_GSE166555_epithelial_h5ad.py      # needs the GEO download
python GSE166555_FIG4_DETERMINISTIC_COSINOR_V1.py   # Fig4A core curves + S1 12-gene subset + Fig4B inputs
python FIG4A_alluvial_final.py                      # Fig4B alluvial
python FIG4D_GSE166555_UNIQUE_RHYTHMIC_ORA.py       # Fig4C/D ORA (needs MSigDB v2026.1 GMTs)
python FIG4D_TumorOnly_Top4.py                      # S3 top-4 curves
python FIG4D_TumorOnly_Top4_GeneHeatmap.py          # Fig4E heatmaps
python GSE166555_ALL_PATHWAYS_NOMINAL_P_ONLY.py     # S2 all-pathway paired harmonic
python FIG45_V7_COMPLETE_REPLOT.py                  # Fig5A–F (v7 publication replot)
python FIG5E_NETWORK_MATCH_FIG5B.py                 # Fig5G ChEMBL network
```

MSigDB GMTs (`h.all.v2026.1.Hs.symbols.gmt`, `c2.cp.reactome.v2026.1.Hs.symbols.gmt`) must be downloaded from MSigDB (registration required; not redistributable). ChEMBL mechanism export: local export 2026-10-24/25 (see `docs/data_sources.md`).

### C5. Supplementary discovery figures S4–S5 (fast, offline given S4 inputs)

```bash
cd scripts/supplementary_discovery
RUN_ALL_SUPPLEMENTARY_DISCOVERY.bat   # or bash run_all_supplementary_discovery.sh
```

~1–2 minutes. Reads the frozen GSE166555 patient-level harmonic tables; regenerates S4/S5 exactly (bootstrap 2000, seed 20261023).

## D. Statistical conventions

- Circular absolute error (AE) in hours on a 24-h clock; medians over cells, and sample-level (n=10 liver, n=18 dermis) summaries weighted by posterior resultant length R.
- Permutation/falsification: 5 prespecified matched-control panels (NOT a sampled null; stated on panel 2B), phase permutation, rot8 wrong-anchor control.
- ORA: hypergeometric; BH q across BOTH Hallmark+Reactome collections and ALL tested gene sets.
- GSE166555 paired harmonic tests: exact paired sign-flip when <=15 pairs, else 10,000 Monte Carlo. Nominal P only in S2 (explicitly stated in the figure).
