# Data sources — AnchorPhase release

All raw data are public. NOTHING raw is redistributed in this repository.

| Role | Dataset | accession / link | Units | Preprocessing shipped |
|---|---|---|---|---|
| Development | Mouse aorta smooth-muscle (Tempo's dataset) | GEO **GSE206583**; also Auerbach et al., Nat Commun 2022 | 22,573 cells | raw 10x -> pooled prep via `scripts/data_preparation/task11_pooled_prep.py`; anchored-gene freeze `configs/GENESET_FREEZE.json` |
| External validation 1 | Mouse liver hepatocytes | GEO **GSE145197** | 18,378 cells, 10 samples, ZT 0–20 | sparse h5ad `gse145197_sparse.h5ad` (18,378 x 14,812); build from GEO matrix files with scanpy |
| External validation 2 | Mouse dorsal dermis | GEO **GSE223109** | 23,990 cells, 18 samples | sparse h5ad `gse223109_sparse.h5ad` (23,990 x 32,285) |
| Application | Human colorectal cancer, epithelial compartment — **Uhlitz et al. 2021, EMBO Mol Med, PMID 34409732** | GEO **GSE166555** | 31,247 cells (Normal 17,874 / Tumor 13,373), 11 matched patients | `build_GSE166555_epithelial_h5ad.py` selects epithelial cells from the GEO count matrix + author cell annotations -> `count_matrix.h5ad` (31,247 x 21,854) |
| Anchor priors | CORE17 acrophase priors | frozen in `configs/ACROPHASE_PRIOR_FREEZE.json` (origin: tempo `clock_aorta_acrophase_prior.csv`, SHA-256 9c4d28a7...) | 17 genes | fixed phi_g, never trained |
| Pathway universes | MSigDB **Hallmark v2026.1** and **Reactome (c2.cp) v2026.1**, human symbols | https://www.gsea-msigdb.org (registration; not redistributable) | 50 + 1,416 gene sets | GMTs read directly by the ORA scripts |
| Drug/mechanism | ChEMBL mechanism records | https://www.ebi.ac.uk/chembl (local export 2026-10-24/25) | curated target-molecule edges | `CHEMBL_DRUG_TARGET_MECHANISMS.csv.gz` in the analysis package; audited edges in `source_data/Figure5/ST09_CHEMBL_NETWORK_EDGES_AUDIT.csv` |

## Download sketches

```bash
# GSE145197 / GSE223109 (processed matrices)
#   -> https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE145197  (and GSE223109)
# build sparse h5ad with scanpy: genes x cells, raw UMI, gene symbols; keep all genes.

# GSE166555
#   -> https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE166555
bash scripts/colorectal_cancer_GSE166555/run_build_GSE166555_epithelial_h5ad.sh

# GSE206583 (development): 10x per-sample directories as in GEO supplementary.
```

## Derived artifacts shipped (small, self-contained)

- `data/derived/GSE{145197,223109}_B0_PREDICTIONS_BLINDED.csv.gz` — per-cell posterior point estimates from the frozen B0 refits (with `MODEL_FREEZE.json` hashes).
- `models/frozen_b0_corrected_run1/` — the frozen corrected-offset B0 parameters (17 genes x {m, A, log_alpha}; phi fixed from priors).
- Full posterior arrays (`PHASE_POSTERIOR.npz`, 6.9–8.8 MB each) were too large for this repo; regenerate via a refit or request from the authors. All paper statistics are recoverable from the shipped prediction CSVs and tables.

## Version notes

- MSigDB v2026.1 was used throughout; using another version changes the pathway universe.
- Ensembl GTF Mus_musculus.GRCm39.104 was used for symbol mapping during development preprocessing.
