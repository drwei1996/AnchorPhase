# data/ — derived artifacts only

Raw and processed count matrices are NOT redistributed (GEO terms + size). See `docs/data_sources.md` for accessions, download commands, and h5ad build steps.

Contents here:
- `derived/GSE145197_B0_PREDICTIONS_BLINDED.csv.gz` — frozen per-cell posterior phase outputs, liver validation cohort (sha256 in `..._MODEL_FREEZE.json` next to it).
- `derived/GSE223109_B0_PREDICTIONS_BLINDED.csv.gz` — same, dermis external test cohort.
- `derived/*_GENE_PARAMETERS.csv`, `derived/*_MODEL_FREEZE.json` — per-cohort frozen B0 refit parameters and hashes.

The full posterior grids (`PHASE_POSTERIOR.npz`) are not shipped (6.9–8.8 MB each x several); regenerate with a refit or ask the authors. All paper-level statistics are recoverable from the shipped CSVs.
