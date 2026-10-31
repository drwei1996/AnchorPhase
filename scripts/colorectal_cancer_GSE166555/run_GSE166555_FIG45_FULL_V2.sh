#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT='16_FIG4_FIG5_APPLICATIONS/GSE166555'
DATA="$ROOT/anchorphase_project/data"
[ -s "$DATA/count_matrix.h5ad" ] || { echo 'Missing H5AD; build GSE166555 first'; exit 2; }
[ -s "$DATA/PREDICTIONS_BLINDED.csv.gz" ] || { echo 'Missing B0 inferred phases; run frozen B0 first'; exit 2; }
MODE="${1:---reuse}"
if [ "$MODE" = '--rebuild-fig4' ]; then
  python -u GSE166555_FIG4_DETERMINISTIC_COSINOR_V1.py --root "$ROOT"
  python -u GSE166555_FIG4_FUNCTIONAL_PHASE_V2.py --root "$ROOT"
elif [ "$MODE" != '--reuse' ]; then echo 'Use --reuse or --rebuild-fig4'; exit 2;fi
[ -s "$ROOT/fig4_DETERMINISTIC_COSINOR_v1/01_Normal_gene_rhythmicity.csv.gz" ] || { echo 'Missing Normal V1 gene table. Re-run with --rebuild-fig4'; exit 2; }
[ -s "$ROOT/fig4_DETERMINISTIC_COSINOR_v1/01_Tumor_gene_rhythmicity.csv.gz" ] || { echo 'Missing Tumor V1 gene table. Re-run with --rebuild-fig4'; exit 2; }
if [ ! -s "$ROOT/DRUG_DATABASES/CHEMBL_DRUG_TARGET_MECHANISMS.csv.gz" ]; then
  echo 'ChEMBL cache missing; downloading from official API...'
  python -u FIG45_02_DOWNLOAD_CHEMBL_FIXED.py --out "$ROOT/DRUG_DATABASES"
else echo 'Reusing existing ChEMBL mechanism database';fi
python -u FIG45_V2_REBUILD.py --root "$ROOT"
echo 'PIPELINE SUCCESS'
