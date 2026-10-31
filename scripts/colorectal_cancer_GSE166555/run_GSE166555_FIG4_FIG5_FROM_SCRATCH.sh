#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
# Assumes H5AD and frozen B0 phase inference outputs already exist.
# Does not refit B0; see project run_GSE149614_with_Tabula_B0.py if required.
python -u GSE166555_FIG4_DETERMINISTIC_COSINOR_V1.py
python -u GSE166555_FIG4_FUNCTIONAL_PHASE_V2.py
python -u FIG45_01_BUILD_TARGET_ATLAS.py
python -u FIG45_02_DOWNLOAD_CHEMBL.py
python -u FIG45_03_DRUG_PRIORITIZATION.py
printf '\nFIG4+FIG5 ALL STAGES SUCCESS\n'
