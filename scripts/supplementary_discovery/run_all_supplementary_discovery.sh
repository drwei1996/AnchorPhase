#!/usr/bin/env bash
# One-click reproduction of both new biological supplementary figures (WSL path version).
set -e
cd "$(dirname "$0")"
python -X utf8 supp_discovery_01_genome_wide_fit.py
python -X utf8 supp_discovery_01_analysis.py
python -X utf8 supp_discovery_01_ora.py
python -X utf8 supp_discovery_01_figure.py
python -X utf8 supp_discovery_02_figure.py
echo "ALL SUPPLEMENTARY DISCOVERY FIGURES REBUILT SUCCESSFULLY"
