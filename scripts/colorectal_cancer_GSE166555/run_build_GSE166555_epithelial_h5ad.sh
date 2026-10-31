#!/usr/bin/env bash
set -euo pipefail
ROOT="${ANCHORPHASE_PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
python -u build_GSE166555_epithelial_h5ad.py --data-dir "16_FIG4_FIG5_APPLICATIONS/GSE166555/data"
