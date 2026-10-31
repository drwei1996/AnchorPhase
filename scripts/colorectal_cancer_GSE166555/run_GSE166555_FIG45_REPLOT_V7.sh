#!/usr/bin/env bash
set -euo pipefail
cd "${ANCHORPHASE_PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
python -u FIG45_V7_COMPLETE_REPLOT.py \
  --root '16_FIG4_FIG5_APPLICATIONS/GSE166555' \
  --max-drugs-per-target 2 \
  --top-pathways 25
