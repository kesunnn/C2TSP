#!/usr/bin/env bash
# C2TSP + LKH-3 integration (paper Table 2, level H4 = best).
#
# Prerequisites:
#   1. pip install -e . from release/
#   2. Build LKH-3 (see docs/INSTALL.md), export LKH_BIN=/path/to/LKH
#   3. Generate a tiny dataset:
#        bash examples/quickstart_generate_data.sh
#      or supply your own Concorde-format .txt as DATASET=...

set -euo pipefail
LKH_BIN="${LKH_BIN:-/path/to/LKH}"
DATASET="${DATASET:-./demo_data/tsp50_lkh.txt}"

cd "$(dirname "$0")/.."

python -m pipeline.run_lkh \
  --run_dir ./checkpoints/c2tsp_tsp100 \
  --dataset "${DATASET}" \
  --lkh_bin "${LKH_BIN}" \
  --levels H0,H4 \
  --time_limit_s 30 \
  --output_json ./demo_lkh_results.json

echo
echo "Wrote ./demo_lkh_results.json"
echo "See the 'summary' field for per-level pct_reached_opt and gap_pct_mean."
