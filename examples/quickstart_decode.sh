#!/usr/bin/env bash
# C2TSP pure-decoding regime (paper Table 1) — no external solver needed.
# Reports gap (%) for the six decode levels: L1, L2x{1,10,100}, L3, L4.
#
# Prerequisites:
#   1. pip install -e . from release/
#   2. Generate a tiny dataset (does need LKH for the LABELING step, not for
#      inference; alternatively use any Concorde-format .txt you have):
#        bash examples/quickstart_generate_data.sh
#      Or pass DATASET=/path/to/your_concorde.txt.

set -euo pipefail
DATASET="${DATASET:-./demo_data/tsp50_lkh.txt}"

cd "$(dirname "$0")/.."

python -m pipeline.run_decode \
  --run_dir ./checkpoints/c2tsp_tsp100 \
  --dataset "${DATASET}" \
  --levels L1,L2x1,L2x10,L2x100,L3,L4 \
  --num_gumbel_draws 20 \
  --output_json ./demo_decode_results.json

echo
echo "Wrote ./demo_decode_results.json"
echo "Pure-decoding gaps are printed in the summary section above; compare against paper Table 1."
