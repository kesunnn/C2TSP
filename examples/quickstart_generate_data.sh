#!/usr/bin/env bash
# Generate a tiny TSP50 evaluation set (16 instances) using LKH as the
# ground-truth solver. Replace --lkh_bin with the absolute path to your LKH
# binary (see docs/INSTALL.md).

set -euo pipefail
LKH_BIN="${LKH_BIN:-/path/to/LKH}"

cd "$(dirname "$0")/.."

python -m data_gen.generate_lkh_tsp_datasets \
  --ns 50 \
  --count 16 \
  --base_seed 2000000 \
  --workers 4 \
  --lkh_bin "${LKH_BIN}" \
  --output_dir ./demo_data

echo
echo "Wrote ./demo_data/tsp50_lkh.txt (16 instances)."
echo "Pass it to examples/quickstart_inference.sh as --dataset."
