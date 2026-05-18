# Installation

## 1. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs `torch>=2.0` and `numpy>=1.23`. For CUDA-enabled PyTorch, install
the appropriate wheel from <https://pytorch.org/get-started/locally/> *before*
running `pip install -e .`.

The pipeline runs on CPU as well; the smoke test will pick `cuda` if available
and fall back to `cpu` otherwise.

## 2. LKH-3 (required for inference)

The inference pipeline shells out to the LKH-3 solver. Build it from source:

```bash
wget http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.14.tgz
tar xzf LKH-3.0.14.tgz
cd LKH-3.0.14
make
# The resulting executable is at ./LKH-3.0.14/LKH
```

Pass the absolute path to that executable to every script that needs it:

```bash
--lkh_bin /absolute/path/to/LKH-3.0.14/LKH
```

There is no autodetection — the flag is required by design so the repo never
silently picks up the wrong binary.

## 3. Concorde (optional, for data generation only)

You only need Concorde if you want exact optima for your evaluation set with
`data_gen/generate_concorde_tsp_datasets.py`. For most uses, the LKH-based
generator (`data_gen/generate_lkh_tsp_datasets.py`) is sufficient and uses the
solver you already installed in step 2.

If you do want Concorde:

```bash
# Download from http://www.math.uwaterloo.ca/tsp/concorde.html and follow the
# README. The executable you want is build/TSP/concorde.
--concorde_bin /absolute/path/to/concorde
```

## Sanity check

After installing, confirm the package imports and the checkpoint loads:

```bash
python -m pipeline.smoke_test \
  --run_dir checkpoints/c2tsp_tsp100 \
  --dataset /path/to/any/concorde_format.txt \
  --take 4 --device auto
```

If you don't have a dataset yet, generate one first with the
`examples/quickstart_generate_data.sh` script.
