# C2TSP — Connected by Construction Inference Release

C2TSP is an unsupervised differentiable pipeline that learns near-tour
marginals for the Traveling Salesman Problem from a connected-by-construction
rooted 1-tree family. After a smoothed Held–Karp equilibration layer and a
certificate-guided sharpening step, the model produces an edge-marginal
matrix `μ` (and a node-dual vector `λ_nr`) that can either be **decoded
directly into a tour** or used to **steer LKH-3** as a candidate-and-initial-
tour oracle.

This repository contains the **inference code**, a **pretrained checkpoint**,
and **dataset-generation scripts** — enough to reproduce the headline
numbers in Tables 1 and 2 of the paper. Training code, baseline
comparisons, and pre-built test datasets are **not** included; see
"Limitations" below.

## Contents

```
release/
├── tsp_onetree/                 model + dataset loader
├── pipeline/
│   ├── run_decode.py            pure-decoding regime  (paper Table 1)
│   ├── run_lkh.py               LKH integration regime (paper Table 2)
│   ├── smoke_test.py            forward-only sanity check
│   └── lkh_integration.py       low-level LKH bridge (TSPLIB / CAND / PI files)
├── data_gen/                    Concorde / LKH dataset generators
├── checkpoints/c2tsp_tsp100/    pretrained weights + model config
├── examples/                    one-line quickstart scripts
└── docs/INSTALL.md              build instructions for LKH-3 / Concorde
```

## Install

See [`docs/INSTALL.md`](docs/INSTALL.md). Short version:

```bash
pip install -e .                # installs torch + numpy
# Build LKH-3 separately (only needed for the LKH integration regime
# and for the LKH-based dataset generator).
```

## Quickstart

```bash
# Generate a small 16-instance TSP50 demo set (uses LKH for labels).
export LKH_BIN=/absolute/path/to/LKH
bash examples/quickstart_generate_data.sh

# (a) Pure decoding — no external solver, no LKH binary needed at inference.
bash examples/quickstart_decode.sh

# (b) LKH integration — feeds C2TSP candidates + initial tour to LKH-3.
bash examples/quickstart_inference.sh
```

## Two evaluation regimes

The paper reports C2TSP under two complementary regimes.

### Regime 1 — Pure decoding (paper Table 1)

`python -m pipeline.run_decode --levels L1,L2x1,L2x10,L2x100,L3,L4 ...`

| Level    | What it does                                                     |
| -------- | ---------------------------------------------------------------- |
| **L1**   | greedy degree-2 tour from `μ` (with degree-2 repair)             |
| **L2×k** | L1 followed by 2-opt local search for `k ∈ {1, 10, 100}` passes  |
| **L3**   | best-of-M greedy decodes under Gumbel perturbation of `μ`        |
| **L4**   | MAP rooted 1-tree + backbone-insertion repair                    |

No LKH binary is required. Reported metric: optimality gap (%) vs Concorde
(or LKH ground truth).

### Regime 2 — LKH integration (paper Table 2)

`python -m pipeline.run_lkh --levels H0,H1,H2,H3,H4 ...`

| Level    | What is supplied to LKH-3                                        |
| -------- | ---------------------------------------------------------------- |
| **H0**   | nothing (vanilla LKH baseline)                                   |
| **H1**   | initial tour from L1                                             |
| **H2**   | top-5 candidate set per node (within 20-NN, ranked by `μ`)       |
| **H3**   | H2 reordered by 0.5·rank(`μ`) + 0.5·rank(`C_mod`) — needs `λ_nr` |
| **H4**   | H1 initial tour + H3 candidate set (paper default)               |

`C_mod[i,j] = D[i,j] − λ_i − λ_j` is the Held–Karp reduced cost. H3 is
C2TSP-only: it requires a non-root node dual that the heatmap-style
baselines do not produce. H4 is the full integration and produces the
strongest LKH numbers in the paper at every tested size.

## Pretrained checkpoint

`checkpoints/c2tsp_tsp100/` contains:

- `model.pt` — PyTorch state dict (~524 KB, 123,750 parameters).
- `run_config.json` — hyperparameters needed to reconstruct the architecture.

**Trained only on TSP100** (paper §3) and applied zero-shot to every test
size (TSP50, TSP100, TSP200, TSP500, TSP1000, TSP2000). No checkpoint
swapping or fine-tuning between sizes.

## Dataset format

Concorde-style, one instance per line:

```
x1 y1 x2 y2 ... xn yn output t1 t2 ... tn t1
```

Coordinates are floats in `[0, 1]`. The tour is 1-indexed and repeats the
start node at the end. The loaders convert to 0-indexed internally.

Two generators ship with this repo:

- `data_gen/generate_concorde_tsp_datasets.py` — exact optima via Concorde.
- `data_gen/generate_lkh_tsp_datasets.py` — near-optima via LKH (recommended
  when you already have LKH installed for the integration regime).

Both write the same line format.

## Limitations

- **No training code.** Only the forward pass, decoders, and the LKH bridge
  are shipped.
- **No baselines.** The paper compares against DIFUSCO, Fast-T2T, DIMES,
  UTSP, and NeuroLKH; each has its own license and install path and lives
  outside this repo.
- **No pre-built datasets.** Generate your own with the scripts in
  `data_gen/`; this avoids redistributing the Joshi et al. test files.
- **No bundled LKH / Concorde binaries.** Build them yourself per
  `docs/INSTALL.md`; the scripts take `--lkh_bin` and `--concorde_bin` as
  required arguments.
- **GPU optional, recommended.** Network forward is ~30 ms / instance on a
  modern GPU and several times slower on CPU.

## Citation

```bibtex
@inproceedings{c2tsp2026,
  title  = {Connected by Construction: Learning Tractable Near-Tour Marginals for Traveling Salesman Problems},
  author = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year   = {2026},
}
```

(BibTeX placeholder — update with the camera-ready citation when available.)

## License

MIT — see [LICENSE](LICENSE).
