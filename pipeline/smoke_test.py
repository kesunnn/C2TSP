#!/usr/bin/env python
"""Smoke test: load the pretrained checkpoint and confirm the forward pass
emits node duals (lambda_nr) and edge probabilities (mu) on a Concorde-format
dataset. Does not invoke LKH.

Usage:
    python -m pipeline.smoke_test \
        --run_dir checkpoints/c2tsp_tsp100 \
        --dataset ./tsp50_demo.txt \
        --take 8 --device auto
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from tsp_onetree.data import ConcordeTSPDataset
from tsp_onetree.model_v19 import TSPEntropicOneTreeModel


MODEL_CTOR_KEYS = [
    "node_dim", "edge_dim", "num_gnn_layers", "beta", "tau", "prior_weight",
    "candidate_k", "non_candidate_penalty", "lam_iters", "lam_tol", "lam_step",
    "ift_ridge", "loss_mode", "entropy_weight", "deg_penalty_weight",
    "resid_penalty_weight", "bern_penalty_weight", "logit_clamp", "root",
    "ift_backward_tol", "inner_homotopy", "inner_tau_start", "inner_tau_mid",
    "inner_final_frac", "cov_shrink", "lm_damping", "detach_refine_state",
    "round2_use_struct_gate", "round2_gate_detach_features",
    "round2_gate_hidden_dim", "round2_struct_gate_floor",
    "round2_struct_gate_temp", "round2_struct_bonus",
    "stage2_struct_target", "stage2_struct_linear_weight",
    "stage2_struct_quad_weight", "stage2_struct_uncertainty_weight",
    "stage2_entropy_penalty_weight", "nontour_entropy_weight",
    "stage2_objective_mode", "stage2_bound_weight", "round0_loss_weight",
    "sharpen_beta", "cert_alpha", "var_tilt_weight",
    "stage2_coupled_steps", "stage2_coupled_damping",
    "edge_head_with_cost", "edge_hidden_mult",
]


def _build_model_from_args(ckpt_args: dict) -> TSPEntropicOneTreeModel:
    kwargs = {k: ckpt_args[k] for k in MODEL_CTOR_KEYS if k in ckpt_args}
    kwargs["gradient_checkpoint"] = bool(ckpt_args.get("gradient_checkpoint", 0))
    return TSPEntropicOneTreeModel(**kwargs)


def _pick_device(preferred: str) -> torch.device:
    if preferred in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)


def _reference_tour_cost(tour: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    idx_from = tour
    idx_to = torch.roll(tour, -1)
    return dist[idx_from, idx_to].sum()


def main() -> int:
    ap = argparse.ArgumentParser()
    default_run_dir = Path(__file__).resolve().parent.parent / "checkpoints" / "c2tsp_tsp100"
    ap.add_argument("--run_dir", type=str, default=str(default_run_dir),
                    help="Directory with run_config.json and model.pt.")
    ap.add_argument("--ckpt_name", type=str, default="model.pt")
    ap.add_argument("--dataset", type=str, required=True,
                    help="Concorde-format .txt instance file.")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--take", type=int, default=8)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    ckpt_path = run_dir / args.ckpt_name
    cfg_path = run_dir / "run_config.json"
    if not ckpt_path.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")
    if not cfg_path.exists():
        raise SystemExit(f"Missing run_config.json under {run_dir}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    ckpt_args = cfg["args"]

    device = _pick_device(args.device)
    print(f"[smoke] device = {device}")
    print(f"[smoke] checkpoint = {ckpt_path}")
    print(f"[smoke] dataset    = {args.dataset}")

    model = _build_model_from_args(ckpt_args).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=True)
    print(f"[smoke] state_dict loaded; missing={len(missing) if missing else 0} "
          f"unexpected={len(unexpected) if unexpected else 0}")
    model.eval()

    dataset = ConcordeTSPDataset(path=args.dataset, take=int(args.take))
    print(f"[smoke] loaded {len(dataset)} instances (n={dataset.num_cities})")

    coords = torch.stack([dataset.coords[i] for i in range(len(dataset))]).to(device)
    dist = torch.stack([dataset.dist_matrices[i] for i in range(len(dataset))]).to(device)
    opt_tours = [dataset.opt_tours[i] for i in range(len(dataset))]

    with torch.no_grad():
        mu, loss, stats, aux = model(coords, dist, return_decode_aux=True)

    B, n, _ = mu.shape
    lambda_nr = aux["lambda_nr"]
    pair_prob = aux["pair_prob"]
    C_theta = aux["C_theta"]
    C_mod = aux["C_mod"]
    cand_mask = aux["cand_mask"]
    tau = aux["tau"]

    mu_row_sum = mu.sum(dim=-1)
    mu_sym_err = (mu - mu.transpose(-2, -1)).abs().max().item()
    full_mask = ~torch.eye(n, device=mu.device, dtype=torch.bool)
    mu_min_off = mu.masked_select(full_mask.unsqueeze(0).expand_as(mu)).min().item()
    mu_max = mu.max().item()

    ref_cost = torch.stack([_reference_tour_cost(opt_tours[b], dist[b]) for b in range(B)])
    relax_cost_per = 0.5 * (dist * mu).sum(dim=(-2, -1))

    print("\n=== Output tensor shapes ===")
    print(f"  mu         : {tuple(mu.shape)}   (edge probability, batched)")
    print(f"  lambda_nr  : {tuple(lambda_nr.shape)}   (node dual, non-root, n-1 values)")
    print(f"  pair_prob  : {tuple(pair_prob.shape)}   (root pair probs)")
    print(f"  C_theta    : {tuple(C_theta.shape)}")
    print(f"  C_mod      : {tuple(C_mod.shape)}")
    print(f"  cand_mask  : {tuple(cand_mask.shape)}   (nnz per instance = {int(cand_mask[0].sum())})")
    print(f"  tau        : {tau}")

    print("\n=== Sanity checks ===")
    print(f"  mu symmetric-err (max abs)     : {mu_sym_err:.3e}")
    print(f"  mu in [0,1]?                   : min={mu_min_off:.3e} max={mu_max:.3e}")
    print(f"  mu row-sum mean (per node)     : {mu_row_sum.mean().item():.4f}   "
          f"(expect ~2 on non-root once converged)")
    print(f"  mu row-sum std                 : {mu_row_sum.std().item():.4e}")
    print(f"  lambda_nr norm mean            : {lambda_nr.norm(dim=-1).mean().item():.4f}")
    print(f"  lambda_nr abs max              : {lambda_nr.abs().max().item():.4f}")

    print("\n=== Relaxation vs Concorde optimum (first 4 instances) ===")
    for i in range(min(B, 4)):
        rc = float(relax_cost_per[i].item())
        oc = float(ref_cost[i].item())
        gap = 100.0 * (rc - oc) / oc if oc > 0 else float("nan")
        print(f"  [{i}] relax_cost={rc:.4f}  opt_cost={oc:.4f}  gap={gap:+.2f}%")

    print("\n[smoke] mu (edge probability) and lambda_nr (node dual) produced successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
