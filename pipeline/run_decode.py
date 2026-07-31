"""Pure decoding regime (no external solver).

Recovers a tour directly from the network's edge marginals mu. Reports
optimality gaps for six decode levels matching paper Table 1:

  L1       : greedy degree-2 repair from mu
  L2 x 1   : L1 + 2-opt local search (1 pass)
  L2 x 10  : L1 + 2-opt local search (10 passes)
  L2 x 100 : L1 + 2-opt local search (100 passes)
  L3       : Gumbel perturbation of mu, best-of-M greedy decodes
  L4       : best-of-M covariance-perturbed MAP rooted 1-tree + backbone-insertion repair

Internally we call `decode_tours_ablation` once per requested twoopt-pass
budget and read the relevant `raw_cost` / `lk_cost` entry. The portfolio
column reports the per-instance best across all enabled strategies.

Usage:
    python -m pipeline.run_decode \\
        --run_dir checkpoints/c2tsp_tsp100 \\
        --dataset ./tsp50_demo.txt \\
        --levels L1,L2x1,L2x10,L2x100,L3,L4 \\
        --output_json ./decode_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from tsp_onetree.data import ConcordeTSPDataset, MetricTSPDataset
from tsp_onetree.model import TSPEntropicOneTreeModel
from tsp_onetree.decode import decode_gumbel, decode_tours_ablation
from tsp_onetree.graph import build_candidate_mask
from tsp_onetree.rooting import (
    ensemble_root_outputs,
    relabel_inputs_to_root_zero,
    restore_node_matrix,
    validate_root,
)

from pipeline.run_lkh import _load_model


DECODE_LEVELS = ("L1", "L2x1", "L2x10", "L2x100", "L3", "L4")


def _reference_tour_cost(tour_np: np.ndarray, dist_np: np.ndarray) -> float:
    idx_from = tour_np
    idx_to = np.roll(tour_np, -1)
    return float(dist_np[idx_from, idx_to].sum())


def _twoopt_passes_for(level: str) -> int:
    if level == "L2x1":
        return 1
    if level == "L2x10":
        return 10
    if level == "L2x100":
        return 100
    return 0


def _extract_cost(out: dict, level: str, B: int) -> np.ndarray:
    """Pick the right cost array from a decode result for `level`."""
    if level == "L1":
        return out["s1_mu_greedy"]["raw_cost"]
    if level.startswith("L2x"):
        return out["s1_mu_greedy"]["lk_cost"]
    if level == "L3":
        return out["s4_gumbel_muM"]["raw_cost"]
    if level == "L4":
        return out["best_cost"]
    raise ValueError(f"Unknown decode level {level!r}")


def _summarize(level: str, costs: np.ndarray, opt_costs: np.ndarray) -> dict[str, Any]:
    gaps = 100.0 * (costs - opt_costs) / np.where(opt_costs > 0, opt_costs, 1.0)
    finite = np.isfinite(gaps)
    return {
        "level": level,
        "n_total": int(costs.size),
        "n_finite": int(finite.sum()),
        "gap_pct_mean": float(np.mean(gaps[finite])) if finite.any() else None,
        "gap_pct_median": float(np.median(gaps[finite])) if finite.any() else None,
        "gap_pct_max": float(np.max(gaps[finite])) if finite.any() else None,
        "gap_pct_min": float(np.min(gaps[finite])) if finite.any() else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    default_run_dir = Path(__file__).resolve().parent.parent / "checkpoints" / "c2tsp_tsp100"
    ap.add_argument("--run_dir", type=str, default=str(default_run_dir))
    ap.add_argument("--dataset", type=str, required=True,
                    help="Concorde-format .txt file or metric manifest.jsonl / directory.")
    ap.add_argument("--dataset_format", choices=("auto", "concorde", "metric"), default="auto",
                    help="Input format. auto detects a metric manifest or directory.")
    ap.add_argument("--levels", type=str, default="L1,L2x1,L2x10,L2x100,L3,L4",
                    help=f"Comma-list from {DECODE_LEVELS}")
    ap.add_argument("--take", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_gumbel_draws", type=int, default=20,
                    help="M for L3 (Gumbel perturbation best-of-M). Paper default = 20.")
    ap.add_argument("--gumbel_scale", type=float, default=0.20)
    ap.add_argument("--l4_num_draws", type=int, default=0,
                    help="M for L4 best-of-M covariance MAP-repair. "
                    "If <= 0, reuses --num_gumbel_draws.")
    ap.add_argument("--l4_noise_type", type=str, default="covariance",
                    help="Noise type for L4 sampled MAP-repair.")
    ap.add_argument("--l4_tau", type=float, default=0.20,
                    help="Temperature used by covariance-based L4 noise.")
    ap.add_argument("--root", type=int, default=0,
                    help="Zero-based original-city index used as the 1-tree root. "
                         "The model is relabeled internally so its fixed root remains index 0.")
    ap.add_argument("--root_ensemble_size", type=int, default=0,
                    help="0 preserves fixed-root inference. K>0 averages restored mu and "
                         "C_mod from model roots 0,...,K-1 before every decode; --root then "
                         "selects the decoder anchor root.")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--device", type=str,
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output_json", type=str, required=True)
    args = ap.parse_args()

    levels: list[str] = []
    for tok in args.levels.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok not in DECODE_LEVELS:
            raise ValueError(f"Unknown level {tok!r}; allowed: {DECODE_LEVELS}")
        levels.append(tok)
    if not levels:
        raise ValueError("--levels produced an empty list")

    device = torch.device(args.device)
    print(f"[decode] loading model from {args.run_dir}")
    model, cfg = _load_model(Path(args.run_dir).resolve(), device)

    print(f"[decode] loading dataset: {args.dataset}")
    take = args.take if args.take > 0 else None
    dataset_path = Path(args.dataset).expanduser()
    is_metric = args.dataset_format == "metric" or (
        args.dataset_format == "auto"
        and (dataset_path.is_dir() or dataset_path.name == "manifest.jsonl")
    )
    if is_metric:
        ds = MetricTSPDataset(path=dataset_path, take=take, skip=args.skip)
    else:
        ds = ConcordeTSPDataset(path=args.dataset, take=take, skip=args.skip)
    n = ds.num_cities
    min_n = n if n is not None else min(ds.sizes)
    root = validate_root(args.root, min_n)
    root_ensemble_size = int(args.root_ensemble_size)
    if not 0 <= root_ensemble_size <= min_n:
        ap.error(
            f"--root_ensemble_size must be in [0, {min_n}], got {root_ensemble_size}"
        )
    B_total = len(ds)
    size_desc = str(n) if n is not None else f"mixed sizes={list(ds.sizes)}"
    print(f"[decode] n={size_desc}, {B_total} instances")

    if is_metric:
        opt_costs = np.array(
            [float(ds.metadata(i)["reference_cost_int"]) for i in range(B_total)],
            dtype=np.float64,
        )
    else:
        coords_list = [ds.coords[i] for i in range(B_total)]
        dist_list = [ds.dist_matrices[i] for i in range(B_total)]
        opt_costs = np.array(
            [_reference_tour_cost(ds.opt_tours[i].numpy(), ds.dist_matrices[i].numpy())
             for i in range(B_total)],
            dtype=np.float64,
        )

    # 2-opt budgets we actually need: union of {0} and any L2xK.
    needed_2opt = sorted({_twoopt_passes_for(l) for l in levels})
    if not any(l == "L1" or l == "L3" or l == "L4" for l in levels) and 0 in needed_2opt:
        needed_2opt.remove(0)
    if not needed_2opt:
        needed_2opt = [0]
    print(f"[decode] running decode for twoopt budgets = {needed_2opt}")

    # Forward + decode in batches.
    cost_by_level: dict[str, np.ndarray] = {l: np.full(B_total, np.nan) for l in levels}
    t_decode_total = 0.0
    t_forward_total = 0.0
    ensemble_diagnostic_sums: dict[str, float] = {}
    ensemble_diagnostic_max: dict[str, float] = {}
    if is_metric and n is None:
        by_size: dict[int, list[int]] = {}
        for idx in range(B_total):
            by_size.setdefault(int(ds.metadata(idx)["n"]), []).append(idx)
        batch_index_groups = [
            indices[start:start + args.batch_size]
            for size in sorted(by_size)
            for indices in [by_size[size]]
            for start in range(0, len(indices), args.batch_size)
        ]
    else:
        batch_index_groups = [
            list(range(start, min(start + args.batch_size, B_total)))
            for start in range(0, B_total, args.batch_size)
        ]

    for batch_no, indices in enumerate(batch_index_groups, start=1):
        start = indices[0]
        end = indices[-1] + 1
        n_batch = n if n is not None else int(ds.metadata(start)["n"])
        root_batch = validate_root(args.root, n_batch)
        model_roots = list(range(root_ensemble_size)) if root_ensemble_size > 0 else [root_batch]
        if is_metric:
            items = [ds[i] for i in indices]
            coords_b = torch.stack([item[0] for item in items]).to(device)
            # The network sees unit-scale costs, while all decoded/reference
            # costs use the official integer matrix.
            dist_b = torch.stack([item[1] for item in items]).to(device)
            decode_dist_b = torch.stack([item[3] for item in items]).to(device=device, dtype=torch.float32)
        else:
            coords_b = torch.stack([coords_list[i] for i in indices]).to(device)
            dist_b = torch.stack([dist_list[i] for i in indices]).to(device)
            decode_dist_b = dist_b
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            if root_ensemble_size == 0:
                # Preserve the established fixed-root release-decoder path.
                coords_model, dist_model, perm = relabel_inputs_to_root_zero(coords_b, dist_b, root_batch)
                mu, _, _, aux = model(coords_model, dist_model, return_decode_aux=True)
            else:
                mu, C_mod, diagnostics = ensemble_root_outputs(
                    model, coords_b, dist_b, model_roots
                )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_forward_total += time.perf_counter() - t0

        if root_ensemble_size == 0:
            mu = restore_node_matrix(mu, perm)
            C_mod = restore_node_matrix(aux["C_mod"], perm)
            cand_mask = restore_node_matrix(aux["cand_mask"], perm)
            pair_prob = aux.get("pair_prob")
            if pair_prob is not None:
                pair_prob = restore_node_matrix(pair_prob, perm)
        else:
            # Do not borrow root-dependent candidate or root-pair auxiliaries
            # from one forward.  This release evaluator has num_root_pairs=0,
            # so pair_prob is unused by every L1--L4 level.
            cand_mask = build_candidate_mask(dist_b, int(model.candidate_k))
            C_mod = C_mod * (~torch.eye(n_batch, device=device, dtype=torch.bool)).unsqueeze(0)
            pair_prob = None
            batch_n = len(indices)
            for name, value in diagnostics.items():
                value_float = float(value.detach().item())
                if name.endswith("_max"):
                    ensemble_diagnostic_max[name] = max(
                        ensemble_diagnostic_max.get(name, float("-inf")), value_float
                    )
                else:
                    ensemble_diagnostic_sums[name] = (
                        ensemble_diagnostic_sums.get(name, 0.0) + value_float * batch_n
                    )

        # One decode_tours_ablation call per requested twoopt budget. For levels
        # that don't depend on twoopt (L1/L3 raw), the first call suffices.
        for budget in needed_2opt:
            need_gumbel = ("L3" in levels)
            t1 = time.perf_counter()
            out = decode_tours_ablation(
                mu, C_mod, cand_mask, decode_dist_b,
                root=root_batch,
                twoopt_passes=int(budget),
                num_root_pairs=0,
                num_gumbel_draws=int(args.num_gumbel_draws) if need_gumbel else 0,
                gumbel_scale=float(args.gumbel_scale),
                seed_base=int(args.seed + start * 7919),
                pair_prob=pair_prob,
                use_lk_alpha=False,
            )
            t_decode_total += time.perf_counter() - t1

            for level in levels:
                if level in cost_by_level and np.any(np.isnan(cost_by_level[level][indices])):
                    # Only fill in cells matching this budget (or budget-independent levels
                    # which take the first available result).
                    if (
                        (level == "L1" and budget == needed_2opt[0])
                        or (level == "L2x1" and budget == 1)
                        or (level == "L2x10" and budget == 10)
                        or (level == "L2x100" and budget == 100)
                        or (level == "L3" and budget == needed_2opt[0])
                    ):
                        cost_by_level[level][indices] = _extract_cost(out, level, len(indices))

            if "L4" in levels and budget == needed_2opt[0]:
                l4_draws = int(args.l4_num_draws) if int(args.l4_num_draws) > 0 else int(args.num_gumbel_draws)
                t2 = time.perf_counter()
                l4_out = decode_gumbel(
                    mu, C_mod, cand_mask, decode_dist_b,
                    root=root_batch,
                    twoopt_passes=0,
                    num_draws=l4_draws,
                    gumbel_scale=float(args.gumbel_scale),
                    seed_base=int(args.seed + start * 7919),
                    use_lk_alpha=False,
                    noise_type=str(args.l4_noise_type),
                    tau=float(args.l4_tau),
                    seed_split=0.0,
                    use_mu_repair=False,
                )
                t_decode_total += time.perf_counter() - t2
                cost_by_level["L4"][indices] = _extract_cost(l4_out, "L4", len(indices))

        print(f"[decode]   batch {batch_no}/{len(batch_index_groups)} (n={n_batch}) done")

    summaries = {l: _summarize(l, cost_by_level[l], opt_costs) for l in levels}
    output = {
        "dataset": str(Path(args.dataset).resolve()),
        "dataset_format": "metric" if is_metric else "concorde",
        "run_dir": str(Path(args.run_dir).resolve()),
        "n": n,
        "sizes": list(ds.sizes) if is_metric else [n],
        "root": root,
        "num_instances": B_total,
        "levels": levels,
        "num_gumbel_draws": int(args.num_gumbel_draws),
        "gumbel_scale": float(args.gumbel_scale),
        "l4_num_draws": int(args.l4_num_draws) if int(args.l4_num_draws) > 0 else int(args.num_gumbel_draws),
        "l4_noise_type": str(args.l4_noise_type),
        "l4_tau": float(args.l4_tau),
        "seed": int(args.seed),
        "t_forward_total_s": float(t_forward_total),
        "t_decode_total_s": float(t_decode_total),
        "t_forward_per_inst_ms": float(1000 * t_forward_total / max(B_total, 1)),
        "t_decode_per_inst_ms": float(1000 * t_decode_total / max(B_total, 1)),
        "summary": summaries,
        "per_instance": {
            l: {"cost": cost_by_level[l].tolist(),
                "opt_cost": opt_costs.tolist()} for l in levels
        },
    }
    if root_ensemble_size > 0:
        output["root_ensemble"] = {
            "size": root_ensemble_size,
            "model_roots": list(range(root_ensemble_size)) if root_ensemble_size > 0 else [root],
            "decoder_anchor_root": root,
            "c_mod_policy": "mean_restored_c_mod",
            "candidate_mask_policy": "canonical_metric_knn",
            "pair_prob_policy": "unused_num_root_pairs_zero",
            "diagnostics": {
                **{
                    name: value / max(B_total, 1)
                    for name, value in ensemble_diagnostic_sums.items()
                },
                **ensemble_diagnostic_max,
            },
        }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n[decode] wrote {out_path}")
    print("[decode] summary (gap %):")
    for lv in levels:
        s = summaries[lv]
        mean = s["gap_pct_mean"]
        med = s["gap_pct_median"]
        print(f"  {lv:>7}  mean={mean:.3f}%  median={med:.3f}%  (n={s['n_finite']}/{s['n_total']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
