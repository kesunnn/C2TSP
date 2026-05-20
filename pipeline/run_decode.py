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

from tsp_onetree.data import ConcordeTSPDataset
from tsp_onetree.model import TSPEntropicOneTreeModel
from tsp_onetree.decode import decode_gumbel, decode_tours_ablation

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
                    help="Concorde-format .txt instance file.")
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
    ds = ConcordeTSPDataset(path=args.dataset, take=take, skip=args.skip)
    n = ds.num_cities
    B_total = len(ds)
    print(f"[decode] n={n}, {B_total} instances")

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
    for start in range(0, B_total, args.batch_size):
        end = min(start + args.batch_size, B_total)
        coords_b = torch.stack(coords_list[start:end]).to(device)
        dist_b = torch.stack(dist_list[start:end]).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            mu, _, _, aux = model(coords_b, dist_b, return_decode_aux=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_forward_total += time.perf_counter() - t0

        C_mod = aux["C_mod"]
        cand_mask = aux["cand_mask"]
        pair_prob = aux.get("pair_prob")

        # One decode_tours_ablation call per requested twoopt budget. For levels
        # that don't depend on twoopt (L1/L3 raw), the first call suffices.
        for budget in needed_2opt:
            need_gumbel = ("L3" in levels)
            t1 = time.perf_counter()
            out = decode_tours_ablation(
                mu, C_mod, cand_mask, dist_b,
                root=0,
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
                if level in cost_by_level and np.any(np.isnan(cost_by_level[level][start:end])):
                    # Only fill in cells matching this budget (or budget-independent levels
                    # which take the first available result).
                    if (
                        (level == "L1" and budget == needed_2opt[0])
                        or (level == "L2x1" and budget == 1)
                        or (level == "L2x10" and budget == 10)
                        or (level == "L2x100" and budget == 100)
                        or (level == "L3" and budget == needed_2opt[0])
                    ):
                        cost_by_level[level][start:end] = _extract_cost(out, level, end - start)

            if "L4" in levels and budget == needed_2opt[0]:
                l4_draws = int(args.l4_num_draws) if int(args.l4_num_draws) > 0 else int(args.num_gumbel_draws)
                t2 = time.perf_counter()
                l4_out = decode_gumbel(
                    mu, C_mod, cand_mask, dist_b,
                    root=0,
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
                cost_by_level["L4"][start:end] = _extract_cost(l4_out, "L4", end - start)

        print(f"[decode]   batch {start}/{B_total} done")

    summaries = {l: _summarize(l, cost_by_level[l], opt_costs) for l in levels}
    output = {
        "dataset": str(Path(args.dataset).resolve()),
        "run_dir": str(Path(args.run_dir).resolve()),
        "n": n,
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
