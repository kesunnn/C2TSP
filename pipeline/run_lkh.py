"""LKH integration regime (C2TSP as candidate-and-initial-tour oracle for LKH-3).

Five integration levels (paper Table 2), all candidate selections restricted to
each node's 20 Euclidean-nearest neighbors. Our PI_FILE contains all zeros: it
is a technical no-op that routes LKH through the "preserve external candidates"
code path in CreateCandidateSet (without it, LKH calls GenerateCandidates and
overwrites our candidate file). lambda_nr is NOT used as PI; it is only used to
compute C_mod for H3/H4 ordering.

  H0   naive LKH                       — no warm start, no candidate file
  H1   initial tour only               — INITIAL_TOUR_FILE = best-of-K (from mu)
  H2   candidate set                   — top-5 by mu within 20-NN,
                                          alpha = (0, 100, 200, 300, 400) by mu order
  H3   candidate set, rank-reordered   — same 5 as H2 (C2TSP only; needs lambda),
                                          alpha by 0.5*rank_mu + 0.5*rank_Cmod
  H4   initial tour + H3 candidate set — combines H1's tour and H3's candidates

C_mod[i,j] = D[i,j] - lambda_i - lambda_j (reduced cost, lower = better).

LKH parameters at H2/H3/H4:
  - PI_FILE  = all zeros (just to bypass GenerateCandidates)
  - SUBGRADIENT = NO  (no ascent; our candidates kept verbatim)
  - MAX_CANDIDATES = 5 (matches our top-K and LKH default)
  - No MERGE_TOUR_FILE, no ASCENT_POLISH
At H0 / H1: no PI / no CAND, SUBGRADIENT = YES (LKH default).

Per-instance timing breakdown:
    t_model_amortized_s   network forward / num_instances
    t_decode_s            initial-tour sampling + decode (H1 / H4 only)
    t_setup_s             writing CAND / TOUR / zero-PI files
    t_subprocess_s        wall time for the LKH subprocess
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from tsp_onetree.data import ConcordeTSPDataset
from tsp_onetree.model_v19 import TSPEntropicOneTreeModel
from pipeline import lkh_integration as L


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


_TRIAL_RE = re.compile(r"\*\s+(\d+):\s+Cost\s*=\s*(-?\d+)")
LEVEL_NAMES = ("H0", "H1", "H2", "H3", "H4")
LEVEL_INDEX = {name: i for i, name in enumerate(LEVEL_NAMES)}

NEEDS_INIT_TOUR = {"H1", "H4"}
NEEDS_CAND = {"H2", "H3", "H4"}
NEEDS_LAMBDA = {"H3", "H4"}


def _load_model(run_dir: Path, device: torch.device) -> tuple[TSPEntropicOneTreeModel, dict]:
    cfg = json.loads((run_dir / "run_config.json").read_text())["args"]
    kw = {k: cfg[k] for k in MODEL_CTOR_KEYS if k in cfg}
    kw["gradient_checkpoint"] = bool(cfg.get("gradient_checkpoint", 0))
    model = TSPEntropicOneTreeModel(**kw).to(device)
    state = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def _parse_trial_trace(stdout: str, opt_cost_int: int) -> dict:
    trials: list[tuple[int, int]] = []
    for m in _TRIAL_RE.finditer(stdout):
        trials.append((int(m.group(1)), int(m.group(2))))
    trials_to_first = trials[0][0] if trials else None
    trials_to_opt: int | None = None
    best_seen = None
    for k, c in trials:
        if c <= opt_cost_int:
            trials_to_opt = k
            break
        best_seen = c if (best_seen is None or c < best_seen) else best_seen
    return {
        "trials_to_first_improvement": trials_to_first,
        "trials_to_opt": trials_to_opt,
        "num_star_trials_logged": len(trials),
        "best_seen_cost_int": best_seen,
    }


def _gumbel_topk_for_size(n: int, default_per_size: dict[int, int]) -> int:
    return int(default_per_size.get(int(n), max(8, min(32, int(round(n / 8))))))


def _decode_for_instance(
    mu_np: np.ndarray,
    dist_np: np.ndarray,
    *,
    num_samples: int,
    gumbel_scale: float,
    seed_base: int,
    inst_idx: int,
    root: int,
    twoopt_passes: int,
) -> tuple[list[list[int]], list[float], float]:
    t0 = time.perf_counter()
    tours, costs = L.gumbel_topk_sample_tours(
        mu_np, dist_np, k=int(num_samples),
        root=int(root), gumbel_scale=float(gumbel_scale),
        rng_seed=int(seed_base + 1000003 * int(inst_idx) + 7),
        twoopt_passes=int(twoopt_passes),
    )
    return tours, costs, time.perf_counter() - t0


def _build_candidate_for_level(
    level: str,
    *,
    mu_np: np.ndarray,
    dist_np: np.ndarray,
    lambda_np: np.ndarray | None,
    cand_knn: int,
    cand_top_k: int,
    root: int,
) -> list[list[int]]:
    mu_sym = 0.5 * (mu_np + mu_np.T)
    np.fill_diagonal(mu_sym, -np.inf)

    if level == "H2":
        return L.select_candidates_knn_topk(
            score=mu_sym, dist=dist_np,
            k_knn=int(cand_knn), top_k=int(cand_top_k),
            higher_is_better=True,
        )
    if level in ("H3", "H4"):
        nbrs = L.select_candidates_knn_topk(
            score=mu_sym, dist=dist_np,
            k_knn=int(cand_knn), top_k=int(cand_top_k),
            higher_is_better=True,
        )
        assert lambda_np is not None
        c_mod = L.compute_C_mod(dist_np, lambda_np, root=int(root))
        c_mod_sym = 0.5 * (c_mod + c_mod.T)
        np.fill_diagonal(c_mod_sym, np.inf)
        return [
            L.reorder_by_combined_rank(
                nbrs[i], mu_row=mu_sym[i], cmod_row=c_mod_sym[i],
                weight_mu=0.5, weight_cmod=0.5,
            )
            for i in range(mu_sym.shape[0])
        ]
    raise ValueError(f"_build_candidate_for_level called with unsupported level={level}")


def _run_level(
    level: str,
    *,
    inst_idx: int,
    coords_np: np.ndarray,
    int_coords: np.ndarray,
    dist_np: np.ndarray,
    mu_np: np.ndarray | None,
    lambda_np: np.ndarray | None,
    sampled_tours: list[list[int]] | None,
    opt_tour: np.ndarray,
    tsp_path: Path,
    work_dir: Path,
    lkh_bin: Path,
    n: int,
    cand_knn: int,
    cand_top_k: int,
    cand_max_candidates: int,
    max_trials: int,
    runs: int,
    time_limit_s: float,
    seed: int,
    t_decode_s: float,
) -> dict:
    par_path = work_dir / f"lkh_{level}.par"
    out_tour = work_dir / f"lkh_{level}.out"
    cand_path = work_dir / f"lkh_{level}.can" if level in NEEDS_CAND else None
    init_tour_path = work_dir / f"lkh_{level}.tour" if level in NEEDS_INIT_TOUR else None
    pi_path = work_dir / f"lkh_{level}.pi" if level in NEEDS_CAND else None

    t_setup0 = time.perf_counter()

    initial_tour = sampled_tours[0] if (level in NEEDS_INIT_TOUR and sampled_tours) else None
    if initial_tour is not None:
        L.write_initial_tour_file(init_tour_path, initial_tour, n=n, name=level)

    nbr_mean = nbr_max = None
    if level in NEEDS_CAND:
        assert mu_np is not None
        nbr_lists = _build_candidate_for_level(
            level, mu_np=mu_np, dist_np=dist_np, lambda_np=lambda_np,
            cand_knn=cand_knn, cand_top_k=cand_top_k, root=0,
        )
        nbr_mean, nbr_max = L.write_candidate_file_neurolkh_style(cand_path, nbr_lists)
        L.write_pi_file(pi_path, np.zeros(n - 1, dtype=np.float64),
                        root=0, n=n, alpha=1.0)

    t_setup_s = time.perf_counter() - t_setup0

    max_candidates = int(cand_max_candidates) if level in NEEDS_CAND else 5
    use_subgradient = level not in NEEDS_CAND

    params = L.LKHParams(
        problem_file=tsp_path,
        output_tour_file=out_tour,
        max_candidates=max_candidates,
        max_trials=max_trials,
        runs=runs,
        time_limit_s=time_limit_s,
        seed=seed,
        trace_level=1,
        subgradient=use_subgradient,
        ascent_polish=False,
    )
    if init_tour_path is not None and initial_tour is not None:
        params.initial_tour_file = init_tour_path
    if cand_path is not None:
        params.candidate_file = cand_path
    if pi_path is not None:
        params.pi_file = pi_path
    params.write(par_path)

    t_sub0 = time.perf_counter()
    result = L.run_lkh(lkh_bin, par_path, dimension=n, output_tour_path=out_tour)
    t_subprocess_s = time.perf_counter() - t_sub0

    opt_cost_norm = L.tour_cost_norm(coords_np, opt_tour)
    opt_cost_int = L.tour_cost_euc2d_int(int_coords, opt_tour)
    if result.tour is not None:
        lkh_cost_norm = L.tour_cost_norm(coords_np, result.tour)
        lkh_cost_int = L.tour_cost_euc2d_int(int_coords, result.tour)
    else:
        lkh_cost_norm = float("nan")
        lkh_cost_int = None
    gap_pct = (
        100.0 * (lkh_cost_norm - opt_cost_norm) / opt_cost_norm
        if opt_cost_norm > 0 and np.isfinite(lkh_cost_norm) else float("nan")
    )
    trial_info = _parse_trial_trace(result.stdout, opt_cost_int)

    lkh_solve_s_derived = None
    if result.total_time_s_reported is not None and result.ascent_time_s is not None:
        lkh_solve_s_derived = float(max(0.0,
            result.total_time_s_reported - result.ascent_time_s))

    return {
        "level": level,
        "level_idx": LEVEL_INDEX[level],
        "returncode": result.returncode,
        "t_decode_s": float(t_decode_s) if level in NEEDS_INIT_TOUR else 0.0,
        "t_setup_s": float(t_setup_s),
        "t_subprocess_s": float(t_subprocess_s),
        "lkh_total_s_reported": result.total_time_s_reported,
        "lkh_ascent_s_reported": result.ascent_time_s,
        "lkh_solve_s_derived": lkh_solve_s_derived,
        "lkh_cost_int": int(lkh_cost_int) if lkh_cost_int is not None else None,
        "lkh_cost_norm": float(lkh_cost_norm) if np.isfinite(lkh_cost_norm) else None,
        "opt_cost_int": int(opt_cost_int),
        "opt_cost_norm": float(opt_cost_norm),
        "gap_pct": float(gap_pct) if np.isfinite(gap_pct) else None,
        "reached_opt": bool(lkh_cost_int is not None and lkh_cost_int <= opt_cost_int),
        "cand_nbr_mean": nbr_mean,
        "cand_nbr_max": nbr_max,
        **trial_info,
        "stderr_head": result.stderr.strip()[:200] if result.stderr.strip() else "",
    }


def _batch_network_forward(
    model: TSPEntropicOneTreeModel,
    coords_list: list[torch.Tensor],
    dist_list: list[torch.Tensor],
    device: torch.device,
    batch_size: int,
    need_lambda: bool,
) -> tuple[list[np.ndarray], list[np.ndarray] | None, float]:
    total_t = 0.0
    mus: list[np.ndarray] = []
    lams: list[np.ndarray] | None = [] if need_lambda else None
    for start in range(0, len(coords_list), batch_size):
        end = min(start + batch_size, len(coords_list))
        coords_b = torch.stack(coords_list[start:end]).to(device)
        dist_b = torch.stack(dist_list[start:end]).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            mu, _, _, aux = model(coords_b, dist_b, return_decode_aux=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_t += time.perf_counter() - t0
        mu_np = mu.cpu().numpy()
        if lams is not None:
            lam_np = aux["lambda_nr"].cpu().numpy()
        for i in range(end - start):
            mus.append(mu_np[i])
            if lams is not None:
                lams.append(lam_np[i])
    return mus, lams, total_t


def _summarize(records: list[dict], levels: list[str]) -> dict:
    summary: dict[str, dict[str, Any]] = {}
    for level in levels:
        rs_all = [r for r in records if r["level"] == level]
        rs = [r for r in rs_all if r["returncode"] == 0]
        if not rs_all:
            continue
        gaps = [r["gap_pct"] for r in rs if r["gap_pct"] is not None]
        sub = [r["t_subprocess_s"] for r in rs]
        decode = [r["t_decode_s"] for r in rs]
        setup = [r["t_setup_s"] for r in rs]
        ascent = [r["lkh_ascent_s_reported"] for r in rs if r["lkh_ascent_s_reported"] is not None]
        solve = [r["lkh_solve_s_derived"] for r in rs if r["lkh_solve_s_derived"] is not None]
        t_to_opt = [r["trials_to_opt"] for r in rs if r["trials_to_opt"] is not None]
        t_first = [r["trials_to_first_improvement"] for r in rs if r["trials_to_first_improvement"] is not None]
        reached = [r["reached_opt"] for r in rs]
        summary[level] = {
            "n_total": len(rs_all),
            "n_ok": len(rs),
            "pct_reached_opt": float(100.0 * sum(reached) / max(len(reached), 1)),
            "gap_pct_mean": float(np.mean(gaps)) if gaps else None,
            "gap_pct_median": float(np.median(gaps)) if gaps else None,
            "gap_pct_max": float(np.max(gaps)) if gaps else None,
            "wall_s_mean": float(np.mean(sub)) if sub else None,
            "wall_s_median": float(np.median(sub)) if sub else None,
            "wall_s_sum": float(np.sum(sub)) if sub else None,
            "decode_s_mean": float(np.mean(decode)) if decode else None,
            "setup_s_mean": float(np.mean(setup)) if setup else None,
            "ascent_s_mean": float(np.mean(ascent)) if ascent else None,
            "solve_s_mean": float(np.mean(solve)) if solve else None,
            "trials_to_opt_mean": float(np.mean(t_to_opt)) if t_to_opt else None,
            "trials_to_opt_median": float(np.median(t_to_opt)) if t_to_opt else None,
            "trials_to_first_improvement_mean": float(np.mean(t_first)) if t_first else None,
        }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    default_run_dir = Path(__file__).resolve().parent.parent / "checkpoints" / "c2tsp_tsp100"
    ap.add_argument("--run_dir", type=str, default=str(default_run_dir),
                    help="Directory containing model.pt and run_config.json.")
    ap.add_argument("--dataset", type=str, required=True,
                    help="Path to a Concorde-format .txt instance file.")
    ap.add_argument("--lkh_bin", type=str, required=True,
                    help="Path to the LKH-3 executable (build it from "
                         "http://akira.ruc.dk/~keld/research/LKH-3/).")
    ap.add_argument("--levels", type=str, default="H0,H1,H2,H3,H4",
                    help="Comma list from {H0,H1,H2,H3,H4}.")
    ap.add_argument("--take", type=int, default=0,
                    help="Limit to first --take instances. 0 = all.")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--num_samples", type=int, default=0,
                    help="K for Gumbel-Top-K. 0 -> auto by size: 8/16/32 for n=50/100/200.")
    ap.add_argument("--gumbel_scale", type=float, default=0.20)
    ap.add_argument("--decode_twoopt_passes", type=int, default=4)

    ap.add_argument("--cand_knn", type=int, default=20)
    ap.add_argument("--cand_top_k", type=int, default=5)
    ap.add_argument("--cand_max_candidates", type=int, default=5)

    ap.add_argument("--max_trials", type=int, default=0,
                    help="LKH MAX_TRIALS. 0 -> use instance size n as default.")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--time_limit_s", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--coord_scale", type=int, default=L.DEFAULT_COORD_SCALE)
    ap.add_argument("--output_json", type=str, required=True)
    args = ap.parse_args()

    levels: list[str] = []
    for tok in args.levels.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok not in LEVEL_INDEX:
            raise ValueError(f"Unknown level {tok!r}; allowed: {LEVEL_NAMES}")
        levels.append(tok)
    if not levels:
        raise ValueError("--levels produced an empty list")
    levels = sorted(set(levels), key=lambda s: LEVEL_INDEX[s])
    device = torch.device(args.device)

    print(f"[pipeline] loading model from {args.run_dir}")
    model, cfg = _load_model(Path(args.run_dir).resolve(), device)

    print(f"[pipeline] loading dataset: {args.dataset}")
    take = args.take if args.take > 0 else None
    ds = ConcordeTSPDataset(path=args.dataset, take=take, skip=args.skip)
    n = ds.num_cities
    max_trials = args.max_trials if args.max_trials > 0 else n
    print(f"[pipeline] n={n}, {len(ds)} instances, max_trials={max_trials}")

    auto_K_by_size = {50: 8, 100: 16, 200: 32}
    num_samples = int(args.num_samples) if args.num_samples > 0 else _gumbel_topk_for_size(n, auto_K_by_size)

    print(f"[pipeline] num_samples={num_samples}, gumbel_scale={args.gumbel_scale}")
    print(f"[pipeline] candidate: knn={args.cand_knn}, top_k={args.cand_top_k}, "
          f"max_candidates={args.cand_max_candidates}")

    need_nn = any(l in (NEEDS_INIT_TOUR | NEEDS_CAND) for l in levels)
    need_decode = any(l in NEEDS_INIT_TOUR for l in levels)
    need_lambda = any(l in NEEDS_LAMBDA for l in levels)
    mus: list[np.ndarray] = []
    lams: list[np.ndarray] | None = None
    nn_total_time = 0.0
    if need_nn:
        print(f"[pipeline] running network forward on {len(ds)} instances "
              f"(batch={args.batch_size}, need_lambda={need_lambda})...")
        coords_list = [ds.coords[i] for i in range(len(ds))]
        dist_list = [ds.dist_matrices[i] for i in range(len(ds))]
        mus, lams, nn_total_time = _batch_network_forward(
            model, coords_list, dist_list, device,
            batch_size=args.batch_size, need_lambda=need_lambda,
        )
        per_inst_ms = 1000 * nn_total_time / max(len(ds), 1)
        print(f"[pipeline] network forward total: {nn_total_time:.2f}s  "
              f"amortized per instance: {per_inst_ms:.2f}ms")

    lkh_bin = Path(args.lkh_bin).resolve()
    if not lkh_bin.exists():
        raise FileNotFoundError(
            f"LKH binary not found at {lkh_bin}. Build LKH-3 from "
            f"http://akira.ruc.dk/~keld/research/LKH-3/ and pass --lkh_bin."
        )
    records: list[dict] = []
    t_loop0 = time.perf_counter()
    nn_per_inst_s = nn_total_time / max(len(ds), 1) if nn_total_time > 0 else 0.0
    with tempfile.TemporaryDirectory(prefix="lkh_pipe_") as td_shared:
        td_shared = Path(td_shared)
        for idx in range(len(ds)):
            coords = ds.coords[idx].numpy()
            dist_np = ds.dist_matrices[idx].numpy()
            opt_tour = ds.opt_tours[idx].numpy()
            tsp_path = td_shared / f"inst_{idx:05d}.tsp"
            int_coords = L.write_tsplib(tsp_path, coords, name=f"inst_{idx}",
                                        coord_scale=args.coord_scale)
            mu_np = mus[idx] if mus else None
            lam_np = lams[idx] if lams is not None else None

            sampled_tours: list[list[int]] | None = None
            t_decode_s = 0.0
            if need_decode and mu_np is not None:
                sampled_tours, _, t_decode_s = _decode_for_instance(
                    mu_np, dist_np,
                    num_samples=num_samples,
                    gumbel_scale=args.gumbel_scale,
                    seed_base=args.seed,
                    inst_idx=idx, root=0,
                    twoopt_passes=args.decode_twoopt_passes,
                )

            for level in levels:
                rec = _run_level(
                    level,
                    inst_idx=idx,
                    coords_np=coords, int_coords=int_coords, dist_np=dist_np,
                    mu_np=mu_np, lambda_np=lam_np,
                    sampled_tours=sampled_tours,
                    opt_tour=opt_tour,
                    tsp_path=tsp_path, work_dir=td_shared, lkh_bin=lkh_bin,
                    n=n,
                    cand_knn=args.cand_knn,
                    cand_top_k=args.cand_top_k,
                    cand_max_candidates=args.cand_max_candidates,
                    max_trials=max_trials,
                    runs=args.runs, time_limit_s=args.time_limit_s, seed=args.seed,
                    t_decode_s=t_decode_s,
                )
                rec["idx"] = idx
                rec["n"] = n
                rec["t_model_amortized_s"] = (
                    float(nn_per_inst_s) if level != "H0" else 0.0
                )
                records.append(rec)

            if (idx + 1) % max(1, len(ds) // 20) == 0 or (idx + 1) == len(ds):
                elapsed = time.perf_counter() - t_loop0
                print(f"[pipeline]   {idx + 1}/{len(ds)}  elapsed={elapsed:.1f}s")
    loop_time = time.perf_counter() - t_loop0

    summary = _summarize(records, levels)
    output = {
        "dataset": str(Path(args.dataset).resolve()),
        "run_dir": str(Path(args.run_dir).resolve()),
        "lkh_bin": str(lkh_bin),
        "n": n,
        "num_instances": len(ds),
        "levels": levels,
        "num_samples": num_samples,
        "gumbel_scale": args.gumbel_scale,
        "cand": {
            "knn": args.cand_knn,
            "top_k": args.cand_top_k,
            "max_candidates": args.cand_max_candidates,
        },
        "max_trials": max_trials,
        "runs": args.runs,
        "time_limit_s": args.time_limit_s,
        "seed": args.seed,
        "nn_total_time_s": nn_total_time,
        "nn_per_inst_mean_ms": 1000 * nn_total_time / max(len(ds), 1),
        "loop_total_time_s": loop_time,
        "summary": summary,
        "records": records,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n[pipeline] wrote {out_path}")
    print("[pipeline] summary:")
    for k, v in summary.items():
        print(f"  {k}: {json.dumps(v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
