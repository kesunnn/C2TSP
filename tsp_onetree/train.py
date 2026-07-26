import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ConcordeTSPDataset, TSPDataset, cosine_anneal
# Preserve the decoder used for training-time model selection.  The public
# inference decoder has evolved independently since the reference experiment.
from .training_decode import decode_gumbel, decode_gumbel_hybrid
from .model import TSPEntropicOneTreeModel
from .onetree import build_inner_tau_path
from .repro import make_torch_generator, sample_batch_permutations


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_safe(value.item())
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")


def _gap_stats(costs: list[float], refs: list[float]) -> tuple[float, float]:
    if not costs or not refs or len(costs) != len(refs):
        return float("nan"), float("nan")
    cost_arr = np.asarray(costs, dtype=np.float64)
    ref_arr = np.asarray(refs, dtype=np.float64)
    valid = np.isfinite(cost_arr) & np.isfinite(ref_arr) & (ref_arr > 0.0)
    if not bool(valid.any()):
        return float("nan"), float("nan")
    gaps = 100.0 * (cost_arr[valid] - ref_arr[valid]) / ref_arr[valid]
    return float(np.mean(gaps)), float(np.std(gaps))


def _stat_float(stats: dict, key: str, default: float = 0.0) -> float:
    value = stats.get(key, default)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float(default)
        return float(value.detach().mean().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _build_decode_lengths_payload(
    *,
    epoch: int,
    job_name: str,
    result_dir: Path,
    checkpoint_path: Path,
    full_decode: dict,
) -> dict:
    return {
        "epoch": epoch,
        "job_name": job_name,
        "result_dir": result_dir,
        "checkpoint_path": checkpoint_path,
        "deterministic_decoder": {
            "name": "decode_gumbel.det_cost",
            "tour_lengths": full_decode["decoded_costs"],
            "mean": float(np.mean(full_decode["decoded_costs"])) if full_decode["decoded_costs"] else None,
        },
        "sampled_decoder": {
            "name": "decode_gumbel.best_cost",
            "tour_lengths": full_decode["sampled_best_costs"],
            "mean": float(np.mean(full_decode["sampled_best_costs"])) if full_decode["sampled_best_costs"] else None,
        },
        "sampled_mean_decoder": {
            "name": "decode_gumbel.mean_cost",
            "tour_lengths": full_decode["sampled_mean_costs"],
            "mean": float(np.mean(full_decode["sampled_mean_costs"])) if full_decode["sampled_mean_costs"] else None,
        },
        "reference": {
            "tour_lengths": full_decode["reference_costs"],
            "mean": float(np.mean(full_decode["reference_costs"])) if full_decode["reference_costs"] else None,
        },
        "num_instances": full_decode["num_instances"],
    }


def _create_result_dir(job_name: str, results_root: str | Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(results_root) / f"{job_name}_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=False)
    return result_dir


def _stage2_struct_target_for_epoch(args, epoch: int) -> float:
    if args.stage2_struct_target_start is None or args.stage2_struct_target_anneal_epochs is None:
        return float(args.stage2_struct_target)
    anneal_epochs = int(args.stage2_struct_target_anneal_epochs)
    if anneal_epochs <= 0:
        return float(args.stage2_struct_target)
    return float(
        cosine_anneal(
            epoch,
            float(args.stage2_struct_target_start),
            float(args.stage2_struct_target),
            anneal_epochs,
        )
    )


def _effective_stage2_steps(args) -> int:
    if getattr(args, "stage2_steps", None) is not None:
        return max(0, int(args.stage2_steps))
    if int(getattr(args, "num_refine_rounds", 2)) <= 1:
        return 0
    legacy_coupled = max(0, int(getattr(args, "stage2_coupled_steps", 0)))
    return 1 if legacy_coupled == 0 else legacy_coupled

def random_permute_batch(
    coords: torch.Tensor,
    dist: torch.Tensor,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly permute city labels independently for each instance.

    This preserves the Euclidean instance while removing the artificial meaning
    of the fixed root index 0. With root=0 in solver coordinates, a random
    permutation makes that root correspond to a random original city.
    """
    B, n, _ = coords.shape
    device = coords.device
    perms = sample_batch_permutations(B, n, device, generator)
    coords_perm = coords.gather(1, perms.unsqueeze(-1).expand(-1, -1, coords.shape[-1]))
    row_idx = perms.unsqueeze(-1).expand(-1, -1, n)
    dist_perm = dist.gather(1, row_idx)
    col_idx = perms.unsqueeze(1).expand(-1, n, -1)
    dist_perm = dist_perm.gather(2, col_idx)
    return coords_perm, dist_perm


def unpack_batch(batch):
    if len(batch) == 2:
        coords, dist = batch
        opt_tour = None
    elif len(batch) == 3:
        coords, dist, opt_tour = batch
    else:
        raise ValueError(f"Unexpected batch structure of length {len(batch)}")
    return coords, dist, opt_tour


def permute_tour_batch(opt_tour: torch.Tensor, perms: torch.Tensor) -> torch.Tensor:
    """Convert original-city tours to the permuted indexing used by the solver."""
    if opt_tour is None:
        return None
    inv = torch.empty_like(perms)
    inv.scatter_(1, perms, torch.arange(perms.size(1), device=perms.device).unsqueeze(0).expand_as(perms))
    return inv.gather(1, opt_tour)


def tour_cost_from_order(D: torch.Tensor, tours: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Compute closed-tour cost for a batch of tours on distance matrices D."""
    if isinstance(tours, np.ndarray):
        tours = torch.as_tensor(tours, device=D.device, dtype=torch.long)
    else:
        tours = tours.to(device=D.device, dtype=torch.long)
    if tours.dim() == 1:
        tours = tours.unsqueeze(0)
    nxt = torch.roll(tours, shifts=-1, dims=1)
    return D.gather(1, tours.unsqueeze(-1).expand(-1, -1, D.size(-1))).gather(2, nxt.unsqueeze(-1)).squeeze(-1).sum(dim=1)


def _tour_edge_set(tour: np.ndarray | list[int]) -> set[tuple[int, int]]:
    arr = np.asarray(tour, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return set()
    edges: set[tuple[int, int]] = set()
    for idx, u in enumerate(arr.tolist()):
        v = int(arr[(idx + 1) % arr.size])
        a, b = (int(u), int(v))
        if a > b:
            a, b = b, a
        edges.add((a, b))
    return edges


def _tour_edge_coverage_stats(
    decoded_tour: np.ndarray | list[int],
    optimal_tour: np.ndarray | list[int],
    dist_single: np.ndarray,
) -> tuple[float, float]:
    opt_edges = _tour_edge_set(optimal_tour)
    dec_edges = _tour_edge_set(decoded_tour)
    if not opt_edges:
        return float("nan"), float("nan")
    overlap = opt_edges & dec_edges
    unweighted = float(len(overlap)) / float(len(opt_edges))
    denom = float(sum(float(dist_single[u, v]) for u, v in opt_edges))
    if denom <= 0.0:
        return unweighted, float("nan")
    numer = float(sum(float(dist_single[u, v]) for u, v in overlap))
    return unweighted, numer / denom


def load_dataset_from_args(path: str | None, num_instances: int, num_cities: int, seed: int, take: int | None, skip: int):
    if path:
        ds = ConcordeTSPDataset(path, take=take, skip=skip)
        if ds.num_cities != num_cities:
            raise ValueError(f"Dataset at {path} has {ds.num_cities} cities, expected --num_cities {num_cities}.")
        return ds
    return TSPDataset(num_instances, num_cities, seed=seed)


def _set_decode_lam_iters(model, args) -> int:
    train_lam_iters = int(model.lam_iters)
    if int(getattr(args, "decode_lam_iters", 0)) > 0:
        model.lam_iters = int(args.decode_lam_iters)
    return train_lam_iters


def collect_full_decode_lengths(model, val_loader, args, device):
    decoded_costs = []
    sampled_mean_costs = []
    sampled_best_costs = []
    reference_costs = []
    decoded_seen = 0

    model.eval()
    train_lam_iters = _set_decode_lam_iters(model, args)
    try:
        with torch.no_grad():
            for batch in val_loader:
                coords, dist, opt_tour = unpack_batch(batch)
                coords, dist = coords.to(device), dist.to(device)
                opt_tour = opt_tour.to(device) if opt_tour is not None else None
                if bool(args.debug_eval_timing) and device.type == "cuda":
                    torch.cuda.synchronize(device)
                forward_start = time.perf_counter()
                mu, _, _, aux = model(coords, dist, return_decode_aux=True)
                if bool(args.debug_eval_timing) and device.type == "cuda":
                    torch.cuda.synchronize(device)
                if bool(args.debug_eval_timing):
                    forward_elapsed = time.perf_counter() - forward_start
                    batch_start = decoded_seen
                    batch_end = decoded_seen + coords.shape[0] - 1
                    print(
                        f"[eval-debug][full-save] forward batch {batch_start}-{batch_end} finished in {forward_elapsed:.3f}s",
                        flush=True,
                    )

                seed_base = args.decode_seed + decoded_seen
                decode_fn = decode_gumbel_hybrid if bool(args.decode_hybrid) else decode_gumbel
                decode_kwargs = dict(
                    root=args.root,
                    twoopt_passes=args.decode_twoopt_passes,
                    num_draws=args.decode_gumbel_M,
                    gumbel_scale=args.decode_gumbel_scale,
                    seed_base=seed_base,
                    use_lk_alpha=bool(args.decode_lk_alpha),
                    noise_type=args.decode_noise_type,
                    tau=float(model.tau),
                    seed_split=float(args.decode_seed_split),
                    use_mu_repair=bool(args.decode_mu_repair),
                    repair_mode=str(args.decode_repair_mode),
                )
                if bool(args.decode_hybrid):
                    decode_kwargs["num_workers"] = int(args.decode_workers)
                gbl = decode_fn(
                    mu,
                    aux["C_mod"],
                    aux.get("cand_mask") if aux.get("cand_mask") is not None else None,
                    dist,
                    **decode_kwargs,
                )
                decoded_costs.extend(gbl["det_cost"].tolist())
                sampled_mean_costs.extend(gbl["mean_cost"].tolist())
                sampled_best_costs.extend(gbl["best_cost"].tolist())

                if opt_tour is not None:
                    oc = tour_cost_from_order(dist, opt_tour).detach().cpu().numpy()
                    reference_costs.extend(oc.tolist())

                decoded_seen += coords.shape[0]
    finally:
        model.lam_iters = train_lam_iters

    return {
        "decoded_costs": decoded_costs,
        "sampled_mean_costs": sampled_mean_costs,
        "sampled_best_costs": sampled_best_costs,
        "reference_costs": reference_costs,
        "num_instances": len(decoded_costs),
    }


def collect_dataset_decode_metrics(model, data_loader, args, device):
    decoded_costs = []
    sampled_best_costs = []
    opt_costs = []
    edge_cov_greedy = []
    edge_cov_best = []
    edge_cost_cov_greedy = []
    edge_cost_cov_best = []
    top2_sum = 0.0
    relax_n = 0
    decoded_seen = 0

    model.eval()
    train_lam_iters = _set_decode_lam_iters(model, args)
    try:
        with torch.no_grad():
            for batch in data_loader:
                coords, dist, opt_tour = unpack_batch(batch)
                coords, dist = coords.to(device), dist.to(device)
                opt_tour = opt_tour.to(device) if opt_tour is not None else None
                batch_n = coords.shape[0]

                mu, _, stats, aux = model(coords, dist, return_decode_aux=True)
                top2_sum += _stat_float(stats, "top2_concentration") * batch_n
                relax_n += batch_n

                seed_base = args.decode_seed + decoded_seen
                decode_fn = decode_gumbel_hybrid if bool(args.decode_hybrid) else decode_gumbel
                decode_kwargs = dict(
                    root=args.root,
                    twoopt_passes=args.decode_twoopt_passes,
                    num_draws=args.decode_gumbel_M,
                    gumbel_scale=args.decode_gumbel_scale,
                    seed_base=seed_base,
                    use_lk_alpha=bool(args.decode_lk_alpha),
                    noise_type=args.decode_noise_type,
                    tau=float(model.tau),
                    seed_split=float(args.decode_seed_split),
                    use_mu_repair=bool(args.decode_mu_repair),
                    repair_mode=str(args.decode_repair_mode),
                )
                if bool(args.decode_hybrid):
                    decode_kwargs["num_workers"] = int(args.decode_workers)
                gbl = decode_fn(
                    mu,
                    aux["C_mod"],
                    aux.get("cand_mask") if aux.get("cand_mask") is not None else None,
                    dist,
                    **decode_kwargs,
                )
                decoded_costs.extend(gbl["det_cost"].tolist())
                sampled_best_costs.extend(gbl["best_cost"].tolist())

                if opt_tour is not None:
                    oc = tour_cost_from_order(dist, opt_tour).detach().cpu().numpy()
                    opt_costs.extend(oc.tolist())
                    opt_tour_np = opt_tour.detach().cpu().numpy()
                    dist_np = dist.detach().cpu().numpy()
                    det_tour_np = gbl["det_tour"]
                    best_tour_np = gbl["best_tour"]
                    for bi in range(batch_n):
                        cov_g, wcov_g = _tour_edge_coverage_stats(det_tour_np[bi], opt_tour_np[bi], dist_np[bi])
                        cov_b, wcov_b = _tour_edge_coverage_stats(best_tour_np[bi], opt_tour_np[bi], dist_np[bi])
                        edge_cov_greedy.append(cov_g)
                        edge_cov_best.append(cov_b)
                        edge_cost_cov_greedy.append(wcov_g)
                        edge_cost_cov_best.append(wcov_b)

                decoded_seen += batch_n
    finally:
        model.lam_iters = train_lam_iters

    decoded_avg = float(np.mean(decoded_costs)) if decoded_costs else float("nan")
    sampled_best_avg = float(np.mean(sampled_best_costs)) if sampled_best_costs else float("nan")
    opt_avg = float(np.mean(opt_costs)) if opt_costs else float("nan")
    top2_conc = top2_sum / float(max(1, relax_n))
    return {
        "avg_decoded_cost_greedy": decoded_avg,
        "avg_decoded_cost_bestM": sampled_best_avg,
        "avg_optimal_cost": opt_avg,
        "top2_concentration": top2_conc,
        "optimal_edge_coverage_greedy": float(np.mean(edge_cov_greedy)) if edge_cov_greedy else float("nan"),
        "optimal_edge_coverage_bestM": float(np.mean(edge_cov_best)) if edge_cov_best else float("nan"),
        "optimal_edge_cost_coverage_greedy": float(np.mean(edge_cost_cov_greedy)) if edge_cost_cov_greedy else float("nan"),
        "optimal_edge_cost_coverage_bestM": float(np.mean(edge_cost_cov_best)) if edge_cost_cov_best else float("nan"),
    }


def train(args):
    device_arg = str(getattr(args, "device", "auto")).strip().lower()
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(getattr(args, "device"))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"Requested --device {args.device}, but CUDA is not available.")
    result_dir = _create_result_dir(args.job_name, args.results_root)
    checkpoint_path = result_dir / Path(args.checkpoint_path).name
    last_checkpoint_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_last{checkpoint_path.suffix}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Results: {result_dir}")
    stage2_steps = _effective_stage2_steps(args)
    print(
        f"TSP-{args.num_cities} | tau={args.tau_start}->{args.tau} beta={args.beta} loss={args.loss_mode} "
        f"cand_k={args.candidate_k} prior_w={args.prior_weight} | proj=candidate_weighted | refine={1 if stage2_steps == 0 else 2}x-shared"
    )
    tau_path = build_inner_tau_path(
        args.tau,
        inner_homotopy=bool(args.inner_homotopy),
        inner_tau_start=args.inner_tau_start,
        inner_tau_mid=args.inner_tau_mid,
    )
    print(
        f"Lambda iters={args.lam_iters} tol={args.lam_tol:g} step={args.lam_step:g} | "
        f"IFT ridge={args.ift_ridge:g} | resid_pen={args.resid_penalty_weight} | "
        f"stage2={args.stage2_objective_mode} rb_w={args.stage2_bound_weight:g} "
        f"cert(target/start/anneal/lin/quad)={args.stage2_struct_target:.3f}/"
        f"{(args.stage2_struct_target_start if args.stage2_struct_target_start is not None else args.stage2_struct_target):.3f}/"
        f"{(args.stage2_struct_target_anneal_epochs if args.stage2_struct_target_anneal_epochs is not None else 0)}/"
        f"{args.stage2_struct_linear_weight:.3f}/{args.stage2_struct_quad_weight:.3f} "
        f"eta>={max(1.0, args.stage2_struct_uncertainty_weight):.3f} | "
        f"H2pen={args.stage2_entropy_penalty_weight:.3f} | ntH={args.nontour_entropy_weight:.3f} | "
        f"sharp={args.sharpen_beta:g}/{args.sharpen_anneal_epochs}@+{args.beta_delay_after_tau} "
        f"| cert_alpha={args.cert_alpha:g} | var_tilt={args.var_tilt_weight:g} | stage2_steps={stage2_steps} extra_coupled={args.stage2_coupled_steps}@{args.stage2_coupled_damping:g} "
        f"| logit_clamp={args.logit_clamp} | inner_tau_path={'/'.join(f'{t:.3f}' for t in tau_path)} "
        f"| final_frac={args.inner_final_frac:.2f} | cov_shrink={args.cov_shrink:g} | lm={args.lm_damping:g} "
        f"| detach_refine={int(bool(args.detach_refine_state))} "
        f"| st2_gnn={0 if bool(args.disable_stage2_gnn_forward) else 1} "
        f"| skip_st1_hk={int(bool(args.skip_stage1_hk))} "
        f"| grad_ckpt={int(bool(args.gradient_checkpoint))} | edge_hidden_mult={int(args.edge_hidden_mult)} "
        f"| decode M={args.decode_gumbel_M} sig={args.decode_gumbel_scale:g} noise={args.decode_noise_type} "
        f"lam_eval={args.decode_lam_iters if args.decode_lam_iters > 0 else args.lam_iters}"
    )
    print("-" * 72)

    train_data = load_dataset_from_args(
        args.train_dataset_path, args.num_train, args.num_cities, seed=args.seed,
        take=args.train_take, skip=args.train_skip,
    )
    val_data = load_dataset_from_args(
        args.val_dataset_path, args.num_val, args.num_cities, seed=args.seed + 1,
        take=args.val_take, skip=args.val_skip,
    )
    train_loader_gen = make_torch_generator(args.seed, device="cpu")
    train_perm_gen = make_torch_generator(args.seed + 1, device=device)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=train_loader_gen,
    )
    train_eval_loader = DataLoader(train_data, batch_size=args.val_batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.val_batch_size, shuffle=False, num_workers=0)

    model = TSPEntropicOneTreeModel(
        node_dim=args.node_dim,
        edge_dim=args.edge_dim,
        num_gnn_layers=args.num_gnn_layers,
        beta=args.beta,
        tau=args.tau,
        prior_weight=args.prior_weight,
        candidate_k=args.candidate_k,
        non_candidate_penalty=args.non_candidate_penalty,
        lam_iters=args.lam_iters,
        lam_tol=args.lam_tol,
        lam_step=args.lam_step,
        ift_ridge=args.ift_ridge,
        loss_mode=args.loss_mode,
        entropy_weight=args.entropy_weight,
        deg_penalty_weight=args.deg_penalty_weight,
        resid_penalty_weight=args.resid_penalty_weight,
        bern_penalty_weight=args.bern_penalty_weight,
        logit_clamp=args.logit_clamp,
        stage2_struct_target=args.stage2_struct_target,
        stage2_struct_linear_weight=args.stage2_struct_linear_weight,
        stage2_struct_quad_weight=args.stage2_struct_quad_weight,
        stage2_struct_uncertainty_weight=args.stage2_struct_uncertainty_weight,
        stage2_entropy_penalty_weight=args.stage2_entropy_penalty_weight,
        nontour_entropy_weight=args.nontour_entropy_weight,
        stage2_objective_mode=args.stage2_objective_mode,
        stage2_bound_weight=args.stage2_bound_weight,
        round0_loss_weight=args.round0_loss_weight,
        sharpen_beta=args.sharpen_beta,
        cert_alpha=args.cert_alpha,
        var_tilt_weight=args.var_tilt_weight,
        stage2_steps=stage2_steps,
        stage2_coupled_steps=args.stage2_coupled_steps,
        stage2_coupled_damping=args.stage2_coupled_damping,
        edge_quotient_projection=args.edge_quotient_projection,
        round2_use_struct_gate=args.round2_use_struct_gate,
        round2_gate_detach_features=args.round2_gate_detach_features,
        round2_gate_hidden_dim=args.round2_gate_hidden_dim,
        round2_struct_gate_floor=args.round2_struct_gate_floor,
        round2_struct_gate_temp=args.round2_struct_gate_temp,
        round2_struct_bonus=args.round2_struct_bonus,
        edge_head_with_cost=bool(args.edge_head_with_cost),
        edge_hidden_mult=args.edge_hidden_mult,
        root=args.root,
        ift_backward_tol=args.ift_backward_tol,
        inner_homotopy=args.inner_homotopy,
        inner_tau_start=args.inner_tau_start,
        inner_tau_mid=args.inner_tau_mid,
        inner_final_frac=args.inner_final_frac,
        cov_shrink=args.cov_shrink,
        lm_damping=args.lm_damping,
        detach_refine_state=args.detach_refine_state,
        disable_stage2_gnn_forward=args.disable_stage2_gnn_forward,
        skip_stage1_hk=args.skip_stage1_hk,
        gradient_checkpoint=bool(args.gradient_checkpoint),
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_decoded = float("inf")
    best_relax = float("inf")
    train_history = []
    test_history = []
    best_detailed_decode_path = result_dir / "best_decoded_tour_lengths.json"
    last_detailed_decode_path = result_dir / "last_decoded_tour_lengths.json"
    _write_json(
        result_dir / "run_config.json",
        {
            "args": vars(args),
            "device": str(device),
            "result_dir": result_dir,
            "checkpoint_path": checkpoint_path,
            "last_checkpoint_path": last_checkpoint_path,
            "best_decoded_tour_lengths_path": best_detailed_decode_path,
            "last_decoded_tour_lengths_path": last_detailed_decode_path,
        },
    )

    for epoch in range(1, args.epochs + 1):
        train_idx=0
        model.train()
        model.tau = cosine_anneal(epoch, args.tau_start, args.tau, args.tau_anneal_epochs)
        model.stage2_struct_target = _stage2_struct_target_for_epoch(args, epoch)
        if args.sharpen_beta <= 0.0 or stage2_steps <= 0:
            model.sharpen_beta = 0.0
        else:
            beta_start_epoch = int(args.tau_anneal_epochs) + int(args.beta_delay_after_tau)
            if epoch <= beta_start_epoch:
                model.sharpen_beta = 0.0
            elif args.sharpen_anneal_epochs > 0:
                local_epoch = epoch - beta_start_epoch
                model.sharpen_beta = cosine_anneal(
                    local_epoch,
                    0.0,
                    float(args.sharpen_beta),
                    args.sharpen_anneal_epochs,
                )
            else:
                model.sharpen_beta = float(args.sharpen_beta)
        ep_loss, ep_cost, ep_struct_active_pct, ep_top2_sum, ep_n = 0.0, 0.0, 0.0, 0.0, 0
        t0 = time.time()

        skipped_nonfinite = 0
        for batch in train_loader:
            train_idx+=1
            coords, dist, opt_tour = unpack_batch(batch)
            coords, dist = coords.to(device), dist.to(device)
            if args.random_train_permute:
                coords, dist = random_permute_batch(coords, dist, train_perm_gen)
            _, loss, stats = model(coords, dist)
            loss_mean = loss.mean()
            cost_mean = stats["cost"].mean()
            if not torch.isfinite(loss_mean):
                optimizer.zero_grad(set_to_none=True)
                skipped_nonfinite += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            loss_mean.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=False)
            if not torch.isfinite(total_norm):
                optimizer.zero_grad(set_to_none=True)
                skipped_nonfinite += 1
                continue
            bad_grad = False
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    bad_grad = True
                    break
            if bad_grad:
                optimizer.zero_grad(set_to_none=True)
                skipped_nonfinite += 1
                continue
            optimizer.step()
            ep_loss += loss_mean.item() * coords.shape[0]
            ep_cost += cost_mean.item() * coords.shape[0]
            ep_struct_active_pct += _stat_float(stats, "struct_target_active_percent") * coords.shape[0]
            ep_top2_sum += _stat_float(stats, "top2_concentration") * coords.shape[0]
            ep_n += coords.shape[0]

        scheduler.step()
        train_avg = ep_loss / ep_n
        train_cost_avg = ep_cost / ep_n
        train_struct_active_pct_avg = ep_struct_active_pct / ep_n
        train_top2_conc_avg = ep_top2_sum / ep_n
        dt = time.time() - t0
        train_history.append({
            "epoch": epoch,
            "train_loss": train_avg,
            "train_cost": train_cost_avg,
            "stage2_struct_target": float(model.stage2_struct_target),
            "stage2_struct_target_active_percent": train_struct_active_pct_avg,
            "top2_concentration": None,
            "top2_concentration_batch": train_top2_conc_avg,
            "avg_decoded_cost_greedy": None,
            "avg_decoded_cost_bestM": None,
            "avg_optimal_cost": None,
            "optimal_edge_coverage_greedy": None,
            "optimal_edge_coverage_bestM": None,
            "optimal_edge_cost_coverage_greedy": None,
            "optimal_edge_cost_coverage_bestM": None,
            "tau": float(model.tau),
            "skipped_nonfinite": skipped_nonfinite,
            "seconds": dt,
        })

        do_train_decode = (
            int(args.train_decode_every) > 0
            and ((epoch % int(args.train_decode_every) == 1) or (epoch == args.epochs))
        )
        if do_train_decode:
            train_decode_metrics = collect_dataset_decode_metrics(model, train_eval_loader, args, device)
            train_history[-1].update({
                "avg_decoded_cost_greedy": train_decode_metrics["avg_decoded_cost_greedy"] if math.isfinite(train_decode_metrics["avg_decoded_cost_greedy"]) else None,
                "avg_decoded_cost_bestM": train_decode_metrics["avg_decoded_cost_bestM"] if math.isfinite(train_decode_metrics["avg_decoded_cost_bestM"]) else None,
                "avg_optimal_cost": train_decode_metrics["avg_optimal_cost"] if math.isfinite(train_decode_metrics["avg_optimal_cost"]) else None,
                "top2_concentration": train_decode_metrics["top2_concentration"] if math.isfinite(train_decode_metrics["top2_concentration"]) else None,
                "optimal_edge_coverage_greedy": train_decode_metrics["optimal_edge_coverage_greedy"] if math.isfinite(train_decode_metrics["optimal_edge_coverage_greedy"]) else None,
                "optimal_edge_coverage_bestM": train_decode_metrics["optimal_edge_coverage_bestM"] if math.isfinite(train_decode_metrics["optimal_edge_coverage_bestM"]) else None,
                "optimal_edge_cost_coverage_greedy": train_decode_metrics["optimal_edge_cost_coverage_greedy"] if math.isfinite(train_decode_metrics["optimal_edge_cost_coverage_greedy"]) else None,
                "optimal_edge_cost_coverage_bestM": train_decode_metrics["optimal_edge_cost_coverage_bestM"] if math.isfinite(train_decode_metrics["optimal_edge_cost_coverage_bestM"]) else None,
            })

        do_val = (epoch == 1) or (epoch % args.val_every == 0) or (epoch == args.epochs)
        skip_tag = f" | skip {skipped_nonfinite}" if skipped_nonfinite else ""
        if not do_val:
            print(
                f"E{epoch:4d} | train {train_avg:.4f} | train_cost {train_cost_avg:.4f} | "
                f"Uact_tr {train_struct_active_pct_avg:.1f}% | tau {model.tau:.4f} | Utgt {model.stage2_struct_target:.3f}{skip_tag} | {dt:.1f}s"
            )
            continue

        model.eval()
        val_loss_sum, relax_cost_sum, relax_n = 0.0, 0.0, 0
        decoded_costs, sampled_mean_costs, sampled_best_costs, opt_costs = [], [], [], []
        edge_cov_greedy = []
        edge_cov_best = []
        edge_cost_cov_greedy = []
        edge_cost_cov_best = []
        deg_mismatch_sum = 0.0
        deg_mismatch_max_sum = 0.0
        residual_sum = 0.0
        residual_max_sum = 0.0
        struct_cert_sum = 0.0
        struct_cert_norm_sum = 0.0
        struct_mean_defect_sum = 0.0
        struct_mean_uncert_sum = 0.0
        struct_defect_cert_sum = 0.0
        struct_uncert_cert_sum = 0.0
        struct_excess_sum = 0.0
        struct_penalty_sum = 0.0
        struct_target_active_pct_sum = 0.0
        repair_bound_weight_sum = 0.0
        repair_upper_bound_sum = 0.0
        edge_drift_sum = 0.0
        sum_nr_sum = 0.0
        fnorm_sum = 0.0
        gnorm_sum = 0.0
        rootdeg_sum = 0.0
        top2_sum = 0.0
        outcand_sum = 0.0
        clamp_sum = 0.0
        sclamp_sum = 0.0
        ift_trusted_sum = 0.0
        gumbel_unique_frac_sum = 0.0
        gumbel_a_mean_sum = 0.0
        gumbel_a_best_sum = 0.0
        gumbel_b_mean_sum = 0.0
        gumbel_b_best_sum = 0.0
        proj_row_sum = 0.0
        proj_grand_sum = 0.0
        proj_removed_sum = 0.0
        proj_align_sum = 0.0
        proj_fb_sum = 0.0
        round0_cost_sum = 0.0
        round1_cost_sum = 0.0
        round1_update_rms_sum = 0.0
        round1_raw_rms_sum = 0.0
        round1_scale_sum = 0.0
        round1_free_scale_sum = 0.0
        stage2_steps_sum = 0.0
        stage2_delta_mu_sum = 0.0
        stage2_delta_mu_first_sum = 0.0
        stage2_delta_mu_last_sum = 0.0
        stage2_delta_mu_max_last_sum = 0.0
        stage2_delta_mu_contract_sum = 0.0
        stage2_cert_rms_sum = 0.0
        stage2_tilt_proj_rms_sum = 0.0
        stage2_tilt_proj_fb_sum = 0.0
        val_seen = 0
        decoded_seen = 0
        decode_infer_total = 0.0
        avg_rec_total = 0.0
        train_lam_iters = _set_decode_lam_iters(model, args)
        try:
            with torch.no_grad():
                val_idx=0
                for batch in val_loader:
                    val_idx+=1
                    coords, dist, opt_tour = unpack_batch(batch)
                    coords, dist = coords.to(device), dist.to(device)
                    opt_tour = opt_tour.to(device) if opt_tour is not None else None
                    batch_n = coords.shape[0]
                    decode_timer_start = time.perf_counter() if decoded_seen < args.decode_eval_limit else None
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    forward_start = time.perf_counter()
                    mu, vloss, stats, aux = model(coords, dist, return_decode_aux=True)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    forward_elapsed = time.perf_counter() - forward_start
                    if bool(args.debug_eval_timing):
                        batch_start = val_seen
                        batch_end = val_seen + batch_n - 1
                        print(
                            f"[eval-debug] forward batch {batch_start}-{batch_end} finished in {forward_elapsed:.3f}s",
                            flush=True,
                        )
                    val_loss_sum += vloss.sum().item()
                    relax_cost_sum += stats["cost"].sum().item()
                    relax_n += batch_n
                    deg_mismatch_sum += _stat_float(stats, "degree_mismatch") * batch_n
                    deg_mismatch_max_sum += _stat_float(stats, "degree_mismatch_max") * batch_n
                    residual_sum += _stat_float(stats, "solver_residual") * batch_n
                    residual_max_sum += _stat_float(stats, "solver_residual_max") * batch_n
                    struct_cert_sum += _stat_float(stats, "struct_cert") * batch_n
                    struct_cert_norm_sum += _stat_float(stats, "struct_cert_norm") * batch_n
                    struct_defect_cert_sum += _stat_float(stats, "struct_defect_cert") * batch_n
                    struct_uncert_cert_sum += _stat_float(stats, "struct_uncert_cert") * batch_n
                    struct_mean_defect_sum += _stat_float(stats, "struct_mean_defect") * batch_n
                    struct_mean_uncert_sum += _stat_float(stats, "struct_mean_uncert") * batch_n
                    struct_excess_sum += _stat_float(stats, "struct_excess") * batch_n
                    struct_penalty_sum += _stat_float(stats, "struct_penalty") * batch_n
                    struct_target_active_pct_sum += _stat_float(stats, "struct_target_active_percent") * batch_n
                    repair_bound_weight_sum += _stat_float(stats, "repair_bound_weight") * batch_n
                    repair_upper_bound_sum += _stat_float(stats, "repair_upper_bound") * batch_n
                    top2_sum += _stat_float(stats, "top2_concentration") * batch_n
                    outcand_sum += _stat_float(stats, "outside_candidate_mass") * batch_n
                    clamp_sum += _stat_float(stats, "logit_clamp_frac") * batch_n
                    sclamp_sum += _stat_float(stats, "scaled_clamp_frac") * batch_n
                    ift_trusted_sum += _stat_float(stats, "ift_trusted_frac") * batch_n
                    edge_drift_sum += _stat_float(stats, "edge_count_drift") * batch_n
                    sum_nr_sum += _stat_float(stats, "sum_nonroot_residual") * batch_n
                    fnorm_sum += _stat_float(stats, "F_nr_norm") * batch_n
                    gnorm_sum += _stat_float(stats, "G_norm") * batch_n
                    rootdeg_sum += _stat_float(stats, "root_degree_dev") * batch_n
                    proj_row_sum += _stat_float(stats, "proj_row_leak") * batch_n
                    proj_grand_sum += _stat_float(stats, "proj_grand_leak") * batch_n
                    proj_removed_sum += _stat_float(stats, "proj_removed_energy_frac") * batch_n
                    proj_align_sum += _stat_float(stats, "proj_lambda_align") * batch_n
                    proj_fb_sum += _stat_float(stats, "proj_fallback_frac") * batch_n
                    round0_cost_sum += _stat_float(stats, "round0_cost") * batch_n
                    round1_cost_sum += _stat_float(stats, "round1_cost") * batch_n
                    round1_update_rms_sum += _stat_float(stats, "round1_update_rms") * batch_n
                    round1_raw_rms_sum += _stat_float(stats, "round1_raw_rms") * batch_n
                    round1_scale_sum += _stat_float(stats, "round1_scale") * batch_n
                    round1_free_scale_sum += _stat_float(stats, "round1_free_scale") * batch_n
                    stage2_steps_sum += _stat_float(stats, "stage2_coupled_steps") * batch_n
                    stage2_delta_mu_sum += _stat_float(stats, "stage2_coupled_delta_mu") * batch_n
                    stage2_delta_mu_first_sum += _stat_float(stats, "stage2_delta_mu_first") * batch_n
                    stage2_delta_mu_last_sum += _stat_float(stats, "stage2_delta_mu_last") * batch_n
                    stage2_delta_mu_max_last_sum += _stat_float(stats, "stage2_delta_mu_max_last") * batch_n
                    stage2_delta_mu_contract_sum += _stat_float(stats, "stage2_delta_mu_contract", 1.0) * batch_n
                    stage2_cert_rms_sum += _stat_float(stats, "stage2_certificate_rms") * batch_n
                    stage2_tilt_proj_rms_sum += _stat_float(stats, "stage2_tilt_proj_rms") * batch_n
                    stage2_tilt_proj_fb_sum += _stat_float(stats, "stage2_tilt_proj_fallback_frac") * batch_n
                    if decoded_seen < args.decode_eval_limit:
                        take = min(batch_n, args.decode_eval_limit - decoded_seen)
                        seed_base = args.decode_seed + decoded_seen
                        decode_fn = decode_gumbel_hybrid if bool(args.decode_hybrid) else decode_gumbel
                        decode_kwargs = dict(
                            root=args.root,
                            twoopt_passes=args.decode_twoopt_passes,
                            num_draws=args.decode_gumbel_M,
                            gumbel_scale=args.decode_gumbel_scale,
                            seed_base=seed_base,
                            use_lk_alpha=bool(args.decode_lk_alpha),
                            noise_type=args.decode_noise_type,
                            tau=float(model.tau),
                            seed_split=float(args.decode_seed_split),
                            use_mu_repair=bool(args.decode_mu_repair),
                            repair_mode=str(args.decode_repair_mode),
                        )
                        if bool(args.decode_hybrid):
                            decode_kwargs["report_timing"] = bool(args.debug_eval_timing)
                            decode_kwargs["num_workers"] = int(args.decode_workers)
                        gbl = decode_fn(
                            mu[:take],
                            aux["C_mod"][:take],
                            aux.get("cand_mask")[:take] if aux.get("cand_mask") is not None else None,
                            dist[:take],
                            **decode_kwargs,
                        )
                        decode_infer_total += time.perf_counter() - decode_timer_start
                        decoded_costs.extend(gbl["det_cost"].tolist())
                        sampled_mean_costs.extend(gbl["mean_cost"].tolist())
                        sampled_best_costs.extend(gbl["best_cost"].tolist())
                        if "per_instance_seconds" in gbl:
                            avg_rec_total += float(forward_elapsed) * take + float(np.sum(gbl["per_instance_seconds"]))
                        elif "_timing" in gbl:
                            avg_rec_total += float(forward_elapsed) * take + float(gbl["_timing"]["total_s"])
                        else:
                            avg_rec_total += float(forward_elapsed) * take
                        gumbel_unique_frac_sum += float(gbl["unique_frac"]) * take
                        gumbel_a_mean_sum += float(np.mean(gbl["a_mean_cost"])) * take
                        gumbel_a_best_sum += float(np.mean(gbl["a_best_cost"])) * take
                        gumbel_b_mean_sum += float(np.mean(gbl["b_mean_cost"])) * take
                        gumbel_b_best_sum += float(np.mean(gbl["b_best_cost"])) * take
                        if opt_tour is not None:
                            oc = tour_cost_from_order(dist[:take], opt_tour[:take]).detach().cpu().numpy()
                            opt_costs.extend(oc.tolist())
                            opt_tour_np = opt_tour[:take].detach().cpu().numpy()
                            dist_np = dist[:take].detach().cpu().numpy()
                            det_tour_np = gbl["det_tour"]
                            best_tour_np = gbl["best_tour"]
                            for ti in range(take):
                                cov_g, wcov_g = _tour_edge_coverage_stats(det_tour_np[ti], opt_tour_np[ti], dist_np[ti])
                                cov_b, wcov_b = _tour_edge_coverage_stats(best_tour_np[ti], opt_tour_np[ti], dist_np[ti])
                                edge_cov_greedy.append(cov_g)
                                edge_cov_best.append(cov_b)
                                edge_cost_cov_greedy.append(wcov_g)
                                edge_cost_cov_best.append(wcov_b)
                        decoded_seen += take
                    val_seen += batch_n
        finally:
            model.lam_iters = train_lam_iters

        val_loss_avg = val_loss_sum / max(1, relax_n)
        relax_avg = relax_cost_sum / max(1, relax_n)
        decoded_avg = float(np.mean(decoded_costs)) if decoded_costs else float("nan")
        sampled_mean_avg = float(np.mean(sampled_mean_costs)) if sampled_mean_costs else float("nan")
        sampled_best_avg = float(np.mean(sampled_best_costs)) if sampled_best_costs else float("nan")
        opt_avg = float(np.mean(opt_costs)) if opt_costs else float("nan")
        ref_avg = opt_avg
        gap = 100.0 * (decoded_avg - ref_avg) / ref_avg if math.isfinite(ref_avg) and ref_avg > 0 else float("nan")
        gap_bestk = 100.0 * (sampled_best_avg - ref_avg) / ref_avg if sampled_best_costs and math.isfinite(ref_avg) and ref_avg > 0 else float("nan")
        gap_inst_mean, gap_inst_std = _gap_stats(decoded_costs, opt_costs)
        gapm_inst_mean, gapm_inst_std = _gap_stats(sampled_best_costs, opt_costs)
        deg_mismatch = deg_mismatch_sum / float(max(1, relax_n))
        deg_mismatch_max = deg_mismatch_max_sum / float(max(1, relax_n))
        solver_resid = residual_sum / float(max(1, relax_n))
        solver_resid_max = residual_max_sum / float(max(1, relax_n))
        struct_cert_avg = struct_cert_sum / float(max(1, relax_n))
        struct_cert_norm_avg = struct_cert_norm_sum / float(max(1, relax_n))
        struct_defect_cert_avg = struct_defect_cert_sum / float(max(1, relax_n))
        struct_uncert_cert_avg = struct_uncert_cert_sum / float(max(1, relax_n))
        struct_mean_defect_avg = struct_mean_defect_sum / float(max(1, relax_n))
        struct_mean_uncert_avg = struct_mean_uncert_sum / float(max(1, relax_n))
        struct_excess_avg = struct_excess_sum / float(max(1, relax_n))
        struct_penalty_avg = struct_penalty_sum / float(max(1, relax_n))
        struct_target_active_pct_avg = struct_target_active_pct_sum / float(max(1, relax_n))
        repair_bound_weight_avg = repair_bound_weight_sum / float(max(1, relax_n))
        repair_upper_bound_avg = repair_upper_bound_sum / float(max(1, relax_n))
        top2_conc = top2_sum / float(max(1, relax_n))
        outcand_mass = outcand_sum / float(max(1, relax_n))
        clamp_frac = clamp_sum / float(max(1, relax_n))
        scaled_clamp_frac = sclamp_sum / float(max(1, relax_n))
        ift_trusted_frac = ift_trusted_sum / float(max(1, relax_n))
        edge_drift = edge_drift_sum / float(max(1, relax_n))
        sum_nr_resid = sum_nr_sum / float(max(1, relax_n))
        fnorm_avg = fnorm_sum / float(max(1, relax_n))
        gnorm_avg = gnorm_sum / float(max(1, relax_n))
        rootdeg_avg = rootdeg_sum / float(max(1, relax_n))
        proj_row_avg = proj_row_sum / float(max(1, relax_n))
        proj_grand_avg = proj_grand_sum / float(max(1, relax_n))
        proj_removed_avg = proj_removed_sum / float(max(1, relax_n))
        proj_align_avg = proj_align_sum / float(max(1, relax_n))
        proj_fb_avg = proj_fb_sum / float(max(1, relax_n))
        round0_cost_avg = round0_cost_sum / float(max(1, relax_n))
        round1_cost_avg = round1_cost_sum / float(max(1, relax_n))
        round1_update_rms_avg = round1_update_rms_sum / float(max(1, relax_n))
        round1_raw_rms_avg = round1_raw_rms_sum / float(max(1, relax_n))
        round1_scale_avg = round1_scale_sum / float(max(1, relax_n))
        round1_free_scale_avg = round1_free_scale_sum / float(max(1, relax_n))
        stage2_steps_avg = stage2_steps_sum / float(max(1, relax_n))
        stage2_delta_mu_avg = stage2_delta_mu_sum / float(max(1, relax_n))
        stage2_delta_mu_first_avg = stage2_delta_mu_first_sum / float(max(1, relax_n))
        stage2_delta_mu_last_avg = stage2_delta_mu_last_sum / float(max(1, relax_n))
        stage2_delta_mu_max_last_avg = stage2_delta_mu_max_last_sum / float(max(1, relax_n))
        stage2_delta_mu_contract_avg = stage2_delta_mu_contract_sum / float(max(1, relax_n))
        stage2_cert_rms_avg = stage2_cert_rms_sum / float(max(1, relax_n))
        stage2_tilt_proj_rms_avg = stage2_tilt_proj_rms_sum / float(max(1, relax_n))
        stage2_tilt_proj_fb_avg = stage2_tilt_proj_fb_sum / float(max(1, relax_n))
        gumbel_unique_frac_avg = gumbel_unique_frac_sum / float(max(1, decoded_seen)) if decoded_seen > 0 else float("nan")
        gumbel_a_mean_avg = gumbel_a_mean_sum / float(max(1, decoded_seen)) if decoded_seen > 0 else float("nan")
        gumbel_a_best_avg = gumbel_a_best_sum / float(max(1, decoded_seen)) if decoded_seen > 0 else float("nan")
        gumbel_b_mean_avg = gumbel_b_mean_sum / float(max(1, decoded_seen)) if decoded_seen > 0 else float("nan")
        gumbel_b_best_avg = gumbel_b_best_sum / float(max(1, decoded_seen)) if decoded_seen > 0 else float("nan")
        decode_infer_avg = decode_infer_total / float(max(1, decoded_seen)) if decoded_seen > 0 else float("nan")
        avg_rec = avg_rec_total / float(max(1, decoded_seen)) if decoded_seen > 0 else float("nan")
        optimal_edge_coverage_greedy = float(np.mean(edge_cov_greedy)) if edge_cov_greedy else float("nan")
        optimal_edge_coverage_bestM = float(np.mean(edge_cov_best)) if edge_cov_best else float("nan")
        optimal_edge_cost_coverage_greedy = float(np.mean(edge_cost_cov_greedy)) if edge_cost_cov_greedy else float("nan")
        optimal_edge_cost_coverage_bestM = float(np.mean(edge_cost_cov_best)) if edge_cost_cov_best else float("nan")

        improved = False
        metric_for_ckpt = sampled_best_avg if sampled_best_costs else decoded_avg
        if decoded_costs:
            if metric_for_ckpt < best_decoded:
                best_decoded = metric_for_ckpt
                improved = True
        elif val_loss_avg < best_relax:
            best_relax = val_loss_avg
            improved = True

        if improved:
            torch.save(model.state_dict(), checkpoint_path)
            full_decode = collect_full_decode_lengths(model, val_loader, args, device)
            _write_json(
                best_detailed_decode_path,
                _build_decode_lengths_payload(
                    epoch=epoch,
                    job_name=args.job_name,
                    result_dir=result_dir,
                    checkpoint_path=checkpoint_path,
                    full_decode=full_decode,
                ),
            )

        extra = ""
        if sampled_best_costs:
            extra = (
                f" | meanM {sampled_mean_avg:.4f} | bestM {sampled_best_avg:.4f} "
                f"| gapM {gap_bestk:+.1f}% | uniq {gumbel_unique_frac_avg:.2f}"
            )

        test_history.append({
            "epoch": epoch,
            "val_loss": val_loss_avg,
            "relax": relax_avg,
            "stage2_struct_target": float(model.stage2_struct_target),
            "decoded": decoded_avg,
            "avg_decoded_cost_greedy": decoded_avg,
            "mean_k": sampled_mean_avg,
            "best_k": sampled_best_avg,
            "avg_decoded_cost_bestM": sampled_best_avg,
            "reference": ref_avg,
            "avg_optimal_cost": ref_avg,
            "gap_percent": gap,
            "gap_bestk_percent": gap_bestk,
            "gap_instance_mean_percent": gap_inst_mean if math.isfinite(gap_inst_mean) else None,
            "gap_instance_std_percent": gap_inst_std if math.isfinite(gap_inst_std) else None,
            "gapM_instance_mean_percent": gapm_inst_mean if math.isfinite(gapm_inst_mean) else None,
            "gapM_instance_std_percent": gapm_inst_std if math.isfinite(gapm_inst_std) else None,
            "optimal_edge_coverage_greedy": optimal_edge_coverage_greedy if math.isfinite(optimal_edge_coverage_greedy) else None,
            "optimal_edge_coverage_bestM": optimal_edge_coverage_bestM if math.isfinite(optimal_edge_coverage_bestM) else None,
            "optimal_edge_cost_coverage_greedy": optimal_edge_cost_coverage_greedy if math.isfinite(optimal_edge_cost_coverage_greedy) else None,
            "optimal_edge_cost_coverage_bestM": optimal_edge_cost_coverage_bestM if math.isfinite(optimal_edge_cost_coverage_bestM) else None,
            "decode_inference_total_seconds": decode_infer_total,
            "decode_inference_avg_seconds_per_instance": decode_infer_avg,
            "avg_rec_seconds_per_instance": avg_rec,
            "gumbel_unique_frac": gumbel_unique_frac_avg,
            "gumbel_a_mean": gumbel_a_mean_avg,
            "gumbel_a_best": gumbel_a_best_avg,
            "gumbel_b_mean": gumbel_b_mean_avg,
            "gumbel_b_best": gumbel_b_best_avg,
            "degree_mismatch": deg_mismatch,
            "degree_mismatch_max": deg_mismatch_max,
            "solver_residual": solver_resid,
            "solver_residual_max": solver_resid_max,
            "struct_cert": struct_cert_avg,
            "struct_cert_norm": struct_cert_norm_avg,
            "stage2_struct_target_active_percent": struct_target_active_pct_avg,
            "repair_upper_bound": repair_upper_bound_avg,
            "top2_concentration": top2_conc,
            "outside_candidate_mass": outcand_mass,
            "ift_trusted_frac": ift_trusted_frac,
            "round0_cost": round0_cost_avg,
            "round1_cost": round1_cost_avg,
            "stage2_coupled_steps": stage2_steps_avg,
            "stage2_coupled_delta_mu": stage2_delta_mu_avg,
            "stage2_delta_mu_first": stage2_delta_mu_first_avg,
            "stage2_delta_mu_last": stage2_delta_mu_last_avg,
            "stage2_delta_mu_max_last": stage2_delta_mu_max_last_avg,
            "stage2_delta_mu_contract": stage2_delta_mu_contract_avg,
            "stage2_certificate_rms": stage2_cert_rms_avg,
            "stage2_tilt_proj_rms": stage2_tilt_proj_rms_avg,
            "stage2_tilt_proj_fallback_frac": stage2_tilt_proj_fb_avg,
            "improved": improved,
            "seconds": dt,
        })

        print(
            f"E{epoch:4d} | train {train_avg:.4f} | train_cost {train_cost_avg:.4f} | tau {model.tau:.4f} | val {val_loss_avg:.4f} | relax {relax_avg:.4f} | decoded {decoded_avg:.4f}{extra} | "
            f"{'OPT' if opt_costs else 'REF'} {ref_avg:.4f} | gap {gap:+.1f}% | r0/r1 {round0_cost_avg:.4f}/{round1_cost_avg:.4f} | "
            f"gap_i {gap_inst_mean:+.2f}/{gap_inst_std:.2f}% | gapM_i {gapm_inst_mean:+.2f}/{gapm_inst_std:.2f}% | "
            f"adec {gumbel_a_mean_avg:.4f}/{gumbel_a_best_avg:.4f} | bdec {gumbel_b_mean_avg:.4f}/{gumbel_b_best_avg:.4f} | "
            f"tinfer {decode_infer_total:.2f}s avg {decode_infer_avg:.3f}s | avg_rec {avg_rec:.3f}s | "
            f"u1 {round1_update_rms_avg:.2e}/{round1_raw_rms_avg:.2e}@{round1_scale_avg:.2f} free {round1_free_scale_avg:.2f} | degmis {deg_mismatch:.2e}/{deg_mismatch_max:.2e} | "
            f"edge {edge_drift:.2e} | sumF {sum_nr_resid:.2e} | F/G {fnorm_avg:.2e}/{gnorm_avg:.2e} | root {rootdeg_avg:.2e} | "
            f"solver {solver_resid:.2e}/{solver_resid_max:.2e} | Uproxy {struct_cert_avg:.2e} (avg {struct_cert_norm_avg:.2e}) = defc {struct_defect_cert_avg:.2e} + uncc {struct_uncert_cert_avg:.2e} | "
            f"mn(def/unc) {struct_mean_defect_avg:.2e}/{struct_mean_uncert_avg:.2e} | "
            f"rcoef {repair_bound_weight_avg:.2e} Rb {repair_upper_bound_avg:.4f} | Utgt {model.stage2_struct_target:.3f} Ux {struct_excess_avg:.2e} Up {struct_penalty_avg:.2e} Uact {struct_target_active_pct_avg:.1f}% | ift {ift_trusted_frac:.2f} | proj {proj_row_avg:.1e}/{proj_grand_avg:.1e} rem {proj_removed_avg:.2f} lcos {proj_align_avg:.2f} pfb {proj_fb_avg:.2f} | "
            f"st2 {stage2_steps_avg:.1f} dmu {stage2_delta_mu_avg:.2e} {stage2_delta_mu_first_avg:.2e}->{stage2_delta_mu_last_avg:.2e} max {stage2_delta_mu_max_last_avg:.2e} rho {stage2_delta_mu_contract_avg:.2f} | cert {stage2_cert_rms_avg:.2e} tilt {stage2_tilt_proj_rms_avg:.2e}/{stage2_tilt_proj_fb_avg:.2f} | "
            f"conc {top2_conc:.3f} | outcand {outcand_mass:.3f} | clamp {clamp_frac:.3f}/{scaled_clamp_frac:.3f} | skip {skipped_nonfinite} | {dt:.1f}s" + (" *" if improved else "")
        )

    print("=" * 72)
    if math.isfinite(best_decoded):
        print(f"Done. Best decoded metric: {best_decoded:.4f}")
    else:
        print(f"Done. Best validation loss: {best_relax:.4f}")
    if not checkpoint_path.exists():
        torch.save(model.state_dict(), checkpoint_path)
    torch.save(model.state_dict(), last_checkpoint_path)
    last_full_decode = collect_full_decode_lengths(model, val_loader, args, device)
    _write_json(
        last_detailed_decode_path,
        _build_decode_lengths_payload(
            epoch=args.epochs,
            job_name=args.job_name,
            result_dir=result_dir,
            checkpoint_path=last_checkpoint_path,
            full_decode=last_full_decode,
        ),
    )
    _write_json(
        result_dir / "train_metrics.json",
        {
            "job_name": args.job_name,
            "result_dir": result_dir,
            "history": train_history,
        },
    )
    _write_json(
        result_dir / "test_metrics.json",
        {
            "job_name": args.job_name,
            "result_dir": result_dir,
            "history": test_history,
            "best_decoded": best_decoded if math.isfinite(best_decoded) else None,
            "best_validation_loss": best_relax if math.isfinite(best_relax) else None,
            "checkpoint_path": checkpoint_path,
            "last_checkpoint_path": last_checkpoint_path,
            "best_decoded_tour_lengths_path": best_detailed_decode_path,
            "last_decoded_tour_lengths_path": last_detailed_decode_path,
        },
    )
    return model
