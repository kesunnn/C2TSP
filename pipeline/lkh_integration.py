"""Helpers that bridge our network outputs (mu, lambda_nr) to LKH-3.

Three integration levels are supported. For each level we produce a bundle of
input files and a PARAMETER_FILE and then invoke the compiled LKH binary:

  level 0 (naive baseline):  tsp only                          — LKH computes pi via Ascent
                                                                 and generates its own candidates.
  level 1 (initial tour):    tsp + initial_tour                — LKH unchanged, but starts from
                                                                 our mu-decoded tour.
  level 2 (top-k candidates): tsp + candidate (from mu top-k)  — LKH still runs Ascent for pi,
                                                                 but the candidate set is ours.
  level 3 (pi + top-k):      tsp + pi + candidate              — Ascent is skipped entirely;
                                                                 pi comes from lambda_nr and
                                                                 candidates come from mu top-k.

All coordinates are written in the same integer scale that Concorde used when
generating the reference optimum (default 1,000,000). LKH's own PRECISION
multiplier stays at its default value of 100. The two factors together mean
internal LKH distances live on the scale  PRECISION * coord_scale * euclid(x,y)
= 1e8 * euclid(x,y), and node duals lambda_nr (which live on the euclid(x,y)
scale) must therefore be rescaled by coord_scale * PRECISION before being
written into PI_FILE.
"""
from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_COORD_SCALE = 1_000_000
DEFAULT_PRECISION = 100


def load_calibration(path: Path, n: int, which: str = "median") -> float:
    """Load lambda->pi calibration factor for TSP size n from a JSON file
    produced by scripts/calibrate_pi_scale.py. Falls back to alpha=1.0 if the
    file is missing or does not contain an entry for this size.
    """
    path = Path(path)
    if not path.exists():
        return 1.0
    import json as _json
    data = _json.loads(path.read_text())
    per = data.get("per_size", {})
    row = per.get(str(int(n)))
    if row is None:
        return 1.0
    key = {"median": "alpha_median",
           "mean": "alpha_mean",
           "geomean": "alpha_geomean"}.get(which, "alpha_median")
    return float(row[key])


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def write_tsplib(
    path: Path,
    coords_unit_sq: np.ndarray,
    name: str,
    coord_scale: int = DEFAULT_COORD_SCALE,
) -> np.ndarray:
    """Write a TSPLIB EUC_2D instance file using integer-rounded coordinates.

    Returns the integer coord matrix that was actually written (so callers can
    reuse it for cost recomputation in the same int units LKH uses).
    """
    coords = np.asarray(coords_unit_sq, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape [n, 2], got {coords.shape}")
    n = coords.shape[0]
    int_coords = np.rint(coords * float(coord_scale)).astype(np.int64)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"NAME: {name}\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {n}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i in range(n):
            f.write(f"{i + 1} {int(int_coords[i, 0])} {int(int_coords[i, 1])}\n")
        f.write("EOF\n")
    return int_coords


def write_initial_tour_file(path: Path, tour_0idx: Iterable[int], n: int, name: str) -> None:
    """Write a TSPLIB .tour file. `tour_0idx` is a permutation of 0..n-1."""
    tour = list(tour_0idx)
    if len(tour) != n or sorted(tour) != list(range(n)):
        raise ValueError("tour_0idx must be a permutation of 0..n-1")
    with path.open("w", encoding="utf-8") as f:
        f.write(f"NAME: {name}.tour\n")
        f.write("TYPE: TOUR\n")
        f.write(f"DIMENSION: {n}\n")
        f.write("TOUR_SECTION\n")
        for v in tour:
            f.write(f"{int(v) + 1}\n")
        f.write("-1\n")
        f.write("EOF\n")


def write_pi_file(
    path: Path,
    lambda_nr: np.ndarray,
    root: int,
    n: int,
    coord_scale: int = DEFAULT_COORD_SCALE,
    precision: int = DEFAULT_PRECISION,
    alpha: float = 1.0,
) -> None:
    """Write LKH PI_FILE encoding our node duals.

    LKH stores Pi as an integer on the internal cost scale
    (PRECISION * coord_scale * euclid(x,y)). Empirically our lambda_nr sits at
    a larger magnitude than LKH's Ascent-optimized pi at that nominal scale, so
    callers pass a calibration factor `alpha` (typically in the 0.05-0.15 range,
    see scripts/calibrate_pi_scale.py) to bring the two distributions into
    agreement. The net written value is

        pi_int = round(lambda_nr * coord_scale * precision * alpha).

    The root dual is fixed to 0 by our gauge.
    """
    lambda_nr = np.asarray(lambda_nr, dtype=np.float64).reshape(-1)
    if lambda_nr.shape[0] != n - 1:
        raise ValueError(
            f"lambda_nr must have length n-1 = {n - 1}, got {lambda_nr.shape[0]}"
        )
    root = int(root)
    scale = float(coord_scale) * float(precision) * float(alpha)
    pi_full = np.zeros(n, dtype=np.int64)
    nonroot = [i for i in range(n) if i != root]
    for j, i in enumerate(nonroot):
        pi_full[i] = int(np.rint(lambda_nr[j] * scale))
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{n}\n")
        for i in range(n):
            f.write(f"{i + 1} {int(pi_full[i])}\n")


def gumbel_topk_sample_tours(
    mu: np.ndarray,
    dist: np.ndarray,
    k: int,
    *,
    root: int = 0,
    gumbel_scale: float = 0.20,
    rng_seed: int = 0,
    twoopt_passes: int = 6,
) -> tuple[list[list[int]], list[float]]:
    """Sample up to k unique tours via Gumbel-perturbed greedy degree-2 decode.

    Uses the existing decode primitives in src/tsp_onetree/decode.py. The first
    draw is the unperturbed argmax (greedy). Subsequent draws add iid symmetric
    Gumbel noise to log(p_ij) and rerun greedy. Each tour gets a cheap LK-lite
    polish to remove obvious 2-opt slack so the cost ranking is meaningful when
    we hand only the best one (or the top-N) to LKH.

    Returns (tours, costs) sorted by cost ascending. May return fewer than k
    tours if the perturbations collapse to duplicates.
    """
    # Deferred import: this module is otherwise pure-Python and we want to keep
    # `import lkh_integration` cheap for callers that never sample.
    from tsp_onetree.decode import (  # type: ignore[import-not-found]
        _mu_score_matrix,
        _symmetric_gumbel_noise,
        build_move_candidates,
        greedy_degree2_tour_from_scores,
        lk_lite_improve_tour,
        _tour_cost_numpy,
        _tour_key,
    )

    mu = np.asarray(mu, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64)
    n = mu.shape[0]
    k = max(1, int(k))

    base_score = _mu_score_matrix(mu)
    candidate_sets = build_move_candidates(mu, dist, None, extra_topk=5)
    rng = np.random.default_rng(int(rng_seed))

    tours: list[list[int]] = []
    costs: list[float] = []
    seen: set[tuple[int, ...]] = set()

    def _add(tour: list[int]) -> None:
        if not tour:
            return
        polished = lk_lite_improve_tour(tour, dist, candidate_sets,
                                        twoopt_passes=int(twoopt_passes))
        # rotate root to front
        if int(polished[0]) != int(root):
            ridx = polished.index(int(root))
            polished = polished[ridx:] + polished[:ridx]
        key = _tour_key(polished)
        if key in seen:
            return
        seen.add(key)
        tours.append(polished)
        costs.append(float(_tour_cost_numpy(polished, dist)))

    # Draw 0: unperturbed greedy.
    score0 = base_score.copy()
    np.fill_diagonal(score0, -np.inf)
    seed0, _ = greedy_degree2_tour_from_scores(score0, dist, root=int(root))
    _add(seed0)

    for m in range(1, k):
        noise = _symmetric_gumbel_noise(n, rng)
        score_m = base_score + float(gumbel_scale) * noise
        np.fill_diagonal(score_m, -np.inf)
        seed_m, _ = greedy_degree2_tour_from_scores(score_m, dist, root=int(root))
        _add(seed_m)

    order = sorted(range(len(tours)), key=lambda i: costs[i])
    tours_sorted = [tours[i] for i in order]
    costs_sorted = [costs[i] for i in order]
    return tours_sorted, costs_sorted


def write_tour_files_for_merge(
    prefix: Path,
    tours: list[Iterable[int]],
    n: int,
) -> list[Path]:
    """Write each tour as its own .tour file. Returns the list of written paths
    suitable for LKH's MERGE_TOUR_FILE = ... entries (LKH allows multiple)."""
    out: list[Path] = []
    for k, tour in enumerate(tours):
        p = prefix.parent / f"{prefix.name}_{k}.tour"
        write_initial_tour_file(p, tour, n=n, name=f"{prefix.name}_{k}")
        out.append(p)
    return out


def write_candidate_file(
    path: Path,
    mu: np.ndarray,
    k: int,
    root: int = 0,
    symmetric: bool = True,
) -> None:
    """Write LKH CANDIDATE_FILE using the top-k mu neighbors per node.

    The alpha values we emit are a monotone transform of mu so that higher
    mu -> smaller alpha ("better" candidate under LKH's ordering):
        alpha_int = round((1 - clip(mu, 0, 1)) * 1000)
    LKH will trim to MAX_CANDIDATES at read time so `k` can be upper bounded
    by that parameter too.
    """
    mu = np.asarray(mu, dtype=np.float64)
    n = mu.shape[0]
    if mu.shape != (n, n):
        raise ValueError(f"mu must be square, got {mu.shape}")
    if k < 1 or k >= n:
        raise ValueError(f"k must be in [1, n-1], got k={k}, n={n}")

    sym = 0.5 * (mu + mu.T)
    np.fill_diagonal(sym, -np.inf)
    mu_clip = np.clip(sym, 0.0, 1.0)

    # For each node, pick the top-k neighbors by mu.
    topk_idx = np.argpartition(-sym, kth=k - 1, axis=1)[:, :k]
    # Sort those k by decreasing mu for determinism.
    for i in range(n):
        order = np.argsort(-sym[i, topk_idx[i]])
        topk_idx[i] = topk_idx[i][order]

    if symmetric:
        # Symmetrize: if j is in N(i), also ensure i is in N(j).
        nbr_set = [set(int(v) for v in topk_idx[i]) for i in range(n)]
        for i in range(n):
            for j in list(nbr_set[i]):
                nbr_set[j].add(i)
        # Preserve mu-ordering of the expanded sets.
        ordered_nbrs = []
        for i in range(n):
            candidates = list(nbr_set[i])
            candidates.sort(key=lambda j: (-sym[i, j], j))
            ordered_nbrs.append(candidates)
    else:
        ordered_nbrs = [list(topk_idx[i]) for i in range(n)]

    with path.open("w", encoding="utf-8") as f:
        f.write(f"{n}\n")
        for i in range(n):
            nbrs = ordered_nbrs[i]
            # Dad=0 means "no parent 1-tree edge recorded".
            f.write(f"{i + 1} 0 {len(nbrs)}")
            for j in nbrs:
                alpha_int = int(round((1.0 - float(mu_clip[i, j])) * 1000.0))
                f.write(f" {int(j) + 1} {alpha_int}")
            f.write("\n")
        f.write("-1\n")


def compute_composite_score(
    mu: np.ndarray,
    dist: np.ndarray,
    lambda_nr: np.ndarray,
    root: int = 0,
    alpha_w: float = 1.0,
    beta_w: float = 1.0,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Composite per-edge score blending edge probability and reduced cost.

    Inputs are all per-instance numpy arrays:
      mu          (n, n)  symmetric edge probability ~ marginals
      dist        (n, n)  Euclidean distances
      lambda_nr   (n-1,)  node duals from the network (root excluded)
      root        int

    Returns (score, reduced_cost) both shape (n, n), symmetric.

      reduced_cost[i, j] = dist[i, j] - lambda_full[i] - lambda_full[j]
      score[i, j]        = alpha_w * log(p_ij) + beta_w * (-c_tilde_norm[i, j])
                          where c_tilde_norm[i, j] = (c[i, j] - mu_i) / sigma_i
                          (per-row znorm of reduced costs over j != i, then symmetrized).

    The diagonal is set to -inf for the score and 0 for reduced_cost.
    """
    mu = np.asarray(mu, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64)
    lam = np.asarray(lambda_nr, dtype=np.float64).reshape(-1)
    n = dist.shape[0]
    if mu.shape != (n, n):
        raise ValueError(f"mu shape {mu.shape} != ({n}, {n})")
    if lam.shape[0] != n - 1:
        raise ValueError(f"lambda_nr length {lam.shape[0]} != n-1 = {n - 1}")

    lam_full = np.zeros(n, dtype=np.float64)
    nonroot = [i for i in range(n) if i != int(root)]
    for j, i in enumerate(nonroot):
        lam_full[i] = float(lam[j])

    reduced_cost = dist - lam_full[:, None] - lam_full[None, :]
    np.fill_diagonal(reduced_cost, 0.0)

    # Per-row z-normalization of reduced cost (excluding diagonal).
    mask = ~np.eye(n, dtype=bool)
    rc_for_stats = np.where(mask, reduced_cost, np.nan)
    mu_i = np.nanmean(rc_for_stats, axis=1, keepdims=True)
    sigma_i = np.nanstd(rc_for_stats, axis=1, keepdims=True)
    sigma_i = np.where(sigma_i > eps, sigma_i, 1.0)
    rc_norm = (reduced_cost - mu_i) / sigma_i
    rc_norm = 0.5 * (rc_norm + rc_norm.T)  # symmetrize

    p_sym = 0.5 * (mu + mu.T)
    p_clip = np.clip(p_sym, eps, 1.0 - eps)
    log_p = np.log(p_clip)

    score = float(alpha_w) * log_p + float(beta_w) * (-rc_norm)
    np.fill_diagonal(score, -np.inf)
    return score, reduced_cost


def write_candidate_file_from_score(
    path: Path,
    selection_score: np.ndarray,
    alpha_score: np.ndarray | None = None,
    p_target: float = 0.95,
    k_min: int = 5,
    k_max: int = 20,
    top_k_fixed: int | None = None,
    alpha_scale: int = 1000,
    symmetric: bool = True,
) -> tuple[float, float]:
    """Write a CANDIDATE_FILE using a composite per-edge selection score.

    Two selection modes:
    - `top_k_fixed=K` (preferred for apples-to-apples vs NeuroLKH): every node
      gets exactly K neighbors (top-K by `selection_score`).
    - Adaptive (default, when `top_k_fixed` is None): cumulative-softmax
      threshold on `selection_score`, clamped to [k_min, k_max].

    `symmetric=True` ensures (i, j) in N(i) iff (j, i) in N(j) (may exceed K
    after symmetrization).

    The per-edge alpha is `round(alpha_scale * (s_max_i - s_ij))` using
    `alpha_score` if provided, else `selection_score` itself. Smaller alpha is
    higher priority in LKH's candidate walk, so the highest-score neighbor at
    each node always gets alpha 0.

    Returns (mean_neighbor_count, max_neighbor_count) for telemetry.
    """
    sel = np.asarray(selection_score, dtype=np.float64).copy()
    n = sel.shape[0]
    if sel.shape != (n, n):
        raise ValueError(f"selection_score must be square, got {sel.shape}")
    np.fill_diagonal(sel, -np.inf)
    sel_sym = 0.5 * (sel + sel.T)
    np.fill_diagonal(sel_sym, -np.inf)

    if alpha_score is None:
        alpha_sym = sel_sym
    else:
        a = np.asarray(alpha_score, dtype=np.float64).copy()
        if a.shape != (n, n):
            raise ValueError(f"alpha_score must be square, got {a.shape}")
        np.fill_diagonal(a, -np.inf)
        alpha_sym = 0.5 * (a + a.T)
        np.fill_diagonal(alpha_sym, -np.inf)

    if top_k_fixed is not None:
        k_fixed = max(1, min(int(top_k_fixed), n - 1))
    else:
        p_target = float(np.clip(p_target, 1e-3, 1.0 - 1e-9))
        k_min = max(1, int(k_min))
        k_max = max(k_min, int(k_max))
        k_max = min(k_max, n - 1)
        k_fixed = None

    nbr_lists: list[list[int]] = []
    for i in range(n):
        order = np.argsort(-sel_sym[i], kind="mergesort")
        order = [int(j) for j in order if int(j) != i]
        if k_fixed is not None:
            nbr_lists.append(order[:k_fixed])
            continue
        s = sel_sym[i, order]
        s_max = float(s[0])
        e = np.exp(s - s_max)
        z = e.sum()
        if z <= 0 or not np.isfinite(z):
            cum = np.linspace(0.0, 1.0, len(order), endpoint=False)
        else:
            cum = np.cumsum(e / z)
        k_adapt = int(np.searchsorted(cum, p_target) + 1)
        k_adapt = max(k_min, min(k_max, k_adapt))
        nbr_lists.append(order[:k_adapt])

    if symmetric:
        sets = [set(int(v) for v in nbr_lists[i]) for i in range(n)]
        for i in range(n):
            for j in list(sets[i]):
                sets[j].add(i)
        nbr_lists = []
        for i in range(n):
            ordered = sorted(sets[i], key=lambda j: (-float(alpha_sym[i, j]), int(j)))
            nbr_lists.append(ordered)
    else:
        # Re-order by alpha so LKH walks our preferred edges first.
        nbr_lists = [sorted(nbrs, key=lambda j: (-float(alpha_sym[i, j]), int(j)))
                     for i, nbrs in enumerate(nbr_lists)]

    a_max_per_node = np.array([float(alpha_sym[i, nbr_lists[i][0]]) if nbr_lists[i] else 0.0
                               for i in range(n)], dtype=np.float64)

    counts = [len(nbrs) for nbrs in nbr_lists]
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{n}\n")
        for i in range(n):
            nbrs = nbr_lists[i]
            f.write(f"{i + 1} 0 {len(nbrs)}")
            for j in nbrs:
                a_ij = float(alpha_sym[i, j])
                alpha_int = int(round(float(alpha_scale) * max(0.0, a_max_per_node[i] - a_ij)))
                f.write(f" {int(j) + 1} {alpha_int}")
            f.write("\n")
        f.write("-1\n")
    return (float(np.mean(counts)) if counts else 0.0,
            float(np.max(counts)) if counts else 0.0)


# ---------------------------------------------------------------------------
# v20-design candidate helpers (μ + C_mod, 20-NN restricted, NeuroLKH-style alpha)
# ---------------------------------------------------------------------------

def compute_C_mod(
    dist: np.ndarray,
    lambda_nr: np.ndarray,
    root: int = 0,
) -> np.ndarray:
    """Reduced-cost-style 'modified cost': C_mod[i,j] = D[i,j] - λ_i - λ_j.

    Lower is better (a more promising edge under the dual).
    Diagonal is set to +inf to keep it out of any min-selection.
    """
    dist = np.asarray(dist, dtype=np.float64)
    lam = np.asarray(lambda_nr, dtype=np.float64).reshape(-1)
    n = dist.shape[0]
    if lam.shape[0] != n - 1:
        raise ValueError(f"lambda_nr length {lam.shape[0]} != n-1 = {n - 1}")
    lam_full = np.zeros(n, dtype=np.float64)
    nonroot = [i for i in range(n) if i != int(root)]
    for j, i in enumerate(nonroot):
        lam_full[i] = float(lam[j])
    C = dist - lam_full[:, None] - lam_full[None, :]
    np.fill_diagonal(C, np.inf)
    return C


def per_row_znorm(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-row z-normalization over off-diagonal entries; symmetrized output.

    Used to put μ (in [0,1]) and C_mod (raw cost units) on the same scale so a
    linear combination 0.5·X_z + 0.5·Y_z weights each signal equally.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    mask = ~np.eye(n, dtype=bool)
    Xm = np.where(mask, X, np.nan)
    m = np.nanmean(Xm, axis=1, keepdims=True)
    s = np.nanstd(Xm, axis=1, keepdims=True)
    s = np.where(s > eps, s, 1.0)
    Z = (X - m) / s
    Z = 0.5 * (Z + Z.T)
    return Z


def select_candidates_knn_topk(
    score: np.ndarray,
    dist: np.ndarray,
    *,
    k_knn: int = 20,
    top_k: int = 5,
    higher_is_better: bool = True,
) -> list[list[int]]:
    """Per-node candidate selection: first restrict to k_knn nearest neighbors
    by Euclidean distance, then take top_k within that neighborhood by `score`.

    Matches NeuroLKH's structural prior (candidates drawn from 20-NN sparse
    graph) so both methods are restricted to the same edge pool.

    Returns per-node neighbor lists, each sorted by `score` (descending if
    `higher_is_better`, ascending otherwise).
    """
    score = np.asarray(score, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64)
    n = dist.shape[0]
    if score.shape != (n, n):
        raise ValueError(f"score shape {score.shape} != ({n},{n})")
    k_knn = max(1, min(int(k_knn), n - 1))
    top_k = max(1, min(int(top_k), k_knn))

    d_self = dist.copy()
    np.fill_diagonal(d_self, np.inf)
    # 20-NN per row by ascending distance.
    knn_idx = np.argpartition(d_self, kth=k_knn - 1, axis=1)[:, :k_knn]

    nbr_lists: list[list[int]] = []
    for i in range(n):
        cand = knn_idx[i]
        scores_i = score[i, cand]
        if higher_is_better:
            order = np.argsort(-scores_i, kind="mergesort")
        else:
            order = np.argsort(scores_i, kind="mergesort")
        chosen = cand[order[:top_k]]
        nbr_lists.append([int(v) for v in chosen])
    return nbr_lists


def write_candidate_file_neurolkh_style(
    path: Path,
    neighbor_lists: list[list[int]],
) -> tuple[float, float]:
    """Write a CANDIDATE_FILE in NeuroLKH's exact format:
        node_idx 0 K cand_1 0 cand_2 100 cand_3 200 ...

    `neighbor_lists` is one ordered list per node (already sorted from "best"
    to "worst"), and the slot index is encoded in the alpha as `slot * 100`.
    The number of neighbors should be the same for every node (typically 5);
    if it varies, alpha is still slot * 100.

    Returns (mean_neighbor_count, max_neighbor_count) for telemetry.
    """
    n = len(neighbor_lists)
    counts = [len(nbrs) for nbrs in neighbor_lists]
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{n}\n")
        for i in range(n):
            nbrs = neighbor_lists[i]
            f.write(f"{i + 1} 0 {len(nbrs)}")
            for slot, j in enumerate(nbrs):
                f.write(f" {int(j) + 1} {slot * 100}")
            f.write("\n")
        f.write("-1\n")
    return (
        float(np.mean(counts)) if counts else 0.0,
        float(np.max(counts)) if counts else 0.0,
    )


def reorder_by_combined_rank(
    nbrs: list[int],
    mu_row: np.ndarray,
    cmod_row: np.ndarray,
    weight_mu: float = 0.5,
    weight_cmod: float = 0.5,
) -> list[int]:
    """Given a fixed set of candidates for node i, reorder them by
    `weight_mu * rank_mu + weight_cmod * rank_cmod` (ascending — lower
    combined rank = better candidate). Ranks are 0..K-1 within the K
    candidates: rank_mu sorts by μ descending, rank_cmod sorts by C_mod
    ascending.
    """
    if not nbrs:
        return []
    nbrs_arr = np.asarray(nbrs, dtype=np.int64)
    mu_vals = mu_row[nbrs_arr]
    cmod_vals = cmod_row[nbrs_arr]
    # rank_mu: position when sorted by mu descending
    order_mu = np.argsort(-mu_vals, kind="mergesort")
    rank_mu = np.empty(len(nbrs), dtype=np.float64)
    rank_mu[order_mu] = np.arange(len(nbrs))
    # rank_cmod: position when sorted by C_mod ascending
    order_c = np.argsort(cmod_vals, kind="mergesort")
    rank_c = np.empty(len(nbrs), dtype=np.float64)
    rank_c[order_c] = np.arange(len(nbrs))
    combined = float(weight_mu) * rank_mu + float(weight_cmod) * rank_c
    final_order = np.argsort(combined, kind="mergesort")
    return [int(nbrs_arr[idx]) for idx in final_order]


# ---------------------------------------------------------------------------
# Parameter file
# ---------------------------------------------------------------------------

@dataclass
class LKHParams:
    problem_file: Path
    output_tour_file: Path | None = None
    pi_file: Path | None = None
    candidate_file: Path | None = None
    initial_tour_file: Path | None = None
    merge_tour_files: list[Path] = field(default_factory=list)
    recombination: str | None = None  # IPT (default) | GPX2 | CLARIST
    max_candidates: int = 5
    max_trials: int | None = None  # None -> LKH default (= DIMENSION)
    runs: int = 1
    seed: int = 12345
    time_limit_s: float | None = 60.0
    trace_level: int = 0
    subgradient: bool = True  # LKH default is YES
    ascent_polish: bool = False  # Custom patch: warm-start Ascent from PI_FILE.
    precision: int = DEFAULT_PRECISION

    def write(self, path: Path) -> None:
        lines: list[str] = []
        lines.append(f"PROBLEM_FILE = {self.problem_file}")
        lines.append(f"TRACE_LEVEL = {int(self.trace_level)}")
        lines.append(f"RUNS = {int(self.runs)}")
        lines.append(f"SEED = {int(self.seed)}")
        lines.append(f"PRECISION = {int(self.precision)}")
        if self.max_trials is not None:
            lines.append(f"MAX_TRIALS = {int(self.max_trials)}")
        lines.append(f"MAX_CANDIDATES = {int(self.max_candidates)}")
        if not self.subgradient:
            lines.append("SUBGRADIENT = NO")
        if self.ascent_polish:
            lines.append("ASCENT_POLISH = YES")
        if self.time_limit_s is not None and self.time_limit_s > 0:
            lines.append(f"TIME_LIMIT = {float(self.time_limit_s)}")
        if self.output_tour_file is not None:
            lines.append(f"OUTPUT_TOUR_FILE = {self.output_tour_file}")
        if self.pi_file is not None:
            lines.append(f"PI_FILE = {self.pi_file}")
        if self.candidate_file is not None:
            lines.append(f"CANDIDATE_FILE = {self.candidate_file}")
        if self.initial_tour_file is not None:
            lines.append(f"INITIAL_TOUR_FILE = {self.initial_tour_file}")
        for mp in self.merge_tour_files:
            lines.append(f"MERGE_TOUR_FILE = {mp}")
        if self.recombination is not None:
            lines.append(f"RECOMBINATION = {self.recombination}")
        path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class LKHResult:
    returncode: int
    wall_time_s: float
    stdout: str
    stderr: str
    tour: np.ndarray | None  # 0-indexed permutation
    reported_cost_int: int | None  # LKH-reported best int cost on EUC_2D scale
    ascent_time_s: float | None = None
    total_time_s_reported: float | None = None


def _parse_tour_file(path: Path, dimension: int) -> np.ndarray:
    """Parse a TSPLIB-style tour file (what OUTPUT_TOUR_FILE writes)."""
    text = path.read_text()
    tokens: list[int] = []
    in_section = False
    for raw in text.splitlines():
        line = raw.strip()
        if not in_section:
            if line.upper().startswith("TOUR_SECTION"):
                in_section = True
            continue
        if not line:
            continue
        if line.upper() in ("EOF", "-1"):
            break
        for tok in line.split():
            try:
                v = int(tok)
            except ValueError:
                continue
            if v == -1:
                tokens = tokens  # break outer after loop
                in_section = False
                break
            tokens.append(v)
        if not in_section:
            break
    if len(tokens) != dimension:
        raise RuntimeError(
            f"Tour file {path} has {len(tokens)} nodes, expected {dimension}."
        )
    tour = np.asarray(tokens, dtype=np.int64) - 1
    if sorted(tour.tolist()) != list(range(dimension)):
        raise RuntimeError(f"Tour is not a permutation of 0..{dimension-1}: {path}")
    return tour


_ASCENT_RE = re.compile(r"Ascent\s+time\s*=\s*([0-9eE.+-]+)\s*sec")
_PREPROC_RE = re.compile(r"Preprocessing\s+time\s*=\s*([0-9eE.+-]+)\s*sec")
_TOTAL_RE = re.compile(r"(?:Time\.total|Total\s+Running\s+Time)\s*[:=]\s*([0-9eE.+-]+)\s*sec")
_COSTMIN_RE = re.compile(r"Cost\.min\s*=\s*(-?\d+)")


def _parse_stdout_for_cost_and_time(stdout: str) -> tuple[int | None, float | None, float | None]:
    """Parse LKH stdout for best cost, ascent time, and total time. Uses
    regexes so 'Lower bound = X, Ascent time = Y sec.' (single line) matches."""
    best_cost: int | None = None
    ascent_time: float | None = None
    total_time: float | None = None
    m = _COSTMIN_RE.search(stdout)
    if m:
        try:
            best_cost = int(m.group(1))
        except ValueError:
            pass
    m = _ASCENT_RE.search(stdout)
    if m:
        try:
            ascent_time = float(m.group(1))
        except ValueError:
            pass
    m = _TOTAL_RE.search(stdout)
    if m:
        try:
            total_time = float(m.group(1))
        except ValueError:
            pass
    return best_cost, ascent_time, total_time


def run_lkh(
    lkh_bin: Path,
    par_path: Path,
    dimension: int,
    output_tour_path: Path | None,
    timeout_s: float | None = None,
) -> LKHResult:
    """Invoke the LKH binary with the given parameter file and collect results."""
    lkh_bin = Path(lkh_bin)
    par_path = Path(par_path)
    if not lkh_bin.is_file():
        raise FileNotFoundError(f"LKH binary not found: {lkh_bin}")
    if not par_path.is_file():
        raise FileNotFoundError(f"Parameter file not found: {par_path}")

    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(lkh_bin), str(par_path)],
        cwd=par_path.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout_s,
    )
    wall = time.perf_counter() - t0
    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")
    best_cost, ascent_time, total_time = _parse_stdout_for_cost_and_time(stdout)

    tour_arr: np.ndarray | None = None
    if output_tour_path is not None and output_tour_path.exists():
        try:
            tour_arr = _parse_tour_file(output_tour_path, dimension)
        except Exception:
            tour_arr = None

    return LKHResult(
        returncode=proc.returncode,
        wall_time_s=wall,
        stdout=stdout,
        stderr=stderr,
        tour=tour_arr,
        reported_cost_int=best_cost,
        ascent_time_s=ascent_time,
        total_time_s_reported=total_time,
    )


# ---------------------------------------------------------------------------
# Cost utilities
# ---------------------------------------------------------------------------

def tour_cost_euc2d_int(int_coords: np.ndarray, tour: np.ndarray) -> int:
    """Cost of a tour in LKH's EUC_2D integer-distance convention (nearest int)."""
    n = tour.shape[0]
    total = 0
    for k in range(n):
        u = int(tour[k])
        v = int(tour[(k + 1) % n])
        dx = float(int_coords[u, 0] - int_coords[v, 0])
        dy = float(int_coords[u, 1] - int_coords[v, 1])
        total += int(np.rint(np.sqrt(dx * dx + dy * dy)))
    return total


def tour_cost_norm(coords: np.ndarray, tour: np.ndarray) -> float:
    """Cost of a tour in normalized [0,1]^2 Euclidean units."""
    coords = np.asarray(coords, dtype=np.float64)
    tour = np.asarray(tour, dtype=np.int64)
    rolled = np.roll(tour, -1)
    diffs = coords[tour] - coords[rolled]
    return float(np.sqrt((diffs ** 2).sum(axis=1)).sum())
