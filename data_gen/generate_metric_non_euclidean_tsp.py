#!/usr/bin/env python
"""Generate explicit symmetric-metric TSP test datasets.

Dataset A is an Erdős--Rényi graph with Euclidean edge lengths, closed under
all-pairs shortest paths.  Dataset B converts HCP edge lists to 1--2 metrics.
Both are stored as packed integer matrices because the release's coordinate-only
``data/tsp`` text format cannot represent a non-Euclidean metric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra
except ImportError as exc:  # pragma: no cover - depends on environment setup
    raise RuntimeError(
        "Metric dataset generation requires scipy. Install project dependencies first."
    ) from exc

from tsp_onetree.data import pack_upper_triangle


INT_COST_SCALE = 1_000_000


def write_explicit_tsplib(path: Path, cost_int: np.ndarray, name: str) -> None:
    """Write a symmetric integer matrix as a TSPLIB explicit full matrix."""
    costs = np.asarray(cost_int)
    if costs.ndim != 2 or costs.shape[0] != costs.shape[1]:
        raise ValueError(f"cost_int must be square, got {costs.shape}")
    n = costs.shape[0]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"NAME: {name}\n")
        handle.write("TYPE: TSP\n")
        handle.write(f"DIMENSION: {n}\n")
        handle.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        handle.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        handle.write("EDGE_WEIGHT_SECTION\n")
        for row in costs:
            handle.write(" ".join(str(int(value)) for value in row))
            handle.write("\n")
        handle.write("EOF\n")


def _parse_concorde_tour(path: Path, n: int) -> np.ndarray:
    tokens: list[int] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        for token in raw.split():
            try:
                tokens.append(int(token))
            except ValueError:
                continue
    if not tokens or tokens[0] != n or len(tokens[1:]) != n:
        raise RuntimeError(f"Malformed Concorde tour file: {path}")
    tour = np.asarray(tokens[1:], dtype=np.int64)
    if not np.array_equal(np.sort(tour), np.arange(n)):
        raise RuntimeError(f"Concorde tour is not a permutation for n={n}: {path}")
    return tour


def _parse_lkh_tour(path: Path, n: int) -> np.ndarray:
    in_section = False
    tokens: list[int] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not in_section:
            in_section = line.upper().startswith("TOUR_SECTION")
            continue
        if not line or line.upper() in {"EOF", "-1"}:
            break
        for token in line.split():
            try:
                tokens.append(int(token))
            except ValueError:
                continue
    if len(tokens) != n:
        raise RuntimeError(f"LKH tour file has {len(tokens)} nodes; expected {n}: {path}")
    tour = np.asarray(tokens, dtype=np.int64) - 1
    if not np.array_equal(np.sort(tour), np.arange(n)):
        raise RuntimeError(f"LKH tour is not a permutation for n={n}: {path}")
    return tour


def _tour_cost_int(cost_int: np.ndarray, tour: np.ndarray) -> int:
    order = np.asarray(tour, dtype=np.int64)
    return int(cost_int[order, np.roll(order, -1)].sum(dtype=np.int64))


def solve_concorde_explicit(
    cost_int: np.ndarray, *, seed: int, concorde_bin: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    if not concorde_bin.is_file():
        raise FileNotFoundError(f"Concorde binary not found: {concorde_bin}")
    n = int(cost_int.shape[0])
    with tempfile.TemporaryDirectory(prefix="metric_concorde_") as tmp:
        work_dir = Path(tmp)
        tsp_path = work_dir / "instance.tsp"
        sol_path = work_dir / "instance.sol"
        write_explicit_tsplib(tsp_path, cost_int, name=f"metric_{seed}")
        started = time.perf_counter()
        proc = subprocess.run(
            [str(concorde_bin), "-s", str(int(seed)), "-x", "-o", str(sol_path), str(tsp_path)],
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        wall_s = time.perf_counter() - started
        # Some Concorde builds return nonzero after successful cleanup. The
        # solution file is the authoritative completion signal, as in the
        # bundled Euclidean generator.
        if not sol_path.is_file():
            raise RuntimeError(
                f"Concorde produced no solution (exit={proc.returncode}): "
                f"{proc.stderr.decode('utf-8', errors='replace')[-1000:]}"
            )
        tour = _parse_concorde_tour(sol_path, n)
    return tour, {
        "solver_name": "concorde",
        "solver_binary": str(concorde_bin),
        "solver_version": concorde_bin.parent.name,
        "solver_parameters": {"seed": int(seed), "exact": True},
        "solver_returncode": int(proc.returncode),
        "solver_runtime_seconds": float(wall_s),
        "label_status": "exact",
    }


def solve_lkh_explicit(
    cost_int: np.ndarray, *, seed: int, lkh_bin: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    if not lkh_bin.is_file():
        raise FileNotFoundError(f"LKH binary not found: {lkh_bin}")
    n = int(cost_int.shape[0])
    with tempfile.TemporaryDirectory(prefix="metric_lkh_") as tmp:
        work_dir = Path(tmp)
        tsp_path = work_dir / "instance.tsp"
        par_path = work_dir / "instance.par"
        tour_path = work_dir / "instance.tour"
        write_explicit_tsplib(tsp_path, cost_int, name=f"metric_{seed}")
        par_path.write_text(
            "\n".join(
                [
                    f"PROBLEM_FILE = {tsp_path}",
                    f"OUTPUT_TOUR_FILE = {tour_path}",
                    "RUNS = 1",
                    f"MAX_TRIALS = {n}",
                    f"SEED = {int(seed)}",
                    "TRACE_LEVEL = 1",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        started = time.perf_counter()
        proc = subprocess.run(
            [str(lkh_bin), str(par_path)],
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        wall_s = time.perf_counter() - started
        if not tour_path.is_file():
            raise RuntimeError(
                f"LKH produced no solution (exit={proc.returncode}): "
                f"{proc.stderr.decode('utf-8', errors='replace')[-1000:]}"
            )
        tour = _parse_lkh_tour(tour_path, n)
    return tour, {
        "solver_name": "lkh-3",
        "solver_binary": str(lkh_bin),
        "solver_version": lkh_bin.parent.name,
        "solver_parameters": {"seed": int(seed), "runs": 1, "max_trials": n, "time_limit": None},
        "solver_returncode": int(proc.returncode),
        "solver_runtime_seconds": float(wall_s),
        "label_status": "near_optimal",
    }


def _sample_connected_er(
    n: int, gamma: float, coords: np.ndarray, rng: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Sample a conditional connected G(n,p) without allocating an n-by-n RNG array."""
    p = min(1.0, float(gamma) * math.log(float(n)) / float(n - 1))
    attempts = 0
    while True:
        attempts += 1
        row_chunks: list[np.ndarray] = []
        col_chunks: list[np.ndarray] = []
        for i in range(n - 1):
            chosen = np.flatnonzero(rng.random_sample(n - i - 1) < p).astype(np.int32) + i + 1
            if chosen.size:
                row_chunks.append(np.full(chosen.size, i, dtype=np.int32))
                col_chunks.append(chosen)
        rows = np.concatenate(row_chunks) if row_chunks else np.empty(0, dtype=np.int32)
        cols = np.concatenate(col_chunks) if col_chunks else np.empty(0, dtype=np.int32)
        if rows.size == 0:
            continue
        delta = coords[rows] - coords[cols]
        weights = np.linalg.norm(delta, axis=1).astype(np.float64)
        graph = csr_matrix(
            (
                np.concatenate((weights, weights)),
                (np.concatenate((rows, cols)), np.concatenate((cols, rows))),
            ),
            shape=(n, n),
        )
        components, _ = connected_components(graph, directed=False, return_labels=True)
        if components == 1:
            return rows, cols, weights.astype(np.float32), attempts, p


def _integerize_metric(cost_float: np.ndarray) -> np.ndarray:
    values = np.asarray(cost_float, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Shortest-path closure contains non-finite distances")
    max_cost = float(values.max())
    if max_cost <= 0:
        raise ValueError("Metric closure has no positive distance")
    cost_int = np.ceil(INT_COST_SCALE * values / max_cost).astype(np.uint32)
    np.fill_diagonal(cost_int, 0)
    return cost_int


def _validate_metric(cost_int: np.ndarray, *, exhaustive: bool, seed: int) -> None:
    costs = np.asarray(cost_int)
    n = costs.shape[0]
    if costs.shape != (n, n) or not np.array_equal(costs, costs.T):
        raise ValueError("Cost matrix must be square and symmetric")
    if not np.all(np.diag(costs) == 0) or np.any(costs[~np.eye(n, dtype=bool)] == 0):
        raise ValueError("Cost matrix must have zero diagonal and positive off-diagonal entries")
    if exhaustive:
        wide = costs.astype(np.int64, copy=False)
        for k in range(n):
            if np.any(wide > wide[:, [k]] + wide[[k], :]):
                raise ValueError(f"Triangle inequality violated through vertex {k}")
    else:
        rng = np.random.RandomState(int(seed) + 17)
        trials = min(1_000_000, max(100_000, 20 * n))
        i = rng.randint(0, n, size=trials)
        j = rng.randint(0, n, size=trials)
        k = rng.randint(0, n, size=trials)
        if np.any(costs[i, j].astype(np.int64) > costs[i, k].astype(np.int64) + costs[k, j].astype(np.int64)):
            raise ValueError("Sampled triangle inequality violation")


def _archive_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["metadata_json"].item()))


def _write_archive(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_path, **arrays, metadata_json=np.array(json.dumps(metadata, sort_keys=True)))
    os.replace(tmp_path, path)


def _generate_er_instance(index: int, config: dict[str, Any]) -> dict[str, Any]:
    n = int(config["n"])
    output_dir = Path(config["output_dir"])
    archive_path = output_dir / "instances" / f"instance_{index:05d}.npz"
    if archive_path.is_file():
        return _archive_metadata(archive_path)

    base_seed = int(config["base_seed"])
    coordinate_seed = base_seed + 2 * int(index)
    graph_seed = coordinate_seed + 1
    coords = np.random.RandomState(coordinate_seed).uniform(0.0, 1.0, size=(n, 2)).astype(np.float32)
    rows, cols, weights, attempts, p = _sample_connected_er(
        n, float(config["gamma"]), coords, np.random.RandomState(graph_seed)
    )
    graph = csr_matrix(
        (
            np.concatenate((weights, weights)),
            (np.concatenate((rows, cols)), np.concatenate((cols, rows))),
        ),
        shape=(n, n),
    )
    cost_float = dijkstra(graph, directed=False, return_predecessors=False)
    cost_int = _integerize_metric(cost_float)
    _validate_metric(cost_int, exhaustive=n <= 500, seed=coordinate_seed)

    solver_seed = base_seed + 1_000_003 * int(index) + 7
    solver = str(config["label_solver"])
    if solver == "concorde":
        tour, solver_meta = solve_concorde_explicit(
            cost_int, seed=solver_seed, concorde_bin=Path(config["concorde_bin"])
        )
    elif solver == "lkh":
        tour, solver_meta = solve_lkh_explicit(
            cost_int, seed=solver_seed, lkh_bin=Path(config["lkh_bin"])
        )
    else:
        raise ValueError(f"Unknown label solver: {solver}")

    packed = pack_upper_triangle(cost_int)
    metadata: dict[str, Any] = {
        "archive": str(Path("instances") / archive_path.name),
        "instance_id": int(index),
        "dataset": "er_euclidean_shortest_path",
        "n": n,
        "gamma": float(config["gamma"]),
        "p": float(p),
        "coordinate_seed": int(coordinate_seed),
        "graph_seed": int(graph_seed),
        "connectivity_attempts": int(attempts),
        "reference_cost_int": _tour_cost_int(cost_int, tour),
        "matrix_sha256": hashlib.sha256(packed.tobytes()).hexdigest(),
        **solver_meta,
    }
    _write_archive(
        archive_path,
        {
            "coords_raw": coords,
            "cost_upper_int": packed,
            "reference_tour": tour.astype(np.int32),
            "edge_index": np.stack((rows, cols), axis=1).astype(np.int32),
            "edge_weight": weights.astype(np.float32),
        },
        metadata,
    )
    return metadata


def generate_er_dataset(args: argparse.Namespace) -> None:
    n = int(args.n)
    if n < 4 or int(args.count) <= 0:
        raise ValueError("--n must be at least 4 and --count must be positive")
    if float(args.gamma) <= 0:
        raise ValueError("--gamma must be positive")
    solver = str(args.label_solver)
    if solver == "concorde" and not args.concorde_bin:
        raise ValueError("--concorde-bin is required when --label-solver concorde")
    if solver == "lkh" and not args.lkh_bin:
        raise ValueError("--lkh-bin is required when --label-solver lkh")

    output_dir = Path(args.output_dir).expanduser().resolve()
    config = {
        "n": n,
        "gamma": float(args.gamma),
        "base_seed": int(args.base_seed),
        "label_solver": solver,
        "concorde_bin": str(Path(args.concorde_bin).expanduser().resolve()) if args.concorde_bin else "",
        "lkh_bin": str(Path(args.lkh_bin).expanduser().resolve()) if args.lkh_bin else "",
        "output_dir": str(output_dir),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    print(
        f"[MetricGen] n={n} count={args.count} gamma={args.gamma} solver={solver} "
        f"workers={args.solver_workers} -> {output_dir}",
        flush=True,
    )
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    workers = max(1, int(args.solver_workers))
    if workers == 1:
        for index in range(int(args.count)):
            results.append(_generate_er_instance(index, config))
            if (index + 1) % max(1, int(args.progress_every)) == 0 or index + 1 == int(args.count):
                print(f"[MetricGen] n={n}: {index + 1}/{args.count}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_generate_er_instance, index, config): index for index in range(int(args.count))}
            for done, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if done % max(1, int(args.progress_every)) == 0 or done == int(args.count):
                    print(f"[MetricGen] n={n}: {done}/{args.count}", flush=True)
    results.sort(key=lambda item: int(item["instance_id"]))
    if len(results) != int(args.count):
        raise RuntimeError(f"Expected {args.count} records, received {len(results)}")
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in results:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")
    summary = {
        "dataset": "er_euclidean_shortest_path",
        "n": n,
        "count": int(args.count),
        "gamma": float(args.gamma),
        "base_seed": int(args.base_seed),
        "label_solver": solver,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _parse_hcp_edges(path: Path) -> tuple[int, np.ndarray]:
    text = path.read_text(encoding="utf-8", errors="replace")
    # A subset of the published Flinders HCP files contains the historical
    # misspellings ``DIMENTION`` and ``EDGE_DATA_SELECTION``.  Accept these
    # aliases while retaining the ordinary TSPLIB spellings.
    dim_match = re.search(r"(?im)^\s*DIMEN[ST]ION\s*[:=]\s*(\d+)\s*$", text)
    if dim_match is None:
        raise ValueError(f"HCP file lacks DIMENSION: {path}")
    n = int(dim_match.group(1))
    section_match = re.search(r"(?im)^\s*EDGE_DATA_(?:SECTION|SELECTION)\s*$", text)
    if section_match is None:
        raise ValueError(f"HCP file lacks EDGE_DATA_SECTION: {path}")
    body = text[section_match.end() :]
    edges: list[tuple[int, int]] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.upper() == "EOF" or line == "-1":
            break
        values = [int(token) for token in re.findall(r"-?\d+", line)]
        if len(values) < 2:
            continue
        u, v = values[0], values[1]
        if not (1 <= u <= n and 1 <= v <= n) or u == v:
            raise ValueError(f"Invalid HCP edge ({u}, {v}) in {path}")
        edges.append((u - 1, v - 1))
    if not edges:
        raise ValueError(f"HCP file has no edges: {path}")
    return n, np.asarray(edges, dtype=np.int32)


def generate_hcp_dataset(args: argparse.Namespace) -> None:
    source_dir = Path(args.hcp_dir).expanduser().resolve()
    files = sorted(path for path in source_dir.iterdir() if path.is_file())
    if args.max_n is not None:
        files = [path for path in files if _parse_hcp_edges(path)[0] <= int(args.max_n)]
    if len(files) != int(args.expected_instances):
        raise ValueError(
            f"Expected {args.expected_instances} HCP files in {source_dir}, found {len(files)}"
        )
    output_dir = Path(args.output_dir).expanduser().resolve()
    records: list[dict[str, Any]] = []
    for index, source_path in enumerate(files):
        n, edges = _parse_hcp_edges(source_path)
        costs = np.full((n, n), 2, dtype=np.uint32)
        np.fill_diagonal(costs, 0)
        costs[edges[:, 0], edges[:, 1]] = 1
        costs[edges[:, 1], edges[:, 0]] = 1
        _validate_metric(costs, exhaustive=n <= 500, seed=index)
        packed = pack_upper_triangle(costs)
        archive_path = output_dir / "instances" / f"instance_{index:03d}.npz"
        metadata = {
            "archive": str(Path("instances") / archive_path.name),
            "instance_id": index,
            "dataset": "flinders_hcp_1_2",
            "n": n,
            "source_filename": source_path.name,
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "reference_cost_int": n,
            "label_status": "exact_objective",
            "matrix_sha256": hashlib.sha256(packed.tobytes()).hexdigest(),
        }
        _write_archive(
            archive_path,
            {
                "coords_proxy": np.zeros((n, 2), dtype=np.float32),
                "cost_upper_int": packed,
                "reference_tour": np.empty(0, dtype=np.int32),
            },
            metadata,
        )
        records.append(metadata)
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("er", "hcp"), default="er")
    parser.add_argument("--n", type=int)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=4.0)
    parser.add_argument("--base-seed", type=int, default=20_260_730)
    parser.add_argument("--label-solver", choices=("concorde", "lkh"), default="concorde")
    parser.add_argument("--solver-workers", type=int, default=1)
    parser.add_argument("--concorde-bin", type=str, default="")
    parser.add_argument("--lkh-bin", type=str, default="")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--hcp-dir", type=str, default="")
    parser.add_argument("--expected-instances", type=int, default=45)
    parser.add_argument("--max-n", type=int, default=None,
                        help="In --mode hcp, retain only source instances with n at most this value.")
    args = parser.parse_args()
    if args.mode == "er":
        if args.n is None:
            parser.error("--n is required in --mode er")
        generate_er_dataset(args)
    else:
        if not args.hcp_dir:
            parser.error("--hcp-dir is required in --mode hcp")
        generate_hcp_dataset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
