#!/usr/bin/env python
import argparse
import contextlib
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np


@contextlib.contextmanager
def _pushd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _format_concorde_line(coords: np.ndarray, tour: np.ndarray) -> str:
    coord_tokens = " ".join(format(float(x), ".12g") for x in coords.reshape(-1))
    tour_tokens = [str(int(v) + 1) for v in tour.tolist()]
    tour_tokens.append(str(int(tour[0]) + 1))
    return f"{coord_tokens} output {' '.join(tour_tokens)}\n"


def _write_tsplib(path: Path, int_coords: np.ndarray, name: str) -> None:
    n = int_coords.shape[0]
    with path.open("w", encoding="utf-8") as f:
        f.write(f"NAME: {name}\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {n}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i in range(n):
            f.write(f"{i + 1} {int(int_coords[i, 0])} {int(int_coords[i, 1])}\n")
        f.write("EOF\n")


def _parse_concorde_tour(sol_path: Path, n: int) -> np.ndarray:
    tokens: list[int] = []
    with sol_path.open("r", encoding="utf-8") as f:
        for line in f:
            for tok in line.strip().split():
                try:
                    tokens.append(int(tok))
                except ValueError:
                    continue
    if not tokens:
        raise RuntimeError(f"Empty Concorde tour file: {sol_path}")
    declared_n = tokens[0]
    body = tokens[1:]
    if declared_n != n or len(body) != n:
        raise RuntimeError(
            f"Concorde tour file has wrong size: declared {declared_n}, body {len(body)}, expected {n}."
        )
    tour = np.asarray(body, dtype=np.int64)
    if len(np.unique(tour)) != n or tour.min() != 0 or tour.max() != n - 1:
        raise RuntimeError("Concorde returned a tour that is not a permutation of 0..n-1.")
    return tour


def _solve_concorde_tour(
    coords: np.ndarray,
    coordinate_scale: float,
    seed: int,
    concorde_bin: Path,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape [n, 2], got {coords.shape}")
    if not concorde_bin.is_file():
        raise FileNotFoundError(f"Concorde binary not found at {concorde_bin}")
    n = coords.shape[0]

    int_coords = np.rint(coords * float(coordinate_scale)).astype(np.int64)

    with tempfile.TemporaryDirectory(prefix="concorde_dataset_gen_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        stem = f"inst_{int(seed)}"
        tsp_path = tmpdir_path / f"{stem}.tsp"
        sol_path = tmpdir_path / f"{stem}.sol"
        _write_tsplib(tsp_path, int_coords, name=stem)

        cmd = [
            str(concorde_bin),
            "-s", str(int(seed)),
            "-x",
            "-o", str(sol_path),
            str(tsp_path),
        ]
        with _pushd(tmpdir_path):
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        # This Concorde build exits nonzero on its cleanup path even after a
        # successful optimal solve. Trust the solution file as the ground truth
        # of success, and only fail when the file is missing or malformed.
        if not sol_path.exists():
            stdout = proc.stdout.decode("utf-8", errors="replace")
            stderr = proc.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Concorde produced no tour file (exit={proc.returncode}) on instance seed={seed}.\n"
                f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
            )
        return _parse_concorde_tour(sol_path, n)


def _generate_dataset_file(
    output_path: Path,
    num_instances: int,
    num_cities: int,
    seed: int,
    coordinate_scale: float,
    progress_every: int,
    concorde_bin: Path,
) -> None:
    if int(num_instances) <= 0:
        raise ValueError(f"num_instances must be positive, got {num_instances}.")
    if int(num_cities) < 4:
        raise ValueError(f"num_cities must be at least 4, got {num_cities}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    t0 = time.time()
    total = int(num_instances)

    print(
        f"[DatasetGen] Writing {output_path} with {total} TSP-{num_cities} instances "
        f"sampled uniformly in [0,1]^2."
    )
    with output_path.open("w", encoding="utf-8") as f:
        for idx in range(total):
            coords = rng.uniform(low=0.0, high=1.0, size=(num_cities, 2)).astype(np.float64)
            tour = _solve_concorde_tour(
                coords,
                coordinate_scale=coordinate_scale,
                seed=seed + idx,
                concorde_bin=concorde_bin,
            )
            f.write(_format_concorde_line(coords, tour))
            f.flush()
            done = idx + 1
            if done % max(1, int(progress_every)) == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                eta = (total - done) / max(rate, 1e-6)
                print(
                    f"[DatasetGen] {output_path.name}: {done}/{total} done | "
                    f"{elapsed:.1f}s elapsed | {rate:.2f} inst/s | ETA {eta:.0f}s"
                )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate train/eval Concorde-format Euclidean TSP datasets with coordinates "
            "sampled uniformly from [0,1]^2."
        )
    )
    parser.add_argument("--num_cities", type=int, required=True)
    parser.add_argument("--train_size", type=int, required=True)
    parser.add_argument("--eval_size", type=int, required=True)
    parser.add_argument("--train_output", type=str, required=True)
    parser.add_argument("--eval_output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=12345,
                        help="Base RNG seed for coordinate sampling and Concorde tie-breaking.")
    parser.add_argument("--eval_seed", type=int, default=None,
                        help="Optional separate seed for eval generation. Defaults to seed + 1_000_000.")
    parser.add_argument("--coordinate_scale", type=float, default=1000000.0,
                        help="Scale factor used before passing coordinates to Concorde EUC_2D.")
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--concorde_bin", type=str, required=True,
                        help="Path to the compiled Concorde TSP binary. "
                             "Download/build Concorde from "
                             "http://www.math.uwaterloo.ca/tsp/concorde.html.")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip train-set generation (useful for eval-only splits).")
    args = parser.parse_args()

    eval_seed = int(args.eval_seed) if args.eval_seed is not None else int(args.seed) + 1_000_000
    concorde_bin = Path(args.concorde_bin).expanduser().resolve()

    if not args.skip_train:
        _generate_dataset_file(
            output_path=Path(args.train_output),
            num_instances=int(args.train_size),
            num_cities=int(args.num_cities),
            seed=int(args.seed),
            coordinate_scale=float(args.coordinate_scale),
            progress_every=int(args.progress_every),
            concorde_bin=concorde_bin,
        )
    _generate_dataset_file(
        output_path=Path(args.eval_output),
        num_instances=int(args.eval_size),
        num_cities=int(args.num_cities),
        seed=eval_seed,
        coordinate_scale=float(args.coordinate_scale),
        progress_every=int(args.progress_every),
        concorde_bin=concorde_bin,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
