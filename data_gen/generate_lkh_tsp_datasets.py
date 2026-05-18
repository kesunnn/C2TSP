#!/usr/bin/env python
"""Generate Euclidean TSP datasets labelled by LKH-3.

For each (n, count) we sample `count` instances uniformly in [0,1]^2 with seeds
`base_seed + idx`, integer-rescale to `coord_scale`, and solve with the local
LKH binary (RUNS=1, MAX_TRIALS=n, no TIME_LIMIT — matches the benchmark used
to estimate wall time).

Outputs per size:
  <output_dir>/tsp{n}_lkh.txt        — Concorde-compatible: 'flat coords output 1..n.. 1'
  <output_dir>/tsp{n}_lkh_times.json — per-instance wall_s / lkh_total_s / tour_cost_int + summary
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


DEFAULT_COORD_SCALE = 1_000_000


# Globals populated in worker initialiser so each LKH call doesn't re-resolve them.
_W_LKH_BIN: Path | None = None
_W_COORD_SCALE: int = DEFAULT_COORD_SCALE
_W_BASE_SEED: int = 0
_W_N: int = 0


def _worker_init(lkh_bin: str, coord_scale: int, base_seed: int, n: int) -> None:
    global _W_LKH_BIN, _W_COORD_SCALE, _W_BASE_SEED, _W_N
    _W_LKH_BIN = Path(lkh_bin)
    _W_COORD_SCALE = int(coord_scale)
    _W_BASE_SEED = int(base_seed)
    _W_N = int(n)


def _write_tsplib(path: Path, int_coords: np.ndarray, name: str) -> None:
    n = int_coords.shape[0]
    with path.open("w") as f:
        f.write(f"NAME: {name}\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {n}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i in range(n):
            f.write(f"{i + 1} {int(int_coords[i, 0])} {int(int_coords[i, 1])}\n")
        f.write("EOF\n")


_TOTAL_RE = __import__("re").compile(r"(?:Time\.total|Total\s+Running\s+Time)\s*[:=]\s*([0-9eE.+-]+)\s*sec")
_COST_RE = __import__("re").compile(r"Cost\.min\s*=\s*(-?\d+)")


def _parse_tour(path: Path, n: int) -> np.ndarray:
    text = path.read_text()
    in_section = False
    tokens: list[int] = []
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
                tokens.append(int(tok))
            except ValueError:
                pass
    if len(tokens) != n:
        raise RuntimeError(f"Tour file {path}: got {len(tokens)} ids, expected {n}")
    tour = np.asarray(tokens, dtype=np.int64) - 1
    if sorted(tour.tolist()) != list(range(n)):
        raise RuntimeError(f"Tour file {path} is not a permutation of 0..{n-1}")
    return tour


def _format_concorde_line(coords: np.ndarray, tour: np.ndarray) -> str:
    coord_tokens = " ".join(format(float(x), ".12g") for x in coords.reshape(-1))
    tour_tokens = [str(int(v) + 1) for v in tour.tolist()]
    tour_tokens.append(str(int(tour[0]) + 1))
    return f"{coord_tokens} output {' '.join(tour_tokens)}\n"


def _solve_one(idx: int) -> dict:
    """Sample one instance and solve it with LKH. Runs in worker."""
    assert _W_LKH_BIN is not None
    n = _W_N
    seed = _W_BASE_SEED + idx
    rng = np.random.RandomState(seed)
    coords = rng.uniform(0.0, 1.0, size=(n, 2)).astype(np.float64)
    int_coords = np.rint(coords * float(_W_COORD_SCALE)).astype(np.int64)

    with tempfile.TemporaryDirectory(prefix=f"lkh_n{n}_i{idx}_") as td:
        td = Path(td)
        tsp = td / "inst.tsp"
        par = td / "inst.par"
        out = td / "inst.tour"
        _write_tsplib(tsp, int_coords, name=f"inst_{idx}")
        par.write_text(
            f"PROBLEM_FILE = {tsp}\n"
            f"OUTPUT_TOUR_FILE = {out}\n"
            f"RUNS = 1\n"
            f"MAX_TRIALS = {n}\n"
            f"SEED = {seed}\n"
            f"TRACE_LEVEL = 1\n"  # need stdout to capture Cost.min and Time.total
        )
        t0 = time.perf_counter()
        proc = subprocess.run(
            [str(_W_LKH_BIN), str(par)],
            cwd=td,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        wall = time.perf_counter() - t0
        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        if not out.exists():
            return {
                "idx": idx, "ok": False, "wall_s": wall, "returncode": proc.returncode,
                "stdout_tail": stdout[-1500:], "stderr_tail": stderr[-1500:],
            }
        try:
            tour = _parse_tour(out, n)
        except Exception as e:
            return {
                "idx": idx, "ok": False, "wall_s": wall, "returncode": proc.returncode,
                "error": f"{type(e).__name__}: {e}",
            }
        cost_min = None
        m = _COST_RE.search(stdout)
        if m:
            try:
                cost_min = int(m.group(1))
            except ValueError:
                pass
        total_s = None
        m = _TOTAL_RE.search(stdout)
        if m:
            try:
                total_s = float(m.group(1))
            except ValueError:
                pass
    line = _format_concorde_line(coords, tour)
    return {
        "idx": idx, "ok": True, "wall_s": wall, "returncode": proc.returncode,
        "lkh_total_s": total_s, "tour_cost_int": cost_min, "line": line,
    }


def _generate_size(
    n: int,
    count: int,
    base_seed: int,
    workers: int,
    coord_scale: int,
    lkh_bin: Path,
    out_dir: Path,
    progress_every: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"tsp{n}_lkh.txt"
    json_path = out_dir / f"tsp{n}_lkh_times.json"

    print(f"[LKHGen] n={n} count={count} workers={workers} -> {txt_path}", flush=True)
    t_total = time.time()
    results: list[dict] = [None] * count  # type: ignore
    done = 0
    last_log = t_total
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(str(lkh_bin), int(coord_scale), int(base_seed), int(n)),
    ) as pool:
        futures = {pool.submit(_solve_one, idx): idx for idx in range(count)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"idx": idx, "ok": False, "wall_s": 0.0, "error": f"{type(e).__name__}: {e}"}
            results[idx] = r
            done += 1
            now = time.time()
            if done % max(1, progress_every) == 0 or done == count or (now - last_log) > 30:
                elapsed = now - t_total
                rate = done / max(elapsed, 1e-6)
                eta = (count - done) / max(rate, 1e-6)
                ok_n = sum(1 for r in results if r and r.get("ok"))
                print(
                    f"[LKHGen] n={n}: {done}/{count} done | ok={ok_n} | "
                    f"{elapsed:.1f}s elapsed | {rate:.2f} inst/s | ETA {eta:.0f}s",
                    flush=True,
                )
                last_log = now

    with txt_path.open("w") as f:
        for r in results:
            if r is None or not r.get("ok"):
                continue
            f.write(r["line"])

    walls = np.array([r["wall_s"] for r in results if r and r.get("ok")], dtype=np.float64)
    summary = {
        "n": n,
        "count_requested": count,
        "count_ok": int(walls.size),
        "count_failed": int(count - walls.size),
        "base_seed": int(base_seed),
        "workers": int(workers),
        "wall_total_s": float(time.time() - t_total),
        "wall_per_inst_s": {
            "mean": float(walls.mean()) if walls.size else None,
            "median": float(np.median(walls)) if walls.size else None,
            "min": float(walls.min()) if walls.size else None,
            "max": float(walls.max()) if walls.size else None,
            "p95": float(np.percentile(walls, 95)) if walls.size else None,
        },
        "instances": [
            {k: v for k, v in r.items() if k != "line"}
            for r in results if r is not None
        ],
    }
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[LKHGen] n={n} done | ok={summary['count_ok']}/{count} | "
        f"wall={summary['wall_total_s']:.1f}s | "
        f"per-inst mean={summary['wall_per_inst_s']['mean']:.2f}s "
        f"med={summary['wall_per_inst_s']['median']:.2f}s",
        flush=True,
    )
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ns", type=str, required=True,
                   help="Comma-separated TSP sizes, e.g. '1000,2000,5000'.")
    p.add_argument("--count", type=int, default=1000,
                   help="Instances per size (default 1000).")
    p.add_argument("--base_seed", type=int, default=2_000_000,
                   help="RNG seed; instance idx i uses base_seed+i.")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2 * 2 - 0),
                   help=f"Parallel workers (default {os.cpu_count()}). "
                        "Each LKH call is single-threaded.")
    p.add_argument("--coord_scale", type=int, default=DEFAULT_COORD_SCALE)
    p.add_argument("--lkh_bin", type=str, required=True,
                   help="Path to the LKH-3 executable. Build LKH-3 from "
                        "http://akira.ruc.dk/~keld/research/LKH-3/.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to write tsp{n}_lkh.txt and tsp{n}_lkh_times.json.")
    p.add_argument("--progress_every", type=int, default=25)
    args = p.parse_args()

    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    lkh_bin = Path(args.lkh_bin).expanduser().resolve()
    if not lkh_bin.is_file():
        raise FileNotFoundError(f"LKH binary not found: {lkh_bin}")
    out_dir = Path(args.output_dir).expanduser().resolve()

    print(f"[LKHGen] LKH={lkh_bin}\n[LKHGen] output_dir={out_dir}\n"
          f"[LKHGen] sizes={ns} count_per_size={args.count} workers={args.workers}",
          flush=True)
    summaries = []
    for n in ns:
        summaries.append(_generate_size(
            n=n, count=args.count, base_seed=args.base_seed,
            workers=args.workers, coord_scale=args.coord_scale,
            lkh_bin=lkh_bin, out_dir=out_dir,
            progress_every=args.progress_every,
        ))

    print("\n=== Aggregate timing summary ===", flush=True)
    print(f"{'n':>6} {'ok':>6} {'fail':>5} {'wall_total_s':>14} {'mean_s':>10} {'med_s':>10} {'p95_s':>10}", flush=True)
    for s in summaries:
        w = s["wall_per_inst_s"]
        print(
            f"{s['n']:>6} {s['count_ok']:>6} {s['count_failed']:>5} "
            f"{s['wall_total_s']:>14.1f} "
            f"{(w['mean'] or 0):>10.2f} {(w['median'] or 0):>10.2f} {(w['p95'] or 0):>10.2f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
