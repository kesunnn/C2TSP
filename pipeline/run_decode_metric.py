"""Pure L1--L4 decoder for stored explicit metric TSP datasets.

This is deliberately separate from :mod:`pipeline.run_decode`: it retains the
v1 fixed-root decoding path and adds only the reader required by Dataset A/B.
Input must be a metric ``manifest.jsonl`` or its containing directory.  The
network receives a per-instance [0, 1] cost matrix; decoded tours and gaps are
computed against the original integer cost matrix.
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

from pipeline.run_lkh import _load_model
from tsp_onetree.decode import decode_gumbel, decode_tours_ablation


DECODE_LEVELS = ("L1", "L2x1", "L2x10", "L2x100", "L3", "L4")


def _unpack_upper_triangle(packed: np.ndarray, n: int) -> np.ndarray:
    """Restore a symmetric zero-diagonal matrix from its strict upper triangle."""
    values = np.asarray(packed)
    expected = n * (n - 1) // 2
    if values.ndim != 1 or values.size != expected:
        raise ValueError(f"Packed matrix for n={n} needs {expected} values, got {values.shape}")
    matrix = np.zeros((n, n), dtype=values.dtype)
    offset = 0
    for row in range(n - 1):
        width = n - row - 1
        matrix[row, row + 1:] = values[offset:offset + width]
        matrix[row + 1:, row] = values[offset:offset + width]
        offset += width
    return matrix


class MetricDecodeDataset:
    """Minimal, lazy manifest reader local to this metric-only evaluator."""

    def __init__(self, path: str | Path, take: int | None = None, skip: int = 0):
        requested = Path(path).expanduser().resolve()
        self.manifest_path = requested / "manifest.jsonl" if requested.is_dir() else requested
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Metric manifest not found: {self.manifest_path}")
        if skip < 0:
            raise ValueError("skip must be nonnegative")
        entries: list[dict[str, Any]] = []
        for line_no, line in enumerate(self.manifest_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {self.manifest_path}:{line_no}") from exc
            if not isinstance(entry, dict) or "archive" not in entry or "n" not in entry:
                raise ValueError(f"Manifest entry {line_no} needs 'archive' and 'n'")
            if "reference_cost_int" not in entry:
                raise ValueError(f"Manifest entry {line_no} needs 'reference_cost_int'")
            entries.append(entry)
        self.entries = entries[skip:] if take is None else entries[skip:skip + take]
        if not self.entries:
            raise ValueError(f"No metric instances loaded from {self.manifest_path}")
        self.sizes = tuple(sorted({int(entry["n"]) for entry in self.entries}))
        self.num_cities = self.sizes[0] if len(self.sizes) == 1 else None

    def __len__(self) -> int:
        return len(self.entries)

    def metadata(self, idx: int) -> dict[str, Any]:
        return dict(self.entries[idx])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        entry = self.entries[idx]
        archive = Path(str(entry["archive"]))
        archive_path = archive if archive.is_absolute() else self.manifest_path.parent / archive
        if not archive_path.is_file():
            raise FileNotFoundError(f"Metric archive not found: {archive_path}")
        with np.load(archive_path, allow_pickle=False) as data:
            coord_key = "coords_raw" if "coords_raw" in data.files else "coords_proxy"
            if coord_key not in data.files or "cost_upper_int" not in data.files:
                raise ValueError(f"Malformed metric archive: {archive_path}")
            coords = np.asarray(data[coord_key], dtype=np.float32)
            packed = np.asarray(data["cost_upper_int"], dtype=np.uint32)
        n = int(entry["n"])
        if coords.shape != (n, 2):
            raise ValueError(f"{archive_path}: expected coordinates {(n, 2)}, got {coords.shape}")
        cost_int = _unpack_upper_triangle(packed, n)
        max_cost = int(cost_int.max())
        if max_cost <= 0:
            raise ValueError(f"{archive_path}: metric has no positive off-diagonal cost")
        d_model = cost_int.astype(np.float32, copy=False) / float(max_cost)
        return (
            torch.from_numpy(coords),
            torch.from_numpy(d_model),
            torch.from_numpy(cost_int.astype(np.float32, copy=False)),
            dict(entry),
        )


def _twoopt_passes_for(level: str) -> int:
    return {"L2x1": 1, "L2x10": 10, "L2x100": 100}.get(level, 0)


def _extract_cost(out: dict, level: str) -> np.ndarray:
    if level == "L1":
        return out["s1_mu_greedy"]["raw_cost"]
    if level.startswith("L2x"):
        return out["s1_mu_greedy"]["lk_cost"]
    if level == "L3":
        return out["s4_gumbel_muM"]["raw_cost"]
    if level == "L4":
        return out["best_cost"]
    raise ValueError(f"Unknown decode level {level!r}")


def _summarize(level: str, costs: np.ndarray, reference_costs: np.ndarray) -> dict[str, Any]:
    gaps = 100.0 * (costs - reference_costs) / np.where(reference_costs > 0, reference_costs, 1.0)
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
    ap.add_argument("--dataset", type=str, required=True, help="Metric manifest.jsonl or its directory.")
    ap.add_argument("--levels", type=str, default="L1,L2x1,L2x10,L2x100,L3,L4")
    ap.add_argument("--take", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_gumbel_draws", type=int, default=20)
    ap.add_argument("--gumbel_scale", type=float, default=0.20)
    ap.add_argument("--l4_num_draws", type=int, default=0)
    ap.add_argument("--l4_noise_type", type=str, default="covariance")
    ap.add_argument("--l4_tau", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output_json", type=str, required=True)
    args = ap.parse_args()

    levels = [token.strip() for token in args.levels.split(",") if token.strip()]
    if not levels or any(level not in DECODE_LEVELS for level in levels):
        raise ValueError(f"--levels must be a nonempty comma-list from {DECODE_LEVELS}")
    device = torch.device(args.device)
    print(f"[decode-metric] loading model from {args.run_dir}")
    model, _ = _load_model(Path(args.run_dir).resolve(), device)
    take = args.take if args.take > 0 else None
    dataset = MetricDecodeDataset(args.dataset, take=take, skip=args.skip)
    total = len(dataset)
    print(f"[decode-metric] n={dataset.num_cities or list(dataset.sizes)}, {total} instances")

    reference_costs = np.asarray(
        [float(dataset.metadata(i)["reference_cost_int"]) for i in range(total)], dtype=np.float64
    )
    needed_2opt = sorted({_twoopt_passes_for(level) for level in levels})
    if not any(level in {"L1", "L3", "L4"} for level in levels) and 0 in needed_2opt:
        needed_2opt.remove(0)
    if not needed_2opt:
        needed_2opt = [0]

    by_size: dict[int, list[int]] = {}
    for idx in range(total):
        by_size.setdefault(int(dataset.metadata(idx)["n"]), []).append(idx)
    batch_groups = [
        indices[start:start + args.batch_size]
        for n in sorted(by_size)
        for indices in [by_size[n]]
        for start in range(0, len(indices), args.batch_size)
    ]
    costs_by_level = {level: np.full(total, np.nan) for level in levels}
    t_forward_total = t_decode_total = 0.0
    for batch_no, indices in enumerate(batch_groups, 1):
        items = [dataset[idx] for idx in indices]
        coords = torch.stack([item[0] for item in items]).to(device)
        d_model = torch.stack([item[1] for item in items]).to(device)
        d_decode = torch.stack([item[2] for item in items]).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            mu, _, _, aux = model(coords, d_model, return_decode_aux=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_forward_total += time.perf_counter() - started

        for budget in needed_2opt:
            started = time.perf_counter()
            out = decode_tours_ablation(
                mu, aux["C_mod"], aux["cand_mask"], d_decode, root=0,
                twoopt_passes=int(budget), num_root_pairs=0,
                num_gumbel_draws=int(args.num_gumbel_draws) if "L3" in levels else 0,
                gumbel_scale=float(args.gumbel_scale),
                seed_base=int(args.seed + indices[0] * 7919), pair_prob=aux.get("pair_prob"),
                use_lk_alpha=False,
            )
            t_decode_total += time.perf_counter() - started
            for level in levels:
                needs_this_budget = (
                    (level in {"L1", "L3"} and budget == needed_2opt[0])
                    or (level == "L2x1" and budget == 1)
                    or (level == "L2x10" and budget == 10)
                    or (level == "L2x100" and budget == 100)
                )
                if needs_this_budget:
                    costs_by_level[level][indices] = _extract_cost(out, level)
            if "L4" in levels and budget == needed_2opt[0]:
                draws = int(args.l4_num_draws) if args.l4_num_draws > 0 else int(args.num_gumbel_draws)
                started = time.perf_counter()
                l4_out = decode_gumbel(
                    mu, aux["C_mod"], aux["cand_mask"], d_decode, root=0,
                    twoopt_passes=0, num_draws=draws, gumbel_scale=float(args.gumbel_scale),
                    seed_base=int(args.seed + indices[0] * 7919), use_lk_alpha=False,
                    noise_type=str(args.l4_noise_type), tau=float(args.l4_tau),
                    seed_split=0.0, use_mu_repair=False,
                )
                t_decode_total += time.perf_counter() - started
                costs_by_level["L4"][indices] = _extract_cost(l4_out, "L4")
        print(f"[decode-metric] batch {batch_no}/{len(batch_groups)} (n={coords.shape[1]}) done")

    summary = {level: _summarize(level, costs_by_level[level], reference_costs) for level in levels}
    output = {
        "dataset": str(Path(args.dataset).resolve()),
        "dataset_format": "metric",
        "run_dir": str(Path(args.run_dir).resolve()),
        "n": dataset.num_cities,
        "sizes": list(dataset.sizes),
        "num_instances": total,
        "levels": levels,
        "root": 0,
        "seed": int(args.seed),
        "t_forward_total_s": float(t_forward_total),
        "t_decode_total_s": float(t_decode_total),
        "t_forward_per_inst_ms": float(1000 * t_forward_total / total),
        "t_decode_per_inst_ms": float(1000 * t_decode_total / total),
        "summary": summary,
        "per_instance": {
            level: {"cost": costs_by_level[level].tolist(), "opt_cost": reference_costs.tolist()}
            for level in levels
        },
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[decode-metric] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
