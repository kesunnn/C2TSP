import json
import math
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def pack_upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Pack the strict upper triangle of a square symmetric matrix row-wise."""
    arr = np.asarray(matrix)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"matrix must be square, got {arr.shape}")
    n = arr.shape[0]
    packed = np.empty(n * (n - 1) // 2, dtype=arr.dtype)
    offset = 0
    for i in range(n - 1):
        width = n - i - 1
        packed[offset : offset + width] = arr[i, i + 1 :]
        offset += width
    return packed


def unpack_upper_triangle(packed: np.ndarray, n: int, *, dtype: np.dtype | None = None) -> np.ndarray:
    """Restore a symmetric, zero-diagonal matrix packed by :func:`pack_upper_triangle`."""
    values = np.asarray(packed)
    n = int(n)
    expected = n * (n - 1) // 2
    if values.ndim != 1 or values.size != expected:
        raise ValueError(
            f"packed upper triangle for n={n} must have {expected} values, got {values.shape}"
        )
    out = np.zeros((n, n), dtype=dtype or values.dtype)
    offset = 0
    for i in range(n - 1):
        width = n - i - 1
        out[i, i + 1 :] = values[offset : offset + width]
        out[i + 1 :, i] = values[offset : offset + width]
        offset += width
    return out

class TSPDataset(Dataset):
    """Random Euclidean TSP instances in [0,1]^2."""

    def __init__(self, num_instances: int, num_cities: int, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.coords = torch.tensor(
            rng.uniform(size=(num_instances, num_cities, 2)), dtype=torch.float32
        )
        diff = self.coords.unsqueeze(2) - self.coords.unsqueeze(1)
        self.dist_matrices = diff.norm(dim=-1)

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        return self.coords[idx], self.dist_matrices[idx]


def _euclidean_dist_matrix(coords: torch.Tensor) -> torch.Tensor:
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)
    return diff.norm(dim=-1)


def _parse_concorde_line(line: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parse one line from the official Concorde TSP datasets.

    Expected format used by the Joshi et al. TSP datasets:
      x1 y1 x2 y2 ... xn yn output t1 t2 ... tn t1

    The stored tour is typically 1-indexed and repeats the start node at the end.
    We convert to zero-based indexing and drop the repeated final node if present.
    """
    toks = line.strip().split()
    if not toks:
        raise ValueError("Encountered an empty line in Concorde dataset.")
    try:
        out_idx = toks.index("output")
    except ValueError as exc:
        raise ValueError("Concorde line missing 'output' separator.") from exc

    coord_toks = toks[:out_idx]
    if len(coord_toks) % 2 != 0:
        raise ValueError("Coordinate token count must be even.")
    n = len(coord_toks) // 2
    coords = torch.tensor([float(x) for x in coord_toks], dtype=torch.float32).view(n, 2)

    tour_toks = toks[out_idx + 1:]
    if len(tour_toks) < n:
        raise ValueError(f"Tour token count {len(tour_toks)} shorter than num cities {n}.")
    tour = torch.tensor([int(x) for x in tour_toks], dtype=torch.long)

    # Convert 1-indexed tours to 0-indexed if needed.
    if int(tour.min().item()) >= 1:
        tour = tour - 1

    # Drop repeated closing node if present.
    if len(tour) >= n + 1 and int(tour[0].item()) == int(tour[-1].item()):
        tour = tour[:-1]

    if len(tour) != n:
        raise ValueError(f"Parsed tour length {len(tour)} != num cities {n}.")
    return coords, tour


class ConcordeTSPDataset(Dataset):
    """File-backed dataset for official Concorde-labeled Euclidean TSP instances."""

    def __init__(self, path: str, take: int | None = None, skip: int = 0):
        self.path = path
        self.coords: List[torch.Tensor] = []
        self.dist_matrices: List[torch.Tensor] = []
        self.opt_tours: List[torch.Tensor] = []

        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if skip < 0:
            raise ValueError("skip must be nonnegative")
        lines = lines[skip:]
        if take is not None:
            lines = lines[:take]
        if not lines:
            raise ValueError(f"No samples loaded from {path} with skip={skip}, take={take}.")

        for ln in lines:
            coords, tour = _parse_concorde_line(ln)
            self.coords.append(coords)
            self.dist_matrices.append(_euclidean_dist_matrix(coords))
            self.opt_tours.append(tour)

        nset = {c.shape[0] for c in self.coords}
        if len(nset) != 1:
            raise ValueError("This loader expects a fixed-size dataset per file/run.")
        self.num_cities = next(iter(nset))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        return self.coords[idx], self.dist_matrices[idx], self.opt_tours[idx]


class MetricTSPDataset(Dataset):
    """Lazy reader for explicit symmetric-metric TSP archives.

    The dataset generator writes one ``.npz`` archive per instance and a
    ``manifest.jsonl`` inventory.  Unlike :class:`ConcordeTSPDataset`, this
    reader never derives distances from coordinates: it expands the packed
    integer cost matrix and exposes an instance-normalized float matrix for the
    frozen network.  The original integer matrix is returned as well so callers
    can report costs in the solver's reference units.
    """

    def __init__(self, path: str | Path, take: int | None = None, skip: int = 0):
        requested = Path(path).expanduser().resolve()
        self.manifest_path = requested / "manifest.jsonl" if requested.is_dir() else requested
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"MetricTSPDataset manifest not found: {self.manifest_path}")
        if skip < 0:
            raise ValueError("skip must be nonnegative")

        entries: list[dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {self.manifest_path}:{line_no}"
                    ) from exc
                if not isinstance(entry, dict) or "archive" not in entry or "n" not in entry:
                    raise ValueError(
                        f"Manifest entry {line_no} must contain at least 'archive' and 'n'"
                    )
                entries.append(entry)
        entries = entries[skip:]
        if take is not None:
            entries = entries[:take]
        if not entries:
            raise ValueError(
                f"No metric instances loaded from {self.manifest_path} with skip={skip}, take={take}."
            )

        self.entries = entries
        self.sizes = tuple(sorted({int(entry["n"]) for entry in entries}))
        # Dataset A has a fixed size per manifest.  Dataset B (the Flinders
        # HCP collection) deliberately spans several sizes; callers must
        # group those entries into size-homogeneous network batches.
        self.num_cities = self.sizes[0] if len(self.sizes) == 1 else None

    def __len__(self) -> int:
        return len(self.entries)

    def metadata(self, idx: int) -> dict[str, Any]:
        """Return the manifest metadata for one instance without expanding its matrix."""
        return dict(self.entries[idx])

    def _archive_path(self, idx: int) -> Path:
        archive = Path(str(self.entries[idx]["archive"]))
        return archive if archive.is_absolute() else self.manifest_path.parent / archive

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        archive_path = self._archive_path(idx)
        if not archive_path.is_file():
            raise FileNotFoundError(f"Metric instance archive not found: {archive_path}")
        with np.load(archive_path, allow_pickle=False) as data:
            coords_key = "coords_raw" if "coords_raw" in data.files else "coords_proxy"
            if coords_key not in data.files or "cost_upper_int" not in data.files:
                raise ValueError(f"Malformed metric archive: {archive_path}")
            coords_np = np.asarray(data[coords_key], dtype=np.float32)
            packed = np.asarray(data["cost_upper_int"], dtype=np.uint32)
            tour_np = (
                np.asarray(data["reference_tour"], dtype=np.int64)
                if "reference_tour" in data.files
                else np.empty(0, dtype=np.int64)
            )
            stored_meta: dict[str, Any] = {}
            if "metadata_json" in data.files:
                stored_meta = json.loads(str(data["metadata_json"].item()))

        n = int(entry["n"])
        if coords_np.shape != (n, 2):
            raise ValueError(f"{archive_path}: coordinates must have shape {(n, 2)}, got {coords_np.shape}")
        cost_int_np = unpack_upper_triangle(packed, n, dtype=np.uint32)
        max_cost = int(cost_int_np.max())
        if max_cost <= 0:
            raise ValueError(f"{archive_path}: cost matrix has no positive off-diagonal entry")
        metadata = {**entry, **stored_meta}
        coords = torch.from_numpy(coords_np)
        d_model = torch.from_numpy(cost_int_np.astype(np.float32, copy=False)) / float(max_cost)
        # Individual costs are bounded by 1e6 in Dataset A and by 2 in Dataset
        # B, so int32 avoids doubling peak memory for TSP5000. Tour sums are
        # explicitly promoted to int64 by evaluation helpers.
        cost_int = torch.from_numpy(cost_int_np.astype(np.int32, copy=False))
        tour = torch.from_numpy(tour_np)
        return coords, d_model, tour, cost_int, metadata

def cosine_anneal(epoch: int, start: float, end: float, total_epochs: int) -> float:
    """Smoothly anneal a scalar from start to end over total_epochs."""
    if total_epochs <= 1:
        return float(end)
    t = min(max(epoch - 1, 0), total_epochs - 1) / float(total_epochs - 1)
    w = 0.5 * (1.0 + math.cos(math.pi * t))
    return float(end + (start - end) * w)
