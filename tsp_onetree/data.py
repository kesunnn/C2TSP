import math
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

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

def cosine_anneal(epoch: int, start: float, end: float, total_epochs: int) -> float:
    """Smoothly anneal a scalar from start to end over total_epochs."""
    if total_epochs <= 1:
        return float(end)
    t = min(max(epoch - 1, 0), total_epochs - 1) / float(total_epochs - 1)
    w = 0.5 * (1.0 + math.cos(math.pi * t))
    return float(end + (start - end) * w)
