from typing import Tuple

import torch
from torch.utils.data import Dataset


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

    if int(tour.min().item()) >= 1:
        tour = tour - 1

    if len(tour) >= n + 1 and int(tour[0].item()) == int(tour[-1].item()):
        tour = tour[:-1]

    if len(tour) != n:
        raise ValueError(f"Parsed tour length {len(tour)} != num cities {n}.")
    return coords, tour


class ConcordeTSPDataset(Dataset):
    """File-backed dataset for Concorde-labeled Euclidean TSP instances."""

    def __init__(self, path: str, take: int | None = None, skip: int = 0):
        self.path = path
        self.coords = []
        self.dist_matrices = []
        self.opt_tours = []

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
