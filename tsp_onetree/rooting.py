"""Relabel helpers for evaluating a fixed-root 1-tree model.

The implicit 1-tree layer is deliberately implemented with internal node 0 as
its root.  These helpers move a requested *original* node to index 0 before a
model forward and map model outputs back afterwards.  Geometry and tour costs
are unchanged; only the solver's node labels are changed.
"""
from __future__ import annotations

import torch


def validate_root(root: int, n: int) -> int:
    """Validate and return a zero-based original-city root index."""
    root = int(root)
    if not 0 <= root < int(n):
        raise ValueError(f"--root must be in [0, {int(n) - 1}], got {root}")
    return root


def root_zero_permutation(n: int, root: int, device: torch.device) -> torch.Tensor | None:
    """Return ``new_index -> original_index``; ``None`` preserves root zero."""
    root = validate_root(root, n)
    if root == 0:
        return None
    original = torch.arange(n, device=device)
    return torch.cat((original[root:root + 1], original[:root], original[root + 1:]))


def relabel_inputs_to_root_zero(
    coords: torch.Tensor,
    dist: torch.Tensor,
    root: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Place original ``root`` at the model's internal index zero."""
    if coords.ndim != 3 or dist.ndim != 3:
        raise ValueError("coords and dist must be batched tensors")
    n = int(coords.shape[1])
    if dist.shape[1:] != (n, n):
        raise ValueError("dist shape must be (B, n, n) for coords shape (B, n, d)")
    perm = root_zero_permutation(n, root, coords.device)
    if perm is None:
        return coords, dist, None
    coords_model = coords.index_select(1, perm)
    dist_model = dist.index_select(1, perm).index_select(2, perm)
    return coords_model, dist_model, perm


def restore_node_matrix(matrix: torch.Tensor, perm: torch.Tensor | None) -> torch.Tensor:
    """Map a batched ``(..., n, n)`` model output back to original labels."""
    if perm is None:
        return matrix
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device)
    return matrix.index_select(-2, inv).index_select(-1, inv)


def restore_lambda_nonroot(
    lambda_nr: torch.Tensor,
    perm: torch.Tensor | None,
    root: int,
) -> torch.Tensor:
    """Map internal-root duals to original-label, non-root ordering.

    ``lambda_nr`` is stored in ascending internal indices 1..n-1.  The result
    follows the ordering expected by :func:`compute_C_mod`: ascending original
    indices excluding the selected original root.
    """
    if perm is None:
        return lambda_nr
    if lambda_nr.ndim != 2 or lambda_nr.shape[1] != perm.numel() - 1:
        raise ValueError("lambda_nr must have shape (B, n - 1)")
    n = int(perm.numel())
    root = validate_root(root, n)
    full = torch.zeros((lambda_nr.shape[0], n), dtype=lambda_nr.dtype, device=lambda_nr.device)
    full[:, perm[1:]] = lambda_nr
    nonroot = torch.arange(n, device=lambda_nr.device)
    nonroot = nonroot[nonroot != root]
    return full.index_select(1, nonroot)
