"""Relabel helpers for evaluating a fixed-root 1-tree model.

The implicit 1-tree layer is deliberately implemented with internal node 0 as
its root.  These helpers move a requested *original* node to index 0 before a
model forward and map model outputs back afterwards.  Geometry and tour costs
are unchanged; only the solver's node labels are changed.
"""
from __future__ import annotations

from collections.abc import Sequence

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


@torch.no_grad()
def ensemble_root_outputs(
    model: torch.nn.Module,
    coords: torch.Tensor,
    dist: torch.Tensor,
    roots: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Average root-specific marginal and decoder-score outputs in original order.

    ``model`` always uses internal node zero as its rooted 1-tree node. This
    helper relabels every requested original-city root to that position, restores
    both outputs to the original labels, and averages them. The returned
    ``c_mod_mean`` is a decoder score ensemble; it is not associated with one
    common Held--Karp dual vector.
    """
    if coords.ndim != 3 or dist.ndim != 3:
        raise ValueError("coords and dist must be batched tensors")
    n = int(coords.shape[1])
    if dist.shape[1:] != (n, n):
        raise ValueError("dist shape must be (B, n, n) for coords shape (B, n, d)")

    root_list = [validate_root(root, n) for root in roots]
    if not root_list:
        raise ValueError("roots must contain at least one root index")
    if len(set(root_list)) != len(root_list):
        raise ValueError("roots must not contain duplicate root indices")
    root_list.sort()

    mu_sum = None
    c_mod_sum = None
    mu_sq_sum = None
    c_mod_sq_sum = None
    for root in root_list:
        coords_model, dist_model, perm = relabel_inputs_to_root_zero(coords, dist, root)
        mu, _, _, aux = model(coords_model, dist_model, return_decode_aux=True)
        if "C_mod" not in aux:
            raise KeyError("model decode auxiliary output is missing C_mod")
        mu_restored = restore_node_matrix(mu, perm)
        c_mod_restored = restore_node_matrix(aux["C_mod"], perm)
        if mu_restored.shape != dist.shape or c_mod_restored.shape != dist.shape:
            raise ValueError("model outputs must have shape (B, n, n)")

        if mu_sum is None:
            mu_sum = mu_restored.clone()
            c_mod_sum = c_mod_restored.clone()
            mu_sq_sum = mu_restored.square()
            c_mod_sq_sum = c_mod_restored.square()
        else:
            mu_sum = mu_sum + mu_restored
            c_mod_sum = c_mod_sum + c_mod_restored
            mu_sq_sum = mu_sq_sum + mu_restored.square()
            c_mod_sq_sum = c_mod_sq_sum + c_mod_restored.square()

    assert mu_sum is not None
    assert c_mod_sum is not None
    assert mu_sq_sum is not None
    assert c_mod_sq_sum is not None
    root_count = float(len(root_list))
    mu_mean = mu_sum / root_count
    c_mod_mean = c_mod_sum / root_count
    mu_std = (mu_sq_sum / root_count - mu_mean.square()).clamp_min(0.0).sqrt()
    c_mod_std = (c_mod_sq_sum / root_count - c_mod_mean.square()).clamp_min(0.0).sqrt()
    degree_mismatch = (mu_mean.sum(dim=-1) - 2.0).abs()

    diagnostics = {
        "mu_root_std_mean": mu_std.mean(),
        "c_mod_root_std_mean": c_mod_std.mean(),
        "mu_symmetry_max": (mu_mean - mu_mean.transpose(-2, -1)).abs().amax(),
        "mu_diagonal_abs_max": torch.diagonal(mu_mean, dim1=-2, dim2=-1).abs().amax(),
        "degree_mismatch_mean": degree_mismatch.mean(),
        "degree_mismatch_max": degree_mismatch.amax(),
        "c_mod_symmetry_max": (c_mod_mean - c_mod_mean.transpose(-2, -1)).abs().amax(),
    }
    return mu_mean, c_mod_mean, diagnostics
