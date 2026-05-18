import argparse
import functools
import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.utils.data import DataLoader, Dataset


# ================================================================
# §1  Data generation
# ================================================================


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


# ================================================================
# §2  Candidate-neighborhood graph
# ================================================================


def build_candidate_mask(dist_matrix: torch.Tensor, k: int) -> torch.Tensor:
    """
    Symmetric kNN candidate graph. Returns bool mask of shape (B, n, n).
    Includes reverse-kNN closure and excludes self loops.
    """
    B, n, _ = dist_matrix.shape
    device = dist_matrix.device
    full_mask = ~torch.eye(n, device=device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    if k >= n - 1:
        return full_mask

    d = dist_matrix.clone()
    d = d + torch.eye(n, device=device).unsqueeze(0) * 1e9
    knn_idx = torch.topk(d, k=k, largest=False, dim=-1).indices
    cand = torch.zeros(B, n, n, device=device, dtype=torch.bool)
    cand.scatter_(dim=-1, index=knn_idx, value=True)
    cand = cand | cand.transpose(-2, -1)
    cand = cand & full_mask
    return cand


# ================================================================
# §3  One-tree-compatible primal-dual edge/node GNN
# ================================================================


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int, keepdim: bool = False,
                eps: float = 1e-8) -> torch.Tensor:
    """Numerically safe masked mean for dense batched tensors."""
    w = mask.to(dtype=x.dtype)
    num = (x * w).sum(dim=dim, keepdim=keepdim)
    den = w.sum(dim=dim, keepdim=keepdim).clamp_min(eps)
    return num / den


def _edge_projection_diagnostics(
        raw_logits: torch.Tensor,
        proj_logits: torch.Tensor,
        active_mask: torch.Tensor,
        node_additive: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Diagnostics for primal residual projection on the active edge support."""
    mask = active_mask.to(dtype=raw_logits.dtype)
    mask = 0.5 * (mask + mask.transpose(-2, -1))
    row_den = mask.sum(dim=-1).clamp_min(1.0)
    grand_den = (0.5 * mask.sum(dim=(-2, -1))).clamp_min(1.0)
    row_leak = (((proj_logits * mask).sum(dim=-1) / row_den).abs()).mean()
    grand_leak = ((0.5 * (proj_logits * mask).sum(dim=(-2, -1)) / grand_den).abs()).mean()
    removed = raw_logits - proj_logits
    removed_num = (mask * removed.pow(2)).sum(dim=(-2, -1))
    removed_den = (mask * raw_logits.pow(2)).sum(dim=(-2, -1)).clamp_min(1e-12)
    info = {
        "row_leak": row_leak,
        "grand_leak": grand_leak,
        "removed_energy_frac": (removed_num / removed_den).mean(),
    }
    if node_additive is not None:
        info["node_additive"] = node_additive
    return info


def project_edge_residual_dense(
        sym_logits: torch.Tensor,
        active_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dense double-centering fallback used only when weighted projection fails."""
    B, n, _ = sym_logits.shape
    device = sym_logits.device
    full_mask = ~torch.eye(n, device=device, dtype=torch.bool).unsqueeze(0)
    full_f = full_mask.to(dtype=sym_logits.dtype)

    L = 0.5 * (sym_logits + sym_logits.transpose(-2, -1))
    L = L * full_f
    row_mean = L.mean(dim=-1, keepdim=True)
    grand_mean = L.mean(dim=(-2, -1), keepdim=True)
    proj = L - row_mean - row_mean.transpose(-2, -1) + grand_mean
    proj = 0.5 * (proj + proj.transpose(-2, -1))
    if active_mask is not None:
        proj = proj * active_mask.to(dtype=proj.dtype)
    proj = proj * full_f

    node_additive = row_mean.squeeze(-1) - 0.5 * grand_mean.view(B, 1)
    info = _edge_projection_diagnostics(
        L,
        proj,
        active_mask if active_mask is not None else full_mask,
        node_additive=node_additive,
    )
    info["fallback_frac"] = torch.zeros((), device=device, dtype=sym_logits.dtype)
    return proj, info


def project_edge_residual_candidate_weighted(
        sym_logits: torch.Tensor,
        active_mask: torch.Tensor,
        ridge: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Project logits away from the constant / node-additive subspace on the candidate graph.

    We solve the weighted least-squares problem
        min_{a, c} sum_{i<j} w_ij (L_ij - a_i - a_j - c)^2
    with w_ij given by the active candidate mask, using a mean-zero basis for a.
    The projected residual is then L^⊥ = L - (a_i + a_j + c), restricted back to
    the active support before the implicit 1-tree layer sees it.
    """
    B, n, _ = sym_logits.shape
    device, dtype = sym_logits.device, sym_logits.dtype
    full_mask = ~torch.eye(n, device=device, dtype=torch.bool).unsqueeze(0)
    full_f = full_mask.to(dtype=dtype)

    L = 0.5 * (sym_logits + sym_logits.transpose(-2, -1))
    L = L * full_f
    W = 0.5 * (active_mask.to(dtype=dtype) + active_mask.transpose(-2, -1).to(dtype=dtype))
    W = W * full_f

    d = W.sum(dim=-1)
    b = (W * L).sum(dim=-1)
    M = 0.5 * W.sum(dim=(-2, -1))
    m_rhs = 0.5 * (W * L).sum(dim=(-2, -1))

    Q = mean_zero_basis(n, device=device, dtype=dtype)
    Qb = Q.unsqueeze(0).expand(B, -1, -1)
    core = torch.diag_embed(d) + W
    H = torch.matmul(Qb.transpose(1, 2), torch.matmul(core, Qb))
    v = torch.matmul(Qb.transpose(1, 2), d.unsqueeze(-1)).squeeze(-1)
    rhs_alpha = torch.matmul(Qb.transpose(1, 2), b.unsqueeze(-1)).squeeze(-1)

    q = Q.shape[1]
    system = torch.zeros(B, q + 1, q + 1, device=device, dtype=dtype)
    rhs = torch.zeros(B, q + 1, device=device, dtype=dtype)
    system[:, :q, :q] = H
    system[:, :q, q] = v
    system[:, q, :q] = v
    system[:, q, q] = M
    rhs[:, :q] = rhs_alpha
    rhs[:, q] = m_rhs

    sol = safe_batched_solve(system, rhs.unsqueeze(-1), ridge=ridge).squeeze(-1)
    alpha = sol[:, :q]
    c = sol[:, q]
    node_additive = torch.matmul(Qb, alpha.unsqueeze(-1)).squeeze(-1)

    additive = node_additive.unsqueeze(-1) + node_additive.unsqueeze(-2) + c.view(B, 1, 1)
    proj = (L - additive) * W
    proj = 0.5 * (proj + proj.transpose(-2, -1))
    proj = proj * full_f

    bad = (~torch.isfinite(proj).all(dim=(-2, -1))) | (~torch.isfinite(node_additive).all(dim=-1)) | (M <= 0)
    if bool(bad.any()):
        dense_proj, dense_info = project_edge_residual_dense(L[bad], active_mask=active_mask[bad])
        proj = proj.clone()
        node_additive = node_additive.clone()
        proj[bad] = dense_proj
        node_additive[bad] = dense_info["node_additive"]

    info = _edge_projection_diagnostics(L, proj, active_mask, node_additive=node_additive)
    info["fallback_frac"] = bad.to(dtype=dtype).mean()
    return proj, info


class SymmetricEdgeNodeBlock(nn.Module):
    """Alternating edge->node and node->edge updates on an undirected candidate graph.

    This block is tailored to the rooted 1-tree layer:
      - edge states carry primal edge preference information,
      - node states summarize incident-edge pressure,
      - the edge update is symmetric in (i, j),
      - only candidate edges participate in message passing.

    v10 changes:
      - edge_update hidden layer widened by ``edge_hidden_mult`` (default 3)
        to reduce the compression bottleneck in the first linear.
      - Edge summary enriched: mean + std + max → projected back to edge_dim
        so the edge_update can distinguish neighbourhood variance / extremes.
    """

    def __init__(self, node_dim: int, edge_dim: int, edge_attr_dim: int,
                 edge_hidden_mult: int = 3):
        super().__init__()
        self.edge_to_node_msg = nn.Sequential(
            nn.Linear(edge_dim + edge_attr_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.edge_to_node_gate = nn.Sequential(
            nn.Linear(edge_dim + edge_attr_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 1),
        )
        self.node_update = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.node_norm = nn.LayerNorm(node_dim)

        # -- Rich edge summary: [mean, std, max] projected to edge_dim ------
        self.summary_proj = nn.Sequential(
            nn.Linear(3 * edge_dim, edge_dim),
            nn.SiLU(),
        )

        # -- Wider hidden in edge update (v10) --------------------------------
        edge_update_in = edge_dim + edge_attr_dim + 2 * node_dim + 2 * edge_dim
        edge_hidden = edge_dim * max(1, int(edge_hidden_mult))
        self.edge_update = nn.Sequential(
            nn.Linear(edge_update_in, edge_hidden),
            nn.SiLU(),
            nn.Linear(edge_hidden, edge_dim),
        )
        self.edge_norm = nn.LayerNorm(edge_dim)

    def forward(
            self,
            h: torch.Tensor,
            e: torch.Tensor,
            edge_attr: torch.Tensor,
            cand_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, n, _ = h.shape
        cand_f = cand_mask.unsqueeze(-1).to(dtype=h.dtype)

        # ---- Edge -> node aggregation ----
        edge_input = torch.cat([e, edge_attr], dim=-1)
        msg = self.edge_to_node_msg(edge_input)
        gate = self.edge_to_node_gate(edge_input).squeeze(-1)
        gate = gate.masked_fill(~cand_mask, -1e9)
        attn = torch.softmax(gate, dim=-1)
        attn = attn * cand_mask.to(dtype=attn.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        agg = (attn.unsqueeze(-1) * msg).sum(dim=2)
        h = self.node_norm(h + self.node_update(torch.cat([h, agg], dim=-1)))

        # ---- Rich edge summary: mean / std / max → project to edge_dim ----
        edge_mean = masked_mean(e, cand_f, dim=2, keepdim=False)
        e_centred = e - edge_mean.unsqueeze(2)
        edge_std = masked_mean(e_centred.pow(2), cand_f, dim=2, keepdim=False).clamp_min(1e-8).sqrt()
        # Max pool: fill non-candidate entries with -inf so they don't win
        edge_max = (e - 1e9 * (1.0 - cand_f)).amax(dim=2)
        edge_summary = self.summary_proj(torch.cat([edge_mean, edge_std, edge_max], dim=-1))

        h_i = h.unsqueeze(2).expand(-1, -1, n, -1)
        h_j = h.unsqueeze(1).expand(-1, n, -1, -1)
        s_i = edge_summary.unsqueeze(2).expand(-1, -1, n, -1)
        s_j = edge_summary.unsqueeze(1).expand(-1, n, -1, -1)

        pair_feat = torch.cat(
            [
                e,
                edge_attr,
                h_i + h_j,
                torch.abs(h_i - h_j),
                s_i + s_j,
                torch.abs(s_i - s_j),
            ],
            dim=-1,
        )
        e_prop = self.edge_update(pair_feat)
        e_prop = 0.5 * (e_prop + e_prop.transpose(1, 2))
        e = self.edge_norm(e + e_prop)
        e = e * cand_f
        return h, e


class NeighborhoodEdgePriorEncoder(nn.Module):
    """One-tree-compatible symmetric encoder with optional refinement-state channels.

    The geometric backbone always stays anchored to the original Euclidean
    instance.  Optional residual-state channels let later refinement rounds see
    the accumulated reduced-cost landscape without overloading the dist_matrix
    input with non-Euclidean semantics.

    v10 changes:
      - ``edge_hidden_mult`` forwarded to each GNN block for wider edge MLP.
      - ``gradient_checkpoint``: if True, recompute GNN layer activations during
        backward to halve peak memory (at ~30% speed cost).
      - kNN rank added as an extra edge attribute (base_edge_attr_dim 6 → 7).
    """

    def __init__(
            self,
            node_dim: int = 128,
            edge_dim: int = 32,
            num_layers: int = 3,
            residual_state_dim: int = 3,
            edge_head_with_cost: bool = False,
            edge_hidden_mult: int = 3,
            gradient_checkpoint: bool = False,
    ):
        super().__init__()
        self.node_feat_dim = 5
        self.base_edge_attr_dim = 7  # v10: +1 for kNN rank
        self.residual_state_dim = residual_state_dim
        self.edge_attr_dim = self.base_edge_attr_dim + self.residual_state_dim
        self.edge_head_with_cost = bool(edge_head_with_cost)
        self.gradient_checkpoint = bool(gradient_checkpoint)

        self.node_embed = nn.Sequential(
            nn.Linear(self.node_feat_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.edge_embed = nn.Sequential(
            nn.Linear(self.edge_attr_dim, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim),
        )
        self.layers = nn.ModuleList(
            [SymmetricEdgeNodeBlock(node_dim, edge_dim, self.edge_attr_dim,
                                    edge_hidden_mult=edge_hidden_mult)
             for _ in range(num_layers)]
        )
        if self.edge_head_with_cost:
            # Enriched head: 3 layers, wider first, takes [edge_embedding, normalized_D] as input.
            # The extra +1 channel is the instance-normalized D value for that edge.
            # Final layer is tiny-initialized so at step 1 the head contribution is near-zero,
            # which keeps early training close to v7 behavior without being bit-for-bit identical.
            self.edge_head = nn.Sequential(
                nn.Linear(edge_dim + 1, edge_dim * 2),
                nn.SiLU(),
                nn.Linear(edge_dim * 2, edge_dim),
                nn.SiLU(),
                nn.Linear(edge_dim, 1),
            )
            nn.init.normal_(self.edge_head[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.edge_head[-1].bias)
        else:
            # v7 head: 2 layers, edge embedding only, no D input.
            self.edge_head = nn.Sequential(
                nn.Linear(edge_dim, edge_dim),
                nn.SiLU(),
                nn.Linear(edge_dim, 1),
            )

        # v16+: Certificate refinement head. Produces a per-edge scalar g(e) that
        # additively corrects the (μ−0.5) baseline certificate:
        #     certificate = (μ₀ − 0.5) + α · tanh(g(e))
        #     tilt = −β·τ·P(certificate)
        # At init, g ≈ 0 so tilt ≈ −β·τ·P(μ₀ − 0.5) (baseline Fisher-gradient
        # certificate). As training proceeds, g learns nonlinear corrections
        # that use geometry + residual_state features the marginal μ alone
        # cannot express. tanh + small α keeps the correction in a trust
        # region around the principled baseline direction.
        # Only active in round 2 (round_idx > 0).
        self.certificate_head = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, 1),
        )
        nn.init.normal_(self.certificate_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.certificate_head[-1].bias)

    def _build_invariant_features(
            self,
            coords: torch.Tensor,
            dist_matrix: torch.Tensor,
            cand_mask: torch.Tensor,
            residual_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, n, _ = coords.shape
        device = coords.device
        full_mask = ~torch.eye(n, device=device, dtype=torch.bool).unsqueeze(0)
        full_f = full_mask.to(dtype=coords.dtype)

        centered = coords - coords.mean(dim=1, keepdim=True)
        geom_scale = centered.norm(dim=-1).mean(dim=-1, keepdim=True).clamp_min(1e-6)
        centered = centered / geom_scale.unsqueeze(-1)
        radius = centered.norm(dim=-1, keepdim=True)

        dist_scale = (dist_matrix * full_f).sum(dim=(-2, -1), keepdim=True) / full_f.sum(dim=(-2, -1),
                                                                                         keepdim=True).clamp_min(1.0)
        dist_scale = dist_scale.clamp_min(1e-6)
        dist_norm = dist_matrix / dist_scale

        mean_d = (dist_norm * full_f).sum(dim=-1, keepdim=True) / full_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
        min_d = dist_norm.masked_fill(~full_mask, float("inf")).amin(dim=-1, keepdim=True)
        max_d = dist_norm.masked_fill(~full_mask, 0.0).amax(dim=-1, keepdim=True)
        var_d = ((dist_norm - mean_d) ** 2 * full_f).sum(dim=-1, keepdim=True) / full_f.sum(dim=-1,
                                                                                            keepdim=True).clamp_min(1.0)
        std_d = var_d.clamp_min(1e-12).sqrt()
        node_feat = torch.cat([radius, mean_d, min_d, max_d, std_d], dim=-1)

        r_i = radius.unsqueeze(2).expand(-1, -1, n, -1)
        r_j = radius.unsqueeze(1).expand(-1, n, -1, -1)
        inner = (centered.unsqueeze(2) * centered.unsqueeze(1)).sum(dim=-1, keepdim=True)

        # v10: kNN rank — for each row i, what is the distance-rank of column j?
        # Normalized to [0, 1] and symmetrized.  Gives the model explicit ordinal
        # neighbour-quality information that otherwise must be learned from scratch.
        d_for_rank = dist_norm.masked_fill(~full_mask, float("inf"))
        rank_indices = d_for_rank.argsort(dim=-1).argsort(dim=-1).to(dtype=coords.dtype)
        rank_norm = rank_indices / float(max(n - 2, 1))  # 0 = nearest, 1 = farthest
        rank_norm = 0.5 * (rank_norm + rank_norm.transpose(1, 2))
        rank_norm = rank_norm * full_f

        edge_attr = torch.cat(
            [
                dist_norm.unsqueeze(-1),
                dist_norm.pow(2).unsqueeze(-1),
                r_i + r_j,
                torch.abs(r_i - r_j),
                inner,
                cand_mask.unsqueeze(-1).to(dtype=coords.dtype),
                rank_norm.unsqueeze(-1),  # v10: kNN rank feature
            ],
            dim=-1,
        )
        if residual_state is None:
            residual_state = torch.zeros(B, n, n, self.residual_state_dim, device=device, dtype=coords.dtype)
        else:
            residual_state = residual_state.to(dtype=coords.dtype)
            if residual_state.shape != (B, n, n, self.residual_state_dim):
                raise ValueError(
                    f"residual_state must have shape {(B, n, n, self.residual_state_dim)}, got {tuple(residual_state.shape)}"
                )
        residual_state = 0.5 * (residual_state + residual_state.transpose(1, 2))
        residual_state = residual_state * full_mask.unsqueeze(-1).to(dtype=coords.dtype)
        edge_attr = torch.cat([edge_attr, residual_state], dim=-1)
        edge_attr = 0.5 * (edge_attr + edge_attr.transpose(1, 2))
        edge_attr = edge_attr * full_mask.unsqueeze(-1).to(dtype=edge_attr.dtype)
        return node_feat, edge_attr, full_mask, dist_norm

    def forward(
            self,
            coords: torch.Tensor,
            dist_matrix: torch.Tensor,
            cand_mask: torch.Tensor,
            residual_state: torch.Tensor | None = None,
            round_idx: int = 0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, n, _ = coords.shape
        node_feat, edge_attr, full_mask, dist_norm = self._build_invariant_features(coords, dist_matrix, cand_mask,
                                                                                    residual_state=residual_state)
        h = self.node_embed(node_feat)
        e = self.edge_embed(edge_attr) * cand_mask.unsqueeze(-1).to(dtype=coords.dtype)
        e = 0.5 * (e + e.transpose(1, 2))

        for layer in self.layers:
            if self.gradient_checkpoint and self.training:
                h, e = torch_checkpoint(layer, h, e, edge_attr, cand_mask,
                                        use_reentrant=False)
            else:
                h, e = layer(h, e, edge_attr, cand_mask)

        if self.edge_head_with_cost:
            # Enriched head: include normalized D as an explicit late-stage feature
            # alongside the edge embedding. This gives the head a direct nonlinear channel
            # from D -> logit, complementing whatever the GNN body has already encoded
            # about D in the edge embedding.
            d_feat = dist_norm.unsqueeze(-1).to(dtype=e.dtype)  # (B, n, n, 1)
            head_input = torch.cat([e, d_feat], dim=-1)  # (B, n, n, edge_dim+1)
            edge_logits = self.edge_head(head_input).squeeze(-1)
        else:
            edge_logits = self.edge_head(e).squeeze(-1)

        edge_logits = 0.5 * (edge_logits + edge_logits.transpose(1, 2))
        logits = edge_logits * cand_mask.to(dtype=edge_logits.dtype)

        # v16+: Certificate refinement (round 2 only).
        # Produces tanh-bounded per-edge scalar g(e) ∈ [−1, 1] that will be
        # scaled by α and added to (μ₀ − 0.5) in the model's tilt assembly.
        # Symmetrized and masked. Near-zero at init so tilt starts at the
        # baseline (μ₀ − 0.5) Fisher-gradient certificate.
        cert_correction = None
        if round_idx > 0:
            cert_raw = self.certificate_head(e).squeeze(-1)
            cert_raw = 0.5 * (cert_raw + cert_raw.transpose(1, 2))
            cert_raw = torch.tanh(cert_raw)
            cert_correction = cert_raw * cand_mask.to(dtype=cert_raw.dtype)

        aux = {
            "edge_logits": edge_logits,
            "full_mask": full_mask.to(dtype=coords.dtype),
            "cert_correction": cert_correction,
        }
        return logits, aux


# ================================================================
# §4  Rooted 1-tree partition and marginals
# ================================================================


@functools.lru_cache(maxsize=64)
def _mean_zero_basis_cached(m: int, device_str: str, dtype_str: str) -> torch.Tensor:
    if m <= 1:
        dtype = getattr(torch, dtype_str.split(".")[-1])
        return torch.zeros(m, 0, device=torch.device(device_str), dtype=dtype)
    dtype = getattr(torch, dtype_str.split(".")[-1])
    device = torch.device(device_str)
    H = torch.eye(m, device=device, dtype=dtype) - torch.ones(m, m, device=device, dtype=dtype) / float(m)
    Q, _ = torch.linalg.qr(H[:, : m - 1], mode="reduced")
    return Q


def mean_zero_basis(m: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Orthonormal basis Q in R^{m x (m-1)} spanning {x : 1^T x = 0}. Cached by (m, device, dtype)."""
    return _mean_zero_basis_cached(m, str(device), str(dtype))


def build_lambda_nonroot(z: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Map free gauge-fixed coordinates z to non-root multipliers lambda_nr = Q z."""
    if z.numel() == 0:
        return torch.zeros(z.shape[0], Q.shape[0], device=z.device, dtype=z.dtype)
    return z @ Q.transpose(0, 1)


def safe_batched_solve(
        A: torch.Tensor,
        b: torch.Tensor,
        ridge: float = 0.0,
        min_ridge: float = 1e-10,
        max_tries: int = 7,
) -> torch.Tensor:
    """Robust batched linear solve with guarded fallbacks.

    This routine is used by the implicit backward. It should never hard-crash training
    just because one Jacobian block is nearly singular or because CUDA's SVD driver refuses
    a pathological matrix. We therefore:
      1) try a batched solve first for speed,
      2) fall back per item with increasing ridge,
      3) try CPU least-squares / pseudoinverse as a last resort, and
      4) return zeros for irrecoverable samples instead of raising.
    """
    if A.numel() == 0:
        return b.clone()
    B, q, _ = A.shape
    A64 = A.to(torch.float64)
    b64 = b.to(torch.float64)
    eye = torch.eye(q, device=A.device, dtype=torch.float64).unsqueeze(0)
    eye_single = eye[0]
    base = max(float(ridge), float(min_ridge))

    finite_batch = torch.isfinite(A64).all(dim=(-2, -1)) & torch.isfinite(b64).all(dim=(-2, -1))

    # Fast path on the subset that looks numerically sane.
    if bool(finite_batch.all()):
        for k in range(max_tries):
            reg = base * (10.0 ** k)
            try:
                solve_out = torch.linalg.solve_ex(A64 + reg * eye, b64, check_errors=False)
                sol = solve_out.result
                info = solve_out.info
                if bool((info == 0).all()) and torch.isfinite(sol).all():
                    return sol.to(b.dtype)
            except RuntimeError:
                pass

    out = torch.zeros_like(b64)
    for bi in range(B):
        Ai = A64[bi]
        bi_vec = b64[bi]
        if (not torch.isfinite(Ai).all()) or (not torch.isfinite(bi_vec).all()):
            continue
        solved = False
        for k in range(max_tries):
            reg = base * (10.0 ** k)
            try:
                solve_out = torch.linalg.solve_ex(Ai + reg * eye_single, bi_vec, check_errors=False)
                sol = solve_out.result
                info = solve_out.info
                if bool((info == 0).all()) and torch.isfinite(sol).all():
                    out[bi] = sol
                    solved = True
                    break
            except RuntimeError:
                continue
        if solved:
            continue
        for k in range(max_tries):
            reg = base * (10.0 ** k)
            try:
                sol = torch.linalg.lstsq(Ai + reg * eye_single, bi_vec).solution
                if torch.isfinite(sol).all():
                    out[bi] = sol
                    solved = True
                    break
            except RuntimeError:
                continue
        if solved:
            continue

        # Last-resort CPU fallback. cuSOLVER SVD can fail on very ill-conditioned blocks;
        # moving a tiny q x q system to CPU is cheap and much more stable.
        try:
            reg = base * (10.0 ** max_tries)
            Ai_cpu = (Ai + reg * eye_single).detach().cpu()
            bi_cpu = bi_vec.detach().cpu()
            try:
                sol_cpu = torch.linalg.lstsq(Ai_cpu, bi_cpu).solution
            except RuntimeError:
                sol_cpu = torch.linalg.pinv(Ai_cpu) @ bi_cpu
            if torch.isfinite(sol_cpu).all():
                out[bi] = sol_cpu.to(out.device)
                solved = True
        except RuntimeError:
            solved = False

        # Final guard: keep zero adjoint for irrecoverable items rather than crashing.
        if not solved:
            out[bi].zero_()
    return out.to(b.dtype)


def explicit_jacobian_from_residual(G: torch.Tensor, z_var: torch.Tensor) -> torch.Tensor:
    """Assemble dG/dz using a single batched VJP call, with a row-wise fallback."""
    B, q = z_var.shape
    if q == 0:
        return torch.zeros(B, 0, 0, device=z_var.device, dtype=z_var.dtype)

    try:
        eye = torch.eye(q, device=G.device, dtype=G.dtype).unsqueeze(1).expand(q, B, q)
        J = torch.autograd.grad(
            G,
            z_var,
            grad_outputs=eye,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
            is_grads_batched=True,
        )[0]
        return J.permute(1, 0, 2).contiguous()
    except RuntimeError:
        rows = []
        for k in range(q):
            row_k = torch.autograd.grad(
                G[:, k].sum(),
                z_var,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0]
            rows.append(row_k.unsqueeze(1))
        return torch.cat(rows, dim=1)


# ────────────────────────────────────────────────────────────────
#  Batched CG solver for SPD systems  (H + ridge I) x = b
#  Used by both the forward Newton direction and the backward
#  IFT adjoint solve.  Runs all B instances in lockstep so
#  every matvec is a single batched GPU kernel.
# ────────────────────────────────────────────────────────────────

def batched_cg_solve(
        matvec_fn: Callable[[torch.Tensor], torch.Tensor],
        b: torch.Tensor,
        max_iter: int | None = None,
        tol: float = 1e-7,
        abs_tol: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched conjugate-gradient for SPD systems A x = b."""
    B, q = b.shape
    if q == 0:
        return b.clone(), torch.ones(B, device=b.device, dtype=torch.bool)
    if max_iter is None:
        max_iter = min(q, 20)

    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs = (r * r).sum(dim=-1)
    b_norm_sq = rs.clone()

    best_x = x.clone()
    best_rs = rs.clone()

    for _ in range(max_iter):
        Ap = matvec_fn(p)
        Ap = torch.nan_to_num(Ap, nan=0.0, posinf=0.0, neginf=0.0)

        pAp = (p * Ap).sum(dim=-1)
        safe_pAp = pAp.clamp_min(1e-30)
        alpha = rs / safe_pAp

        x = x + alpha.unsqueeze(-1) * p
        r = r - alpha.unsqueeze(-1) * Ap
        rs_new = (r * r).sum(dim=-1)

        improved = rs_new < best_rs
        best_x = torch.where(improved.unsqueeze(-1), x, best_x)
        best_rs = torch.where(improved, rs_new, best_rs)

        if (rs_new.sqrt().max().item() < abs_tol or
                (rs_new / b_norm_sq.clamp_min(1e-30)).sqrt().max().item() < tol):
            break

        beta = rs_new / rs.clamp_min(1e-30)
        p = r + beta.unsqueeze(-1) * p
        rs = rs_new

    return best_x, best_rs.sqrt() < max(tol, abs_tol)


def rooted_onetree_distribution(
        C_theta: torch.Tensor,
        lambda_nr: torch.Tensor,
        tau: float,
        root: int = 0,
        jitter: float = 1e-6,
        out_dtype: torch.dtype | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    r"""
    Differentiable rooted 1-tree marginals.

    The latent family is:
      - a spanning tree on V \{root\}
      - plus exactly two edges incident to the root.

    Costs are modified by degree multipliers lambda on the non-root nodes only:
      c^lambda_{ij} = c_{ij} + lambda_i + lambda_j.

    Returns:
      mu: symmetric edge marginal matrix, shape (B, n, n)
      info: diagnostics needed by the solver / loss.
    """
    B, n, _ = C_theta.shape
    device, dtype = C_theta.device, C_theta.dtype
    if out_dtype is None:
        out_dtype = dtype
    work_dtype = torch.float64 if dtype != torch.float64 else dtype
    C_theta = C_theta.to(work_dtype)
    lambda_nr = lambda_nr.to(work_dtype)
    if root != 0:
        raise NotImplementedError("Current implementation assumes root = 0 for simplicity.")
    if n < 3:
        raise ValueError("Rooted 1-tree model requires n >= 3.")

    m = n - 1
    lambda_full = torch.zeros(B, n, device=device, dtype=work_dtype)
    lambda_full[:, 1:] = lambda_nr

    C_mod = C_theta + lambda_full.unsqueeze(-1) + lambda_full.unsqueeze(-2)
    C_mod = 0.5 * (C_mod + C_mod.transpose(-2, -1))

    offdiag_mask = ~torch.eye(n, device=device, dtype=torch.bool).unsqueeze(0)
    shift = C_mod.masked_fill(~offdiag_mask, float("inf")).amin(dim=(-2, -1), keepdim=True)
    C_mod = C_mod - shift

    scaled = (-C_mod / tau).clamp(min=-60.0, max=60.0)
    W = torch.exp(scaled) * offdiag_mask.to(work_dtype)
    W = 0.5 * (W + W.transpose(-2, -1))

    # Tree partition and marginals on the non-root graph.
    W_nr = W[:, 1:, 1:]
    W_nr = W_nr * (~torch.eye(m, device=device, dtype=torch.bool).unsqueeze(0)).to(work_dtype)
    deg_nr_graph = W_nr.sum(dim=-1)
    L = torch.diag_embed(deg_nr_graph) - W_nr

    eye_q = torch.eye(m - 1, device=device, dtype=work_dtype).unsqueeze(0)
    K_base = 0.5 * (L[:, 1:, 1:] + L[:, 1:, 1:].transpose(-2, -1))
    chol = None
    logabsdet = None
    used_jitter = torch.zeros(B, device=device, dtype=work_dtype)

    # First try the exact cofactor without additive jitter. For the complete
    # weighted non-root graph this should typically already be SPD; only fall
    # back to jitter when numerical roundoff actually breaks Cholesky.
    chol_try, info = torch.linalg.cholesky_ex(K_base)
    if bool((info == 0).all()):
        chol = chol_try
        logabsdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
    else:
        failed = info != 0
        chol = torch.zeros_like(K_base)
        logabsdet = torch.empty(B, device=device, dtype=work_dtype)
        success = ~failed
        if bool(success.any()):
            chol[success] = chol_try[success]
            logabsdet[success] = 2.0 * torch.log(torch.diagonal(chol_try[success], dim1=-2, dim2=-1)).sum(dim=-1)
        for mult in (1.0, 10.0, 100.0, 1000.0):
            if not bool(failed.any()):
                break
            K_try = K_base[failed] + (mult * jitter) * eye_q.expand(B, -1, -1)[failed]
            chol_sub, info_sub = torch.linalg.cholesky_ex(K_try)
            good = info_sub == 0
            if bool(good.any()):
                idx = failed.nonzero(as_tuple=False).squeeze(-1)[good]
                chol[idx] = chol_sub[good]
                logabsdet[idx] = 2.0 * torch.log(torch.diagonal(chol_sub[good], dim1=-2, dim2=-1)).sum(dim=-1)
                used_jitter[idx] = mult * jitter
            failed_idx = failed.nonzero(as_tuple=False).squeeze(-1)
            new_failed = torch.ones_like(failed_idx, dtype=torch.bool)
            new_failed[good] = False
            failed = torch.zeros_like(failed)
            failed[failed_idx[new_failed]] = True
        if bool(failed.any()):
            raise RuntimeError("Failed to stabilize non-root Laplacian cofactor.")
    K_inv = torch.cholesky_inverse(chol)

    H = torch.zeros(B, m, m, device=device, dtype=work_dtype)
    H[:, 1:, 1:] = K_inv
    diag_H = torch.diagonal(H, dim1=-2, dim2=-1)
    R_eff = diag_H.unsqueeze(-1) + diag_H.unsqueeze(-2) - 2.0 * H
    mu_nr = W_nr * R_eff
    mu_nr = mu_nr * (~torch.eye(m, device=device, dtype=torch.bool).unsqueeze(0)).to(work_dtype)

    # Root-edge partition and marginals in the log domain.
    log_a = scaled[:, 0, 1:]
    pair_logits = log_a.unsqueeze(-1) + log_a.unsqueeze(-2)
    upper_mask = torch.triu(torch.ones(m, m, device=device, dtype=torch.bool), diagonal=1).unsqueeze(0)
    neg_inf = torch.tensor(float('-inf'), device=device, dtype=work_dtype)
    pair_logits_masked = torch.where(upper_mask, pair_logits, neg_inf)
    logZ_root = torch.logsumexp(pair_logits_masked.reshape(B, -1), dim=-1)
    pair_prob = torch.exp(pair_logits_masked - logZ_root.view(B, 1, 1))
    pair_prob = torch.where(upper_mask, pair_prob, torch.zeros_like(pair_prob))
    mu_root = pair_prob.sum(dim=-1) + pair_prob.sum(dim=-2)

    mu = torch.zeros(B, n, n, device=device, dtype=work_dtype)
    mu[:, 1:, 1:] = mu_nr
    mu[:, 0, 1:] = mu_root
    mu[:, 1:, 0] = mu_root

    logZ_tree = logabsdet
    logZ = logZ_tree + logZ_root

    degree = mu.sum(dim=-1)
    degree_nr = degree[:, 1:]

    exp_mod_cost = 0.5 * (C_mod * mu).sum(dim=(-2, -1))
    entropy = exp_mod_cost / tau + logZ

    mu_out = mu.to(out_dtype)
    info = {
        "mu": mu_out,
        "degree": degree.to(out_dtype),
        "degree_nr": degree_nr.to(out_dtype),
        "logZ": logZ.to(out_dtype),
        "entropy": entropy.to(out_dtype),
        "C_mod": C_mod.to(out_dtype),
        "root_edge_marginals": mu_root.to(out_dtype),
        "cofactor_jitter": used_jitter.to(out_dtype),
        "pair_prob": pair_prob.to(out_dtype),
    }
    return mu_out, info


def expected_degree_residual_from_z(
        C_theta: torch.Tensor,
        z: torch.Tensor,
        Q: torch.Tensor,
        tau: float,
        root: int = 0,
        jitter: float = 1e-6,
        out_dtype: torch.dtype | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Gauge-fixed stationarity map G(z) = Q^T (d_nr(mu)-2)."""
    lambda_nr = build_lambda_nonroot(z, Q)
    if out_dtype is None:
        out_dtype = z.dtype
    mu, info = rooted_onetree_distribution(
        C_theta, lambda_nr, tau=tau, root=root, jitter=jitter, out_dtype=out_dtype
    )
    F_nr = info["degree_nr"] - 2.0
    G = F_nr @ Q
    info["lambda_nr"] = lambda_nr.to(out_dtype)
    info["F_nr"] = F_nr
    info["G"] = G
    return G, info


def build_inner_tau_path(
        target_tau: float,
        inner_homotopy: bool = True,
        inner_tau_start: float = 0.30,
        inner_tau_mid: float = 0.22,
) -> List[float]:
    """Build a monotone nonincreasing inner solve path ending at target_tau.

    The returned path is used only as a forward continuation schedule. The final
    stage always runs at target_tau, so the saved equilibrium and backward pass
    remain aligned with the actual outer model.
    """
    target_tau = float(target_tau)
    if not inner_homotopy:
        return [target_tau]

    candidates = [target_tau]
    for val in (inner_tau_start, inner_tau_mid):
        try:
            tau_val = float(val)
        except (TypeError, ValueError):
            continue
        if math.isfinite(tau_val) and tau_val > target_tau + 1e-12:
            candidates.append(tau_val)

    path = []
    for tau_val in sorted(candidates, reverse=True):
        if not path or abs(tau_val - path[-1]) > 1e-10:
            path.append(tau_val)
    if abs(path[-1] - target_tau) > 1e-10:
        path.append(target_tau)
    return path


def split_lam_iters_across_stages(
        total_iters: int,
        num_stages: int,
        inner_final_frac: float = 0.75,
) -> List[int]:
    """Allocate most inner iterations to the final, sharpest continuation stage."""
    total_iters = max(int(total_iters), 1)
    num_stages = max(int(num_stages), 1)
    if num_stages == 1:
        return [total_iters]

    min_early = num_stages - 1
    final_iters = int(round(total_iters * float(inner_final_frac)))
    final_iters = max(1, min(final_iters, total_iters - min_early))
    rem = total_iters - final_iters
    alloc = [0] * num_stages
    base = rem // (num_stages - 1)
    extra = rem % (num_stages - 1)
    for i in range(num_stages - 1):
        alloc[i] = base + (1 if i < extra else 0)
    alloc[-1] = final_iters
    return alloc


# ================================================================
# §5  The implicit degree-balancing layer
# ================================================================


class RootedOneTreeImplicitLayer(torch.autograd.Function):
    """
    Forward:
      Solve G(z) = Q^T (d_nr(mu(C, z)) - 2) = 0 by damped Newton updates
      on the gauge-fixed entropic 1-tree dual system.

    Backward:
      Use the implicit function theorem on G(z, C) = 0:
        dz*/dC = - (dG/dz)^{-1} (dG/dC).
    """

    @staticmethod
    def forward(
            ctx,
            C_theta: torch.Tensor,
            lambda_init_nr: torch.Tensor,
            tau: float,
            root: int,
            lam_iters: int,
            lam_tol: float,
            lam_step: float,
            ift_ridge: float,
            ift_backward_tol: float,
            inner_homotopy: int,
            inner_tau_start: float,
            inner_tau_mid: float,
            inner_final_frac: float,
            cov_shrink: float,
            lm_damping: float,
    ) -> torch.Tensor:
        """
        Two-phase forward solver:
          Phase 1 (cheap):  HK subgradient warmup — no autograd, no Jacobian.
                            Each step is ONE marginal evaluation.
          Phase 2 (precise): Newton polish — explicit Jacobian for final convergence.
                            Also produces the Jacobian needed by the backward pass.

        Important geometry: G is the gradient of a concave dual objective, so dG/dz is
        negative semidefinite. We therefore solve Newton systems against H = -dG/dz,
        which is PSD up to numerical noise, and step by z <- z + H^{-1} G.
        """
        B, n, _ = C_theta.shape
        device, dtype = C_theta.device, C_theta.dtype
        solve_dtype = torch.float64 if dtype != torch.float64 else dtype
        if root != 0:
            raise NotImplementedError("Current implementation assumes root = 0.")
        if n < 3:
            raise ValueError("Need n >= 3 for rooted 1-tree implicit layer.")

        m = n - 1
        q = m - 1
        Q = mean_zero_basis(m, device=device, dtype=solve_dtype)
        C_solve = C_theta.detach().to(solve_dtype)
        tau_path = build_inner_tau_path(
            tau,
            inner_homotopy=bool(inner_homotopy),
            inner_tau_start=inner_tau_start,
            inner_tau_mid=inner_tau_mid,
        )
        stage_iters = split_lam_iters_across_stages(
            lam_iters,
            len(tau_path),
            inner_final_frac=inner_final_frac,
        )
        max_newton_step = 5.0
        tau_softest = float(tau_path[0])
        tau_target = float(tau)

        def _stage_cov_shrink(stage_tau: float) -> float:
            if float(cov_shrink) <= 0.0:
                return 0.0
            if tau_softest <= tau_target + 1e-12:
                return float(cov_shrink)
            frac = (tau_softest - float(stage_tau)) / max(tau_softest - tau_target, 1e-12)
            frac = min(max(frac, 0.0), 1.0)
            return float(cov_shrink) * frac

        def _solve_single_stage(
                stage_tau: float,
                z_init: torch.Tensor,
                stage_total_iters: int,
                is_final_stage: bool,
        ) -> torch.Tensor:
            stage_total_iters = max(int(stage_total_iters), 0)
            rho_stage = _stage_cov_shrink(stage_tau)
            if stage_total_iters <= 0:
                return z_init.clone()
            if is_final_stage:
                if stage_total_iters <= 2:
                    newton_iters = 1
                elif stage_total_iters <= 5:
                    newton_iters = 2
                else:
                    newton_iters = 3
            else:
                newton_iters = 1 if stage_total_iters >= 8 else 0
            newton_iters = min(stage_total_iters, newton_iters)
            subgrad_iters = max(0, stage_total_iters - newton_iters)

            G_init, _ = expected_degree_residual_from_z(
                C_solve, z_init, Q, tau=stage_tau, root=root, out_dtype=solve_dtype
            )
            best_z = z_init.clone()
            best_res = G_init.norm(dim=-1)
            z_stage = z_init.clone()

            # ── Phase 1: HK subgradient warmup for this continuation stage ──
            for t in range(subgrad_iters):
                G, info = expected_degree_residual_from_z(
                    C_solve, z_stage, Q, tau=stage_tau, root=root, out_dtype=solve_dtype
                )
                res = G.norm(dim=-1)

                improved = res < best_res
                best_res = torch.where(improved, res, best_res)
                best_z = torch.where(improved.unsqueeze(-1), z_stage, best_z)

                if torch.isfinite(res).all() and res.max().item() < lam_tol:
                    break

                d_excess = info["degree_nr"] - 2.0
                lam_nr = info["lambda_nr"]
                step = stage_tau * lam_step / (1.0 + t * 0.05)
                lam_new = lam_nr + step * d_excess

                z_try = lam_new @ Q
                if torch.isfinite(z_try).all():
                    z_stage = z_try.detach()
                else:
                    z_stage = best_z.clone()

            z_stage = best_z.clone()

            # ── Phase 2: terminal Newton polish for this continuation stage ──
            # Batched CG with VJP matvec: all B instances run in lockstep.
            # Since J = dG/dz is symmetric (Hessian of concave dual), we can
            # use VJP to compute J^T@v = J@v.  Each CG iteration costs one
            # batched backward pass — no explicit Jacobian ever materialised.
            effective_ridge = float(ift_ridge) + max(0.0, float(lm_damping)) / max(float(stage_tau), 0.01)
            cg_max_iter = min(q, 30)

            for _ in range(newton_iters):
                with torch.enable_grad():
                    z_var_n = z_stage.detach().requires_grad_(True)
                    G, info_newton = expected_degree_residual_from_z(
                        C_solve, z_var_n, Q, tau=stage_tau, root=root, out_dtype=solve_dtype
                    )
                    res = G.norm(dim=-1)
                    phi = -stage_tau * info_newton["logZ"] - 2.0 * info_newton["lambda_nr"].sum(dim=-1)

                    improved = res < best_res
                    best_res = torch.where(improved, res.detach(), best_res)
                    best_z = torch.where(improved.unsqueeze(-1), z_stage, best_z)

                    if torch.isfinite(res).all() and res.max().item() < lam_tol:
                        break

                    G_safe = torch.nan_to_num(G.detach(), nan=0.0, posinf=1e6, neginf=-1e6)

                    def _newton_matvec(v: torch.Tensor, _G=G, _z=z_var_n, _r=effective_ridge) -> torch.Tensor:
                        JTv = torch.autograd.grad(
                            _G, _z, grad_outputs=v, retain_graph=True, create_graph=False,
                        )[0]
                        JTv = torch.nan_to_num(JTv, nan=0.0, posinf=0.0, neginf=0.0)
                        return -JTv + _r * v  # (H + ridge I) @ v

                    delta, _ = batched_cg_solve(
                        _newton_matvec, G_safe,
                        max_iter=cg_max_iter, tol=1e-5, abs_tol=1e-8,
                    )
                delta = delta.detach()
                finite_delta = torch.isfinite(delta).all(dim=-1)

                delta_norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                delta_scale = torch.clamp(max_newton_step / delta_norm, max=1.0)
                delta = delta * delta_scale

                z_next = z_stage.clone()
                res_det = res.detach()
                phi_det = phi.detach()
                accepted = torch.zeros(B, device=z_stage.device, dtype=torch.bool)
                step_scales = (1.0, 0.5, 0.25)

                for step_scale in step_scales:
                    active = (~accepted) & finite_delta
                    if not bool(active.any()):
                        break

                    idx = active.nonzero(as_tuple=False).squeeze(-1)
                    z_try = z_stage[idx] + step_scale * delta[idx]
                    finite_try = torch.isfinite(z_try).all(dim=-1)
                    if not bool(finite_try.any()):
                        continue

                    idx = idx[finite_try]
                    z_try = z_try[finite_try]
                    G_try, info_try = expected_degree_residual_from_z(
                        C_solve[idx],
                        z_try,
                        Q,
                        tau=stage_tau,
                        root=root,
                        out_dtype=solve_dtype,
                    )
                    res_try = G_try.norm(dim=-1)
                    phi_try = -stage_tau * info_try["logZ"] - 2.0 * info_try["lambda_nr"].sum(dim=-1)
                    ok = (
                            torch.isfinite(res_try)
                            & torch.isfinite(phi_try)
                            & (phi_try >= phi_det[idx] - 1e-12)
                            & (res_try <= 1.05 * res_det[idx] + 1e-10)
                    )
                    if bool(ok.any()):
                        acc_idx = idx[ok]
                        z_next[acc_idx] = z_try[ok].detach()
                        accepted[acc_idx] = True

                still_bad = ~accepted
                if bool(still_bad.any()):
                    bad_idx = still_bad.nonzero(as_tuple=False).squeeze(-1)
                    z_try = z_stage[bad_idx] + 0.1 * stage_tau * G[bad_idx].detach()
                    finite_try = torch.isfinite(z_try).all(dim=-1)
                    if bool(finite_try.any()):
                        z_next[bad_idx[finite_try]] = z_try[finite_try].detach()
                    if bool((~finite_try).any()):
                        z_next[bad_idx[~finite_try]] = best_z[bad_idx[~finite_try]]

                z_stage = z_next

            G_final, _ = expected_degree_residual_from_z(
                C_solve, z_stage, Q, tau=stage_tau, root=root, out_dtype=solve_dtype
            )
            res_final = G_final.norm(dim=-1)
            use_best = res_final > best_res
            z_stage = torch.where(use_best.unsqueeze(-1), best_z, z_stage)
            return z_stage

        with torch.no_grad():
            if lambda_init_nr is None or lambda_init_nr.numel() == 0:
                z = torch.zeros(B, q, device=device, dtype=solve_dtype)
            else:
                if lambda_init_nr.shape != (B, m):
                    raise ValueError(
                        f"lambda_init_nr must have shape {(B, m)}, got {tuple(lambda_init_nr.shape)}"
                    )
                lam0 = lambda_init_nr.detach().to(solve_dtype)
                lam0 = lam0 - lam0.mean(dim=-1, keepdim=True)
                z = lam0 @ Q

            for stage_idx, stage_tau in enumerate(tau_path):
                z = _solve_single_stage(
                    stage_tau,
                    z,
                    stage_iters[stage_idx],
                    is_final_stage=(stage_idx == len(tau_path) - 1),
                )
                if stage_idx + 1 < len(tau_path):
                    next_tau = tau_path[stage_idx + 1]
                    z = z * (float(next_tau) / float(stage_tau))

            _, info = expected_degree_residual_from_z(
                C_solve, z, Q, tau=tau, root=root, out_dtype=solve_dtype
            )
            mu_star = info["mu"].detach().to(dtype)
            lambda_nr_star = info["lambda_nr"].detach().to(dtype)
            degree_star = info["degree"].detach().to(dtype)
            entropy_star = info["entropy"].detach().to(dtype)
            residual_star = info["G"].norm(dim=-1).detach().to(dtype)

        ctx.save_for_backward(C_theta.detach(), z.detach(), Q.detach(), residual_star.detach())
        ctx.tau = tau
        ctx.root = root
        ctx.ift_ridge = ift_ridge
        ctx.ift_backward_tol = ift_backward_tol
        return mu_star, lambda_nr_star, degree_star, entropy_star, residual_star

    @staticmethod
    def backward(ctx, grad_mu, grad_lambda_nr, grad_degree, grad_entropy, grad_residual):
        C_saved, z_saved, Q, residual_saved = ctx.saved_tensors
        tau = ctx.tau
        root = ctx.root
        ift_ridge = ctx.ift_ridge
        ift_backward_tol = ctx.ift_backward_tol

        def _as_output_grad(g: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor | None:
            if g is None:
                return None
            return g.to(ref.dtype)

        with torch.enable_grad():
            C_var = C_saved.detach().requires_grad_(True)
            z_var = z_saved.detach().requires_grad_(True)
            G, info = expected_degree_residual_from_z(
                C_var, z_var, Q, tau=tau, root=root, out_dtype=z_var.dtype
            )
            mu = info["mu"]
            lambda_nr = info["lambda_nr"]
            degree = info["degree"]
            entropy = info["entropy"]
            residual = info["G"].norm(dim=-1)

            outputs = []
            grad_outputs = []
            for out, g in (
                    (mu, _as_output_grad(grad_mu, mu)),
                    (lambda_nr, _as_output_grad(grad_lambda_nr, lambda_nr)),
                    (degree, _as_output_grad(grad_degree, degree)),
                    (entropy, _as_output_grad(grad_entropy, entropy)),
                    (residual, _as_output_grad(grad_residual, residual)),
            ):
                if g is not None:
                    outputs.append(out)
                    grad_outputs.append(g)

            if not outputs:
                return torch.zeros_like(
                    C_saved), None, None, None, None, None, None, None, None, None, None, None, None, None, None

            direct_C, dL_dz = torch.autograd.grad(
                outputs,
                (C_var, z_var),
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )

            B, q = z_var.shape
            finite_direct_C = torch.isfinite(direct_C).all(dim=(-2, -1))
            if q == 0:
                grad_C = torch.zeros_like(direct_C)
                if bool(finite_direct_C.any()):
                    idx = finite_direct_C.nonzero(as_tuple=False).squeeze(-1)
                    grad_C[idx] = direct_C[idx]
            else:
                # Important safety rule: only allow gradients from samples whose forward
                # equilibrium was actually solved to a trustworthy tolerance. Badly solved
                # samples should contribute ZERO gradient, not a direct-through-forward
                # gradient, because that is exactly the path that can inject huge or NaN
                # updates when the Laplacian system becomes ill-conditioned.
                grad_C = torch.zeros_like(direct_C)
                finite_dL_dz = torch.isfinite(dL_dz).all(dim=-1)
                trusted_mask = torch.isfinite(residual_saved) & (residual_saved <= ift_backward_tol)
                trusted_mask = trusted_mask & finite_dL_dz & finite_direct_C

                if bool(trusted_mask.any()):
                    idx = trusted_mask.nonzero(as_tuple=False).squeeze(-1)
                    C_good = C_var[idx]
                    z_good = z_var[idx]
                    dL_dz_good = dL_dz[idx]
                    grad_C[idx] = direct_C[idx]

                    finite_rhs = torch.isfinite(dL_dz_good).all(dim=-1)
                    solve_mask = finite_rhs

                    if bool(solve_mask.any()):
                        idx2 = idx[solve_mask]
                        C_solve = C_good[solve_mask]
                        rhs_sub = dL_dz_good[solve_mask]

                        # Batched CG adjoint solve: (H + ridge I) adj = rhs
                        # where H = -J is PSD.  The matvec uses VJP through
                        # the retained G computation graph — one batched
                        # backward per CG iteration, no explicit Jacobian.
                        z_solve_rg = z_good[solve_mask].detach().requires_grad_(True)
                        G_adj, _ = expected_degree_residual_from_z(
                            C_solve, z_solve_rg,
                            Q, tau=tau, root=root, out_dtype=z_solve_rg.dtype,
                        )
                        adj_ridge = float(ift_ridge)

                        def _adj_matvec(v: torch.Tensor) -> torch.Tensor:
                            JTv = torch.autograd.grad(
                                G_adj, z_solve_rg,
                                grad_outputs=v,
                                retain_graph=True,
                                create_graph=False,
                                allow_unused=False,
                            )[0]
                            JTv = torch.nan_to_num(JTv, nan=0.0, posinf=0.0, neginf=0.0)
                            return -JTv + adj_ridge * v

                        adj_sub, _ = batched_cg_solve(
                            _adj_matvec, rhs_sub,
                            max_iter=min(q, 15), tol=1e-5, abs_tol=1e-8,
                        )

                        if torch.isfinite(adj_sub).all():
                            implicit_C_sub = torch.autograd.grad(
                                G_adj,
                                C_solve,
                                grad_outputs=adj_sub,
                                retain_graph=False,
                                create_graph=False,
                                allow_unused=False,
                            )[0]
                            corrected_sub = direct_C[idx2] + implicit_C_sub
                            finite_corr = torch.isfinite(corrected_sub).all(dim=(-2, -1))
                            if bool(finite_corr.any()):
                                idx3 = idx2[finite_corr]
                                grad_C[idx3] = corrected_sub[finite_corr]

            grad_C = torch.nan_to_num(grad_C, nan=0.0, posinf=0.0, neginf=0.0)

        return grad_C, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def rooted_onetree_implicit_layer(
        C_theta: torch.Tensor,
        tau: float,
        lambda_init_nr: torch.Tensor | None = None,
        root: int = 0,
        lam_iters: int = 50,
        lam_tol: float = 1e-5,
        lam_step: float = 0.5,
        ift_ridge: float = 1e-4,
        ift_backward_tol: float = 1e-2,
        inner_homotopy: int = 1,
        inner_tau_start: float = 0.30,
        inner_tau_mid: float = 0.22,
        inner_final_frac: float = 0.75,
        cov_shrink: float = 0.0,
        lm_damping: float = 0.0,
):
    if lambda_init_nr is None:
        lambda_init_nr = C_theta.new_zeros(C_theta.shape[0], C_theta.shape[1] - 1)
    return RootedOneTreeImplicitLayer.apply(
        C_theta,
        lambda_init_nr,
        tau,
        root,
        lam_iters,
        lam_tol,
        lam_step,
        ift_ridge,
        ift_backward_tol,
        inner_homotopy,
        inner_tau_start,
        inner_tau_mid,
        inner_final_frac,
        cov_shrink,
        lm_damping,
    )


# ================================================================
# §6  Full model
# ================================================================


class TSPEntropicOneTreeModel(nn.Module):
    """Two-round shared-weight residual refinement model over the rooted 1-tree family.

    Round 0 sees only the original Euclidean geometry. Round 1 sees the same
    geometry plus a refinement-state tensor built from the round-0 equilibrium.
    The candidate graph remains fixed on the original instance, and the
    node-additive dual is still solved exactly inside each round. The second
    round is explicitly constrained to act as a residual correction, and the
    gate is now treated only as an optional ablation while the faithful stage-2
    learning signal comes from the true cost plus the explicit U_deg
    structural certificate.
    """

    def __init__(
            self,
            node_dim: int = 128,
            edge_dim: int = 32,
            num_gnn_layers: int = 3,
            beta: float = 1.0,
            tau: float = 0.05,
            prior_weight: float = 0.5,
            candidate_k: int = 16,
            non_candidate_penalty: float = 2.0,
            lam_iters: int = 50,
            lam_tol: float = 1e-5,
            lam_step: float = 0.5,
            ift_ridge: float = 1e-4,
            loss_mode: str = "cost",
            entropy_weight: float = 0.0,
            deg_penalty_weight: float = 10.0,
            resid_penalty_weight: float = 1.0,
            bern_penalty_weight: float = 0.0,
            logit_clamp: float = 5.0,
            root: int = 0,
            ift_backward_tol: float = 1e-2,
            inner_homotopy: int = 1,
            inner_tau_start: float = 0.30,
            inner_tau_mid: float = 0.22,
            inner_final_frac: float = 0.75,
            cov_shrink: float = 0.25,
            lm_damping: float = 0.05,
            detach_refine_state: int = 1,
            round2_use_struct_gate: int = 1,
            round2_gate_detach_features: int = 0,
            round2_gate_hidden_dim: int = 64,
            round2_struct_gate_floor: float = 0.35,
            round2_struct_gate_temp: float = 0.25,
            round2_struct_bonus: float = 0.35,
            stage2_struct_target: float = 0.0,
            stage2_struct_linear_weight: float = 1.0,
            stage2_struct_quad_weight: float = 0.25,
            stage2_struct_uncertainty_weight: float = 1.0,
            stage2_entropy_penalty_weight: float = 0.0,
            nontour_entropy_weight: float = 0.0,
            stage2_objective_mode: str = "cert_budget",
            stage2_bound_weight: float = 0.0,
            round0_loss_weight: float = 0.25,
            sharpen_beta: float = 0.0,
            cert_alpha: float = 0.0,
            var_tilt_weight: float = 0.0,
            stage2_coupled_steps: int = 0,
            stage2_coupled_damping: float = 0.65,
            edge_head_with_cost: bool = False,
            edge_hidden_mult: int = 3,
            gradient_checkpoint: bool = False,
    ):
        super().__init__()
        self.encoder = NeighborhoodEdgePriorEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            num_layers=num_gnn_layers,
            residual_state_dim=3,
            edge_head_with_cost=bool(edge_head_with_cost),
            edge_hidden_mult=int(edge_hidden_mult),
            gradient_checkpoint=bool(gradient_checkpoint),
        )
        self.beta = beta
        self.tau = tau
        self.prior_weight = prior_weight
        self.candidate_k = candidate_k
        self.non_candidate_penalty = non_candidate_penalty
        self.lam_iters = lam_iters
        self.lam_tol = lam_tol
        self.lam_step = lam_step
        self.ift_ridge = ift_ridge
        self.loss_mode = loss_mode
        self.entropy_weight = entropy_weight
        self.deg_penalty_weight = deg_penalty_weight
        self.resid_penalty_weight = resid_penalty_weight
        self.bern_penalty_weight = bern_penalty_weight
        self.logit_clamp = logit_clamp
        self.root = root
        self.ift_backward_tol = ift_backward_tol
        self.inner_homotopy = int(inner_homotopy)
        self.inner_tau_start = inner_tau_start
        self.inner_tau_mid = inner_tau_mid
        self.inner_final_frac = inner_final_frac
        self.cov_shrink = cov_shrink
        self.lm_damping = lm_damping
        self.edge_projection_ridge = 1e-6

        # Fixed internal refinement design choices.
        self.num_refine_rounds = 2
        self.round0_loss_weight = float(round0_loss_weight)
        self.sharpen_beta = float(sharpen_beta)
        # v16+: Certificate refinement strength. Scales the GNN correction g(e)
        # added to the (μ₀−0.5) baseline certificate. 0 disables refinement
        # (pure baseline tilt). Small values (0.05-0.2) keep the tilt within
        # a trust region of the Fisher-gradient direction.
        self.cert_alpha = float(cert_alpha)
        # v16+: Variance-weighted tilt. Weights the (μ - 0.5) tilt direction by
        # the diagonal approximation of endpoint degree variance V_i = Σ_j μ_ij(1-μ_ij).
        # 0 = no weighting (pure Fisher baseline).
        # 1 = full weighting (high-variance vertices get stronger tilt).
        # Uses distributional information from 1-tree Gibbs structure.
        self.var_tilt_weight = float(var_tilt_weight)
        self.stage2_coupled_steps = max(0, int(stage2_coupled_steps))
        self.stage2_coupled_damping = min(max(float(stage2_coupled_damping), 0.0), 1.0)
        self.round1_lam_frac = 0.4
        self.refine_state_scale = 2.0
        self.detach_refine_state = bool(detach_refine_state)
        self.round2_use_struct_gate = bool(round2_use_struct_gate)
        self.round2_gate_detach_features = bool(round2_gate_detach_features)
        gate_hidden = max(16, int(round2_gate_hidden_dim))
        self.round2_gate_hidden_dim = gate_hidden
        self.round2_residual_scale = 0.8
        self.round2_relative_cap = 1.0
        self.round2_struct_gate_floor = float(round2_struct_gate_floor)
        self.round2_struct_gate_temp = float(round2_struct_gate_temp)
        self.round2_struct_bonus = float(round2_struct_bonus)
        self.stage2_struct_target = float(stage2_struct_target)
        self.stage2_struct_linear_weight = float(stage2_struct_linear_weight)
        self.stage2_struct_quad_weight = float(stage2_struct_quad_weight)
        self.stage2_struct_uncertainty_weight = max(1.0, float(stage2_struct_uncertainty_weight))
        self.stage2_entropy_penalty_weight = float(stage2_entropy_penalty_weight)
        self.nontour_entropy_weight = float(nontour_entropy_weight)
        mode = str(stage2_objective_mode).strip().lower()
        if mode in {"hinge", "hinge_penalty", "penalty", "cert_budget", "repair_budget"}:
            mode = "cert_budget"
        elif mode in {"repair_bound", "tour_bound", "bound"}:
            mode = "repair_bound"
        else:
            raise ValueError(
                f"Unknown stage2_objective_mode: {stage2_objective_mode}. "
                "Use 'cert_budget' (default) or 'repair_bound'."
            )
        self.stage2_objective_mode = mode
        self.stage2_bound_weight = float(stage2_bound_weight)

        self.round2_gate_feat_dim = 9
        if self.round2_use_struct_gate:
            self.round2_gate_head = nn.Sequential(
                nn.Linear(self.round2_gate_feat_dim, gate_hidden),
                nn.SiLU(),
                nn.Linear(gate_hidden, gate_hidden),
                nn.SiLU(),
                nn.Linear(gate_hidden, 1),
            )
            nn.init.zeros_(self.round2_gate_head[-1].bias)
        else:
            self.round2_gate_head = None

    def _apply_logit_clamp(self, logits: torch.Tensor) -> torch.Tensor:
        if self.logit_clamp > 0.0:
            # return torch.clamp(logits, -self.logit_clamp,self.logit_clamp)
            return self.logit_clamp * torch.tanh(logits / self.logit_clamp)
        return logits

    def _round_lam_iters(self, round_idx: int) -> int:
        if round_idx <= 0:
            return max(8, int(round(self.round1_lam_frac * float(self.lam_iters))))
        return max(8, int(self.lam_iters))

    def _stage2_coupled_lam_schedule(self) -> list[int]:
        steps = max(1, int(self.stage2_coupled_steps))
        total = max(8, int(self._round_lam_iters(1)))
        alloc = split_lam_iters_across_stages(total, steps, inner_final_frac=self.inner_final_frac)
        return [max(4, int(a)) for a in alloc]

    def _build_stage2_certificate(
            self,
            prev_mu: torch.Tensor,
            enc_aux: dict[str, torch.Tensor],
            cand_mask: torch.Tensor,
            full_f: torch.Tensor,
    ) -> torch.Tensor:
        """Stage-2 certificate field used by the sharpen tilt."""
        certificate = prev_mu - 0.5

        if self.var_tilt_weight > 0:
            with torch.no_grad():
                mu_clamped = prev_mu.clamp(min=0.0, max=1.0)
                edge_var = mu_clamped * (1.0 - mu_clamped) * full_f
                node_var = edge_var.sum(dim=-1)
                node_var_sum = (node_var.unsqueeze(-1) + node_var.unsqueeze(-2)).clamp_min(1e-4)
                w_edge = torch.sqrt(node_var_sum) * full_f
                w_mean = (w_edge * cand_mask).sum(dim=(-2, -1), keepdim=True) / (
                    cand_mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
                )
                w_hat = w_edge / w_mean.clamp_min(1e-4)
                weight = 1.0 + self.var_tilt_weight * (w_hat - 1.0)
            certificate = certificate * weight

        if self.cert_alpha > 0 and enc_aux.get("cert_correction") is not None:
            certificate = certificate + self.cert_alpha * enc_aux["cert_correction"]

        certificate = 0.5 * (certificate + certificate.transpose(-2, -1))
        certificate = certificate * full_f
        return certificate

    def _apply_stage2_sharpen_tilt(
            self,
            C_base: torch.Tensor,
            prev_mu: torch.Tensor,
            enc_aux: dict[str, torch.Tensor],
            cand_mask: torch.Tensor,
            full_f: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        certificate = self._build_stage2_certificate(prev_mu, enc_aux, cand_mask, full_f)
        tilt_raw = self.sharpen_beta * self.tau * certificate
        tilt_raw = 0.5 * (tilt_raw + tilt_raw.transpose(-2, -1))
        tilt_raw = tilt_raw * full_f
        tilt_proj, tilt_proj_info = project_edge_residual_candidate_weighted(
            tilt_raw,
            cand_mask,
            ridge=self.edge_projection_ridge,
        )
        C_eff = C_base - tilt_proj
        C_eff = 0.5 * (C_eff + C_eff.transpose(-2, -1))
        C_eff = C_eff * full_f
        tilt_info = {
            "certificate_rms": self._edge_rms(certificate, cand_mask).mean(),
            "tilt_proj_rms": self._edge_rms(tilt_proj, cand_mask).mean(),
            "tilt_proj_fallback_frac": tilt_proj_info.get("fallback_frac",
                                                          torch.zeros((), device=C_base.device, dtype=C_base.dtype)),
        }
        return C_eff, tilt_info

    def _edge_rms(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.to(dtype=x.dtype)
        denom = mask_f.sum(dim=(-2, -1)).clamp_min(1.0)
        return torch.sqrt(((x * mask_f).pow(2).sum(dim=(-2, -1)) / denom).clamp_min(1e-12))

    def _edge_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.to(dtype=x.dtype)
        denom = mask_f.sum(dim=(-2, -1)).clamp_min(1.0)
        return (x * mask_f).sum(dim=(-2, -1)) / denom

    def _build_round2_gate_features(
            self,
            stage1_state: torch.Tensor,
            stage1_record: dict[str, torch.Tensor],
            bounded_logits: torch.Tensor,
            active_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, n, _ = bounded_logits.shape
        dtype = bounded_logits.dtype
        device = bounded_logits.device

        if self.round2_gate_detach_features:
            stage1_state = stage1_state.detach()
            degree = stage1_record["degree"].detach()
            mu = stage1_record["mu"].detach()
            residual = stage1_record["residual"].detach()
        else:
            degree = stage1_record["degree"]
            mu = stage1_record["mu"]
            residual = stage1_record["residual"]

        stage1_state = stage1_state.to(dtype=dtype)
        degree = degree.to(dtype=dtype)
        mu = mu.to(dtype=dtype)
        residual = residual.to(dtype=dtype)

        deg_err = (degree - 2.0).abs()
        deg_sum = torch.tanh((deg_err.unsqueeze(-1) + deg_err.unsqueeze(-2)) / 2.0)
        deg_diff = torch.tanh((deg_err.unsqueeze(-1) - deg_err.unsqueeze(-2)).abs() / 2.0)

        full_mask = ~torch.eye(n, device=device, dtype=torch.bool).unsqueeze(0)
        mu_prob = mu.clamp(min=0.0, max=1.0)
        edge_var = mu_prob * (1.0 - mu_prob) * full_mask.to(dtype=dtype)
        node_unc = edge_var.sum(dim=-1)
        unc_scale = node_unc[:, 1:].mean(dim=-1, keepdim=True).clamp_min(1e-6)
        unc_pair_scale = 2.0 * unc_scale.unsqueeze(-1)
        unc_sum = torch.tanh((node_unc.unsqueeze(-1) + node_unc.unsqueeze(-2)) / unc_pair_scale)
        unc_diff = torch.tanh((node_unc.unsqueeze(-1) - node_unc.unsqueeze(-2)).abs() / unc_pair_scale)

        cur_scale = self._edge_rms(bounded_logits.detach(), active_mask).view(B, 1, 1).clamp_min(1e-6)
        logit_feat = torch.tanh(bounded_logits / cur_scale)
        residual_feat = torch.tanh(residual.view(B, 1, 1).expand(-1, n, n))

        feat = torch.cat(
            [
                stage1_state,
                logit_feat.unsqueeze(-1),
                deg_sum.unsqueeze(-1),
                deg_diff.unsqueeze(-1),
                unc_sum.unsqueeze(-1),
                unc_diff.unsqueeze(-1),
                residual_feat.unsqueeze(-1),
            ],
            dim=-1,
        )
        feat = 0.5 * (feat + feat.transpose(1, 2))
        feat = feat * full_mask.unsqueeze(-1).to(dtype=dtype)
        return feat

    def _compute_round2_learnable_gate(
            self,
            stage1_state: torch.Tensor,
            stage1_record: dict[str, torch.Tensor],
            bounded_logits: torch.Tensor,
            active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, n, _ = bounded_logits.shape
        dtype = bounded_logits.dtype
        device = bounded_logits.device
        if (not self.round2_use_struct_gate) or (self.round2_gate_head is None) or n <= 2:
            ones = torch.ones_like(bounded_logits)
            zeros = torch.zeros((), device=device, dtype=dtype)
            return ones, {
                "gate_raw": bounded_logits.new_tensor(1.0),
                "gate_scale": bounded_logits.new_tensor(1.0),
                "gate_logit": zeros,
                "gate_residual": zeros,
            }

        gate_feat = self._build_round2_gate_features(stage1_state, stage1_record, bounded_logits, active_mask)
        gate_logits = self.round2_gate_head(gate_feat).squeeze(-1)
        gate_logits = 0.5 * (gate_logits + gate_logits.transpose(-2, -1))
        full_mask = ~torch.eye(n, device=device, dtype=torch.bool).unsqueeze(0)
        gate_logits = gate_logits * full_mask.to(dtype=dtype)

        temp = max(float(self.round2_struct_gate_temp), 1e-8)
        gate_raw = torch.sigmoid(gate_logits / temp)
        lower = min(max(float(self.round2_struct_gate_floor), 0.0), 1.0)
        upper = max(lower, 1.0 + max(0.0, float(self.round2_struct_bonus)))
        gate_scale = lower + (upper - lower) * gate_raw
        gate_scale = gate_scale * active_mask.to(dtype=dtype)

        residual_ref = stage1_record["residual"].detach() if self.round2_gate_detach_features else stage1_record[
            "residual"]
        info = {
            "gate_raw": self._edge_mean(gate_raw, active_mask).mean().detach(),
            "gate_scale": self._edge_mean(gate_scale, active_mask).mean().detach(),
            "gate_logit": self._edge_mean(gate_logits.abs(), active_mask).mean().detach(),
            "gate_residual": residual_ref.mean().detach().to(dtype),
        }
        return gate_scale, info

    def _apply_round_update(
            self,
            round_idx: int,
            bounded_logits: torch.Tensor,
            cumulative_logits: torch.Tensor,
            active_mask: torch.Tensor,
            learned_gate: torch.Tensor | None = None,
            gate_info: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        update = bounded_logits
        info: dict[str, torch.Tensor] = {}
        if round_idx > 0:
            base_scale = bounded_logits.new_tensor(float(self.round2_residual_scale))
            prev_rms = self._edge_rms(cumulative_logits.detach(), active_mask)
            cur_rms = self._edge_rms(bounded_logits.detach(), active_mask)
            ratio_cap = prev_rms * float(self.round2_relative_cap)
            safe = ratio_cap / cur_rms.clamp_min(1e-8)
            rel_scale = torch.clamp(safe, max=1.0)
            free_scale = base_scale * rel_scale.view(-1, 1, 1)
            free_scale_full = free_scale.expand_as(bounded_logits)

            if learned_gate is None:
                learned_gate = torch.ones_like(bounded_logits)
            else:
                learned_gate = learned_gate.to(dtype=bounded_logits.dtype)

            total_scale = free_scale_full * learned_gate
            update = bounded_logits * total_scale
            info["round2_update_rms"] = self._edge_rms(update.detach(), active_mask).mean()
            info["round2_raw_rms"] = cur_rms.mean()
            info["round2_scale"] = self._edge_mean(total_scale, active_mask).mean().detach()
            info["round2_free_scale"] = free_scale.mean().detach()
            if gate_info is not None:
                info["round2_gate"] = gate_info["gate_raw"]
                info["round2_gate_mult"] = gate_info["gate_scale"]
                info["round2_gate_logit"] = gate_info["gate_logit"]
                info["round2_gate_residual"] = gate_info["gate_residual"]
            else:
                zero = bounded_logits.new_tensor(0.0)
                info["round2_gate"] = bounded_logits.new_tensor(1.0)
                info["round2_gate_mult"] = bounded_logits.new_tensor(1.0)
                info["round2_gate_logit"] = zero
                info["round2_gate_residual"] = zero
        else:
            zero = bounded_logits.new_tensor(0.0)
            info["round2_update_rms"] = zero
            info["round2_raw_rms"] = zero
            info["round2_scale"] = bounded_logits.new_tensor(1.0)
            info["round2_free_scale"] = bounded_logits.new_tensor(1.0)
            info["round2_gate"] = bounded_logits.new_tensor(1.0)
            info["round2_gate_mult"] = bounded_logits.new_tensor(1.0)
            info["round2_gate_logit"] = zero
            info["round2_gate_residual"] = zero
        return update, info

    def _build_refine_state(
            self,
            base_cost: torch.Tensor,
            C_theta: torch.Tensor,
            lambda_nr: torch.Tensor,
            mu: torch.Tensor,
            full_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, n, _ = base_cost.shape
        dtype = base_cost.dtype
        device = base_cost.device
        mask_f = full_mask.to(dtype=dtype)
        denom = mask_f.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        scale = (base_cost.abs() * mask_f).sum(dim=(-2, -1), keepdim=True) / denom
        scale = scale.clamp_min(1e-6)

        lambda_full = torch.zeros(B, n, device=device, dtype=lambda_nr.dtype)
        lambda_full[:, 1:] = lambda_nr
        node_shift = lambda_full.unsqueeze(-1) + lambda_full.unsqueeze(-2)
        total_shift = (C_theta + node_shift) - base_cost

        total_feat = torch.tanh(total_shift / (self.refine_state_scale * scale))
        node_feat = torch.tanh(node_shift / (self.refine_state_scale * scale))
        # mu_feat = mu
        mu_feat = torch.zeros_like(mu)
        state = torch.stack([total_feat, node_feat, mu_feat], dim=-1)
        state = 0.5 * (state + state.transpose(1, 2))
        state = state * full_mask.unsqueeze(-1).to(dtype=dtype)
        return state

    def _compute_stage2_repair_bound_weight(self, dist_matrix: torch.Tensor) -> torch.Tensor:
        """Per-instance coefficient for the rigorous repair-cost upper bound.

        The corrected metric-repair theorem gives
            E[cost(repaired tour)] <= E[cost(U)] + D_max * U_proxy,
        where D_max = max_ij D_ij and U_proxy is the rooted 1-tree proxy certificate.
        If stage2_bound_weight > 0 we use that explicit coefficient; otherwise we use
        the instancewise metric constant D_max.
        """
        B, _, _ = dist_matrix.shape
        dtype = dist_matrix.dtype
        device = dist_matrix.device
        if float(self.stage2_bound_weight) > 0.0:
            return torch.full((B,), float(self.stage2_bound_weight), device=device, dtype=dtype)
        return dist_matrix.amax(dim=(-2, -1)).detach()

    def _compute_stage2_structural_certificate(
            self,
            degree: torch.Tensor,
            mu: torch.Tensor,
            full_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rigorous rooted 1-tree proxy certificate and its normalized counterpart.

        The exact combinatorial certificate is
            U_exact = 0.5 * sum_{i != root} (E[d_i]-2)^2
                    + 0.5 * sum_{i != root} Var(d_i),
        which upper-bounds the non-tour mass. We use the computable proxy
            U_proxy = 0.5 * sum_{i != root} (E[d_i]-2)^2
                    + 0.5 * sum_{i != root} sum_j Var(x_{ij}),
        where Var(x_{ij}) = mu_ij (1 - mu_ij). Under negative association of the
        spanning-tree factor, U_proxy >= U_exact.

        We return both the exact-scale sum certificate U_proxy and a normalized
        certificate U_bar = U_proxy / (n-1). The normalized form is the right object
        for a constrained stage-2 budget penalty because it removes the automatic O(n)
        scaling of the sum certificate while preserving an equivalent constraint.
        """
        m = max(int(degree.shape[1] - 1), 1)
        mean_defect_nodes = (degree[:, 1:] - 2.0).pow(2)
        defect_cert = 0.5 * mean_defect_nodes.sum(dim=-1)

        mu_prob = mu.clamp(min=0.0, max=1.0)
        edge_var = mu_prob * (1.0 - mu_prob) * full_mask.to(dtype=mu.dtype)
        node_uncert = edge_var.sum(dim=-1)[:, 1:]
        uncert_cert = 0.5 * node_uncert.sum(dim=-1)

        eta = float(self.stage2_struct_uncertainty_weight)
        struct_cert = defect_cert + eta * uncert_cert
        struct_cert_norm = struct_cert / float(m)
        return (
            struct_cert,
            struct_cert_norm,
            defect_cert,
            uncert_cert,
            mean_defect_nodes.mean(dim=-1),
            node_uncert.mean(dim=-1),
        )

    def _compute_nontour_edge_entropy(
            self,
            mu: torch.Tensor,
            full_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Smooth integrality penalty via Bernoulli variance mu*(1-mu).

        Zero at mu=0 and mu=1 (integral), maximum 0.25 at mu=0.5 (maximally
        fractional). Gradient = 1-2*mu: pushes mu>0.5 toward 1, mu<0.5 toward 0.
        Bounded (max |grad|=1), concave but well-behaved.

        Combined with degree-2 from the solver, this forces concentration on
        exactly 2 edges per vertex.

        Returns per-instance scalar (B,).
        """
        mu_c = mu.clamp(0.0, 1.0)
        edge_var = mu_c * (1.0 - mu_c) * full_mask.to(dtype=mu.dtype)
        return edge_var.sum(dim=(-2, -1)) / 2.0

    def _round_loss(
            self,
            round_idx: int,
            cost: torch.Tensor,
            entropy: torch.Tensor,
            degree: torch.Tensor,
            residual: torch.Tensor,
            mu: torch.Tensor,
            full_mask: torch.Tensor,
            dist_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        (
            struct_cert,
            struct_cert_norm,
            struct_defect_cert,
            struct_uncert_cert,
            struct_mean_defect,
            struct_mean_uncert,
        ) = self._compute_stage2_structural_certificate(degree, mu, full_mask)
        resid_penalty = residual.pow(2)
        repair_bound_weight = self._compute_stage2_repair_bound_weight(dist_matrix)
        repair_upper_bound = cost + repair_bound_weight * struct_cert

        # Non-tour entropy excess: penalizes mass on 3rd+ edges per vertex.
        # Applied only in round 2 — round 1 should explore freely.
        non_tour_H = self._compute_nontour_edge_entropy(mu, full_mask)
        if round_idx > 0:
            nontour_penalty = float(self.nontour_entropy_weight) * self.tau * non_tour_H
        else:
            nontour_penalty = torch.zeros_like(cost)

        if round_idx <= 0:
            struct_excess = torch.zeros_like(struct_cert_norm)
            struct_penalty = torch.zeros_like(struct_cert_norm)
            if self.loss_mode == "cost":
                loss = cost + self.resid_penalty_weight * resid_penalty
            elif self.loss_mode == "cost_entropy":
                loss = (cost
                        - self.entropy_weight * self.tau * entropy
                        + self.resid_penalty_weight * resid_penalty)
            else:
                raise ValueError(f"Unknown loss_mode: {self.loss_mode}")
        else:
            if self.stage2_objective_mode == "repair_bound":
                struct_excess = struct_cert_norm
                struct_penalty = repair_bound_weight * struct_cert
                loss = (
                        cost
                        + struct_penalty
                        + self.resid_penalty_weight * resid_penalty
                        + nontour_penalty
                )
            elif self.stage2_objective_mode == "cert_budget":
                struct_excess = torch.relu(struct_cert_norm - float(self.stage2_struct_target))
                struct_penalty = (
                        float(self.stage2_struct_linear_weight) * struct_excess
                        + 0.5 * float(self.stage2_struct_quad_weight) * struct_excess.pow(2)
                )
                loss = (
                        cost
                        + struct_penalty
                        + self.resid_penalty_weight * resid_penalty
                        + nontour_penalty
                )
            else:
                raise ValueError(
                    f"Unknown stage2_objective_mode: {self.stage2_objective_mode}. "
                    "Use 'cert_budget' or 'repair_bound'."
                )

        info = {
            "loss": loss,
            "resid_penalty": resid_penalty,
            "struct_cert": struct_cert,
            "struct_cert_norm": struct_cert_norm,
            "struct_defect_cert": struct_defect_cert,
            "struct_uncert_cert": struct_uncert_cert,
            "struct_mean_defect": struct_mean_defect,
            "struct_mean_uncert": struct_mean_uncert,
            "struct_excess": non_tour_H,
            "struct_penalty": nontour_penalty,
            "repair_bound_weight": repair_bound_weight,
            "repair_upper_bound": repair_upper_bound,
        }
        return loss, info

    def forward(self, coords: torch.Tensor, dist_matrix: torch.Tensor, return_decode_aux: bool = False):
        B, n, _ = coords.shape
        device = coords.device
        full_mask = ~torch.eye(n, device=device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
        full_f = full_mask.to(dtype=dist_matrix.dtype)
        cand_mask = build_candidate_mask(dist_matrix, self.candidate_k)
        non_cand = (full_mask & ~cand_mask).to(dtype=dist_matrix.dtype)
        base_cost = self.beta * dist_matrix + self.non_candidate_penalty * non_cand
        base_cost = 0.5 * (base_cost + base_cost.transpose(-2, -1))
        base_cost = base_cost * full_f

        residual_state = torch.zeros(B, n, n, 3, device=device, dtype=dist_matrix.dtype)
        cumulative_logits = torch.zeros_like(dist_matrix)
        lambda_init_nr = None
        round_records = []

        for round_idx in range(self.num_refine_rounds):
            round_state = residual_state.detach() if (round_idx > 0 and self.detach_refine_state) else residual_state
            raw_logits, enc_aux = self.encoder(coords, dist_matrix, cand_mask,
                                               residual_state=round_state, round_idx=round_idx)
            raw_logits = 0.5 * (raw_logits + raw_logits.transpose(-2, -1))
            raw_logits = raw_logits * full_f

            bounded_logits, proj_aux = project_edge_residual_candidate_weighted(
                raw_logits,
                cand_mask,
                ridge=self.edge_projection_ridge,
            )

            bounded_logits = self._apply_logit_clamp(bounded_logits)

            # Warm-start λ from the projection's removed node-additive component.
            if round_idx == 0 and lambda_init_nr is None:
                node_add = proj_aux.get("node_additive")
                if node_add is not None and node_add.shape[-1] >= 2:
                    lambda_init_nr = (-self.prior_weight * node_add[:, 1:]).detach()

            # v14: No learnable gate. Round 2 uses integrality tilt instead.
            round2_gate = None
            round2_gate_info = None
            round_update, update_info = self._apply_round_update(
                round_idx,
                bounded_logits,
                cumulative_logits,
                cand_mask,
                learned_gate=round2_gate,
                gate_info=round2_gate_info,
            )
            cumulative_logits = cumulative_logits + round_update
            C_theta_base = base_cost - self.prior_weight * cumulative_logits
            C_theta_base = 0.5 * (C_theta_base + C_theta_base.transpose(-2, -1))
            C_theta_base = C_theta_base * full_f

            stage2_coupled_info = None
            if round_idx > 0 and self.sharpen_beta > 0 and round_records:
                if self.stage2_coupled_steps > 0:
                    C_theta = C_theta_base
                    mu_ref = round_records[-1]["mu"]
                    lambda_loop_init = lambda_init_nr
                    lam_schedule = self._stage2_coupled_lam_schedule()
                    mu_prev_iter = mu_ref
                    coupled_delta_mu = None
                    # v19 Picard diagnostics: track per-step delta trajectory to
                    # measure effective contraction of the outer fixed-point loop.
                    delta_mu_trajectory: list[torch.Tensor] = []
                    delta_mu_max_trajectory: list[torch.Tensor] = []
                    tilt_info = {
                        "certificate_rms": C_theta.new_zeros(()),
                        "tilt_proj_rms": C_theta.new_zeros(()),
                        "tilt_proj_fallback_frac": C_theta.new_zeros(()),
                    }
                    for coupled_step, lam_iters_step in enumerate(lam_schedule):
                        C_target, tilt_info = self._apply_stage2_sharpen_tilt(
                            C_theta_base,
                            mu_ref,
                            enc_aux,
                            cand_mask,
                            full_f,
                        )
                        if coupled_step == 0:
                            C_theta = C_target
                        else:
                            rho = self.stage2_coupled_damping
                            C_theta = (1.0 - rho) * C_theta + rho * C_target
                            C_theta = 0.5 * (C_theta + C_theta.transpose(-2, -1))
                            C_theta = C_theta * full_f

                        mu, lambda_nr, degree, entropy, residual = rooted_onetree_implicit_layer(
                            C_theta,
                            self.tau,
                            lambda_init_nr=lambda_loop_init,
                            root=self.root,
                            lam_iters=lam_iters_step,
                            lam_tol=self.lam_tol,
                            lam_step=self.lam_step,
                            ift_ridge=self.ift_ridge,
                            ift_backward_tol=self.ift_backward_tol,
                            inner_homotopy=self.inner_homotopy,
                            inner_tau_start=self.inner_tau_start,
                            inner_tau_mid=self.inner_tau_mid,
                            inner_final_frac=self.inner_final_frac,
                            cov_shrink=self.cov_shrink,
                            lm_damping=self.lm_damping,
                        )
                        diff_abs = (mu - mu_prev_iter).abs()
                        coupled_delta_mu = diff_abs.mean()
                        coupled_delta_mu_max = diff_abs.amax(dim=(-2, -1)).mean()
                        delta_mu_trajectory.append(coupled_delta_mu.detach())
                        delta_mu_max_trajectory.append(coupled_delta_mu_max.detach())
                        mu_prev_iter = mu
                        mu_ref = mu
                        lambda_loop_init = lambda_nr.detach()

                    lambda_init_nr = lambda_loop_init
                    # Contraction statistics: geometric mean of consecutive Δμ ratios.
                    # For a linearly-contractive Picard map, Δμ_k ≈ ρ^k Δμ_0, so
                    # the geometric mean of (Δμ_k / Δμ_{k-1}) approximates the
                    # effective spectral radius. Safe-guarded for zero denominators.
                    if len(delta_mu_trajectory) >= 2:
                        dmu_first = delta_mu_trajectory[0]
                        dmu_last = delta_mu_trajectory[-1]
                        ratios: list[torch.Tensor] = []
                        for k in range(1, len(delta_mu_trajectory)):
                            denom = delta_mu_trajectory[k - 1].clamp_min(1e-12)
                            ratios.append(delta_mu_trajectory[k] / denom)
                        dmu_contract = torch.stack(ratios).mean()
                    elif len(delta_mu_trajectory) == 1:
                        dmu_first = delta_mu_trajectory[0]
                        dmu_last = dmu_first
                        dmu_contract = C_theta.new_ones(())
                    else:
                        dmu_first = C_theta.new_zeros(())
                        dmu_last = C_theta.new_zeros(())
                        dmu_contract = C_theta.new_ones(())
                    dmu_max_last = (
                        delta_mu_max_trajectory[-1]
                        if delta_mu_max_trajectory
                        else C_theta.new_zeros(())
                    )
                    stage2_coupled_info = {
                        "steps": C_theta.new_tensor(float(len(lam_schedule))),
                        "delta_mu": coupled_delta_mu if coupled_delta_mu is not None else C_theta.new_zeros(()),
                        "delta_mu_first": dmu_first,
                        "delta_mu_last": dmu_last,
                        "delta_mu_max_last": dmu_max_last,
                        "delta_mu_contract": dmu_contract,
                        "certificate_rms": tilt_info["certificate_rms"],
                        "tilt_proj_rms": tilt_info["tilt_proj_rms"],
                        "tilt_proj_fallback_frac": tilt_info["tilt_proj_fallback_frac"],
                    }
                else:
                    C_theta, tilt_info = self._apply_stage2_sharpen_tilt(
                        C_theta_base,
                        round_records[-1]["mu"],
                        enc_aux,
                        cand_mask,
                        full_f,
                    )
                    stage2_coupled_info = {
                        "steps": C_theta.new_tensor(1.0),
                        "delta_mu": C_theta.new_zeros(()),
                        "delta_mu_first": C_theta.new_zeros(()),
                        "delta_mu_last": C_theta.new_zeros(()),
                        "delta_mu_max_last": C_theta.new_zeros(()),
                        "delta_mu_contract": C_theta.new_ones(()),
                        "certificate_rms": tilt_info["certificate_rms"],
                        "tilt_proj_rms": tilt_info["tilt_proj_rms"],
                        "tilt_proj_fallback_frac": tilt_info["tilt_proj_fallback_frac"],
                    }
                    mu, lambda_nr, degree, entropy, residual = rooted_onetree_implicit_layer(
                        C_theta,
                        self.tau,
                        lambda_init_nr=lambda_init_nr,
                        root=self.root,
                        lam_iters=self._round_lam_iters(round_idx),
                        lam_tol=self.lam_tol,
                        lam_step=self.lam_step,
                        ift_ridge=self.ift_ridge,
                        ift_backward_tol=self.ift_backward_tol,
                        inner_homotopy=self.inner_homotopy,
                        inner_tau_start=self.inner_tau_start,
                        inner_tau_mid=self.inner_tau_mid,
                        inner_final_frac=self.inner_final_frac,
                        cov_shrink=self.cov_shrink,
                        lm_damping=self.lm_damping,
                    )
            else:
                C_theta = C_theta_base
                mu, lambda_nr, degree, entropy, residual = rooted_onetree_implicit_layer(
                    C_theta,
                    self.tau,
                    lambda_init_nr=lambda_init_nr,
                    root=self.root,
                    lam_iters=self._round_lam_iters(round_idx),
                    lam_tol=self.lam_tol,
                    lam_step=self.lam_step,
                    ift_ridge=self.ift_ridge,
                    ift_backward_tol=self.ift_backward_tol,
                    inner_homotopy=self.inner_homotopy,
                    inner_tau_start=self.inner_tau_start,
                    inner_tau_mid=self.inner_tau_mid,
                    inner_final_frac=self.inner_final_frac,
                    cov_shrink=self.cov_shrink,
                    lm_damping=self.lm_damping,
                )
            cost = 0.5 * (dist_matrix * mu).sum(dim=(-2, -1))
            loss, loss_info = self._round_loss(
                round_idx, cost, entropy, degree, residual, mu, full_mask, dist_matrix
            )
            lambda_full = torch.zeros(B, n, device=device, dtype=lambda_nr.dtype)
            lambda_full[:, 1:] = lambda_nr
            C_mod = C_theta + lambda_full.unsqueeze(-1) + lambda_full.unsqueeze(-2)
            C_mod = 0.5 * (C_mod + C_mod.transpose(-2, -1))

            round_records.append({
                "mu": mu,
                "lambda_nr": lambda_nr,
                "degree": degree,
                "entropy": entropy,
                "residual": residual,
                "cost": cost,
                "loss": loss,
                "resid_penalty": loss_info["resid_penalty"],
                "struct_cert": loss_info["struct_cert"],
                "struct_cert_norm": loss_info["struct_cert_norm"],
                "struct_defect_cert": loss_info["struct_defect_cert"],
                "struct_uncert_cert": loss_info["struct_uncert_cert"],
                "struct_mean_defect": loss_info["struct_mean_defect"],
                "struct_mean_uncert": loss_info["struct_mean_uncert"],
                "struct_excess": loss_info["struct_excess"],
                "struct_penalty": loss_info["struct_penalty"],
                "repair_bound_weight": loss_info["repair_bound_weight"],
                "repair_upper_bound": loss_info["repair_upper_bound"],
                "C_theta": C_theta,
                "C_mod": C_mod,
                "proj_aux": proj_aux,
                "bounded_logits": bounded_logits,
                "round_update": round_update,
                "update_info": update_info,
                "stage2_coupled_info": stage2_coupled_info,
            })
            lambda_init_nr = lambda_nr.detach()
            residual_state = self._build_refine_state(base_cost, C_theta, lambda_nr, mu, full_mask)

        final_rec = round_records[-1]
        aux_rec = round_records[0]
        # v14: single final loss — round 0 gets gradient through the tilt and
        # residual_state (no detach), so no separate round-0 loss needed.
        loss = final_rec["loss"]
        mu = final_rec["mu"]
        lambda_nr = final_rec["lambda_nr"]
        degree = final_rec["degree"]
        entropy = final_rec["entropy"]
        residual = final_rec["residual"]
        C_theta = final_rec["C_theta"]
        C_mod = final_rec["C_mod"]
        proj_aux = final_rec["proj_aux"]
        bounded_logits = final_rec["bounded_logits"]
        stage2_coupled_info = final_rec.get("stage2_coupled_info")

        with torch.no_grad():
            mu_masked = mu.masked_fill(~full_mask, 0.0)
            top2_mass = torch.topk(mu_masked, k=min(2, n - 1), dim=-1).values.sum(dim=-1)
            concentration = (top2_mass / degree.clamp_min(1e-8)).mean()
            mass_outside_candidates = ((mu * non_cand).sum(dim=(-2, -1)) / mu.sum(dim=(-2, -1)).clamp_min(1e-8)).mean()
            if self.logit_clamp > 0.0:
                clamp_frac = (bounded_logits.abs() >= (0.98 * self.logit_clamp)).float().mean()
            else:
                clamp_frac = torch.zeros((), device=device, dtype=mu.dtype)
            ift_trusted_frac = (residual <= self.ift_backward_tol).float().mean()
            if stage2_coupled_info is None:
                stage2_coupled_steps = torch.zeros((), device=device, dtype=mu.dtype)
                stage2_coupled_delta_mu = torch.zeros((), device=device, dtype=mu.dtype)
                stage2_delta_mu_first = torch.zeros((), device=device, dtype=mu.dtype)
                stage2_delta_mu_last = torch.zeros((), device=device, dtype=mu.dtype)
                stage2_delta_mu_max_last = torch.zeros((), device=device, dtype=mu.dtype)
                stage2_delta_mu_contract = torch.ones((), device=device, dtype=mu.dtype)
                stage2_certificate_rms = torch.zeros((), device=device, dtype=mu.dtype)
                stage2_tilt_proj_rms = torch.zeros((), device=device, dtype=mu.dtype)
                stage2_tilt_proj_fallback_frac = torch.zeros((), device=device, dtype=mu.dtype)
            else:
                stage2_coupled_steps = stage2_coupled_info["steps"]
                stage2_coupled_delta_mu = stage2_coupled_info["delta_mu"]
                stage2_delta_mu_first = stage2_coupled_info.get(
                    "delta_mu_first", torch.zeros((), device=device, dtype=mu.dtype)
                )
                stage2_delta_mu_last = stage2_coupled_info.get(
                    "delta_mu_last", torch.zeros((), device=device, dtype=mu.dtype)
                )
                stage2_delta_mu_max_last = stage2_coupled_info.get(
                    "delta_mu_max_last", torch.zeros((), device=device, dtype=mu.dtype)
                )
                stage2_delta_mu_contract = stage2_coupled_info.get(
                    "delta_mu_contract", torch.ones((), device=device, dtype=mu.dtype)
                )
                stage2_certificate_rms = stage2_coupled_info["certificate_rms"]
                stage2_tilt_proj_rms = stage2_coupled_info["tilt_proj_rms"]
                stage2_tilt_proj_fallback_frac = stage2_coupled_info["tilt_proj_fallback_frac"]

            shift = C_mod.masked_fill(~full_mask, float("inf")).amin(dim=(-2, -1), keepdim=True)
            C_mod_shift = C_mod - shift
            scaled_pre = -C_mod_shift / max(float(self.tau), 1e-8)
            scaled_clamp_mask = ((scaled_pre <= -60.0 + 1e-6) | (scaled_pre >= 60.0 - 1e-6)) & full_mask
            scaled_clamp_frac = scaled_clamp_mask.float().sum() / full_mask.float().sum().clamp_min(1.0)

            F_nr = degree[:, 1:] - 2.0
            edge_count_drift = (0.5 * mu.sum(dim=(-2, -1)) - float(n)).abs().mean()
            sum_nonroot_residual = F_nr.sum(dim=-1).abs().mean()
            F_nr_norm = F_nr.norm(dim=-1).mean()
            root_degree_dev = (degree[:, 0] - 2.0).abs().mean()

            proj_row_leak = proj_aux["row_leak"].detach()
            proj_grand_leak = proj_aux["grand_leak"].detach()
            proj_removed_energy_frac = proj_aux["removed_energy_frac"].detach()
            proj_fallback_frac = proj_aux["fallback_frac"].detach()
            node_additive = proj_aux["node_additive"].detach()
            if node_additive.shape[-1] > 1:
                add_nr = node_additive[:, 1:]
                add_norm = add_nr.norm(dim=-1)
                lam_norm = lambda_nr.norm(dim=-1)
                valid = (add_norm > 1e-8) & (lam_norm > 1e-8)
                if bool(valid.any()):
                    proj_lambda_align = F.cosine_similarity(add_nr[valid], lambda_nr[valid], dim=-1,
                                                            eps=1e-8).abs().mean()
                else:
                    proj_lambda_align = torch.zeros((), device=device, dtype=lambda_nr.dtype)
            else:
                proj_lambda_align = torch.zeros((), device=device, dtype=lambda_nr.dtype)

        stats = {
            "cost": mu.new_tensor(final_rec["cost"]).detach() if not isinstance(final_rec["cost"], torch.Tensor) else
            final_rec["cost"].detach(),
            "loss": loss.detach(),
            "entropy": entropy.detach(),
            "candidate_density": cand_mask.float().mean().detach(),
            "degree_mismatch": (degree[:, 1:] - 2.0).abs().mean().detach(),
            "degree_mismatch_max": (degree[:, 1:] - 2.0).abs().max().detach(),
            "resid_penalty": final_rec["resid_penalty"].mean().detach(),
            "struct_cert": final_rec["struct_cert"].mean().detach(),
            "struct_cert_norm": final_rec["struct_cert_norm"].mean().detach(),
            "struct_defect_cert": final_rec["struct_defect_cert"].mean().detach(),
            "struct_uncert_cert": final_rec["struct_uncert_cert"].mean().detach(),
            "struct_mean_defect": final_rec["struct_mean_defect"].mean().detach(),
            "struct_mean_uncert": final_rec["struct_mean_uncert"].mean().detach(),
            "struct_excess": final_rec["struct_excess"].mean().detach(),
            "struct_penalty": final_rec["struct_penalty"].mean().detach(),
            "repair_bound_weight": final_rec["repair_bound_weight"].mean().detach(),
            "repair_upper_bound": final_rec["repair_upper_bound"].mean().detach(),
            "lambda_norm": lambda_nr.norm(dim=-1).mean().detach(),
            "solver_residual": residual.mean().detach(),
            "solver_residual_max": residual.max().detach(),
            "top2_concentration": concentration.detach(),
            "outside_candidate_mass": mass_outside_candidates.detach(),
            "logit_clamp_frac": clamp_frac.detach(),
            "scaled_clamp_frac": scaled_clamp_frac.detach(),
            "stage2_coupled_steps": stage2_coupled_steps.detach(),
            "stage2_coupled_delta_mu": stage2_coupled_delta_mu.detach(),
            "stage2_delta_mu_first": stage2_delta_mu_first.detach(),
            "stage2_delta_mu_last": stage2_delta_mu_last.detach(),
            "stage2_delta_mu_max_last": stage2_delta_mu_max_last.detach(),
            "stage2_delta_mu_contract": stage2_delta_mu_contract.detach(),
            "stage2_certificate_rms": stage2_certificate_rms.detach(),
            "stage2_tilt_proj_rms": stage2_tilt_proj_rms.detach(),
            "stage2_tilt_proj_fallback_frac": stage2_tilt_proj_fallback_frac.detach(),
            "ift_trusted_frac": ift_trusted_frac.detach(),
            "edge_count_drift": edge_count_drift.detach(),
            "sum_nonroot_residual": sum_nonroot_residual.detach(),
            "F_nr_norm": F_nr_norm.detach(),
            "G_norm": residual.mean().detach(),
            "root_degree_dev": root_degree_dev.detach(),
            "proj_row_leak": proj_row_leak,
            "proj_grand_leak": proj_grand_leak,
            "proj_removed_energy_frac": proj_removed_energy_frac,
            "proj_lambda_align": proj_lambda_align.detach(),
            "proj_fallback_frac": proj_fallback_frac,
            "round0_cost": round_records[0]["cost"].mean().detach(),
            "round1_cost": round_records[1]["cost"].mean().detach(),
            "round0_solver_residual": round_records[0]["residual"].mean().detach(),
            "round1_solver_residual": round_records[1]["residual"].mean().detach(),
            "round1_update_rms": round_records[1]["update_info"]["round2_update_rms"].detach(),
            "round1_raw_rms": round_records[1]["update_info"]["round2_raw_rms"].detach(),
            "round1_scale": round_records[1]["update_info"]["round2_scale"].detach(),
            "round1_gate": round_records[1]["update_info"]["round2_gate"].detach(),
            "round1_gate_mult": round_records[1]["update_info"]["round2_gate_mult"].detach(),
            "round1_free_scale": round_records[1]["update_info"]["round2_free_scale"].detach(),
            "round1_gate_logit": round_records[1]["update_info"]["round2_gate_logit"].detach(),
            "round1_gate_residual": round_records[1]["update_info"]["round2_gate_residual"].detach(),
        }
        if return_decode_aux:
            with torch.no_grad():
                root = int(self.root)
                nonroot = [v for v in range(n) if v != root]
                pair_prob_full = torch.zeros(B, n, n, device=device, dtype=mu.dtype)
                if len(nonroot) >= 2:
                    tau_safe = max(float(self.tau), 1e-8)
                    nr_idx = torch.tensor(nonroot, device=device, dtype=torch.long)
                    root_scores = -C_mod[:, root, nr_idx] / tau_safe
                    m = len(nonroot)
                    upper_mask = torch.triu(torch.ones(m, m, device=device, dtype=torch.bool), diagonal=1)
                    pair_logits = root_scores.unsqueeze(-1) + root_scores.unsqueeze(-2)
                    neg_inf = torch.tensor(float('-inf'), device=device, dtype=root_scores.dtype)
                    pair_logits_masked = torch.where(upper_mask.unsqueeze(0), pair_logits, neg_inf)
                    logZ_root = torch.logsumexp(pair_logits_masked.reshape(B, -1), dim=-1)
                    pair_prob_nr = torch.exp(pair_logits_masked - logZ_root.view(B, 1, 1))
                    pair_prob_nr = torch.where(upper_mask.unsqueeze(0), pair_prob_nr, torch.zeros_like(pair_prob_nr))
                    for ii, u in enumerate(nonroot):
                        for jj, v in enumerate(nonroot):
                            pair_prob_full[:, u, v] = pair_prob_nr[:, ii, jj]
                aux = {
                    "C_theta": C_theta.detach(),
                    "lambda_nr": lambda_nr.detach(),
                    "tau": float(self.tau),
                    "root": int(self.root),
                    "pair_prob": pair_prob_full.detach(),
                    "C_mod": C_mod.detach(),
                    "cand_mask": cand_mask.detach(),
                    "round0_C_mod": round_records[0]["C_mod"].detach(),
                }
            return mu, loss, stats, aux
        return mu, loss, stats


# ================================================================
# §7  Decoding: MAP rooted 1-tree readout from reduced costs
# ================================================================


def prim_minimum_spanning_tree(cost: np.ndarray, root: int = 0) -> np.ndarray:
    """Minimum spanning tree adjacency from a dense symmetric cost matrix."""
    c = np.asarray(cost, dtype=np.float64)
    n = c.shape[0]
    root = int(root)
    in_tree = np.zeros(n, dtype=bool)
    parent = np.full(n, -1, dtype=np.int64)
    best = np.full(n, np.inf, dtype=np.float64)
    best[root] = 0.0

    for _ in range(n):
        masked_best = np.where(in_tree, np.inf, best)
        u = int(np.argmin(masked_best))
        in_tree[u] = True
        better = (~in_tree) & (c[u] < best)
        parent[better] = u
        best = np.where(better, c[u], best)

    adj = np.zeros((n, n), dtype=np.int64)
    rows = np.where(parent >= 0)[0]
    for v in rows.tolist():
        u = int(parent[v])
        adj[u, v] = 1
        adj[v, u] = 1
    return adj


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def cycle_from_degree2_edges(edges: List[Tuple[int, int]], n: int, root: int = 0) -> list[int] | None:
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    if any(len(nei) != 2 for nei in adj):
        return None
    tour = [root]
    prev = -1
    cur = root
    for _ in range(n - 1):
        nxts = adj[cur]
        nxt = nxts[0] if nxts[0] != prev else nxts[1]
        if nxt == root:
            return None
        tour.append(nxt)
        prev, cur = cur, nxt
    if root not in adj[cur]:
        return None
    if len(set(tour)) != n:
        return None
    return tour


def _tour_cost_numpy(tour: list[int], D: np.ndarray) -> float:
    idx = np.asarray(tour, dtype=np.int64)
    nxt = np.asarray(tour[1:] + tour[:1], dtype=np.int64)
    return float(np.sum(D[idx, nxt]))


def two_opt_rooted_tour_preserve_root_endpoints(
        tour: list[int],
        D: np.ndarray,
        max_passes: int = 6,
) -> list[int]:
    """2-opt on a rooted tour [root, a, ..., b] while preserving root neighbors a and b."""
    n = len(tour)
    if n <= 5 or max_passes <= 0:
        return tour
    arr = np.asarray(tour, dtype=np.int64).copy()
    passes = 0
    while passes < max_passes:
        passes += 1
        improved = False
        for i in range(2, n - 2):
            a = int(arr[i - 1])
            b = int(arr[i])
            for j in range(i + 1, n - 1):
                c = int(arr[j])
                d = int(arr[(j + 1) % n])
                delta = float(D[a, c] + D[b, d] - D[a, b] - D[c, d])
                if delta < -1e-12:
                    arr[i: j + 1] = arr[i: j + 1][::-1]
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return arr.tolist()


def _adjacency_lists(adj: np.ndarray) -> list[list[int]]:
    return [np.flatnonzero(adj[u]).astype(int).tolist() for u in range(adj.shape[0])]


def _tree_path(adj: np.ndarray, start: int, goal: int) -> list[int]:
    """Unique path between start and goal in a tree adjacency matrix."""
    n = adj.shape[0]
    start = int(start)
    goal = int(goal)
    parent = np.full(n, -2, dtype=np.int64)
    stack = [start]
    parent[start] = -1
    while stack:
        u = int(stack.pop())
        if u == goal:
            break
        for v in np.flatnonzero(adj[u]).astype(int).tolist():
            if parent[v] != -2:
                continue
            parent[v] = u
            stack.append(v)
    if parent[goal] == -2:
        raise RuntimeError(f"Tree path from {start} to {goal} not found.")
    path = [goal]
    cur = goal
    while parent[cur] != -1:
        cur = int(parent[cur])
        path.append(cur)
    path.reverse()
    return path


def _subtree_preorder_excluding_backbone(
        adj: np.ndarray,
        C_mod: np.ndarray,
        node: int,
        parent: int,
        backbone_set: set[int],
) -> list[int]:
    order = [int(node)]
    children = [
        int(v)
        for v in np.flatnonzero(adj[node]).astype(int).tolist()
        if int(v) != int(parent) and int(v) not in backbone_set
    ]
    children.sort(key=lambda v: (float(C_mod[node, v]), int(v)))
    for child in children:
        order.extend(_subtree_preorder_excluding_backbone(adj, C_mod, child, node, backbone_set))
    return order


def _off_backbone_preorder(
        adj: np.ndarray,
        C_mod: np.ndarray,
        backbone: list[int],
) -> list[int]:
    """Deterministic order of off-backbone nodes guided by the MAP tree and C_mod."""
    backbone_set = set(int(v) for v in backbone)
    order: list[int] = []
    for u in backbone:
        children = [
            int(v)
            for v in np.flatnonzero(adj[u]).astype(int).tolist()
            if int(v) not in backbone_set
        ]
        children.sort(key=lambda v: (float(C_mod[u, v]), int(v)))
        for child in children:
            order.extend(_subtree_preorder_excluding_backbone(adj, C_mod, child, u, backbone_set))
    return order


def _insert_node_cheapest_path_position(path: list[int], node: int, C_mod: np.ndarray) -> None:
    """Insert node into the rooted nonroot path while preserving the endpoints."""
    if len(path) < 2:
        path.append(int(node))
        return
    best_pos = 1
    best_delta = float('inf')
    node = int(node)
    for i in range(len(path) - 1):
        u = int(path[i])
        v = int(path[i + 1])
        delta = float(C_mod[u, node] + C_mod[node, v] - C_mod[u, v])
        cand = (delta, float(C_mod[u, node] + C_mod[node, v]), i + 1)
        if cand < (best_delta, float('inf'), best_pos):
            best_delta = cand[0]
            best_pos = cand[2]
    path.insert(best_pos, node)


def map_rooted_onetree_from_cmod(
        C_mod_single: np.ndarray,
        root: int = 0,
) -> tuple[np.ndarray, list[tuple[int, int]], tuple[int, int], np.ndarray]:
    """Compute the MAP rooted 1-tree under reduced costs C_mod."""
    C = np.asarray(C_mod_single, dtype=np.float64)
    n = C.shape[0]
    root = int(root)
    nonroot = [v for v in range(n) if v != root]
    if len(nonroot) < 2:
        raise ValueError("Need at least two non-root nodes for rooted 1-tree decoding.")

    C_nr = C[np.ix_(nonroot, nonroot)]
    adj_nr = prim_minimum_spanning_tree(C_nr, root=0)

    root_costs = np.asarray(C[root, nonroot], dtype=np.float64)
    order = np.argsort(root_costs, kind='mergesort')
    a = int(nonroot[int(order[0])])
    b = int(nonroot[int(order[1])])

    full_adj = np.zeros((n, n), dtype=np.int64)
    rows, cols = np.where(np.triu(adj_nr, k=1) > 0)
    edges: list[tuple[int, int]] = []
    for i, j in zip(rows.tolist(), cols.tolist()):
        u = int(nonroot[i])
        v = int(nonroot[j])
        full_adj[u, v] = 1
        full_adj[v, u] = 1
        edges.append((u, v))
    full_adj[root, a] = 1
    full_adj[a, root] = 1
    full_adj[root, b] = 1
    full_adj[b, root] = 1
    edges.append((root, a))
    edges.append((root, b))
    return full_adj, edges, (a, b), np.asarray(nonroot, dtype=np.int64)


def _tree_hop_distance(adj: np.ndarray, start: int) -> np.ndarray:
    """Hop distances from start in a tree adjacency matrix."""
    n = adj.shape[0]
    start = int(start)
    dist = np.full(n, n + 1, dtype=np.int64)
    dist[start] = 0
    q = [start]
    head = 0
    while head < len(q):
        u = int(q[head])
        head += 1
        for v in np.flatnonzero(adj[u]).astype(int).tolist():
            if dist[v] <= dist[u] + 1:
                continue
            dist[v] = dist[u] + 1
            q.append(int(v))
    return dist


def backbone_insertion_path_from_rooted_onetree(
        full_adj: np.ndarray,
        C_mod_single: np.ndarray,
        root: int,
        root_pair: tuple[int, int],
) -> list[int]:
    """Construct a non-root Hamiltonian path by preserving the MAP-tree backbone.

    The path starts with the unique tree path between the two MAP root neighbors and then
    inserts off-backbone nodes in a deterministic tree-aware order by cheapest insertion
    under C_mod. This is less myopic than single-ended sequential growth.
    """
    n = full_adj.shape[0]
    root = int(root)
    a, b = int(root_pair[0]), int(root_pair[1])
    nonroot = [int(v) for v in range(n) if int(v) != root]
    idx_of = {int(v): i for i, v in enumerate(nonroot)}
    adj_nr = full_adj[np.ix_(nonroot, nonroot)]
    a_nr = idx_of[a]
    b_nr = idx_of[b]
    backbone_nr = _tree_path(adj_nr, a_nr, b_nr)
    backbone = [int(nonroot[idx]) for idx in backbone_nr]
    path = backbone.copy()
    off_nodes = _off_backbone_preorder(adj_nr, C_mod_single[np.ix_(nonroot, nonroot)], backbone_nr)
    for node_nr in off_nodes:
        node = int(nonroot[int(node_nr)])
        if node in path:
            continue
        _insert_node_cheapest_path_position(path, node, C_mod_single)
    return path


def _tree_edge_priority_for_removal(
        full_adj: np.ndarray,
        u: int,
        mu: np.ndarray,
        root: int,
        root_pair: tuple[int, int],
) -> list[tuple[int, float]]:
    """For vertex u with tree degree > 2 in the rooted 1-tree, enumerate which of
    its incident tree edges are 'safe' to remove (i.e., not a root edge protected
    by the MAP root pair), and return (neighbor, removal_priority) pairs.

    Priority is -log(mu_e + eps): low priority (small mu) = preferred removal.
    """
    EPS = 1e-8
    n = full_adj.shape[0]
    out: list[tuple[int, float]] = []
    protected = {(int(root), int(root_pair[0])), (int(root_pair[0]), int(root)),
                 (int(root), int(root_pair[1])), (int(root_pair[1]), int(root))}
    for v in np.flatnonzero(full_adj[u]).tolist():
        v = int(v)
        if v == u:
            continue
        if (u, v) in protected:
            continue  # Do not remove a protected root edge
        m = float(mu[u, v])
        # Priority: -log(mu+eps) — LOW mu → LARGE priority (good candidate to remove)
        priority = -math.log(m + EPS)
        out.append((v, priority))
    # Sort by priority descending (prefer edges with lowest mu to remove first)
    out.sort(key=lambda pv: -pv[1])
    return out


def mu_weighted_matching_repair(
        full_adj: np.ndarray,
        mu: np.ndarray,
        C_mod_single: np.ndarray,
        D_single: np.ndarray,
        root: int,
        root_pair: tuple[int, int],
        twoopt_passes: int = 0,
) -> tuple[list[int] | None, bool]:
    """Prior-faithful repair: use the model's learned marginals μ to score swaps,
    rather than C_mod cost. Solves a minimum-cost matching on the degree-violation set,
    where swap cost = -log(μ_added) + log(μ_removed), i.e. 'log-likelihood decrease
    under the learned 1-tree measure'.

    Returns (tour, success). If success=False, caller falls back to the existing
    backbone-insertion repair.

    Handles:
      - k=1 (one deg-1 + one deg-3 pair): optimal swap by argmin
      - k=2, 3: brute-force enumeration of small matchings
      - k>=4: scipy Hungarian (if available)

    Connectivity correctness: after swaps, we must have n edges forming a
    Hamiltonian cycle. We check this explicitly and return failure if not.
    """
    n = full_adj.shape[0]
    root = int(root)
    deg = full_adj.sum(axis=1)

    # Identify deg-1 (deficit, need a new edge) and deg-3+ (excess, need to drop one)
    # among non-root vertices. Root always has degree 2 in a rooted 1-tree MAP.
    nonroot_mask = np.ones(n, dtype=bool)
    nonroot_mask[root] = False
    deg1_vertices = [int(v) for v in range(n) if nonroot_mask[v] and deg[v] == 1]
    deg3_vertices = [int(v) for v in range(n) if nonroot_mask[v] and deg[v] >= 3]

    # Guard: require balanced counts. Each deg-3 has excess-1, each deg-1 has deficit-1.
    # If deg-4 or worse appears, this simplified matcher doesn't handle it —
    # fall back.
    if len(deg1_vertices) != len(deg3_vertices):
        return None, False
    if any(deg[u] > 3 for u in deg3_vertices):
        return None, False
    k = len(deg1_vertices)
    if k == 0:
        # Already a tour — shouldn't be called, but handle gracefully
        return None, False
    # Cap problem size; beyond this, fall back to insertion repair.
    if k > 6:
        return None, False

    EPS = 1e-8
    mu_sym = 0.5 * (mu + mu.T)

    # Build the cost matrix for the assignment problem.
    # Row i = deg-3 vertex u_i, col j = deg-1 vertex v_j.
    # SWAP: remove edge (u_i, x) at u_i, and add edge (x, v_j) where x is
    # the orphaned tree neighbor. This properly rebalances degrees:
    #   u_i : deg 3 → 2  (lost one neighbor x)
    #   x   : deg d → d  (lost u_i but gained v_j, unchanged)
    #   v_j : deg 1 → 2  (gained x)
    # Score = -log(mu_{x, v_j} + eps) - (-log(mu_{u_i, x} + eps))
    #       = -log(mu_{x, v_j}) + log(mu_{u_i, x})
    # Low cost = high-probability add AND low-probability remove.
    # We pick the best (x, mu_{u,x}) pair for each (u, v).
    cost_matrix = np.full((k, k), np.inf, dtype=np.float64)
    removal_choice = np.zeros((k, k), dtype=np.int64)  # which x (= neighbor of u_i to remove)
    for i, u in enumerate(deg3_vertices):
        removal_options = _tree_edge_priority_for_removal(full_adj, u, mu_sym, root, root_pair)
        if len(removal_options) == 0:
            continue
        for j, v in enumerate(deg1_vertices):
            if u == v:
                continue
            # Try each removal candidate x and take the best combined swap score.
            best_swap_cost = np.inf
            best_x = -1
            for (x, remove_priority) in removal_options:
                if x == v:
                    continue  # degenerate
                if full_adj[x, v] > 0:
                    continue  # edge (x,v) already exists — swap would not change adjacency,
                    # leaving v with degree 1 and x with degree d-1 (bad)
                add_score = -math.log(float(mu_sym[x, v]) + EPS)
                swap_cost = add_score - remove_priority
                if swap_cost < best_swap_cost:
                    best_swap_cost = swap_cost
                    best_x = x
            if best_x >= 0 and np.isfinite(best_swap_cost):
                cost_matrix[i, j] = best_swap_cost
                removal_choice[i, j] = best_x

    if not np.isfinite(cost_matrix).any():
        return None, False

    # Solve assignment
    if k == 1:
        assignment = np.array([0], dtype=np.int64)
    elif k <= 3:
        # Brute force: enumerate all k! permutations
        from itertools import permutations
        best_cost = np.inf
        best_perm = None
        for perm in permutations(range(k)):
            c = sum(cost_matrix[i, perm[i]] for i in range(k))
            if c < best_cost:
                best_cost = c
                best_perm = perm
        if best_perm is None or not np.isfinite(best_cost):
            return None, False
        assignment = np.array(best_perm, dtype=np.int64)
    else:
        # Hungarian via scipy
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assignment = col_ind
        except Exception:
            return None, False
        if not np.all(np.isfinite(cost_matrix[np.arange(k), assignment])):
            return None, False

    # Apply the swaps
    new_adj = full_adj.copy()
    for i in range(k):
        u = deg3_vertices[i]
        v = deg1_vertices[assignment[i]]
        x = int(removal_choice[i, assignment[i]])  # orphan endpoint
        # Remove edge (u, x)
        new_adj[u, x] = 0
        new_adj[x, u] = 0
        # Add edge (x, v): reconnect orphan to deg-1 vertex
        new_adj[x, v] = 1
        new_adj[v, x] = 1

    # Verify: must now be degree-2 everywhere, connected, one cycle
    new_deg = new_adj.sum(axis=1)
    if not np.all(new_deg == 2):
        return None, False

    edges = [(int(i), int(j)) for i in range(n) for j in range(i + 1, n) if new_adj[i, j] > 0]
    if len(edges) != n:
        return None, False

    # Extract cycle from edges
    tour = cycle_from_degree2_edges(edges, n=n, root=root)
    if tour is None:
        return None, False

    if twoopt_passes > 0:
        tour = two_opt_rooted_tour_preserve_root_endpoints(
            tour, D_single, max_passes=max(1, min(2, int(twoopt_passes)))
        )
    return tour, True


def repair_rooted_onetree_to_tour(
        full_adj: np.ndarray,
        C_mod_single: np.ndarray,
        D_single: np.ndarray,
        root: int,
        root_pair: tuple[int, int],
        twoopt_passes: int = 0,
        mu_single: np.ndarray | None = None,
) -> tuple[list[int], bool]:
    """Deterministic tree-aware repair from a rooted 1-tree to a valid tour.

    Returns (tour, used_repair). Path:
      1. If the MAP rooted 1-tree is already degree-2 everywhere, extract cycle directly.
      2. Else if mu_single is provided, attempt μ-weighted matching repair: swaps
         are chosen to maximize log-likelihood under the learned 1-tree measure.
         Preserves the model's prior structure.
      3. Else (or if matching fails), fall back to backbone-insertion repair
         using C_mod cheapest-insertion.
    """
    n = full_adj.shape[0]
    root = int(root)
    edges = [(int(i), int(j)) for i in range(n) for j in range(i + 1, n) if full_adj[i, j] > 0]
    deg = full_adj.sum(axis=1)
    if np.all(deg == 2):
        exact_tour = cycle_from_degree2_edges(edges, n=n, root=root)
        if exact_tour is not None:
            exact_tour = two_opt_rooted_tour_preserve_root_endpoints(exact_tour, D_single,
                                                                     max_passes=max(0, min(2, int(twoopt_passes))))
            return exact_tour, False

    # μ-weighted matching repair (prior-faithful) — try first if mu available
    if mu_single is not None:
        matched_tour, ok = mu_weighted_matching_repair(
            full_adj, mu_single, C_mod_single, D_single,
            root=root, root_pair=root_pair, twoopt_passes=twoopt_passes,
        )
        if ok and matched_tour is not None:
            return matched_tour, True

    # Fallback: backbone + cheapest-insertion repair (C_mod based)
    path = backbone_insertion_path_from_rooted_onetree(
        full_adj,
        C_mod_single,
        root=root,
        root_pair=root_pair,
    )
    tour = [root] + path
    if twoopt_passes > 0:
        tour = two_opt_rooted_tour_preserve_root_endpoints(tour, D_single,
                                                           max_passes=max(1, min(2, int(twoopt_passes))))
    return tour, True


def two_opt_cycle(tour: list[int], D: np.ndarray, max_passes: int = 6) -> list[int]:
    """Standard 2-opt on a cycle represented as an ordered tour list."""
    n = len(tour)
    if n <= 4 or max_passes <= 0:
        return tour
    arr = np.asarray(tour, dtype=np.int64).copy()
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            a = int(arr[i])
            b = int(arr[(i + 1) % n])
            for j in range(i + 2, n if i > 0 else n - 1):
                c = int(arr[j])
                d = int(arr[(j + 1) % n])
                delta = float(D[a, c] + D[b, d] - D[a, b] - D[c, d])
                if delta < -1e-12:
                    arr[i + 1: j + 1] = arr[i + 1: j + 1][::-1]
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return arr.tolist()


def _rotate_tour_to_root(tour: list[int], root: int) -> list[int]:
    root = int(root)
    if tour[0] == root:
        return tour
    ridx = tour.index(root)
    return tour[ridx:] + tour[:ridx]


def greedy_degree2_tour_from_scores(score: np.ndarray, D: np.ndarray, root: int = 0) -> tuple[list[int], bool]:
    """Greedy Hamiltonian cycle extraction from a dense symmetric edge score matrix."""
    n = score.shape[0]
    deg = np.zeros(n, dtype=int)
    dsu = _DSU(n)
    selected: list[tuple[int, int]] = []
    all_edges: list[tuple[float, float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.append((float(score[i, j]), -float(D[i, j]), i, j))
    all_edges.sort(reverse=True)

    for _, _, u, v in all_edges:
        if deg[u] >= 2 or deg[v] >= 2:
            continue
        same = dsu.find(u) == dsu.find(v)
        if same:
            if len(selected) == n - 1 and deg[u] == 1 and deg[v] == 1:
                selected.append((u, v))
                deg[u] += 1
                deg[v] += 1
                break
            continue
        dsu.union(u, v)
        selected.append((u, v))
        deg[u] += 1
        deg[v] += 1

    if len(selected) == n:
        tour = cycle_from_degree2_edges(selected, n=n, root=root)
        if tour is not None:
            return tour, False

    # Very rare fallback: nearest-neighbor style completion on scores.
    root = int(root)
    vis = {root}
    order = [root]
    cur = root
    while len(order) < n:
        best = None
        best_v = None
        for v in range(n):
            if v in vis:
                continue
            key = (-float(score[cur, v]), float(D[cur, v]), int(v))
            if best is None or key < best:
                best = key
                best_v = int(v)
        order.append(best_v)
        vis.add(best_v)
        cur = best_v
    return order, True


def _topk_indices_desc(values: np.ndarray, k: int, exclude: int | None = None) -> list[int]:
    arr = np.asarray(values)
    n = arr.shape[0]
    if exclude is not None:
        arr = arr.copy()
        arr[int(exclude)] = -np.inf
    k = max(0, min(int(k), n - (1 if exclude is not None else 0)))
    if k <= 0:
        return []
    idx = np.argpartition(-arr, kth=np.arange(k))[:k]
    idx = idx[np.argsort(-arr[idx], kind='mergesort')]
    return [int(i) for i in idx.tolist() if (exclude is None or int(i) != int(exclude))]


def _topk_indices_asc(values: np.ndarray, k: int, exclude: int | None = None) -> list[int]:
    arr = np.asarray(values)
    n = arr.shape[0]
    if exclude is not None:
        arr = arr.copy()
        arr[int(exclude)] = np.inf
    k = max(0, min(int(k), n - (1 if exclude is not None else 0)))
    if k <= 0:
        return []
    idx = np.argpartition(arr, kth=np.arange(k))[:k]
    idx = idx[np.argsort(arr[idx], kind='mergesort')]
    return [int(i) for i in idx.tolist() if (exclude is None or int(i) != int(exclude))]


def build_move_candidates(
        mu_single: np.ndarray,
        C_mod_single: np.ndarray,
        cand_mask_single: np.ndarray | None,
        extra_topk: int = 5,
) -> list[set[int]]:
    """Candidate edge set per node for local search.

    Union of geometric candidates, top-μ support, and lowest-C_mod neighbors.
    """
    n = C_mod_single.shape[0]
    mu_sym = 0.5 * (mu_single + mu_single.T)
    C_sym = 0.5 * (C_mod_single + C_mod_single.T)
    cands: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        if cand_mask_single is not None:
            for j in np.flatnonzero(cand_mask_single[i]).astype(int).tolist():
                if int(j) != i:
                    cands[i].add(int(j))
        for j in _topk_indices_desc(mu_sym[i], extra_topk, exclude=i):
            cands[i].add(int(j))
        for j in _topk_indices_asc(C_sym[i], extra_topk, exclude=i):
            cands[i].add(int(j))
    for i in range(n):
        for j in list(cands[i]):
            cands[j].add(i)
        cands[i].discard(i)
    return cands


def candidate_restricted_two_opt(
        tour: list[int],
        D: np.ndarray,
        candidate_sets: list[set[int]],
        max_passes: int = 6,
        C_mod: np.ndarray | None = None,
) -> list[int]:
    """Candidate-restricted 2-opt on a cycle.

    If C_mod is provided, candidates for each anchor edge (a,b) are tried in ascending
    order of C_mod[a, c] — this is the LK-alpha ordering that biases first-improvement
    search along the learned Lagrangian gradient. The acceptance criterion is unchanged
    (real distance D), so terminal local optima are correct regardless of ordering.
    """
    n = len(tour)
    if n <= 4 or max_passes <= 0:
        return tour
    arr = tour.copy()
    pos = {int(v): i for i, v in enumerate(arr)}
    for _ in range(max_passes):
        improved = False
        for i in range(n):
            a = int(arr[i])
            b = int(arr[(i + 1) % n])
            cand_union = set(candidate_sets[a]) | set(candidate_sets[b])
            if C_mod is not None:
                # Rank candidates ascending by C_mod[a, c] — smallest reduced cost first.
                cand_nodes = sorted(cand_union, key=lambda c: (float(C_mod[a, int(c)]), int(c)))
            else:
                cand_nodes = cand_union
            for c in cand_nodes:
                j = pos.get(int(c), -1)
                if j < 0:
                    continue
                if j == i or j == (i + 1) % n or (j + 1) % n == i:
                    continue
                if i < j:
                    i1, j1 = i, j
                else:
                    i1, j1 = j, i
                a1 = int(arr[i1])
                b1 = int(arr[(i1 + 1) % n])
                c1 = int(arr[j1])
                d1 = int(arr[(j1 + 1) % n])
                if b1 == c1 or d1 == a1:
                    continue
                delta = float(D[a1, c1] + D[b1, d1] - D[a1, b1] - D[c1, d1])
                if delta < -1e-12:
                    arr[i1 + 1: j1 + 1] = arr[i1 + 1: j1 + 1][::-1]
                    pos = {int(v): k for k, v in enumerate(arr)}
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return arr


def candidate_restricted_relocate_one(
        tour: list[int],
        D: np.ndarray,
        candidate_sets: list[set[int]],
        max_passes: int = 4,
        C_mod: np.ndarray | None = None,
) -> list[int]:
    """Candidate-restricted relocate-one on a cycle.

    If C_mod is provided, reinsertion candidates y are tried in ascending order of
    C_mod[x, y] — smallest reduced cost first. Acceptance still uses D.
    """
    n = len(tour)
    if n <= 4 or max_passes <= 0:
        return tour
    arr = tour.copy()
    for _ in range(max_passes):
        pos = {int(v): i for i, v in enumerate(arr)}
        improved = False
        for i in range(n):
            x = int(arr[i])
            prev_x = int(arr[(i - 1) % n])
            next_x = int(arr[(i + 1) % n])
            remove_gain = float(D[prev_x, next_x] - D[prev_x, x] - D[x, next_x])
            cand_iter = candidate_sets[x]
            if C_mod is not None:
                cand_iter = sorted(cand_iter, key=lambda y: (float(C_mod[x, int(y)]), int(y)))
            for y in cand_iter:
                j = pos.get(int(y), -1)
                if j < 0:
                    continue
                if j == i or (j + 1) % n == i or j == (i - 1) % n:
                    continue
                y_next = int(arr[(j + 1) % n])
                delta = remove_gain + float(D[y, x] + D[x, y_next] - D[y, y_next])
                if delta < -1e-12:
                    node = arr.pop(i)
                    if i < j:
                        j -= 1
                    arr.insert(j + 1, node)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return arr


def lk_lite_improve_tour(
        seed_tour: list[int],
        D: np.ndarray,
        candidate_sets: list[set[int]],
        twoopt_passes: int = 6,
        C_mod: np.ndarray | None = None,
) -> list[int]:
    """Lightweight LKH-style improvement: candidate-restricted 2-opt + relocate-1.

    If C_mod is provided, inner candidate orderings use ascending C_mod (LK-alpha).
    The final plain 2-opt cleanup is unchanged (unrestricted, uses D only).
    """
    tour = seed_tour.copy()
    outer = max(1, int(twoopt_passes)) if twoopt_passes > 0 else 0
    if outer <= 0:
        return tour
    for _ in range(outer):
        before = _tour_cost_numpy(tour, D)
        tour = candidate_restricted_two_opt(tour, D, candidate_sets, max_passes=max(1, twoopt_passes), C_mod=C_mod)
        tour = candidate_restricted_relocate_one(tour, D, candidate_sets, max_passes=max(1, twoopt_passes // 2 + 1),
                                                 C_mod=C_mod)
        after = _tour_cost_numpy(tour, D)
        if after >= before - 1e-12:
            break
    tour = two_opt_cycle(tour, D, max_passes=max(1, twoopt_passes))
    return tour


def _mu_score_matrix(mu_single: np.ndarray) -> np.ndarray:
    mu_sym = 0.5 * (mu_single + mu_single.T)
    score = np.log(np.clip(mu_sym, 1e-12, None))
    np.fill_diagonal(score, -np.inf)
    return score


def mu_greedy_seed(mu_single: np.ndarray, D_single: np.ndarray, root: int = 0) -> tuple[list[int], bool]:
    score = _mu_score_matrix(mu_single)
    return greedy_degree2_tour_from_scores(score, D_single, root=root)


def mu_gumbel_seed(
        mu_single: np.ndarray,
        D_single: np.ndarray,
        rng: np.random.Generator,
        root: int = 0,
        gumbel_scale: float = 0.20,
) -> tuple[list[int], bool]:
    score = _mu_score_matrix(mu_single)
    if gumbel_scale > 0.0:
        score = score + float(gumbel_scale) * _symmetric_gumbel_noise(score.shape[0], rng)
        np.fill_diagonal(score, -np.inf)
    return greedy_degree2_tour_from_scores(score, D_single, root=root)


def _gumbel_noise(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    u = rng.uniform(low=1e-12, high=1.0 - 1e-12, size=shape)
    return -np.log(-np.log(u))


def _symmetric_gumbel_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    noise = np.zeros((n, n), dtype=np.float64)
    iu, ju = np.triu_indices(n, k=1)
    vals = _gumbel_noise((len(iu),), rng)
    noise[iu, ju] = vals
    noise[ju, iu] = vals
    return noise


def map_repair_seed(
        C_mod_single: np.ndarray,
        D_single: np.ndarray,
        root: int = 0,
        twoopt_passes: int = 0,
        mu_single: np.ndarray | None = None,
) -> tuple[list[int], bool]:
    full_adj, _, root_pair, _ = map_rooted_onetree_from_cmod(C_mod_single, root=root)
    return repair_rooted_onetree_to_tour(
        full_adj,
        C_mod_single,
        D_single,
        root=root,
        root_pair=root_pair,
        twoopt_passes=twoopt_passes,
        mu_single=mu_single,
    )


def _perturb_cmod(
        C_mod_single: np.ndarray,
        root: int,
        rng: np.random.Generator,
        tree_gumbel_scale: float = 0.35,
        root_gumbel_scale: float = 0.20,
        score_gumbel_scale: float = 0.0,
) -> np.ndarray:
    """Symmetric Gumbel perturbation of C_mod for best-of-K candidate generation."""
    C = np.asarray(C_mod_single, dtype=np.float64).copy()
    n = C.shape[0]
    root = int(root)
    noise = _symmetric_gumbel_noise(n, rng)
    scale = np.full((n, n), float(tree_gumbel_scale + score_gumbel_scale), dtype=np.float64)
    scale[root, :] = float(root_gumbel_scale + score_gumbel_scale)
    scale[:, root] = float(root_gumbel_scale + score_gumbel_scale)
    np.fill_diagonal(scale, 0.0)
    return C + scale * noise


def _sample_pair_from_root_logits(log_a: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    """Exact sample of the two root neighbors from the closed-form pair law."""
    log_a = np.asarray(log_a, dtype=np.float64)
    m = log_a.shape[0]
    if m < 2:
        raise ValueError("Need at least two non-root nodes to sample a rooted 1-tree.")
    iu, ju = np.triu_indices(m, k=1)
    logits = log_a[iu] + log_a[ju]
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs = probs / probs.sum()
    pick = int(rng.choice(len(iu), p=probs))
    return int(iu[pick]), int(ju[pick])


def _sample_weighted_neighbor(weights: np.ndarray, rng: np.random.Generator) -> int:
    probs = np.asarray(weights, dtype=np.float64).copy()
    probs[~np.isfinite(probs)] = 0.0
    probs = np.clip(probs, a_min=0.0, a_max=None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("Encountered a zero/invalid transition row in weighted Wilson sampling.")
    probs /= total
    return int(rng.choice(probs.shape[0], p=probs))


def _sample_weighted_spanning_tree_wilson(
        W: np.ndarray,
        rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Exact weighted spanning-tree sample via Wilson's algorithm on conductances."""
    W = np.asarray(W, dtype=np.float64)
    n = W.shape[0]
    if n <= 1:
        return []
    if W.shape[1] != n:
        raise ValueError("W must be square.")
    if not np.all(np.isfinite(W)):
        raise RuntimeError("Weighted Wilson sampler received non-finite edge weights.")

    in_tree = np.zeros(n, dtype=bool)
    root = 0
    in_tree[root] = True
    sampled: list[tuple[int, int]] = []
    start_order = rng.permutation(n)

    for start in start_order.tolist():
        if in_tree[start]:
            continue
        path = [int(start)]
        loc = {int(start): 0}
        cur = int(start)
        max_steps = max(1000, 20 * n * n)
        steps = 0

        while not in_tree[cur]:
            row = W[cur].copy()
            row[cur] = 0.0
            nxt = _sample_weighted_neighbor(row, rng)
            if in_tree[nxt]:
                path.append(int(nxt))
                break
            if nxt in loc:
                cut = loc[nxt]
                for node in path[cut + 1:]:
                    loc.pop(int(node), None)
                path = path[: cut + 1]
            else:
                path.append(int(nxt))
                loc[int(nxt)] = len(path) - 1
            cur = int(path[-1])
            steps += 1
            if steps > max_steps:
                raise RuntimeError("Weighted Wilson sampler exceeded its step budget.")

        for i in range(len(path) - 1):
            u = int(path[i])
            v = int(path[i + 1])
            if not in_tree[u]:
                in_tree[u] = True
                sampled.append((u, v))

    if len(sampled) != n - 1:
        raise RuntimeError(f"Weighted Wilson sampler returned {len(sampled)} edges; expected {n - 1}.")
    return sampled


def _permute_root_to_zero(C: np.ndarray, root: int) -> tuple[np.ndarray, list[int], dict[int, int]]:
    root = int(root)
    n = C.shape[0]
    if root < 0 or root >= n:
        raise ValueError(f"root index {root} out of range for n={n}.")
    perm = [root] + [i for i in range(n) if i != root]
    inv = {new_i: old_i for new_i, old_i in enumerate(perm)}
    C_perm = C[np.ix_(perm, perm)]
    return C_perm, perm, inv


def _scaled_weight_matrix_from_cmod(
        C_mod_single: np.ndarray,
        tau: float,
        root: int = 0,
) -> tuple[np.ndarray, list[int], dict[int, int], np.ndarray, np.ndarray]:
    """Shared exact Gibbs ingredients for rooted 1-tree decoding on CPU."""
    C = np.asarray(C_mod_single, dtype=np.float64)
    n = C.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 nodes for rooted 1-tree operations.")
    C_perm, perm, inv = _permute_root_to_zero(C, root=root)
    offdiag = ~np.eye(n, dtype=bool)
    shift = float(np.min(C_perm[offdiag]))
    scaled = np.clip(-(C_perm - shift) / float(max(tau, 1e-8)), -60.0, 60.0)
    W = np.exp(scaled) * offdiag.astype(np.float64)
    W = 0.5 * (W + W.T)
    return C_perm, perm, inv, scaled, W


def map_nonroot_tree_edges_from_cmod(
        C_mod_single: np.ndarray,
        root: int = 0,
) -> list[tuple[int, int]]:
    """Deterministic MAP tree on the non-root graph under C_mod."""
    C = np.asarray(C_mod_single, dtype=np.float64)
    n = C.shape[0]
    C_perm, _, inv = _permute_root_to_zero(C, root=root)
    adj_nr = prim_minimum_spanning_tree(C_perm[1:, 1:], root=0)
    rows, cols = np.where(np.triu(adj_nr, k=1) > 0)
    edges: list[tuple[int, int]] = []
    for i, j in zip(rows.tolist(), cols.tolist()):
        edges.append((int(inv[i + 1]), int(inv[j + 1])))
    if len(edges) != n - 2:
        raise RuntimeError(f"Expected {n - 2} non-root tree edges, got {len(edges)}.")
    return edges


def sample_nonroot_tree_edges_from_cmod(
        C_mod_single: np.ndarray,
        tau: float,
        root: int = 0,
        rng: np.random.Generator | None = None,
) -> list[tuple[int, int]]:
    r"""Exact weighted spanning-tree sample on V\{root}."""
    if rng is None:
        rng = np.random.default_rng()
    _, _, inv, _, W = _scaled_weight_matrix_from_cmod(C_mod_single, tau=tau, root=root)
    W_nr = W[1:, 1:]
    nr_edges_perm = _sample_weighted_spanning_tree_wilson(W_nr, rng)
    return [(int(inv[u + 1]), int(inv[v + 1])) for (u, v) in nr_edges_perm]


def assemble_rooted_onetree_from_tree_and_pair(
        n: int,
        root: int,
        tree_edges: list[tuple[int, int]],
        root_pair: tuple[int, int],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Combine a non-root tree with a chosen root pair into a rooted 1-tree."""
    a, b = int(root_pair[0]), int(root_pair[1])
    root = int(root)
    if a == b or a == root or b == root:
        raise ValueError(f"Invalid root pair {root_pair} for root {root}.")
    edges = list(tree_edges) + [(root, a), (root, b)]
    return _adjacency_from_edges(n, edges), edges


def select_root_pair_candidates(
        C_mod_single: np.ndarray,
        root: int = 0,
        pair_prob_single: np.ndarray | None = None,
        k: int = 1,
) -> list[tuple[int, int]]:
    """Deterministic distinct root-pair branches ordered by the exact pair factor."""
    k = max(int(k), 0)
    if k <= 0:
        return []
    n = int(np.asarray(C_mod_single).shape[0])
    _, _, map_pair, _ = map_rooted_onetree_from_cmod(C_mod_single, root=root)
    map_pair = tuple(sorted((int(map_pair[0]), int(map_pair[1]))))
    selected: list[tuple[int, int]] = [map_pair]
    seen = {map_pair}

    scored: list[tuple[float, float, int, int]] = []
    if pair_prob_single is not None:
        P = np.asarray(pair_prob_single, dtype=np.float64)
        for u in range(n):
            for v in range(u + 1, n):
                if u == root or v == root:
                    continue
                prob = float(max(P[u, v], P[v, u]))
                root_cost = float(C_mod_single[root, u] + C_mod_single[root, v])
                scored.append((prob, -root_cost, int(u), int(v)))
        scored.sort(key=lambda t: (-t[0], -t[1], t[2], t[3]))
    else:
        for u in range(n):
            for v in range(u + 1, n):
                if u == root or v == root:
                    continue
                root_cost = float(C_mod_single[root, u] + C_mod_single[root, v])
                scored.append((-root_cost, 0.0, int(u), int(v)))
        scored.sort(key=lambda t: (-t[0], -t[1], t[2], t[3]))

    for _, _, u, v in scored:
        pair = tuple(sorted((int(u), int(v))))
        if pair in seen:
            continue
        selected.append(pair)
        seen.add(pair)
        if len(selected) >= k:
            break
    return selected[:k]


def sample_rooted_onetree_edges_from_cmod(
        C_mod_single: np.ndarray,
        tau: float,
        root: int = 0,
        rng: np.random.Generator | None = None,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """Exact sample from the rooted 1-tree Gibbs law defined by C_mod and tau."""
    if rng is None:
        rng = np.random.default_rng()
    C_perm, _, inv, scaled, W = _scaled_weight_matrix_from_cmod(C_mod_single, tau=tau, root=root)
    i_nr, j_nr = _sample_pair_from_root_logits(scaled[0, 1:], rng)
    root_pair_perm = (i_nr + 1, j_nr + 1)
    W_nr = W[1:, 1:]
    nr_edges_perm = _sample_weighted_spanning_tree_wilson(W_nr, rng)
    edges_perm = [(0, root_pair_perm[0]), (0, root_pair_perm[1])]
    edges_perm.extend((u + 1, v + 1) for (u, v) in nr_edges_perm)

    def _map_back(edge: tuple[int, int]) -> tuple[int, int]:
        u, v = edge
        return int(inv[int(u)]), int(inv[int(v)])

    edges = [_map_back(e) for e in edges_perm]
    root_pair = _map_back((0, root_pair_perm[0]))[1], _map_back((0, root_pair_perm[1]))[1]
    return edges, (int(root_pair[0]), int(root_pair[1]))


def _adjacency_from_edges(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    adj = np.zeros((n, n), dtype=np.int64)
    for u, v in edges:
        u = int(u)
        v = int(v)
        if u == v:
            continue
        adj[u, v] = 1
        adj[v, u] = 1
    return adj


def deterministic_decode_candidate_tours(
        mu_single: np.ndarray,
        C_mod_single: np.ndarray,
        cand_mask_single: np.ndarray | None,
        D_single: np.ndarray,
        root: int = 0,
        twoopt_passes: int = 6,
) -> tuple[list[list[int]], list[float], dict[str, float]]:
    """Deterministic decode portfolio used by both plain decode and best-of-K.

    The two strongest deterministic seeds in this codebase are μ-greedy degree-2
    extraction and MAP rooted-1-tree repair. We improve both with the same LK-lite
    stack and choose the best resulting tour.
    """
    candidate_sets = build_move_candidates(mu_single, C_mod_single, cand_mask_single, extra_topk=5)
    tours: list[list[int]] = []
    costs: list[float] = []
    seen: set[tuple[int, ...]] = set()
    mu_fallback = 0
    map_repair_used = 0

    def _add_candidate(seed_tour: list[int]) -> None:
        improved = lk_lite_improve_tour(seed_tour, D_single, candidate_sets, twoopt_passes=twoopt_passes)
        improved = _rotate_tour_to_root(improved, root=root)
        key = tuple(int(v) for v in improved)
        if key in seen:
            return
        seen.add(key)
        tours.append(improved)
        costs.append(_tour_cost_numpy(improved, D_single))

    seed_tour, used_fallback = mu_greedy_seed(mu_single, D_single, root=root)
    mu_fallback += int(used_fallback)
    _add_candidate(seed_tour)

    map_tour, used_repair = map_repair_seed(C_mod_single, D_single, root=root, twoopt_passes=0)
    map_repair_used += int(used_repair)
    _add_candidate(map_tour)

    if not tours:
        raise RuntimeError("Deterministic decode portfolio failed to produce any tour.")
    info = {
        "fallback_rate": float(mu_fallback),
        "map_repair_rate": float(map_repair_used),
        "num_unique": float(len(tours)),
        "proposal": "best_of{mu_greedy,map_repair}+candidate_restricted_lk_lite",
    }
    return tours, costs, info


def decode_tours_from_cmod(
        mu: torch.Tensor,
        C_mod: torch.Tensor,
        cand_mask: torch.Tensor | None,
        D: torch.Tensor,
        root: int = 0,
        twoopt_passes: int = 6,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Plain deterministic decode: best of μ-greedy and MAP-repair, both improved by LK-lite."""
    mu_np = mu.detach().cpu().numpy()
    C_np = C_mod.detach().cpu().numpy()
    cand_np = None if cand_mask is None else cand_mask.detach().cpu().numpy().astype(bool)
    D_np = D.detach().cpu().numpy()
    B, _, _ = C_np.shape
    tours, costs = [], []
    mu_fallback_count = 0.0
    map_repair_count = 0.0
    unique_count = 0.0

    for b in range(B):
        cand_single = None if cand_np is None else cand_np[b]
        cand_tours, cand_costs, dinfo = deterministic_decode_candidate_tours(
            mu_np[b], C_np[b], cand_single, D_np[b], root=root, twoopt_passes=twoopt_passes,
        )
        best_idx = int(np.argmin(cand_costs))
        tours.append(cand_tours[best_idx])
        costs.append(float(cand_costs[best_idx]))
        mu_fallback_count += float(dinfo["fallback_rate"])
        map_repair_count += float(dinfo["map_repair_rate"])
        unique_count += float(dinfo["num_unique"])

    info = {
        'fallback_rate': mu_fallback_count / float(max(1, B)),
        'map_repair_rate': map_repair_count / float(max(1, B)),
        'avg_unique_candidates': unique_count / float(max(1, B)),
        'proposal': 'best_of{mu_greedy,map_repair}+candidate_restricted_lk_lite',
    }
    return np.array(tours, dtype=object), np.array(costs), info


def decode_tours_from_sampled_onetrees(
        mu: torch.Tensor,
        C_mod: torch.Tensor,
        cand_mask: torch.Tensor | None,
        D: torch.Tensor,
        tau: float,
        root: int = 0,
        num_samples: int = 8,
        num_pair_samples: int | None = None,
        num_proposals_per_pair: int = 1,
        sample_bonus: float = 5.0,
        root_pair_bonus: float = 2.5,
        root_other_penalty: float = 0.0,
        seed_base: int = 0,
        tree_gumbel_scale: float = 0.35,
        root_gumbel_scale: float = 0.20,
        score_gumbel_scale: float = 0.0,
        twoopt_passes: int = 6,
        pair_prob: torch.Tensor | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Best-of-K decode with an explicit deterministic baseline and pair-aware branches.

    Portfolio per instance:
      - deterministic baseline: the same best-of{μ-greedy, MAP-repair} logic used by
        the plain decoder;
      - for each selected root pair, proposal 0 is the conditional MAP-tree branch;
      - additional proposals for that pair are exact Wilson tree samples. Because the
        rooted 1-tree family factorizes into an independent root-pair factor and a
        non-root spanning-tree factor, these are exact conditional proposals.
    """
    del sample_bonus, root_pair_bonus, root_other_penalty, tree_gumbel_scale, root_gumbel_scale, score_gumbel_scale, num_samples
    mu_np = mu.detach().cpu().numpy()
    C_np = C_mod.detach().cpu().numpy()
    cand_np = None if cand_mask is None else cand_mask.detach().cpu().numpy().astype(bool)
    D_np = D.detach().cpu().numpy()
    pair_np = None if pair_prob is None else pair_prob.detach().cpu().numpy()
    B, n, _ = C_np.shape

    pair_count = max(int(num_pair_samples) if num_pair_samples is not None else 0, 0)
    proposals_per_pair = max(int(num_proposals_per_pair), 0)
    if pair_count <= 0 or proposals_per_pair <= 0:
        pair_count = 0
        proposals_per_pair = 0

    tours_best, mean_costs, best_costs = [], [], []
    repair_count = 0
    exact_count = 0
    sampler_fail_count = 0
    sampled_attempts = 0
    total_branches = 0
    unique_portfolio_size = 0.0

    for b in range(B):
        cand_single = None if cand_np is None else cand_np[b]
        det_tours, _, _ = deterministic_decode_candidate_tours(
            mu_np[b], C_np[b], cand_single, D_np[b], root=root, twoopt_passes=twoopt_passes,
        )
        portfolio_tours: list[list[int]] = []
        portfolio_costs: list[float] = []
        seen: set[tuple[int, ...]] = set()

        def _add_portfolio_tour(tour: list[int]) -> None:
            key = tuple(int(v) for v in tour)
            if key in seen:
                return
            seen.add(key)
            portfolio_tours.append(tour)
            portfolio_costs.append(_tour_cost_numpy(tour, D_np[b]))

        for tour in det_tours:
            _add_portfolio_tour(tour)
            total_branches += 1

        selected_pairs = select_root_pair_candidates(
            C_np[b],
            root=root,
            pair_prob_single=None if pair_np is None else pair_np[b],
            k=pair_count,
        )
        map_tree_edges = map_nonroot_tree_edges_from_cmod(C_np[b], root=root)
        candidate_sets = build_move_candidates(mu_np[b], C_np[b], cand_single, extra_topk=5)

        for pair_idx, root_pair in enumerate(selected_pairs):
            for prop_idx in range(max(proposals_per_pair, 1)):
                total_branches += 1
                sampled_attempts += 1
                if prop_idx == 0:
                    full_adj, _ = assemble_rooted_onetree_from_tree_and_pair(n, root, map_tree_edges, root_pair)
                else:
                    rng = np.random.default_rng(
                        int(seed_base + 1000003 * b + 9176 * (pair_idx + 1) + 131 * prop_idx + 53))
                    try:
                        sampled_tree_edges = sample_nonroot_tree_edges_from_cmod(C_np[b], tau=float(tau), root=root,
                                                                                 rng=rng)
                        full_adj, _ = assemble_rooted_onetree_from_tree_and_pair(n, root, sampled_tree_edges, root_pair)
                    except RuntimeError:
                        sampler_fail_count += 1
                        full_adj, _ = assemble_rooted_onetree_from_tree_and_pair(n, root, map_tree_edges, root_pair)

                tour, used_repair = repair_rooted_onetree_to_tour(
                    full_adj,
                    C_np[b],
                    D_np[b],
                    root=root,
                    root_pair=root_pair,
                    twoopt_passes=0,
                )
                repair_count += int(used_repair)
                exact_count += int(not used_repair)
                improved = lk_lite_improve_tour(tour, D_np[b], candidate_sets, twoopt_passes=twoopt_passes)
                improved = _rotate_tour_to_root(improved, root=root)
                _add_portfolio_tour(improved)

        if not portfolio_tours:
            raise RuntimeError("Sampled decoder portfolio is empty.")
        unique_portfolio_size += float(len(portfolio_tours))
        mean_costs.append(float(np.mean(portfolio_costs)))
        best_idx = int(np.argmin(portfolio_costs))
        best_costs.append(float(portfolio_costs[best_idx]))
        tours_best.append(portfolio_tours[best_idx])

    info = {
        'fallback_rate': float(repair_count) / float(max(1, sampled_attempts)),
        'exact_rate': float(exact_count) / float(max(1, sampled_attempts)),
        'sampler_fail_rate': float(sampler_fail_count) / float(max(1, sampled_attempts)),
        'avg_unique_portfolio_size': unique_portfolio_size / float(max(1, B)),
        'proposal': 'deterministic_baseline + top_pair_conditional_map/tree_samples + candidate_restricted_lk_lite',
        'num_pair_samples': float(pair_count),
        'num_proposals_per_pair': float(proposals_per_pair),
        'total_branches': float(total_branches) / float(max(1, B)),
    }
    return np.array(tours_best, dtype=object), np.array(mean_costs), np.array(best_costs), info


# ================================================================
# §7b  Ablation decoder: run 4 strategies independently, report each
# ================================================================


def _tour_key(tour: list[int]) -> tuple[int, ...]:
    """Hashable key for a tour, rotation-invariant via min-index normalization."""
    if not tour:
        return tuple()
    n = len(tour)
    k = int(np.argmin(np.asarray(tour)))
    rot = tour[k:] + tour[:k]
    # Also canonicalize direction.
    if n >= 3 and rot[1] > rot[-1]:
        rot = [rot[0]] + rot[:0:-1]
    return tuple(int(v) for v in rot)


def _eval_seed(
        seed_tour: list[int],
        D_single: np.ndarray,
        candidate_sets: list[set[int]],
        root: int,
        twoopt_passes: int,
        C_mod_single: np.ndarray | None = None,
) -> tuple[list[int], float, list[int], float]:
    """Return (raw_tour, raw_cost, lk_tour, lk_cost) for one seed.

    If C_mod_single is provided, LK-alpha ordering is used inside lk_lite_improve_tour.
    """
    raw_tour = _rotate_tour_to_root(list(seed_tour), root=root)
    raw_cost = float(_tour_cost_numpy(raw_tour, D_single))
    if twoopt_passes > 0:
        lk_tour = lk_lite_improve_tour(raw_tour, D_single, candidate_sets, twoopt_passes=twoopt_passes,
                                       C_mod=C_mod_single)
        lk_tour = _rotate_tour_to_root(lk_tour, root=root)
        lk_cost = float(_tour_cost_numpy(lk_tour, D_single))
    else:
        lk_tour = raw_tour
        lk_cost = raw_cost
    return raw_tour, raw_cost, lk_tour, lk_cost


def _greedy_degree2_with_forced_pair(
        score: np.ndarray,
        D: np.ndarray,
        root: int,
        forced_pair: tuple[int, int],
) -> tuple[list[int], bool]:
    """Like greedy_degree2_tour_from_scores but forces the root pair edges.

    Forces (root, a) and (root, b) into the matching first, then greedily
    completes the degree-2 Hamiltonian cycle using `score` (with D as tiebreak).
    """
    n = score.shape[0]
    a, b = int(forced_pair[0]), int(forced_pair[1])
    root = int(root)
    if a == b or a == root or b == root:
        raise ValueError(f"Invalid forced pair {forced_pair} for root {root}.")

    deg = np.zeros(n, dtype=int)
    dsu = _DSU(n)
    selected: list[tuple[int, int]] = []

    # Force the two root-incident edges.
    for (u, v) in ((root, a), (root, b)):
        if dsu.find(u) == dsu.find(v):
            # Shouldn't happen with fresh DSU, but guard anyway.
            break
        dsu.union(u, v)
        selected.append((u, v))
        deg[u] += 1
        deg[v] += 1

    all_edges: list[tuple[float, float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if i == root or j == root:
                # Root's two edges are already fixed.
                continue
            all_edges.append((float(score[i, j]), -float(D[i, j]), i, j))
    all_edges.sort(reverse=True)

    for _, _, u, v in all_edges:
        if deg[u] >= 2 or deg[v] >= 2:
            continue
        same = dsu.find(u) == dsu.find(v)
        if same:
            if len(selected) == n - 1 and deg[u] == 1 and deg[v] == 1:
                selected.append((u, v))
                deg[u] += 1
                deg[v] += 1
                break
            continue
        dsu.union(u, v)
        selected.append((u, v))
        deg[u] += 1
        deg[v] += 1

    if len(selected) == n:
        tour = cycle_from_degree2_edges(selected, n=n, root=root)
        if tour is not None:
            return tour, False

    # Fallback: drop the forcing constraint and run the unconstrained greedy.
    return greedy_degree2_tour_from_scores(score, D, root=root)


def decode_tours_ablation(
        mu: torch.Tensor,
        C_mod: torch.Tensor,
        cand_mask: torch.Tensor | None,
        D: torch.Tensor,
        root: int = 0,
        twoopt_passes: int = 6,
        num_root_pairs: int = 4,
        num_gumbel_draws: int = 8,
        gumbel_scale: float = 0.20,
        seed_base: int = 0,
        pair_prob: torch.Tensor | None = None,
        use_lk_alpha: bool = False,
) -> Dict[str, np.ndarray | Dict]:
    r"""Run four decode strategies independently and report each plus the envelope.

    Strategies (each yields one or more tour candidates, all optionally LK-lite improved):
      S1 mu_greedy       : deterministic degree-2 greedy from \log \mu^{sym} with D tiebreak.
      S2 map_repair      : deterministic MAP rooted 1-tree + backbone-insertion repair.
      S3 rootpair_mapK   : top-K root pairs ranked by pair_prob (or root_cost fallback);
                            for each pair, use MAP non-root tree and repair.
      S4 gumbel_muM      : M independent Gumbel-perturbed mu-greedy degree-2 tours.

    Portfolio: best cost across all tours from S1..S4.
    Attribution: which strategy provided the portfolio winner per instance.

    Returns a dict keyed by strategy name with np arrays of length B for raw_cost_best
    and lk_cost_best (plus best tours), and 'portfolio' / 'attribution' / 'info'.
    """
    mu_np = mu.detach().cpu().numpy()
    C_np = C_mod.detach().cpu().numpy()
    cand_np = None if cand_mask is None else cand_mask.detach().cpu().numpy().astype(bool)
    D_np = D.detach().cpu().numpy()
    pair_np = None if pair_prob is None else pair_prob.detach().cpu().numpy()
    B, n, _ = C_np.shape

    K = max(0, int(num_root_pairs))
    M = max(0, int(num_gumbel_draws))

    # Per-strategy arrays.
    names = ["s1_mu_greedy", "s2_map_repair", "s3_rootpair_mapK", "s4_gumbel_muM"]
    results: Dict[str, Dict[str, list]] = {
        name: {"raw_tour": [], "raw_cost": [], "lk_tour": [], "lk_cost": []} for name in names
    }
    portfolio_tours: list[list[int]] = []
    portfolio_costs: list[float] = []
    attribution: list[str] = []

    # Per-strategy aggregate info.
    info_counts = {
        "s1_fallback_frac": 0.0,
        "s2_repair_frac": 0.0,
        "s3_repair_frac": 0.0,
        "s3_num_pairs_mean": 0.0,
        "s4_unique_frac": 0.0,  # mean unique tours / M
        "s4_draws": float(M),
    }

    for b in range(B):
        cand_single = None if cand_np is None else cand_np[b]
        mu_b = mu_np[b]
        C_b = C_np[b]
        D_b = D_np[b]
        pair_b = None if pair_np is None else pair_np[b]

        # Shared candidate sets for LK-lite across all strategies for this instance.
        candidate_sets = build_move_candidates(mu_b, C_b, cand_single, extra_topk=5)
        # LK-alpha: if enabled, pass C_mod to the inner moves so candidate ordering
        # follows the learned Lagrangian gradient. If disabled, C_mod_for_lk is None
        # and LK behavior is bit-for-bit identical to v6.
        C_mod_for_lk = C_b if use_lk_alpha else None

        best_cost_b = float("inf")
        best_tour_b: list[int] = []
        best_strategy_b = ""

        # ---------- S1: mu-greedy ----------
        seed1, used_fb1 = mu_greedy_seed(mu_b, D_b, root=root)
        info_counts["s1_fallback_frac"] += float(used_fb1)
        raw_t, raw_c, lk_t, lk_c = _eval_seed(seed1, D_b, candidate_sets, root, twoopt_passes,
                                              C_mod_single=C_mod_for_lk)
        results["s1_mu_greedy"]["raw_tour"].append(raw_t)
        results["s1_mu_greedy"]["raw_cost"].append(raw_c)
        results["s1_mu_greedy"]["lk_tour"].append(lk_t)
        results["s1_mu_greedy"]["lk_cost"].append(lk_c)
        if lk_c < best_cost_b:
            best_cost_b, best_tour_b, best_strategy_b = lk_c, lk_t, "s1_mu_greedy"

        # ---------- S2: MAP rooted 1-tree + repair ----------
        map_full_adj, _, map_root_pair, _ = map_rooted_onetree_from_cmod(C_b, root=root)
        seed2, used_repair2 = repair_rooted_onetree_to_tour(
            map_full_adj, C_b, D_b, root=root, root_pair=map_root_pair, twoopt_passes=0,
        )
        info_counts["s2_repair_frac"] += float(used_repair2)
        raw_t, raw_c, lk_t, lk_c = _eval_seed(seed2, D_b, candidate_sets, root, twoopt_passes,
                                              C_mod_single=C_mod_for_lk)
        results["s2_map_repair"]["raw_tour"].append(raw_t)
        results["s2_map_repair"]["raw_cost"].append(raw_c)
        results["s2_map_repair"]["lk_tour"].append(lk_t)
        results["s2_map_repair"]["lk_cost"].append(lk_c)
        if lk_c < best_cost_b:
            best_cost_b, best_tour_b, best_strategy_b = lk_c, lk_t, "s2_map_repair"

        # ---------- S3: top-K root-pair conditional MAP trees ----------
        s3_best_raw, s3_best_lk = float("inf"), float("inf")
        s3_best_raw_tour: list[int] = []
        s3_best_lk_tour: list[int] = []
        if K > 0:
            selected_pairs = select_root_pair_candidates(
                C_b, root=root, pair_prob_single=pair_b, k=K,
            )
            # Compute the MAP non-root tree ONCE; it's independent of the root pair.
            map_tree_edges = map_nonroot_tree_edges_from_cmod(C_b, root=root)
            s3_repair_count = 0
            s3_branches = 0
            for root_pair in selected_pairs:
                s3_branches += 1
                full_adj, _ = assemble_rooted_onetree_from_tree_and_pair(
                    n, root, map_tree_edges, root_pair,
                )
                seed3, used_repair3 = repair_rooted_onetree_to_tour(
                    full_adj, C_b, D_b, root=root, root_pair=root_pair, twoopt_passes=0,
                )
                s3_repair_count += int(used_repair3)
                raw_t, raw_c, lk_t, lk_c = _eval_seed(seed3, D_b, candidate_sets, root, twoopt_passes,
                                                      C_mod_single=C_mod_for_lk)
                if raw_c < s3_best_raw:
                    s3_best_raw, s3_best_raw_tour = raw_c, raw_t
                if lk_c < s3_best_lk:
                    s3_best_lk, s3_best_lk_tour = lk_c, lk_t
            info_counts["s3_repair_frac"] += float(s3_repair_count) / float(max(1, s3_branches))
            info_counts["s3_num_pairs_mean"] += float(s3_branches)
        else:
            # If K=0, report NaN and skip.
            s3_best_raw = float("nan")
            s3_best_lk = float("nan")
        results["s3_rootpair_mapK"]["raw_tour"].append(s3_best_raw_tour)
        results["s3_rootpair_mapK"]["raw_cost"].append(s3_best_raw)
        results["s3_rootpair_mapK"]["lk_tour"].append(s3_best_lk_tour)
        results["s3_rootpair_mapK"]["lk_cost"].append(s3_best_lk)
        if math.isfinite(s3_best_lk) and s3_best_lk < best_cost_b:
            best_cost_b, best_tour_b, best_strategy_b = s3_best_lk, s3_best_lk_tour, "s3_rootpair_mapK"

        # ---------- S4: Gumbel mu-greedy (M draws) ----------
        s4_best_raw, s4_best_lk = float("inf"), float("inf")
        s4_best_raw_tour: list[int] = []
        s4_best_lk_tour: list[int] = []
        if M > 0:
            rng = np.random.default_rng(int(seed_base + 1000003 * b + 31))
            base_score = _mu_score_matrix(mu_b)
            seen_keys: set[tuple[int, ...]] = set()
            unique = 0
            for m in range(M):
                noise = _symmetric_gumbel_noise(n, rng)
                score_m = base_score + float(gumbel_scale) * noise
                np.fill_diagonal(score_m, -np.inf)
                seed4, _ = greedy_degree2_tour_from_scores(score_m, D_b, root=root)
                raw_t, raw_c, lk_t, lk_c = _eval_seed(seed4, D_b, candidate_sets, root, twoopt_passes,
                                                      C_mod_single=C_mod_for_lk)
                k_raw = _tour_key(raw_t)
                if k_raw not in seen_keys:
                    seen_keys.add(k_raw)
                    unique += 1
                if raw_c < s4_best_raw:
                    s4_best_raw, s4_best_raw_tour = raw_c, raw_t
                if lk_c < s4_best_lk:
                    s4_best_lk, s4_best_lk_tour = lk_c, lk_t
            info_counts["s4_unique_frac"] += float(unique) / float(max(1, M))
        else:
            s4_best_raw = float("nan")
            s4_best_lk = float("nan")
        results["s4_gumbel_muM"]["raw_tour"].append(s4_best_raw_tour)
        results["s4_gumbel_muM"]["raw_cost"].append(s4_best_raw)
        results["s4_gumbel_muM"]["lk_tour"].append(s4_best_lk_tour)
        results["s4_gumbel_muM"]["lk_cost"].append(s4_best_lk)
        if math.isfinite(s4_best_lk) and s4_best_lk < best_cost_b:
            best_cost_b, best_tour_b, best_strategy_b = s4_best_lk, s4_best_lk_tour, "s4_gumbel_muM"

        portfolio_costs.append(best_cost_b)
        portfolio_tours.append(best_tour_b)
        attribution.append(best_strategy_b)

    # Finalize.
    Bf = float(max(1, B))
    info = {
        "s1_fallback_frac": info_counts["s1_fallback_frac"] / Bf,
        "s2_repair_frac": info_counts["s2_repair_frac"] / Bf,
        "s3_repair_frac": info_counts["s3_repair_frac"] / Bf if K > 0 else 0.0,
        "s3_num_pairs_mean": info_counts["s3_num_pairs_mean"] / Bf if K > 0 else 0.0,
        "s4_unique_frac": info_counts["s4_unique_frac"] / Bf if M > 0 else 0.0,
        "s4_draws": float(M),
        "num_root_pairs": float(K),
        "num_gumbel_draws": float(M),
        "gumbel_scale": float(gumbel_scale),
        "twoopt_passes": float(twoopt_passes),
        "lk_alpha": float(bool(use_lk_alpha)),
    }

    out: Dict[str, np.ndarray | Dict] = {}
    for name in names:
        out[name] = {
            "raw_tour": np.array(results[name]["raw_tour"], dtype=object),
            "raw_cost": np.array(results[name]["raw_cost"], dtype=np.float64),
            "lk_tour": np.array(results[name]["lk_tour"], dtype=object),
            "lk_cost": np.array(results[name]["lk_cost"], dtype=np.float64),
        }
    out["portfolio"] = {
        "tour": np.array(portfolio_tours, dtype=object),
        "cost": np.array(portfolio_costs, dtype=np.float64),
    }
    out["attribution"] = np.array(attribution, dtype=object)
    out["info"] = info
    return out


def _generate_decode_noise(
        noise_type: str,
        n: int,
        root: int,
        rng: np.random.Generator,
        mu_b: np.ndarray | None = None,
        C_mod_b: np.ndarray | None = None,
        tau: float = 0.2,
) -> np.ndarray:
    """Generate symmetric (n, n) perturbation noise for one decode draw.

    Types:
      gumbel                 : symmetric Gumbel(0,1) — heavy right tail, original v9 behavior
      gaussian               : symmetric N(0,1) — no tail bias
      uncertainty            : N(0,1) scaled by mu*(1-mu) per edge — explores uncertain edges only
      dual                   : N(0,1) at node level, broadcast to edges as delta_i + delta_j
      covariance             : |phi_i - phi_j| with phi ~ N(0, L^{-1}) — Kirchhoff magnitude,
                               large on high effective resistance edges (topologically ambiguous).
                               Normalized to std 1 overall.
      covariance_uncertainty : covariance noise further weighted by sqrt(mu(1-mu)) per edge —
                               combines graph-topology scale (covariance) with model-uncertainty
                               targeting (only perturbs edges model is still unsure about).
    """
    if noise_type == "gumbel":
        return _symmetric_gumbel_noise(n, rng)
    elif noise_type == "gaussian":
        raw = rng.standard_normal((n, n))
        noise = 0.5 * (raw + raw.T)
        np.fill_diagonal(noise, 0.0)
        return noise
    elif noise_type == "uncertainty":
        raw = rng.standard_normal((n, n))
        noise = 0.5 * (raw + raw.T)
        np.fill_diagonal(noise, 0.0)
        if mu_b is not None:
            mu_sym = 0.5 * (mu_b + mu_b.T)
            unc = np.clip(mu_sym, 0.0, 1.0) * (1.0 - np.clip(mu_sym, 0.0, 1.0))
            # Normalize so max uncertainty edge gets scale 1.0
            unc_max = unc.max()
            if unc_max > 1e-12:
                unc = unc / unc_max
            noise = noise * unc
        return noise
    elif noise_type == "dual":
        delta = rng.standard_normal(n)
        delta[root] = 0.0
        noise = delta[:, None] + delta[None, :]
        np.fill_diagonal(noise, 0.0)
        return noise
    elif noise_type == "covariance":
        return _covariance_noise(n, root, rng, C_mod_b=C_mod_b, tau=tau,
                                 mu_b=mu_b, uncertainty_weight=False)
    elif noise_type == "covariance_uncertainty":
        return _covariance_noise(n, root, rng, C_mod_b=C_mod_b, tau=tau,
                                 mu_b=mu_b, uncertainty_weight=True)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type!r}. "
                         f"Use gumbel/gaussian/uncertainty/dual/covariance/covariance_uncertainty.")


def _covariance_noise(
        n: int,
        root: int,
        rng: np.random.Generator,
        C_mod_b: np.ndarray | None = None,
        tau: float = 0.2,
        mu_b: np.ndarray | None = None,
        uncertainty_weight: bool = False,
) -> np.ndarray:
    """Sample structurally coherent noise from the 1-tree Gibbs covariance.

    The weighted Laplacian L on V\\{root} defines a Gaussian field
        phi ~ N(0, L^{-1}), Cov(phi_i, phi_j) = L^{-1}_{ij}.
    Edge perturbations are the Kirchhoff form noise_ij = phi_i - phi_j, with
        Var(noise_ij) = R^eff_ij = L^{-1}_{ii} + L^{-1}_{jj} - 2 L^{-1}_{ij}
        Cov(noise_ij, noise_ik) = L^{-1}_{ii} - L^{-1}_{ij} - L^{-1}_{ik} + L^{-1}_{jk}
    i.e. competing edges at a shared vertex get anti-correlated noise, and
    magnitude scales with effective resistance.

    The decoder expects symmetric noise matrices, so we emit |phi_i - phi_j|.
    This preserves the resistance-scaled magnitude structure (large on
    topologically ambiguous edges) but loses the anti-correlation sign. For
    best-of-K sampling via score perturbation, magnitude-based exploration
    is what matters — the sign is washed out by the max anyway.

    Fixes in v17+ vs v16:
      - noise = |phi_i - phi_j| (Kirchhoff magnitude), NOT phi_i + phi_j
        (which was node-additive and absorbed by lambda, carrying no edge
        signal).
      - No std-normalization on phi: the heterogeneous per-node variances
        ARE the structural signal. Normalizing phi to unit std destroys it.
      - L is rescaled by its mean diagonal to keep Cholesky well-conditioned
        across cost scales, without flattening relative variances.
      - Final noise matrix is normalized to unit std overall (not per-node),
        so gumbel_scale retains interpretable magnitude across noise types.

    Optional: set uncertainty_weight=True to multiply by sqrt(mu(1-mu)), 
    concentrating structural noise on edges the model is still uncertain 
    about. Combines graph topology with marginal uncertainty.

    Cost: one Cholesky O((n-1)^3) + one triangular solve O((n-1)^2).
    """
    if C_mod_b is None:
        # Fallback to gaussian if C_mod not available
        raw = rng.standard_normal((n, n))
        noise = 0.5 * (raw + raw.T)
        np.fill_diagonal(noise, 0.0)
        return noise

    tau_safe = max(float(tau), 1e-6)
    m = n - 1

    nonroot = [i for i in range(n) if i != root]
    nonroot_arr = np.array(nonroot)

    # Build weighted Laplacian on V\{root}, grounded by root-edge weights on
    # diagonal (standard non-root rooted-spanning-tree Laplacian).
    C_sym = 0.5 * (C_mod_b + C_mod_b.T)
    C_sub = C_sym[np.ix_(nonroot_arr, nonroot_arr)]
    W_sub = np.exp(-C_sub / tau_safe)
    W_sub = np.clip(W_sub, 1e-30, 1e30)
    np.fill_diagonal(W_sub, 0.0)
    L = np.diag(W_sub.sum(axis=1)) - W_sub
    root_w = np.exp(-C_sym[nonroot_arr, root] / tau_safe)
    root_w = np.clip(root_w, 1e-30, 1e30)
    L = L + np.diag(root_w)

    # Rescale L by its mean diagonal: preserves relative per-node variance,
    # keeps Cholesky numerically stable across cost magnitudes.
    L_diag_mean = max(float(np.mean(np.diag(L))), 1e-8)
    L_scaled = L / L_diag_mean

    # Cholesky: L_scaled = R^T R
    try:
        R = np.linalg.cholesky(L_scaled + 1e-6 * np.eye(m))
    except np.linalg.LinAlgError:
        raw = rng.standard_normal((n, n))
        noise = 0.5 * (raw + raw.T)
        np.fill_diagonal(noise, 0.0)
        return noise

    # Sample phi ~ N(0, L_scaled^{-1}):  phi = R^{-T} z,  z ~ N(0, I)
    z = rng.standard_normal(m)
    phi_local = np.linalg.solve(R.T, z)

    # Embed back to full n nodes (root gets phi=0 by grounding)
    phi = np.zeros(n, dtype=np.float64)
    phi[nonroot_arr] = phi_local

    # Symmetric Kirchhoff-magnitude edge noise:
    noise = np.abs(phi[:, None] - phi[None, :])
    np.fill_diagonal(noise, 0.0)

    # Optional uncertainty weighting: concentrate on edges model is uncertain about
    if uncertainty_weight and mu_b is not None:
        mu_sym = 0.5 * (mu_b + mu_b.T)
        mu_c = np.clip(mu_sym, 0.0, 1.0)
        unc = mu_c * (1.0 - mu_c)
        unc_max = float(unc.max()) if unc.size else 0.0
        if unc_max > 1e-10:
            noise = noise * np.sqrt(unc / unc_max)
        np.fill_diagonal(noise, 0.0)

    # Normalize so overall noise std is 1 — keeps gumbel_scale comparable
    # across noise types. Relative structure (what carries signal) is preserved.
    offdiag = noise[~np.eye(n, dtype=bool)]
    ns_std = float(offdiag.std())
    if ns_std > 1e-10:
        noise = noise / ns_std
    np.fill_diagonal(noise, 0.0)
    return noise


# =========================================================================
# v17+: Hybrid CPU/GPU decoder — batched noise + batched Prim's on GPU,
# per-instance greedy/repair/cycle-extraction on CPU.
# =========================================================================

def _batched_prim_gpu(cost: torch.Tensor, root: int = 0) -> torch.Tensor:
    """Batched Prim's MST on GPU.

    Args:
      cost: (B, n, n) symmetric cost matrix on GPU.
      root: starting vertex for Prim's.

    Returns:
      adj: (B, n, n) int64 adjacency in {0,1} on GPU. Each row corresponds to
           the MST of one instance. Diagonal is zero.

    Complexity: O(n) GPU kernel launches, each doing O(B·n) elementwise work.
    Replaces B serial CPU Prim's calls with n batched GPU ops.
    """
    B, n, _ = cost.shape
    device = cost.device
    INF = float("inf")
    in_tree = torch.zeros(B, n, dtype=torch.bool, device=device)
    parent = torch.full((B, n), -1, dtype=torch.long, device=device)
    best = torch.full((B, n), INF, device=device, dtype=cost.dtype)
    best[:, root] = 0.0

    batch_idx = torch.arange(B, device=device)

    for _ in range(n):
        masked = torch.where(in_tree, torch.full_like(best, INF), best)
        u = masked.argmin(dim=1)  # (B,)
        in_tree[batch_idx, u] = True
        c_u = cost[batch_idx, u]  # (B, n) — row u for each instance
        # Mask already-in-tree vertices out of the update
        better = (~in_tree) & (c_u < best)
        parent = torch.where(better, u.unsqueeze(1).expand_as(parent), parent)
        best = torch.where(better, c_u, best)

    # Build adjacency from parent pointers
    adj = torch.zeros(B, n, n, dtype=torch.long, device=device)
    v_idx = torch.arange(n, device=device).unsqueeze(0).expand(B, -1)  # (B, n)
    valid = parent >= 0  # (B, n) bool
    # Flatten valid entries for scatter
    b_flat = batch_idx.unsqueeze(1).expand(-1, n)[valid]  # (K,)
    v_flat = v_idx[valid]  # (K,)
    p_flat = parent[valid]  # (K,)
    adj[b_flat, p_flat, v_flat] = 1
    adj[b_flat, v_flat, p_flat] = 1
    return adj


def _batched_noise_gpu(
        noise_type: str,
        M: int,
        B: int,
        n: int,
        root: int,
        mu_b: torch.Tensor,  # (B, n, n) marginals
        C_mod_b: torch.Tensor,  # (B, n, n) reduced cost
        tau: float,
        seed: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate all (M, B) noise matrices on GPU.

    Returns: (M, B, n, n) noise tensor. Draw 0 is the zero-noise draw; draws
    1..M-1 are sampled.

    Noise type semantics match _generate_decode_noise for apples-to-apples
    comparability. Per-instance normalization (std=1) is applied on GPU.
    """
    # We use a separate generator to ensure reproducibility without perturbing
    # global torch rng state.
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    out = torch.zeros(M, B, n, n, device=device, dtype=dtype)
    eye_off = ~torch.eye(n, dtype=torch.bool, device=device)  # off-diagonal mask

    if noise_type == "gumbel":
        # Symmetric Gumbel(0,1) via -log(-log(U)), symmetrize
        # Only generate M-1 draws (draw 0 is zero)
        u_raw = torch.rand(M - 1, B, n, n, device=device, generator=g, dtype=dtype)
        u_raw = u_raw.clamp(min=1e-12, max=1 - 1e-12)
        raw = -torch.log(-torch.log(u_raw))
        noise = 0.5 * (raw + raw.transpose(-1, -2))
        out[1:] = noise
    elif noise_type == "gaussian":
        raw = torch.randn(M - 1, B, n, n, device=device, generator=g, dtype=dtype)
        noise = 0.5 * (raw + raw.transpose(-1, -2))
        out[1:] = noise
    elif noise_type == "uncertainty":
        raw = torch.randn(M - 1, B, n, n, device=device, generator=g, dtype=dtype)
        noise = 0.5 * (raw + raw.transpose(-1, -2))
        mu_sym = 0.5 * (mu_b + mu_b.transpose(-1, -2))  # (B, n, n)
        mu_c = mu_sym.clamp(0.0, 1.0)
        unc = mu_c * (1.0 - mu_c)  # (B, n, n)
        unc_max = unc.amax(dim=(-1, -2), keepdim=True).clamp(min=1e-12)
        unc_scale = unc / unc_max  # (B, n, n) in [0,1]
        noise = noise * unc_scale.unsqueeze(0)  # broadcast over M
        out[1:] = noise
    elif noise_type == "dual":
        delta = torch.randn(M - 1, B, n, device=device, generator=g, dtype=dtype)
        delta[..., root] = 0.0
        noise = delta.unsqueeze(-1) + delta.unsqueeze(-2)
        out[1:] = noise
    elif noise_type in ("covariance", "covariance_uncertainty"):
        # Batched Cholesky over B Laplacians.
        tau_safe = max(float(tau), 1e-6)
        # Build non-root index
        nonroot = [i for i in range(n) if i != root]
        nr_idx = torch.tensor(nonroot, device=device, dtype=torch.long)
        m = n - 1
        # C_mod_b symmetrized and indexed on non-root
        C_sym = 0.5 * (C_mod_b + C_mod_b.transpose(-1, -2))  # (B, n, n)
        # Gather non-root submatrix
        C_sub = C_sym.index_select(-2, nr_idx).index_select(-1, nr_idx)  # (B, m, m)
        # Weights (clip exponent to avoid overflow)
        exponent = (-C_sub / tau_safe).clamp(-60.0, 60.0)
        W_sub = torch.exp(exponent)  # (B, m, m)
        # Zero diagonal
        W_sub = W_sub * (1.0 - torch.eye(m, device=device, dtype=dtype))
        # Laplacian
        L = torch.diag_embed(W_sub.sum(dim=-1)) - W_sub  # (B, m, m)
        # Add root-edge diagonal
        root_exponent = (-C_sym[:, nr_idx, root] / tau_safe).clamp(-60.0, 60.0)
        root_w = torch.exp(root_exponent)  # (B, m)
        L = L + torch.diag_embed(root_w)
        # Rescale by mean diagonal
        L_diag_mean = L.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
        L_diag_mean = L_diag_mean.clamp(min=1e-8)
        L_scaled = L / L_diag_mean
        # Cholesky with ridge
        ridge = 1e-6 * torch.eye(m, device=device, dtype=dtype)
        try:
            R = torch.linalg.cholesky(L_scaled + ridge)  # (B, m, m)
        except Exception:
            # Fallback to Gaussian noise on failure
            fallback_raw = torch.randn(M - 1, B, n, n, device=device, generator=g, dtype=dtype)
            fallback = 0.5 * (fallback_raw + fallback_raw.transpose(-1, -2))
            out[1:] = fallback
            return out
        # Sample phi: R^T phi = z,  z ~ N(0, I). Batched triangular solve.
        z = torch.randn(M - 1, B, m, device=device, generator=g, dtype=dtype)  # (M-1, B, m)
        # Need to solve R.transpose @ phi = z per batch, broadcast over M-1 draws
        # torch.linalg.solve_triangular: (B, m, m) @ (B, m, K) -> (B, m, K)
        # Flatten M-1 draws into the rhs dim
        z_flat = z.permute(1, 2, 0).contiguous()  # (B, m, M-1)
        phi_flat = torch.linalg.solve_triangular(
            R.transpose(-1, -2), z_flat, upper=True, unitriangular=False
        )  # (B, m, M-1)
        phi = phi_flat.permute(2, 0, 1)  # (M-1, B, m)
        # Embed back to full n
        phi_full = torch.zeros(M - 1, B, n, device=device, dtype=dtype)
        phi_full[:, :, nr_idx] = phi  # root stays 0
        # Kirchhoff magnitude: |phi_i - phi_j|
        noise = torch.abs(phi_full.unsqueeze(-1) - phi_full.unsqueeze(-2))  # (M-1, B, n, n)
        # Optional uncertainty weighting
        if noise_type == "covariance_uncertainty":
            mu_sym = 0.5 * (mu_b + mu_b.transpose(-1, -2))
            mu_c = mu_sym.clamp(0.0, 1.0)
            unc = mu_c * (1.0 - mu_c)
            unc_max = unc.amax(dim=(-1, -2), keepdim=True).clamp(min=1e-12)
            unc_scale = torch.sqrt(unc / unc_max)  # (B, n, n)
            noise = noise * unc_scale.unsqueeze(0)
        out[1:] = noise
    else:
        raise ValueError(f"Unknown noise_type for GPU path: {noise_type!r}")

    # Zero diagonals
    out = out * eye_off.unsqueeze(0).unsqueeze(0)
    # Per-(M,B) std-normalize: compute std over off-diagonal entries
    # shape (M, B, 1, 1)
    flat = out.reshape(M, B, -1)
    # off-diag count per instance
    od_count = int(eye_off.sum().item())
    # variance of off-diag entries per (m, b)
    # Use the mask: sum of squares over off-diag / od_count
    sq = (out * out).sum(dim=(-1, -2)) / max(od_count, 1)  # (M, B)
    sd = sq.clamp(min=1e-20).sqrt()  # (M, B)
    # Avoid dividing draw 0 by (effectively 0) — guard
    sd = sd.where(sd > 1e-10, torch.ones_like(sd))
    out = out / sd.unsqueeze(-1).unsqueeze(-1)
    # Re-zero diagonal (the division may have left tiny noise on diagonal)
    out = out * eye_off.unsqueeze(0).unsqueeze(0)
    # Draw 0 stays zero — re-zero it to be safe
    out[0] = 0.0
    return out


# ---------------------------------------------------------------------------
# v19 decode parallelization: per-instance CPU work extracted to a top-level
# function so it can be sent to a ProcessPoolExecutor.
# ---------------------------------------------------------------------------
def _decode_single_instance_cpu(
        bi: int,
        M: int,
        n: int,
        root: int,
        twoopt_passes: int,
        seed_split: float,
        use_lk_alpha: bool,
        use_mu_repair: bool,
        mu_b: np.ndarray,
        D_b: np.ndarray,
        scores_bi: np.ndarray,  # (M, n, n)
        C_mod_bi: np.ndarray,  # (M, n, n)
        full_adj_bi: np.ndarray,  # (M, n, n)
        root_pair_bi: np.ndarray,  # (M, 2)
        cand_bi: np.ndarray | None,  # (n, n) or None
) -> dict:
    """Process a single batch instance's M Gumbel draws; pickle-safe.

    Mirrors the body of the ``for bi in range(B)`` loop in
    :func:`decode_gumbel_hybrid`.  All captured closures (``_try_seed``) have
    been flattened out so the function stands alone.
    """
    # Build candidate sets for LK
    if cand_bi is not None:
        candidate_sets: list[set[int]] = [
            {int(j) for j in range(n) if j != i and cand_bi[i, j]}
            for i in range(n)
        ]
    else:
        full = set(range(n))
        candidate_sets = [full - {i} for i in range(n)]
    C_mod_for_lk = C_mod_bi[0] if use_lk_alpha else None

    all_costs: list[float] = []
    a_costs: list[float] = []
    b_costs: list[float] = []
    best_cost_b = float("inf")
    best_tour_b: list[int] = []
    best_a_cost = float("inf")
    best_b_cost = float("inf")
    det_cost_b = float("inf")
    seen_keys: set[tuple[int, ...]] = set()

    def try_seed_local(seed_tour: list[int]) -> float:
        nonlocal best_cost_b, best_tour_b
        _, _, lk_tour, lk_cost = _eval_seed(
            seed_tour, D_b, candidate_sets, root, twoopt_passes,
            C_mod_single=C_mod_for_lk,
        )
        all_costs.append(lk_cost)
        seen_keys.add(_tour_key(lk_tour))
        if lk_cost < best_cost_b:
            best_cost_b = lk_cost
            best_tour_b = lk_tour
        return lk_cost

    # Seed-split routing
    run_a_for_draw = [False] * M
    run_b_for_draw = [False] * M
    if seed_split < 0.0:
        for m_i in range(M):
            run_a_for_draw[m_i] = True
            run_b_for_draw[m_i] = True
    else:
        s = max(0.0, min(1.0, float(seed_split)))
        n_a = int(round(s * M))
        if n_a > 0 and n_a < M:
            run_a_for_draw[0] = True
            run_b_for_draw[0] = True
            remaining_a = n_a - 1
            remaining_b = M - 1 - remaining_a
            for m_i in range(1, M):
                if remaining_a > 0 and (remaining_b == 0 or
                                        ((m_i - 1) * n_a) % M < ((m_i) * n_a) % M):
                    run_a_for_draw[m_i] = True
                    remaining_a -= 1
                else:
                    run_b_for_draw[m_i] = True
                    remaining_b -= 1
        elif n_a == 0:
            for m_i in range(M):
                run_b_for_draw[m_i] = True
        else:
            for m_i in range(M):
                run_a_for_draw[m_i] = True

    for m_i in range(M):
        ca = float("inf")
        cb = float("inf")

        if run_a_for_draw[m_i]:
            score_m = scores_bi[m_i].copy()
            np.fill_diagonal(score_m, -np.inf)
            seed_a, _ = greedy_degree2_tour_from_scores(score_m, D_b, root=root)
            ca = try_seed_local(seed_a)
            a_costs.append(ca)
            if ca < best_a_cost:
                best_a_cost = ca

        if run_b_for_draw[m_i]:
            full_adj_b = full_adj_bi[m_i]
            root_pair = (int(root_pair_bi[m_i, 0]), int(root_pair_bi[m_i, 1]))
            C_mod_m = C_mod_bi[m_i]
            seed_b, _ = repair_rooted_onetree_to_tour(
                full_adj_b, C_mod_m, D_b,
                root=root, root_pair=root_pair, twoopt_passes=0,
                mu_single=(mu_b if use_mu_repair else None),
            )
            cb = try_seed_local(seed_b)
            b_costs.append(cb)
            if cb < best_b_cost:
                best_b_cost = cb

        if m_i == 0:
            det_cost_b = min(ca, cb)

    total_candidates = max(1, len(all_costs))
    return {
        "bi": bi,
        "mean_cost": float(np.mean(all_costs)) if all_costs else float("inf"),
        "best_cost": best_cost_b,
        "best_tour": best_tour_b,
        "unique_frac": float(len(seen_keys)) / float(total_candidates),
        "a_mean_cost": float(np.mean(a_costs)) if a_costs else float("nan"),
        "a_best_cost": best_a_cost if a_costs else float("nan"),
        "b_mean_cost": float(np.mean(b_costs)) if b_costs else float("nan"),
        "b_best_cost": best_b_cost if b_costs else float("nan"),
        "det_cost": det_cost_b,
    }


# Module-level lazy process pool for decode parallelization.  Created on first
# use via ``_get_decode_executor`` and shut down at process exit.
_DECODE_EXECUTOR = None
_DECODE_EXECUTOR_WORKERS = 0


def _get_decode_executor(num_workers: int):
    """Return a persistent ProcessPoolExecutor, or None if num_workers<=1."""
    global _DECODE_EXECUTOR, _DECODE_EXECUTOR_WORKERS
    if num_workers is None or num_workers <= 1:
        return None
    if _DECODE_EXECUTOR is not None and _DECODE_EXECUTOR_WORKERS == num_workers:
        return _DECODE_EXECUTOR
    # Worker count changed or first call; (re)create.
    if _DECODE_EXECUTOR is not None:
        try:
            _DECODE_EXECUTOR.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
    import concurrent.futures as _cf
    import multiprocessing as _mp
    # On Linux, fork is fastest (no re-import cost, COW memory).  Workers here
    # never touch CUDA, so fork-after-CUDA-init is safe in our use.  If the
    # user reports deadlocks, switch to "spawn" by overriding the context.
    try:
        ctx = _mp.get_context("fork")
    except ValueError:
        ctx = _mp.get_context()
    _DECODE_EXECUTOR = _cf.ProcessPoolExecutor(max_workers=int(num_workers), mp_context=ctx)
    _DECODE_EXECUTOR_WORKERS = int(num_workers)
    import atexit as _atexit
    _atexit.register(lambda: _DECODE_EXECUTOR.shutdown(wait=False) if _DECODE_EXECUTOR is not None else None)
    return _DECODE_EXECUTOR


def decode_gumbel_hybrid(
        mu: torch.Tensor,
        C_mod: torch.Tensor,
        cand_mask: torch.Tensor | None,
        D: torch.Tensor,
        root: int = 0,
        twoopt_passes: int = 6,
        num_draws: int = 20,
        gumbel_scale: float = 0.20,
        seed_base: int = 0,
        use_lk_alpha: bool = False,
        noise_type: str = "gumbel",
        tau: float = 0.2,
        seed_split: float = -1.0,
        use_mu_repair: bool = True,
        report_timing: bool = False,
        num_workers: int = 1,
) -> Dict[str, np.ndarray | float]:
    r"""Hybrid GPU/CPU decoder. Same interface as decode_gumbel but parallelizes
    noise generation, perturbation, and Prim's MSP on GPU across (M, B) at once.

    Per-instance logic (greedy extraction, repair, cycle extraction, best-of-M
    tracking) remains sequential on CPU but receives precomputed inputs, so the
    heavy compute (Cholesky + Prim's + perturbation) is amortized.

    Returns same dict as decode_gumbel, with an added "_timing" key if
    report_timing=True: {"gpu_s": ..., "cpu_s": ..., "total_s": ...}.
    """
    device = mu.device
    dtype = mu.dtype if mu.is_floating_point() else torch.float32
    B, n = mu.shape[0], mu.shape[-1]
    M = max(1, int(num_draws))

    t0 = time.perf_counter() if report_timing else 0.0

    # ---------- GPU PHASE ----------
    # (1) Batched noise: (M, B, n, n)
    noise_all = _batched_noise_gpu(
        noise_type=noise_type,
        M=M, B=B, n=n, root=root,
        mu_b=mu.to(dtype), C_mod_b=C_mod.to(dtype),
        tau=tau, seed=seed_base, device=device, dtype=dtype,
    )
    # (2) Build Seed B perturbed costs: (M, B, n, n)
    C_mod_perturbed = C_mod.to(dtype).unsqueeze(0) + float(gumbel_scale) * noise_all
    # Symmetrize to guarantee Prim input is symmetric
    C_mod_perturbed = 0.5 * (C_mod_perturbed + C_mod_perturbed.transpose(-1, -2))
    # Zero diagonal
    eye_n = torch.eye(n, device=device, dtype=dtype)
    C_mod_perturbed = C_mod_perturbed * (1.0 - eye_n)

    # (3) Batched Prim's on non-root submatrix per draw
    # Reshape (M, B, n, n) -> (M*B, n, n) and run Prim's on full matrices
    # but we need to mimic "prim on non-root, then add root pair" like map_rooted_onetree_from_cmod.
    # Simpler: run Prim's on full (n, n) excluding root by setting row/col root to inf.
    # Then add root pair (top-2 nearest from root's perspective) afterwards on CPU.
    # Let's do the MSP on non-root subgraph: extract non-root x non-root submatrix.
    nonroot_idx = torch.tensor([i for i in range(n) if i != root], device=device, dtype=torch.long)
    m_size = n - 1
    C_nr = C_mod_perturbed.index_select(-2, nonroot_idx).index_select(-1, nonroot_idx)
    # Flatten (M, B) for Prim
    C_nr_flat = C_nr.reshape(M * B, m_size, m_size)
    adj_nr_flat = _batched_prim_gpu(C_nr_flat, root=0)  # (M*B, m_size, m_size)
    adj_nr = adj_nr_flat.reshape(M, B, m_size, m_size)

    # (4) Root pair: top-2 nearest vertices from root under perturbed cost
    # For each (m, b), argsort C_mod_perturbed[m, b, root, non-root] and take top 2.
    root_costs = C_mod_perturbed[:, :, root, :].index_select(-1, nonroot_idx)  # (M, B, m_size)
    _, root_order = torch.sort(root_costs, dim=-1)  # (M, B, m_size)
    root_pair_local = root_order[..., :2]  # (M, B, 2) — indices into nonroot
    # Map back to full n indices
    root_pair_full = nonroot_idx[root_pair_local]  # (M, B, 2)

    # (5) Seed A scores: log(mu) + noise, precomputed on GPU
    #     Handle mu=0 safely via log(mu + eps)
    EPS_MU = 1e-12
    base_score = torch.log(mu.to(dtype).clamp(min=EPS_MU))  # (B, n, n)
    scores_perturbed = base_score.unsqueeze(0) + float(gumbel_scale) * noise_all
    # Symmetrize scores (matches CPU path)
    scores_perturbed = 0.5 * (scores_perturbed + scores_perturbed.transpose(-1, -2))

    # Transfer to CPU as numpy in one shot
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_gpu_end = time.perf_counter() if report_timing else 0.0

    adj_nr_cpu = adj_nr.cpu().numpy().astype(np.int64)  # (M, B, m_size, m_size)
    root_pair_cpu = root_pair_full.cpu().numpy().astype(np.int64)  # (M, B, 2)
    scores_cpu = scores_perturbed.cpu().numpy()  # (M, B, n, n)
    C_mod_cpu = C_mod_perturbed.cpu().numpy()  # (M, B, n, n)
    mu_np = mu.detach().cpu().numpy()
    D_np = D.detach().cpu().numpy()
    cand_np = None if cand_mask is None else cand_mask.detach().cpu().numpy().astype(bool)

    # Build full_adj on CPU from non-root adj + root pair edges
    # (need full (n, n) adjacency for repair function)
    full_adj_all = np.zeros((M, B, n, n), dtype=np.int64)
    nonroot_np = np.array([i for i in range(n) if i != root], dtype=np.int64)
    # Scatter non-root adjacency into full matrix
    # adj_nr_cpu is indexed by local non-root indices; map to global
    for mi in range(M):
        for bi in range(B):
            nr = adj_nr_cpu[mi, bi]  # (m_size, m_size)
            rows, cols = np.where(np.triu(nr, k=1) > 0)
            for i, j in zip(rows.tolist(), cols.tolist()):
                u = int(nonroot_np[i])
                v = int(nonroot_np[j])
                full_adj_all[mi, bi, u, v] = 1
                full_adj_all[mi, bi, v, u] = 1
            # Add root pair edges
            a, b = int(root_pair_cpu[mi, bi, 0]), int(root_pair_cpu[mi, bi, 1])
            full_adj_all[mi, bi, root, a] = 1
            full_adj_all[mi, bi, a, root] = 1
            full_adj_all[mi, bi, root, b] = 1
            full_adj_all[mi, bi, b, root] = 1

    # ---------- CPU PHASE ----------
    mean_costs: list[float] = [0.0] * B
    best_costs: list[float] = [0.0] * B
    best_tours: list[list[int]] = [[] for _ in range(B)]
    unique_fracs: list[float] = [0.0] * B
    a_mean_costs: list[float] = [0.0] * B
    a_best_costs: list[float] = [0.0] * B
    b_mean_costs: list[float] = [0.0] * B
    b_best_costs: list[float] = [0.0] * B
    det_costs: list[float] = [0.0] * B

    executor = _get_decode_executor(num_workers) if num_workers and num_workers > 1 else None

    def _per_instance_args(bi: int):
        return dict(
            bi=bi,
            M=M, n=n, root=root,
            twoopt_passes=twoopt_passes,
            seed_split=seed_split,
            use_lk_alpha=use_lk_alpha,
            use_mu_repair=use_mu_repair,
            mu_b=mu_np[bi],
            D_b=D_np[bi],
            scores_bi=scores_cpu[:, bi],  # (M, n, n)
            C_mod_bi=C_mod_cpu[:, bi],  # (M, n, n)
            full_adj_bi=full_adj_all[:, bi],  # (M, n, n)
            root_pair_bi=root_pair_cpu[:, bi],  # (M, 2)
            cand_bi=(None if cand_np is None else cand_np[bi]),
        )

    def _gather(res: dict):
        bi = int(res["bi"])
        mean_costs[bi] = res["mean_cost"]
        best_costs[bi] = res["best_cost"]
        best_tours[bi] = res["best_tour"]
        unique_fracs[bi] = res["unique_frac"]
        a_mean_costs[bi] = res["a_mean_cost"]
        a_best_costs[bi] = res["a_best_cost"]
        b_mean_costs[bi] = res["b_mean_cost"]
        b_best_costs[bi] = res["b_best_cost"]
        det_costs[bi] = res["det_cost"]

    if executor is None:
        # Serial path (original behavior).
        for bi in range(B):
            res = _decode_single_instance_cpu(**_per_instance_args(bi))
            _gather(res)
    else:
        # Parallel path via persistent ProcessPoolExecutor.  The slice dict
        # passed per bi is the only payload pickled to workers; we avoid sending
        # the full (M, B, n, n) arrays by indexing ahead of submit().
        futures = [
            executor.submit(_decode_single_instance_cpu, **_per_instance_args(bi))
            for bi in range(B)
        ]
        for fut in futures:
            _gather(fut.result())

    t_total_end = time.perf_counter() if report_timing else 0.0

    result: Dict[str, np.ndarray | float] = {
        "mean_cost": np.array(mean_costs, dtype=np.float64),
        "best_cost": np.array(best_costs, dtype=np.float64),
        "det_cost": np.array(det_costs, dtype=np.float64),
        "best_tour": np.array(best_tours, dtype=object),
        "unique_frac": float(np.mean(unique_fracs)),
        "num_draws": M,
        "gumbel_scale": float(gumbel_scale),
        "noise_type": str(noise_type),
        "a_mean_cost": np.array(a_mean_costs, dtype=np.float64),
        "a_best_cost": np.array(a_best_costs, dtype=np.float64),
        "b_mean_cost": np.array(b_mean_costs, dtype=np.float64),
        "b_best_cost": np.array(b_best_costs, dtype=np.float64),
    }
    if report_timing:
        result["_timing"] = {
            "gpu_s": t_gpu_end - t0,
            "cpu_s": t_total_end - t_gpu_end,
            "total_s": t_total_end - t0,
        }
    return result


def decode_gumbel(
        mu: torch.Tensor,
        C_mod: torch.Tensor,
        cand_mask: torch.Tensor | None,
        D: torch.Tensor,
        root: int = 0,
        twoopt_passes: int = 6,
        num_draws: int = 20,
        gumbel_scale: float = 0.20,
        seed_base: int = 0,
        use_lk_alpha: bool = False,
        noise_type: str = "gumbel",
        tau: float = 0.2,
        seed_split: float = -1.0,
        use_mu_repair: bool = True,
) -> Dict[str, np.ndarray | float]:
    r"""Dual-seed decoder with configurable noise: mu-greedy + MAP-repair per draw.

    For each of M draws, the same symmetric noise matrix perturbs both paths:
      - score path:  log(mu) + scale * noise  → greedy degree-2 → tour_a  (Seed A)
      - C_mod path:  C_mod   + scale * noise  → MAP 1-tree → μ-weighted repair → tour_b  (Seed B)

    Draw 0 is always deterministic (no noise). All tours optionally LK-improved.

    seed_split controls sample allocation across the two seeds:
      seed_split = -1.0 (default): run BOTH A and B on every draw (legacy). Total tours = 2M.
      seed_split = 0.0: run only Seed B on every draw. Total tours = M.
      seed_split = 1.0: run only Seed A on every draw. Total tours = M.
      seed_split in (0,1): fraction of draws allocated to A; remainder to B.
                           E.g. 0.3 means 30% A, 70% B. Total tours = M.

    use_mu_repair controls whether Seed B uses μ-weighted matching repair (prior-faithful)
    or falls back to backbone+cheapest-insertion repair (legacy, C_mod-based).
    Default True (prior-faithful).
    """
    mu_np = mu.detach().cpu().numpy()
    C_np = C_mod.detach().cpu().numpy()
    cand_np = None if cand_mask is None else cand_mask.detach().cpu().numpy().astype(bool)
    D_np = D.detach().cpu().numpy()
    B, n, _ = D_np.shape
    M = max(1, int(num_draws))

    mean_costs: list[float] = []
    best_costs: list[float] = []
    best_tours: list[list[int]] = []
    unique_fracs: list[float] = []
    a_mean_costs: list[float] = []
    a_best_costs: list[float] = []
    b_mean_costs: list[float] = []
    b_best_costs: list[float] = []
    det_costs: list[float] = []

    for b in range(B):
        mu_b = mu_np[b]
        D_b = D_np[b]
        C_mod_b = C_np[b]
        C_mod_for_lk = C_mod_b if use_lk_alpha else None

        # Build candidate sets for LK
        candidate_sets: list[set[int]] = []
        if cand_np is not None:
            for i in range(n):
                candidate_sets.append({int(j) for j in range(n) if j != i and cand_np[b, i, j]})
        else:
            full = set(range(n))
            candidate_sets = [full - {i} for i in range(n)]

        rng = np.random.default_rng(int(seed_base + 1000003 * b + 31))
        base_score = _mu_score_matrix(mu_b)

        all_costs: list[float] = []
        a_costs: list[float] = []
        b_costs: list[float] = []
        best_cost_b = float("inf")
        best_tour_b: list[int] = []
        best_a_cost = float("inf")
        best_b_cost = float("inf")
        det_cost_b = float("inf")  # deterministic (no-noise) best of A+B
        seen_keys: set[tuple[int, ...]] = set()

        def _try_seed(seed_tour: list[int]) -> float:
            nonlocal best_cost_b, best_tour_b
            _, _, lk_tour, lk_cost = _eval_seed(
                seed_tour, D_b, candidate_sets, root, twoopt_passes,
                C_mod_single=C_mod_for_lk,
            )
            all_costs.append(lk_cost)
            seen_keys.add(_tour_key(lk_tour))
            if lk_cost < best_cost_b:
                best_cost_b = lk_cost
                best_tour_b = lk_tour
            return lk_cost

        # Determine which seeds each draw will exercise based on seed_split.
        #   seed_split < 0  → legacy behavior: both A and B every draw
        #   seed_split == 0 → only B
        #   seed_split == 1 → only A
        #   0 < s < 1       → fraction s of draws do A, rest do B. Deterministic
        #                     interleaving (no randomness in allocation).
        run_a_for_draw: list[bool] = [False] * M
        run_b_for_draw: list[bool] = [False] * M
        if seed_split < 0.0:
            for m in range(M):
                run_a_for_draw[m] = True
                run_b_for_draw[m] = True
        else:
            s = max(0.0, min(1.0, float(seed_split)))
            n_a = int(round(s * M))
            # Deterministic interleave: assign the first n_a draws to A, rest to B.
            # Draw 0 (deterministic, no noise) is always run on whichever seed is primary.
            # To preserve the "best-of-both for draw 0" invariant as much as possible:
            # if both seeds get nonzero allocation, run both on draw 0.
            if n_a > 0 and n_a < M:
                run_a_for_draw[0] = True
                run_b_for_draw[0] = True
                # Allocate remaining draws deterministically
                remaining_a = n_a - 1
                remaining_b = M - 1 - remaining_a
                for m in range(1, M):
                    # Alternate based on ratio
                    if remaining_a > 0 and (remaining_b == 0 or
                                            ((m - 1) * n_a) % M < ((m) * n_a) % M):
                        run_a_for_draw[m] = True
                        remaining_a -= 1
                    else:
                        run_b_for_draw[m] = True
                        remaining_b -= 1
            elif n_a == 0:
                for m in range(M):
                    run_b_for_draw[m] = True
            else:  # n_a == M
                for m in range(M):
                    run_a_for_draw[m] = True

        for m in range(M):
            if m == 0:
                noise = np.zeros((n, n), dtype=np.float64)
            else:
                noise = _generate_decode_noise(noise_type, n, root, rng, mu_b=mu_b,
                                               C_mod_b=C_mod_b, tau=tau)

            ca = float("inf")
            cb = float("inf")

            # Seed A: perturbed mu-greedy degree-2
            if run_a_for_draw[m]:
                score_m = base_score + float(gumbel_scale) * noise
                np.fill_diagonal(score_m, -np.inf)
                seed_a, _ = greedy_degree2_tour_from_scores(score_m, D_b, root=root)
                ca = _try_seed(seed_a)
                a_costs.append(ca)
                if ca < best_a_cost:
                    best_a_cost = ca

            # Seed B: perturbed MAP 1-tree → (μ-weighted) repair
            if run_b_for_draw[m]:
                C_mod_m = C_mod_b + float(gumbel_scale) * noise
                np.fill_diagonal(C_mod_m, 0.0)
                C_mod_m = 0.5 * (C_mod_m + C_mod_m.T)
                seed_b, _ = map_repair_seed(
                    C_mod_m, D_b, root=root, twoopt_passes=0,
                    mu_single=(mu_b if use_mu_repair else None),
                )
                cb = _try_seed(seed_b)
                b_costs.append(cb)
                if cb < best_b_cost:
                    best_b_cost = cb

            # Capture deterministic (no-noise) cost from draw 0
            if m == 0:
                det_cost_b = min(ca, cb)

        total_candidates = max(1, len(all_costs))
        mean_costs.append(float(np.mean(all_costs)))
        best_costs.append(best_cost_b)
        best_tours.append(best_tour_b)
        unique_fracs.append(float(len(seen_keys)) / float(total_candidates))
        a_mean_costs.append(float(np.mean(a_costs)) if a_costs else float("nan"))
        a_best_costs.append(best_a_cost if a_costs else float("nan"))
        b_mean_costs.append(float(np.mean(b_costs)) if b_costs else float("nan"))
        b_best_costs.append(best_b_cost if b_costs else float("nan"))
        det_costs.append(det_cost_b)

    return {
        "mean_cost": np.array(mean_costs, dtype=np.float64),
        "best_cost": np.array(best_costs, dtype=np.float64),
        "det_cost": np.array(det_costs, dtype=np.float64),
        "best_tour": np.array(best_tours, dtype=object),
        "unique_frac": float(np.mean(unique_fracs)),
        "num_draws": M,
        "gumbel_scale": float(gumbel_scale),
        "noise_type": str(noise_type),
        # Per-seed diagnostics
        "a_mean_cost": np.array(a_mean_costs, dtype=np.float64),
        "a_best_cost": np.array(a_best_costs, dtype=np.float64),
        "b_mean_cost": np.array(b_mean_costs, dtype=np.float64),
        "b_best_cost": np.array(b_best_costs, dtype=np.float64),
    }


def nearest_neighbor_cost(D: torch.Tensor) -> np.ndarray:
    D_np = D.detach().cpu().numpy()
    B, n, _ = D_np.shape
    costs = []
    for b in range(B):
        best = float("inf")
        for start in range(min(n, 5)):
            vis = {start}
            tc, cur = 0.0, start
            for _ in range(n - 1):
                row = D_np[b, cur].copy()
                row[list(vis)] = float("inf")
                nxt = row.argmin()
                tc += row[nxt]
                vis.add(nxt)
                cur = nxt
            tc += D_np[b, cur, start]
            best = min(best, tc)
        costs.append(best)
    return np.array(costs)


# ================================================================
# §8  Training loop
# ================================================================


def random_permute_batch(coords: torch.Tensor, dist: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly permute city labels independently for each instance.

    This preserves the Euclidean instance while removing the artificial meaning
    of the fixed root index 0. With root=0 in solver coordinates, a random
    permutation makes that root correspond to a random original city.
    """
    B, n, _ = coords.shape
    device = coords.device
    perms = torch.stack([torch.randperm(n, device=device) for _ in range(B)], dim=0)
    coords_perm = coords.gather(1, perms.unsqueeze(-1).expand(-1, -1, coords.shape[-1]))
    row_idx = perms.unsqueeze(-1).expand(-1, -1, n)
    dist_perm = dist.gather(1, row_idx)
    col_idx = perms.unsqueeze(1).expand(-1, n, -1)
    dist_perm = dist_perm.gather(2, col_idx)
    return coords_perm, dist_perm


def unpack_batch(batch):
    if len(batch) == 2:
        coords, dist = batch
        opt_tour = None
    elif len(batch) == 3:
        coords, dist, opt_tour = batch
    else:
        raise ValueError(f"Unexpected batch structure of length {len(batch)}")
    return coords, dist, opt_tour


def permute_tour_batch(opt_tour: torch.Tensor, perms: torch.Tensor) -> torch.Tensor:
    """Convert original-city tours to the permuted indexing used by the solver."""
    if opt_tour is None:
        return None
    inv = torch.empty_like(perms)
    inv.scatter_(1, perms, torch.arange(perms.size(1), device=perms.device).unsqueeze(0).expand_as(perms))
    return inv.gather(1, opt_tour)


def tour_cost_from_order(D: torch.Tensor, tours: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Compute closed-tour cost for a batch of tours on distance matrices D."""
    if isinstance(tours, np.ndarray):
        tours = torch.as_tensor(tours, device=D.device, dtype=torch.long)
    else:
        tours = tours.to(device=D.device, dtype=torch.long)
    if tours.dim() == 1:
        tours = tours.unsqueeze(0)
    nxt = torch.roll(tours, shifts=-1, dims=1)
    return D.gather(1, tours.unsqueeze(-1).expand(-1, -1, D.size(-1))).gather(2, nxt.unsqueeze(-1)).squeeze(-1).sum(
        dim=1)


