from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .onetree import mean_zero_basis, safe_batched_solve

def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int, keepdim: bool = False, eps: float = 1e-8) -> torch.Tensor:
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

        dist_scale = (dist_matrix * full_f).sum(dim=(-2, -1), keepdim=True) / full_f.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        dist_scale = dist_scale.clamp_min(1e-6)
        dist_norm = dist_matrix / dist_scale

        mean_d = (dist_norm * full_f).sum(dim=-1, keepdim=True) / full_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
        min_d = dist_norm.masked_fill(~full_mask, float("inf")).amin(dim=-1, keepdim=True)
        max_d = dist_norm.masked_fill(~full_mask, 0.0).amax(dim=-1, keepdim=True)
        var_d = ((dist_norm - mean_d) ** 2 * full_f).sum(dim=-1, keepdim=True) / full_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
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
        rank_norm = rank_indices / float(max(n - 2, 1))   # 0 = nearest, 1 = farthest
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
                rank_norm.unsqueeze(-1),   # v10: kNN rank feature
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
        node_feat, edge_attr, full_mask, dist_norm = self._build_invariant_features(coords, dist_matrix, cand_mask, residual_state=residual_state)
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
            d_feat = dist_norm.unsqueeze(-1).to(dtype=e.dtype)     # (B, n, n, 1)
            head_input = torch.cat([e, d_feat], dim=-1)            # (B, n, n, edge_dim+1)
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
