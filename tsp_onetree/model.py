import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import NeighborhoodEdgePriorEncoder, project_edge_residual_candidate_weighted
from .graph import build_candidate_mask
from .implicit_layer import rooted_onetree_implicit_layer
from .onetree import split_lam_iters_across_stages

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
        disable_stage2_gnn_forward: int = 0,
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
        stage2_steps: int | None = None,
        stage2_coupled_steps: int = 0,
        stage2_coupled_damping: float = 0.65,
        edge_quotient_projection: int = 1,
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
        self.edge_quotient_projection = bool(edge_quotient_projection)

        # Single user-facing stage-2 control.
        #   stage2_steps = 0: pure stage-1 rooted one-tree only.
        #   stage2_steps = 1: one stage-2 residual round with the legacy one-shot tilt.
        #   stage2_steps = K: one stage-2 residual round with K self-consistent tilt solves.
        # If stage2_steps is omitted, preserve legacy CLI behavior: stage2_coupled_steps=0
        # means the one-shot stage-2 correction, while stage2_coupled_steps>0 gives that many
        # self-consistent stage-2 solves.
        if stage2_steps is None:
            legacy_coupled_steps = max(0, int(stage2_coupled_steps))
            stage2_steps_int = 1 if legacy_coupled_steps == 0 else legacy_coupled_steps
        else:
            stage2_steps_int = max(0, int(stage2_steps))
        self.stage2_steps = stage2_steps_int
        self.num_refine_rounds = 1 if self.stage2_steps == 0 else 2

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
        # Backward-compatible name: this now records extra coupled sweeps beyond
        # the one-shot stage-2 solve. The actual coupled-loop length is stage2_steps.
        self.stage2_coupled_steps = max(0, self.stage2_steps - 1)
        self.stage2_coupled_damping = min(max(float(stage2_coupled_damping), 0.0), 1.0)
        # In pure stage-1 mode, give the single one-tree solve the full iteration budget.
        # In two-round mode, preserve the old cheap first-round warm start.
        self.round1_lam_frac = 1.0 if self.num_refine_rounds == 1 else 0.4
        self.refine_state_scale = 2.0
        self.detach_refine_state = bool(detach_refine_state)
        self.disable_stage2_gnn_forward = bool(disable_stage2_gnn_forward)
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
        steps = max(1, int(self.stage2_steps))
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
        if self.edge_quotient_projection:
            tilt_proj, tilt_proj_info = project_edge_residual_candidate_weighted(
                tilt_raw,
                cand_mask,
                ridge=self.edge_projection_ridge,
            )
        else:
            tilt_proj = tilt_raw * cand_mask.to(dtype=tilt_raw.dtype)
            tilt_proj_info = self._zero_projection_info(C_base.shape[0], C_base.shape[1], C_base.device, C_base.dtype)
        C_eff = C_base - tilt_proj
        C_eff = 0.5 * (C_eff + C_eff.transpose(-2, -1))
        C_eff = C_eff * full_f
        tilt_info = {
            "certificate_rms": self._edge_rms(certificate, cand_mask).mean(),
            "tilt_proj_rms": self._edge_rms(tilt_proj, cand_mask).mean(),
            "tilt_proj_fallback_frac": tilt_proj_info.get("fallback_frac", torch.zeros((), device=C_base.device, dtype=C_base.dtype)),
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

    def _zero_projection_info(
        self,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        zero = torch.zeros((), device=device, dtype=dtype)
        return {
            "row_leak": zero,
            "grand_leak": zero,
            "removed_energy_frac": zero,
            "fallback_frac": zero,
            "node_additive": torch.zeros(batch_size, num_nodes, device=device, dtype=dtype),
        }

    def _zero_round2_update_info(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        zero = torch.zeros((), device=device, dtype=dtype)
        return {
            "round2_update_rms": zero,
            "round2_raw_rms": zero,
            "round2_scale": zero,
            "round2_free_scale": zero,
            "round2_gate": zero,
            "round2_gate_mult": zero,
            "round2_gate_logit": zero,
            "round2_gate_residual": zero,
        }

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

        residual_ref = stage1_record["residual"].detach() if self.round2_gate_detach_features else stage1_record["residual"]
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
            disable_round2_gnn = bool(round_idx > 0 and self.disable_stage2_gnn_forward)
            if disable_round2_gnn:
                enc_aux = {}
                raw_logits = torch.zeros_like(dist_matrix)
                bounded_logits = torch.zeros_like(dist_matrix)
                proj_aux = self._zero_projection_info(B, n, device, dist_matrix.dtype)
            else:
                round_state = residual_state.detach() if (round_idx > 0 and self.detach_refine_state) else residual_state
                raw_logits, enc_aux = self.encoder(
                    coords,
                    dist_matrix,
                    cand_mask,
                    residual_state=round_state,
                    round_idx=round_idx,
                )
                raw_logits = 0.5 * (raw_logits + raw_logits.transpose(-2, -1))
                raw_logits = raw_logits * full_f

                if self.edge_quotient_projection:
                    bounded_logits, proj_aux = project_edge_residual_candidate_weighted(
                        raw_logits,
                        cand_mask,
                        ridge=self.edge_projection_ridge,
                    )
                else:
                    bounded_logits = raw_logits * cand_mask.to(dtype=raw_logits.dtype)
                    proj_aux = self._zero_projection_info(B, n, device, dist_matrix.dtype)
                bounded_logits = self._apply_logit_clamp(bounded_logits)

            # Warm-start λ from the projection's removed node-additive component.
            if round_idx == 0 and lambda_init_nr is None:
                node_add = proj_aux.get("node_additive")
                if node_add is not None and node_add.shape[-1] >= 2:
                    lambda_init_nr = (-self.prior_weight * node_add[:, 1:]).detach()

            # v14: No learnable gate. Round 2 uses integrality tilt instead.
            round2_gate = None
            round2_gate_info = None
            if disable_round2_gnn:
                round_update = torch.zeros_like(cumulative_logits)
                update_info = self._zero_round2_update_info(device, dist_matrix.dtype)
            else:
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
                if self.stage2_steps > 1:
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
        round0_rec = round_records[0]
        round1_rec = round_records[1] if len(round_records) > 1 else round_records[0]
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
                    proj_lambda_align = F.cosine_similarity(add_nr[valid], lambda_nr[valid], dim=-1, eps=1e-8).abs().mean()
                else:
                    proj_lambda_align = torch.zeros((), device=device, dtype=lambda_nr.dtype)
            else:
                proj_lambda_align = torch.zeros((), device=device, dtype=lambda_nr.dtype)

        stats = {
            "cost": mu.new_tensor(final_rec["cost"]).detach() if not isinstance(final_rec["cost"], torch.Tensor) else final_rec["cost"].detach(),
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
            "stage2_steps": stage2_coupled_steps.detach(),
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
            "round0_cost": round0_rec["cost"].mean().detach(),
            "round1_cost": round1_rec["cost"].mean().detach(),
            "round0_solver_residual": round0_rec["residual"].mean().detach(),
            "round1_solver_residual": round1_rec["residual"].mean().detach(),
            "round1_update_rms": round1_rec["update_info"]["round2_update_rms"].detach(),
            "round1_raw_rms": round1_rec["update_info"]["round2_raw_rms"].detach(),
            "round1_scale": round1_rec["update_info"]["round2_scale"].detach(),
            "round1_gate": round1_rec["update_info"]["round2_gate"].detach(),
            "round1_gate_mult": round1_rec["update_info"]["round2_gate_mult"].detach(),
            "round1_free_scale": round1_rec["update_info"]["round2_free_scale"].detach(),
            "round1_gate_logit": round1_rec["update_info"]["round2_gate_logit"].detach(),
            "round1_gate_residual": round1_rec["update_info"]["round2_gate_residual"].detach(),
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
