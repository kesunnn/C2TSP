import torch

from .onetree import (
    batched_cg_solve,
    build_inner_tau_path,
    expected_degree_residual_from_z,
    mean_zero_basis,
    split_lam_iters_across_stages,
)

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
                return torch.zeros_like(C_saved), None, None, None, None, None, None, None, None, None, None, None, None, None, None

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
