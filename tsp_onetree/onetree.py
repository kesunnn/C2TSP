import functools
import math
from typing import Callable, Dict, List, Tuple

import torch

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
