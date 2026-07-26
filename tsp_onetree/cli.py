import argparse

from .repro import set_global_seed
from .train import train


def normalize_decode_repair_mode(value) -> str:
    text = str(value).strip().lower()
    alias_map = {
        "": "auto",
        "auto": "auto",
        "0": "backbone_insert",
        "backbone_insert": "backbone_insert",
        "1": "mu_match",
        "mu_match": "mu_match",
        "2": "mu_christofides",
        "mu_christofides": "mu_christofides",
        "3": "cmod_christofides",
        "cmod_christofides": "cmod_christofides",
        "4": "c_christofides",
        "c_christofides": "c_christofides",
    }
    if text not in alias_map:
        raise ValueError(
            f"Unsupported --decode_repair_mode {value!r}. "
            "Use one of auto, backbone_insert, mu_match, mu_christofides, "
            "cmod_christofides, c_christofides, or legacy aliases 0..4."
        )
    return alias_map[text]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=12345,
                   help="Global seed controlling initialization, synthetic datasets, dataloader shuffling, and training-time permutations.")
    p.add_argument("--device", type=str, default="auto",
                   help="Device to use: 'auto', 'cpu', 'cuda', 'cuda:0', etc.")
    p.add_argument("--job_name", type=str, default="tsp_onetree_2round",
                   help="Job name used to create results/<job_name>_<timestamp>/ for checkpoints and metric JSON files.")
    p.add_argument("--results_root", type=str, default="results",
                   help="Root directory under which each run creates a timestamped result folder.")
    p.add_argument("--num_cities", type=int, default=30)
    p.add_argument("--num_train", type=int, default=300)
    p.add_argument("--num_val", type=int, default=100)
    p.add_argument("--train_dataset_path", type=str, default="",
                   help="Optional path to an official Concorde-labeled dataset file for training.")
    p.add_argument("--val_dataset_path", type=str, default="",
                   help="Optional path to an official Concorde-labeled dataset file for validation.")
    p.add_argument("--train_take", type=int, default=0,
                   help="If >0, use only the first train_take samples from train_dataset_path after skipping train_skip.")
    p.add_argument("--val_take", type=int, default=0,
                   help="If >0, use only the first val_take samples from val_dataset_path after skipping val_skip.")
    p.add_argument("--train_skip", type=int, default=0,
                   help="Skip this many samples at the start of train_dataset_path.")
    p.add_argument("--val_skip", type=int, default=0,
                   help="Skip this many samples at the start of val_dataset_path.")
    p.add_argument("--node_dim", type=int, default=0,
                   help="Node embedding dimension. If 0 (default), set to edge_dim so node and edge widths match.")
    p.add_argument("--edge_dim", type=int, default=32)
    p.add_argument("--edge_hidden_mult", type=int, default=3,
                   help="Width multiplier for the edge-update hidden layer inside each GNN block.")
    p.add_argument("--num_gnn_layers", type=int, default=4)
    p.add_argument("--gradient_checkpoint", type=int, default=1,
                   help="If 1, use activation checkpointing in the GNN layers to reduce peak memory usage during training.")
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=0.05, help="Final training temperature.")
    p.add_argument("--tau_start", type=float, default=0.15, help="Initial temperature for annealed training.")
    p.add_argument("--tau_anneal_epochs", type=int, default=40, help="Number of epochs used to anneal tau_start -> tau.")
    p.add_argument("--prior_weight", type=float, default=0.3)
    p.add_argument("--candidate_k", type=int, default=16)
    p.add_argument("--non_candidate_penalty", type=float, default=2.0)
    p.add_argument("--lam_iters", type=int, default=40)
    p.add_argument("--lam_tol", type=float, default=1e-6)
    p.add_argument("--lam_step", type=float, default=0.5)
    p.add_argument("--ift_ridge", type=float, default=1e-4)
    p.add_argument("--ift_backward_tol", type=float, default=1e-2,
                   help="Only trust implicit backward on samples whose final inner residual is below this threshold.")
    p.add_argument("--inner_homotopy", type=int, default=1,
                   help="If 1, solve the inner equilibrium with a warm-started tau continuation that ends at --tau.")
    p.add_argument("--inner_tau_start", type=float, default=0.30,
                   help="Largest tau used only inside the forward continuation path when --inner_homotopy=1.")
    p.add_argument("--inner_tau_mid", type=float, default=0.22,
                   help="Optional intermediate tau used only inside the forward continuation path when --inner_homotopy=1.")
    p.add_argument("--inner_final_frac", type=float, default=0.75,
                   help="Fraction of lam_iters allocated to the final, sharpest continuation stage.")
    p.add_argument("--cov_shrink", type=float, default=0.25,
                   help="Forward-only covariance shrinkage for Newton metric: H <- (1-rho)H + rho*diag(H), annealed across homotopy stages.")
    p.add_argument("--lm_damping", type=float, default=0.05,
                   help="Forward-only Levenberg-Marquardt damping added to the Newton metric as lm_damping * mean(diag(H)) * I.")
    p.add_argument("--detach_refine_state", type=int, default=1,
                   help="If 1, detach the round-1 refinement state before the second round so round 2 acts as a cleaner residual correction.")
    p.add_argument("--disable_stage2_gnn_forward", type=int, default=1,
                   help="If 1, skip the second GNN forward and use only the stage-1 edge cost plus stage-2 certificate tilt for the final stage-2 solve/loss.")
    p.add_argument("--skip_stage1_hk", type=int, default=0,
                   help="If 1, bypass the stage-1 HK equilibrium solve and use direct zero-dual rooted 1-tree marginals. Stage 2 remains unchanged.")
    p.add_argument("--num_refine_rounds", type=int, default=2, choices=[1, 2],
                   help="Number of refinement rounds for the shared-weight model. Use 1 to run the refactored code as a strict one-round baseline.")
    p.add_argument("--single_round_use_stage2_loss", type=int, default=0,
                   help="Only relevant when --num_refine_rounds=1. If 1, keep a single forward round but train it with the stage-2 certificate objective instead of the legacy stage-1 loss.")
    p.add_argument("--round2_use_struct_gate", type=int, default=0,
                   help="Deprecated compatibility flag. Ignored by the current modular pipeline.")
    p.add_argument("--round2_gate_detach_features", type=int, default=0,
                   help="Deprecated compatibility flag. Ignored by the current modular pipeline.")
    p.add_argument("--round2_gate_hidden_dim", type=int, default=64,
                   help="Deprecated compatibility flag. Ignored by the current modular pipeline.")
    p.add_argument("--round2_struct_gate_floor", type=float, default=0.35,
                   help="Deprecated compatibility flag. Ignored by the current modular pipeline.")
    p.add_argument("--round2_struct_gate_temp", type=float, default=0.25,
                   help="Deprecated compatibility flag. Ignored by the current modular pipeline.")
    p.add_argument("--round2_struct_bonus", type=float, default=0.35,
                   help="Deprecated compatibility flag. Ignored by the current modular pipeline.")
    p.add_argument("--deg_penalty_weight", type=float, default=10.0,
                   help="Legacy compatibility flag retained for older checkpoints/logs. The faithful mainline now uses the explicit U_deg structural certificate instead.")
    p.add_argument("--resid_penalty_weight", type=float, default=1.0,
                   help="Weight on solver residual penalty to discourage unconverged inner solves")
    p.add_argument("--bern_penalty_weight", type=float, default=0.0,
                   help="Legacy compatibility flag retained for older checkpoints/logs. The faithful mainline now uses nodewise uncertainty inside U_deg instead.")
    p.add_argument("--stage2_struct_target", type=float, default=0.0,
                   help="Only used when --stage2_objective_mode=cert_budget. Target epsilon for the normalized proxy certificate U_proxy/(n-1).")
    p.add_argument("--stage2_struct_target_start", type=float, default=None,
                   help="Optional initial stage-2 structural target for annealed training. If omitted, keep --stage2_struct_target constant.")
    p.add_argument("--stage2_struct_target_anneal_epochs", type=int, default=None,
                   help="Optional number of epochs used to anneal stage2_struct_target_start -> stage2_struct_target. If omitted, keep --stage2_struct_target constant.")
    p.add_argument("--stage2_struct_linear_weight", type=float, default=0.0,
                   help="Only used when --stage2_objective_mode=cert_budget. Linear weight on the normalized proxy-certificate violation.")
    p.add_argument("--stage2_struct_quad_weight", type=float, default=0.0,
                   help="Only used when --stage2_objective_mode=cert_budget. Quadratic weight on the normalized proxy-certificate violation.")
    p.add_argument("--stage2_struct_uncertainty_weight", type=float, default=1.0,
                   help="Multiplier eta on the uncertainty term inside the rigorous proxy certificate U_proxy. Values below 1 are automatically clamped to 1 to preserve the upper-bound guarantee.")
    p.add_argument("--stage2_entropy_penalty_weight", type=float, default=0.0,
                   help="Optional mild stage-2 entropy penalty. Leave at 0 for the cleanest certificate-driven version.")
    p.add_argument("--nontour_entropy_weight", type=float, default=0.0,
                   help="Penalty on non-tour edge entropy, applied only in the second round.")
    p.add_argument("--stage2_use_cost_term", type=int, default=1,
                   help="If 1, keep the round-1 cost term inside the stage-2 objective. If 0, round 1 uses only the stage-2 penalty terms and solver residual/entropy terms.")
    p.add_argument("--stage2_objective_mode", type=str, default="cert_budget",
                   choices=["cert_budget", "repair_bound", "tour_bound", "hinge_penalty", "penalty", "hinge"],
                   help="Stage-2 objective. 'cert_budget' is the default constrained surrogate: minimize cost subject to a normalized U_proxy budget via a hinge penalty. 'repair_bound' optimizes the corrected rigorous repair-cost bound cost + D_max * U_proxy. 'tour_bound' is kept as a backward-compatible alias for 'repair_bound'.")
    p.add_argument("--stage2_bound_weight", type=float, default=0.0,
                   help="Only used when --stage2_objective_mode=repair_bound. If >0, use this coefficient; if <=0, use the instancewise metric constant D_max.")
    p.add_argument("--round0_loss_weight", type=float, default=0.0,
                   help="Auxiliary weight on the stage-1 loss inside the final two-round objective.")
    p.add_argument("--sharpen_beta", type=float, default=0.0,
                   help="Round-2 certificate-tilt strength. If 0, disable sharpen refinement.")
    p.add_argument("--sharpen_anneal_epochs", type=int, default=0,
                   help="Number of epochs used to ramp sharpen_beta from 0 to its target value.")
    p.add_argument("--beta_delay_after_tau", type=int, default=0,
                   help="Extra epochs to wait after tau annealing completes before sharpen_beta starts ramping.")
    p.add_argument("--cert_alpha", type=float, default=0.0,
                   help="Strength of the learned certificate correction added to the baseline (mu-0.5) stage-2 certificate.")
    p.add_argument("--var_tilt_weight", type=float, default=0.0,
                   help="Variance-weighting strength for the round-2 certificate tilt.")
    p.add_argument("--stage2_steps", type=int, default=None,
                   help="Single stage-2 control. 0 = pure stage-1 only; 1 = one-shot stage-2 sharpen; K>1 = K self-consistent stage-2 sharpen solves.")
    p.add_argument("--stage2_coupled_steps", type=int, default=0,
                   help="If >0, run this many damped stage-2 sharpen/re-equilibrate sweeps in round 2.")
    p.add_argument("--stage2_coupled_damping", type=float, default=0.65,
                   help="Damping factor used by the round-2 coupled sharpen loop.")
    p.add_argument("--edge_quotient_projection", type=int, default=1,
                   help="If 1, project raw learned edge logits away from the node-additive / constant quotient-nullspace before the implicit solve. If 0, feed the raw candidate-supported logits directly.")
    p.add_argument("--edge_head_with_cost", type=int, default=0,
                   help="If 1, feed normalized edge cost into the final edge head in addition to the learned edge embedding.")
    p.add_argument("--logit_clamp", type=float, default=5.0,
                   help="Smoothly bound GNN logits with c*tanh(logit/c) to keep C_theta well-conditioned")
    p.add_argument("--dual_pair_weight", type=float, default=0.0,
                   help="Deprecated compatibility flag. Ignored in this primal-only mainline.")
    p.add_argument("--dual_hint_clamp", type=float, default=0.0,
                   help="Deprecated compatibility flag. Ignored in this primal-only mainline.")
    p.add_argument("--root", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--val_batch_size", type=int, default=64,
                   help="Batch size used for validation/test forward passes.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--random_train_permute", type=int, default=1,
                   help="Randomly permute city labels during training to remove fixed-root label bias.")
    p.add_argument("--val_every", type=int, default=3)
    p.add_argument("--train_decode_every", type=int, default=3,
                   help="Run the decoded-train-set metric pass every N epochs and record the results in train_metrics.json. Set <=0 to disable.")
    p.add_argument("--decode_eval_limit", type=int, default=10)
    p.add_argument("--decode_gumbel_M", type=int, default=8,
                   help="Number of dual-seed noisy decode draws used by the default evaluation decoder. Draw 0 is deterministic.")
    p.add_argument("--decode_gumbel_scale", type=float, default=0.20,
                   help="Scale of the shared symmetric decode perturbation used by the default evaluation decoder.")
    p.add_argument("--decode_lam_iters", type=int, default=0,
                   help="If >0, temporarily use this many implicit-layer iterations only during evaluation/decode.")
    p.add_argument("--decode_noise_type", type=str, default="gumbel",
                   choices=["gumbel", "gaussian", "uncertainty", "dual", "covariance", "covariance_uncertainty"],
                   help="Noise family used by the default dual-seed decoder.")
    p.add_argument("--decode_lk_alpha", type=int, default=0,
                   help="If 1, order LK-lite candidate moves by ascending C_mod (LK-alpha) while still accepting moves by true tour cost.")
    p.add_argument("--decode_seed_split", type=float, default=-1.0,
                   help="Decode portfolio split across Seed A (mu-greedy) and Seed B (MAP-repair): 0.0 = B only, 1.0 = A only, values in (0,1) split the draw budget, and values <0 run both seeds on every draw.")
    p.add_argument("--decode_mu_repair", type=int, default=1,
                   help="Use mu-weighted MAP repair in the default dual-seed decoder when set to 1; 0 falls back to the legacy backbone-insertion repair.")
    p.add_argument("--decode_repair_mode", type=str, default="auto",
                   help="Seed-B rooted-1-tree repair mode: auto preserves historical behavior, mu_match is the current local swap repair, and the christofides variants use global parity matching with mu, C_mod, or original metric D weights before Euler shortcutting. Legacy integer aliases are also accepted: 0=backbone_insert, 1=mu_match, 2=mu_christofides, 3=cmod_christofides, 4=c_christofides.")
    p.add_argument("--decode_hybrid", type=int, default=0,
                   help="If 1, use the hybrid GPU/CPU decode path for validation/test decoding.")
    p.add_argument("--decode_workers", type=int, default=1,
                   help="Number of per-instance CPU workers used by the default decoder. 1 = serial; >1 uses a persistent ProcessPoolExecutor.")
    p.add_argument("--decode_samples", type=int, default=4,
                   help="Legacy alias for the old exact sampled decoder path. Retained for backward-compatible CLIs.")
    p.add_argument("--decode_pair_samples", type=int, default=-1,
                   help="Legacy exact sampled-decoder setting retained for backward-compatible CLIs.")
    p.add_argument("--decode_proposals_per_pair", type=int, default=-1,
                   help="Legacy exact sampled-decoder setting retained for backward-compatible CLIs.")
    p.add_argument("--decode_sample_bonus", type=float, default=5.0,
                   help="Deprecated compatibility flag. Ignored by the exact Gibbs sampled decoder.")
    p.add_argument("--decode_root_pair_bonus", type=float, default=2.5,
                   help="Deprecated compatibility flag. Ignored by the exact Gibbs sampled decoder.")
    p.add_argument("--decode_root_other_penalty", type=float, default=0.0,
                   help="Deprecated compatibility flag. Ignored by the exact Gibbs sampled decoder.")
    p.add_argument("--decode_tree_gumbel_scale", type=float, default=0.35,
                   help="Deprecated compatibility flag. Ignored by the exact Gibbs sampled decoder.")
    p.add_argument("--decode_root_gumbel_scale", type=float, default=0.20,
                   help="Deprecated compatibility flag. Ignored by the exact Gibbs sampled decoder.")
    p.add_argument("--decode_score_gumbel_scale", type=float, default=0.0,
                   help="Deprecated compatibility flag. Ignored by the exact Gibbs sampled decoder.")
    p.add_argument("--decode_twoopt_passes", type=int, default=6,
                   help="Number of 2-opt passes used in all decode paths (plain greedy and sampled best-of-K).")
    p.add_argument("--decode_seed", type=int, default=None,
                   help="Optional decoder seed override. If omitted, defaults to --seed.")
    p.add_argument("--debug_eval_timing", type=int, default=0,
                   help="Evaluation timing debug level: 0=off, 1=batch/instance summaries, 2=per-instance sampled-decode breakdown.")
    p.add_argument("--checkpoint_path", type=str, default="best_model_onetree.pt")
    p.add_argument("--loss_mode", type=str, default="cost_entropy", choices=["cost", "cost_entropy"])
    p.add_argument("--entropy_weight", type=float, default=0.05)
    args = p.parse_args()
    args.train_dataset_path = args.train_dataset_path or None
    args.val_dataset_path = args.val_dataset_path or None
    args.train_take = args.train_take if args.train_take > 0 else None
    args.val_take = args.val_take if args.val_take > 0 else None
    args.val_batch_size = max(1, int(args.val_batch_size))
    args.train_decode_every = int(args.train_decode_every)

    if args.decode_pair_samples < 0 and args.decode_proposals_per_pair < 0:
        args.decode_pair_samples = max(0, int(args.decode_samples))
        args.decode_proposals_per_pair = 1 if args.decode_pair_samples > 0 else 0
    else:
        if args.decode_pair_samples < 0:
            args.decode_pair_samples = 1
        if args.decode_proposals_per_pair < 0:
            args.decode_proposals_per_pair = 1
        if args.decode_pair_samples <= 0 or args.decode_proposals_per_pair <= 0:
            args.decode_pair_samples = 0
            args.decode_proposals_per_pair = 0
    if args.decode_seed is None:
        args.decode_seed = int(args.seed)
    args.decode_gumbel_M = max(0, int(args.decode_gumbel_M))
    args.decode_lam_iters = max(0, int(args.decode_lam_iters))
    args.decode_repair_mode = normalize_decode_repair_mode(args.decode_repair_mode)
    args.decode_total_samples = int(args.decode_pair_samples) * int(args.decode_proposals_per_pair)
    if int(args.node_dim) <= 0:
        args.node_dim = int(args.edge_dim)
    if args.stage2_steps is None and int(args.num_refine_rounds) <= 1:
        args.stage2_steps = 0
    return args


def main() -> int:
    args = get_args()
    set_global_seed(args.seed)
    train(args)
    return 0
