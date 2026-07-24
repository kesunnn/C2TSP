#!/usr/bin/env bash
# Reproduce the recorded TSP100 base-s1 training configuration.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export PYTHONHASHSEED="${PYTHONHASHSEED:-12345}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

DEVICE="${DEVICE:-cuda:0}"
RESULTS_ROOT="${RESULTS_ROOT:-results_paper_tsp100_v1}"
TRAIN_DATA="${TRAIN_DATA:-data/tsp/tsp100_train_concorde.txt}"
VAL_DATA="${VAL_DATA:-data/tsp/tsp100_test_concorde.txt}"

python tsp_onetree_train.py \
  --seed 12345 \
  --device "${DEVICE}" \
  --results_root "${RESULTS_ROOT}" \
  --job_name paper_tsp100_base_s1 \
  --checkpoint_path model.pt \
  --num_cities 100 \
  --train_dataset_path "${TRAIN_DATA}" \
  --val_dataset_path "${VAL_DATA}" \
  --train_take 2048 \
  --val_take 10 \
  --epochs 100 \
  --batch_size 256 \
  --val_batch_size 256 \
  --node_dim 32 \
  --edge_dim 32 \
  --edge_hidden_mult 3 \
  --gradient_checkpoint 1 \
  --prior_weight 0.25 \
  --candidate_k 32 \
  --non_candidate_penalty 3.0 \
  --logit_clamp 10.0 \
  --lam_iters 150 \
  --lam_step 0.55 \
  --inner_tau_start 0.3 \
  --inner_tau_mid 0.22 \
  --inner_final_frac 0.75 \
  --resid_penalty_weight 5.0 \
  --ift_backward_tol 0.1 \
  --ift_ridge 3e-3 \
  --lr 5e-3 \
  --val_every 200 \
  --round0_loss_weight 0.0 \
  --decode_twoopt_passes 0 \
  --detach_refine_state 0 \
  --stage2_struct_target 0.1 \
  --stage2_objective_mode cert_budget \
  --stage2_struct_linear_weight 0.0 \
  --stage2_struct_quad_weight 0.0 \
  --nontour_entropy_weight 0.0 \
  --decode_lam_iters 100 \
  --decode_eval_limit 10 \
  --tau_start 0.3 \
  --tau 0.25 \
  --cert_alpha 0.0 \
  --tau_anneal_epochs 40 \
  --beta_delay_after_tau -40 \
  --sharpen_anneal_epochs 80 \
  --sharpen_beta 3 \
  --var_tilt_weight 1 \
  --decode_noise_type covariance \
  --decode_gumbel_M 1 \
  --decode_gumbel_scale 0.5 \
  --decode_seed 12345 \
  --decode_seed_split 0.0 \
  --decode_mu_repair 1 \
  --decode_hybrid 1 \
  --decode_pair_samples 4 \
  --decode_proposals_per_pair 1 \
  --stage2_steps 5 \
  --stage2_coupled_damping 0.65 \
  --decode_workers 8 \
  --disable_stage2_gnn_forward 1
