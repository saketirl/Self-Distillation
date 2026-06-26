#!/bin/bash
#SBATCH --job-name=revisql_proj_30b_test
#SBATCH --output=/home/guests/saket/continual/Self-Distillation/logs/revisql_proj_30b_test_%j.out
#SBATCH --error=/home/guests/saket/continual/Self-Distillation/logs/revisql_proj_30b_test_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16

# 1 GPU (80 GB): Qwen3-30B-A3B BF16 ~60 GB + FrozenResidualLinear UBV state at
# energy_threshold=0.25 (very small rank) fits comfortably in remaining headroom.

set -e

BASE_DIR="/home/guests/saket/continual/Self-Distillation"
REVISQL_DIR="${BASE_DIR}/ReViSQL"
OUTPUT_DIR="${BASE_DIR}/outputs/revisql/proj_grpo_qwen30b_test"
LOGS="${BASE_DIR}/logs"

mkdir -p "${LOGS}" "${OUTPUT_DIR}"

cd "${BASE_DIR}"
export HF_HOME=/home/guests/saket/models
export WANDB_PROJECT=continual-sql
export WANDB_ENTITY=rl-power
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN="$(cat "${BASE_DIR}/scripts/hf_token.out" | tr -d '[:space:]')"

uv run wandb login "$(cat "/home/guests/saket/continual/Self-Distillation/scripts/wandb_token.out" | tr -d '[:space:]')"

uv run python "${REVISQL_DIR}/scripts/train_proj_grpo.py" \
    --model_name Qwen/Qwen3-30B-A3B \
    --train_data "${REVISQL_DIR}/data/bird-verified-train.parquet" \
    --val_data   "${REVISQL_DIR}/data/val_test.parquet" \
    --db_path    /home/guests/saket/data/bird/databases \
    --output_dir "${OUTPUT_DIR}" \
    --exp_id     revisql_proj_30b_test \
    \
    --group_size     4 \
    --batch_size     2 \
    --max_turns      5 \
    --max_new_tokens 1024 \
    --temperature    1.0 \
    --clip_high      1.2 \
    --clip_low       0.8 \
    --n_epochs       1 \
    --num_data       64 \
    \
    --lr                5e-3 \
    --energy_threshold  0.25 \
    --stiefel_update    muon_admm \
    --admm_steps        10 \
    --admm_rho          4.0 \
    --stiefel_max_rms   0.03 \
    --uv_lr_scale       0.3 \
    --b_lr_scale        0.1 \
    --target_modules    q_proj,k_proj,v_proj,o_proj \
    \
    --eval_every    10 \
    --eval_samples  20 \
    --save_every    0 \
    --gradient_checkpointing \
    --download_missing_dbs \
    --wandb_project continual-sql \
    --wandb_entity  rl-power \
    --dtype bfloat16
