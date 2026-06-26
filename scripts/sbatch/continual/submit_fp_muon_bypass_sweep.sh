#!/bin/bash
# Sweep vr_bypass_alpha on tooluse only (single task, not chained).
#
# Usage:
#   bash submit_fp_muon_bypass_sweep.sh
#
# Submits one job per bypass_alpha value: 0.0, 0.1, 0.3, 0.5, 1.0

set -e

BASE_DIR="/home/guests/saket/continual/Self-Distillation"
LOGS="${BASE_DIR}/logs"
SBATCH_COMMON="--parsable --gres=gpu:1 --mem=80G --cpus-per-task=8 --time=06:00:00"

mkdir -p ${LOGS}

for BYPASS in 0.0 0.1 0.3 0.5 1.0; do
    BYPASS_TAG=$(echo ${BYPASS} | tr -d '.')   # 0.1 -> 01
    EXP_ID="fp_muon_tooluse_b${BYPASS_TAG}"
    OUTPUT="${BASE_DIR}/outputs/qwen3/${EXP_ID}/task_0_tooluse"

    JOB=$(sbatch ${SBATCH_COMMON} \
        --job-name=fp_bypass_b${BYPASS_TAG} \
        --output=${LOGS}/fp_bypass_b${BYPASS_TAG}_%j.out \
        --error=${LOGS}/fp_bypass_b${BYPASS_TAG}_%j.err \
        --wrap="
cd ${BASE_DIR}
export HF_HOME=/home/guests/saket/models
export WANDB_PROJECT=qwen3-continual-learning
export WANDB_API_KEY=\$(cat '${BASE_DIR}/scripts/wandb_token.out' | tr -d '[:space:]')
uv run python train_qwen3_dft.py \
    --method dft \
    --task tooluse \
    --model_name Qwen/Qwen3-4B \
    --output_dir ${OUTPUT} \
    --exp_id ${EXP_ID} \
    --use_fp_muon \
    --fp_energy_threshold 0.8 \
    --fp_n_dual_iter 10 \
    --fp_dual_step_size 0.01 \
    --fp_vr_bypass_alpha ${BYPASS} \
    --fp_debug \
    --learning_rate 1e-3 \
    --fp_adamw_lr 3e-4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --save_strategy no \
    --eval_all_tasks \
    --eval_tasks tooluse \
    --eval_max_samples 50
")
    echo "bypass=${BYPASS}  job=${JOB}  exp=${EXP_ID}"
done
