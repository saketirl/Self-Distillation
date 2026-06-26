#!/bin/bash
# Submit a single DFT + ProjectedGradient job for one task and energy threshold.
# Usage: bash submit_dft_proj_single.sh <task> <energy>
# Example: bash submit_dft_proj_single.sh tooluse 0.7

set -e

TASK=${1:?"Usage: $0 <task> <energy>   e.g. $0 tooluse 0.7"}
ENERGY=${2:?"Usage: $0 <task> <energy>   e.g. $0 tooluse 0.7"}
ENERGY_TAG=$(echo ${ENERGY} | tr -d '.')

BASE_DIR="/home/guests/saket/continual/Self-Distillation"
EXP_ID="qwen3_dft_proj_e${ENERGY_TAG}_${TASK}"
OUTPUT_DIR="${BASE_DIR}/outputs/qwen3/${EXP_ID}"
LOGS="${BASE_DIR}/logs"

mkdir -p ${LOGS}

echo "Submitting: DFT + ProjectedGradient | task=${TASK} | energy=${ENERGY}"
echo "Output: ${OUTPUT_DIR}"

sbatch \
    --job-name=dft_e${ENERGY_TAG}_${TASK} \
    --output=${LOGS}/dft_e${ENERGY_TAG}_${TASK}_%j.out \
    --error=${LOGS}/dft_e${ENERGY_TAG}_${TASK}_%j.err \
    --time=12:00:00 \
    --gres=gpu:1 \
    --mem=80G \
    --cpus-per-task=8 \
    --wrap="
cd ${BASE_DIR}
export HF_HOME=/home/guests/saket/models
export WANDB_PROJECT=qwen3-continual-learning
wandb login "$(cat "/home/guests/saket/continual/Self-Distillation/scripts/wandb_token.out" | tr -d '[:space:]')"
uv run python train_qwen3_dft.py \
    --method dft \
    --task ${TASK} \
    --model_name Qwen/Qwen3-4B \
    --output_dir ${OUTPUT_DIR} \
    --exp_id ${EXP_ID} \
    --learning_rate 5e-3 \
    --proj_energy_threshold ${ENERGY} \
    --proj_no_resync \
    --proj_grad_clip 1.0 \
    --proj_admm_steps 20 \
    --stiefel_update projected_qr \
    --stiefel_max_rms 0.03 \
    --uv_lr_scale 0.3 \
    --b_lr_scale 0.1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --save_strategy no \
    --eval_tasks spider,tooluse,cot_math \
    --eval_max_samples 50
"
