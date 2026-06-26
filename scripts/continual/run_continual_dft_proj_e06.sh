#!/bin/bash
# Continual Learning: DFT + ProjectedGradient (energy=0.6)
# Tasks: spider -> tooluse -> cot_math
# Usage: bash run_continual_dft_proj_e06.sh <GPU_ID>

set -e

GPU_ID=${1:?"Usage: $0 <GPU_ID>"}
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="/data/saket/models"
export WANDB_PROJECT="qwen3-continual-learning"

BASE_DIR="/home/saket/continual/Self-Distillation"
cd ${BASE_DIR}
mkdir -p logs

uv run wandb login

# ── Hyperparameters ──────────────────────────────────────────────────────────
MODEL_BASE="Qwen/Qwen3-4B"
LR="5e-3"
ENERGY=0.6
UV_SCALE=0.3
B_SCALE=0.1
GRAD_ACCUM_STEPS=64
NUM_EPOCHS=1
EXP_ID="qwen3_cl_dft_proj_e06"
BASE_OUTPUT="/data/saket/continual/Self-Distillation/outputs/qwen3/${EXP_ID}"

echo "========================================"
echo "Continual Learning: DFT + ProjectedGradient"
echo "Tasks: spider -> tooluse -> cot_math"
echo "GPU: ${GPU_ID} | LR: ${LR} | Energy: ${ENERGY}"
echo "uv_scale: ${UV_SCALE} | b_scale: ${B_SCALE}"
echo "Grad accum: ${GRAD_ACCUM_STEPS} | Epochs: ${NUM_EPOCHS}"
echo "Output: ${BASE_OUTPUT}"
echo "========================================"

# ── Task 1/3: spider ─────────────────────────────────────────────────────────
echo ""
echo "[Task 1/3] Training on spider (base: ${MODEL_BASE})"
uv run python train_qwen3_dft.py \
    --method dft \
    --task spider \
    --model_name ${MODEL_BASE} \
    --output_dir ${BASE_OUTPUT}/task_0_spider \
    --exp_id ${EXP_ID} \
    --learning_rate ${LR} \
    --proj_energy_threshold ${ENERGY} \
    --proj_no_resync \
    --proj_grad_clip 1.0 \
    --proj_admm_steps 20 \
    --stiefel_update projected_qr \
    --stiefel_max_rms 0.03 \
    --uv_lr_scale ${UV_SCALE} \
    --b_lr_scale ${B_SCALE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --save_strategy no \
    --eval_all_tasks \
    --eval_max_samples 50

# ── Task 2/3: tooluse ────────────────────────────────────────────────────────
echo ""
echo "[Task 2/3] Training on tooluse (base: task_0_spider)"
uv run python train_qwen3_dft.py \
    --method dft \
    --task tooluse \
    --model_name ${BASE_OUTPUT}/task_0_spider \
    --output_dir ${BASE_OUTPUT}/task_1_tooluse \
    --exp_id ${EXP_ID} \
    --learning_rate ${LR} \
    --proj_energy_threshold ${ENERGY} \
    --proj_no_resync \
    --proj_grad_clip 1.0 \
    --proj_admm_steps 20 \
    --stiefel_update projected_qr \
    --stiefel_max_rms 0.03 \
    --uv_lr_scale ${UV_SCALE} \
    --b_lr_scale ${B_SCALE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --save_strategy no \
    --skip_before_eval \
    --eval_all_tasks \
    --eval_max_samples 50

# ── Task 3/3: cot_math ───────────────────────────────────────────────────────
echo ""
echo "[Task 3/3] Training on cot_math (base: task_1_tooluse)"
uv run python train_qwen3_dft.py \
    --method dft \
    --task cot_math \
    --model_name ${BASE_OUTPUT}/task_1_tooluse \
    --output_dir ${BASE_OUTPUT}/task_2_cot_math \
    --exp_id ${EXP_ID} \
    --learning_rate ${LR} \
    --proj_energy_threshold ${ENERGY} \
    --proj_no_resync \
    --proj_grad_clip 1.0 \
    --proj_admm_steps 20 \
    --stiefel_update projected_qr \
    --stiefel_max_rms 0.03 \
    --uv_lr_scale ${UV_SCALE} \
    --b_lr_scale ${B_SCALE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --save_strategy no \
    --skip_before_eval \
    --eval_all_tasks \
    --eval_max_samples 50

echo ""
echo "========================================"
echo "DFT continual learning complete: ${EXP_ID}"
echo "Checkpoints:"
echo "  ${BASE_OUTPUT}/task_0_spider"
echo "  ${BASE_OUTPUT}/task_1_tooluse"
echo "  ${BASE_OUTPUT}/task_2_cot_math"
echo "========================================"
