#!/bin/bash
#SBATCH --job-name=qwen3_cl_sft_adamw
#SBATCH --output=logs/qwen3_cl_sft_adamw_%j.out
#SBATCH --error=logs/qwen3_cl_sft_adamw_%j.err
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

# Qwen3-4B Continual Learning: SFT + AdamW (baseline)
# Task sequence: spider -> tooluse -> cot_math
# One checkpoint saved per task; after-eval of task N == before-eval of task N+1.

cd /home/guests/saket/continual/Self-Distillation
mkdir -p logs

export HF_HOME="/home/guests/saket/models"
export WANDB_PROJECT="qwen3-continual-learning"

uv run wandb login "$(cat "/home/guests/saket/continual/Self-Distillation/scripts/wandb_token.out" | tr -d '[:space:]')"
MODEL_BASE="Qwen/Qwen3-4B"
LR="1e-4"
EXP_ID="qwen3_cl_sft_adamw"
BASE_OUTPUT="/home/guests/saket/continual/Self-Distillation/outputs/qwen3/${EXP_ID}"

echo "========================================"
echo "Qwen3-4B Continual Learning: SFT + AdamW"
echo "Tasks: spider -> tooluse -> cot_math"
echo "LR: $LR"
echo "Base output: $BASE_OUTPUT"
echo "========================================"

# ── Task 1/3: spider (from Qwen3-4B base) ──────────────────────────────────
echo ""
echo "[Task 1/3] Training on spider (base: $MODEL_BASE)"
uv run python train_qwen3_dft.py \
    --method sft \
    --adamw_only \
    --task spider \
    --model_name $MODEL_BASE \
    --output_dir ${BASE_OUTPUT}/task_0_spider \
    --exp_id ${EXP_ID} \
    --learning_rate $LR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --save_strategy no \
    --eval_all_tasks \
    --eval_max_samples 50

# ── Task 2/3: tooluse (from task_0_spider checkpoint) ──────────────────────
echo ""
echo "[Task 2/3] Training on tooluse (base: task_0_spider)"
uv run python train_qwen3_dft.py \
    --method sft \
    --adamw_only \
    --task tooluse \
    --model_name ${BASE_OUTPUT}/task_0_spider \
    --output_dir ${BASE_OUTPUT}/task_1_tooluse \
    --exp_id ${EXP_ID} \
    --learning_rate $LR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --save_strategy no \
    --skip_before_eval \
    --eval_all_tasks \
    --eval_max_samples 50

# ── Task 3/3: cot_math (from task_1_tooluse checkpoint) ────────────────────
echo ""
echo "[Task 3/3] Training on cot_math (base: task_1_tooluse)"
uv run python train_qwen3_dft.py \
    --method sft \
    --adamw_only \
    --task cot_math \
    --model_name ${BASE_OUTPUT}/task_1_tooluse \
    --output_dir ${BASE_OUTPUT}/task_2_cot_math \
    --exp_id ${EXP_ID} \
    --learning_rate $LR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --save_strategy no \
    --skip_before_eval \
    --eval_all_tasks \
    --eval_max_samples 50

echo ""
echo "========================================"
echo "Continual Learning complete: ${EXP_ID}"
echo "Checkpoints:"
echo "  ${BASE_OUTPUT}/task_0_spider"
echo "  ${BASE_OUTPUT}/task_1_tooluse"
echo "  ${BASE_OUTPUT}/task_2_cot_math"
echo "========================================"
