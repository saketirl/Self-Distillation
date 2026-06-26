#!/bin/bash
#SBATCH --job-name=sft_cl_cotmath
#SBATCH --output=logs/sft_cl_cotmath_%j.out
#SBATCH --error=logs/sft_cl_cotmath_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

# SFT + AdamW continual chain — Task 2/3: cot_math
# Starts from task_0_tooluse checkpoint

cd /home/guests/saket/continual/Self-Distillation
mkdir -p logs

export HF_HOME="/home/guests/saket/models"
export WANDB_PROJECT="qwen3-continual-learning"

export WANDB_API_KEY=$(cat "/home/guests/saket/continual/Self-Distillation/scripts/wandb_token.out" | tr -d '[:space:]')

EXP_ID="qwen3_cl_sft_adamw"
BASE_OUTPUT="/home/guests/saket/continual/Self-Distillation/outputs/qwen3/${EXP_ID}"

uv run python train_qwen3_dft.py \
    --method sft \
    --adamw_only \
    --task cot_math \
    --model_name ${BASE_OUTPUT}/task_0_tooluse \
    --output_dir ${BASE_OUTPUT}/task_1_cot_math \
    --exp_id ${EXP_ID} \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --save_strategy no \
    --skip_before_eval \
    --eval_all_tasks \
    --eval_tasks spider,tooluse,cot_math \
    --eval_max_samples 50

echo "Done: ${EXP_ID} task_1_cot_math"
