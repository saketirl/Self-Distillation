#!/bin/bash
#SBATCH --job-name=eval_dft_proj_full
#SBATCH --output=logs/eval_dft_proj_full_%j.out
#SBATCH --error=logs/eval_dft_proj_full_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

# Full eval on all DFT + ProjectedGradient continual learning checkpoints
# Evaluates task_0_tooluse, task_1_cot_math, task_2_spider on full spider/tooluse/cot_math sets

cd /home/guests/saket/continual/Self-Distillation
mkdir -p logs

export HF_HOME="/home/guests/saket/models"
export WANDB_PROJECT="qwen3-continual-learning"

export WANDB_API_KEY=$(cat "/home/guests/saket/continual/Self-Distillation/scripts/wandb_token.out" | tr -d '[:space:]')

EXP_ID="qwen3_cl_dft_proj_e06"
EXP_DIR="/home/guests/saket/continual/Self-Distillation/outputs/qwen3/${EXP_ID}"

echo "========================================"
echo "Full Eval: DFT + ProjectedGradient"
echo "Exp dir: $EXP_DIR"
echo "Tasks: spider (1034), tooluse (68), cot_math (750)"
echo "========================================"

python scripts/eval_continual_checkpoints.py \
    --exp_dir ${EXP_DIR} \
    --exp_id ${EXP_ID} \
    --base_model Qwen/Qwen3-4B \
    --max_samples 10000

echo "Done: full eval for ${EXP_ID}"
