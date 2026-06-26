#!/bin/bash
#SBATCH --job-name=dolly_muon_lr5em3
#SBATCH --output=logs/dolly_muon_lr5em3_%j.out
#SBATCH --error=logs/dolly_muon_lr5em3_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Change to working directory
cd /home/guests/saket/continual/Self-Distillation
mkdir -p logs

# Set environment variables
export HF_HOME="/home/guests/saket/models"
export WANDB_PROJECT="continual-learning-sdft"

# Login to wandb
uv run wandb login "$(cat "/home/guests/saket/continual/Self-Distillation/scripts/wandb_token.out" | tr -d '[:space:]')"
MODEL="Qwen/Qwen2.5-3B-Instruct"
TASK="dolly"
LR="5e-3"
MOMENTUM=0.95
EXP_ID="dolly_muon_lr5em3"
OUTPUT_DIR="/home/guests/saket/continual/Self-Distillation/outputs/${EXP_ID}"

echo "========================================"
echo "DFT Muon Baseline"
echo "Task: $TASK"
echo "LR: $LR, Momentum: $MOMENTUM"
echo "Output: $OUTPUT_DIR"
echo "========================================"

uv run python train_single_task.py \
    --method sdft \
    --frozen_teacher \
    --task $TASK \
    --model_name $MODEL \
    --output_dir $OUTPUT_DIR \
    --exp_id $EXP_ID \
    --use_muon_optimizer \
    --muon_momentum $MOMENTUM \
    --learning_rate $LR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64

echo "Done: $EXP_ID"
