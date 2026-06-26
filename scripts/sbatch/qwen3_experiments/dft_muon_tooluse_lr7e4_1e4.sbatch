#!/bin/bash
#SBATCH --job-name=qwen3_tu_muon_7e4
#SBATCH --output=logs/qwen3_tooluse_muon_7e4_1e4_%j.out
#SBATCH --error=logs/qwen3_tooluse_muon_7e4_1e4_%j.err
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

# Qwen3-4B DFT (frozen teacher, Muon+AdamW) on Tooluse
# Muon LR=7e-4, AdamW LR=1e-4

cd /home/guests/saket/continual/Self-Distillation
mkdir -p logs

export HF_HOME="/home/guests/saket/models"
export WANDB_PROJECT="qwen3-continual-learning"

uv run wandb login "$(cat "/home/guests/saket/continual/Self-Distillation/scripts/wandb_token.out" | tr -d '[:space:]')"
MODEL="Qwen/Qwen3-4B"
TASK="tooluse"
MUON_LR="7e-4"
ADAMW_LR="1e-4"
MUON_MOMENTUM="0.95"
EXP_ID="qwen3_tooluse_dft_muon_lr7e4_1e4"
OUTPUT_DIR="/home/guests/saket/continual/Self-Distillation/outputs/qwen3/${EXP_ID}"

echo "========================================"
echo "Qwen3-4B DFT Muon Training (frozen teacher)"
echo "Task: $TASK"
echo "Model: $MODEL"
echo "Muon LR: $MUON_LR, AdamW LR: $ADAMW_LR, Momentum: $MUON_MOMENTUM"
echo "Output: $OUTPUT_DIR"
echo "========================================"

uv run python train_qwen3_dft.py \
    --method dft \
    --use_muon \
    --task $TASK \
    --model_name $MODEL \
    --output_dir $OUTPUT_DIR \
    --exp_id $EXP_ID \
    --muon_lr $MUON_LR \
    --muon_adamw_lr $ADAMW_LR \
    --muon_momentum $MUON_MOMENTUM \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --skip_before_eval \
    --eval_all_tasks \
    --eval_max_samples 50

echo "Done: $EXP_ID"
