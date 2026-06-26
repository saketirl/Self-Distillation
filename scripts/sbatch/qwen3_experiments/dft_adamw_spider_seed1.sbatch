#!/bin/bash
#SBATCH --job-name=qwen3_spider_adamw_s1
#SBATCH --output=logs/qwen3_spider_adamw_seed1_%j.out
#SBATCH --error=logs/qwen3_spider_adamw_seed1_%j.err
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

# Qwen3-4B DFT (frozen teacher, plain AdamW) on spider
# Seed: 1

cd /home/guests/saket/continual/Self-Distillation
mkdir -p logs

export HF_HOME="/home/guests/saket/models"
export WANDB_PROJECT="qwen3-continual-learning"

uv run wandb login "$(cat "/home/guests/saket/continual/Self-Distillation/scripts/wandb_token.out" | tr -d '[:space:]')"
MODEL="Qwen/Qwen3-4B"
TASK="spider"
LR="1e-5"
SEED=1
EXP_ID="qwen3_spider_dft_adamw_seed1"
OUTPUT_DIR="/home/guests/saket/continual/Self-Distillation/outputs/qwen3/${EXP_ID}"

echo "========================================"
echo "Qwen3-4B DFT AdamW Training (frozen teacher)"
echo "Task: $TASK | Seed: $SEED"
echo "LR: $LR"
echo "Output: $OUTPUT_DIR"
echo "========================================"

uv run python train_qwen3_dft.py \
    --method dft \
    --adamw_only \
    --task $TASK \
    --model_name $MODEL \
    --output_dir $OUTPUT_DIR \
    --exp_id $EXP_ID \
    --learning_rate $LR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --skip_before_eval \
    --eval_all_tasks \
    --eval_max_samples 50 \
    --seed $SEED

echo "Done: $EXP_ID"
