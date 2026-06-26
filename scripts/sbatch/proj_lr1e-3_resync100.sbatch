#!/bin/bash
#SBATCH --job-name=proj_lr1e-3_resync100
#SBATCH --output=logs/proj_lr1e-3_resync100_%j.out
#SBATCH --error=logs/proj_lr1e-3_resync100_%j.err
#SBATCH --time=4:00:00
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
TASK="tooluse"
LR="1e-3"
RESYNC=100
ENERGY_THRESHOLD=0.9
GRAD_CLIP=1
EXP_ID="proj_lr1e-3_resync100"
OUTPUT_DIR="/home/guests/saket/continual/Self-Distillation/outputs/projected_${EXP_ID}"

echo "========================================"
echo "LR: $LR, Resync: $RESYNC, Energy: $ENERGY_THRESHOLD"
echo "Output: $OUTPUT_DIR"
echo "========================================"

uv run python train_single_task.py \
    --method sdft \
    --frozen_teacher \
    --task $TASK \
    --model_name $MODEL \
    --output_dir $OUTPUT_DIR \
    --exp_id $EXP_ID \
    --use_projected_optimizer \
    --proj_energy_threshold $ENERGY_THRESHOLD \
    --proj_resync_every $RESYNC \
    --proj_grad_clip $GRAD_CLIP \
    --learning_rate $LR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
   

echo "Done: $EXP_ID"
