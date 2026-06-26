#!/bin/bash
#SBATCH --job-name=qwen3_mq
#SBATCH --output=logs/qwen3_medquad_%j.out
#SBATCH --error=logs/qwen3_medquad_%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

# Qwen3-4B DFT on MedQuad
# Using hyperparameters from proj_lr1em3_energy08.sbatch

cd /home/guests/saket/continual/Self-Distillation
mkdir -p logs

export HF_HOME="/home/guests/saket/models"
export WANDB_PROJECT="qwen3-continual-learning"

uv run wandb login "$(cat "/home/guests/saket/continual/Self-Distillation/scripts/wandb_token.out" | tr -d '[:space:]')"
MODEL="Qwen/Qwen3-4B"
TASK="medquad"
LR="1e-3"
ENERGY_THRESHOLD=0.5
EXP_ID="qwen3_medquad_dft"
OUTPUT_DIR="/home/guests/saket/continual/Self-Distillation/outputs/qwen3/${EXP_ID}"

echo "========================================"
echo "Qwen3-4B DFT Training"
echo "Task: $TASK"
echo "Model: $MODEL"
echo "LR: $LR, Energy: $ENERGY_THRESHOLD"
echo "Output: $OUTPUT_DIR"
echo "========================================"

uv run python train_qwen3_dft.py \
    --method dft \
    --task $TASK \
    --model_name $MODEL \
    --output_dir $OUTPUT_DIR \
    --exp_id $EXP_ID \
    --learning_rate $LR \
    --proj_energy_threshold $ENERGY_THRESHOLD \
    --proj_no_resync \
    --proj_grad_clip 1.0 \
    --proj_admm_steps 20 \
    --stiefel_update projected_qr \
    --stiefel_max_rms 0.03 \
    --uv_lr_scale 0.3 \
    --b_lr_scale 0.3 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --skip_before_eval \
    --eval_all_tasks \
    --eval_max_samples 50

echo "Done: $EXP_ID"
