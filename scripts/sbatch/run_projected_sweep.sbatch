#!/bin/bash
#SBATCH --job-name=proj_sweep
#SBATCH --output=logs/proj_sweep_%A_%a.out
#SBATCH --error=logs/proj_sweep_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-11

# Projected Gradient Optimizer sweep
# Learning rates: 1e-2, 1e-3, 1e-4 (higher than typical Adam due to constrained subspace)
# Resync steps: 50, 100, 200, 500

# Change to working directory
WORKDIR="/home/guests/saket/continual/Self-Distillation"
cd $WORKDIR
mkdir -p logs

# Set environment variables
export HF_HOME="/home/guests/saket/models"
export WANDB_PROJECT="continual-learning-sdft"

# Login to wandb
uv run wandb login "$(cat "/home/guests/saket/continual/Self-Distillation/scripts/wandb_token.out" | tr -d '[:space:]')"
# Define sweep parameters
# Higher LRs for projected gradient (constrained to low-rank subspace)
LRS=("1e-2" "1e-3" "1e-4")
RESYNCS=(50 100 200 500)

# Calculate indices from array task ID
LR_IDX=$((SLURM_ARRAY_TASK_ID / 4))
RESYNC_IDX=$((SLURM_ARRAY_TASK_ID % 4))

LR=${LRS[$LR_IDX]}
RESYNC=${RESYNCS[$RESYNC_IDX]}

# Settings
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
TASK="tooluse"
ENERGY_THRESHOLD=0.9
GRAD_CLIP=100
EXP_ID="proj_lr${LR}_resync${RESYNC}"

OUTPUT_DIR="${WORKDIR}/outputs/projected_${EXP_ID}"

echo "========================================"
echo "Job: $SLURM_ARRAY_TASK_ID"
echo "LR: $LR, Resync: $RESYNC, Energy: $ENERGY_THRESHOLD"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# Run training (DFT = distillation with frozen teacher)
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
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --max_samples 256

echo "Done: $EXP_ID"
