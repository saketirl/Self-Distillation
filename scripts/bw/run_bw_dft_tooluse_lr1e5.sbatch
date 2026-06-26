#!/bin/bash
#SBATCH --job-name=bw_dft_lr1e5
#SBATCH --output=logs/bw_dft_tooluse_lr1e5_%j.out
#SBATCH --error=logs/bw_dft_tooluse_lr1e5_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# BW-Riemannian DFT (frozen teacher) on tooluse
# LR=1e-5, Rank=auto (maintain existing)
# Server: /home/guests/saket/continual/Self-Distillation/

set -e

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Paths
BASE_DIR="/home/guests/saket/continual/Self-Distillation"
MODELS_DIR="${BASE_DIR}/models"
OUTPUT_DIR="${BASE_DIR}/outputs"

# Experiment config
EXP_ID="bw_dft_lr1e5"
MODEL_BASE="${MODELS_DIR}/Qwen/Qwen2.5-3B-Instruct"
OUTPUT_BW="${OUTPUT_DIR}/${EXP_ID}"

# Hyperparameters
LEARNING_RATE="1e-5"
NUM_EPOCHS=1

echo "=============================================="
echo "BW-Riemannian DFT Experiment: ${EXP_ID}"
echo "Model: ${MODEL_BASE}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Optimizer: BWRiemannianSGD (rank=auto, energy=0.8)"
echo "Output: ${OUTPUT_BW}"
echo "=============================================="

cd ${BASE_DIR}

uv run wandb login 9fb4ba17a708de72496774b2e25d219f07de038d

# Create logs directory if needed
mkdir -p logs

echo ""
echo "[BW-DFT] tooluse"
uv run python train_single_task.py \
    --method sdft \
    --model_name ${MODEL_BASE} \
    --task tooluse \
    --output_dir ${OUTPUT_BW}/task_0_tooluse \
    --exp_id ${EXP_ID} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --gradient_accumulation_steps 32 \
    --save_strategy no \
    --vllm_gpu_memory_utilization 0.3 \
    --frozen_teacher \
    --use_bw_optimizer \
    --bw_rank auto \
    --bw_energy_threshold 0.8

echo ""
echo "=============================================="
echo "BW-DFT complete!"
echo "Output: ${OUTPUT_BW}"
echo "=============================================="
