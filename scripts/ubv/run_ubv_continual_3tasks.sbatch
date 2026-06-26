#!/bin/bash
#SBATCH --job-name=ubv_continual
#SBATCH --output=logs/ubv_continual_%j.out
#SBATCH --error=logs/ubv_continual_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# UBV Riemannian Continual Learning on 3 tasks
# Task sequence: tooluse -> gsm8k -> mbpp
# Using rank-preserving UBV^T factorization with Riemannian optimization

set -e

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Paths
BASE_DIR="/home/guests/saket/continual/Self-Distillation"
MODELS_DIR="${BASE_DIR}/models"
OUTPUT_DIR="${BASE_DIR}/outputs"

# Experiment config
EXP_ID="ubv_continual_3tasks"
MODEL_BASE="${MODELS_DIR}/Qwen/Qwen2.5-3B-Instruct"
OUTPUT_UBV="${OUTPUT_DIR}/${EXP_ID}"

# Hyperparameters
LEARNING_RATE="5e-5"
NUM_EPOCHS=1
UBV_ENERGY=0.8
UBV_LAMBDA_MIN=1e-4

echo "=============================================="
echo "UBV Continual Learning: ${EXP_ID}"
echo "Model: ${MODEL_BASE}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Optimizer: UBV Riemannian (rank=auto, energy=${UBV_ENERGY})"
echo "Spectrum floor: lambda_min=${UBV_LAMBDA_MIN}"
echo "Tasks: tooluse -> gsm8k -> mbpp"
echo "Output: ${OUTPUT_UBV}"
echo "=============================================="

cd ${BASE_DIR}

uv run wandb login 9fb4ba17a708de72496774b2e25d219f07de038d

mkdir -p logs

# Task 1: tooluse
echo ""
echo "========================================"
echo "[Task 1/3] Training on tooluse"
echo "========================================"
uv run python train_single_task.py \
    --method sdft \
    --model_name ${MODEL_BASE} \
    --task tooluse \
    --output_dir ${OUTPUT_UBV}/task_0_tooluse \
    --exp_id ${EXP_ID} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --gradient_accumulation_steps 32 \
    --save_strategy no \
    --vllm_gpu_memory_utilization 0.3 \
    --frozen_teacher \
    --use_ubv_optimizer \
    --ubv_rank auto \
    --ubv_energy_threshold ${UBV_ENERGY} \
    --ubv_lambda_min ${UBV_LAMBDA_MIN}

# Task 2: gsm8k (continual from tooluse)
echo ""
echo "========================================"
echo "[Task 2/3] Training on gsm8k (from tooluse)"
echo "========================================"
uv run python train_single_task.py \
    --method sdft \
    --model_name ${OUTPUT_UBV}/task_0_tooluse \
    --task gsm8k \
    --output_dir ${OUTPUT_UBV}/task_1_gsm8k \
    --exp_id ${EXP_ID} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --gradient_accumulation_steps 32 \
    --save_strategy no \
    --vllm_gpu_memory_utilization 0.3 \
    --frozen_teacher \
    --use_ubv_optimizer \
    --ubv_rank auto \
    --ubv_energy_threshold ${UBV_ENERGY} \
    --ubv_lambda_min ${UBV_LAMBDA_MIN}

# Task 3: mbpp (continual from gsm8k)
echo ""
echo "========================================"
echo "[Task 3/3] Training on mbpp (from gsm8k)"
echo "========================================"
uv run python train_single_task.py \
    --method sdft \
    --model_name ${OUTPUT_UBV}/task_1_gsm8k \
    --task mbpp \
    --output_dir ${OUTPUT_UBV}/task_2_mbpp \
    --exp_id ${EXP_ID} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --gradient_accumulation_steps 32 \
    --save_strategy no \
    --vllm_gpu_memory_utilization 0.3 \
    --frozen_teacher \
    --use_ubv_optimizer \
    --ubv_rank auto \
    --ubv_energy_threshold ${UBV_ENERGY} \
    --ubv_lambda_min ${UBV_LAMBDA_MIN}

echo ""
echo "=============================================="
echo "UBV Continual Learning complete!"
echo "Final model: ${OUTPUT_UBV}/task_2_mbpp"
echo "=============================================="
