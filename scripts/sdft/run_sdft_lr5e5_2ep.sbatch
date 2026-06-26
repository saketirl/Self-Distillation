#!/bin/bash
#SBATCH --job-name=sdft_lr5e5_2ep
#SBATCH --output=logs/sdft_lr5e5_2ep_%j.out
#SBATCH --error=logs/sdft_lr5e5_2ep_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# SDFT training with LR=5e-5, 2 epochs per task
# Server: /home/guests/saket/continual/Self-Distillation/

set -e

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Paths
BASE_DIR="/home/guests/saket/continual/Self-Distillation"
MODELS_DIR="${BASE_DIR}/models"
OUTPUT_DIR="${BASE_DIR}/outputs"

# Experiment config
EXP_ID="sdft_lr5e5_2ep"
MODEL_BASE="${MODELS_DIR}/Qwen/Qwen2.5-3B-Instruct"
OUTPUT_SDFT="${OUTPUT_DIR}/${EXP_ID}"

# Hyperparameters
LEARNING_RATE="5e-5"
NUM_EPOCHS=2
REF_MODEL_MIXUP_ALPHA=0.05

echo "=============================================="
echo "SDFT Experiment: ${EXP_ID}"
echo "Model: ${MODEL_BASE}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Epochs per task: ${NUM_EPOCHS}"
echo "Output: ${OUTPUT_SDFT}"
echo "=============================================="

cd ${BASE_DIR}

uv run wandb login 9fb4ba17a708de72496774b2e25d219f07de038d

# Create logs directory if needed
mkdir -p logs



# ========== SDFT ==========
echo ""
echo "[SDFT 1/3] tooluse"
uv run python train_single_task.py \
    --method sdft \
    --model_name ${MODEL_BASE} \
    --task tooluse \
    --output_dir ${OUTPUT_SDFT}/task_0_tooluse \
    --exp_id ${EXP_ID} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --ref_model_mixup_alpha ${REF_MODEL_MIXUP_ALPHA} \
    --gradient_accumulation_steps 32 \
    --save_strategy no \
    --vllm_gpu_memory_utilization 0.3

echo ""
echo "[SDFT 2/3] gsm8k"
uv run python train_single_task.py \
    --method sdft \
    --model_name ${OUTPUT_SDFT}/task_0_tooluse \
    --task gsm8k \
    --output_dir ${OUTPUT_SDFT}/task_1_gsm8k \
    --exp_id ${EXP_ID} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --ref_model_mixup_alpha ${REF_MODEL_MIXUP_ALPHA} \
    --gradient_accumulation_steps 32 \
    --save_strategy no \
    --vllm_gpu_memory_utilization 0.3

echo ""
echo "[SDFT 3/3] mbpp"
uv run python train_single_task.py \
    --method sdft \
    --model_name ${OUTPUT_SDFT}/task_1_gsm8k \
    --task mbpp \
    --output_dir ${OUTPUT_SDFT}/task_2_mbpp \
    --exp_id ${EXP_ID} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --ref_model_mixup_alpha ${REF_MODEL_MIXUP_ALPHA} \
    --gradient_accumulation_steps 32 \
    --save_strategy no \
    --vllm_gpu_memory_utilization 0.3

echo ""
echo "=============================================="
echo "SDFT complete!"
echo "Output: ${OUTPUT_SDFT}"
echo "=============================================="
