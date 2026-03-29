#!/bin/bash
# Run each task as separate Python process to avoid GPU memory issues
# Each task loads from previous task's checkpoint

set -e

export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

METHOD=$1  # sdft or sft
EXP_ID=${2:-"exp01"}
MODEL_BASE="Qwen/Qwen2.5-3B-Instruct"
OUTPUT_BASE="outputs/${METHOD}_${EXP_ID}"

if [ -z "$METHOD" ]; then
    echo "Usage: ./run_sequential_tasks.sh <sdft|sft> [exp_id]"
    exit 1
fi

echo "=============================================="
echo "Sequential Training: ${METHOD}"
echo "Experiment ID: ${EXP_ID}"
echo "Output: ${OUTPUT_BASE}"
echo "=============================================="

# Shared hyperparameters (matching SDFT paper recommendations)
LEARNING_RATE="1e-5"
NUM_EPOCHS=2
REF_MODEL_MIXUP_ALPHA=0.05  # EMA alpha for SDFT teacher (paper sweeps: 0.01, 0.02, 0.05)

# Task 1: tooluse (start from base model)
echo ""
echo "[Task 1/3] Training on tooluse..."
echo ""
python train_single_task.py \
    --method ${METHOD} \
    --model_name ${MODEL_BASE} \
    --task tooluse \
    --output_dir ${OUTPUT_BASE}/task_0_tooluse \
    --exp_id ${EXP_ID} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --ref_model_mixup_alpha ${REF_MODEL_MIXUP_ALPHA} \
    --gradient_accumulation_steps 32 \
    --vllm_gpu_memory_utilization 0.3

echo "[Task 1/3] tooluse complete!"

# Task 2: gsm8k (load from task 1 checkpoint)
echo ""
echo "[Task 2/3] Training on gsm8k..."
echo ""
python train_single_task.py \
    --method ${METHOD} \
    --model_name ${OUTPUT_BASE}/task_0_tooluse \
    --task gsm8k \
    --output_dir ${OUTPUT_BASE}/task_1_gsm8k \
    --exp_id ${EXP_ID} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --ref_model_mixup_alpha ${REF_MODEL_MIXUP_ALPHA} \
    --gradient_accumulation_steps 32 \
    --vllm_gpu_memory_utilization 0.3

echo "[Task 2/3] gsm8k complete!"

# Task 3: mbpp (load from task 2 checkpoint)
echo ""
echo "[Task 3/3] Training on mbpp..."
echo ""
python train_single_task.py \
    --method ${METHOD} \
    --model_name ${OUTPUT_BASE}/task_1_gsm8k \
    --task mbpp \
    --output_dir ${OUTPUT_BASE}/task_2_mbpp \
    --exp_id ${EXP_ID} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --ref_model_mixup_alpha ${REF_MODEL_MIXUP_ALPHA} \
    --gradient_accumulation_steps 32 \
    --vllm_gpu_memory_utilization 0.3

echo "[Task 3/3] mbpp complete!"

echo ""
echo "=============================================="
echo "All tasks complete!"
echo "Final model: ${OUTPUT_BASE}/task_2_mbpp"
echo "=============================================="
