#!/bin/bash
# Quick test run - each task as separate process
# ~2 steps per task with 64 samples
# Hyperparameters match run_fullexp.sh

set -e

export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP_ID="exp_mbpp"
MODEL_BASE="Qwen/Qwen2.5-3B-Instruct"
OUTPUT_SDFT="outputs/test_sdft"
OUTPUT_SFT="outputs/test_sft"

# Shared hyperparameters (match run_fullexp.sh)
LEARNING_RATE="5e-6"
NUM_EPOCHS=1
REF_MODEL_MIXUP_ALPHA=0.05

echo "=============================================="
echo "TEST RUN - Separate process per task"
echo "=============================================="

# Clean up old test outputs
rm -rf ${OUTPUT_SDFT} ${OUTPUT_SFT}

# ========== SDFT ==========
echo ""
echo "[SDFT] Task 1: tooluse"
python train_single_task.py \
    --method sdft \
    --model_name ${MODEL_BASE} \
    --task tooluse \
    --output_dir ${OUTPUT_SDFT}/task_0_tooluse \
    --exp_id ${EXP_ID} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --ref_model_mixup_alpha ${REF_MODEL_MIXUP_ALPHA} \
    --gradient_accumulation_steps 32 \
    --vllm_gpu_memory_utilization 0.3 \
    --max_samples 64

echo ""
echo "[SDFT] Task 2: gsm8k"
python train_single_task.py \
    --method sdft \
    --model_name ${OUTPUT_SDFT}/task_0_tooluse \
    --task gsm8k \
    --output_dir ${OUTPUT_SDFT}/task_1_gsm8k \
    --exp_id ${EXP_ID} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --ref_model_mixup_alpha ${REF_MODEL_MIXUP_ALPHA} \
    --gradient_accumulation_steps 32 \
    --vllm_gpu_memory_utilization 0.3 \
    --max_samples 64

echo ""
echo "[SDFT] Task 3: mbpp"
python train_single_task.py \
    --method sdft \
    --model_name ${OUTPUT_SDFT}/task_1_gsm8k \
    --task mbpp \
    --output_dir ${OUTPUT_SDFT}/task_2_mbpp \
    --exp_id ${EXP_ID} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --ref_model_mixup_alpha ${REF_MODEL_MIXUP_ALPHA} \
    --gradient_accumulation_steps 32 \
    --vllm_gpu_memory_utilization 0.3 \
    --max_samples 64

echo ""
echo "[SDFT] Complete!"

# ========== SFT ==========
echo ""
echo "[SFT] Task 1: tooluse"
python train_single_task.py \
    --method sft \
    --model_name ${MODEL_BASE} \
    --task tooluse \
    --output_dir ${OUTPUT_SFT}/task_0_tooluse \
    --exp_id ${EXP_ID} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --gradient_accumulation_steps 32 \
    --max_samples 64

echo ""
echo "[SFT] Task 2: gsm8k"
python train_single_task.py \
    --method sft \
    --model_name ${OUTPUT_SFT}/task_0_tooluse \
    --task gsm8k \
    --output_dir ${OUTPUT_SFT}/task_1_gsm8k \
    --exp_id ${EXP_ID} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --gradient_accumulation_steps 32 \
    --max_samples 64

echo ""
echo "[SFT] Task 3: mbpp"
python train_single_task.py \
    --method sft \
    --model_name ${OUTPUT_SFT}/task_1_gsm8k \
    --task mbpp \
    --output_dir ${OUTPUT_SFT}/task_2_mbpp \
    --exp_id ${EXP_ID} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --gradient_accumulation_steps 32 \
    --max_samples 64

echo ""
echo "=============================================="
echo "Test complete!"
echo "SDFT results: ${OUTPUT_SDFT}/task_*/results.json"
echo "SFT results: ${OUTPUT_SFT}/task_*/results.json"
echo "=============================================="
