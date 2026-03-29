#!/bin/bash
# Full experiment - each task as separate process
# SDFT then SFT on tooluse -> gsm8k -> mbpp

set -e

export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP_ID="full_mbpp"
MODEL_BASE="Qwen/Qwen2.5-3B-Instruct"
OUTPUT_SDFT="/data/saket/continual/Self-Distillation/outputs/sdft_${EXP_ID}"
OUTPUT_SFT="/data/saket/continual/Self-Distillation/outputs/sft_${EXP_ID}"

# Shared hyperparameters
LEARNING_RATE="5e-6"
NUM_EPOCHS=1
REF_MODEL_MIXUP_ALPHA=0.05  # EMA alpha for SDFT teacher (paper sweeps: 0.01, 0.02, 0.05)

echo "=============================================="
echo "Full Experiment: ${EXP_ID}"
echo "Model: ${MODEL_BASE}"
echo "=============================================="

# ========== SDFT ==========
echo ""
echo "========== SDFT =========="

echo ""
echo "[SDFT 1/3] tooluse"
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
    --vllm_gpu_memory_utilization 0.3

echo ""
echo "[SDFT 2/3] gsm8k"
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
    --vllm_gpu_memory_utilization 0.3

echo ""
echo "[SDFT 3/3] mbpp"
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
    --vllm_gpu_memory_utilization 0.3

echo ""
echo "SDFT complete!"

# ========== SFT ==========
echo ""
echo "========== SFT =========="

echo ""
echo "[SFT 1/3] tooluse"
python train_single_task.py \
    --method sft \
    --model_name ${MODEL_BASE} \
    --task tooluse \
    --output_dir ${OUTPUT_SFT}/task_0_tooluse \
    --exp_id ${EXP_ID} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --gradient_accumulation_steps 32

echo ""
echo "[SFT 2/3] gsm8k"
python train_single_task.py \
    --method sft \
    --model_name ${OUTPUT_SFT}/task_0_tooluse \
    --task gsm8k \
    --output_dir ${OUTPUT_SFT}/task_1_gsm8k \
    --exp_id ${EXP_ID} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --gradient_accumulation_steps 32

echo ""
echo "[SFT 3/3] mbpp"
python train_single_task.py \
    --method sft \
    --model_name ${OUTPUT_SFT}/task_1_gsm8k \
    --task mbpp \
    --output_dir ${OUTPUT_SFT}/task_2_mbpp \
    --exp_id ${EXP_ID} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --gradient_accumulation_steps 32

echo ""
echo "=============================================="
echo "All experiments complete!"
echo "SDFT: ${OUTPUT_SDFT}"
echo "SFT: ${OUTPUT_SFT}"
echo "=============================================="
