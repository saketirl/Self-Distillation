#!/bin/bash
# Ablation study: 4 SDFT ablations + SFT baseline
# Ablations: alpha {0.05, 0.01} x lr {3e-5, 5e-5}
# Distributed across GPUs 0,1,2,3

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_BASE="Qwen/Qwen2.5-3B-Instruct"
OUTPUT_BASE="/data/saket/continual/Self-Distillation/outputs"

# Fixed hyperparameters
NUM_EPOCHS=1
GRAD_ACCUM_STEPS=64

# GPU memory threshold (in MB) - GPU considered "available" if free memory > this
GPU_MEM_THRESHOLD=70000

# Function to check if GPU is available
gpu_available() {
    local GPU_ID=$1
    local FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${GPU_ID} 2>/dev/null | tr -d ' ')
    if [ -z "$FREE_MEM" ]; then
        return 1
    fi
    [ "$FREE_MEM" -gt "$GPU_MEM_THRESHOLD" ]
}

# Function to wait for GPU to be available
wait_for_gpu() {
    local GPU_ID=$1
    echo "Waiting for GPU ${GPU_ID} to be available..."
    while ! gpu_available ${GPU_ID}; do
        sleep 30
    done
    echo "GPU ${GPU_ID} is available!"
}

# Function to run SDFT on all 3 tasks
run_sdft() {
    local GPU_ID=$1
    local LR=$2
    local ALPHA=$3
    local LR_TAG=$4      # e.g., "lr3e5"
    local ALPHA_TAG=$5   # e.g., "a05"

    wait_for_gpu ${GPU_ID}

    export CUDA_VISIBLE_DEVICES=${GPU_ID}

    echo ""
    echo "========== SDFT on GPU ${GPU_ID}: lr=${LR}, alpha=${ALPHA} =========="

    # Task 1: tooluse
    local EXP_ID="sdft_tooluse_${ALPHA_TAG}_${LR_TAG}"
    local OUTPUT_DIR="${OUTPUT_BASE}/${EXP_ID}"
    echo "[SDFT] ${EXP_ID}"
    python train_single_task.py \
        --method sdft \
        --model_name ${MODEL_BASE} \
        --task tooluse \
        --output_dir ${OUTPUT_DIR} \
        --exp_id ${EXP_ID} \
        --learning_rate ${LR} \
        --num_train_epochs ${NUM_EPOCHS} \
        --ref_model_mixup_alpha ${ALPHA} \
        --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
        --vllm_gpu_memory_utilization 0.3

    # Task 2: gsm8k
    local PREV_OUTPUT=${OUTPUT_DIR}
    EXP_ID="sdft_gsm8k_${ALPHA_TAG}_${LR_TAG}"
    OUTPUT_DIR="${OUTPUT_BASE}/${EXP_ID}"
    echo "[SDFT] ${EXP_ID}"
    python train_single_task.py \
        --method sdft \
        --model_name ${PREV_OUTPUT} \
        --task gsm8k \
        --output_dir ${OUTPUT_DIR} \
        --exp_id ${EXP_ID} \
        --learning_rate ${LR} \
        --num_train_epochs ${NUM_EPOCHS} \
        --ref_model_mixup_alpha ${ALPHA} \
        --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
        --vllm_gpu_memory_utilization 0.3

    # Task 3: mbpp
    PREV_OUTPUT=${OUTPUT_DIR}
    EXP_ID="sdft_mbpp_${ALPHA_TAG}_${LR_TAG}"
    OUTPUT_DIR="${OUTPUT_BASE}/${EXP_ID}"
    echo "[SDFT] ${EXP_ID}"
    python train_single_task.py \
        --method sdft \
        --model_name ${PREV_OUTPUT} \
        --task mbpp \
        --output_dir ${OUTPUT_DIR} \
        --exp_id ${EXP_ID} \
        --learning_rate ${LR} \
        --num_train_epochs ${NUM_EPOCHS} \
        --ref_model_mixup_alpha ${ALPHA} \
        --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
        --vllm_gpu_memory_utilization 0.3

    echo "SDFT ${ALPHA_TAG}_${LR_TAG} complete on GPU ${GPU_ID}!"
}

# Function to run SFT baseline
run_sft() {
    local GPU_ID=$1
    local LR=$2
    local LR_TAG=$3

    wait_for_gpu ${GPU_ID}

    export CUDA_VISIBLE_DEVICES=${GPU_ID}

    echo ""
    echo "========== SFT Baseline on GPU ${GPU_ID}: lr=${LR} =========="

    # Task 1: tooluse
    local EXP_ID="sft_tooluse_${LR_TAG}"
    local OUTPUT_DIR="${OUTPUT_BASE}/${EXP_ID}"
    echo "[SFT] ${EXP_ID}"
    python train_single_task.py \
        --method sft \
        --model_name ${MODEL_BASE} \
        --task tooluse \
        --output_dir ${OUTPUT_DIR} \
        --exp_id ${EXP_ID} \
        --learning_rate ${LR} \
        --num_train_epochs ${NUM_EPOCHS} \
        --gradient_accumulation_steps ${GRAD_ACCUM_STEPS}

    # Task 2: gsm8k
    local PREV_OUTPUT=${OUTPUT_DIR}
    EXP_ID="sft_gsm8k_${LR_TAG}"
    OUTPUT_DIR="${OUTPUT_BASE}/${EXP_ID}"
    echo "[SFT] ${EXP_ID}"
    python train_single_task.py \
        --method sft \
        --model_name ${PREV_OUTPUT} \
        --task gsm8k \
        --output_dir ${OUTPUT_DIR} \
        --exp_id ${EXP_ID} \
        --learning_rate ${LR} \
        --num_train_epochs ${NUM_EPOCHS} \
        --gradient_accumulation_steps ${GRAD_ACCUM_STEPS}

    # Task 3: mbpp
    PREV_OUTPUT=${OUTPUT_DIR}
    EXP_ID="sft_mbpp_${LR_TAG}"
    OUTPUT_DIR="${OUTPUT_BASE}/${EXP_ID}"
    echo "[SFT] ${EXP_ID}"
    python train_single_task.py \
        --method sft \
        --model_name ${PREV_OUTPUT} \
        --task mbpp \
        --output_dir ${OUTPUT_DIR} \
        --exp_id ${EXP_ID} \
        --learning_rate ${LR} \
        --num_train_epochs ${NUM_EPOCHS} \
        --gradient_accumulation_steps ${GRAD_ACCUM_STEPS}

    echo "SFT ${LR_TAG} complete on GPU ${GPU_ID}!"
}

echo "=============================================="
echo "Ablation Study: alpha x lr"
echo "Model: ${MODEL_BASE}"
echo "GPUs: 0, 1, 2, 3"
echo "=============================================="

# Launch 4 SDFT ablations in parallel on GPUs 0-3
run_sdft 0 "3e-5" "0.05" "lr3e5" "a05" &
run_sdft 1 "5e-5" "0.05" "lr5e5" "a05" &
run_sdft 2 "3e-5" "0.01" "lr3e5" "a01" &
run_sdft 3 "5e-5" "0.01" "lr5e5" "a01" &

# Wait for all SDFT ablations to complete
wait

echo ""
echo "All SDFT ablations complete!"

# Run SFT baseline on GPU 0
run_sft 0 "3e-5" "lr3e5"

echo ""
echo "=============================================="
echo "All ablations complete!"
echo "Results in ${OUTPUT_BASE}/sdft_* and ${OUTPUT_BASE}/sft_*"
echo "=============================================="
