#!/bin/bash
# LR Sweep: 5e-6, 1e-5, 5e-5 with EMA alpha=0.01
# Tasks: tooluse -> gsm8k -> mbpp
# GPUs 1, 2, 3 for parallel runs
# Only saves checkpoint at end of each task (6 total per LR: 3 SDFT + 3 SFT)

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_BASE="Qwen/Qwen2.5-3B-Instruct"
OUTPUT_BASE="/data/saket/continual/Self-Distillation/outputs"

# Fixed hyperparameters
NUM_EPOCHS=1
GRAD_ACCUM_STEPS=32
EMA_ALPHA=0.01

# GPU memory threshold (in MB)
GPU_MEM_THRESHOLD=70000

gpu_available() {
    local GPU_ID=$1
    local FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${GPU_ID} 2>/dev/null | tr -d ' ')
    if [ -z "$FREE_MEM" ]; then
        return 1
    fi
    [ "$FREE_MEM" -gt "$GPU_MEM_THRESHOLD" ]
}

wait_for_gpu() {
    local GPU_ID=$1
    echo "Waiting for GPU ${GPU_ID} to be available..."
    while ! gpu_available ${GPU_ID}; do
        sleep 30
    done
    echo "GPU ${GPU_ID} is available!"
}

# Run SDFT then SFT for a given LR on a given GPU
run_experiment() {
    local GPU_ID=$1
    local LR=$2
    local LR_TAG=$3

    wait_for_gpu ${GPU_ID}
    export CUDA_VISIBLE_DEVICES=${GPU_ID}

    echo ""
    echo "========== GPU ${GPU_ID}: LR=${LR} =========="

    # ========== SDFT ==========
    echo ""
    echo "[SDFT ${LR_TAG}] Task 1/3: tooluse"
    python train_single_task.py \
        --method sdft \
        --model_name ${MODEL_BASE} \
        --task tooluse \
        --output_dir ${OUTPUT_BASE}/sdft_${LR_TAG}/tooluse \
        --exp_id "sweep_${LR_TAG}" \
        --learning_rate ${LR} \
        --num_train_epochs ${NUM_EPOCHS} \
        --ref_model_mixup_alpha ${EMA_ALPHA} \
        --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
        --save_strategy no \
        --vllm_gpu_memory_utilization 0.3

    echo ""
    echo "[SDFT ${LR_TAG}] Task 2/3: gsm8k"
    python train_single_task.py \
        --method sdft \
        --model_name ${OUTPUT_BASE}/sdft_${LR_TAG}/tooluse \
        --task gsm8k \
        --output_dir ${OUTPUT_BASE}/sdft_${LR_TAG}/gsm8k \
        --exp_id "sweep_${LR_TAG}" \
        --learning_rate ${LR} \
        --num_train_epochs ${NUM_EPOCHS} \
        --ref_model_mixup_alpha ${EMA_ALPHA} \
        --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
        --save_strategy no \
        --vllm_gpu_memory_utilization 0.3

    echo ""
    echo "[SDFT ${LR_TAG}] Task 3/3: mbpp"
    python train_single_task.py \
        --method sdft \
        --model_name ${OUTPUT_BASE}/sdft_${LR_TAG}/gsm8k \
        --task mbpp \
        --output_dir ${OUTPUT_BASE}/sdft_${LR_TAG}/mbpp \
        --exp_id "sweep_${LR_TAG}" \
        --learning_rate ${LR} \
        --num_train_epochs ${NUM_EPOCHS} \
        --ref_model_mixup_alpha ${EMA_ALPHA} \
        --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
        --save_strategy no \
        --vllm_gpu_memory_utilization 0.3

    echo ""
    echo "SDFT ${LR_TAG} complete!"

    # ========== SFT ==========
    echo ""
    echo "[SFT ${LR_TAG}] Task 1/3: tooluse"
    python train_single_task.py \
        --method sft \
        --model_name ${MODEL_BASE} \
        --task tooluse \
        --output_dir ${OUTPUT_BASE}/sft_${LR_TAG}/tooluse \
        --exp_id "sweep_${LR_TAG}" \
        --learning_rate ${LR} \
        --num_train_epochs ${NUM_EPOCHS} \
        --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
        --save_strategy no

    echo ""
    echo "[SFT ${LR_TAG}] Task 2/3: gsm8k"
    python train_single_task.py \
        --method sft \
        --model_name ${OUTPUT_BASE}/sft_${LR_TAG}/tooluse \
        --task gsm8k \
        --output_dir ${OUTPUT_BASE}/sft_${LR_TAG}/gsm8k \
        --exp_id "sweep_${LR_TAG}" \
        --learning_rate ${LR} \
        --num_train_epochs ${NUM_EPOCHS} \
        --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
        --save_strategy no

    echo ""
    echo "[SFT ${LR_TAG}] Task 3/3: mbpp"
    python train_single_task.py \
        --method sft \
        --model_name ${OUTPUT_BASE}/sft_${LR_TAG}/gsm8k \
        --task mbpp \
        --output_dir ${OUTPUT_BASE}/sft_${LR_TAG}/mbpp \
        --exp_id "sweep_${LR_TAG}" \
        --learning_rate ${LR} \
        --num_train_epochs ${NUM_EPOCHS} \
        --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
        --save_strategy no

    echo ""
    echo "SFT ${LR_TAG} complete!"
    echo "========== GPU ${GPU_ID}: ${LR_TAG} DONE =========="
}

echo "=============================================="
echo "LR Sweep: 5e-6, 1e-5, 5e-5"
echo "EMA Alpha: ${EMA_ALPHA}"
echo "Model: ${MODEL_BASE}"
echo "Tasks: tooluse -> gsm8k -> mbpp"
echo "GPUs: 1, 2, 3"
echo "=============================================="

# Launch 3 experiments in parallel on GPUs 1, 2, 3
run_experiment 1 "5e-6" "lr5e6" &
run_experiment 2 "1e-5" "lr1e5" &
run_experiment 3 "5e-5" "lr5e5" &

wait

echo ""
echo "=============================================="
echo "All LR sweep experiments complete!"
echo "Results in ${OUTPUT_BASE}/sdft_lr* and ${OUTPUT_BASE}/sft_lr*"
echo "=============================================="
