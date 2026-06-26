#!/bin/bash
# Analyze Stiefel manifold distance for all models

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EIGENSPECTRUM_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$EIGENSPECTRUM_DIR")"

# Base directory for model checkpoints (can be overridden via environment variable)
DATA_ROOT="${DATA_ROOT:-/data/saket/continual/Self-Distillation}"

OUTPUT_DIR="${EIGENSPECTRUM_DIR}/outputs/stiefel"
mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_ROOT"
echo "Working directory: $(pwd)"
echo "Data root (checkpoints): ${DATA_ROOT}"

# Helper function to check if path exists
check_path() {
    if [ ! -d "$1" ]; then
        echo "WARNING: Path does not exist: $1"
        echo "Skipping..."
        return 1
    fi
    return 0
}

# Base model (from HuggingFace)
echo "===== Analyzing Base Model ====="
python -m eigenspectrum.stiefel_distance \
    --model_path "Qwen/Qwen2.5-3B-Instruct" \
    --output_path "${OUTPUT_DIR}/base_model.json" \
    --device cuda

# SFT checkpoints
echo "===== Analyzing SFT Task 0 (ToolUse) ====="
SFT_T0="${DATA_ROOT}/outputs/sft_full_mbpp/task_0_tooluse/checkpoint-127"
if check_path "$SFT_T0"; then
    python -m eigenspectrum.stiefel_distance \
        --model_path "$SFT_T0" \
        --output_path "${OUTPUT_DIR}/sft_task0_tooluse.json" \
        --device cuda
fi

echo "===== Analyzing SFT Task 1 (GSM8K) ====="
SFT_T1="${DATA_ROOT}/outputs/sft_full_mbpp/task_1_gsm8k/checkpoint-234"
if check_path "$SFT_T1"; then
    python -m eigenspectrum.stiefel_distance \
        --model_path "$SFT_T1" \
        --output_path "${OUTPUT_DIR}/sft_task1_gsm8k.json" \
        --device cuda
fi

echo "===== Analyzing SFT Task 2 (MBPP) ====="
SFT_T2="${DATA_ROOT}/outputs/sft_full_mbpp/task_2_mbpp/checkpoint-12"
if check_path "$SFT_T2"; then
    python -m eigenspectrum.stiefel_distance \
        --model_path "$SFT_T2" \
        --output_path "${OUTPUT_DIR}/sft_task2_mbpp.json" \
        --device cuda
fi

# SDFT checkpoints
echo "===== Analyzing SDFT Task 0 (ToolUse) ====="
SDFT_T0="${DATA_ROOT}/outputs/sdft_full_mbpp/task_0_tooluse/checkpoint-1011"
if check_path "$SDFT_T0"; then
    python -m eigenspectrum.stiefel_distance \
        --model_path "$SDFT_T0" \
        --output_path "${OUTPUT_DIR}/sdft_task0_tooluse.json" \
        --device cuda
fi

echo "===== Analyzing SDFT Task 1 (GSM8K) ====="
SDFT_T1="${DATA_ROOT}/outputs/sdft_full_mbpp/task_1_gsm8k/checkpoint-1868"
if check_path "$SDFT_T1"; then
    python -m eigenspectrum.stiefel_distance \
        --model_path "$SDFT_T1" \
        --output_path "${OUTPUT_DIR}/sdft_task1_gsm8k.json" \
        --device cuda
fi

echo "===== Analyzing SDFT Task 2 (MBPP) ====="
SDFT_T2="${DATA_ROOT}/outputs/sdft_full_mbpp/task_2_mbpp/checkpoint-93"
if check_path "$SDFT_T2"; then
    python -m eigenspectrum.stiefel_distance \
        --model_path "$SDFT_T2" \
        --output_path "${OUTPUT_DIR}/sdft_task2_mbpp.json" \
        --device cuda
fi

# Generate visualizations (only if files exist)
echo "===== Generating Visualizations ====="
RESULTS_FILES=""
NAMES=""

if [ -f "${OUTPUT_DIR}/base_model.json" ]; then
    RESULTS_FILES="${OUTPUT_DIR}/base_model.json"
    NAMES="Base"
fi

if [ -f "${OUTPUT_DIR}/sft_task2_mbpp.json" ]; then
    RESULTS_FILES="${RESULTS_FILES} ${OUTPUT_DIR}/sft_task2_mbpp.json"
    NAMES="${NAMES} SFT_MBPP"
fi

if [ -f "${OUTPUT_DIR}/sdft_task2_mbpp.json" ]; then
    RESULTS_FILES="${RESULTS_FILES} ${OUTPUT_DIR}/sdft_task2_mbpp.json"
    NAMES="${NAMES} SDFT_MBPP"
fi

if [ -n "$RESULTS_FILES" ]; then
    python -m eigenspectrum.stiefel_visualize \
        --results $RESULTS_FILES \
        --names $NAMES \
        --output_dir "${OUTPUT_DIR}/plots" \
        --plot_type all
fi

echo "===== Done! ====="
echo "Results saved to: ${OUTPUT_DIR}"
