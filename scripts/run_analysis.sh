#!/bin/bash
# Run checkpoint analysis on trained models
# Evaluates base model + 3 SDFT + 3 SFT checkpoints on all tasks

set -e

# Configuration - adjust these paths as needed
OUTPUT_BASE="${OUTPUT_BASE:-/data/saket/continual/Self-Distillation/outputs}"
EXP_ID="${EXP_ID:-full_mbpp}"
RESULTS_DIR="${RESULTS_DIR:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

# GPU settings
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "Checkpoint Analysis"
echo "=============================================="
echo "Output base: ${OUTPUT_BASE}"
echo "Experiment ID: ${EXP_ID}"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES}"
echo ""

# Build command
CMD="python scripts/analyze_checkpoints.py \
    --output_base ${OUTPUT_BASE} \
    --exp_id ${EXP_ID} \
    --max_new_tokens ${MAX_NEW_TOKENS}"

if [ -n "${RESULTS_DIR}" ]; then
    CMD="${CMD} --results_dir ${RESULTS_DIR}"
fi

# Run analysis
echo "Running: ${CMD}"
echo ""
eval ${CMD}

echo ""
echo "=============================================="
echo "Analysis complete!"
echo "=============================================="
