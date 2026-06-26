#!/bin/bash
#SBATCH --job-name=spec_sft_t2
#SBATCH --output=/home/guests/saket/continual/Self-Distillation/logs/spec_sft_t2_%a_%j.out
#SBATCH --error=/home/guests/saket/continual/Self-Distillation/logs/spec_sft_t2_%a_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-35

set -e

BASE_DIR="/home/guests/saket/continual/Self-Distillation"
MODEL_PATH="${BASE_DIR}/outputs/qwen3/qwen3_cl_sft_adamw/task_2_spider"
OUTPUT_BASE="${BASE_DIR}/eigenspectrum/outputs/qwen3/sft/task_2_spider"
NUM_LAYERS=36

mkdir -p "${OUTPUT_BASE}" "${BASE_DIR}/logs"

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model path not found: ${MODEL_PATH}"
    exit 1
fi

cd "${BASE_DIR}"
export HF_HOME=/home/guests/saket/models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

LAYER=${SLURM_ARRAY_TASK_ID}
OUTPUT_DIR="${OUTPUT_BASE}/layer_${LAYER}"
mkdir -p "${OUTPUT_DIR}"

if [ -f "${OUTPUT_DIR}/summary.json" ]; then
    echo "Layer ${LAYER}: already done, skipping."
    exit 0
fi

echo "========================================"
echo "Layer ${LAYER} / $((NUM_LAYERS - 1))"
echo "========================================"

uv run python -m eigenspectrum.example_analyze \
    --model_path "${MODEL_PATH}" \
    --layer "${LAYER}" \
    --output_dir "${OUTPUT_DIR}" \
    --data_tasks tooluse,spider,cot_math \
    --num_data_samples 500 \
    --lanczos_order 50 \
    --num_samples 10 \
    --batch_size 4 \
    --device cuda \
    --use_bfloat16 \
    --use_gradient_checkpointing

echo "Layer ${LAYER} done."

