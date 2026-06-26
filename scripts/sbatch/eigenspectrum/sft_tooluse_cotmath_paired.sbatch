#!/bin/bash
#SBATCH --job-name=spec_sft_t01
#SBATCH --output=/home/guests/saket/continual/Self-Distillation/logs/spec_sft_t01_%a_%j.out
#SBATCH --error=/home/guests/saket/continual/Self-Distillation/logs/spec_sft_t01_%a_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --array=1

set -e

BASE_DIR="/home/guests/saket/continual/Self-Distillation"
MODEL_T0="${BASE_DIR}/outputs/qwen3/qwen3_cl_sft_adamw/task_0_tooluse"
MODEL_T1="${BASE_DIR}/outputs/qwen3/qwen3_cl_sft_adamw/task_1_cot_math"
OUTPUT_T0="${BASE_DIR}/eigenspectrum/outputs/qwen3/sft/task_0_tooluse"
OUTPUT_T1="${BASE_DIR}/eigenspectrum/outputs/qwen3/sft/task_1_cot_math"
NUM_LAYERS=36

mkdir -p "${OUTPUT_T0}" "${OUTPUT_T1}" "${BASE_DIR}/logs"

for MP in "${MODEL_T0}" "${MODEL_T1}"; do
    if [ ! -d "${MP}" ]; then
        echo "ERROR: Model path not found: ${MP}"
        exit 1
    fi
done

cd "${BASE_DIR}"
export HF_HOME=/home/guests/saket/models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

LAYER=${SLURM_ARRAY_TASK_ID}
echo "========================================"
echo "Layer ${LAYER} / $((NUM_LAYERS - 1)) — running task_0 + task_1 in parallel"
echo "========================================"

# Launch task_0 in background
OUT0="${OUTPUT_T0}/layer_${LAYER}"
mkdir -p "${OUT0}"
if [ -f "${OUT0}/summary.json" ]; then
    echo "task_0 layer ${LAYER}: already done, skipping."
else
    uv run python -m eigenspectrum.example_analyze \
        --model_path "${MODEL_T0}" \
        --layer "${LAYER}" \
        --output_dir "${OUT0}" \
        --data_tasks tooluse,spider,cot_math \
        --num_data_samples 500 \
        --lanczos_order 50 \
        --num_samples 10 \
        --batch_size 4 \
        --device cuda \
        --use_bfloat16 \
        --use_gradient_checkpointing &
    PID0=$!
fi

# Launch task_1 in background
OUT1="${OUTPUT_T1}/layer_${LAYER}"
mkdir -p "${OUT1}"
if [ -f "${OUT1}/summary.json" ]; then
    echo "task_1 layer ${LAYER}: already done, skipping."
else
    uv run python -m eigenspectrum.example_analyze \
        --model_path "${MODEL_T1}" \
        --layer "${LAYER}" \
        --output_dir "${OUT1}" \
        --data_tasks tooluse,spider,cot_math \
        --num_data_samples 500 \
        --lanczos_order 50 \
        --num_samples 10 \
        --batch_size 4 \
        --device cuda \
        --use_bfloat16 \
        --use_gradient_checkpointing &
    PID1=$!
fi

# Wait for both and propagate any failure
FAILED=0
for PID in ${PID0:-} ${PID1:-}; do
    wait "${PID}" || FAILED=1
done
[ ${FAILED} -eq 0 ] || { echo "ERROR: one or more tasks failed for layer ${LAYER}"; exit 1; }

echo "Layer ${LAYER} done (task_0 + task_1)."
