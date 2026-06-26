#!/bin/bash
# Submit full evaluation jobs for all three task checkpoints of a continual run.
#
# Usage:
#   bash scripts/sbatch/continual/submit_full_eval.sh <exp_id>
#
# Example:
#   bash scripts/sbatch/continual/submit_full_eval.sh qwen3_cl_fp_muon_full_proj_e08_1e3

set -e

EXP_ID=${1:?"Usage: $0 <exp_id>"}

BASE_DIR="/home/guests/saket/continual/Self-Distillation"
BASE_OUTPUT="${BASE_DIR}/outputs/qwen3/${EXP_ID}"
LOGS="${BASE_DIR}/logs"

SBATCH_COMMON="--parsable --gres=gpu:1 --mem=40G --cpus-per-task=8 --time=06:00:00"

mkdir -p ${LOGS}

echo "=============================================="
echo "Submitting full eval for: ${EXP_ID}"
echo "=============================================="

for TASK_DIR in task_0_tooluse task_1_cot_math task_2_spider; do
    TAG="after_${TASK_DIR#task_?_}"   # task_0_tooluse -> after_tooluse

    JOB=$(sbatch ${SBATCH_COMMON} \
        --job-name=eval_${TAG} \
        --output=${LOGS}/eval_${TAG}_%j.out \
        --error=${LOGS}/eval_${TAG}_%j.err \
        --wrap="
cd ${BASE_DIR}
export HF_HOME=/home/guests/saket/models
export WANDB_PROJECT=qwen3-continual-learning
export WANDB_API_KEY=\$(cat '${BASE_DIR}/scripts/wandb_token.out' | tr -d '[:space:]')
uv run python scripts/eval_single_checkpoint.py \
    --model_path ${BASE_OUTPUT}/${TASK_DIR} \
    --tag ${TAG} \
    --exp_id ${EXP_ID} \
    --eval_tasks spider,tooluse,cot_math \
    --max_samples 10000
")
    echo "  ${TAG}: job ${JOB}  (${BASE_OUTPUT}/${TASK_DIR})"
done

echo "=============================================="
