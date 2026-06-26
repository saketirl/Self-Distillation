#!/bin/bash
# Evaluate a single model checkpoint on all three continual learning tasks.
#
# Usage:
#   bash scripts/sbatch/continual/eval_checkpoint.sh <model_path> <tag> [exp_id]
#
# Examples:
#   bash scripts/sbatch/continual/eval_checkpoint.sh \
#       outputs/qwen3/qwen3_cl_fp_muon_full_proj_e08_1e3/task_0_tooluse \
#       after_tooluse \
#       qwen3_cl_fp_muon_full_proj_e08_1e3
#
#   bash scripts/sbatch/continual/eval_checkpoint.sh \
#       outputs/qwen3/qwen3_cl_fp_muon_full_proj_e08_1e3/task_1_cot_math \
#       after_cot_math \
#       qwen3_cl_fp_muon_full_proj_e08_1e3
#
#   bash scripts/sbatch/continual/eval_checkpoint.sh \
#       outputs/qwen3/qwen3_cl_fp_muon_full_proj_e08_1e3/task_2_spider \
#       after_spider \
#       qwen3_cl_fp_muon_full_proj_e08_1e3

set -e

MODEL_PATH=${1:?"Usage: $0 <model_path> <tag> [exp_id]"}
TAG=${2:?"Usage: $0 <model_path> <tag> [exp_id]"}
EXP_ID=${3:-${TAG}}

BASE_DIR="/home/guests/saket/continual/Self-Distillation"
LOGS="${BASE_DIR}/logs"
mkdir -p ${LOGS}

JOB=$(sbatch --parsable \
    --job-name=eval_${TAG} \
    --output=${LOGS}/eval_${TAG}_%j.out \
    --error=${LOGS}/eval_${TAG}_%j.err \
    --gres=gpu:1 \
    --mem=40G \
    --cpus-per-task=8 \
    --time=06:00:00 \
    --wrap="
cd ${BASE_DIR}
export HF_HOME=/home/guests/saket/models
export WANDB_PROJECT=qwen3-continual-learning
export WANDB_API_KEY=\$(cat '${BASE_DIR}/scripts/wandb_token.out' | tr -d '[:space:]')
uv run python scripts/eval_single_checkpoint.py \
    --model_path ${MODEL_PATH} \
    --tag ${TAG} \
    --exp_id ${EXP_ID} \
    --eval_tasks spider,tooluse,cot_math \
    --max_samples 10000
")

echo "Submitted eval job: ${JOB}  model=${MODEL_PATH}  tag=${TAG}"
echo "  tail -f ${LOGS}/eval_${TAG}_${JOB}.out"
