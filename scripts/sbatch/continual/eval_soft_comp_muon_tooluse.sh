#!/bin/bash
# Full evaluation of the task_0_tooluse SoftCompMuon checkpoint on all three tasks.
#
# Usage:
#   bash scripts/sbatch/continual/eval_soft_comp_muon_tooluse.sh [lr] [qk_energy] [ov_energy] [mlp_energy]
#
# Examples:
#   bash scripts/sbatch/continual/eval_soft_comp_muon_tooluse.sh
#   bash scripts/sbatch/continual/eval_soft_comp_muon_tooluse.sh 1e-3 0.5 0.5 0.8

set -e

LR=${1:-1e-3}
QK_ENERGY=${2:-0.5}
OV_ENERGY=${3:-0.5}
MLP_ENERGY=${4:-0.5}

LR_TAG=$(echo ${LR} | tr -d '-')
QK_TAG=$(echo ${QK_ENERGY} | tr -d '.')
OV_TAG=$(echo ${OV_ENERGY} | tr -d '.')
MLP_TAG=$(echo ${MLP_ENERGY} | tr -d '.')

BASE_DIR="/home/guests/saket/continual/Self-Distillation"
EXP_ID="qwen3_cl_softcomp_qk${QK_TAG}_ov${OV_TAG}_mlp${MLP_TAG}_${LR_TAG}"
MODEL_PATH="${BASE_DIR}/outputs/qwen3/${EXP_ID}/task_0_tooluse"
LOGS="${BASE_DIR}/logs"

mkdir -p ${LOGS}

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: checkpoint not found at ${MODEL_PATH}"
    exit 1
fi

TAG="softcomp_after_tooluse_${LR_TAG}"

echo "=============================================="
echo "Full eval — SoftCompMuon task_0_tooluse"
echo "Model: ${MODEL_PATH}"
echo "Tasks: tooluse, cot_math, spider (full sets)"
echo "Tag:   ${TAG}"
echo "=============================================="

JOB=$(sbatch --parsable \
    --job-name=eval_softcomp_tooluse \
    --output=${LOGS}/eval_softcomp_tooluse_%j.out \
    --error=${LOGS}/eval_softcomp_tooluse_%j.err \
    --gres=gpu:1 --mem=40G --cpus-per-task=8 --time=06:00:00 \
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

echo "Submitted job ${JOB}"
echo "  tail -f ${LOGS}/eval_softcomp_tooluse_${JOB}.out"
