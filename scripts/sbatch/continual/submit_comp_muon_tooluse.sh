#!/bin/bash
# Run CompositionPreservingMuon DFT on tooluse (single task, no chain).
#
# Usage:
#   bash scripts/sbatch/continual/submit_comp_muon_tooluse.sh [lr] [energy]
#
# Examples:
#   bash scripts/sbatch/continual/submit_comp_muon_tooluse.sh 1e-3
#   bash scripts/sbatch/continual/submit_comp_muon_tooluse.sh 1e-3 0.9

set -e

LR=${1:-1e-3}
ENERGY=${2:-0.8}

LR_TAG=$(echo ${LR} | tr -d '-')        # 1e-3 -> 1e3
ENERGY_TAG=$(echo ${ENERGY} | tr -d '.') # 0.8  -> 08

BASE_DIR="/home/guests/saket/continual/Self-Distillation"
EXP_ID="qwen3_cl_comp_muon_e${ENERGY_TAG}_${LR_TAG}"
BASE_OUTPUT="${BASE_DIR}/outputs/qwen3/${EXP_ID}"
LOGS="${BASE_DIR}/logs"

mkdir -p ${LOGS}

echo "=============================================="
echo "Submitting CompositionPreservingMuon tooluse"
echo "lr: ${LR}  qk_energy: ${ENERGY}  ov_energy: ${ENERGY}"
echo "Exp: ${EXP_ID}"
echo "Output: ${BASE_OUTPUT}/task_0_tooluse"
echo "=============================================="

JOB=$(sbatch --parsable \
    --job-name=comp_muon_tooluse \
    --output=${LOGS}/comp_muon_tooluse_%j.out \
    --error=${LOGS}/comp_muon_tooluse_%j.err \
    --gres=gpu:1 --mem=80G --cpus-per-task=8 --time=12:00:00 \
    --wrap="
cd ${BASE_DIR}
export HF_HOME=/home/guests/saket/models
export WANDB_PROJECT=qwen3-continual-learning
export WANDB_API_KEY=\$(cat '${BASE_DIR}/scripts/wandb_token.out' | tr -d '[:space:]')
uv run python train_qwen3_dft.py \
    --method dft \
    --task tooluse \
    --model_name Qwen/Qwen3-4B \
    --output_dir ${BASE_OUTPUT}/task_0_tooluse \
    --exp_id ${EXP_ID} \
    --use_comp_muon \
    --comp_qk_energy_threshold ${ENERGY} \
    --comp_ov_energy_threshold ${ENERGY} \
    --comp_dual_tol 1e-4 \
    --comp_no_skip_if_fallback_fails \
    --comp_adamw_lr 3e-4 \
    --comp_clip_gradient \
    --comp_delta_norm_cap 1.0 \
    --comp_debug \
    --learning_rate ${LR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --save_strategy no \
    --eval_all_tasks \
    --eval_tasks spider,tooluse,cot_math \
    --eval_max_samples 50
")

echo "Submitted job ${JOB}"
echo "  tail -f ${LOGS}/comp_muon_tooluse_${JOB}.out"
