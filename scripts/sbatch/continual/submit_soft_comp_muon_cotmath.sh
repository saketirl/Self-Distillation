#!/bin/bash
# Run SoftCompPreservingMuon+FPMuon DFT on cot_math (step 2 of continual learning).
# Loads the model saved by submit_soft_comp_muon_tooluse.sh from task_0_tooluse.
#
# Usage:  same args as tooluse script — must match to resolve the checkpoint path.
#   bash scripts/sbatch/continual/submit_soft_comp_muon_cotmath.sh [lr] [qk_energy] [ov_energy] [mlp_energy]
#
# Examples:
#   bash scripts/sbatch/continual/submit_soft_comp_muon_cotmath.sh
#   bash scripts/sbatch/continual/submit_soft_comp_muon_cotmath.sh 1e-3 0.8 0.8 0.8

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
TASK0_DIR="${BASE_DIR}/outputs/qwen3/${EXP_ID}/task_0_tooluse"
OUTPUT_DIR="${BASE_DIR}/outputs/qwen3/${EXP_ID}/task_1_cot_math"
LOGS="${BASE_DIR}/logs"

mkdir -p ${LOGS}

# Verify the task_0 checkpoint exists before submitting
if [ ! -d "${TASK0_DIR}" ]; then
    echo "ERROR: task_0 checkpoint not found at ${TASK0_DIR}"
    echo "Run submit_soft_comp_muon_tooluse.sh first with the same arguments."
    exit 1
fi

echo "=============================================="
echo "SoftCompPreservingMuon + FPMuon — cot_math"
echo "lr: ${LR}  qk_energy: ${QK_ENERGY}  ov_energy: ${OV_ENERGY}  mlp_energy: ${MLP_ENERGY}"
echo "Exp: ${EXP_ID}"
echo "Loading from: ${TASK0_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="

JOB=$(sbatch --parsable \
    --job-name=softcomp_cotmath \
    --output=${LOGS}/softcomp_cotmath_%j.out \
    --error=${LOGS}/softcomp_cotmath_%j.err \
    --gres=gpu:1 --mem=80G --cpus-per-task=8 --time=12:00:00 \
    --wrap="
cd ${BASE_DIR}
export HF_HOME=/home/guests/saket/models
export WANDB_PROJECT=qwen3-continual-learning
export WANDB_API_KEY=\$(cat '${BASE_DIR}/scripts/wandb_token.out' | tr -d '[:space:]')
uv run python train_qwen3_dft.py \
    --method dft \
    --task cot_math \
    --model_name ${TASK0_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --exp_id ${EXP_ID} \
    --use_soft_comp_muon \
    --soft_comp_head_dim 80 \
    --soft_comp_num_kv_heads 8 \
    --soft_comp_qk_energy_threshold ${QK_ENERGY} \
    --soft_comp_ov_energy_threshold ${OV_ENERGY} \
    --soft_comp_fp_energy_threshold ${MLP_ENERGY} \
    --soft_comp_adamw_lr 3e-4 \
    --soft_comp_fallback_tol 1e-2 \
    --soft_comp_hard_fallback_tol 0.5 \
    --soft_comp_sylvester_damping 1e-4 \
    --soft_comp_sylvester_skip_tol 5e-2 \
    --soft_comp_debug \
    --max_prompt_length 1024 \
    --max_completion_length 2048 \
    --learning_rate ${LR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --save_strategy no \
    --skip_before_eval \
    --eval_all_tasks \
    --eval_tasks spider,tooluse,cot_math \
    --eval_max_samples 50
")

echo "Submitted job ${JOB}"
echo "  tail -f ${LOGS}/softcomp_cotmath_${JOB}.out"
