#!/bin/bash
# Run SoftCompPreservingMuon+FPMuon DFT on tooluse (step 1 of continual learning).
#
# Attention (q/k/v/o): SoftCompPreservingMuon
#   - Preserves the k×k soft core of W_Q^T W_K and W_O W_V per head
#   - k chosen by energy threshold: min{j : sum(S[:j]^2)/sum(S^2) >= threshold}
#   - Constraint enforced via Sylvester equation (one-shot closed-form projection)
# MLP (gate/up/down): FeaturePreservingMuon
#   - Projects gradient orthogonal to right singular dirs capturing mlp_energy
# Scalars / embed / lm_head: AdamW
#
# Usage:
#   bash scripts/sbatch/continual/submit_soft_comp_muon_tooluse.sh [lr] [qk_energy] [ov_energy] [mlp_energy]
#
# Examples:
#   bash scripts/sbatch/continual/submit_soft_comp_muon_tooluse.sh
#   bash scripts/sbatch/continual/submit_soft_comp_muon_tooluse.sh 1e-3 0.8 0.8 0.8
#   bash scripts/sbatch/continual/submit_soft_comp_muon_tooluse.sh 5e-4 0.9 0.7 0.7

set -e

LR=${1:-1e-3}
QK_ENERGY=${2:-0.5}
OV_ENERGY=${3:-0.5}
MLP_ENERGY=${4:-0.5}

LR_TAG=$(echo ${LR} | tr -d '-')       # 1e-3 -> 1e3
QK_TAG=$(echo ${QK_ENERGY} | tr -d '.') # 0.8  -> 08
OV_TAG=$(echo ${OV_ENERGY} | tr -d '.')
MLP_TAG=$(echo ${MLP_ENERGY} | tr -d '.')

BASE_DIR="/home/guests/saket/continual/Self-Distillation"
EXP_ID="qwen3_cl_softcomp_qk${QK_TAG}_ov${OV_TAG}_mlp${MLP_TAG}_${LR_TAG}"
OUTPUT_DIR="${BASE_DIR}/outputs/qwen3/${EXP_ID}/task_0_tooluse"
LOGS="${BASE_DIR}/logs"

mkdir -p ${LOGS}

echo "=============================================="
echo "SoftCompPreservingMuon + FPMuon — tooluse"
echo "lr: ${LR}  qk_energy: ${QK_ENERGY}  ov_energy: ${OV_ENERGY}  mlp_energy: ${MLP_ENERGY}"
echo "Exp: ${EXP_ID}"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="

JOB=$(sbatch --parsable \
    --job-name=softcomp_tooluse \
    --output=${LOGS}/softcomp_tooluse_%j.out \
    --error=${LOGS}/softcomp_tooluse_%j.err \
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
    --output_dir ${OUTPUT_DIR} \
    --exp_id ${EXP_ID} \
    --use_soft_comp_muon \
    --soft_comp_head_dim 80 \
    --soft_comp_num_kv_heads 8 \
    --soft_comp_qk_energy_threshold ${QK_ENERGY} \
    --soft_comp_ov_energy_threshold ${OV_ENERGY} \
    --soft_comp_fp_energy_threshold ${MLP_ENERGY} \
    --soft_comp_adamw_lr 3e-4 \
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
    --eval_all_tasks \
    --eval_tasks spider,tooluse,cot_math \
    --eval_max_samples 50
")

echo "Submitted job ${JOB}"
echo "  tail -f ${LOGS}/softcomp_tooluse_${JOB}.out"
