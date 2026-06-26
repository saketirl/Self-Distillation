#!/bin/bash
# FeaturePreservingMuon continual chain — alpha=0.0 (full projection).
# Chains 3 SLURM jobs: tooluse -> cot_math -> spider.
# Every gradient update is constrained orthogonal to V_r (no bypass).
#
# Usage:
#   bash submit_fp_muon_full_proj_chain.sh <energy_threshold> <lr>
#   bash submit_fp_muon_full_proj_chain.sh 0.8 1e-3

set -e

ENERGY=${1:?"Usage: $0 <energy_threshold> <lr>  e.g. 0.8 1e-3"}
LR=${2:?"Usage: $0 <energy_threshold> <lr>  e.g. 0.8 1e-3"}

ENERGY_TAG=$(echo ${ENERGY} | tr -d '.')   # 0.8 -> 08
LR_TAG=$(echo ${LR} | tr -d '-')           # 1e-3 -> 1e3

BASE_DIR="/home/guests/saket/continual/Self-Distillation"
EXP_ID="qwen3_cl_fp_muon_full_proj_e${ENERGY_TAG}_${LR_TAG}"
BASE_OUTPUT="${BASE_DIR}/outputs/qwen3/${EXP_ID}"
LOGS="${BASE_DIR}/logs"

SBATCH_COMMON="--parsable --gres=gpu:1 --mem=80G --cpus-per-task=8 --time=12:00:00"

echo "=============================================="
echo "Submitting FeaturePreservingMuon full-projection chain"
echo "Energy threshold: ${ENERGY}  lr: ${LR}  bypass_alpha: 0.0"
echo "Exp: ${EXP_ID}"
echo "Tasks: tooluse -> cot_math -> spider"
echo "Output: ${BASE_OUTPUT}"
echo "=============================================="

mkdir -p ${LOGS}

# ── Task 1/3: tooluse ────────────────────────────────────────────────────────
JOB1=$(sbatch ${SBATCH_COMMON} \
    --job-name=fp_proj_e${ENERGY_TAG}_tooluse \
    --output=${LOGS}/fp_proj_e${ENERGY_TAG}_tooluse_%j.out \
    --error=${LOGS}/fp_proj_e${ENERGY_TAG}_tooluse_%j.err \
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
    --use_fp_muon \
    --fp_energy_threshold ${ENERGY} \
    --fp_n_dual_iter 10 \
    --fp_dual_step_size 0.01 \
    --fp_vr_bypass_alpha 0.0 \
    --fp_debug \
    --learning_rate ${LR} \
    --fp_adamw_lr 3e-4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --save_strategy no \
    --eval_all_tasks \
    --eval_tasks spider,tooluse,cot_math \
    --eval_max_samples 50
")
echo "Submitted task 1 (tooluse):   job ${JOB1}"

# ── Task 2/3: cot_math ───────────────────────────────────────────────────────
JOB2=$(sbatch ${SBATCH_COMMON} \
    --job-name=fp_proj_e${ENERGY_TAG}_cotmath \
    --output=${LOGS}/fp_proj_e${ENERGY_TAG}_cotmath_%j.out \
    --error=${LOGS}/fp_proj_e${ENERGY_TAG}_cotmath_%j.err \
    --dependency=afterok:${JOB1} \
    --wrap="
cd ${BASE_DIR}
export HF_HOME=/home/guests/saket/models
export WANDB_PROJECT=qwen3-continual-learning
export WANDB_API_KEY=\$(cat '${BASE_DIR}/scripts/wandb_token.out' | tr -d '[:space:]')
uv run python train_qwen3_dft.py \
    --method dft \
    --task cot_math \
    --model_name ${BASE_OUTPUT}/task_0_tooluse \
    --output_dir ${BASE_OUTPUT}/task_1_cot_math \
    --exp_id ${EXP_ID} \
    --use_fp_muon \
    --fp_energy_threshold ${ENERGY} \
    --fp_n_dual_iter 10 \
    --fp_dual_step_size 0.01 \
    --fp_vr_bypass_alpha 0.0 \
    --fp_debug \
    --learning_rate ${LR} \
    --fp_adamw_lr 3e-4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --save_strategy no \
    --skip_before_eval \
    --eval_all_tasks \
    --eval_tasks spider,tooluse,cot_math \
    --eval_max_samples 50
")
echo "Submitted task 2 (cot_math):  job ${JOB2}  [after ${JOB1}]"

# ── Task 3/3: spider ─────────────────────────────────────────────────────────
JOB3=$(sbatch ${SBATCH_COMMON} \
    --job-name=fp_proj_e${ENERGY_TAG}_spider \
    --output=${LOGS}/fp_proj_e${ENERGY_TAG}_spider_%j.out \
    --error=${LOGS}/fp_proj_e${ENERGY_TAG}_spider_%j.err \
    --dependency=afterok:${JOB2} \
    --wrap="
cd ${BASE_DIR}
export HF_HOME=/home/guests/saket/models
export WANDB_PROJECT=qwen3-continual-learning
export WANDB_API_KEY=\$(cat '${BASE_DIR}/scripts/wandb_token.out' | tr -d '[:space:]')
uv run python train_qwen3_dft.py \
    --method dft \
    --task spider \
    --model_name ${BASE_OUTPUT}/task_1_cot_math \
    --output_dir ${BASE_OUTPUT}/task_2_spider \
    --exp_id ${EXP_ID} \
    --use_fp_muon \
    --fp_energy_threshold ${ENERGY} \
    --fp_n_dual_iter 10 \
    --fp_dual_step_size 0.01 \
    --fp_vr_bypass_alpha 0.0 \
    --fp_debug \
    --learning_rate ${LR} \
    --fp_adamw_lr 3e-4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --save_strategy no \
    --skip_before_eval \
    --eval_all_tasks \
    --eval_tasks spider,tooluse,cot_math \
    --eval_max_samples 50
")
echo "Submitted task 3 (spider):    job ${JOB3}  [after ${JOB2}]"

echo ""
echo "=============================================="
echo "All 3 jobs queued. Monitor with:"
echo "  squeue -j ${JOB1},${JOB2},${JOB3}"
echo "  tail -f ${LOGS}/fp_proj_e${ENERGY_TAG}_tooluse_${JOB1}.out"
echo "=============================================="
