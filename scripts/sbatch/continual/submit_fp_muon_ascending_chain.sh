#!/bin/bash
# FeaturePreservingMuon continual chain with ascending energy thresholds.
#
# Each task protects a larger fraction of the current model's feature subspace,
# forcing successive tasks to be learned in increasingly complementary directions:
#
#   Task 1 (tooluse):  energy=E1  — protect 50% of base model features
#   Task 2 (cot_math): energy=E2  — protect 60% of tooluse-adapted features
#   Task 3 (spider):   energy=E3  — protect 70% of cot_math-adapted features
#
# Usage:
#   bash scripts/sbatch/continual/submit_fp_muon_ascending_chain.sh [lr] [e1] [e2] [e3]
#
# Examples:
#   bash scripts/sbatch/continual/submit_fp_muon_ascending_chain.sh
#   bash scripts/sbatch/continual/submit_fp_muon_ascending_chain.sh 1e-3 0.5 0.6 0.7

set -e

LR=${1:-1e-3}
E1=${2:-0.5}   # tooluse
E2=${3:-0.6}   # cot_math
E3=${4:-0.7}   # spider

LR_TAG=$(echo ${LR} | tr -d '-')
E1_TAG=$(echo ${E1} | tr -d '.')
E2_TAG=$(echo ${E2} | tr -d '.')
E3_TAG=$(echo ${E3} | tr -d '.')

BASE_DIR="/home/guests/saket/continual/Self-Distillation"
EXP_ID="qwen3_cl_fp_muon_asc_e${E1_TAG}e${E2_TAG}e${E3_TAG}_${LR_TAG}"
BASE_OUTPUT="${BASE_DIR}/outputs/qwen3/${EXP_ID}"
LOGS="${BASE_DIR}/logs"

SBATCH_TRAIN="--parsable --gres=gpu:1 --mem=80G --cpus-per-task=8 --time=12:00:00"
SBATCH_EVAL="--parsable --gres=gpu:1 --mem=40G --cpus-per-task=8 --time=06:00:00"

echo "=============================================="
echo "FeaturePreservingMuon — ascending energy chain"
echo "lr: ${LR}"
echo "tooluse  energy: ${E1}"
echo "cot_math energy: ${E2}"
echo "spider   energy: ${E3}"
echo "Exp: ${EXP_ID}"
echo "Output: ${BASE_OUTPUT}"
echo "=============================================="

mkdir -p ${LOGS}

# ── Task 1/3: tooluse ────────────────────────────────────────────────────────
JOB1=$(sbatch ${SBATCH_TRAIN} \
    --job-name=fp_asc_tooluse \
    --output=${LOGS}/fp_asc_tooluse_%j.out \
    --error=${LOGS}/fp_asc_tooluse_%j.err \
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
    --fp_energy_threshold ${E1} \
    --fp_n_dual_iter 10 \
    --fp_dual_step_size 0.01 \
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
echo "Submitted task 1 (tooluse,  energy=${E1}): job ${JOB1}"

# ── Eval after tooluse ───────────────────────────────────────────────────────
EVAL1=$(sbatch ${SBATCH_EVAL} \
    --job-name=fp_asc_eval_tooluse \
    --output=${LOGS}/fp_asc_eval_tooluse_%j.out \
    --error=${LOGS}/fp_asc_eval_tooluse_%j.err \
    --dependency=afterok:${JOB1} \
    --wrap="
cd ${BASE_DIR}
export HF_HOME=/home/guests/saket/models
export WANDB_PROJECT=qwen3-continual-learning
export WANDB_API_KEY=\$(cat '${BASE_DIR}/scripts/wandb_token.out' | tr -d '[:space:]')
uv run python scripts/eval_single_checkpoint.py \
    --model_path ${BASE_OUTPUT}/task_0_tooluse \
    --tag after_tooluse \
    --exp_id ${EXP_ID} \
    --eval_tasks spider,tooluse,cot_math \
    --max_samples 10000
")
echo "Submitted eval  1 (after tooluse):          job ${EVAL1}  [after ${JOB1}]"

# ── Task 2/3: cot_math ───────────────────────────────────────────────────────
JOB2=$(sbatch ${SBATCH_TRAIN} \
    --job-name=fp_asc_cotmath \
    --output=${LOGS}/fp_asc_cotmath_%j.out \
    --error=${LOGS}/fp_asc_cotmath_%j.err \
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
    --fp_energy_threshold ${E2} \
    --fp_n_dual_iter 10 \
    --fp_dual_step_size 0.01 \
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
echo "Submitted task 2 (cot_math, energy=${E2}): job ${JOB2}  [after ${JOB1}]"

# ── Eval after cot_math ──────────────────────────────────────────────────────
EVAL2=$(sbatch ${SBATCH_EVAL} \
    --job-name=fp_asc_eval_cotmath \
    --output=${LOGS}/fp_asc_eval_cotmath_%j.out \
    --error=${LOGS}/fp_asc_eval_cotmath_%j.err \
    --dependency=afterok:${JOB2} \
    --wrap="
cd ${BASE_DIR}
export HF_HOME=/home/guests/saket/models
export WANDB_PROJECT=qwen3-continual-learning
export WANDB_API_KEY=\$(cat '${BASE_DIR}/scripts/wandb_token.out' | tr -d '[:space:]')
uv run python scripts/eval_single_checkpoint.py \
    --model_path ${BASE_OUTPUT}/task_1_cot_math \
    --tag after_cot_math \
    --exp_id ${EXP_ID} \
    --eval_tasks spider,tooluse,cot_math \
    --max_samples 10000
")
echo "Submitted eval  2 (after cot_math):         job ${EVAL2}  [after ${JOB2}]"

# ── Task 3/3: spider ─────────────────────────────────────────────────────────
JOB3=$(sbatch ${SBATCH_TRAIN} \
    --job-name=fp_asc_spider \
    --output=${LOGS}/fp_asc_spider_%j.out \
    --error=${LOGS}/fp_asc_spider_%j.err \
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
    --fp_energy_threshold ${E3} \
    --fp_n_dual_iter 10 \
    --fp_dual_step_size 0.01 \
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
echo "Submitted task 3 (spider,   energy=${E3}): job ${JOB3}  [after ${JOB2}]"

# ── Eval after spider ────────────────────────────────────────────────────────
EVAL3=$(sbatch ${SBATCH_EVAL} \
    --job-name=fp_asc_eval_spider \
    --output=${LOGS}/fp_asc_eval_spider_%j.out \
    --error=${LOGS}/fp_asc_eval_spider_%j.err \
    --dependency=afterok:${JOB3} \
    --wrap="
cd ${BASE_DIR}
export HF_HOME=/home/guests/saket/models
export WANDB_PROJECT=qwen3-continual-learning
export WANDB_API_KEY=\$(cat '${BASE_DIR}/scripts/wandb_token.out' | tr -d '[:space:]')
uv run python scripts/eval_single_checkpoint.py \
    --model_path ${BASE_OUTPUT}/task_2_spider \
    --tag after_spider \
    --exp_id ${EXP_ID} \
    --eval_tasks spider,tooluse,cot_math \
    --max_samples 10000
")
echo "Submitted eval  3 (after spider):           job ${EVAL3}  [after ${JOB3}]"

echo ""
echo "=============================================="
echo "6 jobs queued (3 train + 3 eval). Monitor with:"
echo "  squeue -j ${JOB1},${EVAL1},${JOB2},${EVAL2},${JOB3},${EVAL3}"
echo "  tail -f ${LOGS}/fp_asc_tooluse_${JOB1}.out"
echo "=============================================="
