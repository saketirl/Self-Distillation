#!/bin/bash
# FeaturePreservingMuon continual chain: tooluse → cot_math → spider
# with full eval after every task.
#
# Energy thresholds ascend per task: 0.5 → 0.6 → 0.7
#
# Usage:
#   bash scripts/sbatch/continual/submit_fp_muon_chain.sh [lr] [bypass_alpha]
#
# Examples:
#   bash scripts/sbatch/continual/submit_fp_muon_chain.sh
#   bash scripts/sbatch/continual/submit_fp_muon_chain.sh 1e-3
#   bash scripts/sbatch/continual/submit_fp_muon_chain.sh 1e-3 0.1

set -e

LR=${1:-1e-3}
BYPASS=${2:-0.0}

ENERGY_T0=0.5
ENERGY_T1=0.6
ENERGY_T2=0.7

LR_TAG=$(echo ${LR} | tr -d '-')
BYPASS_TAG=$(echo ${BYPASS} | tr -d '.')

BASE_DIR="/home/guests/saket/continual/Self-Distillation"
EXP_ID="qwen3_cl_fp_muon_asc_e05e06e07_${LR_TAG}_b${BYPASS_TAG}"
BASE_OUTPUT="${BASE_DIR}/outputs/qwen3/${EXP_ID}"
LOGS="${BASE_DIR}/logs"

mkdir -p ${LOGS}

SBATCH_TRAIN="--parsable --gres=gpu:1 --mem=80G --cpus-per-task=8 --time=12:00:00"
SBATCH_EVAL="--parsable  --gres=gpu:1 --mem=40G --cpus-per-task=8 --time=06:00:00"

SETUP="cd ${BASE_DIR}; export HF_HOME=/home/guests/saket/models; export WANDB_PROJECT=qwen3-continual-learning; export WANDB_API_KEY=\$(cat '${BASE_DIR}/scripts/wandb_token.out' | tr -d '[:space:]')"

COMMON_ARGS="--method dft --use_fp_muon --fp_n_dual_iter 10 --fp_dual_step_size 0.01 --fp_vr_bypass_alpha ${BYPASS} --fp_debug --fp_adamw_lr 5e-5 --learning_rate ${LR} --max_prompt_length 1024 --max_completion_length 2048 --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 64 --save_strategy no --eval_tasks spider,tooluse,cot_math --eval_max_samples 50"

echo "=============================================="
echo "FeaturePreservingMuon chain: tooluse → cot_math → spider"
echo "lr=${LR}  bypass=${BYPASS}"
echo "Energy thresholds: tooluse=${ENERGY_T0}  cot_math=${ENERGY_T1}  spider=${ENERGY_T2}"
echo "Exp: ${EXP_ID}"
echo "Output: ${BASE_OUTPUT}"
echo "=============================================="

# ── Task 1/3: tooluse ────────────────────────────────────────────────────────
JOB1=$(sbatch ${SBATCH_TRAIN} \
    --job-name=fp_tooluse \
    --output=${LOGS}/fp_tooluse_%j.out \
    --error=${LOGS}/fp_tooluse_%j.err \
    --wrap="${SETUP}; uv run python train_qwen3_dft.py ${COMMON_ARGS} --fp_energy_threshold ${ENERGY_T0} --task tooluse --model_name Qwen/Qwen3-4B --output_dir ${BASE_OUTPUT}/task_0_tooluse --exp_id ${EXP_ID} --eval_all_tasks")
echo "Submitted task 1 (tooluse):  job ${JOB1}"

# ── Eval after tooluse ───────────────────────────────────────────────────────
EVAL1=$(sbatch ${SBATCH_EVAL} \
    --job-name=fp_eval_tooluse \
    --output=${LOGS}/fp_eval_tooluse_%j.out \
    --error=${LOGS}/fp_eval_tooluse_%j.err \
    --dependency=afterok:${JOB1} \
    --wrap="${SETUP}; uv run python scripts/eval_single_checkpoint.py --model_path ${BASE_OUTPUT}/task_0_tooluse --tag after_tooluse --exp_id ${EXP_ID} --eval_tasks spider,tooluse,cot_math --max_samples 10000")
echo "Submitted eval  1 (after tooluse):   job ${EVAL1}  [after ${JOB1}]"

# ── Task 2/3: cot_math ───────────────────────────────────────────────────────
JOB2=$(sbatch ${SBATCH_TRAIN} \
    --job-name=fp_cotmath \
    --output=${LOGS}/fp_cotmath_%j.out \
    --error=${LOGS}/fp_cotmath_%j.err \
    --dependency=afterok:${JOB1} \
    --wrap="${SETUP}; uv run python train_qwen3_dft.py ${COMMON_ARGS} --fp_energy_threshold ${ENERGY_T1} --task cot_math --model_name ${BASE_OUTPUT}/task_0_tooluse --output_dir ${BASE_OUTPUT}/task_1_cot_math --exp_id ${EXP_ID} --skip_before_eval --eval_all_tasks")
echo "Submitted task 2 (cot_math): job ${JOB2}  [after ${JOB1}]"

# ── Eval after cot_math ──────────────────────────────────────────────────────
EVAL2=$(sbatch ${SBATCH_EVAL} \
    --job-name=fp_eval_cotmath \
    --output=${LOGS}/fp_eval_cotmath_%j.out \
    --error=${LOGS}/fp_eval_cotmath_%j.err \
    --dependency=afterok:${JOB2} \
    --wrap="${SETUP}; uv run python scripts/eval_single_checkpoint.py --model_path ${BASE_OUTPUT}/task_1_cot_math --tag after_cot_math --exp_id ${EXP_ID} --eval_tasks spider,tooluse,cot_math --max_samples 10000")
echo "Submitted eval  2 (after cot_math):  job ${EVAL2}  [after ${JOB2}]"

# ── Task 3/3: spider ─────────────────────────────────────────────────────────
JOB3=$(sbatch ${SBATCH_TRAIN} \
    --job-name=fp_spider \
    --output=${LOGS}/fp_spider_%j.out \
    --error=${LOGS}/fp_spider_%j.err \
    --dependency=afterok:${JOB2} \
    --wrap="${SETUP}; uv run python train_qwen3_dft.py ${COMMON_ARGS} --fp_energy_threshold ${ENERGY_T2} --task spider --model_name ${BASE_OUTPUT}/task_1_cot_math --output_dir ${BASE_OUTPUT}/task_2_spider --exp_id ${EXP_ID} --skip_before_eval --eval_all_tasks")
echo "Submitted task 3 (spider):   job ${JOB3}  [after ${JOB2}]"

# ── Eval after spider ────────────────────────────────────────────────────────
EVAL3=$(sbatch ${SBATCH_EVAL} \
    --job-name=fp_eval_spider \
    --output=${LOGS}/fp_eval_spider_%j.out \
    --error=${LOGS}/fp_eval_spider_%j.err \
    --dependency=afterok:${JOB3} \
    --wrap="${SETUP}; uv run python scripts/eval_single_checkpoint.py --model_path ${BASE_OUTPUT}/task_2_spider --tag after_spider --exp_id ${EXP_ID} --eval_tasks spider,tooluse,cot_math --max_samples 10000")
echo "Submitted eval  3 (after spider):    job ${EVAL3}  [after ${JOB3}]"

echo ""
echo "=============================================="
echo "6 jobs queued (3 train + 3 eval)"
echo "  squeue -j ${JOB1},${EVAL1},${JOB2},${EVAL2},${JOB3},${EVAL3}"
echo "  tail -f ${LOGS}/fp_tooluse_${JOB1}.out"
echo "  tail -f ${LOGS}/fp_cotmath_${JOB2}.out"
echo "  tail -f ${LOGS}/fp_spider_${JOB3}.out"
echo "=============================================="
