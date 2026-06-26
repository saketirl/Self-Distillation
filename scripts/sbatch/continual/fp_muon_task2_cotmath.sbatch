#!/bin/bash
#SBATCH --job-name=fp_muon_cotmath
#SBATCH --output=logs/fp_muon_cotmath_%j.out
#SBATCH --error=logs/fp_muon_cotmath_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

# FeaturePreservingMuon continual chain — Task 2/3: cot_math
# Loads checkpoint from task_0_tooluse. Requires task 1 to have completed.
# Env vars:  FP_ENERGY (default 0.8)  FP_LR (default 1e-3)

cd /home/guests/saket/continual/Self-Distillation
mkdir -p logs

export HF_HOME="/home/guests/saket/models"
export WANDB_PROJECT="qwen3-continual-learning"

export WANDB_API_KEY=$(cat "/home/guests/saket/continual/Self-Distillation/scripts/wandb_token.out" | tr -d '[:space:]')

ENERGY="${FP_ENERGY:-0.8}"
LR="${FP_LR:-1e-3}"
ENERGY_TAG=$(echo ${ENERGY} | tr -d '.')
LR_TAG=$(echo ${LR} | tr -d '-')
EXP_ID="${EXP_ID:-qwen3_cl_fp_muon_e${ENERGY_TAG}_${LR_TAG}}"
BASE_OUTPUT="/home/guests/saket/continual/Self-Distillation/outputs/qwen3/${EXP_ID}"

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

echo "Done: ${EXP_ID} task_1_cot_math"
