#!/bin/bash
#SBATCH --job-name=dft_cl_spider
#SBATCH --output=logs/dft_cl_spider_%j.out
#SBATCH --error=logs/dft_cl_spider_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

# DFT + ProjectedGradient continual chain — Task 3/3: spider
# Starts from task_1_cot_math checkpoint

cd /home/guests/saket/continual/Self-Distillation
mkdir -p logs

export HF_HOME="/home/guests/saket/models"
export WANDB_PROJECT="qwen3-continual-learning"

export WANDB_API_KEY=$(cat "/home/guests/saket/continual/Self-Distillation/scripts/wandb_token.out" | tr -d '[:space:]')

EXP_ID="qwen3_cl_dft_proj_e06"
BASE_OUTPUT="/home/guests/saket/continual/Self-Distillation/outputs/qwen3/${EXP_ID}"

uv run python train_qwen3_dft.py \
    --method dft \
    --task spider \
    --model_name ${BASE_OUTPUT}/task_1_cot_math \
    --output_dir ${BASE_OUTPUT}/task_2_spider \
    --exp_id ${EXP_ID} \
    --learning_rate 5e-3 \
    --proj_energy_threshold 0.6 \
    --proj_no_resync \
    --proj_grad_clip 1.0 \
    --proj_admm_steps 20 \
    --stiefel_update projected_qr \
    --stiefel_max_rms 0.03 \
    --uv_lr_scale 0.3 \
    --b_lr_scale 0.1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --save_strategy no \
    --skip_before_eval \
    --eval_all_tasks \
    --eval_tasks spider,tooluse,cot_math \
    --eval_max_samples 50

echo "Done: ${EXP_ID} task_2_spider"
