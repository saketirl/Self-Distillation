#!/bin/bash
# Grid search over projected gradient hyperparameters for DFT on three datasets.
#
# Grid:
#   --proj_energy_threshold : 0.5  (fixed)
#   --learning_rate         : 5e-4  1e-3  5e-3
#   --uv_lr_scale           : 0.1  0.3
#   --b_lr_scale            : 0.1  0.3
#
# Total: 3 x 2 x 2 = 12 combos per dataset x 3 datasets = 36 jobs
#
# Other params held fixed (extend the arrays below to sweep them too):
#   --stiefel_max_rms   0.03  (try 0.01 for more conservative manifold steps)
#   --proj_no_resync          (try removing to enable resync every 100 steps)
#   --proj_grad_clip    1.0
#
# Usage:
#   bash submit_grid_search.sh          # dry-run (prints commands)
#   bash submit_grid_search.sh --submit # actually submits to SLURM

set -e

SUBMIT=false
if [[ "$1" == "--submit" ]]; then
    SUBMIT=true
fi

BASE_DIR="/home/guests/saket/continual/Self-Distillation"
OUTPUT_BASE="${BASE_DIR}/outputs/qwen3/grid_search"

DATASETS=("spider" "tooluse" "limo")
ENERGY=0.5
LRS=("5e-4" "1e-3" "5e-3")
UV_LR_SCALES=(0.1 0.3)
B_LR_SCALES=(0.1 0.3)

# Fixed settings
ADMM_STEPS=20
ADMM_RHO=4.0
GRAD_CLIP=1.0
STIEFEL_UPDATE="projected_qr"
STIEFEL_MAX_RMS=0.03

n_jobs=0

for TASK in "${DATASETS[@]}"; do
    for LR in "${LRS[@]}"; do
        for UV_SCALE in "${UV_LR_SCALES[@]}"; do
            for B_SCALE in "${B_LR_SCALES[@]}"; do

                EXP_ID="gs_${TASK}_e${ENERGY}_lr${LR}_uv${UV_SCALE}_b${B_SCALE}"
                EXP_ID="${EXP_ID//./_}"
                OUTPUT_DIR="${OUTPUT_BASE}/${EXP_ID}"

                SBATCH_SCRIPT=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=${EXP_ID}
#SBATCH --output=${BASE_DIR}/logs/${EXP_ID}_%j.out
#SBATCH --error=${BASE_DIR}/logs/${EXP_ID}_%j.err
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

cd ${BASE_DIR}
mkdir -p logs

export HF_HOME="/home/guests/saket/models"
export WANDB_PROJECT="qwen3-continual-learning"

uv run wandb login "$(cat "/home/guests/saket/continual/Self-Distillation/scripts/wandb_token.out" | tr -d '[:space:]')"
uv run python train_qwen3_dft.py \\
    --method dft \\
    --task ${TASK} \\
    --model_name Qwen/Qwen3-4B \\
    --output_dir ${OUTPUT_DIR} \\
    --exp_id ${EXP_ID} \\
    --learning_rate ${LR} \\
    --proj_energy_threshold ${ENERGY} \\
    --proj_no_resync \\
    --proj_grad_clip ${GRAD_CLIP} \\
    --proj_admm_steps ${ADMM_STEPS} \\
    --proj_admm_rho ${ADMM_RHO} \\
    --stiefel_update ${STIEFEL_UPDATE} \\
    --stiefel_max_rms ${STIEFEL_MAX_RMS} \\
    --uv_lr_scale ${UV_SCALE} \\
    --b_lr_scale ${B_SCALE} \\
    --num_train_epochs 1 \\
    --per_device_train_batch_size 1 \\
    --gradient_accumulation_steps 64 \\
    --skip_before_eval \\
    --eval_all_tasks \\
    --eval_max_samples 50 \\
    --save_strategy no

echo "Done: ${EXP_ID}"
EOF
)

                if $SUBMIT; then
                    echo "$SBATCH_SCRIPT" | sbatch
                    echo "Submitted: ${EXP_ID}"
                else
                    echo "--- DRY RUN: ${EXP_ID} ---"
                    echo "  task=${TASK}  energy=${ENERGY}  lr=${LR}  uv_lr_scale=${UV_SCALE}  b_lr_scale=${B_SCALE}"
                    echo "  output=${OUTPUT_DIR}"
                fi

                n_jobs=$((n_jobs + 1))
            done
        done
    done
done

echo ""
echo "Total jobs: ${n_jobs}"
if ! $SUBMIT; then
    echo "Dry run complete. Run with --submit to actually submit."
fi
