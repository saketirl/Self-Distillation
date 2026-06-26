#!/bin/bash
# Generate individual sbatch files for projected gradient sweep

mkdir -p scripts/sbatch
mkdir -p logs

# Higher LRs for projected gradient (constrained to low-rank subspace)
LRS=("1e-2" "1e-3" "1e-4")
RESYNCS=(50 100 200 500)

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
TASK="tooluse"
ENERGY_THRESHOLD=0.9
GRAD_CLIP=100
WORKDIR="/home/guests/saket/continual/Self-Distillation"
MODELS_DIR="/home/guests/saket/models"
WANDB_PROJECT="continual-learning-sdft"

for LR in "${LRS[@]}"; do
    for RESYNC in "${RESYNCS[@]}"; do
        EXP_ID="proj_lr${LR}_resync${RESYNC}"
        FILENAME="scripts/sbatch/proj_lr${LR}_resync${RESYNC}.sbatch"

        cat > "$FILENAME" << EOF
#!/bin/bash
#SBATCH --job-name=${EXP_ID}
#SBATCH --output=logs/${EXP_ID}_%j.out
#SBATCH --error=logs/${EXP_ID}_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Change to working directory
cd ${WORKDIR}
mkdir -p logs

# Set environment variables
export HF_HOME="${MODELS_DIR}"
export WANDB_PROJECT="${WANDB_PROJECT}"

# Login to wandb
uv run wandb login "$(cat "/home/guests/saket/continual/Self-Distillation/scripts/wandb_token.out" | tr -d '[:space:]')"
MODEL="$MODEL"
TASK="$TASK"
LR="$LR"
RESYNC=$RESYNC
ENERGY_THRESHOLD=$ENERGY_THRESHOLD
GRAD_CLIP=$GRAD_CLIP
EXP_ID="$EXP_ID"
OUTPUT_DIR="${WORKDIR}/outputs/projected_\${EXP_ID}"

echo "========================================"
echo "LR: \$LR, Resync: \$RESYNC, Energy: \$ENERGY_THRESHOLD"
echo "Output: \$OUTPUT_DIR"
echo "========================================"

uv run python train_single_task.py \\
    --method sdft \\
    --frozen_teacher \\
    --task \$TASK \\
    --model_name \$MODEL \\
    --output_dir \$OUTPUT_DIR \\
    --exp_id \$EXP_ID \\
    --use_projected_optimizer \\
    --proj_energy_threshold \$ENERGY_THRESHOLD \\
    --proj_resync_every \$RESYNC \\
    --proj_grad_clip \$GRAD_CLIP \\
    --learning_rate \$LR \\
    --num_train_epochs 1 \\
    --per_device_train_batch_size 1 \\
    --gradient_accumulation_steps 32 \\
    --max_samples 256

echo "Done: \$EXP_ID"
EOF

        echo "Created: $FILENAME"
    done
done

echo ""
echo "To submit all jobs:"
echo "  for f in scripts/sbatch/proj_lr*.sbatch; do sbatch \$f; done"
echo ""
echo "Or use the array job:"
echo "  sbatch scripts/sbatch/run_projected_sweep.sbatch"
