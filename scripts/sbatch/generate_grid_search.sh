#!/bin/bash
# Generate sbatch files for grid search over grad_clip, energy_threshold, lr
# Uses --proj_no_resync (no resyncing of UBV factorization)
# Dataset: databricks/databricks-dolly-15k

OUTPUT_DIR="scripts/sbatch/grid_search"
mkdir -p "$OUTPUT_DIR"

# Grid values
GRAD_CLIPS=(0.5 1)
ENERGY_THRESHOLDS=(0.9 0.95)
LRS=(1e-3 2.5e-3 5e-3)

MODEL="Qwen/Qwen2.5-3B-Instruct"
TASK="dolly"

for gc in "${GRAD_CLIPS[@]}"; do
    for et in "${ENERGY_THRESHOLDS[@]}"; do
        for lr in "${LRS[@]}"; do
            # Create safe filename (replace . with p)
            gc_safe=$(echo "$gc" | tr '.' 'p')
            et_safe=$(echo "$et" | tr '.' 'p')
            lr_safe=$(echo "$lr" | tr '.' 'p' | tr '-' 'm')

            EXP_ID="dolly_gc${gc_safe}_et${et_safe}_lr${lr_safe}"
            FILENAME="${OUTPUT_DIR}/proj_${EXP_ID}.sbatch"

            cat > "$FILENAME" << EOF
#!/bin/bash
#SBATCH --job-name=${EXP_ID}
#SBATCH --output=logs/${EXP_ID}_%j.out
#SBATCH --error=logs/${EXP_ID}_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Change to working directory
cd /home/guests/saket/continual/Self-Distillation
mkdir -p logs

# Set environment variables
export HF_HOME="/home/guests/saket/models"
export WANDB_PROJECT="continual-learning-sdft"

# Login to wandb
uv run wandb login "$(cat "/home/guests/saket/continual/Self-Distillation/scripts/wandb_token.out" | tr -d '[:space:]')"
MODEL="${MODEL}"
TASK="${TASK}"
LR="${lr}"
ENERGY_THRESHOLD=${et}
GRAD_CLIP=${gc}
EXP_ID="${EXP_ID}"
OUTPUT_DIR="/home/guests/saket/continual/Self-Distillation/outputs/projected_\${EXP_ID}"

echo "========================================"
echo "Grid Search Experiment (no resync)"
echo "Task: \$TASK"
echo "LR: \$LR"
echo "Energy: \$ENERGY_THRESHOLD, Grad Clip: \$GRAD_CLIP"
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
    --proj_no_resync \\
    --proj_grad_clip \$GRAD_CLIP \\
    --learning_rate \$LR \\
    --num_train_epochs 1 \\
    --per_device_train_batch_size 1 \\
    --gradient_accumulation_steps 64

echo "Done: \$EXP_ID"
EOF

            echo "Created: $FILENAME"
        done
    done
done

echo ""
echo "Generated $(ls -1 ${OUTPUT_DIR}/*.sbatch | wc -l) sbatch files in ${OUTPUT_DIR}/"
echo ""
echo "To submit all jobs:"
echo "  for f in ${OUTPUT_DIR}/*.sbatch; do sbatch \$f; done"
