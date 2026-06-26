#!/bin/bash
# Generate sbatch files for tooluse LR sweep with projected gradient optimizer

OUTPUT_DIR="scripts/sbatch/tooluse_lr_sweep"
mkdir -p "$OUTPUT_DIR"

# Learning rates to sweep
LRS=(1e-4 5e-5 1e-5 1e-6)

MODEL="Qwen/Qwen2.5-3B-Instruct"
TASK="tooluse"
ENERGY_THRESHOLD=0.9
GRAD_CLIP=1

for lr in "${LRS[@]}"; do
    # Create safe filename (replace . with p, - with m)
    lr_safe=$(echo "$lr" | tr '.' 'p' | tr '-' 'm')

    EXP_ID="tooluse_proj_lr${lr_safe}"
    FILENAME="${OUTPUT_DIR}/proj_${EXP_ID}.sbatch"

    cat > "$FILENAME" << EOF
#!/bin/bash
#SBATCH --job-name=${EXP_ID}
#SBATCH --output=logs/${EXP_ID}_%j.out
#SBATCH --error=logs/${EXP_ID}_%j.err
#SBATCH --time=6:00:00
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
ENERGY_THRESHOLD=${ENERGY_THRESHOLD}
GRAD_CLIP=${GRAD_CLIP}
EXP_ID="${EXP_ID}"
OUTPUT_DIR="/home/guests/saket/continual/Self-Distillation/outputs/projected_\${EXP_ID}"

echo "========================================"
echo "Projected Gradient LR Sweep"
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

echo ""
echo "Generated $(ls -1 ${OUTPUT_DIR}/*.sbatch | wc -l) sbatch files in ${OUTPUT_DIR}/"
echo ""
echo "To submit all jobs:"
echo "  for f in ${OUTPUT_DIR}/*.sbatch; do sbatch \$f; done"
