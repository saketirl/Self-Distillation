#!/bin/bash
# Generate sbatch files for DFT with Adam baseline (no projected optimizer)
# This serves as a benchmark against the projected gradient optimizer
# Dataset: databricks/databricks-dolly-15k

OUTPUT_DIR="scripts/sbatch/dft_adam_baseline"
mkdir -p "$OUTPUT_DIR"

# Learning rates to sweep
LRS=(5e-6 1e-5 5e-5)

MODEL="Qwen/Qwen2.5-3B-Instruct"
TASK="dolly"

for lr in "${LRS[@]}"; do
    # Create safe filename (replace . with p, - with m)
    lr_safe=$(echo "$lr" | tr '.' 'p' | tr '-' 'm')

    EXP_ID="dolly_adam_lr${lr_safe}"
    FILENAME="${OUTPUT_DIR}/${EXP_ID}.sbatch"

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
EXP_ID="${EXP_ID}"
OUTPUT_DIR="/home/guests/saket/continual/Self-Distillation/outputs/\${EXP_ID}"

echo "========================================"
echo "DFT Adam Baseline"
echo "Task: \$TASK"
echo "LR: \$LR"
echo "Output: \$OUTPUT_DIR"
echo "========================================"

uv run python train_single_task.py \\
    --method sdft \\
    --frozen_teacher \\
    --task \$TASK \\
    --model_name \$MODEL \\
    --output_dir \$OUTPUT_DIR \\
    --exp_id \$EXP_ID \\
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
