#!/bin/bash
# Generate sbatch files for DFT with Muon optimizer baseline
# Muon typically uses higher learning rates than Adam (default 0.02)
# Dataset: databricks/databricks-dolly-15k

OUTPUT_DIR="scripts/sbatch/dft_muon_baseline"
mkdir -p "$OUTPUT_DIR"

# Learning rates to sweep (Muon uses higher LRs than Adam)
# Range: conservative (1e-3) to Muon default (2e-2)
LRS=(1e-3 5e-3 1e-2 2e-2)

MODEL="Qwen/Qwen2.5-3B-Instruct"
TASK="dolly"
MOMENTUM=0.95

for lr in "${LRS[@]}"; do
    # Create safe filename (replace . with p, - with m)
    lr_safe=$(echo "$lr" | tr '.' 'p' | tr '-' 'm')

    EXP_ID="dolly_muon_lr${lr_safe}"
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
MOMENTUM=${MOMENTUM}
EXP_ID="${EXP_ID}"
OUTPUT_DIR="/home/guests/saket/continual/Self-Distillation/outputs/\${EXP_ID}"

echo "========================================"
echo "DFT Muon Baseline"
echo "Task: \$TASK"
echo "LR: \$LR, Momentum: \$MOMENTUM"
echo "Output: \$OUTPUT_DIR"
echo "========================================"

uv run python train_single_task.py \\
    --method sdft \\
    --frozen_teacher \\
    --task \$TASK \\
    --model_name \$MODEL \\
    --output_dir \$OUTPUT_DIR \\
    --exp_id \$EXP_ID \\
    --use_muon_optimizer \\
    --muon_momentum \$MOMENTUM \\
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
