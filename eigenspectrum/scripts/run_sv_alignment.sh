#!/bin/bash
# Run singular vector alignment analysis for SFT vs SDFT

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Configuration
DATA_ROOT="${DATA_ROOT:-/data/saket/continual/Self-Distillation}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/../outputs/sv_alignment}"

cd "$PROJECT_ROOT"
echo "Working directory: $(pwd)"
echo "Data root: ${DATA_ROOT}"
echo "Output directory: ${OUTPUT_DIR}"

# Check if checkpoints exist
echo ""
echo "Checking checkpoints..."
for ckpt in \
    "${DATA_ROOT}/outputs/sft_full_mbpp/task_0_tooluse/checkpoint-127" \
    "${DATA_ROOT}/outputs/sft_full_mbpp/task_1_gsm8k/checkpoint-234" \
    "${DATA_ROOT}/outputs/sft_full_mbpp/task_2_mbpp/checkpoint-12" \
    "${DATA_ROOT}/outputs/sdft_full_mbpp/task_0_tooluse/checkpoint-1011" \
    "${DATA_ROOT}/outputs/sdft_full_mbpp/task_1_gsm8k/checkpoint-1868" \
    "${DATA_ROOT}/outputs/sdft_full_mbpp/task_2_mbpp/checkpoint-93"
do
    if [ -d "$ckpt" ]; then
        echo "  ✓ $(basename $(dirname $ckpt))/$(basename $ckpt)"
    else
        echo "  ✗ MISSING: $ckpt"
    fi
done

echo ""
echo "Starting analysis..."
python eigenspectrum/scripts/analyze_sv_alignment_all.py \
    --output_dir "$OUTPUT_DIR" \
    --data_root "$DATA_ROOT" \
    --methods sft sdft \
    --top_k 50

echo ""
echo "Done! Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Key outputs:"
echo "  - all_results.json: Detailed metrics for each layer/param"
echo "  - summary.json: Summary metrics"
echo "  - comparison_U_mean_diag.png: U alignment across layers"
echo "  - comparison_V_mean_diag.png: V alignment across layers"
echo "  - comparison_heatmap.png: Heatmap summary"
echo "  - comparison_average.png: Average drift across all params"
