#!/bin/bash
# Analyze eigenspectrum for all layers of the base Qwen model
# Saves results in eigenspectrum/outputs/qwen2.5-3b-instruct/layer_*/

set -e

MODEL_PATH="${MODEL_PATH:-/data/saket/continual/Self-Distillation/models/qwen2.5-3b-instruct}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/saket/continual/Self-Distillation/eigenspectrum/outputs/qwen2.5-3b-instruct}"
NUM_LAYERS="${NUM_LAYERS:-36}"  # Qwen2.5-3B has 36 layers
NUM_DATA_SAMPLES="${NUM_DATA_SAMPLES:-500}"
LANCZOS_ORDER="${LANCZOS_ORDER:-50}"
NUM_LANCZOS_SAMPLES="${NUM_LANCZOS_SAMPLES:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DEVICE="${DEVICE:-cuda}"

# GPU settings
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

START_TIME=$(date +%s)

echo "=============================================="
echo "Eigenspectrum Analysis - All Layers"
echo "=============================================="
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_BASE}"
echo "Layers: 0 to $((NUM_LAYERS-1))"
echo "Data samples: ${NUM_DATA_SAMPLES}"
echo "Lanczos order: ${LANCZOS_ORDER}"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES}"
echo "=============================================="

# Create output base directory
mkdir -p "${OUTPUT_BASE}"

# Loop through all layers (skip layer 0, already done)
for layer in $(seq 0 $((NUM_LAYERS-1))); do
    OUTPUT_DIR="${OUTPUT_BASE}/layer_${layer}"

    echo ""
    echo "=============================================="
    echo "[Layer ${layer}/${NUM_LAYERS}] Analyzing..."
    echo "Output: ${OUTPUT_DIR}"
    echo "=============================================="

    # Skip if already done
    if [ -f "${OUTPUT_DIR}/summary.json" ]; then
        echo "  Already completed, skipping. Delete ${OUTPUT_DIR}/summary.json to rerun."
        continue
    fi

    python -m eigenspectrum.example_analyze \
        --model_path "${MODEL_PATH}" \
        --layer ${layer} \
        --output_dir "${OUTPUT_DIR}" \
        --num_data_samples ${NUM_DATA_SAMPLES} \
        --lanczos_order ${LANCZOS_ORDER} \
        --num_samples ${NUM_LANCZOS_SAMPLES} \
        --batch_size ${BATCH_SIZE} \
        --device ${DEVICE} \
        --use_bfloat16 \
        --use_gradient_checkpointing

    # Generate density plots
    echo "  Generating density plots..."
    python -m eigenspectrum.plot_spectrum \
        --input_dir "${OUTPUT_DIR}" \
        --num_bins 50

    echo "  Layer ${layer} complete!"
done

echo ""
echo "=============================================="
echo "All layers complete!"
echo "Results saved in: ${OUTPUT_BASE}/layer_*/"
echo "=============================================="

# Generate comparison across layers (optional)
echo ""
echo "Generating cross-layer comparison plots..."
python -c "
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

output_base = '${OUTPUT_BASE}'
layers = sorted(glob.glob(f'{output_base}/layer_*/summary.json'))

if len(layers) > 0:
    import json

    # Collect data
    data = {}
    for layer_path in layers:
        layer_dir = Path(layer_path).parent
        layer_idx = int(layer_dir.name.split('_')[1])

        with open(layer_path) as f:
            summary = json.load(f)

        for param_name, stats in summary.items():
            short_name = param_name.split('.')[-2] + '.' + param_name.split('.')[-1]
            if short_name not in data:
                data[short_name] = {'layers': [], 'top_eig': [], 'eff_rank': []}
            data[short_name]['layers'].append(layer_idx)
            data[short_name]['top_eig'].append(stats['top_eigenvalue'])
            data[short_name]['eff_rank'].append(stats['effective_rank'])

    # Plot top eigenvalue across layers
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for param_type, vals in data.items():
        sorted_idx = np.argsort(vals['layers'])
        layers_sorted = np.array(vals['layers'])[sorted_idx]
        top_eig_sorted = np.array(vals['top_eig'])[sorted_idx]
        eff_rank_sorted = np.array(vals['eff_rank'])[sorted_idx]

        axes[0].plot(layers_sorted, top_eig_sorted, 'o-', label=param_type, markersize=4)
        axes[1].plot(layers_sorted, eff_rank_sorted, 'o-', label=param_type, markersize=4)

    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Top Eigenvalue')
    axes[0].set_title('Top Eigenvalue Across Layers')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Effective Rank')
    axes[1].set_title('Effective Rank Across Layers')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_base}/cross_layer_summary.png', dpi=150)
    print(f'Saved: {output_base}/cross_layer_summary.png')
"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=============================================="
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=============================================="
echo "Done!"
