#!/usr/bin/env python3
"""
Run singular vector alignment analysis across all layers and parameters
for both SFT and SDFT training trajectories.

Measures how much U and V rotate over the course of continual learning.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eigenspectrum.singular_vector_alignment import (
    analyze_trajectory,
    load_matrices_from_checkpoints,
    plot_trajectory_heatmaps,
    plot_trajectory_metrics,
)


# =============================================================================
# Configuration
# =============================================================================

DATA_ROOT = "/data/saket/continual/Self-Distillation"

# Checkpoint sequences
CHECKPOINTS = {
    'sft': [
        "Qwen/Qwen2.5-3B-Instruct",  # Base model (reference)
        f"{DATA_ROOT}/outputs/sft_full_mbpp/task_0_tooluse/checkpoint-127",
        f"{DATA_ROOT}/outputs/sft_full_mbpp/task_1_gsm8k/checkpoint-234",
        f"{DATA_ROOT}/outputs/sft_full_mbpp/task_2_mbpp/checkpoint-12",
    ],
    'sdft': [
        "Qwen/Qwen2.5-3B-Instruct",  # Base model (reference)
        f"{DATA_ROOT}/outputs/sdft_full_mbpp/task_0_tooluse/checkpoint-1011",
        f"{DATA_ROOT}/outputs/sdft_full_mbpp/task_1_gsm8k/checkpoint-1868",
        f"{DATA_ROOT}/outputs/sdft_full_mbpp/task_2_mbpp/checkpoint-93",
    ],
}

CHECKPOINT_LABELS = ['base', 'tooluse', 'gsm8k', 'mbpp']

# Parameters to analyze
PARAM_NAMES = [
    'self_attn.q_proj.weight',
    'self_attn.k_proj.weight',
    'self_attn.v_proj.weight',
    'self_attn.o_proj.weight',
    'mlp.gate_proj.weight',
    'mlp.up_proj.weight',
    'mlp.down_proj.weight',
]

PARAM_SHORT_NAMES = {
    'self_attn.q_proj.weight': 'q_proj',
    'self_attn.k_proj.weight': 'k_proj',
    'self_attn.v_proj.weight': 'v_proj',
    'self_attn.o_proj.weight': 'o_proj',
    'mlp.gate_proj.weight': 'gate_proj',
    'mlp.up_proj.weight': 'up_proj',
    'mlp.down_proj.weight': 'down_proj',
}

NUM_LAYERS = 36
TOP_K = 50


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_single_param(
    method: str,
    layer_idx: int,
    param_name: str,
    output_dir: Path,
    top_k: int = TOP_K,
    use_gpu: bool = True,
) -> Optional[Dict]:
    """
    Analyze singular vector alignment for a single parameter.

    Returns:
        Dictionary with final metrics, or None if failed
    """
    checkpoint_paths = CHECKPOINTS[method]
    param_short = PARAM_SHORT_NAMES[param_name]

    try:
        # Load matrices
        matrices, _ = load_matrices_from_checkpoints(
            checkpoint_paths,
            layer_idx,
            param_name,
        )

        # Run analysis
        trajectory = analyze_trajectory(
            matrices,
            checkpoint_ids=CHECKPOINT_LABELS,
            reference_idx=0,
            top_k=top_k,
            use_gpu=use_gpu,
        )

        # Extract final metrics (comparing last checkpoint to base)
        final_metrics = trajectory.alignments[-1].metrics

        # Save detailed results
        result = {
            'method': method,
            'layer': layer_idx,
            'param': param_name,
            'checkpoint_ids': CHECKPOINT_LABELS,
            'metrics_over_time': trajectory.metrics_over_time,
            'final_metrics': final_metrics,
        }

        return result

    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def run_full_analysis(
    output_dir: str,
    methods: List[str] = ['sft', 'sdft'],
    layers: Optional[List[int]] = None,
    params: Optional[List[str]] = None,
    top_k: int = TOP_K,
    save_heatmaps: bool = False,
    use_gpu: bool = True,
):
    """
    Run alignment analysis across all layers and parameters.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if layers is None:
        layers = list(range(NUM_LAYERS))
    if params is None:
        params = PARAM_NAMES

    all_results = {
        'sft': {},
        'sdft': {},
    }

    # Summary metrics for comparison plots
    summary = {
        method: {
            param: {
                'U_mean_diag': [],
                'V_mean_diag': [],
                'U_diag_ratio': [],
                'V_diag_ratio': [],
                'U_frob_dist_identity': [],
                'V_frob_dist_identity': [],
            }
            for param in params
        }
        for method in methods
    }

    total = len(methods) * len(layers) * len(params)
    count = 0

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Analyzing {method.upper()}")
        print('='*60)

        method_dir = output_dir / method
        method_dir.mkdir(exist_ok=True)

        for layer_idx in layers:
            print(f"\nLayer {layer_idx}/{NUM_LAYERS-1}")

            for param_name in params:
                count += 1
                param_short = PARAM_SHORT_NAMES[param_name]
                print(f"  [{count}/{total}] {param_short}...", end=' ', flush=True)

                result = analyze_single_param(
                    method, layer_idx, param_name, method_dir, top_k, use_gpu
                )

                if result is not None:
                    # Store result
                    key = f"layer{layer_idx}_{param_short}"
                    all_results[method][key] = result

                    # Extract final metrics for summary
                    final = result['final_metrics']
                    for metric_name in summary[method][param_name]:
                        if metric_name in final:
                            summary[method][param_name][metric_name].append(final[metric_name])
                        else:
                            summary[method][param_name][metric_name].append(np.nan)

                    print(f"U_diag={final['U_mean_diag']:.3f}, V_diag={final['V_mean_diag']:.3f}")
                else:
                    # Fill with NaN
                    for metric_name in summary[method][param_name]:
                        summary[method][param_name][metric_name].append(np.nan)
                    print("FAILED")

    # Save all results
    results_path = output_dir / 'all_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved detailed results to {results_path}")

    # Save summary
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    # Generate comparison plots
    print("\nGenerating comparison plots...")
    plot_comparison(summary, layers, params, output_dir)

    return all_results, summary


def plot_comparison(
    summary: Dict,
    layers: List[int],
    params: List[str],
    output_dir: Path,
):
    """
    Generate comparison plots between SFT and SDFT.
    """
    # Plot 1: U mean diagonal across layers for each param
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, param_name in enumerate(params):
        ax = axes[idx]
        param_short = PARAM_SHORT_NAMES[param_name]

        for method, color in [('sft', 'blue'), ('sdft', 'red')]:
            values = summary[method][param_name]['U_mean_diag']
            ax.plot(layers, values, 'o-', color=color, label=method.upper(),
                    linewidth=1.5, markersize=3, alpha=0.8)

        ax.set_xlabel('Layer')
        ax.set_ylabel('U mean |diagonal|')
        ax.set_title(param_short)
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()

    axes[-1].axis('off')
    fig.suptitle('Left Singular Vector Alignment (U): Mean Diagonal Entry', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_U_mean_diag.png', dpi=150)
    plt.close()

    # Plot 2: V mean diagonal across layers for each param
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, param_name in enumerate(params):
        ax = axes[idx]
        param_short = PARAM_SHORT_NAMES[param_name]

        for method, color in [('sft', 'blue'), ('sdft', 'red')]:
            values = summary[method][param_name]['V_mean_diag']
            ax.plot(layers, values, 'o-', color=color, label=method.upper(),
                    linewidth=1.5, markersize=3, alpha=0.8)

        ax.set_xlabel('Layer')
        ax.set_ylabel('V mean |diagonal|')
        ax.set_title(param_short)
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()

    axes[-1].axis('off')
    fig.suptitle('Right Singular Vector Alignment (V): Mean Diagonal Entry', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_V_mean_diag.png', dpi=150)
    plt.close()

    # Plot 3: Frobenius distance from identity
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, param_name in enumerate(params):
        ax = axes[idx]
        param_short = PARAM_SHORT_NAMES[param_name]

        for method, color in [('sft', 'blue'), ('sdft', 'red')]:
            values_U = summary[method][param_name]['U_frob_dist_identity']
            values_V = summary[method][param_name]['V_frob_dist_identity']
            ax.plot(layers, values_U, 'o-', color=color, label=f'{method.upper()} U',
                    linewidth=1.5, markersize=3, alpha=0.8)
            ax.plot(layers, values_V, 's--', color=color, label=f'{method.upper()} V',
                    linewidth=1.5, markersize=3, alpha=0.5)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Frobenius dist from I')
        ax.set_title(param_short)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    axes[-1].axis('off')
    fig.suptitle('Alignment Matrix Distance from Identity', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_frob_dist.png', dpi=150)
    plt.close()

    # Plot 4: Heatmap summary - mean across params for each layer
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (method, title) in enumerate([('sft', 'SFT'), ('sdft', 'SDFT')]):
        ax = axes[ax_idx]

        # Build matrix: params x layers
        matrix = np.zeros((len(params), len(layers)))
        for i, param_name in enumerate(params):
            values = summary[method][param_name]['U_mean_diag']
            matrix[i, :] = values

        im = ax.imshow(matrix, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        ax.set_xticks(range(0, len(layers), 4))
        ax.set_xticklabels([str(l) for l in layers[::4]])
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels([PARAM_SHORT_NAMES[p] for p in params])
        ax.set_xlabel('Layer')
        ax.set_ylabel('Parameter')
        ax.set_title(f'{title}: U Mean Diagonal (1=preserved)')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_heatmap.png', dpi=150)
    plt.close()

    # Plot 5: Average drift summary
    fig, ax = plt.subplots(figsize=(10, 6))

    # Average across all params
    for method, color in [('sft', 'blue'), ('sdft', 'red')]:
        all_U_diag = []
        all_V_diag = []
        for param_name in params:
            all_U_diag.append(summary[method][param_name]['U_mean_diag'])
            all_V_diag.append(summary[method][param_name]['V_mean_diag'])

        avg_U = np.nanmean(all_U_diag, axis=0)
        avg_V = np.nanmean(all_V_diag, axis=0)

        ax.plot(layers, avg_U, 'o-', color=color, label=f'{method.upper()} U',
                linewidth=2, markersize=4)
        ax.plot(layers, avg_V, 's--', color=color, label=f'{method.upper()} V',
                linewidth=2, markersize=4, alpha=0.7)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Diagonal Alignment')
    ax.set_title('Average Singular Vector Preservation Across All Parameters')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect alignment')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_average.png', dpi=150)
    plt.close()

    print(f"Saved comparison plots to {output_dir}")


# =============================================================================
# Main
# =============================================================================

def get_checkpoints(data_root: str) -> Dict[str, List[str]]:
    """Build checkpoint paths for given data root."""
    return {
        'sft': [
            "Qwen/Qwen2.5-3B-Instruct",
            f"{data_root}/outputs/sft_full_mbpp/task_0_tooluse/checkpoint-127",
            f"{data_root}/outputs/sft_full_mbpp/task_1_gsm8k/checkpoint-234",
            f"{data_root}/outputs/sft_full_mbpp/task_2_mbpp/checkpoint-12",
        ],
        'sdft': [
            "Qwen/Qwen2.5-3B-Instruct",
            f"{data_root}/outputs/sdft_full_mbpp/task_0_tooluse/checkpoint-1011",
            f"{data_root}/outputs/sdft_full_mbpp/task_1_gsm8k/checkpoint-1868",
            f"{data_root}/outputs/sdft_full_mbpp/task_2_mbpp/checkpoint-93",
        ],
    }


def main():
    global CHECKPOINTS

    parser = argparse.ArgumentParser(
        description='Analyze singular vector alignment for SFT vs SDFT'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='eigenspectrum/outputs/sv_alignment',
        help='Output directory'
    )
    parser.add_argument(
        '--data_root', type=str,
        default=DATA_ROOT,
        help='Root directory for checkpoint data'
    )
    parser.add_argument(
        '--layers', type=int, nargs='+', default=None,
        help='Specific layers to analyze (default: all 36)'
    )
    parser.add_argument(
        '--methods', type=str, nargs='+', default=['sft', 'sdft'],
        choices=['sft', 'sdft'],
        help='Which methods to analyze'
    )
    parser.add_argument(
        '--top_k', type=int, default=TOP_K,
        help='Number of top singular directions'
    )
    parser.add_argument(
        '--use_gpu', action='store_true', default=True,
        help='Use GPU for SVD computation (default: True)'
    )
    parser.add_argument(
        '--cpu', action='store_true',
        help='Force CPU computation'
    )

    args = parser.parse_args()

    # Update checkpoints based on data root
    CHECKPOINTS = get_checkpoints(args.data_root)

    use_gpu = args.use_gpu and not args.cpu

    run_full_analysis(
        output_dir=args.output_dir,
        methods=args.methods,
        layers=args.layers,
        top_k=args.top_k,
        use_gpu=use_gpu,
    )


if __name__ == '__main__':
    main()
