"""
Visualization for Stiefel manifold distance analysis.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import argparse


def load_results(path: str) -> Dict:
    """Load results from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def plot_layer_distances(
    results: Dict,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    Plot spectral distance to Stiefel manifold across layers.
    """
    param_types = [
        'self_attn.q_proj.weight',
        'self_attn.k_proj.weight',
        'self_attn.v_proj.weight',
        'self_attn.o_proj.weight',
        'mlp.gate_proj.weight',
        'mlp.up_proj.weight',
        'mlp.down_proj.weight',
    ]

    layers = sorted([int(k) for k in results['layers'].keys()])

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # Plot each parameter type
    for idx, param_name in enumerate(param_types):
        ax = axes[idx]
        distances = []
        max_svs = []
        min_svs = []

        for layer_idx in layers:
            layer_data = results['layers'][str(layer_idx)]
            if param_name in layer_data and 'error' not in layer_data[param_name]:
                metrics = layer_data[param_name]
                distances.append(metrics['spectral_distance'])
                max_svs.append(metrics['max_singular_value'])
                min_svs.append(metrics['min_singular_value'])
            else:
                distances.append(np.nan)
                max_svs.append(np.nan)
                min_svs.append(np.nan)

        ax.plot(layers, distances, 'b-', linewidth=2, label='Spectral dist')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Spectral Distance')
        ax.set_title(param_name.split('.')[-2])
        ax.grid(True, alpha=0.3)

    # Use last subplot for legend/summary
    ax = axes[-1]
    ax.axis('off')
    summary = results['summary']
    text = f"Summary:\n"
    text += f"Mean dist: {summary['mean_spectral_distance']:.4f}\n\n"
    text += "Top 3 furthest:\n"
    for i, item in enumerate(summary['top_10_furthest_from_stiefel'][:3]):
        text += f"{i+1}. L{item['layer']} {item['param'].split('.')[-2]}\n"
        text += f"   dist={item['spectral_distance']:.4f}\n"
    ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_singular_value_spectrum(
    results: Dict,
    layer_idx: int,
    output_path: Optional[str] = None,
):
    """
    Plot singular value distribution for all params in a layer.
    """
    layer_data = results['layers'][str(layer_idx)]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    param_names = list(layer_data.keys())

    for idx, param_name in enumerate(param_names[:7]):
        ax = axes[idx]
        metrics = layer_data[param_name]

        if 'error' in metrics:
            ax.text(0.5, 0.5, f"Error: {metrics['error']}", ha='center', va='center')
            continue

        # Plot sample singular values
        svs = metrics['singular_values_sample']
        ax.bar(range(len(svs)), svs, alpha=0.7)
        ax.axhline(y=1.0, color='red', linestyle='--', label='σ=1 (Stiefel)')
        ax.set_xlabel('Singular Value Index')
        ax.set_ylabel('σ')
        ax.set_title(f"{param_name.split('.')[-2]}\n(dist={metrics['spectral_distance']:.4f})")
        ax.legend(fontsize=8)

    axes[-1].axis('off')

    fig.suptitle(f"Layer {layer_idx} - Singular Values (first 10)", fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def compare_models_plot(
    results_list: List[Dict],
    model_names: List[str],
    output_path: Optional[str] = None,
):
    """
    Compare spectral distances between multiple models.
    """
    param_types = [
        'self_attn.q_proj.weight',
        'self_attn.k_proj.weight',
        'self_attn.v_proj.weight',
        'self_attn.o_proj.weight',
        'mlp.gate_proj.weight',
        'mlp.up_proj.weight',
        'mlp.down_proj.weight',
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    colors = ['blue', 'red', 'green', 'orange']

    for idx, param_name in enumerate(param_types):
        ax = axes[idx]

        for model_idx, (results, model_name) in enumerate(zip(results_list, model_names)):
            layers = sorted([int(k) for k in results['layers'].keys()])
            distances = []

            for layer_idx in layers:
                layer_data = results['layers'][str(layer_idx)]
                if param_name in layer_data and 'error' not in layer_data[param_name]:
                    distances.append(layer_data[param_name]['spectral_distance'])
                else:
                    distances.append(np.nan)

            ax.plot(layers, distances, color=colors[model_idx % len(colors)],
                    linewidth=2, label=model_name, alpha=0.8)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Spectral Distance')
        ax.set_title(param_name.split('.')[-2])
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    # Summary in last subplot
    ax = axes[-1]
    ax.axis('off')
    text = "Mean Spectral Distance:\n\n"
    for results, model_name in zip(results_list, model_names):
        mean_dist = results['summary']['mean_spectral_distance']
        text += f"{model_name}: {mean_dist:.4f}\n"
    ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace')

    fig.suptitle("Stiefel Distance Comparison: SFT vs SDFT", fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_heatmap(
    results: Dict,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    Plot heatmap of spectral distances across layers and params.
    """
    param_types = [
        'self_attn.q_proj.weight',
        'self_attn.k_proj.weight',
        'self_attn.v_proj.weight',
        'self_attn.o_proj.weight',
        'mlp.gate_proj.weight',
        'mlp.up_proj.weight',
        'mlp.down_proj.weight',
    ]
    param_labels = ['Q', 'K', 'V', 'O', 'Gate', 'Up', 'Down']

    layers = sorted([int(k) for k in results['layers'].keys()])

    # Build matrix
    matrix = np.zeros((len(param_types), len(layers)))

    for j, layer_idx in enumerate(layers):
        layer_data = results['layers'][str(layer_idx)]
        for i, param_name in enumerate(param_types):
            if param_name in layer_data and 'error' not in layer_data[param_name]:
                matrix[i, j] = layer_data[param_name]['spectral_distance']
            else:
                matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(14, 4))

    im = ax.imshow(matrix, aspect='auto', cmap='viridis')
    ax.set_xticks(range(0, len(layers), 4))
    ax.set_xticklabels([str(l) for l in layers[::4]])
    ax.set_yticks(range(len(param_labels)))
    ax.set_yticklabels(param_labels)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Parameter')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Spectral Distance to Stiefel')

    if title:
        ax.set_title(title)
    else:
        ax.set_title('Spectral Distance to Stiefel Manifold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize Stiefel distance analysis")
    parser.add_argument('--results', type=str, nargs='+', required=True,
                        help='Path(s) to results JSON file(s)')
    parser.add_argument('--names', type=str, nargs='+', default=None,
                        help='Names for each model (for comparison)')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save plots')
    parser.add_argument('--plot_type', type=str, default='all',
                        choices=['layers', 'heatmap', 'compare', 'all'],
                        help='Type of plot to generate')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_list = [load_results(p) for p in args.results]

    if args.names is None:
        args.names = [f"Model {i}" for i in range(len(results_list))]

    if args.plot_type in ['layers', 'all']:
        for results, name in zip(results_list, args.names):
            plot_layer_distances(
                results,
                output_path=output_dir / f"{name.replace(' ', '_')}_layers.png",
                title=name
            )

    if args.plot_type in ['heatmap', 'all']:
        for results, name in zip(results_list, args.names):
            plot_heatmap(
                results,
                output_path=output_dir / f"{name.replace(' ', '_')}_heatmap.png",
                title=name
            )

    if args.plot_type in ['compare', 'all'] and len(results_list) > 1:
        compare_models_plot(
            results_list,
            args.names,
            output_path=output_dir / "comparison.png"
        )


if __name__ == '__main__':
    main()
