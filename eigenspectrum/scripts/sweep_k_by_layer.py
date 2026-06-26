#!/usr/bin/env python3
"""
Analyze subspace drift by layer for different k values.

For each layer and each k, compute drift and plot:
- Drift vs layer (for each k)
- SFT vs SDFT comparison across layers
- Heatmaps of drift (layers x k)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Configuration
# =============================================================================

DATA_ROOT = "/data/saket/continual/Self-Distillation"

K_VALUES = [5, 10, 20, 50, 100, 250, 500]

def get_checkpoint_paths(data_root: str) -> Dict[str, List[str]]:
    base = "Qwen/Qwen2.5-3B-Instruct"
    return {
        'sft': [
            base,
            f"{data_root}/outputs/sft_full_mbpp/task_0_tooluse/checkpoint-127",
            f"{data_root}/outputs/sft_full_mbpp/task_1_gsm8k/checkpoint-234",
            f"{data_root}/outputs/sft_full_mbpp/task_2_mbpp/checkpoint-12",
        ],
        'sdft': [
            base,
            f"{data_root}/outputs/sdft_full_mbpp/task_0_tooluse/checkpoint-1011",
            f"{data_root}/outputs/sdft_full_mbpp/task_1_gsm8k/checkpoint-1868",
            f"{data_root}/outputs/sdft_full_mbpp/task_2_mbpp/checkpoint-93",
        ],
    }

PARAM_NAMES = [
    'self_attn.q_proj.weight',
    'self_attn.k_proj.weight',
    'self_attn.v_proj.weight',
    'self_attn.o_proj.weight',
    'mlp.gate_proj.weight',
    'mlp.up_proj.weight',
    'mlp.down_proj.weight',
]

PARAM_SHORT = {p: p.split('.')[-2] for p in PARAM_NAMES}

NUM_LAYERS = 36


# =============================================================================
# Core Functions
# =============================================================================

def load_weight(path: str, layer: int, param: str) -> np.ndarray:
    """Load weight matrix from checkpoint."""
    import torch
    from safetensors import safe_open

    full_name = f"model.layers.{layer}.{param}"
    p = Path(path)

    if p.exists() and p.is_dir():
        for sf in p.glob("*.safetensors"):
            with safe_open(sf, framework="pt", device="cpu") as f:
                if full_name in f.keys():
                    return f.get_tensor(full_name).float().numpy()
        for pt in list(p.glob("*.bin")) + list(p.glob("pytorch_model*.bin")):
            sd = torch.load(pt, map_location='cpu', weights_only=True)
            if full_name in sd:
                w = sd[full_name].float().numpy()
                del sd
                return w
        raise FileNotFoundError(f"{full_name} not in {path}")
    else:
        from huggingface_hub import hf_hub_download, list_repo_files
        for sf in [f for f in list_repo_files(path) if f.endswith('.safetensors')]:
            local = hf_hub_download(path, sf)
            with safe_open(local, framework="pt", device="cpu") as f:
                if full_name in f.keys():
                    return f.get_tensor(full_name).float().numpy()
        raise FileNotFoundError(f"{full_name} not in {path}")


def svd_full(W: np.ndarray, gpu: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute full SVD. Returns U, S, V."""
    if gpu:
        import torch
        if torch.cuda.is_available():
            t = torch.tensor(W, dtype=torch.float32, device='cuda')
            U, S, Vh = torch.linalg.svd(t, full_matrices=False)
            return U.cpu().numpy(), S.cpu().numpy(), Vh.T.cpu().numpy()
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    return U, S, Vh.T


def alignment_topk(U1: np.ndarray, U2: np.ndarray, k: int) -> float:
    """Compute alignment of top-k subspaces."""
    k = min(k, U1.shape[1], U2.shape[1])
    A = U1[:, :k].T @ U2[:, :k]
    return float(np.mean(np.abs(np.diag(A))))


def drift_bp(alignment: float) -> float:
    """Convert alignment to drift in basis points."""
    return (1 - alignment) * 10000


# =============================================================================
# Analysis
# =============================================================================

def analyze_all_layers(
    data_root: str,
    output_dir: str,
    layers: List[int] = None,
    params: List[str] = None,
    k_values: List[int] = None,
    gpu: bool = True,
):
    """Analyze drift by layer for all k values, for all intermediate checkpoints."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = get_checkpoint_paths(data_root)

    if layers is None:
        layers = list(range(NUM_LAYERS))
    if params is None:
        params = PARAM_NAMES
    if k_values is None:
        k_values = K_VALUES

    n_layers = len(layers)
    n_k = len(k_values)
    n_params = len(params)

    # Checkpoint comparisons: base -> each intermediate checkpoint
    # Index 0 = base, 1 = tooluse, 2 = gsm8k, 3 = mbpp
    comparisons = [
        ('base_to_tooluse', 0, 1),
        ('base_to_gsm8k', 0, 2),
        ('base_to_mbpp', 0, 3),
    ]

    # Storage: results[comparison][method][param]['U'/'V'] -> (n_layers, n_k)
    results = {
        comp_name: {
            method: {
                param: {
                    'U': np.zeros((n_layers, n_k)),
                    'V': np.zeros((n_layers, n_k)),
                }
                for param in params
            }
            for method in ['sft', 'sdft']
        }
        for comp_name, _, _ in comparisons
    }

    total = n_layers * n_params
    count = 0

    for pi, param in enumerate(params):
        pshort = PARAM_SHORT[param]

        for li, layer in enumerate(layers):
            count += 1
            print(f"[{count}/{total}] Layer {layer:2d} {pshort}...", end=' ', flush=True)

            try:
                # Load all checkpoints for this layer/param
                sft_weights = [load_weight(paths['sft'][i], layer, param) for i in range(4)]
                sdft_weights = [load_weight(paths['sdft'][i], layer, param) for i in range(4)]

                # Compute SVDs
                sft_svds = [svd_full(W, gpu) for W in sft_weights]
                sdft_svds = [svd_full(W, gpu) for W in sdft_weights]

                # Compute alignments for each comparison
                for comp_name, idx_from, idx_to in comparisons:
                    for ki, k in enumerate(k_values):
                        # SFT
                        results[comp_name]['sft'][param]['U'][li, ki] = alignment_topk(
                            sft_svds[idx_from][0], sft_svds[idx_to][0], k)
                        results[comp_name]['sft'][param]['V'][li, ki] = alignment_topk(
                            sft_svds[idx_from][2], sft_svds[idx_to][2], k)
                        # SDFT
                        results[comp_name]['sdft'][param]['U'][li, ki] = alignment_topk(
                            sdft_svds[idx_from][0], sdft_svds[idx_to][0], k)
                        results[comp_name]['sdft'][param]['V'][li, ki] = alignment_topk(
                            sdft_svds[idx_from][2], sdft_svds[idx_to][2], k)

                # Print k=50 summary for base->mbpp
                ki_50 = k_values.index(50) if 50 in k_values else -1
                if ki_50 >= 0:
                    sft_d = drift_bp(results['base_to_mbpp']['sft'][param]['U'][li, ki_50])
                    sdft_d = drift_bp(results['base_to_mbpp']['sdft'][param]['U'][li, ki_50])
                    print(f"k=50 base→mbpp: SFT={sft_d:5.1f}bp, SDFT={sdft_d:5.1f}bp")
                else:
                    print("done")

            except Exception as e:
                print(f"ERROR: {e}")
                for comp_name, _, _ in comparisons:
                    for ki in range(n_k):
                        for method in ['sft', 'sdft']:
                            results[comp_name][method][param]['U'][li, ki] = np.nan
                            results[comp_name][method][param]['V'][li, ki] = np.nan

    # Save results
    save_results_all(results, layers, k_values, params, comparisons, output_dir)

    # Generate plots for each comparison
    print("\nGenerating plots...")
    for comp_name, _, _ in comparisons:
        print(f"  Plotting {comp_name}...")
        comp_results = results[comp_name]
        comp_dir = output_dir / comp_name
        comp_dir.mkdir(exist_ok=True)

        plot_drift_by_layer(comp_results, layers, k_values, params, comp_dir, comp_name)
        plot_heatmaps(comp_results, layers, k_values, params, comp_dir, comp_name)
        plot_difference_by_layer(comp_results, layers, k_values, params, comp_dir, comp_name)

    # Combined comparison plot
    plot_all_comparisons(results, layers, k_values, params, comparisons, output_dir)
    print_layer_summary_all(results, layers, k_values, params, comparisons)

    return results


def save_results(results: Dict, layers: List[int], k_values: List[int],
                 params: List[str], output_dir: Path):
    """Save results to JSON (legacy single comparison)."""
    json_results = {
        'layers': layers,
        'k_values': k_values,
        'params': params,
        'data': {}
    }

    for method in results:
        json_results['data'][method] = {}
        for param in results[method]:
            json_results['data'][method][param] = {
                'U': results[method][param]['U'].tolist(),
                'V': results[method][param]['V'].tolist(),
            }

    with open(output_dir / 'layer_k_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved results to {output_dir / 'layer_k_results.json'}")


def save_results_all(results: Dict, layers: List[int], k_values: List[int],
                     params: List[str], comparisons: List, output_dir: Path):
    """Save results for all comparisons to JSON, including per-matrix delta."""
    json_results = {
        'layers': layers,
        'k_values': k_values,
        'params': params,
        'param_short_names': {p: PARAM_SHORT[p] for p in params},
        'comparisons': [c[0] for c in comparisons],
        'data': {}
    }

    for comp_name, _, _ in comparisons:
        json_results['data'][comp_name] = {
            'sft': {},
            'sdft': {},
            'delta': {},  # SDFT - SFT for each matrix
        }

        for param in params:
            # Store SFT alignment (per matrix)
            json_results['data'][comp_name]['sft'][param] = {
                'U': results[comp_name]['sft'][param]['U'].tolist(),
                'V': results[comp_name]['sft'][param]['V'].tolist(),
            }

            # Store SDFT alignment (per matrix)
            json_results['data'][comp_name]['sdft'][param] = {
                'U': results[comp_name]['sdft'][param]['U'].tolist(),
                'V': results[comp_name]['sdft'][param]['V'].tolist(),
            }

            # Compute and store delta (SDFT - SFT) per matrix
            # Delta is in drift basis points: (1 - sdft) - (1 - sft) = sft - sdft
            # But for alignment: sdft_align - sft_align (positive = SDFT aligns better)
            # For drift: (1-sdft)*10000 - (1-sft)*10000 = (sft - sdft)*10000
            delta_U = results[comp_name]['sdft'][param]['U'] - results[comp_name]['sft'][param]['U']
            delta_V = results[comp_name]['sdft'][param]['V'] - results[comp_name]['sft'][param]['V']

            # Also compute drift delta (positive = SDFT drifts MORE)
            drift_delta_U = (1 - results[comp_name]['sdft'][param]['U']) - (1 - results[comp_name]['sft'][param]['U'])
            drift_delta_V = (1 - results[comp_name]['sdft'][param]['V']) - (1 - results[comp_name]['sft'][param]['V'])

            json_results['data'][comp_name]['delta'][param] = {
                'U_alignment': delta_U.tolist(),  # positive = SDFT aligns better
                'V_alignment': delta_V.tolist(),
                'U_drift_bp': (drift_delta_U * 10000).tolist(),  # positive = SDFT drifts more
                'V_drift_bp': (drift_delta_V * 10000).tolist(),
            }

    # Save main results file
    with open(output_dir / 'all_comparisons_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved all results to {output_dir / 'all_comparisons_results.json'}")

    # Also save per-matrix CSV files for easier analysis
    save_per_matrix_csv(results, layers, k_values, params, comparisons, output_dir)


def save_per_matrix_csv(results: Dict, layers: List[int], k_values: List[int],
                        params: List[str], comparisons: List, output_dir: Path):
    """Save per-matrix results as CSV files for easy loading."""
    import csv

    csv_dir = output_dir / 'csv'
    csv_dir.mkdir(exist_ok=True)

    for comp_name, _, _ in comparisons:
        for param in params:
            pshort = PARAM_SHORT[param]

            # Create CSV with columns: layer, k, sft_U, sft_V, sdft_U, sdft_V, delta_U, delta_V
            csv_path = csv_dir / f'{comp_name}_{pshort}.csv'

            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['layer', 'k', 'sft_U', 'sft_V', 'sdft_U', 'sdft_V',
                                'delta_U_align', 'delta_V_align', 'delta_U_drift_bp', 'delta_V_drift_bp'])

                for li, layer in enumerate(layers):
                    for ki, k in enumerate(k_values):
                        sft_U = results[comp_name]['sft'][param]['U'][li, ki]
                        sft_V = results[comp_name]['sft'][param]['V'][li, ki]
                        sdft_U = results[comp_name]['sdft'][param]['U'][li, ki]
                        sdft_V = results[comp_name]['sdft'][param]['V'][li, ki]
                        delta_U = sdft_U - sft_U
                        delta_V = sdft_V - sft_V
                        drift_delta_U = (1 - sdft_U - (1 - sft_U)) * 10000
                        drift_delta_V = (1 - sdft_V - (1 - sft_V)) * 10000

                        writer.writerow([layer, k, sft_U, sft_V, sdft_U, sdft_V,
                                        delta_U, delta_V, drift_delta_U, drift_delta_V])

    print(f"Saved per-matrix CSV files to {csv_dir}")


def plot_drift_by_layer(results: Dict, layers: List[int], k_values: List[int],
                        params: List[str], output_dir: Path, comp_name: str = ""):
    """Plot drift vs layer for each k value."""
    title_suffix = f" ({comp_name})" if comp_name else ""

    # Average across all params
    n_layers = len(layers)
    n_k = len(k_values)

    # Aggregate across params
    sft_U_avg = np.zeros((n_layers, n_k))
    sft_V_avg = np.zeros((n_layers, n_k))
    sdft_U_avg = np.zeros((n_layers, n_k))
    sdft_V_avg = np.zeros((n_layers, n_k))

    for param in params:
        sft_U_avg += results['sft'][param]['U']
        sft_V_avg += results['sft'][param]['V']
        sdft_U_avg += results['sdft'][param]['U']
        sdft_V_avg += results['sdft'][param]['V']

    sft_U_avg /= len(params)
    sft_V_avg /= len(params)
    sdft_U_avg /= len(params)
    sdft_V_avg /= len(params)

    # Plot 1: Drift vs layer for each k (U vectors)
    n_plots = len(k_values)
    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(k_values)))

    for ki, k in enumerate(k_values):
        ax = axes[ki] if ki < len(axes) else None
        if ax is None:
            break

        sft_drift = drift_bp(sft_U_avg[:, ki])
        sdft_drift = drift_bp(sdft_U_avg[:, ki])

        ax.plot(layers, sft_drift, 'b-o', label='SFT', linewidth=2, markersize=4)
        ax.plot(layers, sdft_drift, 'r-s', label='SDFT', linewidth=2, markersize=4)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Drift (basis points)')
        ax.set_title(f'k = {k}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for i in range(len(k_values), len(axes)):
        axes[i].axis('off')

    fig.suptitle(f'U Subspace Drift vs Layer{title_suffix}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'drift_by_layer_U.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: All k values on same plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, method, title in [(axes[0], 'sft', 'SFT'), (axes[1], 'sdft', 'SDFT')]:
        data = sft_U_avg if method == 'sft' else sdft_U_avg

        for ki, k in enumerate(k_values):
            drift = drift_bp(data[:, ki])
            ax.plot(layers, drift, '-o', color=colors[ki], label=f'k={k}',
                    linewidth=2, markersize=4)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Drift (basis points)')
        ax.set_title(f'{title}: U Drift by Layer')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Drift by Layer for Different k{title_suffix}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'drift_by_layer_all_k.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_heatmaps(results: Dict, layers: List[int], k_values: List[int],
                  params: List[str], output_dir: Path, comp_name: str = ""):
    """Plot heatmaps of drift (layers x k)."""
    title_suffix = f" ({comp_name})" if comp_name else ""

    # Average across params
    n_layers = len(layers)
    n_k = len(k_values)

    sft_U_avg = np.zeros((n_layers, n_k))
    sdft_U_avg = np.zeros((n_layers, n_k))

    for param in params:
        sft_U_avg += results['sft'][param]['U']
        sdft_U_avg += results['sdft'][param]['U']

    sft_U_avg /= len(params)
    sdft_U_avg /= len(params)

    # Convert to drift
    sft_drift = drift_bp(sft_U_avg)
    sdft_drift = drift_bp(sdft_U_avg)
    diff_drift = sdft_drift - sft_drift

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Common colorbar range
    vmax = max(np.nanmax(sft_drift), np.nanmax(sdft_drift))

    for ax, data, title, cmap in [
        (axes[0], sft_drift, 'SFT: Drift (bp)', 'Blues'),
        (axes[1], sdft_drift, 'SDFT: Drift (bp)', 'Reds'),
        (axes[2], diff_drift, 'SDFT - SFT (bp)', 'RdBu_r'),
    ]:
        if 'SDFT - SFT' in title:
            vmin_plot = -np.nanmax(np.abs(diff_drift))
            vmax_plot = np.nanmax(np.abs(diff_drift))
        else:
            vmin_plot = 0
            vmax_plot = vmax

        im = ax.imshow(data.T, aspect='auto', cmap=cmap, vmin=vmin_plot, vmax=vmax_plot,
                       origin='lower')

        ax.set_xlabel('Layer')
        ax.set_ylabel('k')
        ax.set_yticks(range(len(k_values)))
        ax.set_yticklabels([str(k) for k in k_values])

        # Show every 5th layer on x-axis
        xticks = list(range(0, len(layers), 5))
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(layers[i]) for i in xticks])

        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    fig.suptitle(f'U Subspace Drift Heatmap (Layer × k){title_suffix}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_layer_k.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_difference_by_layer(results: Dict, layers: List[int], k_values: List[int],
                             params: List[str], output_dir: Path, comp_name: str = ""):
    """Plot SDFT - SFT difference by layer."""
    title_suffix = f" ({comp_name})" if comp_name else ""

    n_layers = len(layers)
    n_k = len(k_values)

    # Aggregate
    sft_U_avg = np.zeros((n_layers, n_k))
    sdft_U_avg = np.zeros((n_layers, n_k))

    for param in params:
        sft_U_avg += results['sft'][param]['U']
        sdft_U_avg += results['sdft'][param]['U']

    sft_U_avg /= len(params)
    sdft_U_avg /= len(params)

    # Difference in drift
    diff = drift_bp(sdft_U_avg) - drift_bp(sft_U_avg)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(k_values)))

    for ki, k in enumerate(k_values):
        ax.plot(layers, diff[:, ki], '-o', color=colors[ki], label=f'k={k}',
                linewidth=2, markersize=4)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.fill_between(layers, 0, np.max(diff, axis=1), alpha=0.1, color='red')
    ax.fill_between(layers, np.min(diff, axis=1), 0, alpha=0.1, color='blue')

    ax.set_xlabel('Layer')
    ax.set_ylabel('SDFT - SFT Drift (basis points)')
    ax.set_title(f'Difference in U Drift{title_suffix}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'difference_by_layer.png', dpi=150, bbox_inches='tight')
    plt.close()


def print_layer_summary(results: Dict, layers: List[int], k_values: List[int],
                        params: List[str]):
    """Print summary table by layer."""

    n_layers = len(layers)
    n_k = len(k_values)

    # Aggregate
    sft_U_avg = np.zeros((n_layers, n_k))
    sdft_U_avg = np.zeros((n_layers, n_k))

    for param in params:
        sft_U_avg += results['sft'][param]['U']
        sdft_U_avg += results['sdft'][param]['U']

    sft_U_avg /= len(params)
    sdft_U_avg /= len(params)

    print("\n" + "="*90)
    print("DRIFT BY LAYER (U, basis points, averaged over params)")
    print("="*90)

    # Header
    header = f"{'Layer':>6}"
    for k in k_values:
        header += f"  {'SFT k='+str(k):>10} {'SDFT k='+str(k):>10} {'Δ':>8}"
    print(header)
    print("-"*90)

    for li, layer in enumerate(layers):
        row = f"{layer:>6}"
        for ki, k in enumerate(k_values):
            sft_d = drift_bp(sft_U_avg[li, ki])
            sdft_d = drift_bp(sdft_U_avg[li, ki])
            diff = sdft_d - sft_d
            row += f"  {sft_d:>10.1f} {sdft_d:>10.1f} {diff:>+8.1f}"
        print(row)

    print("="*90)


def plot_all_comparisons(results: Dict, layers: List[int], k_values: List[int],
                         params: List[str], comparisons: List, output_dir: Path):
    """Plot comparison of all intermediate checkpoints."""

    n_layers = len(layers)
    n_k = len(k_values)

    # Aggregate across params for each comparison
    agg = {}
    for comp_name, _, _ in comparisons:
        sft_U = np.zeros((n_layers, n_k))
        sdft_U = np.zeros((n_layers, n_k))
        for param in params:
            sft_U += results[comp_name]['sft'][param]['U']
            sdft_U += results[comp_name]['sdft'][param]['U']
        agg[comp_name] = {
            'sft': sft_U / len(params),
            'sdft': sdft_U / len(params),
        }

    # Plot 1: Drift by layer for k=50, all comparisons
    ki_50 = k_values.index(50) if 50 in k_values else len(k_values) // 2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    comp_colors = {'base_to_tooluse': 'green', 'base_to_gsm8k': 'orange', 'base_to_mbpp': 'purple'}
    comp_labels = {'base_to_tooluse': 'base→tooluse', 'base_to_gsm8k': 'base→gsm8k', 'base_to_mbpp': 'base→mbpp'}

    for ax, method, title in [(axes[0], 'sft', 'SFT'), (axes[1], 'sdft', 'SDFT')]:
        for comp_name, _, _ in comparisons:
            drift = drift_bp(agg[comp_name][method][:, ki_50])
            ax.plot(layers, drift, '-o', color=comp_colors[comp_name],
                    label=comp_labels[comp_name], linewidth=2, markersize=4)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Drift (basis points)')
        ax.set_title(f'{title} (k={k_values[ki_50]})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Cumulative Drift: Base → Each Checkpoint', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'all_comparisons_by_layer.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Heatmap comparing all (rows = comparisons, cols = layers) for k=50
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    for ax, method, title in [(axes[0], 'sft', 'SFT'), (axes[1], 'sdft', 'SDFT')]:
        data = np.zeros((len(comparisons), n_layers))
        for ci, (comp_name, _, _) in enumerate(comparisons):
            data[ci, :] = drift_bp(agg[comp_name][method][:, ki_50])

        im = ax.imshow(data, aspect='auto', cmap='Reds', origin='upper')
        ax.set_yticks(range(len(comparisons)))
        ax.set_yticklabels([comp_labels[c[0]] for c in comparisons])

        xticks = list(range(0, len(layers), 5))
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(layers[i]) for i in xticks])
        ax.set_xlabel('Layer')
        ax.set_title(f'{title} Drift (k={k_values[ki_50]})')
        plt.colorbar(im, ax=ax, label='Drift (bp)')

    plt.tight_layout()
    plt.savefig(output_dir / 'all_comparisons_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: SDFT - SFT difference for each comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    for comp_name, _, _ in comparisons:
        diff = drift_bp(agg[comp_name]['sdft'][:, ki_50]) - drift_bp(agg[comp_name]['sft'][:, ki_50])
        ax.plot(layers, diff, '-o', color=comp_colors[comp_name],
                label=comp_labels[comp_name], linewidth=2, markersize=4)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('SDFT - SFT Drift (basis points)')
    ax.set_title(f'Difference (SDFT - SFT) by Layer (k={k_values[ki_50]})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'all_comparisons_difference.png', dpi=150, bbox_inches='tight')
    plt.close()


def print_layer_summary_all(results: Dict, layers: List[int], k_values: List[int],
                            params: List[str], comparisons: List):
    """Print summary table for all comparisons."""

    n_layers = len(layers)
    n_k = len(k_values)
    ki_50 = k_values.index(50) if 50 in k_values else len(k_values) // 2
    k_selected = k_values[ki_50]

    comp_labels = {'base_to_tooluse': 'base→tooluse', 'base_to_gsm8k': 'base→gsm8k', 'base_to_mbpp': 'base→mbpp'}

    print("\n" + "="*100)
    print(f"DRIFT SUMMARY BY LAYER (U, k={k_selected}, basis points, averaged over params)")
    print("="*100)

    # Header
    header = f"{'Layer':>6}"
    for comp_name, _, _ in comparisons:
        header += f"  {comp_labels[comp_name]+' SFT':>14} {comp_labels[comp_name]+' SDFT':>14} {'Δ':>8}"
    print(header)
    print("-"*100)

    # Aggregate
    agg = {}
    for comp_name, _, _ in comparisons:
        sft_U = np.zeros((n_layers, n_k))
        sdft_U = np.zeros((n_layers, n_k))
        for param in params:
            sft_U += results[comp_name]['sft'][param]['U']
            sdft_U += results[comp_name]['sdft'][param]['U']
        agg[comp_name] = {
            'sft': sft_U / len(params),
            'sdft': sdft_U / len(params),
        }

    for li, layer in enumerate(layers):
        row = f"{layer:>6}"
        for comp_name, _, _ in comparisons:
            sft_d = drift_bp(agg[comp_name]['sft'][li, ki_50])
            sdft_d = drift_bp(agg[comp_name]['sdft'][li, ki_50])
            diff = sdft_d - sft_d
            row += f"  {sft_d:>14.1f} {sdft_d:>14.1f} {diff:>+8.1f}"
        print(row)

    print("="*100)

    # Summary across all layers
    print("\nAVERAGE ACROSS ALL LAYERS:")
    print("-"*60)
    for comp_name, _, _ in comparisons:
        sft_avg = np.mean(drift_bp(agg[comp_name]['sft'][:, ki_50]))
        sdft_avg = np.mean(drift_bp(agg[comp_name]['sdft'][:, ki_50]))
        print(f"  {comp_labels[comp_name]:>15}: SFT={sft_avg:6.1f}bp, SDFT={sdft_avg:6.1f}bp, Δ={sdft_avg-sft_avg:+6.1f}bp")
    print("-"*60)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze subspace drift by layer for different k values'
    )
    parser.add_argument('--data_root', type=str, default=DATA_ROOT)
    parser.add_argument('--output_dir', type=str,
                        default='eigenspectrum/outputs/layer_k_sweep')
    parser.add_argument('--layers', type=int, nargs='+', default=None,
                        help='Specific layers (default: all 36)')
    parser.add_argument('--k_values', type=int, nargs='+', default=K_VALUES,
                        help='k values to sweep (default: 5 10 20 50 100)')
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()

    analyze_all_layers(
        data_root=args.data_root,
        output_dir=args.output_dir,
        layers=args.layers,
        k_values=args.k_values,
        gpu=not args.cpu,
    )


if __name__ == '__main__':
    main()
