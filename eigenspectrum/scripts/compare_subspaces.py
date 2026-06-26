#!/usr/bin/env python3
"""
Pairwise subspace comparison between checkpoints.

For each pair of checkpoints (i, j), compute:
- Alignment matrix A_U = U_i^T @ U_j
- Alignment matrix A_V = V_i^T @ V_j
- Summary metrics (diagonal mass, Grassmann distance, etc.)

Produces comparison matrices showing how subspaces evolve.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Configuration
# =============================================================================

DATA_ROOT = "/data/saket/continual/Self-Distillation"

def get_checkpoints(data_root: str) -> Dict[str, List[Tuple[str, str]]]:
    """Return (label, path) pairs for each method."""
    return {
        'sft': [
            ("base", "Qwen/Qwen2.5-3B-Instruct"),
            ("tooluse", f"{data_root}/outputs/sft_full_mbpp/task_0_tooluse/checkpoint-127"),
            ("gsm8k", f"{data_root}/outputs/sft_full_mbpp/task_1_gsm8k/checkpoint-234"),
            ("mbpp", f"{data_root}/outputs/sft_full_mbpp/task_2_mbpp/checkpoint-12"),
        ],
        'sdft': [
            ("base", "Qwen/Qwen2.5-3B-Instruct"),
            ("tooluse", f"{data_root}/outputs/sdft_full_mbpp/task_0_tooluse/checkpoint-1011"),
            ("gsm8k", f"{data_root}/outputs/sdft_full_mbpp/task_1_gsm8k/checkpoint-1868"),
            ("mbpp", f"{data_root}/outputs/sdft_full_mbpp/task_2_mbpp/checkpoint-93"),
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

PARAM_SHORT = {
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
# Matrix Loading
# =============================================================================

def load_weight_matrix(checkpoint_path: str, layer_idx: int, param_name: str) -> np.ndarray:
    """Load a single weight matrix from checkpoint."""
    import torch
    from safetensors import safe_open

    path = Path(checkpoint_path)
    full_param_name = f"model.layers.{layer_idx}.{param_name}"

    if path.exists() and path.is_dir():
        # Local checkpoint - try safetensors
        safetensor_files = list(path.glob("*.safetensors"))
        if safetensor_files:
            for sf_path in safetensor_files:
                with safe_open(sf_path, framework="pt", device="cpu") as f:
                    if full_param_name in f.keys():
                        return f.get_tensor(full_param_name).float().numpy()

        # Try pytorch bin
        pytorch_files = list(path.glob("*.bin")) + list(path.glob("pytorch_model*.bin"))
        for pt_path in pytorch_files:
            state_dict = torch.load(pt_path, map_location='cpu', weights_only=True)
            if full_param_name in state_dict:
                W = state_dict[full_param_name].float().numpy()
                del state_dict
                return W

        raise FileNotFoundError(f"Could not find {full_param_name} in {path}")

    else:
        # HuggingFace - download safetensors
        from huggingface_hub import hf_hub_download, list_repo_files

        files = list_repo_files(checkpoint_path)
        safetensor_files = [f for f in files if f.endswith('.safetensors')]

        for sf_file in safetensor_files:
            local_path = hf_hub_download(checkpoint_path, sf_file)
            with safe_open(local_path, framework="pt", device="cpu") as f:
                if full_param_name in f.keys():
                    return f.get_tensor(full_param_name).float().numpy()

        raise FileNotFoundError(f"Could not find {full_param_name} in {checkpoint_path}")


# =============================================================================
# SVD and Subspace Comparison
# =============================================================================

def compute_svd(W: np.ndarray, use_gpu: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute SVD, return (U, S, V) where V = Vh.T"""
    if use_gpu:
        import torch
        if torch.cuda.is_available():
            W_t = torch.tensor(W, dtype=torch.float32, device='cuda')
            U, S, Vh = torch.linalg.svd(W_t, full_matrices=False)
            return U.cpu().numpy(), S.cpu().numpy(), Vh.T.cpu().numpy()

    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    return U, S, Vh.T


def subspace_alignment(U1: np.ndarray, U2: np.ndarray, top_k: int) -> Dict[str, float]:
    """
    Compute alignment metrics between two sets of singular vectors.

    Args:
        U1, U2: Singular vectors (n x k)
        top_k: Number of top directions to consider

    Returns:
        Dictionary of alignment metrics
    """
    k = min(top_k, U1.shape[1], U2.shape[1])
    U1 = U1[:, :k]
    U2 = U2[:, :k]

    # Alignment matrix
    A = U1.T @ U2  # (k x k)

    # Absolute alignment (handles sign ambiguity)
    A_abs = np.abs(A)

    # Diagonal metrics
    diag = np.diag(A_abs)
    mean_diag = float(np.mean(diag))
    min_diag = float(np.min(diag))

    # Off-diagonal
    mask = ~np.eye(k, dtype=bool)
    off_diag = A_abs[mask]
    mean_off_diag = float(np.mean(off_diag))
    max_off_diag = float(np.max(off_diag))

    # Diagonal ratio
    diag_ratio = float(np.sum(diag) / np.sum(A_abs))

    # Grassmann distance (geodesic on Grassmann manifold)
    # cos(theta_i) = singular values of A
    s = np.linalg.svd(A, compute_uv=False)
    s = np.clip(s, -1, 1)
    principal_angles = np.arccos(s)
    grassmann_dist = float(np.linalg.norm(principal_angles))

    # Frobenius distance from identity
    frob_dist = float(np.linalg.norm(A_abs - np.eye(k), 'fro'))

    return {
        'mean_diag': mean_diag,
        'min_diag': min_diag,
        'mean_off_diag': mean_off_diag,
        'max_off_diag': max_off_diag,
        'diag_ratio': diag_ratio,
        'grassmann_dist': grassmann_dist,
        'frob_dist': frob_dist,
    }


def compare_checkpoints_pairwise(
    checkpoints: List[Tuple[str, str]],
    layer_idx: int,
    param_name: str,
    top_k: int = 50,
    use_gpu: bool = True,
) -> Dict:
    """
    Compare all pairs of checkpoints for a given layer/param.

    Returns:
        Dictionary with pairwise comparison matrices
    """
    n = len(checkpoints)
    labels = [c[0] for c in checkpoints]
    paths = [c[1] for c in checkpoints]

    # Load all matrices and compute SVDs
    svds = []
    for label, path in checkpoints:
        W = load_weight_matrix(path, layer_idx, param_name)
        U, S, V = compute_svd(W, use_gpu=use_gpu)
        svds.append({'U': U, 'S': S, 'V': V, 'label': label})

    # Pairwise comparison matrices
    U_mean_diag = np.zeros((n, n))
    V_mean_diag = np.zeros((n, n))
    U_grassmann = np.zeros((n, n))
    V_grassmann = np.zeros((n, n))

    pairwise_results = {}

    for i in range(n):
        for j in range(n):
            U_metrics = subspace_alignment(svds[i]['U'], svds[j]['U'], top_k)
            V_metrics = subspace_alignment(svds[i]['V'], svds[j]['V'], top_k)

            U_mean_diag[i, j] = U_metrics['mean_diag']
            V_mean_diag[i, j] = V_metrics['mean_diag']
            U_grassmann[i, j] = U_metrics['grassmann_dist']
            V_grassmann[i, j] = V_metrics['grassmann_dist']

            key = f"{labels[i]}_vs_{labels[j]}"
            pairwise_results[key] = {
                'U': U_metrics,
                'V': V_metrics,
            }

    return {
        'labels': labels,
        'U_mean_diag': U_mean_diag,
        'V_mean_diag': V_mean_diag,
        'U_grassmann': U_grassmann,
        'V_grassmann': V_grassmann,
        'pairwise': pairwise_results,
    }


# =============================================================================
# Full Analysis
# =============================================================================

def run_analysis(
    data_root: str,
    output_dir: str,
    layers: Optional[List[int]] = None,
    params: Optional[List[str]] = None,
    top_k: int = 50,
    use_gpu: bool = True,
):
    """Run full pairwise comparison for SFT and SDFT."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = get_checkpoints(data_root)

    if layers is None:
        layers = list(range(NUM_LAYERS))
    if params is None:
        params = PARAM_NAMES

    results = {'sft': {}, 'sdft': {}}

    # Aggregate matrices for summary plots
    # Shape: (n_layers, n_checkpoints, n_checkpoints)
    n_ckpts = len(checkpoints['sft'])
    agg_U_diag = {m: {p: np.zeros((len(layers), n_ckpts, n_ckpts)) for p in params}
                  for m in ['sft', 'sdft']}
    agg_V_diag = {m: {p: np.zeros((len(layers), n_ckpts, n_ckpts)) for p in params}
                  for m in ['sft', 'sdft']}

    total = len(layers) * len(params) * 2
    count = 0

    for method in ['sft', 'sdft']:
        print(f"\n{'='*60}")
        print(f"{method.upper()}")
        print('='*60)

        for layer_idx in layers:
            for param_name in params:
                count += 1
                param_short = PARAM_SHORT[param_name]
                print(f"[{count}/{total}] {method} L{layer_idx} {param_short}...", end=' ', flush=True)

                try:
                    result = compare_checkpoints_pairwise(
                        checkpoints[method],
                        layer_idx,
                        param_name,
                        top_k=top_k,
                        use_gpu=use_gpu,
                    )

                    key = f"layer{layer_idx}_{param_short}"
                    results[method][key] = {
                        'labels': result['labels'],
                        'pairwise': result['pairwise'],
                    }

                    # Store in aggregate
                    l_idx = layers.index(layer_idx)
                    agg_U_diag[method][param_name][l_idx] = result['U_mean_diag']
                    agg_V_diag[method][param_name][l_idx] = result['V_mean_diag']

                    # Print summary (base vs final)
                    base_vs_mbpp = result['pairwise']['base_vs_mbpp']
                    print(f"base→mbpp: U={base_vs_mbpp['U']['mean_diag']:.3f}, V={base_vs_mbpp['V']['mean_diag']:.3f}")

                except Exception as e:
                    print(f"ERROR: {e}")

    # Save raw results
    results_path = output_dir / 'pairwise_results.json'
    with open(results_path, 'w') as f:
        # Convert numpy arrays in pairwise to lists for JSON
        json_results = {}
        for method in results:
            json_results[method] = {}
            for key, val in results[method].items():
                json_results[method][key] = val
        json.dump(json_results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Generate plots
    print("\nGenerating plots...")
    plot_pairwise_matrices(agg_U_diag, agg_V_diag, layers, params, checkpoints, output_dir)
    plot_comparison_summary(agg_U_diag, agg_V_diag, layers, params, output_dir)

    return results


def plot_pairwise_matrices(
    agg_U: Dict,
    agg_V: Dict,
    layers: List[int],
    params: List[str],
    checkpoints: Dict,
    output_dir: Path,
):
    """Plot pairwise alignment matrices averaged over layers."""

    labels = [c[0] for c in checkpoints['sft']]
    n = len(labels)

    for param_name in params:
        param_short = PARAM_SHORT[param_name]

        # Average over layers
        sft_U = np.mean(agg_U['sft'][param_name], axis=0)
        sft_V = np.mean(agg_V['sft'][param_name], axis=0)
        sdft_U = np.mean(agg_U['sdft'][param_name], axis=0)
        sdft_V = np.mean(agg_V['sdft'][param_name], axis=0)

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        for ax, data, title in [
            (axes[0, 0], sft_U, 'SFT - U alignment'),
            (axes[0, 1], sdft_U, 'SDFT - U alignment'),
            (axes[1, 0], sft_V, 'SFT - V alignment'),
            (axes[1, 1], sdft_V, 'SDFT - V alignment'),
        ]:
            im = ax.imshow(data, cmap='viridis', vmin=0, vmax=1)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(labels, rotation=45)
            ax.set_yticklabels(labels)
            ax.set_title(title)

            # Annotate cells
            for i in range(n):
                for j in range(n):
                    color = 'white' if data[i, j] < 0.5 else 'black'
                    ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                            color=color, fontsize=10)

            plt.colorbar(im, ax=ax, fraction=0.046)

        fig.suptitle(f'{param_short}: Pairwise Subspace Alignment (mean |diag|, averaged over layers)', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / f'pairwise_{param_short}.png', dpi=150)
        plt.close()

    print(f"Saved pairwise matrices to {output_dir}")


def plot_comparison_summary(
    agg_U: Dict,
    agg_V: Dict,
    layers: List[int],
    params: List[str],
    output_dir: Path,
):
    """Plot summary comparing SFT vs SDFT drift."""

    # Extract base→final comparison for each layer
    # Index 0 = base, index 3 = mbpp (final)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, param_name in enumerate(params):
        ax = axes[idx]
        param_short = PARAM_SHORT[param_name]

        # base→mbpp alignment (row 0, col 3)
        sft_U_drift = agg_U['sft'][param_name][:, 0, 3]  # (n_layers,)
        sft_V_drift = agg_V['sft'][param_name][:, 0, 3]
        sdft_U_drift = agg_U['sdft'][param_name][:, 0, 3]
        sdft_V_drift = agg_V['sdft'][param_name][:, 0, 3]

        ax.plot(layers, sft_U_drift, 'b-o', label='SFT U', markersize=3, linewidth=1.5)
        ax.plot(layers, sdft_U_drift, 'r-o', label='SDFT U', markersize=3, linewidth=1.5)
        ax.plot(layers, sft_V_drift, 'b--s', label='SFT V', markersize=3, linewidth=1.5, alpha=0.6)
        ax.plot(layers, sdft_V_drift, 'r--s', label='SDFT V', markersize=3, linewidth=1.5, alpha=0.6)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Alignment (mean |diag|)')
        ax.set_title(param_short)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    axes[-1].axis('off')

    fig.suptitle('Base → MBPP: Singular Vector Preservation by Layer', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'base_to_final_comparison.png', dpi=150)
    plt.close()

    # Summary bar chart: average drift per method
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(params))
    width = 0.35

    for ax, agg, title in [(axes[0], agg_U, 'U (left singular vectors)'),
                            (axes[1], agg_V, 'V (right singular vectors)')]:
        sft_means = []
        sdft_means = []

        for param_name in params:
            sft_means.append(np.mean(agg['sft'][param_name][:, 0, 3]))
            sdft_means.append(np.mean(agg['sdft'][param_name][:, 0, 3]))

        bars1 = ax.bar(x - width/2, sft_means, width, label='SFT', color='blue', alpha=0.7)
        bars2 = ax.bar(x + width/2, sdft_means, width, label='SDFT', color='red', alpha=0.7)

        ax.set_ylabel('Mean Alignment (base→mbpp)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([PARAM_SHORT[p] for p in params], rotation=45)
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars1, sft_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', fontsize=8)
        for bar, val in zip(bars2, sdft_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', fontsize=8)

    fig.suptitle('Average Subspace Preservation: Base → Final (higher = less drift)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_bar_chart.png', dpi=150)
    plt.close()

    # Incremental drift: base→tooluse→gsm8k→mbpp
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    transitions = [
        ('base→tooluse', 0, 1),
        ('tooluse→gsm8k', 1, 2),
        ('gsm8k→mbpp', 2, 3),
    ]

    for idx, param_name in enumerate(params):
        ax = axes[idx]
        param_short = PARAM_SHORT[param_name]

        for method, color in [('sft', 'blue'), ('sdft', 'red')]:
            values = []
            for _, i, j in transitions:
                values.append(np.mean(agg_U[method][param_name][:, i, j]))

            ax.plot([t[0] for t in transitions], values, 'o-', color=color,
                    label=f'{method.upper()}', linewidth=2, markersize=6)

        ax.set_ylabel('Alignment')
        ax.set_title(param_short)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()

    axes[-1].axis('off')

    fig.suptitle('Incremental Subspace Drift (U) Between Consecutive Tasks', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'incremental_drift.png', dpi=150)
    plt.close()

    print(f"Saved summary plots to {output_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Pairwise subspace comparison')
    parser.add_argument('--data_root', type=str, default=DATA_ROOT)
    parser.add_argument('--output_dir', type=str, default='eigenspectrum/outputs/subspace_comparison')
    parser.add_argument('--layers', type=int, nargs='+', default=None)
    parser.add_argument('--top_k', type=int, default=TOP_K)
    parser.add_argument('--cpu', action='store_true', help='Force CPU')

    args = parser.parse_args()

    run_analysis(
        data_root=args.data_root,
        output_dir=args.output_dir,
        layers=args.layers,
        top_k=args.top_k,
        use_gpu=not args.cpu,
    )


if __name__ == '__main__':
    main()
