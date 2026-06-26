#!/usr/bin/env python3
"""
Sweep over k values to analyze how subspace alignment changes
with the number of top singular vectors considered.

For each k in [5, 10, 20, 50, 100]:
- Compute alignment of top-k singular subspace between base and final checkpoint
- Compare SFT vs SDFT
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

K_VALUES = [5, 10, 20, 50, 100]

CHECKPOINT_LABELS = ['base', 'tooluse', 'gsm8k', 'mbpp']

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
    """
    Compute alignment of top-k subspaces.
    Returns mean |diagonal| of U1[:,:k].T @ U2[:,:k].
    """
    k = min(k, U1.shape[1], U2.shape[1])
    A = U1[:, :k].T @ U2[:, :k]
    return float(np.mean(np.abs(np.diag(A))))


# =============================================================================
# Analysis
# =============================================================================

def analyze_layer_sweep_k(
    paths: Dict[str, List[str]],
    layer: int,
    param: str,
    k_values: List[int],
    gpu: bool = True,
) -> Dict:
    """
    For one layer/param, compute alignment for each k value.

    Returns dict with alignment values for each k.
    """
    # Load base and final checkpoints
    W_base = load_weight(paths['sft'][0], layer, param)  # Same for both
    W_sft_final = load_weight(paths['sft'][3], layer, param)
    W_sdft_final = load_weight(paths['sdft'][3], layer, param)

    # Compute full SVD
    U_base, _, V_base = svd_full(W_base, gpu)
    U_sft, _, V_sft = svd_full(W_sft_final, gpu)
    U_sdft, _, V_sdft = svd_full(W_sdft_final, gpu)

    results = {
        'k_values': k_values,
        'sft_U': [],
        'sft_V': [],
        'sdft_U': [],
        'sdft_V': [],
    }

    for k in k_values:
        results['sft_U'].append(alignment_topk(U_base, U_sft, k))
        results['sft_V'].append(alignment_topk(V_base, V_sft, k))
        results['sdft_U'].append(alignment_topk(U_base, U_sdft, k))
        results['sdft_V'].append(alignment_topk(V_base, V_sdft, k))

    return results


def run_sweep(
    data_root: str,
    output_dir: str,
    layers: List[int] = None,
    params: List[str] = None,
    k_values: List[int] = None,
    gpu: bool = True,
):
    """Run k-sweep analysis across all layers and params."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = get_checkpoint_paths(data_root)

    if layers is None:
        layers = list(range(NUM_LAYERS))
    if params is None:
        params = PARAM_NAMES
    if k_values is None:
        k_values = K_VALUES

    # Storage: results[param][layer] = {k_values, sft_U, sft_V, sdft_U, sdft_V}
    results = {p: {} for p in params}

    # Aggregated: agg[method][param][k_idx] = array of values across layers
    agg = {
        method: {
            param: {
                'U': {k: [] for k in k_values},
                'V': {k: [] for k in k_values},
            }
            for param in params
        }
        for method in ['sft', 'sdft']
    }

    total = len(layers) * len(params)
    count = 0

    for param in params:
        pshort = PARAM_SHORT[param]

        for layer in layers:
            count += 1
            print(f"[{count}/{total}] Layer {layer} {pshort}...", end=' ', flush=True)

            try:
                res = analyze_layer_sweep_k(paths, layer, param, k_values, gpu)
                results[param][layer] = res

                # Store in aggregated
                for ki, k in enumerate(k_values):
                    agg['sft'][param]['U'][k].append(res['sft_U'][ki])
                    agg['sft'][param]['V'][k].append(res['sft_V'][ki])
                    agg['sdft'][param]['U'][k].append(res['sdft_U'][ki])
                    agg['sdft'][param]['V'][k].append(res['sdft_V'][ki])

                # Print summary for k=50
                idx_50 = k_values.index(50) if 50 in k_values else -1
                if idx_50 >= 0:
                    sft_drift = (1 - res['sft_U'][idx_50]) * 10000
                    sdft_drift = (1 - res['sdft_U'][idx_50]) * 10000
                    print(f"k=50: SFT={sft_drift:.1f}bp, SDFT={sdft_drift:.1f}bp")
                else:
                    print("done")

            except Exception as e:
                print(f"ERROR: {e}")

    # Save results
    save_results(results, k_values, output_dir)

    # Generate plots
    print("\nGenerating plots...")
    plot_k_sweep_by_param(agg, k_values, layers, params, output_dir)
    plot_k_sweep_summary(agg, k_values, params, output_dir)
    print_summary(agg, k_values, params)

    return results, agg


def save_results(results: Dict, k_values: List[int], output_dir: Path):
    """Save results to JSON."""
    json_results = {}
    for param in results:
        json_results[param] = {}
        for layer in results[param]:
            json_results[param][str(layer)] = results[param][layer]

    with open(output_dir / 'k_sweep_results.json', 'w') as f:
        json.dump({
            'k_values': k_values,
            'data': json_results,
        }, f, indent=2)


def plot_k_sweep_by_param(
    agg: Dict,
    k_values: List[int],
    layers: List[int],
    params: List[str],
    output_dir: Path,
):
    """Plot drift vs k for each parameter."""

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, param in enumerate(params):
        ax = axes[idx]
        pshort = PARAM_SHORT[param]

        # Average drift across layers for each k
        for method, color, marker in [('sft', 'blue', 'o'), ('sdft', 'red', 's')]:
            drift_U = [(1 - np.mean(agg[method][param]['U'][k])) * 10000 for k in k_values]
            drift_V = [(1 - np.mean(agg[method][param]['V'][k])) * 10000 for k in k_values]

            ax.plot(k_values, drift_U, f'-{marker}', color=color,
                    label=f'{method.upper()} U', linewidth=2, markersize=6)
            ax.plot(k_values, drift_V, f'--{marker}', color=color,
                    label=f'{method.upper()} V', linewidth=1.5, markersize=5, alpha=0.6)

        ax.set_xlabel('k (top singular vectors)')
        ax.set_ylabel('Drift (basis points)')
        ax.set_title(pshort)
        ax.set_xscale('log')
        ax.set_xticks(k_values)
        ax.set_xticklabels([str(k) for k in k_values])
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(fontsize=7)

    axes[-1].axis('off')

    fig.suptitle('Subspace Drift vs k: Base → MBPP (averaged over layers)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'drift_vs_k_by_param.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_k_sweep_summary(
    agg: Dict,
    k_values: List[int],
    params: List[str],
    output_dir: Path,
):
    """Summary plot: average across all params."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, sv, title in [(axes[0], 'U', 'Left Singular Vectors (U)'),
                           (axes[1], 'V', 'Right Singular Vectors (V)')]:

        for method, color, marker in [('sft', 'blue', 'o'), ('sdft', 'red', 's')]:
            # Average across all params and layers
            drift_by_k = []
            for k in k_values:
                all_vals = []
                for param in params:
                    all_vals.extend(agg[method][param][sv][k])
                drift_by_k.append((1 - np.mean(all_vals)) * 10000)

            ax.plot(k_values, drift_by_k, f'-{marker}', color=color,
                    label=method.upper(), linewidth=2, markersize=8)

        ax.set_xlabel('k (top singular vectors)')
        ax.set_ylabel('Drift (basis points)')
        ax.set_title(title)
        ax.set_xscale('log')
        ax.set_xticks(k_values)
        ax.set_xticklabels([str(k) for k in k_values])
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Average Subspace Drift vs k (all params, all layers)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'drift_vs_k_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Also plot DIFFERENCE (SDFT - SFT)
    fig, ax = plt.subplots(figsize=(8, 5))

    for sv, linestyle, label in [('U', '-', 'U (left)'), ('V', '--', 'V (right)')]:
        diff_by_k = []
        for k in k_values:
            sft_vals = []
            sdft_vals = []
            for param in params:
                sft_vals.extend(agg['sft'][param][sv][k])
                sdft_vals.extend(agg['sdft'][param][sv][k])

            sft_drift = (1 - np.mean(sft_vals)) * 10000
            sdft_drift = (1 - np.mean(sdft_vals)) * 10000
            diff_by_k.append(sdft_drift - sft_drift)

        ax.plot(k_values, diff_by_k, f'{linestyle}o', label=label, linewidth=2, markersize=8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('k (top singular vectors)')
    ax.set_ylabel('SDFT - SFT Drift (basis points)')
    ax.set_title('Difference in Drift: Positive = SDFT drifts more')
    ax.set_xscale('log')
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'drift_vs_k_difference.png', dpi=150, bbox_inches='tight')
    plt.close()


def print_summary(agg: Dict, k_values: List[int], params: List[str]):
    """Print summary table."""

    print("\n" + "="*80)
    print("SUMMARY: Drift (basis points) by k")
    print("="*80)

    # Header
    header = f"{'k':>6}"
    for method in ['SFT', 'SDFT']:
        header += f"  {method + ' U':>10} {method + ' V':>10}"
    header += f"  {'Δ U':>10} {'Δ V':>10}"
    print(header)
    print("-"*80)

    for k in k_values:
        sft_u_vals = []
        sft_v_vals = []
        sdft_u_vals = []
        sdft_v_vals = []

        for param in params:
            sft_u_vals.extend(agg['sft'][param]['U'][k])
            sft_v_vals.extend(agg['sft'][param]['V'][k])
            sdft_u_vals.extend(agg['sdft'][param]['U'][k])
            sdft_v_vals.extend(agg['sdft'][param]['V'][k])

        sft_u = (1 - np.mean(sft_u_vals)) * 10000
        sft_v = (1 - np.mean(sft_v_vals)) * 10000
        sdft_u = (1 - np.mean(sdft_u_vals)) * 10000
        sdft_v = (1 - np.mean(sdft_v_vals)) * 10000

        print(f"{k:>6}  {sft_u:>10.2f} {sft_v:>10.2f}  {sdft_u:>10.2f} {sdft_v:>10.2f}"
              f"  {sdft_u - sft_u:>+10.2f} {sdft_v - sft_v:>+10.2f}")

    print("="*80)
    print("Δ > 0 means SDFT drifts MORE than SFT")
    print("Δ < 0 means SDFT drifts LESS than SFT (preserves subspace better)")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Sweep over k values to analyze subspace alignment'
    )
    parser.add_argument('--data_root', type=str, default=DATA_ROOT)
    parser.add_argument('--output_dir', type=str,
                        default='eigenspectrum/outputs/k_sweep')
    parser.add_argument('--layers', type=int, nargs='+', default=None,
                        help='Specific layers (default: all 36)')
    parser.add_argument('--k_values', type=int, nargs='+', default=K_VALUES,
                        help='k values to sweep (default: 5 10 20 50 100)')
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()

    run_sweep(
        data_root=args.data_root,
        output_dir=args.output_dir,
        layers=args.layers,
        k_values=args.k_values,
        gpu=not args.cpu,
    )


if __name__ == '__main__':
    main()
