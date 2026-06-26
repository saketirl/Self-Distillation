#!/usr/bin/env python3
"""
Compare subspace drift between SFT and SDFT across all layers.

Checkpoints:
- Base (same for both)
- After tooluse training (SFT vs SDFT)
- After gsm8k training (SFT vs SDFT)
- After mbpp training (SFT vs SDFT)

For each method, compute 4x4 pairwise alignment matrix.
Compare how much subspaces drift in SFT vs SDFT.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Configuration
# =============================================================================

DATA_ROOT = "/data/saket/continual/Self-Distillation"

CHECKPOINT_LABELS = ['base', 'tooluse', 'gsm8k', 'mbpp']

def get_checkpoint_paths(data_root: str) -> Dict[str, List[str]]:
    """Get checkpoint paths for SFT and SDFT."""
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
TOP_K = 50


# =============================================================================
# Loading and SVD
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


def svd(W: np.ndarray, top_k: int = None, gpu: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return U, S, V (not Vh).

    If top_k is specified, only return top-k singular vectors/values.
    """
    if gpu:
        import torch
        if torch.cuda.is_available():
            t = torch.tensor(W, dtype=torch.float32, device='cuda')
            U, S, Vh = torch.linalg.svd(t, full_matrices=False)
            U, S, V = U.cpu().numpy(), S.cpu().numpy(), Vh.T.cpu().numpy()
            if top_k is not None:
                return U[:, :top_k], S[:top_k], V[:, :top_k]
            return U, S, V

    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    if top_k is not None:
        return U[:, :top_k], S[:top_k], Vh.T[:, :top_k]
    return U, S, Vh.T


def alignment_score(U1: np.ndarray, U2: np.ndarray, k: int) -> float:
    """
    Compute mean |diagonal| of U1[:,:k].T @ U2[:,:k].

    Returns value in [0, 1]:
    - 1.0 = identical subspaces
    - 0.0 = orthogonal subspaces
    """
    k = min(k, U1.shape[1], U2.shape[1])
    A = U1[:, :k].T @ U2[:, :k]
    return float(np.mean(np.abs(np.diag(A))))


# =============================================================================
# Main Analysis
# =============================================================================

def analyze_layer(
    paths_sft: List[str],
    paths_sdft: List[str],
    layer: int,
    param: str,
    top_k: int,
    gpu: bool,
) -> Dict:
    """
    Compute pairwise alignment matrices for one layer/param.

    Only computes SVD for top-k singular vectors (faster and focuses on
    the most important directions).

    Returns dict with:
    - sft_U: 4x4 alignment matrix for U (SFT)
    - sft_V: 4x4 alignment matrix for V (SFT)
    - sdft_U: 4x4 alignment matrix for U (SDFT)
    - sdft_V: 4x4 alignment matrix for V (SDFT)
    """
    n = len(CHECKPOINT_LABELS)

    # Load and compute SVD for all checkpoints (only top-k)
    sft_svds = []
    sdft_svds = []

    for i in range(n):
        # SFT
        W_sft = load_weight(paths_sft[i], layer, param)
        U, S, V = svd(W_sft, top_k=top_k, gpu=gpu)
        sft_svds.append((U, S, V))

        # SDFT (base is same, but load anyway for simplicity)
        W_sdft = load_weight(paths_sdft[i], layer, param)
        U, S, V = svd(W_sdft, top_k=top_k, gpu=gpu)
        sdft_svds.append((U, S, V))

    # Compute pairwise alignments (U and V are already truncated to top-k)
    sft_U = np.zeros((n, n))
    sft_V = np.zeros((n, n))
    sdft_U = np.zeros((n, n))
    sdft_V = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Now U is already (m, top_k), so use all columns
            k = sft_svds[i][0].shape[1]  # = top_k
            sft_U[i, j] = alignment_score(sft_svds[i][0], sft_svds[j][0], k)
            sft_V[i, j] = alignment_score(sft_svds[i][2], sft_svds[j][2], k)
            sdft_U[i, j] = alignment_score(sdft_svds[i][0], sdft_svds[j][0], k)
            sdft_V[i, j] = alignment_score(sdft_svds[i][2], sdft_svds[j][2], k)

    return {
        'sft_U': sft_U,
        'sft_V': sft_V,
        'sdft_U': sdft_U,
        'sdft_V': sdft_V,
    }


def run_analysis(
    data_root: str,
    output_dir: str,
    layers: Optional[List[int]] = None,
    params: Optional[List[str]] = None,
    top_k: int = TOP_K,
    gpu: bool = True,
):
    """Run full analysis across all layers and params."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = get_checkpoint_paths(data_root)

    if layers is None:
        layers = list(range(NUM_LAYERS))
    if params is None:
        params = PARAM_NAMES

    # Storage: results[param][layer] = {sft_U, sft_V, sdft_U, sdft_V}
    results = {p: {} for p in params}

    # Aggregated across layers for plotting
    # Shape: (n_layers, 4, 4) for each method/param
    n = len(CHECKPOINT_LABELS)
    agg = {
        method: {
            param: {
                'U': np.zeros((len(layers), n, n)),
                'V': np.zeros((len(layers), n, n)),
            }
            for param in params
        }
        for method in ['sft', 'sdft']
    }

    total = len(layers) * len(params)
    count = 0

    for param in params:
        pshort = PARAM_SHORT[param]

        for li, layer in enumerate(layers):
            count += 1
            print(f"[{count}/{total}] Layer {layer} {pshort}...", end=' ', flush=True)

            try:
                res = analyze_layer(
                    paths['sft'], paths['sdft'],
                    layer, param, top_k, gpu
                )

                results[param][layer] = res

                agg['sft'][param]['U'][li] = res['sft_U']
                agg['sft'][param]['V'][li] = res['sft_V']
                agg['sdft'][param]['U'][li] = res['sdft_U']
                agg['sdft'][param]['V'][li] = res['sdft_V']

                # Print base→mbpp comparison
                sft_drift = res['sft_U'][0, 3]
                sdft_drift = res['sdft_U'][0, 3]
                print(f"base→mbpp: SFT={sft_drift:.3f}, SDFT={sdft_drift:.3f}, diff={sdft_drift-sft_drift:+.3f}")

            except Exception as e:
                print(f"ERROR: {e}")

    # Save results
    print("\nSaving results...")
    save_results(results, layers, output_dir)

    # Generate plots
    print("Generating plots...")
    plot_pairwise_heatmaps(agg, layers, params, output_dir)
    plot_drift_by_layer(agg, layers, params, output_dir)
    plot_summary(agg, layers, params, output_dir)

    print(f"\nDone! Results saved to {output_dir}")

    return results, agg


def save_results(results: Dict, layers: List[int], output_dir: Path):
    """Save results to JSON."""
    # Convert numpy to lists
    json_results = {}
    for param in results:
        json_results[param] = {}
        for layer in results[param]:
            json_results[param][str(layer)] = {
                k: v.tolist() for k, v in results[param][layer].items()
            }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump({
            'labels': CHECKPOINT_LABELS,
            'layers': layers,
            'data': json_results,
        }, f, indent=2)


def plot_pairwise_heatmaps(agg: Dict, layers: List[int], params: List[str], output_dir: Path):
    """Plot 4x4 pairwise matrices averaged over all layers."""

    n = len(CHECKPOINT_LABELS)

    for param in params:
        pshort = PARAM_SHORT[param]

        # Average over layers
        sft_U = np.mean(agg['sft'][param]['U'], axis=0)
        sft_V = np.mean(agg['sft'][param]['V'], axis=0)
        sdft_U = np.mean(agg['sdft'][param]['U'], axis=0)
        sdft_V = np.mean(agg['sdft'][param]['V'], axis=0)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Row 1: U alignments
        for ax, data, title in [
            (axes[0, 0], sft_U, 'SFT: U alignment'),
            (axes[0, 1], sdft_U, 'SDFT: U alignment'),
            (axes[0, 2], sdft_U - sft_U, 'SDFT - SFT (U)'),
        ]:
            if 'SDFT - SFT' in title:
                im = ax.imshow(data, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
            else:
                im = ax.imshow(data, cmap='viridis', vmin=0, vmax=1)

            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(CHECKPOINT_LABELS, rotation=45)
            ax.set_yticklabels(CHECKPOINT_LABELS)
            ax.set_title(title)

            for i in range(n):
                for j in range(n):
                    color = 'white' if abs(data[i,j]) > 0.5 else 'black'
                    ax.text(j, i, f'{data[i,j]:.2f}', ha='center', va='center',
                            color=color, fontsize=9)

            plt.colorbar(im, ax=ax, fraction=0.046)

        # Row 2: V alignments
        for ax, data, title in [
            (axes[1, 0], sft_V, 'SFT: V alignment'),
            (axes[1, 1], sdft_V, 'SDFT: V alignment'),
            (axes[1, 2], sdft_V - sft_V, 'SDFT - SFT (V)'),
        ]:
            if 'SDFT - SFT' in title:
                im = ax.imshow(data, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
            else:
                im = ax.imshow(data, cmap='viridis', vmin=0, vmax=1)

            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(CHECKPOINT_LABELS, rotation=45)
            ax.set_yticklabels(CHECKPOINT_LABELS)
            ax.set_title(title)

            for i in range(n):
                for j in range(n):
                    color = 'white' if abs(data[i,j]) > 0.5 else 'black'
                    ax.text(j, i, f'{data[i,j]:.2f}', ha='center', va='center',
                            color=color, fontsize=9)

            plt.colorbar(im, ax=ax, fraction=0.046)

        fig.suptitle(f'{pshort}: Pairwise Subspace Alignment (averaged over {len(layers)} layers)', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f'pairwise_{pshort}.png', dpi=150, bbox_inches='tight')
        plt.close()


def plot_drift_by_layer(agg: Dict, layers: List[int], params: List[str], output_dir: Path):
    """Plot base→mbpp DRIFT (1-alignment) by layer for each param."""

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, param in enumerate(params):
        ax = axes[idx]
        pshort = PARAM_SHORT[param]

        # base→mbpp is [0, 3] entry - convert to drift in basis points
        sft_U = (1 - agg['sft'][param]['U'][:, 0, 3]) * 10000
        sdft_U = (1 - agg['sdft'][param]['U'][:, 0, 3]) * 10000
        sft_V = (1 - agg['sft'][param]['V'][:, 0, 3]) * 10000
        sdft_V = (1 - agg['sdft'][param]['V'][:, 0, 3]) * 10000

        ax.plot(layers, sft_U, 'b-', label='SFT U', linewidth=2)
        ax.plot(layers, sdft_U, 'r-', label='SDFT U', linewidth=2)
        ax.plot(layers, sft_V, 'b--', label='SFT V', linewidth=1.5, alpha=0.7)
        ax.plot(layers, sdft_V, 'r--', label='SDFT V', linewidth=1.5, alpha=0.7)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Drift (basis points)')
        ax.set_title(pshort)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(fontsize=8)

    axes[-1].axis('off')

    fig.suptitle('Base → MBPP Subspace Drift by Layer (higher = more drift)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'drift_by_layer.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Also plot the DIFFERENCE (SDFT - SFT)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, param in enumerate(params):
        ax = axes[idx]
        pshort = PARAM_SHORT[param]

        diff_U = agg['sdft'][param]['U'][:, 0, 3] - agg['sft'][param]['U'][:, 0, 3]
        diff_V = agg['sdft'][param]['V'][:, 0, 3] - agg['sft'][param]['V'][:, 0, 3]

        ax.plot(layers, diff_U, 'g-', label='U (SDFT-SFT)', linewidth=2)
        ax.plot(layers, diff_V, 'm-', label='V (SDFT-SFT)', linewidth=2)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        ax.set_xlabel('Layer')
        ax.set_ylabel('SDFT - SFT')
        ax.set_title(pshort)
        ax.set_ylim(-0.3, 0.3)
        ax.grid(True, alpha=0.3)

        # Shade regions
        ax.fill_between(layers, 0, diff_U, where=(diff_U > 0), alpha=0.3, color='green', label='SDFT better')
        ax.fill_between(layers, 0, diff_U, where=(diff_U < 0), alpha=0.3, color='red', label='SFT better')

        if idx == 0:
            ax.legend(fontsize=8)

    axes[-1].axis('off')

    fig.suptitle('SDFT - SFT: Positive = SDFT preserves subspace better', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'drift_difference.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary(agg: Dict, layers: List[int], params: List[str], output_dir: Path):
    """Summary bar charts."""

    # Bar chart: average base→mbpp DRIFT (in basis points)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(params))
    width = 0.35

    for ax, sv, title in [(axes[0], 'U', 'Left Singular Vectors (U)'),
                           (axes[1], 'V', 'Right Singular Vectors (V)')]:

        # Convert to drift in basis points
        sft_vals = [(1 - np.mean(agg['sft'][p][sv][:, 0, 3])) * 10000 for p in params]
        sdft_vals = [(1 - np.mean(agg['sdft'][p][sv][:, 0, 3])) * 10000 for p in params]

        bars1 = ax.bar(x - width/2, sft_vals, width, label='SFT', color='blue', alpha=0.7)
        bars2 = ax.bar(x + width/2, sdft_vals, width, label='SDFT', color='red', alpha=0.7)

        ax.set_ylabel('Drift (basis points)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([PARAM_SHORT[p] for p in params], rotation=45)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.legend()

        # Labels
        for bar, val in zip(bars1, sft_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', fontsize=8)
        for bar, val in zip(bars2, sdft_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', fontsize=8)

    fig.suptitle('Average Subspace Drift: Base → MBPP (lower = less drift)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_bar.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Incremental drift plot
    transitions = [
        ('base→tooluse', 0, 1),
        ('tooluse→gsm8k', 1, 2),
        ('gsm8k→mbpp', 2, 3),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, param in enumerate(params):
        ax = axes[idx]
        pshort = PARAM_SHORT[param]

        for method, color, marker in [('sft', 'blue', 'o'), ('sdft', 'red', 's')]:
            vals = []
            for _, i, j in transitions:
                vals.append(np.mean(agg[method][param]['U'][:, i, j]))

            ax.plot(range(len(transitions)), vals, f'{color[0]}-{marker}',
                    label=method.upper(), linewidth=2, markersize=8)

        ax.set_xticks(range(len(transitions)))
        ax.set_xticklabels([t[0] for t in transitions], rotation=30)
        ax.set_ylabel('Alignment')
        ax.set_title(pshort)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend()

    axes[-1].axis('off')

    fig.suptitle('Incremental Subspace Drift (U) Between Tasks', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'incremental_drift.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Print summary stats - show DRIFT (1 - alignment) in basis points for clarity
    print("\n" + "="*80)
    print("SUMMARY: Subspace Drift = (1 - alignment) × 10000 basis points")
    print("         (0 = no drift, higher = more drift)")
    print("="*80)
    print(f"{'Parameter':<12} {'SFT U':>10} {'SDFT U':>10} {'Δ U':>10} {'SFT V':>10} {'SDFT V':>10} {'Δ V':>10}")
    print("-"*80)

    for param in params:
        pshort = PARAM_SHORT[param]
        # Convert to drift in basis points (× 10000)
        sft_u = (1 - np.mean(agg['sft'][param]['U'][:, 0, 3])) * 10000
        sdft_u = (1 - np.mean(agg['sdft'][param]['U'][:, 0, 3])) * 10000
        sft_v = (1 - np.mean(agg['sft'][param]['V'][:, 0, 3])) * 10000
        sdft_v = (1 - np.mean(agg['sdft'][param]['V'][:, 0, 3])) * 10000

        print(f"{pshort:<12} {sft_u:>10.2f} {sdft_u:>10.2f} {sdft_u-sft_u:>+10.2f} "
              f"{sft_v:>10.2f} {sdft_v:>10.2f} {sdft_v-sft_v:>+10.2f}")

    print("-"*80)
    # Overall average
    all_sft_u = (1 - np.mean([np.mean(agg['sft'][p]['U'][:, 0, 3]) for p in params])) * 10000
    all_sdft_u = (1 - np.mean([np.mean(agg['sdft'][p]['U'][:, 0, 3]) for p in params])) * 10000
    all_sft_v = (1 - np.mean([np.mean(agg['sft'][p]['V'][:, 0, 3]) for p in params])) * 10000
    all_sdft_v = (1 - np.mean([np.mean(agg['sdft'][p]['V'][:, 0, 3]) for p in params])) * 10000

    print(f"{'AVERAGE':<12} {all_sft_u:>10.2f} {all_sdft_u:>10.2f} {all_sdft_u-all_sft_u:>+10.2f} "
          f"{all_sft_v:>10.2f} {all_sdft_v:>10.2f} {all_sdft_v-all_sft_v:>+10.2f}")
    print("="*80)
    print("\nInterpretation: Δ > 0 means SDFT drifts MORE, Δ < 0 means SDFT drifts LESS")

    # Also print raw alignment for reference
    print("\n" + "="*80)
    print("RAW ALIGNMENT VALUES (for reference):")
    print("="*80)
    for param in params:
        pshort = PARAM_SHORT[param]
        sft_u = np.mean(agg['sft'][param]['U'][:, 0, 3])
        sdft_u = np.mean(agg['sdft'][param]['U'][:, 0, 3])
        print(f"{pshort:<12} SFT_U={sft_u:.6f}  SDFT_U={sdft_u:.6f}  diff={sdft_u-sft_u:+.6f}")


# =============================================================================
# Main
# =============================================================================

def analyze_alignment_vs_k(
    data_root: str,
    output_dir: Path,
    layer: int = 0,
    param: str = 'self_attn.q_proj.weight',
    k_values: List[int] = None,
    gpu: bool = True,
):
    """
    Analyze how alignment changes with different k values.

    This helps understand whether drift is concentrated in top singular
    vectors or spread across all directions.
    """
    if k_values is None:
        k_values = [5, 10, 20, 50, 100, 200, 500]

    paths = get_checkpoint_paths(data_root)
    pshort = PARAM_SHORT[param]

    print(f"\nAnalyzing alignment vs k for layer {layer}, {pshort}")

    # Load weights
    W_base_sft = load_weight(paths['sft'][0], layer, param)
    W_mbpp_sft = load_weight(paths['sft'][3], layer, param)
    W_mbpp_sdft = load_weight(paths['sdft'][3], layer, param)

    # Compute full SVD
    U_base, S_base, V_base = svd(W_base_sft, top_k=None, gpu=gpu)
    U_sft, S_sft, V_sft = svd(W_mbpp_sft, top_k=None, gpu=gpu)
    U_sdft, S_sdft, V_sdft = svd(W_mbpp_sdft, top_k=None, gpu=gpu)

    max_k = min(U_base.shape[1], max(k_values))
    k_values = [k for k in k_values if k <= max_k]

    results = {'k': [], 'sft_U': [], 'sdft_U': [], 'sft_V': [], 'sdft_V': []}

    for k in k_values:
        sft_u = alignment_score(U_base, U_sft, k)
        sdft_u = alignment_score(U_base, U_sdft, k)
        sft_v = alignment_score(V_base, V_sft, k)
        sdft_v = alignment_score(V_base, V_sdft, k)

        results['k'].append(k)
        results['sft_U'].append(sft_u)
        results['sdft_U'].append(sdft_u)
        results['sft_V'].append(sft_v)
        results['sdft_V'].append(sdft_v)

        drift_sft = (1 - sft_u) * 10000
        drift_sdft = (1 - sdft_u) * 10000
        print(f"  k={k:4d}: SFT_U drift={drift_sft:6.1f} bp, SDFT_U drift={drift_sdft:6.1f} bp, Δ={drift_sdft-drift_sft:+6.1f} bp")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, sv, title in [(axes[0], 'U', 'Left Singular Vectors (U)'),
                           (axes[1], 'V', 'Right Singular Vectors (V)')]:
        # Convert to drift
        sft_drift = [(1 - v) * 10000 for v in results[f'sft_{sv}']]
        sdft_drift = [(1 - v) * 10000 for v in results[f'sdft_{sv}']]

        ax.plot(results['k'], sft_drift, 'b-o', label='SFT', linewidth=2, markersize=6)
        ax.plot(results['k'], sdft_drift, 'r-s', label='SDFT', linewidth=2, markersize=6)

        ax.set_xlabel('k (number of top singular vectors)')
        ax.set_ylabel('Drift (basis points)')
        ax.set_title(title)
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Subspace Drift vs k: Layer {layer}, {pshort}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'alignment_vs_k_L{layer}_{pshort}.png', dpi=150)
    plt.close()

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Compare SFT vs SDFT subspace drift across all layers'
    )
    parser.add_argument('--data_root', type=str, default=DATA_ROOT)
    parser.add_argument('--output_dir', type=str,
                        default='eigenspectrum/outputs/sft_vs_sdft_subspaces')
    parser.add_argument('--layers', type=int, nargs='+', default=None,
                        help='Specific layers (default: all 36)')
    parser.add_argument('--top_k', type=int, default=TOP_K,
                        help='Number of top singular vectors to analyze (default: 50)')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--analyze_k', action='store_true',
                        help='Also analyze how alignment changes with k')
    parser.add_argument('--analyze_k_layer', type=int, default=0,
                        help='Layer to use for k analysis')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Main analysis
    run_analysis(
        data_root=args.data_root,
        output_dir=args.output_dir,
        layers=args.layers,
        top_k=args.top_k,
        gpu=not args.cpu,
    )

    # Optional: analyze alignment vs k
    if args.analyze_k:
        print("\n" + "="*70)
        print("Analyzing alignment vs k")
        print("="*70)
        for param in PARAM_NAMES[:3]:  # Just first 3 params for speed
            analyze_alignment_vs_k(
                data_root=args.data_root,
                output_dir=output_dir,
                layer=args.analyze_k_layer,
                param=param,
                gpu=not args.cpu,
            )


if __name__ == '__main__':
    main()
