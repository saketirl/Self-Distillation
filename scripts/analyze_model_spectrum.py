"""
Analyze singular value spectrum for all layers of a model.

Computes rank and singular values for all 2D weight matrices,
saves results as CSVs and plots as PNGs.

Usage:
    python scripts/analyze_model_spectrum.py --model_path models/Qwen/Qwen2.5-3B-Instruct --output_dir outputs/spectrum_analysis
"""

import argparse
import os
from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM
from tqdm import tqdm


def compute_numerical_rank(S: torch.Tensor, tol: float = 1e-5) -> int:
    """Compute numerical rank based on singular value threshold."""
    if S.numel() == 0 or S.max() == 0:
        return 0
    threshold = tol * S[0].item()
    return int((S > threshold).sum().item())


def analyze_layer(weight: torch.Tensor, name: str) -> dict:
    """Analyze a single weight matrix."""
    if weight.dim() != 2:
        return None

    # Move to CPU and convert to float32 for SVD stability
    W = weight.detach().float().cpu()
    m, n = W.shape

    # Compute SVD
    try:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    except Exception as e:
        print(f"  SVD failed for {name}: {e}")
        return None

    S = S.numpy()

    # Compute ranks at different tolerances
    rank_1e5 = compute_numerical_rank(torch.from_numpy(S), tol=1e-5)
    rank_1e3 = compute_numerical_rank(torch.from_numpy(S), tol=1e-3)
    rank_1e2 = compute_numerical_rank(torch.from_numpy(S), tol=1e-2)

    # Compute other statistics
    full_rank = min(m, n)
    condition_number = S[0] / S[-1] if S[-1] > 0 else float('inf')
    effective_rank = np.sum(S) ** 2 / np.sum(S ** 2) if np.sum(S ** 2) > 0 else 0

    return {
        'name': name,
        'shape_m': m,
        'shape_n': n,
        'full_rank': full_rank,
        'rank_tol_1e5': rank_1e5,
        'rank_tol_1e3': rank_1e3,
        'rank_tol_1e2': rank_1e2,
        'effective_rank': effective_rank,
        'condition_number': condition_number,
        'max_sv': S[0],
        'min_sv': S[-1],
        'singular_values': S,
    }


def plot_singular_values(S: np.ndarray, name: str, output_path: str):
    """Plot singular value spectrum."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Linear scale
    axes[0].plot(S, 'b-', linewidth=0.8)
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Singular Value')
    axes[0].set_title(f'{name}\n(Linear Scale)')
    axes[0].grid(True, alpha=0.3)

    # Log scale
    S_pos = S[S > 0]
    if len(S_pos) > 0:
        axes[1].semilogy(range(len(S_pos)), S_pos, 'b-', linewidth=0.8)
        axes[1].set_xlabel('Index')
        axes[1].set_ylabel('Singular Value (log)')
        axes[1].set_title(f'{name}\n(Log Scale)')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary(df: pd.DataFrame, output_dir: str):
    """Plot summary visualizations."""
    # Filter to layers with meaningful data
    df_valid = df[df['full_rank'] > 1].copy()

    if len(df_valid) == 0:
        return

    # Plot 1: Rank ratio across layers
    fig, ax = plt.subplots(figsize=(14, 5))
    x = range(len(df_valid))
    ax.bar(x, df_valid['rank_tol_1e5'] / df_valid['full_rank'], alpha=0.7, label='tol=1e-5')
    ax.bar(x, df_valid['rank_tol_1e3'] / df_valid['full_rank'], alpha=0.5, label='tol=1e-3')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Rank / Full Rank')
    ax.set_title('Numerical Rank Ratio Across Layers')
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rank_ratio_summary.png'), dpi=150)
    plt.close()

    # Plot 2: Effective rank across layers
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x, df_valid['effective_rank'], alpha=0.7, color='green')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Effective Rank')
    ax.set_title('Effective Rank Across Layers')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'effective_rank_summary.png'), dpi=150)
    plt.close()

    # Plot 3: Condition number (log scale)
    fig, ax = plt.subplots(figsize=(14, 5))
    cond = df_valid['condition_number'].replace([np.inf], np.nan).dropna()
    if len(cond) > 0:
        ax.semilogy(range(len(cond)), cond.values, 'ro-', markersize=3)
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Condition Number (log)')
        ax.set_title('Condition Number Across Layers')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'condition_number_summary.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze model singular value spectrum")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--plot_all", action="store_true", help="Plot SVs for every layer (can be many files)")
    parser.add_argument("--layer_filter", type=str, default=None, help="Filter layers by name (e.g., 'attn', 'mlp')")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sv_dir = output_dir / "singular_values"
    sv_dir.mkdir(exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"Loading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="cpu",  # Load on CPU for analysis
    )

    print(f"Analyzing layers...")
    results = []

    for name, param in tqdm(list(model.named_parameters())):
        if param.dim() != 2:
            continue

        if args.layer_filter and args.layer_filter not in name:
            continue

        result = analyze_layer(param, name)
        if result is None:
            continue

        # Save singular values to CSV
        sv_df = pd.DataFrame({
            'index': range(len(result['singular_values'])),
            'singular_value': result['singular_values']
        })
        safe_name = name.replace('.', '_').replace('/', '_')
        sv_df.to_csv(sv_dir / f"{safe_name}.csv", index=False)

        # Plot if requested
        if args.plot_all:
            plot_singular_values(
                result['singular_values'],
                name,
                str(plots_dir / f"{safe_name}.png")
            )

        # Store summary (without full SV array)
        summary = {k: v for k, v in result.items() if k != 'singular_values'}
        results.append(summary)

    # Save summary CSV
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "layer_summary.csv", index=False)

    # Plot summary visualizations
    print("Generating summary plots...")
    plot_summary(df, str(output_dir))

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total 2D layers analyzed: {len(df)}")
    print(f"\nRank statistics (tol=1e-5):")
    print(f"  Mean rank ratio: {(df['rank_tol_1e5'] / df['full_rank']).mean():.3f}")
    print(f"  Min rank ratio:  {(df['rank_tol_1e5'] / df['full_rank']).min():.3f}")
    print(f"  Max rank ratio:  {(df['rank_tol_1e5'] / df['full_rank']).max():.3f}")
    print(f"\nEffective rank:")
    print(f"  Mean: {df['effective_rank'].mean():.1f}")
    print(f"  Min:  {df['effective_rank'].min():.1f}")
    print(f"  Max:  {df['effective_rank'].max():.1f}")
    print(f"\nCondition number:")
    cond_finite = df['condition_number'].replace([np.inf], np.nan).dropna()
    if len(cond_finite) > 0:
        print(f"  Mean: {cond_finite.mean():.2e}")
        print(f"  Max:  {cond_finite.max():.2e}")
    print(f"\nOutput saved to: {output_dir}")
    print(f"  - layer_summary.csv")
    print(f"  - singular_values/*.csv ({len(df)} files)")
    if args.plot_all:
        print(f"  - plots/*.png ({len(df)} files)")
    print(f"  - rank_ratio_summary.png")
    print(f"  - effective_rank_summary.png")
    print(f"  - condition_number_summary.png")


if __name__ == "__main__":
    main()
