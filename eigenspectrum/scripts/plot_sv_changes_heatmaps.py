#!/usr/bin/env python3
"""
Generate heatmaps for singular value change data from CSV files.

For each matrix and comparison:
- X-axis: layer
- Y-axis: k value
- Color: mean_abs_change, mean_rel_change, mean_ratio, mass_change_pct
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================

COMPARISONS = ['base_to_tooluse', 'base_to_gsm8k', 'base_to_mbpp']
MATRICES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
COMP_LABELS = {
    'base_to_tooluse': 'Base → ToolUse',
    'base_to_gsm8k': 'Base → GSM8K',
    'base_to_mbpp': 'Base → MBPP',
}

# Key metrics to visualize (aggregate over k)
METRICS = ['mean_abs_change', 'mean_ratio', 'mass_change_pct', 'percentile_50_ratio']
METRIC_LABELS = {
    'mean_abs_change': 'Mean Δσ (absolute)',
    'mean_rel_change': 'Mean Δσ/σ (relative)',
    'mean_ratio': 'Mean σ_task/σ_base',
    'mass_change_pct': 'Mass Change (%)',
    'percentile_50_abs_change': 'Median Δσ',
    'percentile_50_ratio': 'Median Ratio',
}

# Singular value indices tracked (1-indexed)
SV_INDICES = [1, 5, 10, 20, 50, 100, 250, 500]


def load_csv(csv_dir: Path, comparison: str, matrix: str) -> pd.DataFrame:
    """Load CSV file for a given comparison and matrix."""
    csv_path = csv_dir / f'{comparison}_{matrix}_sv_changes.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def pivot_to_heatmap(df: pd.DataFrame, value_col: str) -> tuple:
    """Pivot dataframe to heatmap format."""
    layers = sorted(df['layer'].unique())
    k_values = sorted(df['k'].unique())
    pivot = df.pivot(index='k', columns='layer', values=value_col)
    pivot = pivot.sort_index(ascending=True)
    return pivot.values, layers, k_values


def plot_single_matrix_all_metrics(
    csv_dir: Path,
    output_dir: Path,
    comparison: str,
    matrix: str,
):
    """
    Plot heatmaps for a single matrix showing key metrics.
    Layout: 4 rows (metrics) x 3 cols (SFT, SDFT, Delta)
    """
    df = load_csv(csv_dir, comparison, matrix)

    fig, axes = plt.subplots(4, 3, figsize=(16, 16))

    for ri, metric in enumerate(METRICS):
        sft_col = f'sft_{metric}'
        sdft_col = f'sdft_{metric}'
        delta_col = f'delta_{metric}'

        sft_data, layers, k_values = pivot_to_heatmap(df, sft_col)
        sdft_data, _, _ = pivot_to_heatmap(df, sdft_col)
        delta_data, _, _ = pivot_to_heatmap(df, delta_col)

        # Determine color ranges (per-metric)
        if metric == 'mean_ratio':
            # Ratio: centered around 1
            vmax_abs = max(np.nanmax(np.abs(sft_data - 1)), np.nanmax(np.abs(sdft_data - 1)))
            vmin_sft, vmax_sft = 1 - vmax_abs, 1 + vmax_abs
            cmap_base = 'RdBu_r'
        else:
            # Other metrics: symmetric around 0
            vmax_sft = max(np.nanmax(np.abs(sft_data)), np.nanmax(np.abs(sdft_data)))
            vmin_sft = -vmax_sft
            cmap_base = 'RdBu_r'

        vmax_delta = np.nanmax(np.abs(delta_data))

        xticks = list(range(0, len(layers), 5))

        # Plot SFT
        ax = axes[ri, 0]
        im = ax.imshow(sft_data, aspect='auto', cmap=cmap_base, vmin=vmin_sft, vmax=vmax_sft, origin='lower')
        ax.set_ylabel(f'{METRIC_LABELS[metric]}\nk')
        if ri == 0:
            ax.set_title('SFT')
        if ri == len(METRICS) - 1:
            ax.set_xlabel('Layer')
        ax.set_yticks(range(len(k_values)))
        ax.set_yticklabels([str(k) for k in k_values])
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(layers[i]) for i in xticks])
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Plot SDFT
        ax = axes[ri, 1]
        im = ax.imshow(sdft_data, aspect='auto', cmap=cmap_base, vmin=vmin_sft, vmax=vmax_sft, origin='lower')
        if ri == 0:
            ax.set_title('SDFT')
        if ri == len(METRICS) - 1:
            ax.set_xlabel('Layer')
        ax.set_yticks(range(len(k_values)))
        ax.set_yticklabels([str(k) for k in k_values])
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(layers[i]) for i in xticks])
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Plot Delta
        ax = axes[ri, 2]
        im = ax.imshow(delta_data, aspect='auto', cmap='RdBu_r', vmin=-vmax_delta, vmax=vmax_delta, origin='lower')
        if ri == 0:
            ax.set_title('Δ (SDFT - SFT)')
        if ri == len(METRICS) - 1:
            ax.set_xlabel('Layer')
        ax.set_yticks(range(len(k_values)))
        ax.set_yticklabels([str(k) for k in k_values])
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(layers[i]) for i in xticks])
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f'{matrix} | {COMP_LABELS[comparison]} | Singular Value Changes', fontsize=14)
    plt.tight_layout()

    out_path = output_dir / f'{comparison}_{matrix}_sv_changes_heatmap.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    return out_path


def plot_metric_all_matrices(
    csv_dir: Path,
    output_dir: Path,
    comparison: str,
    metric: str,
):
    """
    Plot one metric for all 7 matrices in a single figure.
    Rows: matrices, Cols: SFT, SDFT, Delta
    Each matrix has its own color scale.
    """
    fig, axes = plt.subplots(7, 3, figsize=(16, 28))

    all_data = {}
    for matrix in MATRICES:
        df = load_csv(csv_dir, comparison, matrix)
        sft_data, layers, k_values = pivot_to_heatmap(df, f'sft_{metric}')
        sdft_data, _, _ = pivot_to_heatmap(df, f'sdft_{metric}')
        delta_data = sdft_data - sft_data

        # Per-matrix color ranges
        if metric == 'mean_ratio':
            vmax_abs = max(np.nanmax(np.abs(sft_data - 1)), np.nanmax(np.abs(sdft_data - 1)))
            vmin_sft, vmax_sft = 1 - vmax_abs, 1 + vmax_abs
        else:
            vmax_sft = max(np.nanmax(np.abs(sft_data)), np.nanmax(np.abs(sdft_data)))
            vmin_sft = -vmax_sft

        vmax_delta = np.nanmax(np.abs(delta_data))

        all_data[matrix] = {
            'sft': sft_data, 'sdft': sdft_data, 'delta': delta_data,
            'layers': layers, 'k_values': k_values,
            'vmin_sft': vmin_sft, 'vmax_sft': vmax_sft, 'vmax_delta': vmax_delta
        }

    for mi, matrix in enumerate(MATRICES):
        data = all_data[matrix]
        layers = data['layers']
        k_values = data['k_values']
        vmin_sft = data['vmin_sft']
        vmax_sft = data['vmax_sft']
        vmax_delta = data['vmax_delta']

        for ci, (col_data, title, vmin, vmax) in enumerate([
            (data['sft'], 'SFT', vmin_sft, vmax_sft),
            (data['sdft'], 'SDFT', vmin_sft, vmax_sft),
            (data['delta'], 'Δ', -vmax_delta, vmax_delta),
        ]):
            ax = axes[mi, ci]
            im = ax.imshow(col_data, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')

            xticks = list(range(0, len(layers), 10))
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(layers[i]) for i in xticks], fontsize=8)
            ax.set_yticks(range(len(k_values)))
            ax.set_yticklabels([str(k) for k in k_values], fontsize=8)

            if mi == 0:
                ax.set_title(title, fontsize=12)
            if ci == 0:
                ax.set_ylabel(f'{matrix}\nk', fontsize=10)
            if mi == len(MATRICES) - 1:
                ax.set_xlabel('Layer', fontsize=10)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f'{METRIC_LABELS[metric]} | {COMP_LABELS[comparison]} | Per-Matrix Scale', fontsize=16, y=1.01)
    plt.tight_layout()

    out_path = output_dir / f'{comparison}_all_matrices_{metric}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    return out_path


def plot_matrix_across_comparisons(
    csv_dir: Path,
    output_dir: Path,
    matrix: str,
    metric: str,
):
    """
    Plot a single matrix across all 3 comparisons for one metric.
    Rows: comparisons, Cols: SFT, SDFT, Delta
    Each comparison has its own color scale.
    """
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    all_data = {}
    for comparison in COMPARISONS:
        df = load_csv(csv_dir, comparison, matrix)
        sft_data, layers, k_values = pivot_to_heatmap(df, f'sft_{metric}')
        sdft_data, _, _ = pivot_to_heatmap(df, f'sdft_{metric}')
        delta_data = sdft_data - sft_data

        # Per-comparison color ranges
        if metric == 'mean_ratio':
            vmax_abs = max(np.nanmax(np.abs(sft_data - 1)), np.nanmax(np.abs(sdft_data - 1)))
            vmin_sft, vmax_sft = 1 - vmax_abs, 1 + vmax_abs
        else:
            vmax_sft = max(np.nanmax(np.abs(sft_data)), np.nanmax(np.abs(sdft_data)))
            vmin_sft = -vmax_sft

        vmax_delta = np.nanmax(np.abs(delta_data))

        all_data[comparison] = {
            'sft': sft_data, 'sdft': sdft_data, 'delta': delta_data,
            'layers': layers, 'k_values': k_values,
            'vmin_sft': vmin_sft, 'vmax_sft': vmax_sft, 'vmax_delta': vmax_delta
        }

    for ri, comparison in enumerate(COMPARISONS):
        data = all_data[comparison]
        layers = data['layers']
        k_values = data['k_values']
        vmin_sft = data['vmin_sft']
        vmax_sft = data['vmax_sft']
        vmax_delta = data['vmax_delta']

        for ci, (col_data, title, vmin, vmax) in enumerate([
            (data['sft'], 'SFT', vmin_sft, vmax_sft),
            (data['sdft'], 'SDFT', vmin_sft, vmax_sft),
            (data['delta'], 'Δ', -vmax_delta, vmax_delta),
        ]):
            ax = axes[ri, ci]
            im = ax.imshow(col_data, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')

            xticks = list(range(0, len(layers), 5))
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(layers[i]) for i in xticks])
            ax.set_yticks(range(len(k_values)))
            ax.set_yticklabels([str(k) for k in k_values])

            if ri == 0:
                ax.set_title(title, fontsize=12)
            if ci == 0:
                ax.set_ylabel(f'{COMP_LABELS[comparison]}\nk', fontsize=10)
            if ri == len(COMPARISONS) - 1:
                ax.set_xlabel('Layer', fontsize=10)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f'{matrix} | {METRIC_LABELS[metric]} Across Training | Per-Comparison Scale', fontsize=14)
    plt.tight_layout()

    out_path = output_dir / f'{matrix}_{metric}_all_comparisons.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    return out_path


def load_per_index_csv(csv_dir: Path, comparison: str, matrix: str) -> pd.DataFrame:
    """Load per-index CSV file for a given comparison and matrix."""
    csv_path = csv_dir / f'{comparison}_{matrix}_sv_per_index.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def plot_per_index_changes(
    csv_dir: Path,
    output_dir: Path,
    comparison: str,
    matrix: str,
):
    """
    Plot per-index singular value changes across layers.
    X-axis: layer, Y-axis: change value, Lines: different σ indices
    """
    df = load_per_index_csv(csv_dir, comparison, matrix)
    layers = df['layer'].values

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Colors for different indices
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(SV_INDICES)))

    # Plot 1: Absolute change (SFT)
    ax = axes[0, 0]
    for i, idx in enumerate(SV_INDICES):
        col = f'sft_abs_change_{idx}'
        if col in df.columns:
            ax.plot(layers, df[col], '-', color=colors[i], label=f'σ_{idx}', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Δσ')
    ax.set_title('SFT: Absolute Change')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Plot 2: Absolute change (SDFT)
    ax = axes[0, 1]
    for i, idx in enumerate(SV_INDICES):
        col = f'sdft_abs_change_{idx}'
        if col in df.columns:
            ax.plot(layers, df[col], '-', color=colors[i], label=f'σ_{idx}', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Δσ')
    ax.set_title('SDFT: Absolute Change')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Plot 3: Absolute change (Delta)
    ax = axes[0, 2]
    for i, idx in enumerate(SV_INDICES):
        col = f'delta_abs_change_{idx}'
        if col in df.columns:
            ax.plot(layers, df[col], '-', color=colors[i], label=f'σ_{idx}', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Δ(SDFT-SFT)')
    ax.set_title('Delta: Absolute Change')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Plot 4: Ratio (SFT)
    ax = axes[1, 0]
    for i, idx in enumerate(SV_INDICES):
        col = f'sft_ratio_{idx}'
        if col in df.columns:
            ax.plot(layers, df[col], '-', color=colors[i], label=f'σ_{idx}', linewidth=1.5)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('σ_task / σ_base')
    ax.set_title('SFT: Ratio')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Plot 5: Ratio (SDFT)
    ax = axes[1, 1]
    for i, idx in enumerate(SV_INDICES):
        col = f'sdft_ratio_{idx}'
        if col in df.columns:
            ax.plot(layers, df[col], '-', color=colors[i], label=f'σ_{idx}', linewidth=1.5)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('σ_task / σ_base')
    ax.set_title('SDFT: Ratio')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Plot 6: Ratio (Delta)
    ax = axes[1, 2]
    for i, idx in enumerate(SV_INDICES):
        col = f'delta_ratio_{idx}'
        if col in df.columns:
            ax.plot(layers, df[col], '-', color=colors[i], label=f'σ_{idx}', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Δ Ratio')
    ax.set_title('Delta: Ratio')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'{matrix} | {COMP_LABELS[comparison]} | Per-Index SV Changes', fontsize=14)
    plt.tight_layout()

    out_path = output_dir / f'{comparison}_{matrix}_per_index_changes.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    return out_path


def plot_per_index_all_matrices(
    csv_dir: Path,
    output_dir: Path,
    comparison: str,
    sv_index: int,
):
    """
    Plot changes for a specific singular value index across all matrices.
    X-axis: layer, Y-axis: change value, Lines: different matrices
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(MATRICES)))

    for mi, matrix in enumerate(MATRICES):
        try:
            df = load_per_index_csv(csv_dir, comparison, matrix)
            layers = df['layer'].values

            # SFT
            col = f'sft_ratio_{sv_index}'
            if col in df.columns:
                axes[0].plot(layers, df[col], '-', color=colors[mi], label=matrix, linewidth=1.5)

            # SDFT
            col = f'sdft_ratio_{sv_index}'
            if col in df.columns:
                axes[1].plot(layers, df[col], '-', color=colors[mi], label=matrix, linewidth=1.5)

            # Delta
            col = f'delta_ratio_{sv_index}'
            if col in df.columns:
                axes[2].plot(layers, df[col], '-', color=colors[mi], label=matrix, linewidth=1.5)
        except FileNotFoundError:
            continue

    for ax, title in [(axes[0], 'SFT'), (axes[1], 'SDFT'), (axes[2], 'Delta')]:
        if title != 'Delta':
            ax.axhline(y=1, color='black', linestyle='--', linewidth=0.5)
        else:
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Layer')
        ax.set_ylabel('σ_task / σ_base' if title != 'Delta' else 'Δ Ratio')
        ax.set_title(f'{title}: σ_{sv_index} Ratio')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'σ_{sv_index} Changes | {COMP_LABELS[comparison]} | All Matrices', fontsize=14)
    plt.tight_layout()

    out_path = output_dir / f'{comparison}_all_matrices_sv{sv_index}_ratio.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    return out_path


def main():
    parser = argparse.ArgumentParser(description='Plot singular value change heatmaps')
    parser.add_argument('--csv_dir', type=str,
                        default='eigenspectrum/outputs/sv_changes/csv',
                        help='Directory containing CSV files')
    parser.add_argument('--output_dir', type=str,
                        default='eigenspectrum/outputs/sv_changes/heatmaps',
                        help='Directory to save heatmap images')

    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading CSVs from: {csv_dir}")
    print(f"Saving heatmaps to: {output_dir}")

    # 1. Individual matrix heatmaps (aggregate metrics with k-sweep)
    print("\n1. Generating individual matrix heatmaps (aggregate metrics)...")
    for comparison in COMPARISONS:
        for matrix in MATRICES:
            try:
                out_path = plot_single_matrix_all_metrics(csv_dir, output_dir, comparison, matrix)
                print(f"   Saved: {out_path.name}")
            except FileNotFoundError as e:
                print(f"   Skipped: {e}")

    # 2. All matrices combined per metric
    print("\n2. Generating combined matrix heatmaps per metric...")
    for comparison in COMPARISONS:
        for metric in ['mean_abs_change', 'mean_ratio', 'mass_change_pct']:
            try:
                out_path = plot_metric_all_matrices(csv_dir, output_dir, comparison, metric)
                print(f"   Saved: {out_path.name}")
            except FileNotFoundError as e:
                print(f"   Skipped: {e}")

    # 3. Per-matrix across all comparisons
    print("\n3. Generating per-matrix comparison grids...")
    for matrix in MATRICES:
        for metric in ['mean_abs_change', 'mean_ratio']:
            try:
                out_path = plot_matrix_across_comparisons(csv_dir, output_dir, matrix, metric)
                print(f"   Saved: {out_path.name}")
            except FileNotFoundError as e:
                print(f"   Skipped: {e}")

    # 4. Per-index plots (line plots across layers)
    print("\n4. Generating per-index change plots...")
    for comparison in COMPARISONS:
        for matrix in MATRICES:
            try:
                out_path = plot_per_index_changes(csv_dir, output_dir, comparison, matrix)
                print(f"   Saved: {out_path.name}")
            except FileNotFoundError as e:
                print(f"   Skipped: {e}")

    # 5. Per-index across all matrices
    print("\n5. Generating per-index all-matrices plots...")
    for comparison in COMPARISONS:
        for sv_idx in [1, 5, 10, 50]:  # Key indices
            try:
                out_path = plot_per_index_all_matrices(csv_dir, output_dir, comparison, sv_idx)
                print(f"   Saved: {out_path.name}")
            except FileNotFoundError as e:
                print(f"   Skipped: {e}")

    print(f"\nDone! All heatmaps saved to {output_dir}")


if __name__ == '__main__':
    main()
