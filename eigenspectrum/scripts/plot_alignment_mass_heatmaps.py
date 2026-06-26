#!/usr/bin/env python3
"""
Generate heatmaps for alignment mass metrics from CSV data.

For each matrix and comparison:
- X-axis: layer
- Y-axis: k value
- Color: diag_mass, off_diag_mass, or diag_ratio
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

METRICS = ['diag_mass', 'off_diag_mass', 'diag_ratio']


def load_csv(csv_dir: Path, comparison: str, matrix: str) -> pd.DataFrame:
    """Load CSV file for a given comparison and matrix."""
    csv_path = csv_dir / f'{comparison}_{matrix}_masses.csv'
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
    Plot heatmaps for a single matrix showing all metrics.

    Layout: 3 rows (diag_mass, off_diag_mass, diag_ratio) x 3 cols (SFT, SDFT, Delta)
    For U vectors only (can add V separately if needed).
    """
    df = load_csv(csv_dir, comparison, matrix)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    metrics_info = [
        ('U_diag_mass', 'U Diagonal Mass', 'viridis', 0, 1),
        ('U_off_diag_mass', 'U Off-Diagonal Mass', 'magma', None, None),
        ('U_diag_ratio', 'U Diagonal Ratio', 'viridis', 0, 1),
    ]

    for ri, (metric_suffix, metric_label, cmap_base, vmin_fixed, vmax_fixed) in enumerate(metrics_info):
        # Get data
        sft_col = f'sft_{metric_suffix}'
        sdft_col = f'sdft_{metric_suffix}'
        delta_col = f'delta_{metric_suffix}'

        sft_data, layers, k_values = pivot_to_heatmap(df, sft_col)
        sdft_data, _, _ = pivot_to_heatmap(df, sdft_col)
        delta_data, _, _ = pivot_to_heatmap(df, delta_col)

        # Determine color ranges
        if vmin_fixed is not None:
            vmin_sft = vmin_fixed
            vmax_sft = vmax_fixed
        else:
            vmax_sft = max(np.nanmax(sft_data), np.nanmax(sdft_data))
            vmin_sft = 0

        vmax_delta = np.nanmax(np.abs(delta_data))

        # Plot SFT
        ax = axes[ri, 0]
        im = ax.imshow(sft_data, aspect='auto', cmap=cmap_base, vmin=vmin_sft, vmax=vmax_sft, origin='lower')
        ax.set_ylabel(f'{metric_label}\nk')
        if ri == 0:
            ax.set_title('SFT')
        if ri == 2:
            ax.set_xlabel('Layer')
        ax.set_yticks(range(len(k_values)))
        ax.set_yticklabels([str(k) for k in k_values])
        xticks = list(range(0, len(layers), 5))
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(layers[i]) for i in xticks])
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Plot SDFT
        ax = axes[ri, 1]
        im = ax.imshow(sdft_data, aspect='auto', cmap=cmap_base, vmin=vmin_sft, vmax=vmax_sft, origin='lower')
        if ri == 0:
            ax.set_title('SDFT')
        if ri == 2:
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
        if ri == 2:
            ax.set_xlabel('Layer')
        ax.set_yticks(range(len(k_values)))
        ax.set_yticklabels([str(k) for k in k_values])
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(layers[i]) for i in xticks])
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f'{matrix} | {COMP_LABELS[comparison]} | Alignment Masses (U)', fontsize=14)
    plt.tight_layout()

    out_path = output_dir / f'{comparison}_{matrix}_masses_heatmap.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    return out_path


def plot_diag_mass_all_matrices(
    csv_dir: Path,
    output_dir: Path,
    comparison: str,
):
    """
    Plot diagonal mass for all 7 matrices in a single figure.
    Rows: matrices, Cols: SFT, SDFT, Delta
    Each matrix has its own color scale.
    """
    fig, axes = plt.subplots(7, 3, figsize=(16, 28))

    all_data = {}
    for matrix in MATRICES:
        df = load_csv(csv_dir, comparison, matrix)
        sft_data, layers, k_values = pivot_to_heatmap(df, 'sft_U_diag_mass')
        sdft_data, _, _ = pivot_to_heatmap(df, 'sdft_U_diag_mass')
        delta_data = sdft_data - sft_data
        # Per-matrix color range for delta
        vmax_delta = np.nanmax(np.abs(delta_data))
        all_data[matrix] = {'sft': sft_data, 'sdft': sdft_data, 'delta': delta_data,
                            'layers': layers, 'k_values': k_values, 'vmax_delta': vmax_delta}

    for mi, matrix in enumerate(MATRICES):
        data = all_data[matrix]
        layers = data['layers']
        k_values = data['k_values']
        vmax_delta = data['vmax_delta']

        for ci, (col_data, title, cmap, vmin, vmax) in enumerate([
            (data['sft'], 'SFT', 'viridis', 0, 1),
            (data['sdft'], 'SDFT', 'viridis', 0, 1),
            (data['delta'], 'Δ', 'RdBu_r', -vmax_delta, vmax_delta),
        ]):
            ax = axes[mi, ci]
            im = ax.imshow(col_data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')

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

    fig.suptitle(f'U Diagonal Mass | {COMP_LABELS[comparison]} | Per-Matrix Scale', fontsize=16, y=1.01)
    plt.tight_layout()

    out_path = output_dir / f'{comparison}_all_matrices_diag_mass.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    return out_path


def plot_off_diag_mass_all_matrices(
    csv_dir: Path,
    output_dir: Path,
    comparison: str,
):
    """
    Plot off-diagonal mass for all 7 matrices in a single figure.
    Each matrix has its own color scale.
    """
    fig, axes = plt.subplots(7, 3, figsize=(16, 28))

    all_data = {}
    for matrix in MATRICES:
        df = load_csv(csv_dir, comparison, matrix)
        sft_data, layers, k_values = pivot_to_heatmap(df, 'sft_U_off_diag_mass')
        sdft_data, _, _ = pivot_to_heatmap(df, 'sdft_U_off_diag_mass')
        delta_data = sdft_data - sft_data
        # Per-matrix color ranges
        vmax_abs = max(np.nanmax(sft_data), np.nanmax(sdft_data))
        vmax_delta = np.nanmax(np.abs(delta_data))
        all_data[matrix] = {'sft': sft_data, 'sdft': sdft_data, 'delta': delta_data,
                            'layers': layers, 'k_values': k_values,
                            'vmax_abs': vmax_abs, 'vmax_delta': vmax_delta}

    for mi, matrix in enumerate(MATRICES):
        data = all_data[matrix]
        layers = data['layers']
        k_values = data['k_values']
        vmax_abs = data['vmax_abs']
        vmax_delta = data['vmax_delta']

        for ci, (col_data, title, cmap, vmin, vmax) in enumerate([
            (data['sft'], 'SFT', 'magma', 0, vmax_abs),
            (data['sdft'], 'SDFT', 'magma', 0, vmax_abs),
            (data['delta'], 'Δ', 'RdBu_r', -vmax_delta, vmax_delta),
        ]):
            ax = axes[mi, ci]
            im = ax.imshow(col_data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')

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

    fig.suptitle(f'U Off-Diagonal Mass | {COMP_LABELS[comparison]} | Per-Matrix Scale', fontsize=16, y=1.01)
    plt.tight_layout()

    out_path = output_dir / f'{comparison}_all_matrices_off_diag_mass.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    return out_path


def plot_matrix_across_comparisons(
    csv_dir: Path,
    output_dir: Path,
    matrix: str,
    metric: str = 'U_diag_mass',
):
    """
    Plot a single matrix across all 3 comparisons.
    Rows: comparisons, Cols: SFT, SDFT, Delta
    Each comparison row has its own color scale.
    """
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    all_data = {}
    for comparison in COMPARISONS:
        df = load_csv(csv_dir, comparison, matrix)
        sft_data, layers, k_values = pivot_to_heatmap(df, f'sft_{metric}')
        sdft_data, _, _ = pivot_to_heatmap(df, f'sdft_{metric}')
        delta_data = sdft_data - sft_data
        # Per-comparison color ranges
        is_ratio = 'ratio' in metric
        if is_ratio:
            vmin_abs, vmax_abs = 0, 1
        else:
            vmax_abs = max(np.nanmax(sft_data), np.nanmax(sdft_data))
            vmin_abs = 0
        vmax_delta = np.nanmax(np.abs(delta_data))
        all_data[comparison] = {'sft': sft_data, 'sdft': sdft_data, 'delta': delta_data,
                                'layers': layers, 'k_values': k_values,
                                'vmin_abs': vmin_abs, 'vmax_abs': vmax_abs, 'vmax_delta': vmax_delta}

    cmap_base = 'viridis' if 'ratio' in metric or 'diag_mass' in metric else 'magma'

    for ri, comparison in enumerate(COMPARISONS):
        data = all_data[comparison]
        layers = data['layers']
        k_values = data['k_values']
        vmin_abs = data['vmin_abs']
        vmax_abs = data['vmax_abs']
        vmax_delta = data['vmax_delta']

        for ci, (col_data, title, cmap, vmin, vmax) in enumerate([
            (data['sft'], 'SFT', cmap_base, vmin_abs, vmax_abs),
            (data['sdft'], 'SDFT', cmap_base, vmin_abs, vmax_abs),
            (data['delta'], 'Δ', 'RdBu_r', -vmax_delta, vmax_delta),
        ]):
            ax = axes[ri, ci]
            im = ax.imshow(col_data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')

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

    metric_label = metric.replace('_', ' ').title()
    fig.suptitle(f'{matrix} | {metric_label} Across Training | Per-Comparison Scale', fontsize=14)
    plt.tight_layout()

    out_path = output_dir / f'{matrix}_{metric}_all_comparisons.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    return out_path


def main():
    parser = argparse.ArgumentParser(description='Plot alignment mass heatmaps')
    parser.add_argument('--csv_dir', type=str,
                        default='eigenspectrum/outputs/alignment_masses/csv',
                        help='Directory containing CSV files')
    parser.add_argument('--output_dir', type=str,
                        default='eigenspectrum/outputs/alignment_masses/heatmaps',
                        help='Directory to save heatmap images')

    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading CSVs from: {csv_dir}")
    print(f"Saving heatmaps to: {output_dir}")

    # 1. Individual matrix heatmaps (all metrics)
    print("\n1. Generating individual matrix heatmaps (all metrics)...")
    for comparison in COMPARISONS:
        for matrix in MATRICES:
            out_path = plot_single_matrix_all_metrics(csv_dir, output_dir, comparison, matrix)
            print(f"   Saved: {out_path.name}")

    # 2. All matrices combined - diagonal mass
    print("\n2. Generating combined diagonal mass heatmaps...")
    for comparison in COMPARISONS:
        out_path = plot_diag_mass_all_matrices(csv_dir, output_dir, comparison)
        print(f"   Saved: {out_path.name}")

    # 3. All matrices combined - off-diagonal mass
    print("\n3. Generating combined off-diagonal mass heatmaps...")
    for comparison in COMPARISONS:
        out_path = plot_off_diag_mass_all_matrices(csv_dir, output_dir, comparison)
        print(f"   Saved: {out_path.name}")

    # 4. Per-matrix across all comparisons
    print("\n4. Generating per-matrix comparison grids...")
    for matrix in MATRICES:
        for metric in ['U_diag_mass', 'U_off_diag_mass', 'U_diag_ratio']:
            out_path = plot_matrix_across_comparisons(csv_dir, output_dir, matrix, metric)
            print(f"   Saved: {out_path.name}")

    print(f"\nDone! All heatmaps saved to {output_dir}")


if __name__ == '__main__':
    main()
