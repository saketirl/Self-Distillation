#!/usr/bin/env python3
"""
Generate heatmaps per individual matrix from CSV data.

For each matrix (q_proj, k_proj, etc.) and each comparison (base_to_tooluse, etc.):
- X-axis: layer
- Y-axis: k value
- Color: drift (basis points)

Creates separate plots for SFT, SDFT, and delta.
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


def load_csv(csv_dir: Path, comparison: str, matrix: str) -> pd.DataFrame:
    """Load CSV file for a given comparison and matrix."""
    csv_path = csv_dir / f'{comparison}_{matrix}.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def pivot_to_heatmap(df: pd.DataFrame, value_col: str) -> tuple:
    """
    Pivot dataframe to heatmap format.

    Returns:
        (data, layers, k_values) where data[k_idx, layer_idx] = value
    """
    layers = sorted(df['layer'].unique())
    k_values = sorted(df['k'].unique())

    # Create pivot table
    pivot = df.pivot(index='k', columns='layer', values=value_col)
    pivot = pivot.sort_index(ascending=True)  # k ascending

    return pivot.values, layers, k_values


def plot_single_matrix_heatmaps(
    csv_dir: Path,
    output_dir: Path,
    comparison: str,
    matrix: str,
):
    """
    Plot heatmaps for a single matrix: SFT, SDFT, and Delta.
    """
    df = load_csv(csv_dir, comparison, matrix)

    # Convert alignment to drift in basis points
    df['sft_U_drift'] = (1 - df['sft_U']) * 10000
    df['sft_V_drift'] = (1 - df['sft_V']) * 10000
    df['sdft_U_drift'] = (1 - df['sdft_U']) * 10000
    df['sdft_V_drift'] = (1 - df['sdft_V']) * 10000

    # Get heatmap data for U vectors
    sft_U, layers, k_values = pivot_to_heatmap(df, 'sft_U_drift')
    sdft_U, _, _ = pivot_to_heatmap(df, 'sdft_U_drift')
    delta_U = sdft_U - sft_U  # positive = SDFT drifts more

    # Get heatmap data for V vectors
    sft_V, _, _ = pivot_to_heatmap(df, 'sft_V_drift')
    sdft_V, _, _ = pivot_to_heatmap(df, 'sdft_V_drift')
    delta_V = sdft_V - sft_V

    # Create figure with 2 rows (U, V) x 3 cols (SFT, SDFT, Delta)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Common colorbar range for SFT/SDFT
    vmax_drift = max(np.nanmax(sft_U), np.nanmax(sdft_U), np.nanmax(sft_V), np.nanmax(sdft_V))
    vmin_drift = 0

    # Delta range (symmetric around 0)
    vmax_delta = max(np.nanmax(np.abs(delta_U)), np.nanmax(np.abs(delta_V)))

    # Plot U row
    for ax, data, title, cmap, vmin, vmax in [
        (axes[0, 0], sft_U, 'SFT - U Drift', 'Blues', vmin_drift, vmax_drift),
        (axes[0, 1], sdft_U, 'SDFT - U Drift', 'Reds', vmin_drift, vmax_drift),
        (axes[0, 2], delta_U, 'Δ (SDFT-SFT) - U', 'RdBu_r', -vmax_delta, vmax_delta),
    ]:
        im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')

        # X-axis: layers
        xticks = list(range(0, len(layers), 5))
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(layers[i]) for i in xticks])
        ax.set_xlabel('Layer')

        # Y-axis: k values
        ax.set_yticks(range(len(k_values)))
        ax.set_yticklabels([str(k) for k in k_values])
        ax.set_ylabel('k')

        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Drift (bp)')

    # Plot V row
    for ax, data, title, cmap, vmin, vmax in [
        (axes[1, 0], sft_V, 'SFT - V Drift', 'Blues', vmin_drift, vmax_drift),
        (axes[1, 1], sdft_V, 'SDFT - V Drift', 'Reds', vmin_drift, vmax_drift),
        (axes[1, 2], delta_V, 'Δ (SDFT-SFT) - V', 'RdBu_r', -vmax_delta, vmax_delta),
    ]:
        im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')

        xticks = list(range(0, len(layers), 5))
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(layers[i]) for i in xticks])
        ax.set_xlabel('Layer')

        ax.set_yticks(range(len(k_values)))
        ax.set_yticklabels([str(k) for k in k_values])
        ax.set_ylabel('k')

        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Drift (bp)')

    fig.suptitle(f'{matrix} | {COMP_LABELS[comparison]}', fontsize=16)
    plt.tight_layout()

    # Save
    out_path = output_dir / f'{comparison}_{matrix}_heatmap.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    return out_path


def plot_all_matrices_single_comparison(
    csv_dir: Path,
    output_dir: Path,
    comparison: str,
):
    """
    Plot all 7 matrices in a single large figure for one comparison.
    Shows U drift for SFT, SDFT, and Delta side by side.
    """
    fig, axes = plt.subplots(7, 3, figsize=(18, 28))

    all_data = {}

    # Load all data first to get common color scale
    for mi, matrix in enumerate(MATRICES):
        df = load_csv(csv_dir, comparison, matrix)
        df['sft_U_drift'] = (1 - df['sft_U']) * 10000
        df['sdft_U_drift'] = (1 - df['sdft_U']) * 10000

        sft_U, layers, k_values = pivot_to_heatmap(df, 'sft_U_drift')
        sdft_U, _, _ = pivot_to_heatmap(df, 'sdft_U_drift')
        delta_U = sdft_U - sft_U

        all_data[matrix] = {
            'sft': sft_U,
            'sdft': sdft_U,
            'delta': delta_U,
            'layers': layers,
            'k_values': k_values,
        }

    # Get global color ranges
    vmax_drift = max(
        max(np.nanmax(d['sft']), np.nanmax(d['sdft']))
        for d in all_data.values()
    )
    vmax_delta = max(np.nanmax(np.abs(d['delta'])) for d in all_data.values())

    # Plot
    for mi, matrix in enumerate(MATRICES):
        data = all_data[matrix]
        layers = data['layers']
        k_values = data['k_values']

        for ci, (col_data, title, cmap, vmin, vmax) in enumerate([
            (data['sft'], 'SFT', 'Blues', 0, vmax_drift),
            (data['sdft'], 'SDFT', 'Reds', 0, vmax_drift),
            (data['delta'], 'Δ', 'RdBu_r', -vmax_delta, vmax_delta),
        ]):
            ax = axes[mi, ci]
            im = ax.imshow(col_data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')

            # X-axis
            xticks = list(range(0, len(layers), 10))
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(layers[i]) for i in xticks], fontsize=8)

            # Y-axis
            ax.set_yticks(range(len(k_values)))
            ax.set_yticklabels([str(k) for k in k_values], fontsize=8)

            if mi == 0:
                ax.set_title(title, fontsize=12)
            if ci == 0:
                ax.set_ylabel(f'{matrix}\nk', fontsize=10)
            else:
                ax.set_ylabel('k', fontsize=8)
            if mi == len(MATRICES) - 1:
                ax.set_xlabel('Layer', fontsize=10)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f'U Drift Heatmaps | {COMP_LABELS[comparison]}', fontsize=16, y=1.01)
    plt.tight_layout()

    out_path = output_dir / f'{comparison}_all_matrices_heatmap.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    return out_path


def plot_comparison_grid(
    csv_dir: Path,
    output_dir: Path,
    matrix: str,
):
    """
    For a single matrix, plot all 3 comparisons in one figure.
    Rows: comparisons, Cols: SFT, SDFT, Delta
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    all_data = {}

    # Load all data
    for comparison in COMPARISONS:
        df = load_csv(csv_dir, comparison, matrix)
        df['sft_U_drift'] = (1 - df['sft_U']) * 10000
        df['sdft_U_drift'] = (1 - df['sdft_U']) * 10000

        sft_U, layers, k_values = pivot_to_heatmap(df, 'sft_U_drift')
        sdft_U, _, _ = pivot_to_heatmap(df, 'sdft_U_drift')
        delta_U = sdft_U - sft_U

        all_data[comparison] = {
            'sft': sft_U,
            'sdft': sdft_U,
            'delta': delta_U,
            'layers': layers,
            'k_values': k_values,
        }

    # Global color ranges
    vmax_drift = max(
        max(np.nanmax(d['sft']), np.nanmax(d['sdft']))
        for d in all_data.values()
    )
    vmax_delta = max(np.nanmax(np.abs(d['delta'])) for d in all_data.values())

    # Plot
    for ri, comparison in enumerate(COMPARISONS):
        data = all_data[comparison]
        layers = data['layers']
        k_values = data['k_values']

        for ci, (col_data, title, cmap, vmin, vmax) in enumerate([
            (data['sft'], 'SFT', 'Blues', 0, vmax_drift),
            (data['sdft'], 'SDFT', 'Reds', 0, vmax_drift),
            (data['delta'], 'Δ (SDFT-SFT)', 'RdBu_r', -vmax_delta, vmax_delta),
        ]):
            ax = axes[ri, ci]
            im = ax.imshow(col_data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')

            xticks = list(range(0, len(layers), 5))
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(layers[i]) for i in xticks])
            ax.set_xlabel('Layer')

            ax.set_yticks(range(len(k_values)))
            ax.set_yticklabels([str(k) for k in k_values])
            ax.set_ylabel('k')

            if ri == 0:
                ax.set_title(title, fontsize=12)
            if ci == 0:
                ax.set_ylabel(f'{COMP_LABELS[comparison]}\nk', fontsize=10)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f'{matrix} | U Drift Across Training', fontsize=16)
    plt.tight_layout()

    out_path = output_dir / f'{matrix}_all_comparisons_heatmap.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    return out_path


def main():
    parser = argparse.ArgumentParser(description='Plot heatmaps per matrix from CSV data')
    parser.add_argument('--csv_dir', type=str,
                        default='eigenspectrum/outputs/layer_k_sweep/csv',
                        help='Directory containing CSV files')
    parser.add_argument('--output_dir', type=str,
                        default='eigenspectrum/outputs/layer_k_sweep/heatmaps',
                        help='Directory to save heatmap images')

    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading CSVs from: {csv_dir}")
    print(f"Saving heatmaps to: {output_dir}")

    # 1. Individual matrix heatmaps (per comparison)
    print("\n1. Generating individual matrix heatmaps...")
    for comparison in COMPARISONS:
        for matrix in MATRICES:
            out_path = plot_single_matrix_heatmaps(csv_dir, output_dir, comparison, matrix)
            print(f"   Saved: {out_path.name}")

    # 2. All matrices in single figure (per comparison)
    print("\n2. Generating combined matrix heatmaps per comparison...")
    for comparison in COMPARISONS:
        out_path = plot_all_matrices_single_comparison(csv_dir, output_dir, comparison)
        print(f"   Saved: {out_path.name}")

    # 3. All comparisons in single figure (per matrix)
    print("\n3. Generating comparison grids per matrix...")
    for matrix in MATRICES:
        out_path = plot_comparison_grid(csv_dir, output_dir, matrix)
        print(f"   Saved: {out_path.name}")

    print(f"\nDone! All heatmaps saved to {output_dir}")


if __name__ == '__main__':
    main()
