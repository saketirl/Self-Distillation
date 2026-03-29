"""Plot eigenspectrum from saved .npz files.

Usage:
    python -m eigenspectrum.plot_spectrum --file outputs/qwen2.5-3b-instruct/model_layers_0_self_attn_q_proj_weight_spectrum.npz
    python -m eigenspectrum.plot_spectrum --input_dir outputs/qwen2.5-3b-instruct
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob


def plot_single_spectrum(npz_path: str, num_bins: int = 50, save: bool = True):
    """Plot eigenspectrum density and log density histograms."""
    data = np.load(npz_path, allow_pickle=True)

    param_name = str(data["param_name"])
    eigenvalues = data["eigenvalues"].flatten()
    effective_rank = float(data["effective_rank"])
    top_eigenvalue = float(data["top_eigenvalue"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Density histogram
    ax1 = axes[0]
    counts, bin_edges, _ = ax1.hist(eigenvalues, bins=num_bins, density=True,
                                     alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel("Eigenvalue (λ)")
    ax1.set_ylabel("Density")
    ax1.set_title(f"Eigenvalue Density\nTop λ={top_eigenvalue:.2e}, Eff. Rank={effective_rank:.1f}")
    ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='λ=0')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Log density histogram
    ax2 = axes[1]
    # Compute histogram manually for log density
    counts, bin_edges = np.histogram(eigenvalues, bins=num_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Take log, filter out zeros
    nonzero = counts > 0
    log_counts = np.log10(counts[nonzero])
    bin_centers_nz = bin_centers[nonzero]

    ax2.bar(bin_centers_nz, log_counts, width=(bin_edges[1] - bin_edges[0]) * 0.9,
            alpha=0.7, color='steelblue', edgecolor='black')
    ax2.set_xlabel("Eigenvalue (λ)")
    ax2.set_ylabel("log₁₀(Density)")
    ax2.set_title(f"Eigenvalue Log Density\n({len(eigenvalues)} eigenvalues, {num_bins} bins)")
    ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='λ=0')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(param_name, fontsize=10)
    plt.tight_layout()

    if save:
        out_path = npz_path.replace("_spectrum.npz", "_density.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")

    return fig


def plot_comparison(npz_files: list, num_bins: int = 50, save_path: str = None):
    """Plot multiple spectra on the same axes for comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(npz_files)))

    for i, npz_path in enumerate(npz_files):
        data = np.load(npz_path, allow_pickle=True)

        param_name = str(data["param_name"])
        short_name = ".".join(param_name.split(".")[-2:])
        eigenvalues = data["eigenvalues"].flatten()

        # Compute histogram
        counts, bin_edges = np.histogram(eigenvalues, bins=num_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Plot density
        axes[0].plot(bin_centers, counts, label=short_name, color=colors[i], linewidth=1.5)

        # Plot log density
        nonzero = counts > 0
        log_counts = np.log10(counts[nonzero])
        axes[1].plot(bin_centers[nonzero], log_counts, label=short_name,
                     color=colors[i], linewidth=1.5)

    axes[0].set_xlabel("Eigenvalue (λ)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Eigenvalue Density Comparison")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.5)

    axes[1].set_xlabel("Eigenvalue (λ)")
    axes[1].set_ylabel("log₁₀(Density)")
    axes[1].set_title("Eigenvalue Log Density Comparison")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot eigenspectrum density histograms")
    parser.add_argument("--file", type=str, help="Single .npz file to plot")
    parser.add_argument("--input_dir", type=str, help="Directory with .npz files")
    parser.add_argument("--compare", action="store_true", help="Plot all spectra on same axes")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--num_bins", type=int, default=50,
                        help="Number of histogram bins (default: 50)")
    args = parser.parse_args()

    if args.file:
        plot_single_spectrum(args.file, num_bins=args.num_bins)
        if args.show:
            plt.show()

    elif args.input_dir:
        npz_files = sorted(glob.glob(f"{args.input_dir}/*_spectrum.npz"))

        if not npz_files:
            print(f"No *_spectrum.npz files found in {args.input_dir}")
            return

        print(f"Found {len(npz_files)} spectrum files")

        if args.compare:
            save_path = f"{args.input_dir}/comparison_density.png"
            plot_comparison(npz_files, num_bins=args.num_bins, save_path=save_path)
        else:
            for npz_path in npz_files:
                print(f"\nPlotting: {npz_path}")
                plot_single_spectrum(npz_path, num_bins=args.num_bins)

        if args.show:
            plt.show()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
