#!/usr/bin/env python3
"""
Measure how singular values change during training.

For each comparison (base→task), compute:
  - Absolute change: σ_task[i] - σ_base[i]
  - Relative change: (σ_task[i] - σ_base[i]) / σ_base[i]
  - Ratio: σ_task[i] / σ_base[i]

Sweep over k values (top-k singular values) and store statistics.
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Configuration
# =============================================================================

DATA_ROOT = "/data/saket/continual/Self-Distillation"

K_VALUES = [5, 10, 20, 50, 100, 250, 500]

# Specific indices to track per-singular-value changes (0-indexed)
SV_INDICES = [0, 4, 9, 19, 49, 99, 249, 499]  # σ_1, σ_5, σ_10, σ_20, σ_50, σ_100, σ_250, σ_500

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

COMPARISONS = [
    ('base_to_tooluse', 0, 1),
    ('base_to_gsm8k', 0, 2),
    ('base_to_mbpp', 0, 3),
]


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


def svd_singular_values(W: np.ndarray, gpu: bool = True) -> np.ndarray:
    """Compute singular values only (faster than full SVD)."""
    if gpu:
        import torch
        if torch.cuda.is_available():
            t = torch.tensor(W, dtype=torch.float32, device='cuda')
            S = torch.linalg.svdvals(t)
            return S.cpu().numpy()
    S = np.linalg.svd(W, compute_uv=False)
    return S


def compute_sv_changes(S_base: np.ndarray, S_task: np.ndarray, k: int) -> Dict[str, float]:
    """
    Compute statistics for how top-k singular values change.

    Returns:
        - mean_abs_change: mean(σ_task[i] - σ_base[i]) for i in 0..k-1
        - mean_rel_change: mean((σ_task[i] - σ_base[i]) / σ_base[i])
        - mean_ratio: mean(σ_task[i] / σ_base[i])
        - max_abs_change: max absolute change
        - min_abs_change: min absolute change
        - num_increased: count of σ that increased
        - num_decreased: count of σ that decreased
        - total_mass_base: sum(σ_base[:k])
        - total_mass_task: sum(σ_task[:k])
        - mass_change_pct: percentage change in total mass
        - percentile_25_abs_change: 25th percentile of absolute changes
        - percentile_50_abs_change: median of absolute changes
        - percentile_75_abs_change: 75th percentile of absolute changes
        - percentile_25_ratio: 25th percentile of ratios
        - percentile_50_ratio: median of ratios
        - percentile_75_ratio: 75th percentile of ratios
    """
    k = min(k, len(S_base), len(S_task))

    s_base = S_base[:k]
    s_task = S_task[:k]

    # Absolute change
    abs_change = s_task - s_base
    mean_abs_change = float(np.mean(abs_change))
    max_abs_change = float(np.max(abs_change))
    min_abs_change = float(np.min(abs_change))

    # Relative change (avoid division by zero)
    rel_change = abs_change / (s_base + 1e-10)
    mean_rel_change = float(np.mean(rel_change))

    # Ratio
    ratio = s_task / (s_base + 1e-10)
    mean_ratio = float(np.mean(ratio))

    # Counts
    num_increased = int(np.sum(abs_change > 0))
    num_decreased = int(np.sum(abs_change < 0))

    # Total mass (sum of singular values = nuclear norm contribution)
    total_mass_base = float(np.sum(s_base))
    total_mass_task = float(np.sum(s_task))
    mass_change_pct = (total_mass_task - total_mass_base) / (total_mass_base + 1e-10) * 100

    # Percentiles for absolute change
    percentile_25_abs_change = float(np.percentile(abs_change, 25))
    percentile_50_abs_change = float(np.percentile(abs_change, 50))  # median
    percentile_75_abs_change = float(np.percentile(abs_change, 75))

    # Percentiles for ratio
    percentile_25_ratio = float(np.percentile(ratio, 25))
    percentile_50_ratio = float(np.percentile(ratio, 50))  # median
    percentile_75_ratio = float(np.percentile(ratio, 75))

    return {
        'mean_abs_change': mean_abs_change,
        'mean_rel_change': mean_rel_change,
        'mean_ratio': mean_ratio,
        'max_abs_change': max_abs_change,
        'min_abs_change': min_abs_change,
        'num_increased': num_increased,
        'num_decreased': num_decreased,
        'total_mass_base': total_mass_base,
        'total_mass_task': total_mass_task,
        'mass_change_pct': mass_change_pct,
        'percentile_25_abs_change': percentile_25_abs_change,
        'percentile_50_abs_change': percentile_50_abs_change,
        'percentile_75_abs_change': percentile_75_abs_change,
        'percentile_25_ratio': percentile_25_ratio,
        'percentile_50_ratio': percentile_50_ratio,
        'percentile_75_ratio': percentile_75_ratio,
        'k': k,
    }


def compute_per_index_changes(S_base: np.ndarray, S_task: np.ndarray, indices: List[int]) -> Dict[str, float]:
    """
    Compute changes for specific singular value indices.

    Args:
        S_base: Base singular values (sorted descending)
        S_task: Task singular values (sorted descending)
        indices: List of 0-indexed positions to track (e.g., [0, 4, 9] for σ_1, σ_5, σ_10)

    Returns:
        Dictionary with per-index metrics:
        - sv_base_i: base singular value at index i
        - sv_task_i: task singular value at index i
        - abs_change_i: absolute change at index i
        - rel_change_i: relative change at index i
        - ratio_i: ratio at index i
    """
    results = {}
    max_idx = min(len(S_base), len(S_task))

    for idx in indices:
        suffix = f'_{idx+1}'  # 1-indexed for readability (σ_1, σ_5, etc.)

        if idx < max_idx:
            s_base = float(S_base[idx])
            s_task = float(S_task[idx])
            abs_change = s_task - s_base
            rel_change = abs_change / (s_base + 1e-10)
            ratio = s_task / (s_base + 1e-10)

            results[f'sv_base{suffix}'] = s_base
            results[f'sv_task{suffix}'] = s_task
            results[f'abs_change{suffix}'] = abs_change
            results[f'rel_change{suffix}'] = rel_change
            results[f'ratio{suffix}'] = ratio
        else:
            # Index out of range for this matrix
            results[f'sv_base{suffix}'] = np.nan
            results[f'sv_task{suffix}'] = np.nan
            results[f'abs_change{suffix}'] = np.nan
            results[f'rel_change{suffix}'] = np.nan
            results[f'ratio{suffix}'] = np.nan

    return results


# =============================================================================
# Main Analysis
# =============================================================================

def run_analysis(
    data_root: str,
    output_dir: str,
    layers: List[int] = None,
    params: List[str] = None,
    k_values: List[int] = None,
    gpu: bool = True,
):
    """Run singular value change analysis."""

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

    # Storage structure:
    # results[comparison][method][param] = {
    #     'mean_abs_change': array(n_layers, n_k),
    #     'mean_rel_change': array(n_layers, n_k),
    #     'mean_ratio': array(n_layers, n_k),
    #     'mass_change_pct': array(n_layers, n_k),
    #     ...
    # }

    # Aggregate metrics (over top-k)
    aggregate_metrics = ['mean_abs_change', 'mean_rel_change', 'mean_ratio',
                         'max_abs_change', 'min_abs_change',
                         'num_increased', 'num_decreased',
                         'total_mass_base', 'total_mass_task', 'mass_change_pct',
                         'percentile_25_abs_change', 'percentile_50_abs_change', 'percentile_75_abs_change',
                         'percentile_25_ratio', 'percentile_50_ratio', 'percentile_75_ratio']

    # Per-index metrics
    per_index_metrics = []
    for idx in SV_INDICES:
        suffix = f'_{idx+1}'
        per_index_metrics.extend([
            f'sv_base{suffix}', f'sv_task{suffix}',
            f'abs_change{suffix}', f'rel_change{suffix}', f'ratio{suffix}'
        ])

    results = {
        comp_name: {
            method: {
                param: {
                    metric: np.zeros((n_layers, n_k))
                    for metric in aggregate_metrics
                }
                for param in params
            }
            for method in ['sft', 'sdft']
        }
        for comp_name, _, _ in COMPARISONS
    }

    # Separate storage for per-index metrics (no k dimension, just layer)
    per_index_results = {
        comp_name: {
            method: {
                param: {
                    metric: np.zeros(n_layers)
                    for metric in per_index_metrics
                }
                for param in params
            }
            for method in ['sft', 'sdft']
        }
        for comp_name, _, _ in COMPARISONS
    }

    total = n_layers * len(params)
    count = 0

    for pi, param in enumerate(params):
        pshort = PARAM_SHORT[param]

        for li, layer in enumerate(layers):
            count += 1
            print(f"[{count}/{total}] Layer {layer:2d} {pshort}...", end=' ', flush=True)

            try:
                # Load all checkpoints and compute singular values
                sft_svs = [svd_singular_values(load_weight(paths['sft'][i], layer, param), gpu)
                           for i in range(4)]
                sdft_svs = [svd_singular_values(load_weight(paths['sdft'][i], layer, param), gpu)
                            for i in range(4)]

                # For each comparison
                for comp_name, idx_from, idx_to in COMPARISONS:
                    # Aggregate metrics (sweep over k)
                    for ki, k in enumerate(k_values):
                        # SFT
                        sv_changes_sft = compute_sv_changes(sft_svs[idx_from], sft_svs[idx_to], k)
                        for metric in aggregate_metrics:
                            results[comp_name]['sft'][param][metric][li, ki] = sv_changes_sft[metric]

                        # SDFT
                        sv_changes_sdft = compute_sv_changes(sdft_svs[idx_from], sdft_svs[idx_to], k)
                        for metric in aggregate_metrics:
                            results[comp_name]['sdft'][param][metric][li, ki] = sv_changes_sdft[metric]

                    # Per-index metrics (no k sweep)
                    per_idx_sft = compute_per_index_changes(sft_svs[idx_from], sft_svs[idx_to], SV_INDICES)
                    per_idx_sdft = compute_per_index_changes(sdft_svs[idx_from], sdft_svs[idx_to], SV_INDICES)

                    for metric in per_index_metrics:
                        per_index_results[comp_name]['sft'][param][metric][li] = per_idx_sft[metric]
                        per_index_results[comp_name]['sdft'][param][metric][li] = per_idx_sdft[metric]

                print("done")

            except Exception as e:
                print(f"ERROR: {e}")
                # Fill with NaN
                for comp_name, _, _ in COMPARISONS:
                    for method in ['sft', 'sdft']:
                        for metric in aggregate_metrics:
                            results[comp_name][method][param][metric][li, :] = np.nan
                        for metric in per_index_metrics:
                            per_index_results[comp_name][method][param][metric][li] = np.nan

    # Save results
    save_results(results, per_index_results, layers, k_values, params,
                 aggregate_metrics, per_index_metrics, output_dir)
    save_csv(results, per_index_results, layers, k_values, params,
             aggregate_metrics, per_index_metrics, output_dir)
    print_summary(results, per_index_results, layers, k_values, params)

    return results, per_index_results


def save_results(results: Dict, per_index_results: Dict, layers: List[int], k_values: List[int],
                 params: List[str], aggregate_metrics: List[str], per_index_metrics: List[str],
                 output_dir: Path):
    """Save results to JSON."""

    json_results = {
        'layers': layers,
        'k_values': k_values,
        'sv_indices': [idx + 1 for idx in SV_INDICES],  # 1-indexed for readability
        'params': params,
        'param_short_names': {p: PARAM_SHORT[p] for p in params},
        'comparisons': [c[0] for c in COMPARISONS],
        'aggregate_metrics': aggregate_metrics,
        'per_index_metrics': per_index_metrics,
        'aggregate_data': {},
        'per_index_data': {},
    }

    for comp_name, _, _ in COMPARISONS:
        json_results['aggregate_data'][comp_name] = {}
        json_results['per_index_data'][comp_name] = {}
        for method in ['sft', 'sdft']:
            json_results['aggregate_data'][comp_name][method] = {}
            json_results['per_index_data'][comp_name][method] = {}
            for param in params:
                json_results['aggregate_data'][comp_name][method][param] = {
                    metric: results[comp_name][method][param][metric].tolist()
                    for metric in aggregate_metrics
                }
                json_results['per_index_data'][comp_name][method][param] = {
                    metric: per_index_results[comp_name][method][param][metric].tolist()
                    for metric in per_index_metrics
                }

    with open(output_dir / 'singular_value_changes.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved JSON to {output_dir / 'singular_value_changes.json'}")


def save_csv(results: Dict, per_index_results: Dict, layers: List[int], k_values: List[int],
             params: List[str], aggregate_metrics: List[str], per_index_metrics: List[str],
             output_dir: Path):
    """Save per-matrix CSV files."""

    csv_dir = output_dir / 'csv'
    csv_dir.mkdir(exist_ok=True)

    # 1. Aggregate metrics CSV (with k sweep)
    for comp_name, _, _ in COMPARISONS:
        for param in params:
            pshort = PARAM_SHORT[param]
            csv_path = csv_dir / f'{comp_name}_{pshort}_sv_changes.csv'

            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)

                # Header
                header = ['layer', 'k']
                for metric in aggregate_metrics:
                    header.extend([f'sft_{metric}', f'sdft_{metric}', f'delta_{metric}'])
                writer.writerow(header)

                for li, layer in enumerate(layers):
                    for ki, k in enumerate(k_values):
                        row = [layer, k]
                        for metric in aggregate_metrics:
                            sft_val = results[comp_name]['sft'][param][metric][li, ki]
                            sdft_val = results[comp_name]['sdft'][param][metric][li, ki]
                            delta_val = sdft_val - sft_val
                            row.extend([sft_val, sdft_val, delta_val])
                        writer.writerow(row)

    # 2. Per-index metrics CSV (no k, just layer)
    for comp_name, _, _ in COMPARISONS:
        for param in params:
            pshort = PARAM_SHORT[param]
            csv_path = csv_dir / f'{comp_name}_{pshort}_sv_per_index.csv'

            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)

                # Header
                header = ['layer']
                for metric in per_index_metrics:
                    header.extend([f'sft_{metric}', f'sdft_{metric}', f'delta_{metric}'])
                writer.writerow(header)

                for li, layer in enumerate(layers):
                    row = [layer]
                    for metric in per_index_metrics:
                        sft_val = per_index_results[comp_name]['sft'][param][metric][li]
                        sdft_val = per_index_results[comp_name]['sdft'][param][metric][li]
                        delta_val = sdft_val - sft_val
                        row.extend([sft_val, sdft_val, delta_val])
                    writer.writerow(row)

    print(f"Saved CSV files to {csv_dir}")


def print_summary(results: Dict, per_index_results: Dict, layers: List[int],
                  k_values: List[int], params: List[str]):
    """Print summary statistics."""

    ki_50 = k_values.index(50) if 50 in k_values else len(k_values) // 2
    k_selected = k_values[ki_50]

    print("\n" + "="*120)
    print(f"SUMMARY: Singular Value Changes (k={k_selected}, averaged over layers and params)")
    print("="*120)

    print(f"\n{'Comparison':<20} {'Method':<8} {'Mean Δσ':>12} {'Median Δσ':>12} {'Mean Ratio':>12} {'Median Ratio':>12} {'Mass Δ%':>12}")
    print("-"*120)

    for comp_name, _, _ in COMPARISONS:
        for method in ['sft', 'sdft']:
            abs_changes = []
            median_abs = []
            ratios = []
            median_ratios = []
            mass_pcts = []

            for param in params:
                abs_changes.extend(results[comp_name][method][param]['mean_abs_change'][:, ki_50])
                median_abs.extend(results[comp_name][method][param]['percentile_50_abs_change'][:, ki_50])
                ratios.extend(results[comp_name][method][param]['mean_ratio'][:, ki_50])
                median_ratios.extend(results[comp_name][method][param]['percentile_50_ratio'][:, ki_50])
                mass_pcts.extend(results[comp_name][method][param]['mass_change_pct'][:, ki_50])

            print(f"{comp_name:<20} {method.upper():<8} "
                  f"{np.nanmean(abs_changes):>12.6f} {np.nanmean(median_abs):>12.6f} "
                  f"{np.nanmean(ratios):>12.6f} {np.nanmean(median_ratios):>12.6f} "
                  f"{np.nanmean(mass_pcts):>12.4f}%")

    # Per-index summary
    print("\n" + "="*120)
    print("PER-INDEX SUMMARY: Changes at specific singular value positions (averaged over layers and params)")
    print("="*120)

    indices_to_show = [0, 4, 9, 49]  # σ_1, σ_5, σ_10, σ_50
    print(f"\n{'Comparison':<20} {'Method':<8}", end='')
    for idx in indices_to_show:
        print(f" {'Δσ_'+str(idx+1):>10} {'ratio_'+str(idx+1):>10}", end='')
    print()
    print("-"*120)

    for comp_name, _, _ in COMPARISONS:
        for method in ['sft', 'sdft']:
            print(f"{comp_name:<20} {method.upper():<8}", end='')

            for idx in indices_to_show:
                suffix = f'_{idx+1}'
                abs_changes = []
                ratios = []

                for param in params:
                    if f'abs_change{suffix}' in per_index_results[comp_name][method][param]:
                        abs_changes.extend(per_index_results[comp_name][method][param][f'abs_change{suffix}'])
                        ratios.extend(per_index_results[comp_name][method][param][f'ratio{suffix}'])

                print(f" {np.nanmean(abs_changes):>10.6f} {np.nanmean(ratios):>10.6f}", end='')
            print()

    print("="*120)
    print("\nInterpretation:")
    print("  - Mean Δσ > 0: singular values increased on average")
    print("  - Mean Ratio > 1: singular values grew; < 1: singular values shrunk")
    print("  - Mass Δ% > 0: total spectral mass (sum of top-k σ) increased")
    print("  - Per-index: σ_1 is largest, σ_50 is 50th largest")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Measure how singular values change during training'
    )
    parser.add_argument('--data_root', type=str, default=DATA_ROOT)
    parser.add_argument('--output_dir', type=str,
                        default='eigenspectrum/outputs/sv_changes')
    parser.add_argument('--layers', type=int, nargs='+', default=None)
    parser.add_argument('--k_values', type=int, nargs='+', default=K_VALUES)
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()

    run_analysis(
        data_root=args.data_root,
        output_dir=args.output_dir,
        layers=args.layers,
        k_values=args.k_values,
        gpu=not args.cpu,
    )


if __name__ == '__main__':
    main()
