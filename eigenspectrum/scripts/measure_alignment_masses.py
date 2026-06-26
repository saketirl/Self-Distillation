#!/usr/bin/env python3
"""
Measure diagonal and off-diagonal mass of alignment matrices.

For each comparison (base→task), compute:
  A_U = U_base[:,:k]^T @ U_task[:,:k]
  A_V = V_base[:,:k]^T @ V_task[:,:k]

Then measure:
  - Diagonal mass: mean(|A[i,i]|) over k diagonal elements
  - Off-diagonal mass: mean(|A[i,j]|) over k*(k-1) off-diagonal elements

Sweep over k values and store all data.
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


def compute_alignment_masses(U1: np.ndarray, U2: np.ndarray, k: int) -> Dict[str, float]:
    """
    Compute diagonal and off-diagonal mass of alignment matrix.

    A = U1[:,:k]^T @ U2[:,:k]  (k x k matrix)

    Returns:
        - diag_mass: mean(|A[i,i]|) over k elements
        - off_diag_mass: mean(|A[i,j]| for i≠j) over k*(k-1) elements
        - diag_sum: sum(|A[i,i]|)
        - off_diag_sum: sum(|A[i,j]| for i≠j)
    """
    k = min(k, U1.shape[1], U2.shape[1])

    A = U1[:, :k].T @ U2[:, :k]  # k x k
    A_abs = np.abs(A)

    # Diagonal elements
    diag = np.diag(A_abs)
    diag_sum = float(np.sum(diag))
    diag_mass = float(np.mean(diag))  # normalized by k

    # Off-diagonal elements
    mask = ~np.eye(k, dtype=bool)
    off_diag = A_abs[mask]
    off_diag_sum = float(np.sum(off_diag))
    off_diag_mass = float(np.mean(off_diag))  # normalized by k*(k-1)

    # Total mass
    total_mass = diag_sum + off_diag_sum

    # Ratio of diagonal to total
    diag_ratio = diag_sum / total_mass if total_mass > 0 else 0.0

    return {
        'diag_mass': diag_mass,
        'off_diag_mass': off_diag_mass,
        'diag_sum': diag_sum,
        'off_diag_sum': off_diag_sum,
        'diag_ratio': diag_ratio,
        'k': k,
    }


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
    """Run alignment mass analysis."""

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
    #     'U_diag_mass': array(n_layers, n_k),
    #     'U_off_diag_mass': array(n_layers, n_k),
    #     'V_diag_mass': array(n_layers, n_k),
    #     'V_off_diag_mass': array(n_layers, n_k),
    #     'U_diag_ratio': array(n_layers, n_k),
    #     'V_diag_ratio': array(n_layers, n_k),
    # }

    results = {
        comp_name: {
            method: {
                param: {
                    'U_diag_mass': np.zeros((n_layers, n_k)),
                    'U_off_diag_mass': np.zeros((n_layers, n_k)),
                    'U_diag_ratio': np.zeros((n_layers, n_k)),
                    'V_diag_mass': np.zeros((n_layers, n_k)),
                    'V_off_diag_mass': np.zeros((n_layers, n_k)),
                    'V_diag_ratio': np.zeros((n_layers, n_k)),
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
                # Load all checkpoints
                sft_weights = [load_weight(paths['sft'][i], layer, param) for i in range(4)]
                sdft_weights = [load_weight(paths['sdft'][i], layer, param) for i in range(4)]

                # Compute SVDs
                sft_svds = [svd_full(W, gpu) for W in sft_weights]  # Each is (U, S, V)
                sdft_svds = [svd_full(W, gpu) for W in sdft_weights]

                # For each comparison
                for comp_name, idx_from, idx_to in COMPARISONS:
                    for ki, k in enumerate(k_values):
                        # SFT
                        U_base_sft = sft_svds[idx_from][0]
                        U_task_sft = sft_svds[idx_to][0]
                        V_base_sft = sft_svds[idx_from][2]
                        V_task_sft = sft_svds[idx_to][2]

                        u_masses_sft = compute_alignment_masses(U_base_sft, U_task_sft, k)
                        v_masses_sft = compute_alignment_masses(V_base_sft, V_task_sft, k)

                        results[comp_name]['sft'][param]['U_diag_mass'][li, ki] = u_masses_sft['diag_mass']
                        results[comp_name]['sft'][param]['U_off_diag_mass'][li, ki] = u_masses_sft['off_diag_mass']
                        results[comp_name]['sft'][param]['U_diag_ratio'][li, ki] = u_masses_sft['diag_ratio']
                        results[comp_name]['sft'][param]['V_diag_mass'][li, ki] = v_masses_sft['diag_mass']
                        results[comp_name]['sft'][param]['V_off_diag_mass'][li, ki] = v_masses_sft['off_diag_mass']
                        results[comp_name]['sft'][param]['V_diag_ratio'][li, ki] = v_masses_sft['diag_ratio']

                        # SDFT
                        U_base_sdft = sdft_svds[idx_from][0]
                        U_task_sdft = sdft_svds[idx_to][0]
                        V_base_sdft = sdft_svds[idx_from][2]
                        V_task_sdft = sdft_svds[idx_to][2]

                        u_masses_sdft = compute_alignment_masses(U_base_sdft, U_task_sdft, k)
                        v_masses_sdft = compute_alignment_masses(V_base_sdft, V_task_sdft, k)

                        results[comp_name]['sdft'][param]['U_diag_mass'][li, ki] = u_masses_sdft['diag_mass']
                        results[comp_name]['sdft'][param]['U_off_diag_mass'][li, ki] = u_masses_sdft['off_diag_mass']
                        results[comp_name]['sdft'][param]['U_diag_ratio'][li, ki] = u_masses_sdft['diag_ratio']
                        results[comp_name]['sdft'][param]['V_diag_mass'][li, ki] = v_masses_sdft['diag_mass']
                        results[comp_name]['sdft'][param]['V_off_diag_mass'][li, ki] = v_masses_sdft['off_diag_mass']
                        results[comp_name]['sdft'][param]['V_diag_ratio'][li, ki] = v_masses_sdft['diag_ratio']

                print("done")

            except Exception as e:
                print(f"ERROR: {e}")
                # Fill with NaN
                for comp_name, _, _ in COMPARISONS:
                    for method in ['sft', 'sdft']:
                        for key in results[comp_name][method][param]:
                            results[comp_name][method][param][key][li, :] = np.nan

    # Save results
    save_results(results, layers, k_values, params, output_dir)
    save_csv(results, layers, k_values, params, output_dir)
    print_summary(results, layers, k_values, params)

    return results


def save_results(results: Dict, layers: List[int], k_values: List[int],
                 params: List[str], output_dir: Path):
    """Save results to JSON."""

    json_results = {
        'layers': layers,
        'k_values': k_values,
        'params': params,
        'param_short_names': {p: PARAM_SHORT[p] for p in params},
        'comparisons': [c[0] for c in COMPARISONS],
        'metrics': ['U_diag_mass', 'U_off_diag_mass', 'U_diag_ratio',
                    'V_diag_mass', 'V_off_diag_mass', 'V_diag_ratio'],
        'data': {}
    }

    for comp_name, _, _ in COMPARISONS:
        json_results['data'][comp_name] = {}
        for method in ['sft', 'sdft']:
            json_results['data'][comp_name][method] = {}
            for param in params:
                json_results['data'][comp_name][method][param] = {
                    key: val.tolist()
                    for key, val in results[comp_name][method][param].items()
                }

    with open(output_dir / 'alignment_masses.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved JSON to {output_dir / 'alignment_masses.json'}")


def save_csv(results: Dict, layers: List[int], k_values: List[int],
             params: List[str], output_dir: Path):
    """Save per-matrix CSV files."""

    csv_dir = output_dir / 'csv'
    csv_dir.mkdir(exist_ok=True)

    for comp_name, _, _ in COMPARISONS:
        for param in params:
            pshort = PARAM_SHORT[param]
            csv_path = csv_dir / f'{comp_name}_{pshort}_masses.csv'

            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'layer', 'k',
                    'sft_U_diag_mass', 'sft_U_off_diag_mass', 'sft_U_diag_ratio',
                    'sft_V_diag_mass', 'sft_V_off_diag_mass', 'sft_V_diag_ratio',
                    'sdft_U_diag_mass', 'sdft_U_off_diag_mass', 'sdft_U_diag_ratio',
                    'sdft_V_diag_mass', 'sdft_V_off_diag_mass', 'sdft_V_diag_ratio',
                    'delta_U_diag_mass', 'delta_U_off_diag_mass', 'delta_U_diag_ratio',
                    'delta_V_diag_mass', 'delta_V_off_diag_mass', 'delta_V_diag_ratio',
                ])

                for li, layer in enumerate(layers):
                    for ki, k in enumerate(k_values):
                        sft = results[comp_name]['sft'][param]
                        sdft = results[comp_name]['sdft'][param]

                        writer.writerow([
                            layer, k,
                            sft['U_diag_mass'][li, ki],
                            sft['U_off_diag_mass'][li, ki],
                            sft['U_diag_ratio'][li, ki],
                            sft['V_diag_mass'][li, ki],
                            sft['V_off_diag_mass'][li, ki],
                            sft['V_diag_ratio'][li, ki],
                            sdft['U_diag_mass'][li, ki],
                            sdft['U_off_diag_mass'][li, ki],
                            sdft['U_diag_ratio'][li, ki],
                            sdft['V_diag_mass'][li, ki],
                            sdft['V_off_diag_mass'][li, ki],
                            sdft['V_diag_ratio'][li, ki],
                            sdft['U_diag_mass'][li, ki] - sft['U_diag_mass'][li, ki],
                            sdft['U_off_diag_mass'][li, ki] - sft['U_off_diag_mass'][li, ki],
                            sdft['U_diag_ratio'][li, ki] - sft['U_diag_ratio'][li, ki],
                            sdft['V_diag_mass'][li, ki] - sft['V_diag_mass'][li, ki],
                            sdft['V_off_diag_mass'][li, ki] - sft['V_off_diag_mass'][li, ki],
                            sdft['V_diag_ratio'][li, ki] - sft['V_diag_ratio'][li, ki],
                        ])

    print(f"Saved CSV files to {csv_dir}")


def print_summary(results: Dict, layers: List[int], k_values: List[int],
                  params: List[str]):
    """Print summary statistics."""

    ki_50 = k_values.index(50) if 50 in k_values else len(k_values) // 2
    k_selected = k_values[ki_50]

    print("\n" + "="*100)
    print(f"SUMMARY: Alignment Mass (k={k_selected}, averaged over layers and params)")
    print("="*100)

    print(f"\n{'Comparison':<20} {'Method':<8} {'U_diag':>10} {'U_off_diag':>12} {'U_ratio':>10} {'V_diag':>10} {'V_off_diag':>12} {'V_ratio':>10}")
    print("-"*100)

    for comp_name, _, _ in COMPARISONS:
        for method in ['sft', 'sdft']:
            u_diag_all = []
            u_off_diag_all = []
            u_ratio_all = []
            v_diag_all = []
            v_off_diag_all = []
            v_ratio_all = []

            for param in params:
                u_diag_all.extend(results[comp_name][method][param]['U_diag_mass'][:, ki_50])
                u_off_diag_all.extend(results[comp_name][method][param]['U_off_diag_mass'][:, ki_50])
                u_ratio_all.extend(results[comp_name][method][param]['U_diag_ratio'][:, ki_50])
                v_diag_all.extend(results[comp_name][method][param]['V_diag_mass'][:, ki_50])
                v_off_diag_all.extend(results[comp_name][method][param]['V_off_diag_mass'][:, ki_50])
                v_ratio_all.extend(results[comp_name][method][param]['V_diag_ratio'][:, ki_50])

            print(f"{comp_name:<20} {method.upper():<8} "
                  f"{np.nanmean(u_diag_all):>10.4f} {np.nanmean(u_off_diag_all):>12.4f} {np.nanmean(u_ratio_all):>10.4f} "
                  f"{np.nanmean(v_diag_all):>10.4f} {np.nanmean(v_off_diag_all):>12.4f} {np.nanmean(v_ratio_all):>10.4f}")

    print("="*100)
    print("\nInterpretation:")
    print("  - diag_mass ≈ 1.0 means singular vectors are well-aligned")
    print("  - off_diag_mass ≈ 0.0 means no mixing between different singular directions")
    print("  - diag_ratio close to 1.0 means most mass is on diagonal (good alignment)")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Measure diagonal and off-diagonal mass of alignment matrices'
    )
    parser.add_argument('--data_root', type=str, default=DATA_ROOT)
    parser.add_argument('--output_dir', type=str,
                        default='eigenspectrum/outputs/alignment_masses')
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
