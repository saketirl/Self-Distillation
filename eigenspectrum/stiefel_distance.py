"""
Measure distance from weight matrices to the Stiefel manifold.

The Stiefel manifold St(n, p) consists of n×p matrices with orthonormal columns.
For a matrix W with SVD W = UΣV^T, the closest point on St is UV^T.

Spectral norm distance: ||W - UV^T||_2 = max_i |σ_i - 1|

This measures how far singular values deviate from 1.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM
from typing import Dict, List, Tuple, Optional
import argparse
import json
from pathlib import Path


def spectral_distance_to_stiefel(W: torch.Tensor) -> Dict[str, float]:
    """
    Compute the spectral norm distance from matrix W to the Stiefel manifold.

    Args:
        W: 2D tensor (weight matrix)

    Returns:
        Dictionary with distance metrics
    """
    # Ensure 2D
    if W.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {W.dim()}D")

    # Compute SVD
    # Use float32 for numerical stability in SVD
    W_float = W.float()
    U, S, Vh = torch.linalg.svd(W_float, full_matrices=False)

    # Singular values
    singular_values = S.cpu().numpy()

    # Spectral distance to Stiefel: max |σ_i - 1|
    deviations = np.abs(singular_values - 1.0)
    spectral_dist = float(np.max(deviations))

    # Additional metrics
    mean_deviation = float(np.mean(deviations))
    frobenius_dist = float(np.sqrt(np.sum(deviations**2)))  # Frobenius distance

    # Which singular value is furthest from 1?
    max_idx = int(np.argmax(deviations))
    max_sv = float(singular_values[max_idx])

    # Condition number (ratio of largest to smallest singular value)
    condition_number = float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 1e-10 else float('inf')

    return {
        'spectral_distance': spectral_dist,
        'frobenius_distance': frobenius_dist,
        'mean_deviation': mean_deviation,
        'max_singular_value': float(singular_values[0]),
        'min_singular_value': float(singular_values[-1]),
        'furthest_sv_index': max_idx,
        'furthest_sv_value': max_sv,
        'condition_number': condition_number,
        'num_singular_values': len(singular_values),
        'singular_values_sample': singular_values[:10].tolist(),  # First 10 for inspection
    }


def analyze_layer(model, layer_idx: int, device: str = 'cpu') -> Dict[str, Dict]:
    """
    Analyze all weight matrices in a transformer layer.

    Args:
        model: HuggingFace model
        layer_idx: Layer index
        device: Device for computation

    Returns:
        Dictionary mapping parameter names to their Stiefel distances
    """
    layer = model.model.layers[layer_idx]

    # Weight matrices to analyze
    param_names = [
        'self_attn.q_proj.weight',
        'self_attn.k_proj.weight',
        'self_attn.v_proj.weight',
        'self_attn.o_proj.weight',
        'mlp.gate_proj.weight',
        'mlp.up_proj.weight',
        'mlp.down_proj.weight',
    ]

    results = {}

    for param_name in param_names:
        # Navigate to the parameter
        parts = param_name.split('.')
        obj = layer
        for part in parts[:-1]:
            obj = getattr(obj, part)
        weight = getattr(obj, parts[-1])

        if weight is None:
            continue

        # Move to device and compute
        W = weight.data.to(device)

        try:
            metrics = spectral_distance_to_stiefel(W)
            metrics['shape'] = list(W.shape)
            results[param_name] = metrics
        except Exception as e:
            print(f"  Error analyzing {param_name}: {e}")
            results[param_name] = {'error': str(e)}

    return results


def analyze_model(
    model_path: str,
    output_path: Optional[str] = None,
    device: str = 'cuda',
    layers: Optional[List[int]] = None,
) -> Dict:
    """
    Analyze all layers of a model for Stiefel manifold distance.

    Args:
        model_path: Path to model or HuggingFace model ID
        output_path: Optional path to save results
        device: Device for computation
        layers: Optional list of specific layers to analyze

    Returns:
        Dictionary with all results
    """
    print(f"Loading model from {model_path}...")

    # Check if it's a local path
    model_path_obj = Path(model_path)
    is_local = model_path_obj.exists() and model_path_obj.is_dir()

    load_kwargs = {
        'torch_dtype': torch.bfloat16,
        'device_map': 'cpu',  # Load to CPU first
        'trust_remote_code': True,
    }

    if is_local:
        load_kwargs['local_files_only'] = True
        print(f"  (loading from local path)")

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()

    num_layers = len(model.model.layers)
    print(f"Model has {num_layers} layers")

    if layers is None:
        layers = list(range(num_layers))

    all_results = {
        'model_path': model_path,
        'num_layers': num_layers,
        'layers': {},
    }

    for layer_idx in layers:
        print(f"Analyzing layer {layer_idx}/{num_layers-1}...")
        layer_results = analyze_layer(model, layer_idx, device)
        all_results['layers'][str(layer_idx)] = layer_results

        # Print summary for this layer
        for param_name, metrics in layer_results.items():
            if 'error' not in metrics:
                print(f"  {param_name}: spectral_dist={metrics['spectral_distance']:.4f}, "
                      f"max_sv={metrics['max_singular_value']:.4f}, "
                      f"min_sv={metrics['min_singular_value']:.4f}")

    # Compute summary statistics across all layers
    all_spectral_dists = []
    for layer_idx, layer_data in all_results['layers'].items():
        for param_name, metrics in layer_data.items():
            if 'error' not in metrics:
                all_spectral_dists.append({
                    'layer': int(layer_idx),
                    'param': param_name,
                    'spectral_distance': metrics['spectral_distance'],
                    'max_sv': metrics['max_singular_value'],
                    'min_sv': metrics['min_singular_value'],
                })

    # Sort by spectral distance
    all_spectral_dists.sort(key=lambda x: x['spectral_distance'], reverse=True)

    all_results['summary'] = {
        'total_params_analyzed': len(all_spectral_dists),
        'top_10_furthest_from_stiefel': all_spectral_dists[:10],
        'mean_spectral_distance': np.mean([x['spectral_distance'] for x in all_spectral_dists]),
        'max_spectral_distance': all_spectral_dists[0] if all_spectral_dists else None,
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return all_results


def compare_models(
    model_paths: List[str],
    model_names: List[str],
    output_path: Optional[str] = None,
    device: str = 'cuda',
) -> Dict:
    """
    Compare Stiefel distances across multiple models (e.g., SFT vs SDFT).
    """
    all_results = {}

    for model_path, model_name in zip(model_paths, model_names):
        print(f"\n{'='*60}")
        print(f"Analyzing {model_name}")
        print('='*60)
        results = analyze_model(model_path, device=device)
        all_results[model_name] = results

    # Comparison summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print('='*60)

    for model_name, results in all_results.items():
        summary = results['summary']
        print(f"\n{model_name}:")
        print(f"  Mean spectral distance: {summary['mean_spectral_distance']:.4f}")
        if summary['max_spectral_distance']:
            max_info = summary['max_spectral_distance']
            print(f"  Max spectral distance: {max_info['spectral_distance']:.4f} "
                  f"(layer {max_info['layer']}, {max_info['param']})")

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nComparison saved to {output_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Measure distance from weight matrices to Stiefel manifold"
    )
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to model or HuggingFace model ID'
    )
    parser.add_argument(
        '--output_path', type=str, default=None,
        help='Path to save results JSON'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device for computation (cuda or cpu)'
    )
    parser.add_argument(
        '--layers', type=int, nargs='+', default=None,
        help='Specific layers to analyze (default: all)'
    )

    args = parser.parse_args()

    results = analyze_model(
        model_path=args.model_path,
        output_path=args.output_path,
        device=args.device,
        layers=args.layers,
    )

    # Print final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    summary = results['summary']
    print(f"Total parameters analyzed: {summary['total_params_analyzed']}")
    print(f"Mean spectral distance to Stiefel: {summary['mean_spectral_distance']:.4f}")
    print(f"\nTop 5 furthest from Stiefel manifold:")
    for i, item in enumerate(summary['top_10_furthest_from_stiefel'][:5]):
        print(f"  {i+1}. Layer {item['layer']}, {item['param']}: "
              f"dist={item['spectral_distance']:.4f} "
              f"(σ_max={item['max_sv']:.4f}, σ_min={item['min_sv']:.4f})")


if __name__ == '__main__':
    main()
