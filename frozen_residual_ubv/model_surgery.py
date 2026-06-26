"""
Model surgery utilities for converting nn.Linear layers to FactoredLinear.

This module provides utilities to walk an existing model and replace target
linear layers with rank-preserving FactoredLinear modules.

Example:
    >>> from transformers import AutoModelForCausalLM
    >>> from frozen_residual_ubv import convert_model_to_factored
    >>>
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
    >>> model = convert_model_to_factored(model, r_star=64)
    >>> # Now model uses FactoredLinear for attention and MLP layers
"""

import re
from typing import Optional, Callable, Union, Set, Tuple, Dict, List
import torch
import torch.nn as nn
from torch import Tensor

from .factored_linear import FactoredLinear


def compute_energy_rank(
    W: Tensor,
    energy_threshold: float = 0.8,
    min_rank: int = 1,
    max_rank: Optional[int] = None,
) -> int:
    """
    Compute the rank needed to capture a given fraction of spectral energy.

    The spectral energy is defined as the sum of squared singular values.
    This function finds the minimum rank r such that:
        sum(σ_1^2, ..., σ_r^2) / sum(σ_i^2) >= energy_threshold

    Args:
        W: Weight matrix of shape (out_features, in_features).
        energy_threshold: Fraction of energy to capture (0.0 to 1.0).
        min_rank: Minimum rank to return.
        max_rank: Maximum rank (defaults to min(m, n)).

    Returns:
        The computed rank.

    Example:
        >>> W = torch.randn(512, 768)
        >>> rank = compute_energy_rank(W, energy_threshold=0.9)
        >>> print(f"Rank to capture 90% energy: {rank}")
    """
    m, n = W.shape
    if max_rank is None:
        max_rank = min(m, n)

    # Cast to float32 for numerical stability
    W_f32 = W.float()

    # Compute singular values
    try:
        # Use full SVD to get all singular values
        S = torch.linalg.svdvals(W_f32)
    except RuntimeError:
        # Fall back to eigenvalue-based computation
        # For W W^T or W^T W, eigenvalues are σ^2
        if m <= n:
            gram = W_f32 @ W_f32.T
        else:
            gram = W_f32.T @ W_f32
        eigenvalues = torch.linalg.eigvalsh(gram)
        S = torch.sqrt(torch.clamp(eigenvalues.flip(0), min=0))

    # Compute energy (squared singular values)
    S_sq = S ** 2
    total_energy = S_sq.sum()

    if total_energy < 1e-10:
        return min_rank

    # Compute cumulative energy ratio
    cumsum_energy = torch.cumsum(S_sq, dim=0)
    energy_ratio = cumsum_energy / total_energy

    # Find minimum rank that captures threshold energy
    mask = energy_ratio >= energy_threshold
    if mask.any():
        rank = int(mask.nonzero()[0].item()) + 1
    else:
        rank = len(S)

    # Clamp to [min_rank, max_rank]
    rank = max(min_rank, min(rank, max_rank))

    return rank


def compute_tolerance_rank(
    W: Tensor,
    rank_tol: float = 1e-5,
    min_rank: int = 1,
    max_rank: Optional[int] = None,
) -> int:
    """
    Compute rank by counting singular values above a relative tolerance.

    Finds the number of singular values satisfying: σ_i > rank_tol * σ_max

    Args:
        W: Weight matrix of shape (out_features, in_features).
        rank_tol: Relative tolerance for singular value cutoff.
        min_rank: Minimum rank to return.
        max_rank: Maximum rank (defaults to min(m, n)).

    Returns:
        The computed rank.
    """
    m, n = W.shape
    if max_rank is None:
        max_rank = min(m, n)

    # Cast to float32 for numerical stability
    W_f32 = W.float()

    # Compute singular values
    try:
        S = torch.linalg.svdvals(W_f32)
    except RuntimeError:
        # Fallback
        if m <= n:
            gram = W_f32 @ W_f32.T
        else:
            gram = W_f32.T @ W_f32
        eigenvalues = torch.linalg.eigvalsh(gram)
        S = torch.sqrt(torch.clamp(eigenvalues.flip(0), min=0))

    # Compute threshold
    if S[0] > 0:
        threshold = rank_tol * S[0]
    else:
        threshold = rank_tol

    # Count singular values above threshold
    rank = int((S > threshold).sum().item())

    # Clamp to [min_rank, max_rank]
    rank = max(min_rank, min(rank, max_rank))

    return rank


def create_rank_fn(
    rank: Union[int, str],
    energy_threshold: float = 0.8,
    rank_tol: float = 1e-5,
    min_rank: int = 1,
    max_rank: Optional[int] = None,
) -> Callable[[str, nn.Linear], int]:
    """
    Create a rank function for convert_model_to_factored.

    Args:
        rank: Rank specification:
            - int: Fixed rank for all layers
            - "auto": Auto-detect using energy threshold
            - "full": Full rank (min(m, n))
        energy_threshold: For "auto" rank, fraction of energy to capture.
            If 0 or None, uses tolerance-based rank detection.
        rank_tol: Relative tolerance for tolerance-based rank detection.
        min_rank: Minimum rank for auto detection.
        max_rank: Maximum rank for auto detection.

    Returns:
        Callable that takes (layer_name, module) and returns rank.
    """
    if isinstance(rank, int):
        def fixed_rank_fn(name: str, module: nn.Linear) -> int:
            return min(rank, min(module.out_features, module.in_features))
        return fixed_rank_fn

    elif rank == "full":
        def full_rank_fn(name: str, module: nn.Linear) -> int:
            return min(module.out_features, module.in_features)
        return full_rank_fn

    elif rank == "auto":
        def auto_rank_fn(name: str, module: nn.Linear) -> int:
            W = module.weight.data
            layer_max = max_rank if max_rank else min(W.shape[0], W.shape[1])

            if energy_threshold and energy_threshold > 0:
                return compute_energy_rank(
                    W, energy_threshold=energy_threshold,
                    min_rank=min_rank, max_rank=layer_max
                )
            else:
                return compute_tolerance_rank(
                    W, rank_tol=rank_tol,
                    min_rank=min_rank, max_rank=layer_max
                )
        return auto_rank_fn

    else:
        raise ValueError(f"Invalid rank: {rank}. Must be int, 'auto', or 'full'")


# Default patterns for HuggingFace transformer layers to convert
DEFAULT_INCLUDE_PATTERNS = [
    r".*\.q_proj$",      # Query projection
    r".*\.k_proj$",      # Key projection
    r".*\.v_proj$",      # Value projection
    r".*\.o_proj$",      # Output projection
    r".*\.gate_proj$",   # MLP gate (for Llama-style)
    r".*\.up_proj$",     # MLP up projection
    r".*\.down_proj$",   # MLP down projection
    r".*\.fc1$",         # MLP first layer (for GPT-2 style)
    r".*\.fc2$",         # MLP second layer (for GPT-2 style)
    r".*\.c_attn$",      # Combined QKV (GPT-2)
    r".*\.c_proj$",      # Output projection (GPT-2)
    r".*\.c_fc$",        # MLP (GPT-2)
    r".*\.mlp\.dense_h_to_4h$",  # Bloom/Falcon style
    r".*\.mlp\.dense_4h_to_h$",
    r".*\.self_attn\.query_key_value$",  # Falcon combined QKV
    r".*\.self_attn\.dense$",
]

# Patterns to always exclude (embeddings, output heads, layer norms)
DEFAULT_EXCLUDE_PATTERNS = [
    r".*embed.*",        # Embeddings
    r".*lm_head.*",      # Language model head
    r".*wte$",           # Token embeddings
    r".*wpe$",           # Position embeddings
    r".*norm.*",         # Layer norms
    r".*ln_.*",          # Layer norms (alternative naming)
]


def _matches_any_pattern(name: str, patterns: list) -> bool:
    """Check if name matches any regex pattern in the list."""
    for pattern in patterns:
        if re.match(pattern, name):
            return True
    return False


def _get_parent_and_attr(model: nn.Module, name: str) -> Tuple[nn.Module, str]:
    """
    Get parent module and attribute name for a nested module path.

    Args:
        model: Root model.
        name: Dot-separated path like "transformer.h.0.attn.c_proj".

    Returns:
        Tuple of (parent_module, final_attr_name).
    """
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def convert_model_to_factored(
    model: nn.Module,
    r_star: Union[int, Callable[[str, nn.Linear], int]],
    target_pattern: Optional[Union[str, Callable[[str], bool]]] = None,
    exclude_pattern: Optional[Union[str, Callable[[str], bool]]] = None,
    freeze_residual: bool = True,
    verbose: bool = True,
) -> nn.Module:
    """
    Convert target nn.Linear layers to FactoredLinear in an existing model.

    This function walks the model tree, identifies linear layers matching
    the target patterns, and replaces them with FactoredLinear modules
    initialized from the pretrained weights.

    Args:
        model: Model to convert (modified in-place).
        r_star: Target rank for trainable component. Can be:
            - int: Use same rank for all layers
            - Callable: Function (name, module) -> int for per-layer rank
        target_pattern: Pattern for layers to convert. Can be:
            - None: Use default patterns (attention and MLP layers)
            - str: Regex pattern to match layer names
            - Callable: Function (name) -> bool to filter layers
        exclude_pattern: Pattern for layers to exclude. Can be:
            - None: Use default excludes (embeddings, heads, norms)
            - str: Regex pattern to exclude
            - Callable: Function (name) -> bool to exclude
        freeze_residual: If True, freeze W_res components.
        verbose: If True, print conversion summary.

    Returns:
        The modified model (same object, modified in-place).

    Example:
        >>> # Uniform rank across all layers
        >>> model = convert_model_to_factored(model, r_star=64)
        >>>
        >>> # Per-layer rank based on hidden dimension
        >>> def rank_fn(name, module):
        ...     return min(64, module.out_features // 4)
        >>> model = convert_model_to_factored(model, r_star=rank_fn)
    """
    # Build filter function for target layers
    if target_pattern is None:
        include_fn = lambda name: _matches_any_pattern(name, DEFAULT_INCLUDE_PATTERNS)
    elif isinstance(target_pattern, str):
        include_fn = lambda name: re.match(target_pattern, name) is not None
    else:
        include_fn = target_pattern

    # Build exclude function
    if exclude_pattern is None:
        exclude_fn = lambda name: _matches_any_pattern(name, DEFAULT_EXCLUDE_PATTERNS)
    elif isinstance(exclude_pattern, str):
        exclude_fn = lambda name: re.match(exclude_pattern, name) is not None
    else:
        exclude_fn = exclude_pattern

    # Track conversion statistics
    converted_count = 0
    skipped_count = 0
    total_original_params = 0
    total_trainable_params = 0
    total_residual_params = 0
    conversion_log = []

    # Find all nn.Linear modules
    linear_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_modules.append((name, module))

    # Convert matching modules
    for name, module in linear_modules:
        # Check if should be converted
        if not include_fn(name) or exclude_fn(name):
            skipped_count += 1
            continue

        # Determine rank for this layer
        if callable(r_star):
            layer_rank = r_star(name, module)
        else:
            layer_rank = r_star

        # Create FactoredLinear replacement
        factored = FactoredLinear(
            W_pretrained=module.weight.data,
            r_star=layer_rank,
            bias=module.bias.data if module.bias is not None else None,
            freeze_residual=freeze_residual,
        )

        # Replace in parent module
        parent, attr_name = _get_parent_and_attr(model, name)
        setattr(parent, attr_name, factored)

        # Track statistics
        original_params = module.weight.numel()
        if module.bias is not None:
            original_params += module.bias.numel()
        trainable = factored.trainable_params
        residual = factored.residual_params

        total_original_params += original_params
        total_trainable_params += trainable
        total_residual_params += residual
        converted_count += 1

        conversion_log.append({
            'name': name,
            'shape': tuple(module.weight.shape),
            'rank': layer_rank,
            'original': original_params,
            'trainable': trainable,
            'residual': residual,
        })

    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print("FactoredLinear Conversion Summary")
        print(f"{'='*60}")
        print(f"Layers converted: {converted_count}")
        print(f"Layers skipped:   {skipped_count}")
        print(f"\nParameter counts:")
        print(f"  Original (in converted layers): {total_original_params:,}")
        print(f"  Trainable after conversion:     {total_trainable_params:,}")
        print(f"  Frozen residual:                {total_residual_params:,}")
        if total_original_params > 0:
            reduction = 100 * (1 - total_trainable_params / total_original_params)
            print(f"  Trainable parameter reduction:  {reduction:.1f}%")
        print(f"\nConverted layers:")
        for entry in conversion_log[:10]:  # Show first 10
            print(f"  {entry['name']}: {entry['shape']} -> rank {entry['rank']}")
        if len(conversion_log) > 10:
            print(f"  ... and {len(conversion_log) - 10} more")
        print(f"{'='*60}\n")

    return model


def get_factored_parameters(model: nn.Module) -> dict:
    """
    Get all FactoredLinear parameters grouped by role.

    Args:
        model: Model containing FactoredLinear layers.

    Returns:
        Dictionary with keys 'U', 'B', 'V', 'bias', 'other' containing parameter lists.
    """
    params = {'U': [], 'B': [], 'V': [], 'bias': [], 'other': []}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        role = getattr(param, '_ubv_role', None)
        if role in params:
            params[role].append(param)
        else:
            params['other'].append(param)

    return params


def count_parameters(model: nn.Module) -> dict:
    """
    Count parameters in a model with FactoredLinear layers.

    Args:
        model: Model to analyze.

    Returns:
        Dictionary with parameter counts by category.
    """
    counts = {
        'U': 0, 'B': 0, 'V': 0, 'bias': 0, 'other': 0,
        'trainable': 0, 'frozen': 0, 'total': 0
    }

    for name, param in model.named_parameters():
        numel = param.numel()
        counts['total'] += numel

        if param.requires_grad:
            counts['trainable'] += numel
            role = getattr(param, '_ubv_role', 'other')
            if role in counts:
                counts[role] += numel
            else:
                counts['other'] += numel
        else:
            counts['frozen'] += numel

    # Count frozen buffers (W_res)
    for name, buffer in model.named_buffers():
        if buffer is not None:
            counts['frozen'] += buffer.numel()
            counts['total'] += buffer.numel()

    return counts


def print_factored_summary(model: nn.Module) -> None:
    """
    Print a summary of FactoredLinear layers in the model.

    Args:
        model: Model to summarize.
    """
    print("\nFactoredLinear Layer Summary:")
    print("-" * 80)
    print(f"{'Layer Name':<40} {'Shape':<15} {'Rank':<6} {'Params':<10}")
    print("-" * 80)

    for name, module in model.named_modules():
        if isinstance(module, FactoredLinear):
            shape = f"({module.out_features}, {module.in_features})"
            params = module.trainable_params
            print(f"{name:<40} {shape:<15} {module.r_star:<6} {params:<10,}")

    print("-" * 80)
    counts = count_parameters(model)
    print(f"Total trainable: {counts['trainable']:,}")
    print(f"Total frozen:    {counts['frozen']:,}")
    print()
