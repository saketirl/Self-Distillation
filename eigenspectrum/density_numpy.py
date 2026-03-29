"""NumPy implementation of density estimation from Lanczos output."""

import numpy as np
from typing import Tuple, Optional
import math


def eigv_to_density(
    eig_vals: np.ndarray,
    all_weights: np.ndarray = None,
    grids: np.ndarray = None,
    grid_len: int = 10000,
    sigma_squared: float = None,
    grid_expand: float = 1e-2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute smoothed spectral density from eigenvalues.

    Convolves eigenvalues with a Gaussian kernel, weighting by all_weights.

    Args:
        eig_vals: Array of shape [num_draws, order].
        all_weights: Array of shape [num_draws, order]. If None, uniform weights.
        grids: Array of shape [grid_len] for density evaluation points.
        grid_len: Number of grid points if grids is None.
        sigma_squared: Smoothing parameter. If None, auto-computed.
        grid_expand: Padding beyond min/max eigenvalues.

    Returns:
        density: Array of shape [grid_len].
        grids: Array of shape [grid_len].
    """
    if all_weights is None:
        all_weights = np.ones(eig_vals.shape) / float(eig_vals.shape[1])

    num_draws = eig_vals.shape[0]

    lambda_max = np.nanmean(np.max(eig_vals, axis=1)) + grid_expand
    lambda_min = np.nanmean(np.min(eig_vals, axis=1)) - grid_expand

    if grids is None:
        grids = np.linspace(lambda_min, lambda_max, num=grid_len)

    grid_len = grids.shape[0]
    if sigma_squared is None:
        sigma = 1e-5 * max(1, (lambda_max - lambda_min))
    else:
        sigma = sigma_squared * max(1, (lambda_max - lambda_min))

    density_each_draw = np.zeros((num_draws, grid_len))

    for i in range(num_draws):
        for j in range(grid_len):
            x = grids[j]
            vals = _kernel(eig_vals[i, :], x, sigma)
            density_each_draw[i, j] = np.sum(vals * all_weights[i, :])

    density = np.nanmean(density_each_draw, axis=0)
    norm_fact = np.sum(density) * (grids[1] - grids[0])
    if norm_fact > 0:
        density = density / norm_fact

    return density, grids


def tridiag_to_eigv(tridiag_list: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract eigenvalues and weights from tridiagonal matrices.

    Args:
        tridiag_list: Array of shape [num_draws, order, order].

    Returns:
        eig_vals: Array of shape [num_draws, order].
        all_weights: Array of shape [num_draws, order].
    """
    num_draws = len(tridiag_list)
    num_lanczos = tridiag_list[0].shape[0]

    eig_vals = np.zeros((num_draws, num_lanczos))
    all_weights = np.zeros((num_draws, num_lanczos))

    for i in range(num_draws):
        nodes, evecs = np.linalg.eigh(tridiag_list[i])
        index = np.argsort(nodes)
        nodes = nodes[index]
        evecs = evecs[:, index]
        eig_vals[i, :] = nodes
        all_weights[i, :] = evecs[0] ** 2

    return eig_vals, all_weights


def tridiag_to_density(
    tridiag_list: np.ndarray,
    sigma_squared: float = 1e-5,
    grid_len: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate smoothed density from Lanczos output.

    Args:
        tridiag_list: Array of shape [num_draws, order, order].
        sigma_squared: Smoothing parameter.
        grid_len: Density grid resolution.

    Returns:
        density: Array of shape [grid_len].
        grids: Array of shape [grid_len].
    """
    eig_vals, all_weights = tridiag_to_eigv(tridiag_list)
    density, grids = eigv_to_density(
        eig_vals, all_weights, grid_len=grid_len, sigma_squared=sigma_squared
    )
    return density, grids


def _kernel(x: np.ndarray, x0: float, variance: float) -> np.ndarray:
    """Gaussian kernel evaluation."""
    coeff = 1.0 / np.sqrt(2 * math.pi * variance)
    val = -(x0 - x) ** 2 / (2.0 * variance)
    return coeff * np.exp(val)


def effective_rank_from_eigv(eig_vals: np.ndarray, eps: float = 1e-8) -> float:
    """Compute effective rank from eigenvalues.

    Uses entropy-based definition: r_eff = exp(H(p)) where p_i = |λ_i| / Σ|λ_j|.

    Args:
        eig_vals: Array of shape [num_draws, order].
        eps: Small constant for numerical stability.

    Returns:
        Average effective rank over draws.
    """
    abs_vals = np.abs(eig_vals)
    p = abs_vals / (np.sum(abs_vals, axis=1, keepdims=True) + eps)
    entropy = -np.sum(np.where(p > eps, p * np.log(p), 0.0), axis=1)
    ranks = np.exp(entropy)
    return float(np.mean(ranks))


def tridiag_to_density_and_erank(
    tridiag_list: np.ndarray,
    sigma_squared: float = 1e-5,
    grid_len: int = 10000,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute density and effective rank from tridiagonal matrices.

    Args:
        tridiag_list: Array of shape [num_draws, order, order].
        sigma_squared: Smoothing parameter.
        grid_len: Density grid resolution.

    Returns:
        density: Array of shape [grid_len].
        grids: Array of shape [grid_len].
        effective_rank: Scalar.
    """
    eig_vals, all_weights = tridiag_to_eigv(tridiag_list)
    density, grids = eigv_to_density(
        eig_vals, all_weights, grid_len=grid_len, sigma_squared=sigma_squared
    )
    effective_rank = effective_rank_from_eigv(eig_vals)
    return density, grids, effective_rank


# Vectorized version for better performance
def eigv_to_density_fast(
    eig_vals: np.ndarray,
    all_weights: np.ndarray = None,
    grids: np.ndarray = None,
    grid_len: int = 10000,
    sigma_squared: float = None,
    grid_expand: float = 1e-2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized density estimation (faster for large grid_len).

    Same interface as eigv_to_density but uses broadcasting.
    """
    if all_weights is None:
        all_weights = np.ones(eig_vals.shape) / float(eig_vals.shape[1])

    num_draws = eig_vals.shape[0]

    lambda_max = np.nanmean(np.max(eig_vals, axis=1)) + grid_expand
    lambda_min = np.nanmean(np.min(eig_vals, axis=1)) - grid_expand

    if grids is None:
        grids = np.linspace(lambda_min, lambda_max, num=grid_len)

    grid_len = grids.shape[0]
    if sigma_squared is None:
        sigma = 1e-5 * max(1, (lambda_max - lambda_min))
    else:
        sigma = sigma_squared * max(1, (lambda_max - lambda_min))

    # Vectorized computation
    # eig_vals: [num_draws, order]
    # grids: [grid_len]
    # We want: for each draw, for each grid point, sum over eigenvalues

    coeff = 1.0 / np.sqrt(2 * math.pi * sigma)

    # Shape: [num_draws, order, grid_len]
    diff = eig_vals[:, :, np.newaxis] - grids[np.newaxis, np.newaxis, :]
    kernel_vals = coeff * np.exp(-diff**2 / (2.0 * sigma))

    # Weighted sum over eigenvalues: [num_draws, grid_len]
    density_each_draw = np.sum(kernel_vals * all_weights[:, :, np.newaxis], axis=1)

    # Average over draws
    density = np.nanmean(density_each_draw, axis=0)

    # Normalize
    norm_fact = np.sum(density) * (grids[1] - grids[0])
    if norm_fact > 0:
        density = density / norm_fact

    return density, grids
