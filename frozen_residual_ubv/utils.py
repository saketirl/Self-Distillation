"""
Utility functions for UBV factorization with Riemannian optimization.

This module provides geometric primitives:
- msign: Matrix sign function via Polar Express iteration (GPU-friendly, no SVD)
- eigenvalue_clamp: Enforce spectrum floor on symmetric matrices
- stiefel_retraction_modula: Closed-form Stiefel retraction from Modula
- polar_retraction: Fallback polar retraction

References:
    - Buchanan et al., "Polar Express" https://arxiv.org/abs/2505.16932
    - Bernstein et al., "Modula: Manifold Optimization via Dual Ascent"
      https://docs.modula.systems/algorithms/manifold/stiefel/
    - Mishra et al., "R3MC: A Riemannian three-factor algorithm for low-rank
      matrix completion," CDC 2014.
"""

from typing import Optional, Tuple
import torch
from torch import Tensor

# Polar Express polynomial coefficients (a, b, c) per iteration step.
# Reference: https://github.com/sdbuch/thinky-manifolds/blob/main/src/msign.py
_PE_ABC: list[tuple[float, float, float]] = [
    (8.28721201814563,   -23.595886519098837, 17.300387312530933),
    (4.107059111542203,   -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946,  -2.908902115962949,  0.5518191394370137),
    (3.3184196573706015,  -2.488488024314874,  0.51004894012372),
    (2.300652019954817,   -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398,   -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479,  -1.2500016453999487, 0.3750001645474248),
    (1.875,               -1.25,               0.375),
]
# Safety-scaled versions of all but the last step (divide by 1.01^{1,3,5})
_PE_ABC_STABLE: list[tuple[float, float, float]] = [
    (a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in _PE_ABC[:-1]
] + [_PE_ABC[-1]]

# ---------------------------------------------------------------------------
# msign call statistics — reset per optimizer step so each optimizer sees only
# its own msign calls (CompMuon resets, reads; then FPMuon resets, reads).
# ---------------------------------------------------------------------------
_msign_stats: dict = {
    "total":      0,   # total msign calls this step
    "nan_input":  0,   # input was non-finite (returned identity fallback)
    "ns_diverge": 0,   # Newton-Schulz diverged mid-iteration (returned QR fallback)
    "nan_output": 0,   # output non-finite despite all guards
}


def reset_msign_stats() -> None:
    _msign_stats["total"]      = 0
    _msign_stats["nan_input"]  = 0
    _msign_stats["ns_diverge"] = 0
    _msign_stats["nan_output"] = 0


def get_msign_stats() -> dict:
    return dict(_msign_stats)


def msign(X: Tensor, ns_steps: int = 8, eps: float = 1e-7) -> Tensor:
    """
    Compute the matrix sign (polar factor) via the Polar Express algorithm.

    For X = U S V^T, returns msign(X) = U V^T.

    Uses the optimized quintic polynomial iteration from Buchanan et al.
    (https://arxiv.org/abs/2505.16932), which converges faster per step than
    the standard cubic Newton-Schulz iteration. The iteration runs in bfloat16
    and returns float32.

    Args:
        X:        Input matrix, shape (m, n).
        ns_steps: Number of Polar Express iterations (5 is typically enough).
        eps:      Unused; kept for API compatibility with callers.

    Returns:
        Polar factor of X, shape (m, n), same device as input.
    """
    m, n = X.shape
    dtype = X.dtype
    _msign_stats["total"] += 1

    # Check for NaN/Inf before entering the iteration
    if not torch.isfinite(X).all():
        _msign_stats["nan_input"] += 1
        result = torch.zeros(m, n, device=X.device, dtype=dtype)
        k = min(m, n)
        result[:k, :k] = torch.eye(k, device=X.device, dtype=dtype)
        return result

    # Polar Express operates on a wide matrix (rows ≤ cols) so that
    # S = x @ xᵀ is the smaller (rows×rows) Gram matrix.
    transposed = m > n
    x = X.float()
    if transposed:
        x = x.T   # now shape (n, m) with n ≤ m

    # Scale by the exact spectral norm (largest singular value) so σ_max ≈ 1.
    # PE is more sensitive to accurate initial scaling than the cubic NS iteration.
    spectral_norm = torch.linalg.matrix_norm(x, ord=2).item()
    scale = max(spectral_norm * 1.1, eps)
    x = x / scale

    for step in range(ns_steps):
        a, b, c = _PE_ABC_STABLE[step] if step < len(_PE_ABC_STABLE) else _PE_ABC_STABLE[-1]
        s = x @ x.T                        # (r, r) where r = min(m, n)
        y = c * s
        y.diagonal().add_(b)               # y = b*I + c*S
        y = y @ s                          # y = b*S + c*S²
        y.diagonal().add_(a)               # y = a*I + b*S + c*S²
        x = y @ x                          # x = (aI + bS + cS²) x

    if transposed:
        x = x.T

    result = torch.nan_to_num(x).to(dtype)

    if not torch.isfinite(result).all():
        _msign_stats["nan_output"] += 1

    return result


def eigenvalue_clamp(
    B: Tensor,
    lambda_min: float = 1e-4,
    symmetrize: bool = True
) -> Tensor:
    """
    Clamp eigenvalues of a symmetric matrix to enforce a spectrum floor.

    Given a symmetric (or nearly symmetric) matrix B, this function:
    1. Symmetrizes B if requested: B <- (B + B^T) / 2
    2. Eigendecomposes: B = Q @ diag(lambda) @ Q^T
    3. Clamps eigenvalues: lambda_i <- max(lambda_i, lambda_min)
    4. Reconstructs: B_clamped = Q @ diag(lambda_clamped) @ Q^T

    This ensures B remains in S_{++} (symmetric positive definite) with
    eigenvalues bounded below, preventing spectral collapse during training.

    Args:
        B: Symmetric matrix of shape (r, r).
        lambda_min: Minimum allowed eigenvalue (spectrum floor).
        symmetrize: If True, symmetrize B before eigendecomposition.

    Returns:
        SPD matrix with all eigenvalues >= lambda_min, shape (r, r).

    References:
        He et al., "Spectral Collapse Drives Loss of Plasticity in Deep
        Continual Learning," arXiv:2509.22335.
    """
    # Store original dtype for bf16 support
    original_dtype = B.dtype
    device = B.device
    r = B.shape[0]

    # Cast to float32 for numerical stability in eigendecomposition
    B_f32 = B.float()

    # Handle NaN/Inf by falling back to identity-like matrix
    if not torch.isfinite(B_f32).all():
        # Return a safe default: identity scaled by lambda_min
        return (torch.eye(r, device=device, dtype=original_dtype) * lambda_min)

    # Symmetrize to handle numerical drift
    if symmetrize:
        B_f32 = (B_f32 + B_f32.T) / 2

    # Add small regularization to diagonal for numerical stability
    B_f32 = B_f32 + torch.eye(r, device=device, dtype=torch.float32) * 1e-8

    # Eigendecomposition of symmetric matrix
    # eigenvalues are real, eigenvectors are orthonormal
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(B_f32)
    except RuntimeError:
        # Fallback: if eigh fails, return a safe default
        return (torch.eye(r, device=device, dtype=original_dtype) * lambda_min)

    # Clamp eigenvalues to enforce spectrum floor
    eigenvalues_clamped = torch.clamp(eigenvalues, min=lambda_min)

    # Reconstruct: B = Q @ diag(lambda) @ Q^T
    B_clamped = eigenvectors @ torch.diag(eigenvalues_clamped) @ eigenvectors.T

    # Final symmetrization to ensure perfect symmetry
    B_clamped = (B_clamped + B_clamped.T) / 2

    # Cast back to original dtype
    return B_clamped.to(original_dtype)


def stiefel_retraction_modula(
    U: Tensor,
    A: Tensor,
    eta: float,
    ns_steps: int = 5
) -> Tensor:
    """
    Stiefel retraction using polar projection.

    Given U on the Stiefel manifold St(r, m) and a tangent direction A,
    this computes a retraction that maps (U, eta*A) back to the manifold
    using the polar decomposition.

    The polar retraction is:
        U_new = polar(U + eta * A) = (U + eta * A) @ (A_new^T A_new)^{-1/2}

    We compute this via msign (Newton-Schulz iteration on U + eta*A).

    Args:
        U: Current point on Stiefel manifold, shape (m, r) with U^T U = I.
        A: Tangent vector, shape (m, r).
        eta: Step size (learning rate).
        ns_steps: Newton-Schulz iterations.

    Returns:
        Retracted point on Stiefel manifold, shape (m, r).

    References:
        Modula documentation: https://docs.modula.systems/algorithms/manifold/stiefel/
    """
    # Take the step in ambient space
    U_stepped = U + eta * A

    # Check for NaN/Inf
    if not torch.isfinite(U_stepped).all():
        return U  # Fall back to no update

    # Project back to Stiefel manifold via polar retraction
    # This is more reliable than the closed-form Modula formula
    U_new = msign(U_stepped, ns_steps=ns_steps)

    # Verify orthonormality and fall back to QR if needed
    if not torch.isfinite(U_new).all():
        try:
            U_new, _ = torch.linalg.qr(U_stepped)
        except RuntimeError:
            return U  # Fall back to no update

    return U_new


def polar_retraction(U: Tensor, ns_steps: int = 5) -> Tensor:
    """
    Polar retraction to Stiefel manifold via matrix sign function.

    Given a matrix U that's approximately on the Stiefel manifold (columns
    approximately orthonormal), this projects it exactly onto St(r, m) by
    computing the polar factor:

        U_polar = U @ (U^T U)^{-1/2} = msign(U)

    This is a fallback when Modula's closed-form retraction is insufficient.

    Args:
        U: Matrix to project, shape (m, r).
        ns_steps: Newton-Schulz iterations for msign.

    Returns:
        Orthonormal matrix on Stiefel manifold, shape (m, r).
    """
    return msign(U, ns_steps=ns_steps)


def symmetric_part(X: Tensor) -> Tensor:
    """
    Compute the symmetric part of a matrix: Sym(X) = (X + X^T) / 2.

    Args:
        X: Square matrix of shape (n, n).

    Returns:
        Symmetric matrix, shape (n, n).
    """
    return (X + X.T) / 2


def skew_symmetric_part(X: Tensor) -> Tensor:
    """
    Compute the skew-symmetric part of a matrix: Skew(X) = (X - X^T) / 2.

    Args:
        X: Square matrix of shape (n, n).

    Returns:
        Skew-symmetric matrix, shape (n, n).
    """
    return (X - X.T) / 2


def stiefel_project_tangent(U: Tensor, G: Tensor) -> Tensor:
    """
    Project a gradient G onto the tangent space of the Stiefel manifold at U.

    The tangent space at U is:
        T_U St(r, m) = {Z : U^T Z + Z^T U = 0}

    The projection is:
        proj_U(G) = G - U @ Sym(U^T G)

    Args:
        U: Point on Stiefel manifold, shape (m, r).
        G: Euclidean gradient, shape (m, r).

    Returns:
        Tangent vector at U, shape (m, r).
    """
    UtG = U.T @ G
    return G - U @ symmetric_part(UtG)


def compute_dual_variable_init(U: Tensor, G_U: Tensor) -> Tensor:
    """
    Initialize the dual variable Lambda for Stiefel dual ascent.

    From Modula: Lambda_0 = -1/4 * (U^T G_U + G_U^T U)

    This is a good starting point that makes the initial constraint violation
    approximately zero.

    Args:
        U: Point on Stiefel manifold, shape (m, r).
        G_U: Euclidean gradient w.r.t. U, shape (m, r).

    Returns:
        Initial dual variable, shape (r, r), symmetric.
    """
    UtG = U.T @ G_U
    return -0.25 * (UtG + UtG.T)


def check_stiefel_constraint(U: Tensor, tol: float = 1e-5) -> Tuple[bool, float]:
    """
    Check if U satisfies the Stiefel constraint U^T U = I.

    Args:
        U: Matrix to check, shape (m, r).
        tol: Tolerance for constraint violation.

    Returns:
        Tuple of (is_satisfied, violation_norm) where violation_norm = ||U^T U - I||_F.
    """
    r = U.shape[1]
    I_r = torch.eye(r, device=U.device, dtype=U.dtype)
    violation = U.T @ U - I_r
    violation_norm = torch.linalg.norm(violation, 'fro').item()
    return violation_norm < tol, violation_norm


def check_spd_constraint(B: Tensor, lambda_min: float = 0.0) -> Tuple[bool, float]:
    """
    Check if B is symmetric positive definite with eigenvalues >= lambda_min.

    Args:
        B: Matrix to check, shape (r, r).
        lambda_min: Minimum required eigenvalue.

    Returns:
        Tuple of (is_satisfied, min_eigenvalue).
    """
    # Symmetrize and compute eigenvalues
    B_sym = (B + B.T) / 2
    eigenvalues = torch.linalg.eigvalsh(B_sym)
    min_eig = eigenvalues.min().item()
    return min_eig >= lambda_min, min_eig
