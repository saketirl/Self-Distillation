"""
Projected Gradient Optimizer with Frozen Residual.

Factorizes weight W = U @ B @ V.T + R where:
- U, V: Orthonormal matrices on Stiefel manifold
- B: Symmetric positive definite matrix (non-diagonal per Mishra et al. Section 3.2)
- R: Frozen residual capturing energy below threshold

The residual R is frozen during optimization. Only U, B, V are updated,
constraining learning to the principal subspace while preserving the
residual structure of the pretrained weights.

Flow:
    1. Init: W = U @ B @ V.T + R (R = W - U @ B @ V.T, frozen)
    2. Forward: y = x @ W.T (use W directly)
    3. Backward: G_W = dL/dW
    4. Project: G_U, G_B, G_V = chain_rule(G_W, U, B, V)
    5. Constrain: Manifold MUON for U/V, Riemannian SPD for B
    6. Update: U, B, V with constrained gradients
    7. Reconstruct: W = U @ B @ V.T + R
    8. Every m steps: Resync (recompute UBV from W, update R)

References:
    - Mishra et al. 2013, "Low-rank optimization with trace norm penalty"
      arXiv:1209.0430 (Section 3.2 for non-diagonal B, Section 6.3 for quotient issues)
    - Buchanan et al., "Manifold MUON" (thinky-manifolds)
"""

from typing import Optional, Dict, Any, List
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
import torch.nn as nn

from .utils import eigenvalue_clamp, msign, symmetric_part


def energy_truncated_svd(
    W: Tensor,
    energy_threshold: float = 0.9,
) -> tuple[Tensor, Tensor, Tensor, int, float]:
    """
    Compute SVD with energy-based rank truncation using randomized SVD.

    Uses torch.svd_lowrank for memory efficiency. Adaptively increases rank
    until the energy threshold is met.

    Args:
        W: Input matrix, shape (m, n)
        energy_threshold: Fraction of total energy to retain (0.0-1.0)

    Returns:
        U: Left singular vectors, shape (m, r)
        S: Singular values, shape (r,)
        V: Right singular vectors, shape (n, r)
        r: Selected rank
        captured_energy: Actual energy ratio captured (>= threshold)
    """
    W = W.float()
    m, n = W.shape
    min_dim = min(m, n)

    # Total energy (Frobenius norm squared)
    total_energy = torch.linalg.norm(W, 'fro') ** 2

    if total_energy < 1e-10:
        # Near-zero matrix: return rank-1
        U = torch.zeros(m, 1, device=W.device, dtype=W.dtype)
        U[0, 0] = 1.0
        S = torch.zeros(1, device=W.device, dtype=W.dtype)
        V = torch.zeros(n, 1, device=W.device, dtype=W.dtype)
        V[0, 0] = 1.0
        return U, S, V, 1, 1.0

    # Initial rank estimate based on threshold
    # Lower threshold -> fewer components needed
    initial_rank = max(16, int(min_dim * (1 - energy_threshold) * 2))
    initial_rank = min(initial_rank, min_dim - 1)

    # Adaptive loop: increase rank until energy threshold is met
    rank = initial_rank
    max_attempts = 5

    for attempt in range(max_attempts):
        # Use randomized SVD (memory efficient)
        U, S, V = torch.svd_lowrank(W, q=rank, niter=2)

        # Compute captured energy
        captured_energy = (S ** 2).sum() / total_energy

        if captured_energy >= energy_threshold or rank >= min_dim - 1:
            break

        # Increase rank for next attempt
        rank = min(rank * 2, min_dim - 1)

    # Find actual rank needed for threshold within computed singular values
    S_squared = S ** 2
    cumsum_energy = torch.cumsum(S_squared, dim=0)
    energy_ratio = cumsum_energy / total_energy

    mask = energy_ratio >= energy_threshold
    if mask.any():
        r = int(mask.float().argmax().item()) + 1
    else:
        r = len(S)

    r = max(1, min(r, len(S)))

    # Extract top-r components
    U = U[:, :r]
    S = S[:r]
    V = V[:, :r]

    captured_energy = energy_ratio[min(r - 1, len(energy_ratio) - 1)].item()

    return U, S, V, r, captured_energy


def spd_muon_admm(
    B: Tensor,
    G: Tensor,
    Omega: Optional[Tensor] = None,
    steps: int = 10,
    rho: float = 4.0,
    momentum: float = 0.9,
) -> tuple[Tensor, Tensor]:
    """
    DEPRECATED: This function uses affine-invariant metric which causes gradient amplification.
    Kept for backwards compatibility. Use Log-Cholesky parameterization instead.
    """
    r = B.shape[0]
    device, dtype = B.device, B.dtype
    G_sym = (G + G.T) / 2
    G_riem = B @ G_sym @ B
    if Omega is None:
        Omega = torch.zeros(r, r, device=device, dtype=dtype)
    Omega_new = momentum * Omega + (1 - momentum) * G_riem
    Omega_new = (Omega_new + Omega_new.T) / 2
    A = Omega_new
    return A, Omega_new


# =============================================================================
# Log-Cholesky Parameterization for SPD Matrix B
# =============================================================================
# Instead of optimizing B directly on the SPD manifold with affine-invariant metric
# (which causes gradient amplification by ||B||^2), we parameterize B as:
#
#   B = L @ L.T
#
# where L is lower triangular with:
#   - Strict lower triangle: unconstrained real entries (stored in A_lower)
#   - Diagonal: exp(log_diag) + spd_eps (always positive)
#
# This makes the optimization unconstrained in (A_lower, log_diag) space,
# and we can use standard Adam with gradient clipping.
# =============================================================================


def logchol_from_spd(
    B: Tensor,
    spd_eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """
    Convert SPD matrix B to Log-Cholesky coordinates.

    Args:
        B: Symmetric positive definite matrix, shape (r, r)
        spd_eps: Small constant for numerical stability

    Returns:
        A_lower: Full r x r tensor, only strict lower triangle used (zeros elsewhere)
        log_diag: r-vector of log(diag(L) - spd_eps)
    """
    r = B.shape[0]
    device, dtype = B.device, B.dtype

    # Ensure symmetry
    B_sym = (B + B.T) / 2

    # Try Cholesky decomposition
    try:
        L = torch.linalg.cholesky(B_sym)
    except RuntimeError:
        # Fallback: eigenvalue clamp then Cholesky
        B_clamped = eigenvalue_clamp(B_sym, lambda_min=spd_eps)
        L = torch.linalg.cholesky(B_clamped)

    # Extract strict lower triangle (zeros on and above diagonal)
    A_lower = torch.tril(L, diagonal=-1)

    # Extract diagonal and convert to log space
    diag_L = torch.diag(L)
    # Clamp to ensure we can take log
    diag_clamped = torch.clamp(diag_L - spd_eps, min=spd_eps)
    log_diag = torch.log(diag_clamped)

    return A_lower, log_diag


def spd_from_logchol(
    A_lower: Tensor,
    log_diag: Tensor,
    spd_eps: float = 1e-6,
    max_log_diag: float = 8.0,
    min_log_diag: float = -8.0,
) -> tuple[Tensor, Tensor]:
    """
    Reconstruct SPD matrix B from Log-Cholesky coordinates.

    Args:
        A_lower: Full r x r tensor, only strict lower triangle used
        log_diag: r-vector in log space
        spd_eps: Small constant added to diagonal
        max_log_diag: Maximum value for log_diag (prevents explosion)
        min_log_diag: Minimum value for log_diag (prevents collapse)

    Returns:
        B: Symmetric positive definite matrix
        L: Lower triangular Cholesky factor
    """
    # Clamp log_diag to prevent numerical issues
    log_diag_clamped = torch.clamp(log_diag, min=min_log_diag, max=max_log_diag)

    # Construct L: strict lower from A_lower + positive diagonal
    L = torch.tril(A_lower, diagonal=-1) + torch.diag(torch.exp(log_diag_clamped) + spd_eps)

    # B = L @ L.T (guaranteed SPD)
    B = L @ L.T

    # Ensure perfect symmetry
    B = (B + B.T) / 2

    return B, L


def spd_logchol_adam_update(
    A_lower: Tensor,
    log_diag: Tensor,
    G_B: Tensor,
    state: Dict[str, Any],
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.99,
    adam_eps: float = 1e-8,
    spd_eps: float = 1e-6,
    max_log_diag: float = 8.0,
    min_log_diag: float = -8.0,
    b_update_rms: float = 0.03,
) -> tuple[Tensor, Tensor, Tensor, Dict[str, float]]:
    """
    Update Log-Cholesky parameters using Adam optimizer.

    The gradient of loss L w.r.t. L is: grad_L = 2 * G_sym @ L
    where G_sym = (G_B + G_B.T) / 2 is the symmetrized Euclidean gradient.

    Args:
        A_lower: Current strict lower triangle params, shape (r, r)
        log_diag: Current log-diagonal params, shape (r,)
        G_B: Euclidean gradient w.r.t. B = U.T @ G @ V, shape (r, r)
        state: Optimizer state dict (modified in place)
        lr: Learning rate
        beta1, beta2, adam_eps: Adam hyperparameters
        spd_eps: SPD epsilon for numerical stability
        max_log_diag, min_log_diag: Bounds for log_diag
        b_update_rms: RMS threshold for update clipping

    Returns:
        B_new: Updated SPD matrix
        A_lower_new: Updated strict lower params
        log_diag_new: Updated log-diagonal params
        info: Dict with logging info
    """
    r = A_lower.shape[0]
    device, dtype = A_lower.device, A_lower.dtype

    # Symmetrize gradient
    G_sym = (G_B + G_B.T) / 2

    # Reconstruct current L from params
    log_diag_clamped = torch.clamp(log_diag, min=min_log_diag, max=max_log_diag)
    diag_vals = torch.exp(log_diag_clamped) + spd_eps
    L = torch.tril(A_lower, diagonal=-1) + torch.diag(diag_vals)

    # Gradient w.r.t. L: grad_L = 2 * G_sym @ L
    # This comes from: B = L @ L.T, so dL/dB uses chain rule
    grad_L = 2 * G_sym @ L

    # Extract gradients for our parameters:
    # grad_A = strict lower triangle of grad_L
    grad_A = torch.tril(grad_L, diagonal=-1)

    # grad_d = diagonal of grad_L * diag(L) (chain rule for exp parameterization)
    # Since L_ii = exp(log_diag_i) + spd_eps, dL_ii/d(log_diag_i) = exp(log_diag_i) = diag_vals - spd_eps
    grad_d = torch.diag(grad_L) * (diag_vals - spd_eps)

    # Flatten strict lower triangle entries for Adam
    # Number of strict lower entries: r*(r-1)/2
    strict_lower_mask = torch.tril(torch.ones(r, r, device=device, dtype=torch.bool), diagonal=-1)
    flat_A = A_lower[strict_lower_mask]
    flat_grad_A = grad_A[strict_lower_mask]

    # Concatenate into single vector: [flat_A, log_diag]
    params_vec = torch.cat([flat_A, log_diag])
    grad_vec = torch.cat([flat_grad_A, grad_d])

    # Initialize Adam state if needed
    if 'B_adam_m' not in state:
        state['B_adam_m'] = torch.zeros_like(params_vec)
        state['B_adam_v'] = torch.zeros_like(params_vec)
        state['B_adam_t'] = 0

    # Get Adam state
    m = state['B_adam_m']
    v = state['B_adam_v']
    t = state['B_adam_t'] + 1

    # Adam update
    m_new = beta1 * m + (1 - beta1) * grad_vec
    v_new = beta2 * v + (1 - beta2) * (grad_vec ** 2)

    # Bias correction
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)

    # Adam step (before clipping)
    step_vec = m_hat / (torch.sqrt(v_hat) + adam_eps)

    # Compute step norm before clipping
    step_norm_pre = torch.linalg.norm(step_vec).item()

    # Clip by RMS threshold
    numel = step_vec.numel()
    max_norm = b_update_rms * (numel ** 0.5)
    step_norm = torch.linalg.norm(step_vec)
    clipped = False
    if step_norm > max_norm:
        step_vec = step_vec * (max_norm / step_norm)
        clipped = True

    step_norm_post = torch.linalg.norm(step_vec).item()

    # Update state
    state['B_adam_m'] = m_new
    state['B_adam_v'] = v_new
    state['B_adam_t'] = t

    # Apply update
    params_new = params_vec - lr * step_vec

    # Unpack back to A_lower and log_diag
    n_lower = flat_A.numel()
    flat_A_new = params_new[:n_lower]
    log_diag_new = params_new[n_lower:]

    # Reconstruct A_lower matrix
    A_lower_new = torch.zeros_like(A_lower)
    A_lower_new[strict_lower_mask] = flat_A_new

    # Clamp log_diag
    log_diag_new = torch.clamp(log_diag_new, min=min_log_diag, max=max_log_diag)

    # Reconstruct B
    B_new, _ = spd_from_logchol(A_lower_new, log_diag_new, spd_eps, max_log_diag, min_log_diag)

    # Logging info
    info = {
        'B_step_norm': step_norm_pre,
        'B_step_norm_clipped': step_norm_post,
        'B_logdiag_min': log_diag_new.min().item(),
        'B_logdiag_max': log_diag_new.max().item(),
        'B_clipped': 1.0 if clipped else 0.0,
        'B_grad_norm': torch.linalg.norm(grad_vec).item(),
    }

    return B_new, A_lower_new, log_diag_new, info


def manifold_muon_admm(
    W: Tensor,
    G: Tensor,
    steps: int = 10,
    rho: float = 4.0,
) -> Tensor:
    """
    Compute descent direction on Stiefel manifold via ADMM-based Manifold MUON.

    Based on: https://sdbuchanan.com/blog/manifold-muon/

    Uses ADMM to solve the dual problem more efficiently than simple dual ascent.
    Returns the update direction A (after msign), ready to be used for the step.

    Args:
        W: Current point on Stiefel manifold, shape (m, r)
        G: Euclidean gradient, shape (m, r)
        steps: Number of ADMM iterations
        rho: ADMM penalty parameter

    Returns:
        A: Descent direction on Stiefel manifold (after msign)
    """
    # Ensure that W and G are both tall matrices
    should_transpose = W.shape[0] < W.shape[1]
    if should_transpose:
        W = W.T
        G = G.T

    # Initialize the Lagrangian, slack, and dual variable
    Lambda = -0.25 * (W.T @ G + G.T @ W)
    X = G + 2 * W @ Lambda
    Omega = torch.zeros_like(X)

    # Solve the dual problem with ADMM to find the update direction A
    for step in range(steps):
        # Update for Lambda (orthonormal least-squares solve)
        P = W.mT @ (1 / rho * Omega + X - G)
        Lambda_upd = 0.25 * (P + P.mT)

        # Update for X (singular value thresholding)
        B = G + 2 * W @ Lambda_upd - 1 / rho * Omega
        eye = torch.eye(B.shape[1], device=B.device, dtype=B.dtype)
        P_pos = 0.5 * (eye + msign(B.mT @ B - 1 / rho**2 * eye))
        X_upd = (B - 1 / rho * msign(B)) @ P_pos

        # Update for Omega (dual ascent)
        Omega_upd = Omega + rho * (X_upd - 2 * W @ Lambda_upd - G)
        Lambda, X, Omega = Lambda_upd, X_upd, Omega_upd

    # Calculate A from final ADMM solution
    # (at convergence, G + 2 * W @ Lambda ≈ X)
    A = msign(G + 2 * W @ Lambda)

    # Restore the shape of the solution and return
    return A.T if should_transpose else A


# =============================================================================
# Stable Stiefel Manifold Updates for U and V
# =============================================================================
# The Manifold MUON direction from msign(G + 2*W@Lambda) may not be exactly
# tangent at W, causing instability. These helpers provide a stable alternative:
#
#   1. stiefel_project_tangent: Project gradient to tangent space
#   2. normalize_update: RMS-based clipping to bound step size
#   3. stiefel_qr_retraction: QR-based retraction (stable, always orthonormal)
#   4. stiefel_cayley_retraction: Cayley retraction (optional, more expensive)
# =============================================================================


def stiefel_project_tangent(W: Tensor, G: Tensor) -> Tensor:
    """
    Project Euclidean gradient G onto tangent space at Stiefel point W.

    The tangent space at W consists of matrices A such that:
        W.T @ A + A.T @ W = 0  (skew-symmetric)

    The projection is:
        A = G - W @ Sym(W.T @ G)

    where Sym(X) = 0.5 * (X + X.T)

    Args:
        W: Point on Stiefel manifold, shape (m, r), W.T @ W = I_r
        G: Euclidean gradient, shape (m, r)

    Returns:
        A: Tangent vector at W, shape (m, r)
    """
    # Symmetric part of W.T @ G
    WtG = W.T @ G
    Sym = 0.5 * (WtG + WtG.T)

    # Project to tangent space
    A = G - W @ Sym

    return A


def normalize_update(
    A: Tensor,
    max_rms: float = 0.03,
    eps: float = 1e-8,
) -> tuple[Tensor, Dict[str, float]]:
    """
    Normalize/clip update tensor A by RMS threshold.

    Clips the update so that:
        ||A||_F <= max_rms * sqrt(A.numel())

    This bounds the per-element RMS of the update.

    Args:
        A: Update tensor, any shape
        max_rms: Maximum RMS value per element
        eps: Small constant for numerical stability

    Returns:
        A_normalized: Clipped update tensor
        info: Dict with raw_norm, clipped_norm, was_clipped
    """
    raw_norm = torch.linalg.norm(A, 'fro').item()
    max_norm = max_rms * (A.numel() ** 0.5)

    was_clipped = raw_norm > max_norm
    if was_clipped:
        scale = max_norm / (raw_norm + eps)
        A_normalized = A * scale
        clipped_norm = max_norm
    else:
        A_normalized = A
        clipped_norm = raw_norm

    info = {
        'raw_norm': raw_norm,
        'clipped_norm': clipped_norm,
        'was_clipped': 1.0 if was_clipped else 0.0,
    }

    return A_normalized, info


def stiefel_qr_retraction(W: Tensor) -> Tensor:
    """
    Retract matrix W back to Stiefel manifold using QR decomposition.

    This is more stable than polar retraction (msign) for ill-conditioned matrices.

    The QR retraction is:
        Q, R = qr(W)
        Fix sign ambiguity: Q = Q * sign(diag(R))

    Args:
        W: Matrix to retract, shape (m, r), should be close to Stiefel

    Returns:
        Q: Orthonormal matrix on Stiefel manifold, shape (m, r)
    """
    Q, R = torch.linalg.qr(W, mode='reduced')

    # Fix QR sign ambiguity: ensure diagonal of R is positive
    # This makes the retraction continuous
    signs = torch.sign(torch.diagonal(R))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)

    # Apply sign correction to columns of Q
    Q = Q * signs.unsqueeze(0)

    return Q


def stiefel_cayley_retraction(W: Tensor, A: Tensor, lr: float) -> Tensor:
    """
    Retract along tangent direction A using Cayley transform.

    The Cayley retraction is:
        K = A @ W.T - W @ A.T  (skew-symmetric)
        W_new = (I + 0.5*lr*K)^{-1} @ (I - 0.5*lr*K) @ W

    This is exact for small steps but requires solving a linear system.
    For large m x r matrices, QR retraction may be more efficient.

    Args:
        W: Current point on Stiefel manifold, shape (m, r)
        A: Tangent vector at W, shape (m, r)
        lr: Step size

    Returns:
        W_new: New point on Stiefel manifold, shape (m, r)
    """
    m, r = W.shape
    device, dtype = W.device, W.dtype

    # Skew-symmetric matrix K = A @ W.T - W @ A.T
    K = A @ W.T - W @ A.T

    # Identity matrix
    I = torch.eye(m, device=device, dtype=dtype)

    # Cayley transform: (I + 0.5*lr*K)^{-1} @ (I - 0.5*lr*K) @ W
    lhs = I + 0.5 * lr * K
    rhs = (I - 0.5 * lr * K) @ W

    # Solve linear system
    W_new = torch.linalg.solve(lhs, rhs)

    return W_new


class ProjectedGradientOptimizer(Optimizer):
    """
    Optimizer that projects gradients through UBV structure with frozen residual.

    For each weight matrix W, maintains:
    - U, V: Orthonormal factors on Stiefel manifold
    - B: Symmetric positive definite matrix (via Log-Cholesky parameterization)
    - R: Frozen residual (W - U @ B @ V.T)

    The factorization W = U @ B @ V.T + R is maintained throughout training.
    Only U, B, V are updated; R stays frozen to preserve pretrained structure.

    Uses stable tangent-projected gradients with QR retraction for U, V updates (default).
    Uses Log-Cholesky parameterization with Adam for B updates (stable, no gradient amplification).

    Args:
        params: Parameters to optimize (should be weight matrices)
        lr: Base learning rate
        energy_threshold: Energy threshold for rank selection (0.0-1.0)
        resync_every: Re-factor UBV from W every this many steps
        ns_steps: Newton-Schulz iterations for Stiefel retraction (only used with muon_admm)
        lambda_min: Minimum eigenvalue for B (used in init/resync fallback only)
        admm_steps: Number of ADMM iterations for manifold MUON (only used with muon_admm)
        admm_rho: ADMM penalty parameter (only used with muon_admm)
        grad_clip: Gradient clipping threshold (legacy, prefer stiefel_max_rms)
        stiefel_update: Stiefel update method ("projected_qr" or "muon_admm")
        stiefel_max_rms: RMS threshold for U, V update clipping (default 0.03)
        uv_lr_scale: Learning rate scale for U, V relative to lr (default 0.3)
        b_beta1: Adam beta1 for B optimizer (default 0.9)
        b_beta2: Adam beta2 for B optimizer (default 0.99)
        b_adam_eps: Adam epsilon for B optimizer (default 1e-8)
        b_update_rms: RMS threshold for B update clipping (default 0.03)
        b_spd_eps: Small constant for SPD numerical stability (default 1e-6)
        b_min_log_diag: Minimum log-diagonal value (default -8.0)
        b_max_log_diag: Maximum log-diagonal value (default 8.0)
        b_lr_scale: Learning rate scale for B relative to lr (default 0.3)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        energy_threshold: float = 0.9,
        resync_every: int = 100,
        ns_steps: int = 20,
        lambda_min: float = 1e-4,
        admm_steps: int = 10,
        admm_rho: float = 4.0,
        grad_clip: float = 100.0,
        # Stiefel U, V optimizer hyperparameters
        stiefel_update: str = "projected_qr",
        stiefel_max_rms: float = 0.03,
        uv_lr_scale: float = 0.3,
        # Log-Cholesky B optimizer hyperparameters
        b_beta1: float = 0.9,
        b_beta2: float = 0.99,
        b_adam_eps: float = 1e-8,
        b_update_rms: float = 0.03,
        b_spd_eps: float = 1e-6,
        b_min_log_diag: float = -8.0,
        b_max_log_diag: float = 8.0,
        b_lr_scale: float = 0.3,
    ):
        defaults = dict(
            lr=lr,
            energy_threshold=energy_threshold,
            resync_every=resync_every,
            ns_steps=ns_steps,
            lambda_min=lambda_min,
            admm_steps=admm_steps,
            admm_rho=admm_rho,
            grad_clip=grad_clip,
            # Stiefel U, V optimizer
            stiefel_update=stiefel_update,
            stiefel_max_rms=stiefel_max_rms,
            uv_lr_scale=uv_lr_scale,
            # Log-Cholesky B optimizer
            b_beta1=b_beta1,
            b_beta2=b_beta2,
            b_adam_eps=b_adam_eps,
            b_update_rms=b_update_rms,
            b_spd_eps=b_spd_eps,
            b_min_log_diag=b_min_log_diag,
            b_max_log_diag=b_max_log_diag,
            b_lr_scale=b_lr_scale,
        )
        super().__init__(params, defaults)

    def _init_ubv(self, p: Tensor, state: Dict[str, Any], energy_threshold: float):
        """
        Initialize UBV factors from weight matrix via SVD with energy-based rank.

        Per Mishra et al. 2013 (arXiv:1209.0430):
        - Section 6.3: Diagonal B leads to bad optimization due to quotient structure
        - Section 3.2: Apply orthogonal transformation (U, S, V) → (U@O, O.T@S@O, V@O)

        This transformation:
        1. Keeps U, V on Stiefel manifold (U@O is still orthonormal)
        2. Makes B = O.T @ diag(S) @ O non-diagonal but symmetric positive definite
        3. Maintains W = U @ B @ V.T (same product)

        Rank is determined purely by energy threshold: find smallest r such that
        sum(S[:r]^2) / sum(S^2) >= energy_threshold. No heuristic caps.
        """
        W = p.data.float()
        device = W.device
        dtype = W.dtype

        # Energy-based SVD truncation (no heuristics, purely energy-based)
        U_svd, S_r, V_svd, r, captured_energy = energy_truncated_svd(W, energy_threshold)

        # Re-orthogonalize via QR for numerical stability
        U_svd, _ = torch.linalg.qr(U_svd)
        V_svd, _ = torch.linalg.qr(V_svd)

        # Generate random orthogonal matrix O via QR decomposition
        # This is well-conditioned and avoids the diagonal B issue (Section 3.2)
        random_matrix = torch.randn(r, r, device=device, dtype=dtype)
        O, _ = torch.linalg.qr(random_matrix)

        # Apply transformation: (U, S, V) → (U@O, O.T@S@O, V@O)
        U = U_svd @ O  # Still on Stiefel (O is orthogonal)
        V = V_svd @ O  # Still on Stiefel

        # B = O.T @ diag(S) @ O (non-diagonal, symmetric positive definite)
        S_diag = torch.diag(S_r)
        B = O.T @ S_diag @ O

        # B is already symmetric by construction, but ensure numerical symmetry
        B = symmetric_part(B)

        # Ensure positive definite (eigenvalues are the singular values, all positive)
        B = eigenvalue_clamp(B, lambda_min=1e-6)

        # Convert B to Log-Cholesky parameterization
        A_lower, log_diag = logchol_from_spd(B, spd_eps=1e-6)

        # Reconstruct B from Log-Cholesky params to ensure consistency
        B, _ = spd_from_logchol(A_lower, log_diag, spd_eps=1e-6)

        # Compute and store FROZEN residual: R = W - U @ B @ V.T
        # This captures the energy below the threshold (smaller singular values)
        # R stays frozen during optimization to preserve pretrained structure
        low_rank_approx = U @ B @ V.T
        R = W - low_rank_approx

        # Verify orthonormality at initialization
        I_r = torch.eye(r, device=device, dtype=U.dtype)
        U_orth_err = torch.linalg.norm(U.T @ U - I_r).item()
        V_orth_err = torch.linalg.norm(V.T @ V - I_r).item()

        # These should be very small (< 1e-5) if initialization is correct
        if U_orth_err > 0.01 or V_orth_err > 0.01:
            print(f"[ProjectedGradient] WARNING: High orthonormality error at init!")
            print(f"  U shape: {U.shape}, dtype: {U.dtype}")
            print(f"  ||U.T @ U - I||: {U_orth_err:.6f}")
            print(f"  ||V.T @ V - I||: {V_orth_err:.6f}")
            print(f"  U_svd orthonormality: {torch.linalg.norm(U_svd.T @ U_svd - I_r).item():.6f}")
            print(f"  O orthonormality: {torch.linalg.norm(O.T @ O - I_r).item():.6f}")

        state['U'] = U
        state['B'] = B
        state['V'] = V
        state['R'] = R  # Frozen residual
        state['r'] = r
        state['energy_threshold'] = energy_threshold
        state['step'] = 0

        # Log-Cholesky parameters for B
        state['B_A_lower'] = A_lower
        state['B_log_diag'] = log_diag

    def _resync_ubv(self, p: Tensor, state: Dict[str, Any]):
        """
        Re-factor W into UBV with Procrustes alignment and energy-based rank.

        Per Mishra et al. 2013 (Section 3.2): Apply orthogonal transformation
        to make B non-diagonal while keeping U, V on Stiefel.

        Rank is determined purely by energy threshold. No heuristics.
        """
        W = p.data.float()
        device = W.device
        dtype = W.dtype

        energy_threshold = state.get('energy_threshold', 0.9)
        r_old = state['r']
        U_old = state['U']
        V_old = state['V']

        # Energy-based SVD truncation (no heuristics)
        U_svd, S_r, V_svd, r_new, _ = energy_truncated_svd(W, energy_threshold)

        # Re-orthogonalize via QR for numerical stability
        U_svd, _ = torch.linalg.qr(U_svd)
        V_svd, _ = torch.linalg.qr(V_svd)

        # Generate random orthogonal matrix O (Section 3.2)
        random_matrix = torch.randn(r_new, r_new, device=device, dtype=dtype)
        O, _ = torch.linalg.qr(random_matrix)

        # Apply transformation: (U, S, V) → (U@O, O.T@S@O, V@O)
        U_new = U_svd @ O
        V_new = V_svd @ O

        # Procrustes alignment for smooth transition (if ranks match)
        if r_new == r_old:
            # Align new U to old U
            M_U = U_old.T @ U_new
            Ua, Sa, Vta = torch.linalg.svd(M_U)
            R_U = Vta.T @ Ua.T
            U_aligned = U_new @ R_U

            # Align new V to old V
            M_V = V_old.T @ V_new
            Ua, Sa, Vta = torch.linalg.svd(M_V)
            R_V = Vta.T @ Ua.T
            V_aligned = V_new @ R_V

            # Recompute B with aligned factors
            # Note: Since we applied separate rotations R_U and R_V, we need to
            # compute B from the aligned factors to maintain W = U @ B @ V.T
            # B = U_aligned.T @ W @ V_aligned
            # But we can also derive it: B_new = R_U.T @ (O.T @ diag(S) @ O) @ R_V
            # For simplicity and numerical stability, compute directly:
            B_aligned = U_aligned.T @ W @ V_aligned
        else:
            # Rank changed, use the O-transformed factors directly
            U_aligned = U_new
            V_aligned = V_new
            # B = O.T @ diag(S) @ O (non-diagonal, symmetric positive definite)
            S_diag = torch.diag(S_r)
            B_aligned = O.T @ S_diag @ O

        B_aligned = symmetric_part(B_aligned)
        B_aligned = eigenvalue_clamp(B_aligned, lambda_min=1e-6)

        # Convert B to Log-Cholesky parameterization
        A_lower, log_diag = logchol_from_spd(B_aligned, spd_eps=1e-6)

        # Reconstruct B from Log-Cholesky params to ensure consistency
        B_aligned, _ = spd_from_logchol(A_lower, log_diag, spd_eps=1e-6)

        # Recompute frozen residual with new factorization
        low_rank_approx = U_aligned @ B_aligned @ V_aligned.T
        R_new = W - low_rank_approx

        state['U'] = U_aligned
        state['B'] = B_aligned
        state['V'] = V_aligned
        state['R'] = R_new  # Updated frozen residual
        state['r'] = r_new

        # Log-Cholesky parameters for B
        state['B_A_lower'] = A_lower
        state['B_log_diag'] = log_diag

        # Reset B Adam state (rank may have changed)
        state.pop('B_adam_m', None)
        state.pop('B_adam_v', None)
        state.pop('B_adam_t', None)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform optimization step with frozen residual.

        For each parameter W = U @ B @ V.T + R:
        1. Compute gradients G_U, G_B, G_V from chain rule
        2. Apply Riemannian constraints (ADMM Manifold MUON for U/V, SPD for B)
        3. Update U, B, V with constrained gradients (same LR for all)
        4. Reconstruct W = U_new @ B_new @ V_new.T + R (R is frozen)
        5. Periodic resync to update factorization
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Base learning rate
            lr = group['lr']
            energy_threshold = group['energy_threshold']
            resync_every = group['resync_every']
            ns_steps = group['ns_steps']
            lambda_min = group['lambda_min']
            admm_steps = group.get('admm_steps', 10)
            admm_rho = group.get('admm_rho', 4.0)
            grad_clip = group.get('grad_clip', 100.0)

            # Stiefel U, V optimizer hyperparameters
            stiefel_update = group.get('stiefel_update', 'projected_qr')
            stiefel_max_rms = group.get('stiefel_max_rms', 0.03)
            uv_lr_scale = group.get('uv_lr_scale', 0.3)
            lr_UV = lr * uv_lr_scale

            # Log-Cholesky B optimizer hyperparameters
            b_beta1 = group.get('b_beta1', 0.9)
            b_beta2 = group.get('b_beta2', 0.99)
            b_adam_eps = group.get('b_adam_eps', 1e-8)
            b_update_rms = group.get('b_update_rms', 0.03)
            b_spd_eps = group.get('b_spd_eps', 1e-6)
            b_min_log_diag = group.get('b_min_log_diag', -8.0)
            b_max_log_diag = group.get('b_max_log_diag', 8.0)
            b_lr_scale = group.get('b_lr_scale', 0.3)
            lr_B = lr * b_lr_scale

            for p in group['params']:
                if p.grad is None:
                    continue

                # Skip 1D params (biases)
                if p.dim() != 2:
                    p.data.add_(p.grad, alpha=-lr)
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize UBV + R on first step
                if 'U' not in state:
                    self._init_ubv(p, state, energy_threshold)

                # Get factors and frozen residual
                U = state['U']
                B = state['B']
                V = state['V']
                R = state['R']  # Frozen residual

                # Work in float32
                G = grad.float()

                # Skip if gradient is invalid
                if not torch.isfinite(G).all():
                    continue

                # === Step 1: Project gradient to UBV structure ===
                # Chain rule: W = U @ B @ V.T + R (R is frozen, so dR = 0)
                G_U = G @ V @ B.T          # (m, r)
                G_B = U.T @ G @ V          # (r, r)
                G_V = G.T @ U @ B          # (n, r)

                # Track raw gradient norms (before any processing)
                G_U_norm = torch.linalg.norm(G_U, 'fro')
                G_V_norm = torch.linalg.norm(G_V, 'fro')
                G_B_norm = torch.linalg.norm(G_B, 'fro')

                # === Step 2: Compute U, V update directions ===

                if stiefel_update == "projected_qr":
                    # Stable path: tangent projection + RMS clipping + QR retraction

                    # Project gradients to tangent space at U, V
                    A_U_raw = stiefel_project_tangent(U, G_U)
                    A_V_raw = stiefel_project_tangent(V, G_V)

                    # Track tangent error BEFORE projection (should be ~0 since we just projected)
                    # For raw Euclidean gradient, this shows how far off tangent it was
                    tangent_U_before = U.T @ G_U + G_U.T @ U
                    tangent_V_before = V.T @ G_V + G_V.T @ V
                    state['tangent_err_U_before'] = torch.linalg.norm(tangent_U_before, 'fro').item()
                    state['tangent_err_V_before'] = torch.linalg.norm(tangent_V_before, 'fro').item()

                    # Track tangent error AFTER projection (should be near zero)
                    tangent_U_after = U.T @ A_U_raw + A_U_raw.T @ U
                    tangent_V_after = V.T @ A_V_raw + A_V_raw.T @ V
                    state['tangent_err_U_after'] = torch.linalg.norm(tangent_U_after, 'fro').item()
                    state['tangent_err_V_after'] = torch.linalg.norm(tangent_V_after, 'fro').item()

                    # Normalize/clip updates by RMS threshold
                    A_U, U_clip_info = normalize_update(A_U_raw, max_rms=stiefel_max_rms)
                    A_V, V_clip_info = normalize_update(A_V_raw, max_rms=stiefel_max_rms)

                    # Log step norms
                    state['U_step_norm_raw'] = U_clip_info['raw_norm']
                    state['U_step_norm_clipped'] = U_clip_info['clipped_norm']
                    state['V_step_norm_raw'] = V_clip_info['raw_norm']
                    state['V_step_norm_clipped'] = V_clip_info['clipped_norm']
                    state['clipped_U'] = U_clip_info['was_clipped']
                    state['clipped_V'] = V_clip_info['was_clipped']

                    # === Step 3: Update U, V via QR retraction ===
                    U_stepped = U - lr_UV * A_U
                    U_new = stiefel_qr_retraction(U_stepped)

                    V_stepped = V - lr_UV * A_V
                    V_new = stiefel_qr_retraction(V_stepped)

                elif stiefel_update == "muon_admm":
                    # Legacy path: Manifold MUON + tangent projection + QR retraction

                    # Get MUON direction (may not be exactly tangent)
                    A_U_muon = manifold_muon_admm(U, G_U, steps=admm_steps, rho=admm_rho)
                    A_V_muon = manifold_muon_admm(V, G_V, steps=admm_steps, rho=admm_rho)

                    # Track tangent error BEFORE projection
                    tangent_U_before = U.T @ A_U_muon + A_U_muon.T @ U
                    tangent_V_before = V.T @ A_V_muon + A_V_muon.T @ V
                    state['tangent_err_U_before'] = torch.linalg.norm(tangent_U_before, 'fro').item()
                    state['tangent_err_V_before'] = torch.linalg.norm(tangent_V_before, 'fro').item()

                    # Force tangent projection to fix any deviation
                    A_U_raw = stiefel_project_tangent(U, A_U_muon)
                    A_V_raw = stiefel_project_tangent(V, A_V_muon)

                    # Track tangent error AFTER projection
                    tangent_U_after = U.T @ A_U_raw + A_U_raw.T @ U
                    tangent_V_after = V.T @ A_V_raw + A_V_raw.T @ V
                    state['tangent_err_U_after'] = torch.linalg.norm(tangent_U_after, 'fro').item()
                    state['tangent_err_V_after'] = torch.linalg.norm(tangent_V_after, 'fro').item()

                    # Normalize/clip updates by RMS threshold
                    A_U, U_clip_info = normalize_update(A_U_raw, max_rms=stiefel_max_rms)
                    A_V, V_clip_info = normalize_update(A_V_raw, max_rms=stiefel_max_rms)

                    # Log step norms
                    state['U_step_norm_raw'] = U_clip_info['raw_norm']
                    state['U_step_norm_clipped'] = U_clip_info['clipped_norm']
                    state['V_step_norm_raw'] = V_clip_info['raw_norm']
                    state['V_step_norm_clipped'] = V_clip_info['clipped_norm']
                    state['clipped_U'] = U_clip_info['was_clipped']
                    state['clipped_V'] = V_clip_info['was_clipped']

                    # === Step 3: Update U, V via QR retraction (more stable than msign) ===
                    U_stepped = U - lr_UV * A_U
                    U_new = stiefel_qr_retraction(U_stepped)

                    V_stepped = V - lr_UV * A_V
                    V_new = stiefel_qr_retraction(V_stepped)

                else:
                    raise ValueError(f"Unknown stiefel_update method: {stiefel_update}")

                # === Step 4: Update B via Log-Cholesky Adam ===
                # Get Log-Cholesky parameters
                A_lower = state['B_A_lower']
                log_diag = state['B_log_diag']

                # Update B using Log-Cholesky Adam (no B @ G @ B amplification)
                B_new, A_lower_new, log_diag_new, b_info = spd_logchol_adam_update(
                    A_lower=A_lower,
                    log_diag=log_diag,
                    G_B=G_B,
                    state=state,
                    lr=lr_B,
                    beta1=b_beta1,
                    beta2=b_beta2,
                    adam_eps=b_adam_eps,
                    spd_eps=b_spd_eps,
                    max_log_diag=b_max_log_diag,
                    min_log_diag=b_min_log_diag,
                    b_update_rms=b_update_rms,
                )

                # Update Log-Cholesky state
                state['B_A_lower'] = A_lower_new
                state['B_log_diag'] = log_diag_new

                # Track gradient magnitudes for debugging
                state['grad_norm_raw'] = torch.linalg.norm(grad.float(), 'fro').item()
                state['grad_norm_U'] = G_U_norm.item()
                state['grad_norm_V'] = G_V_norm.item()
                state['grad_norm_B'] = G_B_norm.item()

                # B-specific metrics from Log-Cholesky update
                state['B_step_norm'] = b_info['B_step_norm']
                state['B_step_norm_clipped'] = b_info['B_step_norm_clipped']
                state['B_logdiag_min'] = b_info['B_logdiag_min']
                state['B_logdiag_max'] = b_info['B_logdiag_max']
                state['B_grad_norm'] = b_info['B_grad_norm']
                state['clipped_B'] = b_info['B_clipped']

                # Note: U/V clipping and step norms are already set in the Stiefel update section above

                # === Step 5: Reconstruct W = U @ B @ V.T + R (frozen residual) ===
                if torch.isfinite(U_new).all() and torch.isfinite(V_new).all() and torch.isfinite(B_new).all():
                    state['U'] = U_new
                    state['V'] = V_new
                    state['B'] = B_new

                    # W = low-rank part + frozen residual
                    W_new = U_new @ B_new @ V_new.T + R

                    if torch.isfinite(W_new).all():
                        p.data.copy_(W_new.to(p.dtype))

                # === Step 6: Periodic resync ===
                state['step'] = state.get('step', 0) + 1
                if state['step'] % resync_every == 0:
                    self._resync_ubv(p, state)
                    # Note: _resync_ubv resets B Adam state (B_adam_m, B_adam_v, B_adam_t)

        return loss

    def get_ubv_stats(self) -> Dict[str, Any]:
        """Get statistics about UBV factors and frozen residual across all parameters."""
        stats = {
            'eigenvalues': [],
            'orthonormality_U': [],
            'orthonormality_V': [],
            'ranks': [],
            'residual_norms': [],
            'weight_norms': [],
            'residual_ratios': [],  # ||R|| / ||W||
        }

        for group in self.param_groups:
            for p in group['params']:
                if p in self.state and 'B' in self.state[p]:
                    state = self.state[p]
                    B = state['B']
                    U = state['U']
                    V = state['V']
                    R = state.get('R', None)
                    r = state.get('r', U.shape[1])

                    # B eigenvalues
                    eigs = torch.linalg.eigvalsh(symmetric_part(B))
                    stats['eigenvalues'].append(eigs)

                    # Orthonormality errors (ensure same dtype for identity matrix)
                    I_U = torch.eye(U.shape[1], device=U.device, dtype=U.dtype)
                    I_V = torch.eye(V.shape[1], device=V.device, dtype=V.dtype)
                    U_err = torch.linalg.norm(U.T @ U - I_U)
                    V_err = torch.linalg.norm(V.T @ V - I_V)
                    stats['orthonormality_U'].append(U_err.item())
                    stats['orthonormality_V'].append(V_err.item())

                    # Rank
                    stats['ranks'].append(r)

                    # Residual statistics
                    if R is not None:
                        R_norm = torch.linalg.norm(R, 'fro').item()
                        W_norm = torch.linalg.norm(p.data.float(), 'fro').item()
                        stats['residual_norms'].append(R_norm)
                        stats['weight_norms'].append(W_norm)
                        if W_norm > 1e-10:
                            stats['residual_ratios'].append(R_norm / W_norm)

        return stats

    def get_spectrum_stats(self) -> Dict[str, List[torch.Tensor]]:
        """
        Get eigenvalue statistics for all B matrices.

        Compatible with distil_trainer's _get_ubv_metrics() method.

        Returns:
            Dictionary with 'eigenvalues' containing list of eigenvalue tensors.
        """
        eigenvalues_list = []
        for group in self.param_groups:
            for p in group['params']:
                if p in self.state and 'B' in self.state[p]:
                    B = self.state[p]['B'].float()
                    B_sym = symmetric_part(B)
                    eigs = torch.linalg.eigvalsh(B_sym)
                    eigenvalues_list.append(eigs)
        return {'eigenvalues': eigenvalues_list}

    def get_detailed_metrics(self) -> Dict[str, float]:
        """
        Get detailed metrics for logging to wandb.

        Returns dict with:
            - proj/min_eigenvalue: Minimum eigenvalue across all B matrices
            - proj/max_eigenvalue: Maximum eigenvalue across all B matrices
            - proj/mean_rank: Average rank across all layers
            - proj/max_condition_number: Worst condition number
            - proj/mean_orthonormality_error: Average orthonormality error
        """
        stats = self.get_ubv_stats()

        if not stats['eigenvalues']:
            return {}

        metrics = {}

        # Eigenvalue stats
        all_eigs = torch.cat(stats['eigenvalues'])
        metrics['proj/min_eigenvalue'] = all_eigs.min().item()
        metrics['proj/max_eigenvalue'] = all_eigs.max().item()
        metrics['proj/mean_eigenvalue'] = all_eigs.mean().item()

        # Condition numbers per B matrix
        condition_numbers = []
        for eigs in stats['eigenvalues']:
            if eigs.min() > 1e-10:
                cond = eigs.max() / eigs.min()
                condition_numbers.append(cond.item())
        if condition_numbers:
            metrics['proj/max_condition_number'] = max(condition_numbers)
            metrics['proj/mean_condition_number'] = sum(condition_numbers) / len(condition_numbers)

        # Rank stats
        if stats['ranks']:
            metrics['proj/mean_rank'] = sum(stats['ranks']) / len(stats['ranks'])
            metrics['proj/min_rank'] = min(stats['ranks'])
            metrics['proj/max_rank'] = max(stats['ranks'])

        # Orthonormality errors
        if stats['orthonormality_U']:
            metrics['proj/mean_orthonormality_U'] = sum(stats['orthonormality_U']) / len(stats['orthonormality_U'])
            metrics['proj/max_orthonormality_U'] = max(stats['orthonormality_U'])
        if stats['orthonormality_V']:
            metrics['proj/mean_orthonormality_V'] = sum(stats['orthonormality_V']) / len(stats['orthonormality_V'])
            metrics['proj/max_orthonormality_V'] = max(stats['orthonormality_V'])

        # Effective rank (based on eigenvalue distribution)
        effective_ranks = []
        for eigs in stats['eigenvalues']:
            eigs_pos = eigs[eigs > 1e-10]
            if len(eigs_pos) > 0:
                # Effective rank = exp(entropy of normalized eigenvalues)
                eigs_norm = eigs_pos / eigs_pos.sum()
                entropy = -(eigs_norm * torch.log(eigs_norm + 1e-10)).sum()
                eff_rank = torch.exp(entropy).item()
                effective_ranks.append(eff_rank)
        if effective_ranks:
            metrics['proj/mean_effective_rank'] = sum(effective_ranks) / len(effective_ranks)

        # Frozen residual statistics
        if stats['residual_norms']:
            metrics['proj/mean_residual_norm'] = sum(stats['residual_norms']) / len(stats['residual_norms'])
            metrics['proj/max_residual_norm'] = max(stats['residual_norms'])
        if stats['residual_ratios']:
            # Ratio ||R|| / ||W|| shows how much of weight is in residual vs low-rank
            metrics['proj/mean_residual_ratio'] = sum(stats['residual_ratios']) / len(stats['residual_ratios'])
            metrics['proj/max_residual_ratio'] = max(stats['residual_ratios'])

        # Tangent space error statistics (before and after projection)
        # Before: error of raw gradient/MUON direction
        # After: error after tangent projection (should be near zero)
        tangent_errs_U_before = []
        tangent_errs_V_before = []
        tangent_errs_U_after = []
        tangent_errs_V_after = []
        for group in self.param_groups:
            for p in group['params']:
                if p in self.state:
                    state = self.state[p]
                    if 'tangent_err_U_before' in state:
                        tangent_errs_U_before.append(state['tangent_err_U_before'])
                    if 'tangent_err_V_before' in state:
                        tangent_errs_V_before.append(state['tangent_err_V_before'])
                    if 'tangent_err_U_after' in state:
                        tangent_errs_U_after.append(state['tangent_err_U_after'])
                    if 'tangent_err_V_after' in state:
                        tangent_errs_V_after.append(state['tangent_err_V_after'])

        if tangent_errs_U_before:
            metrics['proj/mean_tangent_err_U_before'] = sum(tangent_errs_U_before) / len(tangent_errs_U_before)
            metrics['proj/max_tangent_err_U_before'] = max(tangent_errs_U_before)
        if tangent_errs_V_before:
            metrics['proj/mean_tangent_err_V_before'] = sum(tangent_errs_V_before) / len(tangent_errs_V_before)
            metrics['proj/max_tangent_err_V_before'] = max(tangent_errs_V_before)
        if tangent_errs_U_after:
            metrics['proj/mean_tangent_err_U_after'] = sum(tangent_errs_U_after) / len(tangent_errs_U_after)
            metrics['proj/max_tangent_err_U_after'] = max(tangent_errs_U_after)
        if tangent_errs_V_after:
            metrics['proj/mean_tangent_err_V_after'] = sum(tangent_errs_V_after) / len(tangent_errs_V_after)
            metrics['proj/max_tangent_err_V_after'] = max(tangent_errs_V_after)

        # Gradient magnitude statistics (for debugging)
        grad_norms_raw = []
        grad_norms_U = []
        grad_norms_V = []
        grad_norms_B = []

        # U/V step norm metrics (from stable Stiefel update)
        U_step_norms_raw = []
        U_step_norms_clipped = []
        V_step_norms_raw = []
        V_step_norms_clipped = []

        # Log-Cholesky B-specific metrics
        B_step_norms = []
        B_step_norms_clipped = []
        B_logdiag_mins = []
        B_logdiag_maxs = []
        B_grad_norms = []

        for group in self.param_groups:
            for p in group['params']:
                if p in self.state:
                    state = self.state[p]
                    if 'grad_norm_raw' in state:
                        grad_norms_raw.append(state['grad_norm_raw'])
                    if 'grad_norm_U' in state:
                        grad_norms_U.append(state['grad_norm_U'])
                    if 'grad_norm_V' in state:
                        grad_norms_V.append(state['grad_norm_V'])
                    if 'grad_norm_B' in state:
                        grad_norms_B.append(state['grad_norm_B'])
                    # U/V step norms
                    if 'U_step_norm_raw' in state:
                        U_step_norms_raw.append(state['U_step_norm_raw'])
                    if 'U_step_norm_clipped' in state:
                        U_step_norms_clipped.append(state['U_step_norm_clipped'])
                    if 'V_step_norm_raw' in state:
                        V_step_norms_raw.append(state['V_step_norm_raw'])
                    if 'V_step_norm_clipped' in state:
                        V_step_norms_clipped.append(state['V_step_norm_clipped'])
                    # Log-Cholesky B metrics
                    if 'B_step_norm' in state:
                        B_step_norms.append(state['B_step_norm'])
                    if 'B_step_norm_clipped' in state:
                        B_step_norms_clipped.append(state['B_step_norm_clipped'])
                    if 'B_logdiag_min' in state:
                        B_logdiag_mins.append(state['B_logdiag_min'])
                    if 'B_logdiag_max' in state:
                        B_logdiag_maxs.append(state['B_logdiag_max'])
                    if 'B_grad_norm' in state:
                        B_grad_norms.append(state['B_grad_norm'])

        if grad_norms_raw:
            metrics['proj/mean_grad_norm_raw'] = sum(grad_norms_raw) / len(grad_norms_raw)
            metrics['proj/max_grad_norm_raw'] = max(grad_norms_raw)
        if grad_norms_U:
            metrics['proj/mean_grad_norm_U'] = sum(grad_norms_U) / len(grad_norms_U)
            metrics['proj/max_grad_norm_U'] = max(grad_norms_U)
        if grad_norms_V:
            metrics['proj/mean_grad_norm_V'] = sum(grad_norms_V) / len(grad_norms_V)
            metrics['proj/max_grad_norm_V'] = max(grad_norms_V)
        if grad_norms_B:
            metrics['proj/mean_grad_norm_B'] = sum(grad_norms_B) / len(grad_norms_B)
            metrics['proj/max_grad_norm_B'] = max(grad_norms_B)

        # U/V step norm metrics
        if U_step_norms_raw:
            metrics['proj/mean_U_step_norm_raw'] = sum(U_step_norms_raw) / len(U_step_norms_raw)
            metrics['proj/max_U_step_norm_raw'] = max(U_step_norms_raw)
        if U_step_norms_clipped:
            metrics['proj/mean_U_step_norm_clipped'] = sum(U_step_norms_clipped) / len(U_step_norms_clipped)
        if V_step_norms_raw:
            metrics['proj/mean_V_step_norm_raw'] = sum(V_step_norms_raw) / len(V_step_norms_raw)
            metrics['proj/max_V_step_norm_raw'] = max(V_step_norms_raw)
        if V_step_norms_clipped:
            metrics['proj/mean_V_step_norm_clipped'] = sum(V_step_norms_clipped) / len(V_step_norms_clipped)

        # Log-Cholesky B step metrics
        if B_step_norms:
            metrics['proj/mean_B_step_norm'] = sum(B_step_norms) / len(B_step_norms)
            metrics['proj/max_B_step_norm'] = max(B_step_norms)
        if B_step_norms_clipped:
            metrics['proj/mean_B_step_norm_clipped'] = sum(B_step_norms_clipped) / len(B_step_norms_clipped)
        if B_logdiag_mins:
            metrics['proj/min_B_logdiag'] = min(B_logdiag_mins)
        if B_logdiag_maxs:
            metrics['proj/max_B_logdiag'] = max(B_logdiag_maxs)
        if B_grad_norms:
            metrics['proj/mean_B_grad_norm'] = sum(B_grad_norms) / len(B_grad_norms)
            metrics['proj/max_B_grad_norm'] = max(B_grad_norms)

        # Clipping rate (fraction of layers where clipping was triggered)
        clipped_U = []
        clipped_V = []
        clipped_B = []
        for group in self.param_groups:
            for p in group['params']:
                if p in self.state:
                    state = self.state[p]
                    if 'clipped_U' in state:
                        clipped_U.append(state['clipped_U'])
                    if 'clipped_V' in state:
                        clipped_V.append(state['clipped_V'])
                    if 'clipped_B' in state:
                        clipped_B.append(state['clipped_B'])

        if clipped_U:
            metrics['proj/clip_rate_U'] = sum(clipped_U) / len(clipped_U)
        if clipped_V:
            metrics['proj/clip_rate_V'] = sum(clipped_V) / len(clipped_V)
        if clipped_B:
            metrics['proj/clip_rate_B'] = sum(clipped_B) / len(clipped_B)

        return metrics


def create_projected_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    energy_threshold: float = 0.9,
    resync_every: int = 100,
    admm_steps: int = 10,
    admm_rho: float = 4.0,
    grad_clip: float = 100.0,
    target_modules: Optional[List[str]] = None,
    # Stiefel U, V optimizer hyperparameters
    stiefel_update: str = "projected_qr",
    stiefel_max_rms: float = 0.03,
    uv_lr_scale: float = 0.3,
    # Log-Cholesky B optimizer hyperparameters
    b_beta1: float = 0.9,
    b_beta2: float = 0.99,
    b_adam_eps: float = 1e-8,
    b_update_rms: float = 0.03,
    b_spd_eps: float = 1e-6,
    b_min_log_diag: float = -8.0,
    b_max_log_diag: float = 8.0,
    b_lr_scale: float = 0.3,
) -> ProjectedGradientOptimizer:
    """
    Create a ProjectedGradientOptimizer for a model.

    Args:
        model: Model to optimize
        lr: Learning rate for U, V factors
        energy_threshold: Energy threshold for rank detection (0.0-1.0)
        resync_every: Steps between resyncs
        admm_steps: Number of ADMM iterations for manifold MUON (U, V) - only used with muon_admm
        admm_rho: ADMM penalty parameter - only used with muon_admm
        grad_clip: Gradient clipping threshold (legacy, prefer stiefel_max_rms)
        target_modules: Module name patterns to apply projection (None = all Linear)
        stiefel_update: Stiefel update method ("projected_qr" or "muon_admm")
        stiefel_max_rms: RMS threshold for U, V update clipping
        uv_lr_scale: Learning rate scale for U, V relative to lr
        b_beta1: Adam beta1 for B optimizer
        b_beta2: Adam beta2 for B optimizer
        b_adam_eps: Adam epsilon for B optimizer
        b_update_rms: RMS threshold for B update clipping
        b_spd_eps: Small constant for SPD numerical stability
        b_min_log_diag: Minimum log-diagonal value
        b_max_log_diag: Maximum log-diagonal value
        b_lr_scale: Learning rate scale for B relative to lr

    Returns:
        Configured optimizer
    """
    # Collect parameters
    projected_params = []
    other_params = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if target_modules is None or any(t in name for t in target_modules):
                projected_params.append(module.weight)
                if module.bias is not None:
                    other_params.append(module.bias)
            else:
                other_params.extend(module.parameters())

    # Create optimizer with param groups
    param_groups = [
        {'params': projected_params,
         'lr': lr,
         'energy_threshold': energy_threshold,
         'resync_every': resync_every,
         'admm_steps': admm_steps,
         'admm_rho': admm_rho,
         'grad_clip': grad_clip,
         # Stiefel U, V optimizer
         'stiefel_update': stiefel_update,
         'stiefel_max_rms': stiefel_max_rms,
         'uv_lr_scale': uv_lr_scale,
         # Log-Cholesky B optimizer
         'b_beta1': b_beta1,
         'b_beta2': b_beta2,
         'b_adam_eps': b_adam_eps,
         'b_update_rms': b_update_rms,
         'b_spd_eps': b_spd_eps,
         'b_min_log_diag': b_min_log_diag,
         'b_max_log_diag': b_max_log_diag,
         'b_lr_scale': b_lr_scale,
         },
    ]

    optimizer = ProjectedGradientOptimizer(param_groups)

    # Add other params (biases, non-target modules) with same LR
    if other_params:
        optimizer.add_param_group({
            'params': other_params,
            'lr': lr,
            'energy_threshold': 1.0,  # No projection for other params
            'resync_every': 999999,
            'admm_steps': admm_steps,
            'admm_rho': admm_rho,
            'grad_clip': grad_clip,
            # Stiefel params not used for biases, but include for consistency
            'stiefel_update': stiefel_update,
            'stiefel_max_rms': stiefel_max_rms,
            'uv_lr_scale': uv_lr_scale,
            # B params not used for biases, but include for consistency
            'b_beta1': b_beta1,
            'b_beta2': b_beta2,
            'b_adam_eps': b_adam_eps,
            'b_update_rms': b_update_rms,
            'b_spd_eps': b_spd_eps,
            'b_min_log_diag': b_min_log_diag,
            'b_max_log_diag': b_max_log_diag,
            'b_lr_scale': b_lr_scale,
        })

    return optimizer
