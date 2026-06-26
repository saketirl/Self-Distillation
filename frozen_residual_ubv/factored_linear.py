"""
FactoredLinear module for rank-preserving continual learning.

This module implements the UBV^T factorization:
    W_eff = W_res + U @ B @ V^T

where:
    - U in St(r*, m): Stiefel manifold (orthonormal columns)
    - B in S_{++}(r*): Symmetric positive definite (trainable spectrum)
    - V in St(r*, n): Stiefel manifold (orthonormal columns)
    - W_res: Frozen residual (captures tail singular components)

The factorization preserves rank by construction: the trainable component
UBV^T has exactly rank r*, and the residual W_res has rank (full_rank - r*).

References:
    - He et al., "Spectral Collapse Drives Loss of Plasticity in Deep
      Continual Learning," arXiv:2509.22335.
    - Mishra et al., "R3MC: A Riemannian three-factor algorithm for low-rank
      matrix completion," CDC 2014.
"""

from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor


class FactoredLinear(nn.Module):
    """
    A linear layer with UBV^T + W_res factorization for rank-preserving training.

    This replaces a standard nn.Linear layer with a factored representation
    that enables Riemannian optimization on manifold constraints. The effective
    weight is:

        W_eff = W_res + U @ B @ V^T

    where U, V have orthonormal columns (Stiefel manifold) and B is symmetric
    positive definite. This structure guarantees rank preservation during training.

    The forward pass computes:
        y = x @ W_eff^T + bias = x @ W_res^T + x @ V @ B^T @ U^T + bias

    using efficient parenthesization to avoid materializing the full W_eff matrix.

    Args:
        W_pretrained: Pretrained weight matrix of shape (out_features, in_features).
        r_star: Target rank for the trainable UBV^T component.
        bias: Optional pretrained bias vector of shape (out_features,).
        freeze_residual: If True, W_res is frozen. If False, skip residual entirely.

    Attributes:
        U: Left singular vectors, shape (out_features, r_star), trainable.
        B: Spectrum matrix, shape (r_star, r_star), trainable.
        V: Right singular vectors, shape (in_features, r_star), trainable.
        W_res: Residual weight (frozen buffer) or None.
        bias: Bias parameter (trainable) or None.
    """

    def __init__(
        self,
        W_pretrained: Tensor,
        r_star: int,
        bias: Optional[Tensor] = None,
        freeze_residual: bool = True,
    ):
        super().__init__()

        # Store dimensions
        self.out_features, self.in_features = W_pretrained.shape
        self.r_star = r_star
        self.freeze_residual = freeze_residual

        # Determine full rank
        full_rank = min(self.out_features, self.in_features)

        # Validate r_star
        if r_star < 1:
            raise ValueError(f"r_star must be >= 1, got {r_star}")
        if r_star > full_rank:
            # Silently cap to full rank
            r_star = full_rank
            self.r_star = r_star

        # Store original dtype
        original_dtype = W_pretrained.dtype

        # Compute SVD in float32 for numerical stability
        # W = U_full @ diag(S_full) @ Vt_full
        W_f32 = W_pretrained.detach().float()
        U_full, S_full, Vt_full = torch.linalg.svd(W_f32, full_matrices=False)

        # Extract top-r* components for trainable factors
        U_0 = U_full[:, :r_star].to(original_dtype)  # (out_features, r_star)
        S_0 = S_full[:r_star].to(original_dtype)      # (r_star,)
        V_0 = Vt_full[:r_star, :].T.to(original_dtype)  # (in_features, r_star)

        # Initialize B as diagonal matrix with top-r* singular values
        # B is symmetric positive definite by construction
        B_0 = torch.diag(S_0)  # (r_star, r_star)

        # Register trainable parameters
        # Tag with _ubv_role for optimizer dispatch
        self.U = nn.Parameter(U_0.clone())
        self.U._ubv_role = "U"

        self.B = nn.Parameter(B_0.clone())
        self.B._ubv_role = "B"

        self.V = nn.Parameter(V_0.clone())
        self.V._ubv_role = "V"

        # Handle residual
        if r_star >= full_rank or not freeze_residual:
            # Full-rank mode: no residual needed
            self.register_buffer('W_res', None)
        else:
            # Compute residual from tail singular components
            # W_res = U_tail @ diag(S_tail) @ Vt_tail
            U_tail = U_full[:, r_star:].to(original_dtype)
            S_tail = S_full[r_star:].to(original_dtype)
            Vt_tail = Vt_full[r_star:, :].to(original_dtype)

            # Reconstruct residual (this is materialized once at init, then frozen)
            W_res = U_tail @ torch.diag(S_tail) @ Vt_tail
            self.register_buffer('W_res', W_res)

        # Handle bias
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
            self.bias._ubv_role = "bias"
        else:
            self.register_parameter('bias', None)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass using effective weight.

        Computes: y = x @ W_eff^T + bias

        where W_eff = W_res + U @ B @ V^T.

        We materialize W_eff for numerical stability. This uses more memory
        but ensures the forward pass matches standard nn.Linear behavior,
        which is critical for stable gradients during training.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        # Compute effective weight: W_eff = W_res + U @ B @ V^T
        # This is materialized for numerical stability
        UBVt = self.U @ self.B @ self.V.T  # (out_features, in_features)

        if self.W_res is not None:
            W_eff = self.W_res + UBVt
        else:
            W_eff = UBVt

        # Standard linear forward: y = x @ W_eff^T + bias
        y = x @ W_eff.T

        if self.bias is not None:
            y = y + self.bias

        return y

    def get_effective_weight(self) -> Tensor:
        """
        Materialize the effective weight matrix W_eff = W_res + U @ B @ V^T.

        WARNING: This creates a full m x n matrix. Only use for debugging or
        evaluation, never in training forward pass.

        Returns:
            Effective weight matrix of shape (out_features, in_features).
        """
        UBVt = self.U @ self.B @ self.V.T  # (out_features, in_features)

        if self.W_res is not None:
            return self.W_res + UBVt
        else:
            return UBVt

    def extra_repr(self) -> str:
        """String representation for debugging."""
        residual_info = "frozen" if self.W_res is not None else "none"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r_star={self.r_star}, residual={residual_info}, "
            f"bias={self.bias is not None}"
        )

    @property
    def trainable_params(self) -> int:
        """Count trainable parameters in the factored representation."""
        # U: out_features * r_star
        # B: r_star * r_star
        # V: in_features * r_star
        # bias: out_features (if present)
        count = (
            self.out_features * self.r_star +
            self.r_star * self.r_star +
            self.in_features * self.r_star
        )
        if self.bias is not None:
            count += self.out_features
        return count

    @property
    def residual_params(self) -> int:
        """Count frozen parameters in the residual."""
        if self.W_res is None:
            return 0
        return self.W_res.numel()


def create_factored_linear_from_linear(
    linear: nn.Linear,
    r_star: int,
    freeze_residual: bool = True,
) -> FactoredLinear:
    """
    Create a FactoredLinear module from an existing nn.Linear.

    Args:
        linear: Source nn.Linear module.
        r_star: Target rank for trainable component.
        freeze_residual: If True, freeze the residual component.

    Returns:
        FactoredLinear module initialized from the linear layer's weights.
    """
    return FactoredLinear(
        W_pretrained=linear.weight.data,
        r_star=r_star,
        bias=linear.bias.data if linear.bias is not None else None,
        freeze_residual=freeze_residual,
    )
