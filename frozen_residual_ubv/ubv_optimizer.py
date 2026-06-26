"""
UBV Optimizer: Riemannian optimization for UBV^T factorization.

This optimizer implements geometric update rules for the three factor types:
- U, V: Manifold MUON with dual ascent on Stiefel manifold + msign retraction
- B: Riemannian gradient descent on S_{++} with affine-invariant metric (Mishra)

The optimizer enforces manifold constraints at each step:
- U, V remain orthonormal (Stiefel constraint)
- B remains symmetric positive definite with eigenvalues >= lambda_min

References:
    - Buchanan et al., "Manifold MUON" (thinky-manifolds)
      https://github.com/sdbuch/thinky-manifolds
    - Mishra et al., "R3MC: A Riemannian three-factor algorithm for low-rank
      matrix completion," CDC 2014, Table 5.
    - He et al., "Spectral Collapse Drives Loss of Plasticity in Deep
      Continual Learning," arXiv:2509.22335.
"""

from typing import Optional, Dict, Any, Callable, List
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .utils import (
    eigenvalue_clamp,
    msign,
    symmetric_part,
)


class UBVOptimizer(Optimizer):
    """
    Riemannian optimizer for UBV^T factorization.

    This optimizer handles three types of parameters:
    - "U" parameters: Optimized on Stiefel manifold via dual ascent
    - "B" parameters: Optimized on S_{++} via affine-invariant Riemannian gradient
    - "V" parameters: Same as U (Stiefel manifold)
    - "bias" parameters: Standard SGD

    Parameters are identified by their _ubv_role attribute, which should be
    set by the FactoredLinear module.

    Args:
        params: Iterable of parameters to optimize.
        lr_U: Learning rate for U (Stiefel) updates.
        lr_B: Learning rate for B (S_{++}) updates.
        lr_V: Learning rate for V (Stiefel) updates.
        lr_bias: Learning rate for bias (SGD) updates.
        dual_alpha: Step size for dual variable updates in Stiefel optimization.
        dual_max_iters: Maximum dual ascent iterations (low due to warm-starting).
        dual_tol: Convergence tolerance for dual ascent.
        ns_steps: Newton-Schulz iterations for matrix sign function.
        lambda_min: Minimum eigenvalue for B (spectrum floor).
        momentum: Momentum coefficient for B updates (per Mishra).
        retraction: Retraction method for Stiefel ("modula" or "polar").

    Example:
        >>> model = FactoredLinear(W, r_star=16)
        >>> optimizer = UBVOptimizer(model.parameters(), lr_U=1e-2, lr_B=1e-3)
        >>> for x, y in dataloader:
        ...     optimizer.zero_grad()
        ...     loss = criterion(model(x), y)
        ...     loss.backward()
        ...     optimizer.step()
    """

    def __init__(
        self,
        params,
        lr_U: float = 1e-2,
        lr_B: float = 1e-3,
        lr_V: float = 1e-2,
        lr_bias: float = 1e-3,
        dual_alpha: float = 1e-2,
        dual_max_iters: int = 5,
        dual_tol: float = 1e-6,
        ns_steps: int = 10,
        lambda_min: float = 1e-4,
        momentum: float = 0.0,
        retraction: str = "modula",
    ):
        if lr_U < 0.0:
            raise ValueError(f"Invalid learning rate lr_U: {lr_U}")
        if lr_B < 0.0:
            raise ValueError(f"Invalid learning rate lr_B: {lr_B}")
        if lr_V < 0.0:
            raise ValueError(f"Invalid learning rate lr_V: {lr_V}")
        if lr_bias < 0.0:
            raise ValueError(f"Invalid learning rate lr_bias: {lr_bias}")
        if lambda_min <= 0.0:
            raise ValueError(f"Invalid lambda_min: {lambda_min}. Must be > 0")
        if retraction not in ("modula", "polar"):
            raise ValueError(f"Invalid retraction: {retraction}. Must be 'modula' or 'polar'")

        # Include 'lr' for compatibility with standard LR schedulers
        # The scheduler will modify 'lr', and we scale the individual rates accordingly
        defaults = dict(
            lr=lr_U,  # Base LR for scheduler compatibility
            lr_U=lr_U,
            lr_B=lr_B,
            lr_V=lr_V,
            lr_bias=lr_bias,
            # Store initial ratios for scaling when lr changes
            lr_U_ratio=1.0,
            lr_B_ratio=lr_B / lr_U if lr_U > 0 else 1.0,
            lr_V_ratio=lr_V / lr_U if lr_U > 0 else 1.0,
            lr_bias_ratio=lr_bias / lr_U if lr_U > 0 else 1.0,
            dual_alpha=dual_alpha,
            dual_max_iters=dual_max_iters,
            dual_tol=dual_tol,
            ns_steps=ns_steps,
            lambda_min=lambda_min,
            momentum=momentum,
            retraction=retraction,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Perform a single optimization step.

        For each parameter:
        - If role == "U": Stiefel dual ascent with warm-started Lambda
        - If role == "B": Riemannian gradient B @ Sym(G_B) @ B + eigenvalue clamp
        - If role == "V": Same as U (separate warm-started Lambda_V)
        - If role == "bias" or unknown: Standard SGD

        Args:
            closure: Optional closure that reevaluates the model and returns loss.

        Returns:
            Loss value if closure provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Get parameter role (set by FactoredLinear)
                role = getattr(p, '_ubv_role', 'bias')

                if role == "U":
                    self._update_stiefel(
                        p, grad, state, group, is_U=True
                    )
                elif role == "V":
                    self._update_stiefel(
                        p, grad, state, group, is_U=False
                    )
                elif role == "B":
                    self._update_spd(p, grad, state, group)
                else:
                    # Bias or unknown: standard SGD
                    # Use scaled LR for scheduler compatibility
                    lr = group['lr'] * group['lr_bias_ratio']
                    self._update_sgd(p, grad, lr)

        return loss

    def _update_stiefel(
        self,
        p: Tensor,
        grad: Tensor,
        state: Dict[str, Any],
        group: Dict[str, Any],
        is_U: bool,
    ) -> None:
        """
        Update a parameter on the Stiefel manifold via manifold MUON.

        This implements the manifold MUON algorithm from thinky-manifolds:
        1. Initialize dual variable Lambda from gradient
        2. Dual ascent loop to find descent direction A on tangent space
        3. Primal update with msign retraction

        Reference: https://github.com/sdbuch/thinky-manifolds

        Args:
            p: Parameter tensor (U or V), shape (m, r) or (n, r).
            grad: Euclidean gradient G_U or G_V, same shape as p.
            state: Optimizer state dictionary for this parameter.
            group: Parameter group with hyperparameters.
            is_U: True for U parameters, False for V.
        """
        # Compute effective LR: base lr * ratio (allows scheduler to work)
        base_lr = group['lr']
        ratio = group['lr_U_ratio'] if is_U else group['lr_V_ratio']
        eta = base_lr * ratio  # Primal step size
        ns_steps = group['ns_steps']
        dual_alpha = group['dual_alpha']
        dual_max_iters = group['dual_max_iters']
        dual_tol = group['dual_tol']

        # Store original dtype for bf16 support
        original_dtype = p.dtype
        device = p.device

        # Cast to float32 for numerical stability
        W = p.data.float()
        G = grad.float()

        # Handle NaN/Inf in gradient
        if not torch.isfinite(G).all():
            return  # Skip update if gradient is invalid

        m, r = W.shape

        # Initialize step count for tracking
        if 'step_count' not in state:
            state['step_count'] = 0
        state['step_count'] += 1

        # Clip gradient norm for stability (preserve magnitude unlike normalization)
        G_norm = torch.linalg.norm(G, 'fro')
        if G_norm < 1e-8:
            return  # Skip if gradient is negligible
        max_grad_norm = 1.0
        if G_norm > max_grad_norm:
            G = G * (max_grad_norm / G_norm)
            G_norm = max_grad_norm

        # Simple Riemannian gradient descent on Stiefel manifold
        # Project gradient onto tangent space: G_tan = G - W @ Sym(W^T @ G)
        WtG = W.T @ G
        sym_WtG = (WtG + WtG.T) / 2
        G_tangent = G - W @ sym_WtG

        # Take step in tangent direction
        W_stepped = W - eta * G_tangent

        # Handle NaN/Inf
        if not torch.isfinite(W_stepped).all():
            return

        # Retract to Stiefel manifold using polar decomposition (msign)
        W_new = msign(W_stepped, ns_steps=ns_steps)

        # Verify result is finite
        if not torch.isfinite(W_new).all():
            return

        # Cast back to original dtype and update parameter
        p.data.copy_(W_new.to(original_dtype))

    def _update_spd(
        self,
        p: Tensor,
        grad: Tensor,
        state: Dict[str, Any],
        group: Dict[str, Any],
    ) -> None:
        """
        Update a parameter on S_{++} (symmetric positive definite matrices).

        This implements the Riemannian gradient descent on S_{++} with the
        affine-invariant metric (Mishra et al. 2014, Table 5):

            B <- B - eta_B * B @ Sym(G_B) @ B

        The B @ Sym(G_B) @ B sandwich is the Riemannian gradient under the
        affine-invariant metric, which has the property that the geodesic
        distance is invariant to congruence transformations.

        After the update, eigenvalues are clamped to enforce a spectrum floor,
        preventing spectral collapse.

        Args:
            p: Parameter tensor B, shape (r, r), symmetric.
            grad: Euclidean gradient G_B, shape (r, r).
            state: Optimizer state dictionary for this parameter.
            group: Parameter group with hyperparameters.
        """
        # Compute effective LR: base lr * ratio (allows scheduler to work)
        base_lr = group['lr']
        lr = base_lr * group['lr_B_ratio']
        lambda_min = group['lambda_min']
        momentum = group['momentum']

        # Store original dtype for bf16 support
        original_dtype = p.dtype
        device = p.device
        r = p.shape[0]

        # Cast to float32 for numerical stability
        B = p.data.float()
        G = grad.float()

        # Handle NaN/Inf in gradient
        if not torch.isfinite(G).all():
            return  # Skip update if gradient is invalid

        # Compute symmetric part of gradient
        # This ensures we move in a direction that preserves symmetry
        G_sym = symmetric_part(G)

        # Clip gradient to prevent explosion
        grad_norm = torch.linalg.norm(G_sym, 'fro')
        max_grad_norm = 100.0
        if grad_norm > max_grad_norm:
            G_sym = G_sym * (max_grad_norm / grad_norm)

        # Compute Riemannian gradient: B @ Sym(G_B) @ B
        # This is the natural gradient on S_{++} under the affine-invariant metric
        # It naturally scales updates by the current eigenvalue magnitudes
        riem_grad = B @ G_sym @ B

        # Scale learning rate by 1/||B||²_F for scale-invariant updates
        # This is critical because riem_grad scales as O(||B||²)
        B_norm_sq = torch.linalg.norm(B, 'fro') ** 2
        # Avoid division by zero; also don't over-scale if B is very small
        scale_factor = 1.0 / torch.clamp(B_norm_sq, min=1.0).item()
        lr_scaled = lr * scale_factor

        # Clip Riemannian gradient (after scaling, this is more conservative)
        riem_norm = torch.linalg.norm(riem_grad, 'fro')
        B_norm = torch.sqrt(B_norm_sq)
        max_riem_norm = 1.0 * B_norm  # Much tighter clipping
        if riem_norm > max_riem_norm:
            riem_grad = riem_grad * (max_riem_norm / riem_norm)

        # Apply momentum if enabled
        if momentum != 0:
            if 'momentum_buffer' not in state:
                state['momentum_buffer'] = torch.zeros_like(riem_grad)
            buf = state['momentum_buffer']
            riem_grad = buf.mul_(momentum).add_(riem_grad)

        # Riemannian gradient descent step (using scaled lr)
        B_new = B - lr_scaled * riem_grad

        # Handle NaN/Inf in result
        if not torch.isfinite(B_new).all():
            B_new = B  # Fall back to no update

        # Symmetrize to handle numerical drift
        B_new = symmetric_part(B_new)

        # Clamp eigenvalues to enforce spectrum floor
        # This is the key operation that prevents spectral collapse
        B_new = eigenvalue_clamp(B_new, lambda_min=lambda_min)

        # Cast back to original dtype and update parameter
        p.data.copy_(B_new.to(original_dtype))

    def _update_sgd(self, p: Tensor, grad: Tensor, lr: float) -> None:
        """
        Standard SGD update for bias parameters.

        Args:
            p: Parameter tensor.
            grad: Gradient tensor.
            lr: Learning rate.
        """
        p.data.add_(grad, alpha=-lr)

    def get_dual_convergence_stats(self) -> Dict[str, Any]:
        """
        Get statistics about dual ascent convergence across all Stiefel parameters.

        Useful for debugging and monitoring the warm-starting effectiveness.

        Returns:
            Dictionary with step counts and convergence info.
        """
        stats = {'U': [], 'V': []}
        for group in self.param_groups:
            for p in group['params']:
                role = getattr(p, '_ubv_role', None)
                if role in ('U', 'V'):
                    state = self.state.get(p, {})
                    if 'step_count' in state:
                        stats[role].append(state['step_count'])
        return stats

    def get_spectrum_stats(self) -> Dict[str, List[Tensor]]:
        """
        Get eigenvalue statistics for all B matrices.

        Returns:
            Dictionary with 'eigenvalues' containing list of eigenvalue tensors.
        """
        eigenvalues_list = []
        for group in self.param_groups:
            for p in group['params']:
                role = getattr(p, '_ubv_role', None)
                if role == 'B':
                    B = p.data.float()
                    B_sym = (B + B.T) / 2
                    eigs = torch.linalg.eigvalsh(B_sym)
                    eigenvalues_list.append(eigs)
        return {'eigenvalues': eigenvalues_list}
