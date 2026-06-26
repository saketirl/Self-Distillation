"""
AdaptiveUBVLinear: Gradient projection through UBV structure.

Unlike FactoredLinear which replaces W with U @ B @ V.T + W_res,
this module keeps W as the actual parameter and uses UBV as
"shadow factors" for gradient projection.

Key differences:
1. Forward pass uses W directly (no approximation error)
2. Gradients are projected through UBV structure before applying
3. W is reconstructed from UBV after each update
4. Every m steps, UBV is re-factored from W (warm-started)

This allows:
- Zero forward pass error at each re-sync point
- Riemannian constraints on gradient directions
- The tail (W_res) can slowly adapt rather than being frozen
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import eigenvalue_clamp, msign, symmetric_part


def procrustes_align(X_new: Tensor, X_old: Tensor) -> Tensor:
    """
    Align X_new to X_old via Procrustes (optimal rotation).

    Finds R = argmin ||X_new @ R - X_old||_F s.t. R^T R = I
    Solution: R = V @ U^T where X_old^T @ X_new = U @ S @ V^T

    Args:
        X_new: New orthonormal matrix (m, r)
        X_old: Old orthonormal matrix (m, r) to align to

    Returns:
        X_new rotated to align with X_old
    """
    M = X_old.T @ X_new  # (r, r)
    U, S, Vt = torch.linalg.svd(M)
    R = Vt.T @ U.T  # Optimal rotation
    return X_new @ R


class AdaptiveUBVLinear(nn.Module):
    """
    Linear layer with adaptive UBV gradient projection.

    The actual weight W is the parameter. UBV factors are auxiliary
    variables used to project gradients onto a Riemannian manifold.

    Forward: y = x @ W^T + bias  (uses W directly, no error!)

    Gradient update:
        1. Compute dL/dW via backprop
        2. Project: G_U = dL/dW @ V @ B^T, G_B = U^T @ dL/dW @ V, etc.
        3. Apply Riemannian updates to U, B, V
        4. Reconstruct: W = U @ B @ V^T + W_res

    Every m steps:
        Re-factor W into fresh UBV + W_res (warm-started from previous)

    Args:
        W_pretrained: Initial weight matrix (out_features, in_features)
        r_star: Rank of the trainable UBV component
        bias: Optional bias vector
        resync_every: Re-factor W every this many optimizer steps
    """

    def __init__(
        self,
        W_pretrained: Tensor,
        r_star: int,
        bias: Optional[Tensor] = None,
        resync_every: int = 100,
    ):
        super().__init__()

        self.out_features, self.in_features = W_pretrained.shape
        self.r_star = min(r_star, min(self.out_features, self.in_features))
        self.resync_every = resync_every

        # W is the actual parameter - forward uses this directly!
        self.weight = nn.Parameter(W_pretrained.clone())

        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.register_parameter('bias', None)

        # Initialize UBV factors from SVD
        self._init_ubv_factors()

        # Step counter for resync
        self.register_buffer('_step_count', torch.tensor(0, dtype=torch.long))

    def _init_ubv_factors(self):
        """Initialize UBV factors from current weight via SVD."""
        W = self.weight.data.float()
        U_full, S_full, Vt_full = torch.linalg.svd(W, full_matrices=False)

        r = self.r_star
        dtype = self.weight.dtype

        # Top-r factors (these are buffers, not parameters)
        self.register_buffer('U', U_full[:, :r].to(dtype))
        self.register_buffer('B', torch.diag(S_full[:r]).to(dtype))
        self.register_buffer('V', Vt_full[:r, :].T.to(dtype))

        # Residual (tail components)
        if r < len(S_full):
            U_tail = U_full[:, r:]
            S_tail = S_full[r:]
            Vt_tail = Vt_full[r:, :]
            W_res = (U_tail @ torch.diag(S_tail) @ Vt_tail).to(dtype)
            self.register_buffer('W_res', W_res)
        else:
            self.register_buffer('W_res', None)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass using W directly.

        This has ZERO approximation error - we use the actual weight.
        """
        return F.linear(x, self.weight, self.bias)

    def project_gradient(self, grad_W: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Project weight gradient onto UBV structure.

        Given dL/dW, compute gradients w.r.t. U, B, V using chain rule:
            W = U @ B @ V^T
            dL/dU = dL/dW @ V @ B^T
            dL/dB = U^T @ dL/dW @ V
            dL/dV = dL/dW^T @ U @ B

        Args:
            grad_W: Gradient w.r.t. weight, shape (out_features, in_features)

        Returns:
            (G_U, G_B, G_V) tuple of projected gradients
        """
        G = grad_W.float()
        U = self.U.float()
        B = self.B.float()
        V = self.V.float()

        G_U = G @ V @ B.T      # (out_features, r)
        G_B = U.T @ G @ V      # (r, r)
        G_V = G.T @ U @ B      # (in_features, r)

        return G_U, G_B, G_V

    def apply_ubv_update(
        self,
        G_U: Tensor,
        G_B: Tensor,
        G_V: Tensor,
        lr_U: float,
        lr_B: float,
        lr_V: float,
        ns_steps: int = 10,
        lambda_min: float = 1e-4,
    ):
        """
        Apply Riemannian updates to UBV factors and reconstruct W.

        Args:
            G_U, G_B, G_V: Projected gradients from project_gradient()
            lr_U, lr_B, lr_V: Learning rates for each factor
            ns_steps: Newton-Schulz iterations for msign
            lambda_min: Minimum eigenvalue for B
        """
        dtype = self.weight.dtype

        # Work in float32
        U = self.U.float()
        B = self.B.float()
        V = self.V.float()
        G_U = G_U.float()
        G_B = G_B.float()
        G_V = G_V.float()

        # --- Update U on Stiefel manifold ---
        # Project gradient to tangent space
        UtGU = U.T @ G_U
        G_U_tan = G_U - U @ symmetric_part(UtGU)
        # Gradient descent + retraction
        U_new = msign(U - lr_U * G_U_tan, ns_steps=ns_steps)

        # --- Update V on Stiefel manifold ---
        VtGV = V.T @ G_V
        G_V_tan = G_V - V @ symmetric_part(VtGV)
        V_new = msign(V - lr_V * G_V_tan, ns_steps=ns_steps)

        # --- Update B on S_{++} ---
        # Riemannian gradient: B @ Sym(G_B) @ B
        G_B_sym = symmetric_part(G_B)
        riem_grad = B @ G_B_sym @ B
        # Scale learning rate by 1/||B||^2 for stability
        B_norm_sq = torch.linalg.norm(B, 'fro') ** 2
        lr_B_scaled = lr_B / max(B_norm_sq.item(), 1.0)
        B_new = B - lr_B_scaled * riem_grad
        B_new = symmetric_part(B_new)
        B_new = eigenvalue_clamp(B_new, lambda_min=lambda_min)

        # Update buffers
        self.U.copy_(U_new.to(dtype))
        self.B.copy_(B_new.to(dtype))
        self.V.copy_(V_new.to(dtype))

        # Reconstruct W from updated UBV
        W_new = U_new @ B_new @ V_new.T
        if self.W_res is not None:
            W_new = W_new + self.W_res.float()

        self.weight.data.copy_(W_new.to(dtype))

        # Increment step counter
        self._step_count += 1

        # Periodic resync
        if self._step_count.item() % self.resync_every == 0:
            self.resync_ubv()

    def resync_ubv(self):
        """
        Re-factor current W into UBV + W_res.

        This is warm-started from previous UBV via Procrustes alignment,
        ensuring smooth transitions. The tail (W_res) can evolve to absorb
        components that no longer fit in the rank-r structure.
        """
        W = self.weight.data.float()
        U_full, S_full, Vt_full = torch.linalg.svd(W, full_matrices=False)

        r = self.r_star
        dtype = self.weight.dtype

        # Extract new top-r factors
        U_new = U_full[:, :r]
        V_new = Vt_full[:r, :].T

        # Procrustes alignment for smooth transition
        # This prevents sudden jumps in UBV space
        U_aligned = procrustes_align(U_new, self.U.float())
        V_aligned = procrustes_align(V_new, self.V.float())

        # Compute B that minimizes ||W - U @ B @ V^T||_F
        # Since U, V are orthonormal: B = U^T @ W @ V
        B_new = U_aligned.T @ W @ V_aligned
        B_new = symmetric_part(B_new)
        B_new = eigenvalue_clamp(B_new, lambda_min=1e-6)

        # Compute new residual
        W_approx = U_aligned @ B_new @ V_aligned.T
        W_res_new = W - W_approx

        # Update buffers
        self.U.copy_(U_aligned.to(dtype))
        self.B.copy_(B_new.to(dtype))
        self.V.copy_(V_aligned.to(dtype))
        if self.W_res is not None:
            self.W_res.copy_(W_res_new.to(dtype))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r_star={self.r_star}, resync_every={self.resync_every}, "
            f"bias={self.bias is not None}"
        )


class AdaptiveUBVOptimizer(torch.optim.Optimizer):
    """
    Optimizer for AdaptiveUBVLinear modules.

    For AdaptiveUBVLinear parameters:
        1. Project gradient through UBV structure
        2. Apply Riemannian updates to U, B, V
        3. Reconstruct W from UBV

    For other parameters: standard SGD.

    Args:
        params: Model parameters
        lr_U: Learning rate for U (Stiefel)
        lr_B: Learning rate for B (SPD)
        lr_V: Learning rate for V (Stiefel)
        lr_other: Learning rate for non-UBV params
        ns_steps: Newton-Schulz iterations
        lambda_min: Minimum eigenvalue for B
    """

    def __init__(
        self,
        params,
        lr_U: float = 1e-2,
        lr_B: float = 1e-3,
        lr_V: float = 1e-2,
        lr_other: float = 1e-3,
        ns_steps: int = 10,
        lambda_min: float = 1e-4,
    ):
        defaults = dict(
            lr_U=lr_U, lr_B=lr_B, lr_V=lr_V, lr_other=lr_other,
            ns_steps=ns_steps, lambda_min=lambda_min,
        )

        # Build mapping from parameter to module
        self._param_to_module = {}

        super().__init__(params, defaults)

    def register_module_mapping(self, model: nn.Module):
        """Build mapping from weight parameters to their AdaptiveUBVLinear modules."""
        self._param_to_module = {}
        for name, module in model.named_modules():
            if isinstance(module, AdaptiveUBVLinear):
                self._param_to_module[id(module.weight)] = module

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Check if this is an AdaptiveUBVLinear weight
                module = self._param_to_module.get(id(p))

                if module is not None:
                    # Project gradient and apply Riemannian update
                    G_U, G_B, G_V = module.project_gradient(grad)
                    module.apply_ubv_update(
                        G_U, G_B, G_V,
                        lr_U=group['lr_U'],
                        lr_B=group['lr_B'],
                        lr_V=group['lr_V'],
                        ns_steps=group['ns_steps'],
                        lambda_min=group['lambda_min'],
                    )
                else:
                    # Standard SGD for other params
                    p.data.add_(grad, alpha=-group['lr_other'])

        return loss


def convert_to_adaptive_ubv(
    model: nn.Module,
    r_star: int,
    resync_every: int = 100,
    target_modules: Optional[list] = None,
) -> nn.Module:
    """
    Convert nn.Linear modules to AdaptiveUBVLinear.

    Args:
        model: Model to convert
        r_star: Rank for UBV factorization (or callable)
        resync_every: Steps between re-syncs
        target_modules: List of module name patterns to convert (None = all Linear)

    Returns:
        Model with converted modules
    """
    from .model_surgery import _should_convert, _get_rank_for_module

    replacements = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if target_modules is None or any(t in name for t in target_modules):
                # Determine rank
                if callable(r_star):
                    rank = r_star(name, module)
                else:
                    rank = r_star

                # Create replacement
                new_module = AdaptiveUBVLinear(
                    W_pretrained=module.weight.data,
                    r_star=rank,
                    bias=module.bias.data if module.bias is not None else None,
                    resync_every=resync_every,
                )
                replacements.append((name, new_module))

    # Apply replacements
    for name, new_module in replacements:
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    return model
