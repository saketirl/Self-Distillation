"""
Bures-Wasserstein Riemannian SGD Optimizer for Fixed-Rank Matrix Manifolds.

This optimizer performs Riemannian gradient descent on the manifold of fixed-rank
matrices using a Bures-Wasserstein-style preconditioned metric. The Σ^{-2}
preconditioning amplifies motion along small singular value directions and damps
motion along large singular value directions, improving conditioning for low-rank
optimization problems.

References:
    - Mishra, B. & Sepulchre, R. "Riemannian Preconditioning."
      SIAM J. Optim., 2016.
    - Zhang, F. & Pilanci, M. "Riemannian Preconditioned LoRA for Fine-Tuning
      Foundation Models." ICML 2024.
"""

from typing import Optional, Dict, Any, Iterable, Union
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class BWRiemannianSGD(Optimizer):
    """
    Riemannian SGD with Bures-Wasserstein-style preconditioning for fixed-rank matrices.

    For a matrix A with SVD A = UΣV^T, the Riemannian gradient under the BW metric is:
        grad L(A) = U(U^T G V)V^T + (I - UU^T)GVΣ^{-2}V^T + UΣ^{-2}U^T G(I - VV^T)

    where G is the Euclidean gradient. The Σ^{-2} preconditioning acts as a natural
    gradient that accelerates learning along rank-deficient directions.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate (required).
        rank: Target rank r. Can be:
              - int: Use fixed rank r for all matrices
              - "auto": Compute rank from energy_threshold (default 0.8) or rank_tol
              - None: Use full rank min(m, n) for each matrix (no rank constraint)
        momentum: Heavy-ball momentum coefficient (default: 0.0).
        eps: Regularization for Σ^{-2} to avoid numerical instability.
             Uses (Σ^2 + ε)^{-1} instead of Σ^{-2} (default: 1e-8).
        energy_threshold: When rank="auto", find rank that captures this fraction of
                         total energy (sum of squared singular values). E.g., 0.8 means
                         keep enough singular values to capture 80% of energy.
                         Set to None to use rank_tol instead. (default: 0.8)
        rank_tol: Tolerance for computing numerical rank when rank="auto" and
                  energy_threshold is None. Uses relative threshold tol * σ_max.
                  (default: 1e-5)
        svd_lowrank_niter: Number of subspace iterations for randomized SVD.
                          Higher = more accurate but slower. (default: 2)
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float,
        rank: Optional[Union[int, str]] = None,
        momentum: float = 0.0,
        eps: float = 1e-8,
        energy_threshold: Optional[float] = 0.8,
        rank_tol: float = 1e-5,
        svd_lowrank_niter: int = 2,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if rank is not None and rank != "auto" and (not isinstance(rank, int) or rank < 1):
            raise ValueError(f"Invalid rank: {rank}. Must be int >= 1, 'auto', or None")
        if energy_threshold is not None and (energy_threshold <= 0.0 or energy_threshold > 1.0):
            raise ValueError(f"Invalid energy_threshold: {energy_threshold}. Must be in (0, 1]")
        if svd_lowrank_niter < 1:
            raise ValueError(f"Invalid svd_lowrank_niter: {svd_lowrank_niter}. Must be >= 1")

        defaults = dict(lr=lr, rank=rank, momentum=momentum, eps=eps,
                       energy_threshold=energy_threshold, rank_tol=rank_tol,
                       svd_lowrank_niter=svd_lowrank_niter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> Optional[float]:
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The loss if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            rank = group['rank']
            momentum = group['momentum']
            eps = group['eps']
            energy_threshold = group['energy_threshold']
            rank_tol = group['rank_tol']
            svd_lowrank_niter = group['svd_lowrank_niter']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # For non-2D parameters (biases, 1D, scalars), use vanilla SGD
                if p.dim() != 2:
                    self._vanilla_sgd_step(p, grad, lr, momentum)
                    continue

                # 2D parameter: apply Riemannian update
                self._riemannian_step(p, grad, lr, rank, momentum, eps,
                                      energy_threshold, rank_tol, svd_lowrank_niter)

        return loss

    def _vanilla_sgd_step(
        self,
        p: Tensor,
        grad: Tensor,
        lr: float,
        momentum: float
    ) -> None:
        """Vanilla SGD step for non-matrix parameters."""
        state = self.state[p]

        if momentum != 0:
            if 'momentum_buffer' not in state:
                state['momentum_buffer'] = torch.zeros_like(grad)
            buf = state['momentum_buffer']
            buf.mul_(momentum).add_(grad)
            p.data.add_(buf, alpha=-lr)
        else:
            p.data.add_(grad, alpha=-lr)

    def _riemannian_step(
        self,
        p: Tensor,
        grad: Tensor,
        lr: float,
        rank: Optional[Union[int, str]],
        momentum: float,
        eps: float,
        energy_threshold: Optional[float],
        rank_tol: float,
        svd_lowrank_niter: int,
    ) -> None:
        """
        Riemannian gradient descent step on fixed-rank matrix manifold.

        Computes: A <- A - η * grad_riem(L)
        where grad_riem is the Riemannian gradient under the BW metric.

        Uses torch.svd_lowrank for efficient low-rank SVD when r < min(m, n).
        """
        state = self.state[p]
        device = p.device
        original_dtype = p.dtype

        # Cast to float32 for numerical stability in SVD
        if original_dtype in (torch.float16, torch.bfloat16):
            A = p.data.float()
            G = grad.float()
        else:
            A = p.data
            G = grad

        m, n = A.shape
        k = min(m, n)

        # Determine effective rank
        if rank == "auto":
            # Use stored rank from first step, or compute it now
            if 'auto_rank' not in state:
                # Compute FULL SVD to determine rank (need all singular values)
                _, S_init, _ = torch.linalg.svd(A, full_matrices=False)

                if energy_threshold is not None:
                    # Energy-based rank: find smallest r such that
                    # sum(S[:r]^2) / sum(S^2) >= energy_threshold
                    S_sq = S_init ** 2
                    total_energy = S_sq.sum()
                    if total_energy > 0:
                        cumsum_energy = torch.cumsum(S_sq, dim=0)
                        threshold_energy = energy_threshold * total_energy
                        # Find first index where cumsum >= threshold
                        mask = cumsum_energy >= threshold_energy
                        if mask.any():
                            state['auto_rank'] = int(mask.nonzero()[0].item()) + 1
                        else:
                            state['auto_rank'] = k  # Use full rank if threshold not reached
                    else:
                        state['auto_rank'] = 1  # Degenerate case
                else:
                    # Tolerance-based rank: count singular values above tol * σ_max
                    threshold = rank_tol * S_init[0] if S_init[0] > 0 else rank_tol
                    state['auto_rank'] = int((S_init > threshold).sum().item())

                state['auto_rank'] = max(1, state['auto_rank'])  # At least rank 1
            r = state['auto_rank']
        elif rank is not None:
            r = rank
        else:
            r = k  # Full rank

        r = min(r, k)  # Can't exceed min(m, n)

        # Use low-rank SVD when r < k for efficiency, otherwise full SVD
        use_lowrank = r < k
        try:
            if use_lowrank:
                # torch.svd_lowrank returns (U, S, V) where V is NOT transposed
                U, S, V = torch.svd_lowrank(A, q=r, niter=svd_lowrank_niter)
                # U: (m, r), S: (r,), V: (n, r)
            else:
                # Full SVD for full-rank case
                U, S, Vh = torch.linalg.svd(A, full_matrices=False)
                V = Vh.T  # (n, k)
        except RuntimeError as e:
            raise RuntimeError(
                f"SVD failed for parameter of shape {p.shape}. "
                f"This may indicate numerical issues (e.g., NaN/Inf values). "
                f"Original error: {e}"
            )

        # Check for degenerate case (near-zero matrix)
        if S.max() < eps:
            # Fall back to Euclidean step
            if momentum != 0:
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad)
                p.data.add_(buf, alpha=-lr)
            else:
                p.data.add_(grad, alpha=-lr)
            return

        # V is already (n, r), compute Vh = V.T for formulas
        Vh = V.T  # (r, n)

        # Regularized inverse squared singular values: (σ^2 + ε)^{-1}
        sigma_inv2 = 1.0 / (S ** 2 + eps)  # (r,)

        # Precompute common terms
        UtG = U.T @ G           # (r, n)
        GV = G @ V              # (m, r)
        UtGV = UtG @ V          # (r, r) - this is U^T G V

        # ============================================================
        # Center term: U (U^T G V) V^T
        # This is the tangent component in the "core" subspace
        # ============================================================
        center = U @ UtGV @ Vh  # (m, n)

        # ============================================================
        # Left-horizontal term: (I - UU^T) G V Σ^{-2} V^T
        # = (GV - U(U^T G V)) * σ^{-2} @ V^T
        # This captures motion in the left singular vector directions
        # ============================================================
        left_inner = GV - U @ UtGV                    # (m, r): (I - UU^T) G V
        left_scaled = left_inner * sigma_inv2[None, :]  # (m, r): row-wise scaling by σ^{-2}
        left_horizontal = left_scaled @ Vh            # (m, n)

        # ============================================================
        # Right-horizontal term: U Σ^{-2} U^T G (I - VV^T)
        # = U @ (σ^{-2} * (U^T G - (U^T G V) V^T))
        # This captures motion in the right singular vector directions
        # ============================================================
        right_inner = UtG - UtGV @ Vh                   # (r, n): U^T G (I - VV^T)
        right_scaled = sigma_inv2[:, None] * right_inner  # (r, n): col-wise scaling by σ^{-2}
        right_horizontal = U @ right_scaled             # (m, n)

        # Sum the three components to get full Riemannian gradient
        grad_riem = center + left_horizontal + right_horizontal

        # Apply momentum
        if momentum != 0:
            if 'momentum_buffer' not in state:
                state['momentum_buffer'] = torch.zeros_like(p.data)
            buf = state['momentum_buffer']
            buf.mul_(momentum).add_(grad_riem)
            A_new = A - lr * buf
        else:
            A_new = A - lr * grad_riem

        # ============================================================
        # Retraction: project back to rank-r manifold via SVD truncation
        # This ensures we stay on the fixed-rank manifold after the step
        # Only apply if we have a rank constraint (explicit rank or "auto")
        # Uses low-rank SVD for efficiency
        # ============================================================
        if r < k:
            # Use low-rank SVD for efficient retraction
            U_new, S_new, V_new = torch.svd_lowrank(A_new, q=r, niter=svd_lowrank_niter)
            # Reconstruct rank-r matrix: U @ diag(S) @ V^T
            A_new = U_new @ torch.diag(S_new) @ V_new.T

        # Cast back to original dtype if needed
        if original_dtype in (torch.float16, torch.bfloat16):
            A_new = A_new.to(original_dtype)

        p.data.copy_(A_new)

    def get_detected_ranks(self) -> Dict[str, int]:
        """
        Returns a dictionary mapping parameter IDs to their detected auto-ranks.

        Only populated after at least one step() call when rank="auto".
        Useful for debugging and logging the energy-based rank selection.

        Returns:
            Dict mapping parameter id (str) to detected rank (int).
        """
        ranks = {}
        for group in self.param_groups:
            for p in group['params']:
                state = self.state.get(p, {})
                if 'auto_rank' in state:
                    ranks[id(p)] = state['auto_rank']
        return ranks

    def get_rank_summary(self) -> str:
        """
        Returns a human-readable summary of detected ranks by layer type.

        Useful for logging the rank distribution across the model.

        Returns:
            A formatted string summarizing detected ranks.
        """
        ranks = self.get_detected_ranks()
        if not ranks:
            return "No auto-ranks detected yet (call step() first)"

        rank_list = list(ranks.values())
        lines = [
            f"Detected {len(rank_list)} auto-ranks:",
            f"  Min: {min(rank_list)}",
            f"  Max: {max(rank_list)}",
            f"  Mean: {sum(rank_list) / len(rank_list):.1f}",
        ]
        return "\n".join(lines)
