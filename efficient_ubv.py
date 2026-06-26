"""
efficient_ubv.py — Memory-efficient frozen-residual UBV adapter.

W = U @ B @ V.T + R

  U, B, V  — trainable nn.Parameters; autograd computes gradients through them.
  R        — frozen buffer; zero gradient, zero backward memory for the residual.

Forward uses parenthesized computation to avoid materializing full W:

    y = x @ R.T  +  ((x @ V) @ B.T) @ U.T

Intermediate tensors are (*, r) rather than (*, out_features), so activation
memory and gradient tensors scale with rank r instead of out_features.

Compared to frozen_residual_ubv/:
  - No full-weight gradient p.grad (out, in); gradients are (out, r), (r, r), (in, r).
  - No weight reconstruction U@B@V.T + R written back to p.data each step.
  - Compatible with gradient checkpointing without materializing full W.
  - Existing frozen_residual_ubv/ files are untouched.

Usage:
    from efficient_ubv import create_efficient_optimizer

    optimizer, n_replaced = create_efficient_optimizer(
        model,
        lr=5e-3,
        energy_threshold=0.25,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.optimizer import Optimizer

# Pure utility functions — imported from existing code, not modified.
from frozen_residual_ubv.projected_gradient_optimizer import (
    energy_truncated_svd,
    logchol_from_spd,
    manifold_muon_admm,
    normalize_update,
    spd_from_logchol,
    spd_logchol_adam_update,
    stiefel_qr_retraction,
)
from frozen_residual_ubv.utils import eigenvalue_clamp, stiefel_project_tangent, symmetric_part

logger = logging.getLogger(__name__)


# ── Module ────────────────────────────────────────────────────────────────────

class FrozenResidualLinear(nn.Module):
    """
    Drop-in nn.Linear replacement with W = U @ B @ V.T + R.

    R is a frozen buffer (no gradient). U, B, V are nn.Parameters so autograd
    tracks gradients through them directly — gradient tensors are (out, r),
    (r, r), (in, r) instead of the full (out, in).

    Forward parenthesizes computation to keep intermediates at rank r:
        h = x @ V          # (*, r)
        y = x @ R.T + (h @ B.T) @ U.T
    """

    def __init__(self, W_pretrained: Tensor, r: int, bias: Optional[Tensor] = None):
        super().__init__()
        out_features, in_features = W_pretrained.shape
        self.out_features = out_features
        self.in_features = in_features
        self.r = r

        orig_dtype = W_pretrained.dtype
        W_f32 = W_pretrained.detach().float()

        # Full SVD for clean initialization (done once at layer swap time)
        U_full, S_full, Vt_full = torch.linalg.svd(W_f32, full_matrices=False)

        r = min(r, len(S_full) - 1)   # always leave at least one component in residual
        r = max(r, 1)

        U0 = U_full[:, :r].to(orig_dtype)         # (out, r)
        S0 = S_full[:r].to(orig_dtype)             # (r,)
        V0 = Vt_full[:r, :].T.to(orig_dtype)       # (in, r)
        B0 = torch.diag(S0)                        # (r, r) diagonal SPD init

        # Frozen residual: tail singular components
        U_tail  = U_full[:, r:]
        S_tail  = S_full[r:]
        Vt_tail = Vt_full[r:, :]
        R = (U_tail * S_tail.unsqueeze(0)) @ Vt_tail   # (out, in), float32
        R = R.to(orig_dtype)

        # Trainable parameters — tagged for the optimizer's dispatch
        self.U = nn.Parameter(U0)
        self.U._ubv_role = "U"

        self.B = nn.Parameter(B0)
        self.B._ubv_role = "B"

        self.V = nn.Parameter(V0)
        self.V._ubv_role = "V"

        # Frozen buffer — no gradient, no grad storage
        self.register_buffer("R", R)

        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        # Parenthesized to keep intermediates at rank r:
        #   y = x @ R.T  +  ((x @ V) @ B.T) @ U.T
        y = x @ self.R.T                    # (*, out) — no grad through R
        h = (x @ self.V) @ self.B.T         # (*, r)  — autograd tracks V, B
        y = y + h @ self.U.T                # (*, out) — autograd tracks U
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, r={self.r}, "
            f"residual={'frozen' if self.R is not None else 'none'}"
        )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        energy_threshold: float,
    ) -> "FrozenResidualLinear":
        W = linear.weight.data
        W_f32 = W.float()

        # Determine rank from energy threshold
        total_energy = (W_f32 ** 2).sum()
        _, S_full, _ = torch.linalg.svd(W_f32, full_matrices=False)
        cumsum = torch.cumsum(S_full ** 2, dim=0) / total_energy
        mask = cumsum >= energy_threshold
        if mask.any():
            r = int(mask.float().argmax().item()) + 1
        else:
            r = len(S_full)
        r = max(1, min(r, min(W.shape) - 1))

        bias = linear.bias.data if linear.bias is not None else None
        return cls(W, r, bias)


# ── Model surgery ──────────────────────────────────────────────────────────────

def swap_to_factored(
    model: nn.Module,
    energy_threshold: float,
    target_modules: Optional[List[str]] = None,
) -> int:
    """
    Replace matching nn.Linear layers with FrozenResidualLinear in-place.
    Safe with device_map="auto": each new module is moved to the same device
    and dtype as the layer it replaces.

    Returns the number of layers replaced.
    """
    n_replaced = 0
    for parent_name, parent_module in model.named_modules():
        for child_name, child_module in list(parent_module.named_children()):
            if not isinstance(child_module, nn.Linear):
                continue
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if target_modules is not None and not any(t in full_name for t in target_modules):
                continue

            device = child_module.weight.device
            dtype = child_module.weight.dtype

            new_mod = FrozenResidualLinear.from_linear(child_module, energy_threshold)
            new_mod = new_mod.to(device=device, dtype=dtype)
            setattr(parent_module, child_name, new_mod)
            n_replaced += 1
            logger.info(
                f"  {full_name}: r={new_mod.r} / {min(new_mod.out_features, new_mod.in_features)}"
                f"  (energy_threshold={energy_threshold})"
            )

    return n_replaced


# ── Optimizer ─────────────────────────────────────────────────────────────────

class FrozenResidualOptimizer(Optimizer):
    """
    Manifold-aware optimizer for FrozenResidualLinear parameters.

    Dispatch by _ubv_role tag on each parameter:
      'U', 'V'  → Stiefel update (muon_admm or projected_qr) + QR retraction.
      'B'       → SPD update: Log-Cholesky Adam (no gradient amplification).
      other     → Plain Adam (biases, embedding weights, etc.).

    stiefel_update choices:
      "muon_admm"    — Manifold MUON via ADMM (Buchanan 2025). Solves the
                       nuclear-norm-constrained tangent problem with admm_steps
                       iterations, giving O(1/k) convergence vs O(1/√k) for
                       subgradient. Recommended for large matrices.
      "projected_qr" — Simple Euclidean gradient projected to tangent space,
                       then QR retraction. Cheaper per step, slightly less
                       theoretically motivated but works well in practice.

    All manifold computations are done in float32 regardless of model dtype.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        # Stiefel (U, V)
        stiefel_update: str = "muon_admm",
        uv_lr_scale: float = 0.3,
        stiefel_max_rms: float = 0.03,
        admm_steps: int = 10,
        admm_rho: float = 4.0,
        # SPD (B) via Log-Cholesky Adam
        b_lr_scale: float = 0.1,
        b_beta1: float = 0.9,
        b_beta2: float = 0.99,
        b_adam_eps: float = 1e-8,
        b_update_rms: float = 0.03,
        b_spd_eps: float = 1e-6,
        b_min_log_diag: float = -8.0,
        b_max_log_diag: float = 8.0,
        # Plain Adam (biases, etc.)
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr,
            stiefel_update=stiefel_update,
            uv_lr_scale=uv_lr_scale,
            stiefel_max_rms=stiefel_max_rms,
            admm_steps=admm_steps,
            admm_rho=admm_rho,
            b_lr_scale=b_lr_scale,
            b_beta1=b_beta1,
            b_beta2=b_beta2,
            b_adam_eps=b_adam_eps,
            b_update_rms=b_update_rms,
            b_spd_eps=b_spd_eps,
            b_min_log_diag=b_min_log_diag,
            b_max_log_diag=b_max_log_diag,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_eps=adam_eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                role = getattr(p, "_ubv_role", None)
                if role in ("U", "V"):
                    self._stiefel_step(p, group)
                elif role == "B":
                    self._spd_step(p, group)
                else:
                    self._adam_step(p, group)

        return loss

    def _stiefel_step(self, p: Tensor, group: dict):
        lr_uv = group["lr"] * group["uv_lr_scale"]
        G = p.grad.float()
        W = p.data.float()

        if not torch.isfinite(G).all():
            return

        method = group.get("stiefel_update", "muon_admm")

        if method == "muon_admm":
            # Manifold MUON (Buchanan 2025): solve nuclear-norm-constrained
            # tangent problem via ADMM. Returns the descent direction A = msign(G + 2W@Λ).
            # Then project to tangent space to fix any residual deviation, clip, retract.
            A_muon = manifold_muon_admm(
                W, G,
                steps=group["admm_steps"],
                rho=group["admm_rho"],
            )
            A_raw = stiefel_project_tangent(W, A_muon)
        elif method == "projected_qr":
            # Euclidean gradient projected to tangent space — cheaper, no inner loop.
            A_raw = stiefel_project_tangent(W, G)
        else:
            raise ValueError(f"Unknown stiefel_update: {method!r}. Use 'muon_admm' or 'projected_qr'.")

        A, _ = normalize_update(A_raw, max_rms=group["stiefel_max_rms"])
        W_new = stiefel_qr_retraction(W - lr_uv * A)
        p.data.copy_(W_new.to(p.dtype))

    def _spd_step(self, p: Tensor, group: dict):
        state = self.state[p]
        lr_b = group["lr"] * group["b_lr_scale"]
        G_B = p.grad.float()
        B = p.data.float()

        if not torch.isfinite(G_B).all():
            return

        # Lazily initialize Log-Cholesky state from current B value
        if "B_A_lower" not in state:
            B_sym = symmetric_part(eigenvalue_clamp(B, lambda_min=group["b_spd_eps"]))
            A_lower, log_diag = logchol_from_spd(B_sym, group["b_spd_eps"])
            state["B_A_lower"] = A_lower
            state["B_log_diag"] = log_diag

        B_new, A_lower_new, log_diag_new, _ = spd_logchol_adam_update(
            A_lower=state["B_A_lower"],
            log_diag=state["B_log_diag"],
            G_B=G_B,
            state=state,
            lr=lr_b,
            beta1=group["b_beta1"],
            beta2=group["b_beta2"],
            adam_eps=group["b_adam_eps"],
            spd_eps=group["b_spd_eps"],
            max_log_diag=group["b_max_log_diag"],
            min_log_diag=group["b_min_log_diag"],
            b_update_rms=group["b_update_rms"],
        )
        state["B_A_lower"] = A_lower_new
        state["B_log_diag"] = log_diag_new
        p.data.copy_(B_new.to(p.dtype))

    def _adam_step(self, p: Tensor, group: dict):
        state = self.state[p]
        grad = p.grad
        lr = group["lr"]
        beta1 = group["adam_beta1"]
        beta2 = group["adam_beta2"]
        eps = group["adam_eps"]

        if "adam_m" not in state:
            state["adam_m"] = torch.zeros_like(p.data)
            state["adam_v"] = torch.zeros_like(p.data)
            state["adam_t"] = 0

        m, v = state["adam_m"], state["adam_v"]
        t = state["adam_t"] + 1

        m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)

        p.data.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
        state["adam_t"] = t

    def get_detailed_metrics(self) -> Dict[str, float]:
        """Aggregate per-layer stats for W&B logging. Same interface as ProjectedGradientOptimizer."""
        ranks, orth_U, orth_V = [], [], []

        for group in self.param_groups:
            for p in group["params"]:
                role = getattr(p, "_ubv_role", None)
                if role == "U":
                    r = p.shape[1]
                    ranks.append(r)
                    I = torch.eye(r, device=p.device, dtype=p.dtype)
                    err = torch.linalg.norm(p.T @ p - I).item()
                    orth_U.append(err)
                elif role == "V":
                    r = p.shape[1]
                    I = torch.eye(r, device=p.device, dtype=p.dtype)
                    err = torch.linalg.norm(p.T @ p - I).item()
                    orth_V.append(err)

        metrics: Dict[str, float] = {}
        if ranks:
            metrics["proj/mean_rank"] = sum(ranks) / len(ranks)
            metrics["proj/min_rank"] = float(min(ranks))
            metrics["proj/max_rank"] = float(max(ranks))
        if orth_U:
            metrics["proj/mean_orthonormality_U"] = sum(orth_U) / len(orth_U)
            metrics["proj/max_orthonormality_U"] = max(orth_U)
        if orth_V:
            metrics["proj/mean_orthonormality_V"] = sum(orth_V) / len(orth_V)
            metrics["proj/max_orthonormality_V"] = max(orth_V)
        return metrics


# ── Entry point ───────────────────────────────────────────────────────────────

def freeze_untargeted_linears(model: nn.Module) -> int:
    """
    Freeze weights of all remaining nn.Linear modules (those not replaced by
    FrozenResidualLinear). Called after swap_to_factored to prevent PyTorch from
    allocating .grad tensors for untargeted parameters (e.g. MoE expert weights)
    during backward. Biases are left as-is.

    Returns the number of weight tensors frozen.
    """
    n_frozen = 0
    for module in model.modules():
        if isinstance(module, nn.Linear) and module.weight.requires_grad:
            module.weight.requires_grad_(False)
            n_frozen += 1
    return n_frozen


def create_efficient_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    energy_threshold: float = 0.9,
    target_modules: Optional[List[str]] = None,
    freeze_untargeted: bool = True,
    # Stiefel (U, V)
    stiefel_update: str = "muon_admm",
    uv_lr_scale: float = 0.3,
    stiefel_max_rms: float = 0.03,
    admm_steps: int = 10,
    admm_rho: float = 4.0,
    # SPD (B)
    b_lr_scale: float = 0.1,
    b_beta1: float = 0.9,
    b_beta2: float = 0.99,
    b_adam_eps: float = 1e-8,
    b_update_rms: float = 0.03,
    b_spd_eps: float = 1e-6,
    b_min_log_diag: float = -8.0,
    b_max_log_diag: float = 8.0,
) -> Tuple["FrozenResidualOptimizer", int]:
    """
    Swap eligible nn.Linear layers to FrozenResidualLinear, then build optimizer.

    If using device_map="auto", call after model loading — swap_to_factored moves
    each new module to the same device/dtype as the layer it replaces.

    Args:
        model: Model to modify in-place.
        lr: Base learning rate.
        energy_threshold: Fraction of spectral energy in trainable rank.
        target_modules: Substrings to match layer names. None = all nn.Linear.
        freeze_untargeted: If True (default), freeze .weight of all nn.Linear
            layers NOT replaced by FrozenResidualLinear. Critical for MoE models
            where untargeted expert weights (gate/up/down_proj) would otherwise
            accumulate .grad tensors (~54 GB for Qwen3-30B-A3B) on first backward.
        stiefel_update: "muon_admm" (Manifold MUON via ADMM) or "projected_qr".
        uv_lr_scale: LR multiplier for U, V.
        stiefel_max_rms: RMS clip threshold for U, V updates.
        admm_steps: ADMM iterations per Stiefel step (only used with muon_admm).
        admm_rho: ADMM penalty parameter (only used with muon_admm).
        b_lr_scale: LR multiplier for B.
        b_beta1/b_beta2/b_adam_eps/b_update_rms: Adam params for B.
        b_spd_eps/b_min_log_diag/b_max_log_diag: SPD numerical stability.

    Returns:
        (optimizer, n_replaced)
    """
    logger.info(
        f"Swapping nn.Linear → FrozenResidualLinear  "
        f"energy_threshold={energy_threshold}  "
        f"stiefel_update={stiefel_update}  admm_steps={admm_steps}  "
        f"target_modules={target_modules or 'all'}"
    )
    n_replaced = swap_to_factored(model, energy_threshold, target_modules)
    logger.info(f"Replaced {n_replaced} layers.")

    if freeze_untargeted and target_modules is not None:
        # Freeze weights of nn.Linear layers not replaced (e.g. MoE experts).
        # Without this, PyTorch allocates .grad for every requires_grad param
        # touched during backward, including ~27B MoE expert params in Qwen3-30B.
        n_frozen = freeze_untargeted_linears(model)
        logger.info(
            f"Froze {n_frozen} untargeted nn.Linear weight tensors "
            f"(prevents .grad allocation for non-UBV params)."
        )

    optimizer = FrozenResidualOptimizer(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        stiefel_update=stiefel_update,
        uv_lr_scale=uv_lr_scale,
        stiefel_max_rms=stiefel_max_rms,
        admm_steps=admm_steps,
        admm_rho=admm_rho,
        b_lr_scale=b_lr_scale,
        b_beta1=b_beta1,
        b_beta2=b_beta2,
        b_adam_eps=b_adam_eps,
        b_update_rms=b_update_rms,
        b_spd_eps=b_spd_eps,
        b_min_log_diag=b_min_log_diag,
        b_max_log_diag=b_max_log_diag,
    )
    return optimizer, n_replaced
