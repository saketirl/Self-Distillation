"""
Composition-Preserving Muon (CompMuon) Optimizer.

Preserves the dominant singular directions of the composed attention maps:
    M_QK^(h) = W_Q^(h)^T @ W_K^(j)   [d_model, d_model]
    M_OV^(h) = W_O^(h)   @ W_V^(j)   [d_model, d_model]

where j = h // (H_q // H_kv) is the KV head index for query head h.

The number of protected directions k is set automatically per head via an
energy threshold on the singular value spectrum of the composed map:
    k = min r  s.t.  sum(S[:r]^2) / sum(S^2) >= qk_energy_threshold

Update strategy for each composition pair (QK / OV):
    strict_composition_fallback() is applied directly — algebraic projection
    achieves near-machine-precision residuals (~1e-7).  Dual ascent is NOT
    used for composition pairs: it diverges because adding U_k-aligned terms
    to G_A via Lambda causes msign to produce D_A aligned WITH span(U_k),
    making the violation larger (positive feedback → residual saturates at 2.0).
    Dual ascent requires a smooth/convex primal; msign is non-smooth.

    Accept/skip logic per head:
      1. Apply strict_composition_fallback() → measure constraint residual.
      2. If residual < fallback_tol → accept.
      3. Else if skip_if_fallback_fails → zero out the update (skip this head).

For GQA, K and V receive the mean D_B over all Q-heads sharing each KV head.
Non-attention 2-D params (MLP matrices) receive a FPMuon-style feature-preserving
update: V_r (right singular vectors capturing fp_energy_threshold of spectral
energy) is computed once at first step and gradient updates are constrained to
satisfy D V_r = 0 via warm-start dual ascent.  When the residual ||D V_r||_F
exceeds residual_fallback_tol the update falls back to the exact projection
G(I - V_r V_r^T).  1-D params receive plain SGD.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer

from .utils import msign, reset_msign_stats, get_msign_stats


# ---------------------------------------------------------------------------
# SVD helpers
# ---------------------------------------------------------------------------

def topk_svd_lowrank_product(
    A: Tensor, B: Tensor, k: int
) -> Tuple[Tensor, Tensor, Tensor]:
    """Top-k SVD of A @ B without materializing the product.

    A: [d_out, r]   B: [r, d_in]
    Returns U_k [d_out, k], S_k [k], V_k [d_in, k].
    """
    Q_A, R_A = torch.linalg.qr(A)
    Q_B, R_B = torch.linalg.qr(B.mT)
    C = R_A @ R_B.mT
    U_C, S, Vh_C = torch.linalg.svd(C, full_matrices=False)
    k = min(k, C.shape[0])
    U_k = Q_A @ U_C[:, :k]
    V_k = Q_B @ Vh_C.mT[:, :k]
    return U_k, S[:k], V_k


def energy_svd_lowrank_product(
    A: Tensor, B: Tensor, energy_threshold: float
) -> Tuple[Tensor, Tensor, Tensor, int]:
    """Energy-truncated SVD of A @ B without materializing the product.

    A: [d_out, r]   B: [r, d_in]
    energy_threshold: fraction of Frobenius-norm-squared to capture.

    k is chosen as the smallest integer such that
        sum(S[:k]^2) / sum(S^2) >= energy_threshold.

    Returns U_k [d_out, k], S_k [k], V_k [d_in, k], k.
    """
    Q_A, R_A = torch.linalg.qr(A)
    Q_B, R_B = torch.linalg.qr(B.mT)
    C = R_A @ R_B.mT                               # [r, r]; same singular values as A@B
    U_C, S, Vh_C = torch.linalg.svd(C, full_matrices=False)

    total_energy = (S ** 2).sum()
    if total_energy < 1e-10:
        k = 1
    else:
        cumsum = torch.cumsum(S ** 2, dim=0) / total_energy
        mask = cumsum >= energy_threshold
        k = int(mask.float().argmax().item()) + 1 if mask.any() else len(S)
    k = max(1, min(k, len(S)))

    U_k = Q_A @ U_C[:, :k]
    V_k = Q_B @ Vh_C.mT[:, :k]
    return U_k, S[:k], V_k, k


# ---------------------------------------------------------------------------
# Constraint residual and strict fallback helpers
# ---------------------------------------------------------------------------

def constraint_residual(
    A: Tensor, B: Tensor,
    Delta_A: Tensor, Delta_B: Tensor,
    U_k: Tensor, V_k: Tensor,
    eps: float = 1e-12,
) -> Tuple[float, float, float]:
    """Relative composition constraint residual.

    A: [d_out, r]   B: [r, d_in]
    Delta_A: [d_out, r]   Delta_B: [r, d_in]
    U_k: [d_out, k]   V_k: [d_in, k]

    Returns (total_residual, left_residual, right_residual) where
    total = (||U_k^T ΔM|| + ||ΔM V_k||) / (||ΔM|| + eps).
    """
    if U_k is None or V_k is None or U_k.shape[1] == 0 or V_k.shape[1] == 0:
        return 0.0, 0.0, 0.0

    Delta_M = Delta_A @ B + A @ Delta_B
    H_L = U_k.T @ Delta_M
    H_R = Delta_M @ V_k
    left = torch.linalg.norm(H_L).item()
    right = torch.linalg.norm(H_R).item()
    denom = torch.linalg.norm(Delta_M).item() + eps
    total = (left + right) / denom
    return total, left, right


def orth(X: Tensor, eps: float = 1e-8) -> Optional[Tensor]:
    """Orthonormal basis for the column span of X, or None if numerically zero."""
    if X is None or X.numel() == 0:
        return None
    if torch.linalg.norm(X).item() < eps:
        return None
    Q, R = torch.linalg.qr(X, mode="reduced")
    diag = torch.abs(torch.diagonal(R))
    rank = int((diag > eps).sum().item())
    if rank == 0:
        return None
    return Q[:, :rank]


def project_left(D: Tensor, Q: Optional[Tensor]) -> Tensor:
    """Project D so that Q.T @ D = 0."""
    if Q is None:
        return D
    return D - Q @ (Q.T @ D)


def project_right(D: Tensor, Q: Optional[Tensor]) -> Tensor:
    """Project D so that D @ Q = 0."""
    if Q is None:
        return D
    return D - (D @ Q) @ Q.T


def strict_composition_fallback(
    A: Tensor, B: Tensor,
    G_A: Tensor, G_B: Tensor,
    U_k: Tensor, V_k: Tensor,
    msign_fn,
    ns_steps: int,
    eps: float,
    orth_eps: float = 1e-8,
    scale_to_proj_norm: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Conservative feasible update for M = A @ B.

    Enforces the stronger sufficient conditions:
        U_k^T Delta_A = 0,    Delta_A @ B @ V_k = 0
        U_k^T @ A @ Delta_B = 0,    Delta_B @ V_k = 0
    which together guarantee U_k^T ΔM = 0 and ΔM V_k = 0.

    scale_to_proj_norm: if True, scale (Delta_A, Delta_B) by
        ‖G_A_perp‖_F / ‖G_A‖_F  and  ‖G_B_perp‖_F / ‖G_B‖_F
    respectively.  This prevents msign from amplifying near-zero projected
    gradients (which arise when the task gradient lies mostly in the blocked
    U_k / V_k directions) into full-magnitude adversarial updates.

    Returns (Delta_A, Delta_B) as descent directions.
    Apply with: param.data.add_(Delta, alpha=lr)
    """
    if U_k is None or V_k is None or U_k.shape[1] == 0 or V_k.shape[1] == 0:
        return (
            -msign_fn(G_A, ns_steps=ns_steps, eps=eps),
            -msign_fn(G_B, ns_steps=ns_steps, eps=eps),
        )

    Q_BV = orth(B @ V_k, eps=orth_eps)
    Q_ATU = orth(A.T @ U_k, eps=orth_eps)

    G_A_perp = project_left(G_A, U_k)
    G_A_perp = project_right(G_A_perp, Q_BV)

    G_B_perp = project_left(G_B, Q_ATU)
    G_B_perp = project_right(G_B_perp, V_k)

    Delta_A = -msign_fn(G_A_perp, ns_steps=ns_steps, eps=eps)
    Delta_B = -msign_fn(G_B_perp, ns_steps=ns_steps, eps=eps)

    # Hard-project to remove any Newton-Schulz leakage.
    Delta_A = project_left(Delta_A, U_k)
    Delta_A = project_right(Delta_A, Q_BV)
    Delta_B = project_left(Delta_B, Q_ATU)
    Delta_B = project_right(Delta_B, V_k)

    if scale_to_proj_norm:
        # Scale by the fraction of gradient norm that survived projection.
        # Ratio in [0, 1]: 0 when task gradient is entirely in blocked directions
        # (no update), 1 when gradient was already orthogonal to the constraint.
        scale_A = torch.linalg.norm(G_A_perp).item() / (torch.linalg.norm(G_A).item() + eps)
        scale_B = torch.linalg.norm(G_B_perp).item() / (torch.linalg.norm(G_B).item() + eps)
        Delta_A = Delta_A * scale_A
        Delta_B = Delta_B * scale_B

    return Delta_A, Delta_B


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class CompositionPreservingMuon(Optimizer):
    """Muon variant that preserves singular directions of M_QK and M_OV.

    Handles only attention Q/K/V/O weight matrices. Non-attention parameters
    must be handled by a separate optimizer (e.g. FeaturePreservingMuon for
    MLP matrices, AdamW for 1-D params).

    The number of protected directions per head is determined automatically by
    qk_energy_threshold / ov_energy_threshold: the smallest k such that the
    top-k singular values of the composed map capture that fraction of total
    spectral energy (Frobenius norm squared).

    Args:
        named_params: list of (name, Parameter) from model.named_parameters().
            Should contain only attention q/k/v/o weight matrices; anything
            else is silently ignored (not added to any param group).
        lr: Learning rate.
        head_dim: Per-head dimension d_h (used as a hint; actual d_h is always
            derived from the K weight shape for correctness).
        num_kv_heads: Number of KV heads H_kv.
        qk_energy_threshold: Fraction of spectral energy of M_QK to protect.
        ov_energy_threshold: Fraction of spectral energy of M_OV to protect.
        preserve_qk: Whether to apply QK composition constraint.
        preserve_ov: Whether to apply OV composition constraint.
        dual_tol: Used to set fallback_tol when fallback_tol is None.
        fallback_tol: Residual threshold to accept the strict_composition_fallback
            result for QK/OV pairs. Default: 10 * dual_tol.
        use_strict_fallback: Kept for API compatibility; strict fallback is always
            used for composition pairs regardless of this flag.
        skip_if_fallback_fails: If True and strict fallback residual > fallback_tol,
            zero out the update for that head.
        clip_comp_gradient: Scale (D_A, D_B) by ‖G_perp‖/‖G‖ before applying.
            Prevents msign from amplifying near-zero projected gradients into
            full-magnitude adversarial updates when the task gradient lies mostly
            in the blocked U_k/V_k directions.
        comp_delta_norm_cap: If not None, cap ‖D_A @ B + A @ D_B‖_F to this value
            by rescaling (D_A, D_B) uniformly. Prevents large but directionally
            feasible composed updates from destabilizing training. The relative
            residual check is directional only and does not catch magnitude blow-ups.
        ns_steps: Newton-Schulz iterations inside msign.
        eps: msign numerical floor.
        debug: Print per-head residuals and fallback state.
        attn_name_patterns: Regex patterns to identify attention params.
    """

    def __init__(
        self,
        named_params: List[Tuple[str, Tensor]],
        lr: float = 1e-3,
        head_dim: int = 80,
        num_kv_heads: int = 8,
        qk_energy_threshold: float = 0.8,
        ov_energy_threshold: float = 0.8,
        preserve_qk: bool = True,
        preserve_ov: bool = True,
        dual_tol: float = 1e-4,
        fallback_tol: Optional[float] = None,
        use_strict_fallback: bool = True,
        skip_if_fallback_fails: bool = True,
        clip_comp_gradient: bool = False,
        comp_delta_norm_cap: Optional[float] = None,
        ns_steps: int = 5,
        eps: float = 1e-7,
        debug: bool = False,
        attn_name_patterns: Optional[List[str]] = None,
    ):
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.qk_energy_threshold = qk_energy_threshold
        self.ov_energy_threshold = ov_energy_threshold
        self.preserve_qk = preserve_qk
        self.preserve_ov = preserve_ov
        self.fallback_tol = fallback_tol if fallback_tol is not None else 10.0 * dual_tol
        self.use_strict_fallback = use_strict_fallback
        self.skip_if_fallback_fails = skip_if_fallback_fails
        self.clip_comp_gradient = clip_comp_gradient
        self.comp_delta_norm_cap = comp_delta_norm_cap
        self.ns_steps = ns_steps
        self.eps = eps
        self.debug = debug

        patterns = attn_name_patterns or [
            r".*\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight",
            r".*\.layers\.(\d+)\.attention\.(q|k|v|o)_proj\.weight",
        ]
        self._patterns = [re.compile(p) for p in patterns]

        self._layer_params: Dict[int, Dict[str, Tensor]] = {}

        for name, p in named_params:
            if not p.requires_grad:
                continue
            for pat in self._patterns:
                m = pat.match(name)
                if m:
                    layer_idx = int(m.group(1))
                    role = m.group(2)
                    self._layer_params.setdefault(layer_idx, {})[role] = p
                    break

        all_params = [p for lp in self._layer_params.values() for p in lp.values()]
        super().__init__(all_params, {"lr": lr})

        # Subspace state: keyed by (layer_idx, "qk"|"ov", head_idx)
        self._subspaces: Dict[Tuple[int, str, int], Dict[str, Tensor]] = {}
        self._initialized = False

        self._last_comp_residuals: List[float] = []
        self._last_residuals: List[float] = []
        self._last_skipped: int = 0
        self._last_total: int = 0
        self._last_msign_stats: dict = {"total": 0, "nan_input": 0, "ns_diverge": 0, "nan_output": 0}

    # ------------------------------------------------------------------
    # Subspace initialization
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _init_subspaces(self) -> None:
        for layer_idx, lp in self._layer_params.items():
            if not {"q", "k", "v", "o"}.issubset(lp):
                continue
            p_Q, p_K, p_V, p_O = lp["q"], lp["k"], lp["v"], lp["o"]
            H_kv = self.num_kv_heads
            d_h = p_K.shape[0] // H_kv
            H_q = p_Q.shape[0] // d_h
            groups = H_q // H_kv

            for h in range(H_q):
                j = h // groups
                if self.preserve_qk:
                    A = p_Q.data[h*d_h:(h+1)*d_h, :].T.float()  # [d_model, d_h]
                    B = p_K.data[j*d_h:(j+1)*d_h, :].float()    # [d_h, d_model]
                    U_k, _, V_k, k = energy_svd_lowrank_product(
                        A, B, self.qk_energy_threshold
                    )
                    self._subspaces[(layer_idx, "qk", h)] = {
                        "U_k": U_k.detach().requires_grad_(False),
                        "V_k": V_k.detach().requires_grad_(False),
                        "k": k,
                    }
                if self.preserve_ov:
                    A = p_O.data[:, h*d_h:(h+1)*d_h].float()    # [d_model, d_h]
                    B = p_V.data[j*d_h:(j+1)*d_h, :].float()    # [d_h, d_model]
                    U_k, _, V_k, k = energy_svd_lowrank_product(
                        A, B, self.ov_energy_threshold
                    )
                    self._subspaces[(layer_idx, "ov", h)] = {
                        "U_k": U_k.detach().requires_grad_(False),
                        "V_k": V_k.detach().requires_grad_(False),
                        "k": k,
                    }

        if self.debug:
            qk_ks = [ss["k"] for key, ss in self._subspaces.items() if key[1] == "qk"]
            ov_ks = [ss["k"] for key, ss in self._subspaces.items() if key[1] == "ov"]
            if qk_ks:
                print(f"[CompMuon] QK k: min={min(qk_ks)} max={max(qk_ks)} "
                      f"mean={sum(qk_ks)/len(qk_ks):.1f} (energy>={self.qk_energy_threshold})")
            if ov_ks:
                print(f"[CompMuon] OV k: min={min(ov_ks)} max={max(ov_ks)} "
                      f"mean={sum(ov_ks)/len(ov_ks):.1f} (energy>={self.ov_energy_threshold})")

        self._initialized = True

    # ------------------------------------------------------------------
    # Per-head composition update via strict algebraic projection
    # ------------------------------------------------------------------

    def _apply_composition_update(
        self,
        A: Tensor, B: Tensor,
        G_A: Tensor, G_B: Tensor,
        U_k: Tensor, V_k: Tensor,
        label: str,
        layer_idx: int,
        head: int,
    ) -> Tuple[Tensor, Tensor]:
        """Strict algebraic projection for one (A, B) composition pair.

        Dual ascent is not used here: it diverges for composition constraints
        because the Lagrange update adds U_k-aligned mass back into G_A, causing
        msign to produce D_A aligned with span(U_k) — a positive feedback loop
        that saturates the residual at the maximum value of 2.0.

        strict_composition_fallback() avoids this by projecting G_A and G_B into
        the null space of the constraint before computing msign, achieving
        near-machine-precision residuals (~1e-7).

        Returns (Delta_A, Delta_B) as descent directions.
        """
        # Guard: non-finite weights or gradients → zero update, no metric recorded.
        if not (torch.isfinite(A).all() and torch.isfinite(B).all()
                and torch.isfinite(G_A).all() and torch.isfinite(G_B).all()):
            if self.debug:
                print(f"[CompPreserveMuon layer={layer_idx} head={head} {label}] "
                      f"NaN/Inf in weights or gradient — zeroing update")
            return torch.zeros_like(G_A), torch.zeros_like(G_B)

        D_A, D_B = strict_composition_fallback(
            A, B, G_A, G_B, U_k, V_k,
            msign, self.ns_steps, self.eps,
            scale_to_proj_norm=self.clip_comp_gradient,
        )

        # Absolute norm cap: ‖D_A @ B + A @ D_B‖_F ≤ comp_delta_norm_cap.
        # The relative residual check is directional only; a large but feasible
        # ΔM can still destabilize training. This enforces a magnitude bound.
        if self.comp_delta_norm_cap is not None:
            Delta_M = D_A @ B + A @ D_B
            dm_norm = torch.linalg.norm(Delta_M).item()
            if dm_norm > self.comp_delta_norm_cap:
                scale = self.comp_delta_norm_cap / (dm_norm + self.eps)
                D_A = D_A * scale
                D_B = D_B * scale

        res, _, _ = constraint_residual(A, B, D_A, D_B, U_k, V_k)
        self._last_comp_residuals.append(res)
        self._last_residuals.append(res)
        self._last_total += 1

        skipped = False
        if res >= self.fallback_tol and self.skip_if_fallback_fails:
            D_A = torch.zeros_like(D_A)
            D_B = torch.zeros_like(D_B)
            skipped = True
            self._last_skipped += 1

        if self.debug:
            k = U_k.shape[1]
            print(
                f"[CompPreserveMuon layer={layer_idx} head={head} {label} k={k}]"
                f" strict_res={res:.3e} skipped={skipped}"
            )

        return D_A, D_B

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self._initialized:
            self._init_subspaces()

        lr = self.param_groups[0]["lr"]
        self._last_comp_residuals = []
        self._last_residuals = []
        self._last_skipped = 0
        self._last_total = 0
        reset_msign_stats()

        # ---- Attention layers ----------------------------------------
        for layer_idx in sorted(self._layer_params):
            lp = self._layer_params[layer_idx]
            if not {"q", "k", "v", "o"}.issubset(lp):
                continue
            p_Q, p_K, p_V, p_O = lp["q"], lp["k"], lp["v"], lp["o"]

            if p_Q.grad is None:
                continue

            H_kv = self.num_kv_heads
            d_h = p_K.shape[0] // H_kv
            H_q = p_Q.shape[0] // d_h
            groups = H_q // H_kv

            D_K_sum = torch.zeros_like(p_K.data.float())
            D_V_sum = torch.zeros_like(p_V.data.float())
            K_cnt = torch.zeros(H_kv, device=p_K.device)
            V_cnt = torch.zeros(H_kv, device=p_V.device)
            D_Q: Dict[int, Tensor] = {}
            D_O: Dict[int, Tensor] = {}

            # ---- QK compositions ----
            for h in range(H_q):
                j = h // groups
                A = p_Q.data[h*d_h:(h+1)*d_h, :].T.float()
                B = p_K.data[j*d_h:(j+1)*d_h, :].float()
                G_A = p_Q.grad[h*d_h:(h+1)*d_h, :].T.float()
                G_B = (p_K.grad[j*d_h:(j+1)*d_h, :].float()
                       if p_K.grad is not None
                       else torch.zeros_like(B))

                if self.preserve_qk:
                    ss = self._subspaces[(layer_idx, "qk", h)]
                    D_A, D_B = self._apply_composition_update(
                        A, B, G_A, G_B, ss["U_k"], ss["V_k"], "QK", layer_idx, h,
                    )
                else:
                    D_A = -msign(G_A, ns_steps=self.ns_steps, eps=self.eps)
                    D_B = -msign(G_B, ns_steps=self.ns_steps, eps=self.eps)

                D_Q[h] = D_A                              # Delta_W_Q_h = D_A^T
                D_K_sum[j*d_h:(j+1)*d_h, :] += D_B
                K_cnt[j] += 1

            # ---- OV compositions ----
            for h in range(H_q):
                j = h // groups
                A = p_O.data[:, h*d_h:(h+1)*d_h].float()
                B = p_V.data[j*d_h:(j+1)*d_h, :].float()
                G_A = (p_O.grad[:, h*d_h:(h+1)*d_h].float()
                       if p_O.grad is not None
                       else torch.zeros_like(A))
                G_B = (p_V.grad[j*d_h:(j+1)*d_h, :].float()
                       if p_V.grad is not None
                       else torch.zeros_like(B))

                if self.preserve_ov:
                    ss = self._subspaces[(layer_idx, "ov", h)]
                    D_A, D_B = self._apply_composition_update(
                        A, B, G_A, G_B, ss["U_k"], ss["V_k"], "OV", layer_idx, h,
                    )
                else:
                    D_A = -msign(G_A, ns_steps=self.ns_steps, eps=self.eps)
                    D_B = -msign(G_B, ns_steps=self.ns_steps, eps=self.eps)

                D_O[h] = D_A                              # Delta_W_O_h = D_A
                D_V_sum[j*d_h:(j+1)*d_h, :] += D_B
                V_cnt[j] += 1

            # ---- Apply attention updates ----
            for h, D_A in D_Q.items():
                p_Q.data[h*d_h:(h+1)*d_h, :].add_(D_A.T.to(p_Q.dtype), alpha=lr)
            for j in range(H_kv):
                cnt = K_cnt[j].item()
                if cnt > 0:
                    p_K.data[j*d_h:(j+1)*d_h, :].add_(
                        (D_K_sum[j*d_h:(j+1)*d_h, :] / cnt).to(p_K.dtype), alpha=lr
                    )
            for h, D_A in D_O.items():
                p_O.data[:, h*d_h:(h+1)*d_h].add_(D_A.to(p_O.dtype), alpha=lr)
            for j in range(H_kv):
                cnt = V_cnt[j].item()
                if cnt > 0:
                    p_V.data[j*d_h:(j+1)*d_h, :].add_(
                        (D_V_sum[j*d_h:(j+1)*d_h, :] / cnt).to(p_V.dtype), alpha=lr
                    )

        self._last_msign_stats = get_msign_stats()

        if self.debug:
            s = self._last_msign_stats
            if s["total"] > 0:
                print(f"[CompMuon] msign: total={s['total']}  "
                      f"nan_input={s['nan_input']}  "
                      f"ns_diverge={s['ns_diverge']}  "
                      f"nan_output={s['nan_output']}")
            if self._last_comp_residuals:
                cr = self._last_comp_residuals
                print(f"[CompMuon] comp_residual mean={sum(cr)/len(cr):.3e}  "
                      f"max={max(cr):.3e}  n={len(cr)}  "
                      f"skip_rate={self._last_skipped/max(self._last_total,1):.3f}")

        return loss

    def get_constraint_metrics(self) -> dict:
        """Aggregated residual norms, skip rates, and msign health from the last step."""
        metrics: dict = {}
        if self._last_comp_residuals:
            cr = self._last_comp_residuals
            metrics["comp_muon/comp_residual_mean"] = sum(cr) / len(cr)
            metrics["comp_muon/comp_residual_max"] = max(cr)
            metrics["comp_muon/skip_rate"] = (
                self._last_skipped / max(self._last_total, 1)
            )
        s = self._last_msign_stats
        if s["total"] > 0:
            metrics["comp_muon/msign_total"]      = s["total"]
            metrics["comp_muon/msign_nan_input"]  = s["nan_input"]
            metrics["comp_muon/msign_ns_diverge"] = s["ns_diverge"]
            metrics["comp_muon/msign_nan_output"] = s["nan_output"]
        return metrics
