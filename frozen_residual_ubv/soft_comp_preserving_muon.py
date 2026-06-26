"""
SoftCompPreservingMuon
======================
Preserves the soft k×k routing/content-interaction core of composed attention
matrices   M = AB   under the linearised constraint  U_k^T ΔM V_k = 0.

Unlike CompositionPreservingMuon (which enforces the stronger U_k^T ΔM = 0
and ΔM V_k = 0), this optimizer only constrains the k×k interaction between
the top-k left and right singular spaces of M_0.  The dual variable is k×k
rather than proportional to the full matrix dimension, keeping the iteration
cheap and the dual ascent stable.

k is chosen per head via an energy threshold (same convention as
FeaturePreservingMuon):  k = min{j : Σ_{i≤j} σ_i² / Σ_i σ_i² ≥ threshold}

Dual ascent is well-behaved here because the msign non-smoothness only appears
at the primal; the k×k constraint Jacobian contracts the dual space.

Algorithm per composition pair (A, B)
--------------------------------------
Initialization:
    M_0 = U_k Σ_k V_k^T    (full SVD via factored QR; k from energy threshold)
    Λ   = 0_{k×k}           (persists across steps — warm-start dual)

Each step (Sylvester projection):
    cache:  BV = B @ V_k       [r, k]
            UA = U_k^T @ A     [k, r]
            P  = UA @ UA^T     [k, k]   = U_k^T A A^T U_k
            Q  = BV^T @ BV     [k, k]   = V_k^T B^T B V_k

    1. Δ_A = −msign(G_A),  Δ_B = −msign(G_B)          (plain Muon directions)
    2. H   = (U_k^T Δ_A) BV + UA (Δ_B V_k)            (constraint violation, [k,k])
    3. Solve  P Λ + Λ Q = H                             (Sylvester, Kronecker k²×k² solve)
    4. Δ_A −= U_k (Λ BV^T),  Δ_B −= UA^T (Λ V_k^T)   (project onto constraint manifold)
    5. If ‖H_corrected‖_F/√k² > hard_fallback_tol: zero both updates

    apply: A += lr · Δ_A,  B += lr · Δ_B

Conventions (matching CompositionPreservingMuon):
    QK:  A = W_Q_h^T [d_model, d_h],  B = W_K_j [d_h, d_model]
         apply: W_Q_h += Δ_A^T · lr,  W_K_j += Δ_B · lr  (GQA-averaged)
    OV:  A = W_O_h   [d_model, d_h],  B = W_V_j [d_h, d_model]
         apply: W_O_h += Δ_A · lr,    W_V_j += Δ_B · lr  (GQA-averaged)

Monitoring (per head):
    core_t = (U_k^T A)(B V_k)               [k, k]  (avoids d_model² product)
    drift   = ‖core_t − diag(Σ_k)‖_F / (‖Σ_k‖_F + ε)
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer

from .utils import msign, reset_msign_stats, get_msign_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _factored_svd(A: Tensor, B: Tensor, k: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Top-k SVD of M = A @ B without forming M explicitly.

    A : [m, r]   B : [r, n]   k : number of singular values (≤ r)

    Uses the factored QR trick:
        B^T = Q_B R_B  →  C = A R_B^T  →  SVD(C) = U_C S_C Vh_C
        U_M = U_C[:, :k],  S_M = S_C[:k],  V_M = Q_B Vh_C[:k]^T

    Cost: O(n·r + m·r² + m·r·k) instead of forming the m×n matrix M.

    Returns U_k [m, k], S_k [k], V_k [n, k].
    """
    Q_B, R_B = torch.linalg.qr(B.T)              # Q_B [n, r'], R_B [r', r]
    C = A @ R_B.T                                  # [m, r']
    U_C, S_C, Vh_C = torch.linalg.svd(C, full_matrices=False)
    k_act = min(k, S_C.shape[0])
    return (
        U_C[:, :k_act].contiguous(),               # U_k [m, k]
        S_C[:k_act].contiguous(),                   # S_k [k]
        (Q_B @ Vh_C[:k_act].T).contiguous(),        # V_k [n, k]
    )


def _energy_rank(S: Tensor, threshold: float) -> int:
    """Smallest k s.t. sum(S[:k]^2) / sum(S^2) >= threshold.

    Returns 0 when threshold <= 0 (no constraint), 1 when S is near-zero.
    """
    if threshold <= 0.0:
        return 0
    energy = S ** 2
    total = energy.sum()
    if total < 1e-12:
        return 1
    cumulative = energy.cumsum(0) / total
    k = int((cumulative < threshold).sum().item()) + 1
    return min(k, S.shape[0])


def _factored_svd_energy(
    A: Tensor, B: Tensor, energy_threshold: float
) -> Tuple[Tensor, Tensor, Tensor, int, float]:
    """Full SVD of M = A@B via the QR trick, then energy-based rank truncation.

    Computes all singular values of M (rank ≤ min(r, m, n)) then selects
    k = min{j : Σ_{i≤j} σ_i² / Σ_i σ_i² ≥ energy_threshold}.

    Returns U_k [m,k], S_k [k], V_k [n,k], k_act, captured_energy.
    Returns empty (k=0) tensors when energy_threshold <= 0.
    """
    if energy_threshold <= 0.0:
        m, n = A.shape[0], B.shape[1]
        return (
            torch.zeros(m, 0, device=A.device, dtype=A.dtype),
            torch.zeros(0, device=A.device, dtype=A.dtype),
            torch.zeros(n, 0, device=B.device, dtype=B.dtype),
            0, 0.0,
        )
    Q_B, R_B = torch.linalg.qr(B.T)
    C = A @ R_B.T
    U_C, S_C, Vh_C = torch.linalg.svd(C, full_matrices=False)
    k = _energy_rank(S_C, energy_threshold)
    if k == 0:
        m, n = A.shape[0], B.shape[1]
        return (
            torch.zeros(m, 0, device=A.device, dtype=A.dtype),
            torch.zeros(0, device=A.device, dtype=A.dtype),
            torch.zeros(n, 0, device=B.device, dtype=B.dtype),
            0, 0.0,
        )
    captured = float((S_C[:k] ** 2).sum() / (S_C ** 2).sum().clamp(min=1e-12))
    return (
        U_C[:, :k].contiguous(),
        S_C[:k].contiguous(),
        (Q_B @ Vh_C[:k].T).contiguous(),
        k,
        captured,
    )


def _solve_sylvester_damped(
    P: Tensor, Q: Tensor, R: Tensor, damping: float = 1e-4
) -> Tensor:
    """Solve  P X + X Q = R  via Kronecker vectorisation.

    Returns X [k, k].  P and Q are regularised by adding damping*I before
    solving to keep the k²×k² system well-conditioned.

    Kronecker identity:  vec(PX + XQ) = (P⊗I + I⊗Q^T) vec(X)  [row-major].
    So (P⊗I + I⊗Q^T) vec(X) = vec(R).
    """
    k = P.shape[0]
    I = torch.eye(k, device=P.device, dtype=P.dtype)
    P_reg = P + damping * I
    Q_reg = Q + damping * I
    # Row-major (PyTorch) identity: rvec(PX + XQ) = (P⊗I + I⊗Q^T) rvec(X).
    # kron(P,I) handles the PX term; kron(I,Q^T) handles the XQ term.
    kron_PI  = torch.einsum("ij,kl->ikjl", P_reg, I).reshape(k * k, k * k)
    kron_IQT = torch.einsum("ij,kl->ikjl", I, Q_reg.T.contiguous()).reshape(k * k, k * k)
    sol = torch.linalg.solve(kron_PI + kron_IQT, R.reshape(-1, 1))
    return sol.reshape(k, k)


def _safe_msign(G: Tensor, ns_steps: int, eps: float) -> Tensor:
    """msign with pre-call finite/norm guard.

    Returns zeros_like(G) when G is non-finite or has Frobenius norm < 1e-12,
    preventing msign from amplifying degenerate inputs into adversarial updates.
    """
    if not torch.isfinite(G).all():
        return torch.zeros_like(G)
    g_norm = torch.linalg.norm(G).item()
    if not math.isfinite(g_norm) or g_norm < 1e-12:
        return torch.zeros_like(G)
    return msign(G, ns_steps=ns_steps, eps=eps)


# ---------------------------------------------------------------------------
# SoftCompPreservingMuon
# ---------------------------------------------------------------------------

class SoftCompPreservingMuon(Optimizer):
    """
    Muon optimizer that preserves the k×k soft core of composed attention matrices.

    Hyperparameters
    ---------------
    lr                   : base learning rate
    head_dim             : d_h  (key/value head dimension)
    num_kv_heads         : H_kv (number of KV heads, for GQA)
    qk_energy_threshold  : fraction of QK spectral energy to protect per head;
                           k = min{j : Σ_{i≤j}σ_i²/Σσ_i² ≥ threshold}  (0 = plain Muon)
    ov_energy_threshold  : same for OV composition  (0 = plain Muon)
    preserve_qk          : whether to apply the QK core constraint
    preserve_ov          : whether to apply the OV core constraint
    ns_steps             : Newton-Schulz iterations inside msign
    eps                  : numerical floor for msign and drift normalization
    hard_fallback_tol    : zero updates when Sylvester-corrected ‖H‖_F/√k² > this
    sylvester_damping    : regularisation added to P and Q before the k²×k² solve
    debug                : print per-head diagnostics each step
    attn_name_patterns   : regex patterns used to identify attention weights

    Algorithm (per composition pair A, B):
        1. Compute plain Muon directions: Δ_A = −msign(G_A), Δ_B = −msign(G_B)
        2. Measure constraint violation: H = U_k^T (Δ_A B + A Δ_B) V_k  [k×k]
        3. Solve Sylvester equation P Λ + Λ Q = H (closed-form via Kronecker)
           where P = (U_k^T A)(A^T U_k), Q = (V_k^T B^T)(B V_k)
        4. Apply correction: Δ_A −= U_k (Λ BV^T), Δ_B −= (UA^T) (Λ V_k^T)
        5. If corrected ‖H‖_F/√k² > hard_fallback_tol: zero both updates.

    The Sylvester equation is the one-shot closed-form projection of the plain-Muon
    directions onto the constraint manifold (derived by linearising msign ≈ identity,
    which gives exactly P Λ + Λ Q = H as the stationarity condition for Λ).
    """

    def __init__(
        self,
        named_params: List[Tuple[str, Tensor]],
        lr: float = 1e-3,
        head_dim: int = 128,
        num_kv_heads: int = 8,
        qk_energy_threshold: float = 0.8,
        ov_energy_threshold: float = 0.8,
        preserve_qk: bool = True,
        preserve_ov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-7,
        debug: bool = False,
        attn_name_patterns: Optional[List[str]] = None,
        hard_fallback_tol: float = 0.5,
        sylvester_damping: float = 1e-4,
        sylvester_skip_tol: float = 0.0,
        # Deprecated — kept for backward-compat but ignored
        admm_steps: int = 0,
        dual_lr: float = 0.0,
        dual_tol: float = 0.0,
        use_sylvester_fallback: bool = True,
        fallback_tol: float = 0.0,
    ):
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.qk_energy_threshold = qk_energy_threshold
        self.ov_energy_threshold = ov_energy_threshold
        self.preserve_qk = preserve_qk
        self.preserve_ov = preserve_ov
        self.ns_steps = ns_steps
        self.eps = eps
        self.debug = debug
        self.hard_fallback_tol = hard_fallback_tol
        self.sylvester_damping = sylvester_damping
        self.sylvester_skip_tol = sylvester_skip_tol

        patterns = attn_name_patterns or [
            r".*\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight",
            r".*\.layers\.(\d+)\.attention\.(q|k|v|o)_proj\.weight",
        ]
        self._patterns = [re.compile(p) for p in patterns]

        self._layer_params: Dict[int, Dict[str, Tensor]] = {}
        all_params: List[Tensor] = []

        for name, p in named_params:
            if not p.requires_grad:
                continue
            for pat in self._patterns:
                m = pat.match(name)
                if m:
                    layer_idx = int(m.group(1))
                    proj = m.group(2)
                    self._layer_params.setdefault(layer_idx, {})[proj] = p
                    all_params.append(p)
                    break

        if not all_params:
            raise ValueError(
                "SoftCompPreservingMuon: no attention parameters matched. "
                "Check attn_name_patterns against your model's parameter names."
            )

        super().__init__(all_params, dict(lr=lr))

        self._subspaces: Dict[Tuple[int, str, int], Dict] = {}
        self._initialized = False

        # Per-step diagnostics (reset at start of each step)
        self._last_core_drifts: List[float] = []
        self._last_H_norms: List[float] = []
        self._last_dual_iters: List[int] = []
        self._last_msign_stats: dict = {
            "total": 0, "nan_input": 0, "ns_diverge": 0, "nan_output": 0,
        }
        self._last_resid_before_fallback: List[float] = []
        self._last_resid_after_fallback: List[float] = []
        self._last_sylvester_used: List[bool] = []
        self._last_sylvester_failed: List[bool] = []

        # Cumulative fallback counters (persist across steps; reset via reset_fallback_stats)
        # "soft fallback"  = Sylvester was applied AND succeeded (resid2 ≤ hard_fallback_tol)
        # "hard fallback"  = Sylvester was applied but failed → updates zeroed
        self._step_count: int = 0
        self._cum_heads: int = 0        # total constrained head-pairs processed
        self._cum_soft: int = 0         # soft fallbacks (Sylvester succeeded)
        self._cum_hard: int = 0         # hard fallbacks (updates zeroed)

    # ------------------------------------------------------------------
    # Subspace initialization
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _init_subspaces(self) -> None:
        """Compute and cache top-k SVD of M_0 = A_0 B_0 for every composition pair."""
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

                if self.preserve_qk and self.qk_energy_threshold > 0:
                    A = p_Q.data[h*d_h:(h+1)*d_h, :].float().T   # [d_model, d_h]
                    B = p_K.data[j*d_h:(j+1)*d_h, :].float()      # [d_h, d_model]
                    U_k, S_k, V_k, k_act, cap = _factored_svd_energy(A, B, self.qk_energy_threshold)
                    if k_act == 0:
                        continue
                    # H_scale: plausible magnitude of the constraint matrix H at init.
                    # H = (U_k^T ΔA) BV + UA (ΔB V_k), each term ≤ sqrt(k)*||connector||_F.
                    # Storing ||BV_init||_F + ||UA_init||_F makes hard_fallback_tol
                    # relative to weight magnitude rather than absolute.
                    BV_init = B @ V_k          # [d_h, k]
                    UA_init = U_k.T @ A        # [k, d_h]
                    H_scale = max(
                        (torch.linalg.norm(BV_init) + torch.linalg.norm(UA_init)).item(),
                        1e-6,
                    )
                    self._subspaces[(layer_idx, "qk", h)] = {
                        "U": U_k, "S": S_k, "V": V_k,
                        "captured_energy": cap,
                        "H_scale": H_scale,
                    }

                if self.preserve_ov and self.ov_energy_threshold > 0:
                    A = p_O.data[:, h*d_h:(h+1)*d_h].float()      # [d_model, d_h]
                    B = p_V.data[j*d_h:(j+1)*d_h, :].float()      # [d_h, d_model]
                    U_k, S_k, V_k, k_act, cap = _factored_svd_energy(A, B, self.ov_energy_threshold)
                    if k_act == 0:
                        continue
                    BV_init = B @ V_k
                    UA_init = U_k.T @ A
                    H_scale = max(
                        (torch.linalg.norm(BV_init) + torch.linalg.norm(UA_init)).item(),
                        1e-6,
                    )
                    self._subspaces[(layer_idx, "ov", h)] = {
                        "U": U_k, "S": S_k, "V": V_k,
                        "captured_energy": cap,
                        "H_scale": H_scale,
                    }

        self._initialized = True
        if self.debug:
            n_qk = sum(1 for k in self._subspaces if k[1] == "qk")
            n_ov = sum(1 for k in self._subspaces if k[1] == "ov")
            k_vals_qk = [self._subspaces[k]["S"].shape[0] for k in self._subspaces if k[1] == "qk"]
            k_vals_ov = [self._subspaces[k]["S"].shape[0] for k in self._subspaces if k[1] == "ov"]
            print(
                f"[SoftCompMuon] Initialized {n_qk} QK (k={set(k_vals_qk)}, "
                f"energy_thr={self.qk_energy_threshold}) + "
                f"{n_ov} OV (k={set(k_vals_ov)}, energy_thr={self.ov_energy_threshold}) subspaces"
            )

    # ------------------------------------------------------------------
    # Per-pair soft-core dual ascent
    # ------------------------------------------------------------------

    def _soft_core_update(
        self,
        A: Tensor,    # [m, r]  current weight (float32)
        B: Tensor,    # [r, n]  current weight (float32)
        G_A: Tensor,  # [m, r]  loss gradient w.r.t. A
        G_B: Tensor,  # [r, n]  loss gradient w.r.t. B
        sub: Dict,    # subspace dict: U, S, V
        label: str,
        layer_idx: int,
        head: int,
    ) -> Tuple[Tensor, Tensor]:
        """Return (Δ_A, Δ_B) satisfying U_k^T (Δ_A B + A Δ_B) V_k ≈ 0.

        Algorithm:
            1. Plain Muon: Δ_A = −msign(G_A),  Δ_B = −msign(G_B)
            2. Measure constraint violation: H = U_k^T (Δ_A BV + UA Δ_B V_k)
            3. Solve Sylvester P Λ + Λ Q = H  (closed-form Kronecker solve)
               — this is the one-shot projection of the plain-Muon directions
                 onto the constraint manifold, derived by linearising msign ≈ I.
            4. Correct: Δ_A −= U_k (Λ BV^T),  Δ_B −= UA^T (Λ V_k^T)
            5. If corrected ‖H‖/√k² > hard_fallback_tol: zero both updates.
        """
        U_k   = sub["U"]   # [m, k]
        S_k   = sub["S"]   # [k]
        V_k   = sub["V"]   # [n, k]
        k_act = S_k.shape[0]
        k2    = k_act * k_act

        BV = B @ V_k    # [r, k]
        UA = U_k.T @ A  # [k, r]

        # ---- Step 1: plain Muon directions (unconstrained) -----------------
        Delta_A = -_safe_msign(G_A, self.ns_steps, self.eps)
        Delta_B = -_safe_msign(G_B, self.ns_steps, self.eps)

        # ---- Step 2: constraint violation ----------------------------------
        H = (U_k.T @ Delta_A) @ BV + UA @ (Delta_B @ V_k)
        H_norm = torch.linalg.norm(H).item() / math.sqrt(max(k2, 1))

        resid_before   = H_norm
        resid_after    = H_norm
        sylvester_failed = False

        # ---- Step 3-5: Sylvester projection --------------------------------
        # Skip when H_norm is already small relative to the plausible H scale
        # (‖BV_init‖_F + ‖UA_init‖_F, computed once at init). This avoids the
        # O(k³) Sylvester solve when the unconstrained step barely violates the
        # constraint. sylvester_skip_tol=0 always solves; ~0.05 skips most steps
        # where the gradient is already nearly orthogonal to the protected core.
        H_scale = sub.get("H_scale", 1.0)
        if k_act > 0 and resid_before > self.sylvester_skip_tol * H_scale:
            P = UA @ UA.T   # [k, k]  P = (U_k^T A)(A^T U_k)
            Q = BV.T @ BV   # [k, k]  Q = (V_k^T B^T)(B V_k)
            try:
                Lambda_corr = _solve_sylvester_damped(
                    P, Q, H, damping=self.sylvester_damping
                )
                Delta_A = Delta_A - U_k @ (Lambda_corr @ BV.T)
                Delta_B = Delta_B - UA.T @ (Lambda_corr @ V_k.T)
                H2 = (U_k.T @ Delta_A) @ BV + UA @ (Delta_B @ V_k)
                resid2 = torch.linalg.norm(H2).item() / math.sqrt(max(k2, 1))
                resid_after = resid2
                if not math.isfinite(resid2) or resid2 > self.hard_fallback_tol * H_scale:
                    sylvester_failed = True
                    Delta_A = torch.zeros_like(Delta_A)
                    Delta_B = torch.zeros_like(Delta_B)
            except Exception:
                sylvester_failed = True
                resid_after = float("nan")
                Delta_A = torch.zeros_like(Delta_A)
                Delta_B = torch.zeros_like(Delta_B)

        # ---- Core drift ----------------------------------------------------
        core = UA @ BV
        S_diag = torch.diag(S_k)
        drift = (torch.linalg.norm(core - S_diag).item() /
                 (torch.linalg.norm(S_k).item() + self.eps))

        self._last_core_drifts.append(drift)
        self._last_H_norms.append(H_norm)
        self._last_dual_iters.append(0)
        self._last_resid_before_fallback.append(resid_before)
        self._last_resid_after_fallback.append(resid_after)
        self._last_sylvester_used.append(k_act > 0)
        self._last_sylvester_failed.append(sylvester_failed)

        if self.debug:
            status = "FAILED" if sylvester_failed else f"after={resid_after:.2e}"
            print(
                f"[SoftCompMuon layer={layer_idx} head={head} {label} k={k_act}] "
                f"H_norm={H_norm:.3e}  drift={drift:.3e}  sylvester: before={resid_before:.2e} {status}"
            )

        return Delta_A, Delta_B

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
        self._last_core_drifts = []
        self._last_H_norms = []
        self._last_dual_iters = []
        self._last_resid_before_fallback = []
        self._last_resid_after_fallback = []
        self._last_sylvester_used = []
        self._last_sylvester_failed = []
        reset_msign_stats()

        use_dual_qk = self.preserve_qk and self.qk_energy_threshold > 0
        use_dual_ov = self.preserve_ov and self.ov_energy_threshold > 0

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

            # Per-Q-head updates — stored in [d_h, d_model] (W_Q_h) and
            # [d_model, d_h] (W_O_h) shapes so apply is uniform.
            D_Q: Dict[int, Tensor] = {}   # h → update for W_Q_h [d_h, d_model]
            D_O: Dict[int, Tensor] = {}   # h → update for W_O column h [d_model, d_h]

            # KV accumulation buffers (averaged over Q-heads sharing each KV head)
            D_K_sum = torch.zeros_like(p_K.data.float())
            D_V_sum = torch.zeros_like(p_V.data.float())
            K_cnt = torch.zeros(H_kv, device=p_K.device)
            V_cnt = torch.zeros(H_kv, device=p_V.device)

            for h in range(H_q):
                j = h // groups

                # ---- QK ------------------------------------------------
                G_Q_h = p_Q.grad[h*d_h:(h+1)*d_h, :].float()   # [d_h, d_model]

                if use_dual_qk:
                    A   = p_Q.data[h*d_h:(h+1)*d_h, :].float().T    # [d_model, d_h]
                    B   = p_K.data[j*d_h:(j+1)*d_h, :].float()      # [d_h, d_model]
                    G_A = G_Q_h.T                                      # [d_model, d_h]
                    G_B = (p_K.grad[j*d_h:(j+1)*d_h, :].float()
                           if p_K.grad is not None
                           else torch.zeros_like(B))
                    sub = self._subspaces[(layer_idx, "qk", h)]
                    D_A, D_B = self._soft_core_update(
                        A, B, G_A, G_B, sub, "QK", layer_idx, h
                    )
                    D_Q[h] = D_A.T                         # [d_h, d_model]
                    D_K_sum[j*d_h:(j+1)*d_h, :].add_(D_B)
                    K_cnt[j] += 1
                else:
                    D_Q[h] = -_safe_msign(G_Q_h, self.ns_steps, self.eps)

                # ---- OV ------------------------------------------------
                G_O_h = (p_O.grad[:, h*d_h:(h+1)*d_h].float()
                          if p_O.grad is not None
                          else None)

                if use_dual_ov:
                    A   = p_O.data[:, h*d_h:(h+1)*d_h].float()       # [d_model, d_h]
                    B   = p_V.data[j*d_h:(j+1)*d_h, :].float()       # [d_h, d_model]
                    G_A = (G_O_h if G_O_h is not None
                           else torch.zeros_like(A))
                    G_B = (p_V.grad[j*d_h:(j+1)*d_h, :].float()
                           if p_V.grad is not None
                           else torch.zeros_like(B))
                    sub = self._subspaces[(layer_idx, "ov", h)]
                    D_A, D_B = self._soft_core_update(
                        A, B, G_A, G_B, sub, "OV", layer_idx, h
                    )
                    D_O[h] = D_A                            # [d_model, d_h]
                    D_V_sum[j*d_h:(j+1)*d_h, :].add_(D_B)
                    V_cnt[j] += 1
                else:
                    D_O[h] = (
                        -_safe_msign(G_O_h, self.ns_steps, self.eps)
                        if G_O_h is not None
                        else torch.zeros(p_O.shape[0], d_h, device=p_O.device)
                    )

            # ---- Apply Q and O ----------------------------------------
            for h, dq in D_Q.items():
                p_Q.data[h*d_h:(h+1)*d_h, :].add_(dq.to(p_Q.dtype), alpha=lr)
            for h, do in D_O.items():
                p_O.data[:, h*d_h:(h+1)*d_h].add_(do.to(p_O.dtype), alpha=lr)

            # ---- Apply K ----------------------------------------------
            for j in range(H_kv):
                if use_dual_qk:
                    cnt = K_cnt[j].item()
                    if cnt > 0:
                        p_K.data[j*d_h:(j+1)*d_h, :].add_(
                            (D_K_sum[j*d_h:(j+1)*d_h, :] / cnt).to(p_K.dtype), alpha=lr
                        )
                elif p_K.grad is not None:
                    G_K_j = p_K.grad[j*d_h:(j+1)*d_h, :].float()
                    p_K.data[j*d_h:(j+1)*d_h, :].add_(
                        (-_safe_msign(G_K_j, self.ns_steps, self.eps)).to(p_K.dtype), alpha=lr
                    )

            # ---- Apply V ----------------------------------------------
            for j in range(H_kv):
                if use_dual_ov:
                    cnt = V_cnt[j].item()
                    if cnt > 0:
                        p_V.data[j*d_h:(j+1)*d_h, :].add_(
                            (D_V_sum[j*d_h:(j+1)*d_h, :] / cnt).to(p_V.dtype), alpha=lr
                        )
                elif p_V.grad is not None:
                    G_V_j = p_V.grad[j*d_h:(j+1)*d_h, :].float()
                    p_V.data[j*d_h:(j+1)*d_h, :].add_(
                        (-_safe_msign(G_V_j, self.ns_steps, self.eps)).to(p_V.dtype), alpha=lr
                    )

        self._last_msign_stats = get_msign_stats()

        # Accumulate cumulative fallback counters
        self._step_count += 1
        self._cum_heads += len(self._last_sylvester_used)
        for used, failed in zip(self._last_sylvester_used, self._last_sylvester_failed):
            if used and not failed:
                self._cum_soft += 1
            elif used and failed:
                self._cum_hard += 1

        return loss

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_constraint_metrics(self) -> dict:
        """Per-step + cumulative metrics, suitable for wandb logging.

        Fallback terminology
        --------------------
        soft fallback  Sylvester correction was applied and reduced the
                       constraint residual below hard_fallback_tol.
        hard fallback  Sylvester was applied but could not satisfy the
                       tolerance (or produced non-finite output), so both
                       Delta_A and Delta_B were zeroed for that head-pair.
        """
        metrics: dict = {}

        # ---- Core drift & H_norm -----------------------------------------
        if self._last_core_drifts:
            drifts = self._last_core_drifts
            metrics["soft_comp_muon/core_drift_mean"] = sum(drifts) / len(drifts)
            metrics["soft_comp_muon/core_drift_max"]  = max(drifts)
        if self._last_H_norms:
            hnorms = self._last_H_norms
            metrics["soft_comp_muon/H_norm_mean"] = sum(hnorms) / len(hnorms)
            metrics["soft_comp_muon/H_norm_max"]  = max(hnorms)
        if self._last_dual_iters:
            its = self._last_dual_iters
            metrics["soft_comp_muon/dual_iters_mean"] = sum(its) / len(its)

        # ---- msign health ------------------------------------------------
        s = self._last_msign_stats
        if s.get("total", 0) > 0:
            metrics["soft_comp_muon/msign_total"]      = s["total"]
            metrics["soft_comp_muon/msign_nan_input"]  = s["nan_input"]
            metrics["soft_comp_muon/msign_ns_diverge"] = s["ns_diverge"]
            metrics["soft_comp_muon/msign_nan_output"] = s["nan_output"]

        # ---- Per-step fallback counts & rates ----------------------------
        n_heads = len(self._last_sylvester_used)
        if n_heads > 0:
            used   = self._last_sylvester_used
            failed = self._last_sylvester_failed
            rb     = self._last_resid_before_fallback
            ra     = self._last_resid_after_fallback

            n_soft = sum(u and not f for u, f in zip(used, failed))
            n_hard = sum(u and f     for u, f in zip(used, failed))

            metrics["soft_comp_muon/step_heads"]           = n_heads
            metrics["soft_comp_muon/step_soft_fallbacks"]  = n_soft
            metrics["soft_comp_muon/step_hard_fallbacks"]  = n_hard
            metrics["soft_comp_muon/step_soft_rate"]       = n_soft / n_heads
            metrics["soft_comp_muon/step_hard_rate"]       = n_hard / n_heads

            # Residual before/after (only where Sylvester fired)
            rb_triggered = [b for b, u in zip(rb, used) if u]
            ra_triggered = [a for a, u in zip(ra, used) if u and math.isfinite(a)]
            if rb_triggered:
                metrics["soft_comp_muon/step_resid_before_mean"] = (
                    sum(rb_triggered) / len(rb_triggered)
                )
            if ra_triggered:
                metrics["soft_comp_muon/step_resid_after_mean"] = (
                    sum(ra_triggered) / len(ra_triggered)
                )

        # ---- Cumulative fallback counters --------------------------------
        if self._cum_heads > 0:
            metrics["soft_comp_muon/cum_steps"]      = self._step_count
            metrics["soft_comp_muon/cum_heads"]      = self._cum_heads
            metrics["soft_comp_muon/cum_soft"]       = self._cum_soft
            metrics["soft_comp_muon/cum_hard"]       = self._cum_hard
            metrics["soft_comp_muon/cum_soft_rate"]  = self._cum_soft / self._cum_heads
            metrics["soft_comp_muon/cum_hard_rate"]  = self._cum_hard / self._cum_heads

        return metrics

    def get_fallback_summary(self) -> str:
        """One-line human-readable summary of fallback statistics.

        Suitable for printing to console during training:
            [SoftCompMuon] step=120  heads/step=28  soft=0.8%  hard=0.0%
                           cum: soft=11/1680 (0.7%)  hard=0/1680 (0.0%)
        """
        n = len(self._last_sylvester_used)
        if n == 0:
            return f"[SoftCompMuon] step={self._step_count}  no constrained heads this step"

        used   = self._last_sylvester_used
        failed = self._last_sylvester_failed
        n_soft = sum(u and not f for u, f in zip(used, failed))
        n_hard = sum(u and f     for u, f in zip(used, failed))

        cum_total = self._cum_heads or 1
        return (
            f"[SoftCompMuon] step={self._step_count}  heads/step={n}"
            f"  soft={n_soft}/{n} ({100*n_soft/n:.1f}%)"
            f"  hard={n_hard}/{n} ({100*n_hard/n:.1f}%)"
            f"  |  cum: soft={self._cum_soft}/{self._cum_heads}"
            f" ({100*self._cum_soft/cum_total:.1f}%)"
            f"  hard={self._cum_hard}/{self._cum_heads}"
            f" ({100*self._cum_hard/cum_total:.1f}%)"
        )

    def reset_fallback_stats(self) -> None:
        """Reset cumulative fallback counters (call between continual learning tasks)."""
        self._step_count = 0
        self._cum_heads  = 0
        self._cum_soft   = 0
        self._cum_hard   = 0
