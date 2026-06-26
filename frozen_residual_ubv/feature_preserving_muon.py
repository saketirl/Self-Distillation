"""
Feature-Preserving Muon Optimizer.

For a weight matrix W ∈ R^{m×n}, compute the top-r right singular vectors
V_r ∈ R^{n×r} that capture a given fraction of W's spectral energy, then
constrain every gradient update to be orthogonal to V_r via dual ascent:

    min_D  <G, D>   s.t.  ‖D‖_spectral ≤ 1,  D V_r = 0

Lagrangian:  L(D, Λ) = <G, D> + <Λ, D V_r>
Effective gradient:  B = G + Λ V_r^T
Primal minimizer:    D = -msign(B)
Dual ascent:         Λ ← Λ + ρ (D V_r)

Warm start: set Λ₀ = -(G V_r) so B₀ = G(I - V_r V_r^T).
With n_dual_iter=0 this reduces to plain projection.

The rank r is chosen per-parameter at init via an energy threshold:
    r = min { j : Σ_{i≤j} σ_i² / Σ_i σ_i² ≥ energy_threshold }

V_r is computed once from the *initial* W, QR-reorthogonalized for
numerical precision, and held fixed for the entire run.

Double projection (ur_alpha > 0):
    Saves both U_r (left singular vectors) and V_r (right singular vectors)
    and applies a soft double projection:

        B ← (I - ur_alpha · U_r U_r^T)(I - V_r V_r^T) G

    When ur_alpha=1 this exactly preserves the T00 singular core U_r Σ V_r^T
    across all subsequent tasks, because every update ΔW satisfies
    U_r^T ΔW = 0 and ΔW V_r = 0, so:
        U_r^T (W_0 + ΔW₁ + … + ΔW_t) V_r = U_r^T W_0 V_r = Σ_r

References
----------
- Bernstein & Newhouse, "Old Optimizer, New Norm," 2024
- Buchanan, "Manifold MUON," https://sdbuchanan.com/blog/manifold-muon/
"""

from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .utils import msign, reset_msign_stats, get_msign_stats
from .projected_gradient_optimizer import energy_truncated_svd


class FeaturePreservingMuon(Optimizer):
    """
    Gradient optimizer that preserves the dominant right singular feature
    directions of each weight matrix throughout training.

    For every 2-D (or higher, flattened) parameter W ∈ R^{m×n}:

      1. At init, compute V_r ∈ R^{n×r} whose columns are the right singular
         vectors capturing `energy_threshold` fraction of W's spectral energy.
         Store V_r in state; never update it.

      2. At each step, find D = -msign(G + Λ V_r^T) satisfying D V_r = 0
         via dual ascent with warm start Λ₀ = -(G V_r).

      3. Update: W ← W + lr * D  (D is already the descent direction).

    Parameters with ndim < 2 (biases, LayerNorm scales, embedding vectors)
    receive a plain SGD fallback update instead.

    Args:
        params:            Iterable of parameters or param groups.
        lr:                Learning rate (η).
        energy_threshold:  Fraction of spectral energy to protect (0.0–1.0).
        momentum:          Momentum coefficient β ∈ [0, 1).  0 disables.
        nesterov:          Use Nesterov momentum (requires momentum > 0).
        weight_decay:      L2 penalty added to gradient before momentum.
        ns_steps:          Newton-Schulz iterations inside msign.
        n_dual_iter:       Dual ascent iterations (0 = warm-start projection only).
        dual_tol:          Stop dual ascent early when ‖D V_r‖_F < dual_tol.
        dual_step_size:    Step size ρ for dual variable updates.
        eps:               Numerical floor passed to msign.
        debug:             Print constraint residuals at every step.
        fallback_lr:       Learning rate for 1-D params (defaults to lr).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        energy_threshold: float = 0.8,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        ns_steps: int = 8,
        inner_ns_steps: int = 3,
        n_dual_iter: int = 0,
        dual_tol: float = 1e-4,
        dual_step_size: float = 0.01,
        vr_update_freq: int = 0,
        residual_fallback_tol: float = 0.1,
        vr_bypass_alpha: float = 0.0,
        ur_alpha: float = 0.0,
        fixed_subspaces: Optional[dict] = None,
        eps: float = 1e-7,
        debug: bool = False,
        fallback_lr: Optional[float] = None,
        scale_to_proj_norm: bool = False,
    ):
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires momentum > 0")
        if not 0.0 < energy_threshold <= 1.0:
            raise ValueError(f"energy_threshold must be in (0, 1], got {energy_threshold}")
        if not 0.0 <= vr_bypass_alpha <= 1.0:
            raise ValueError(f"vr_bypass_alpha must be in [0, 1], got {vr_bypass_alpha}")
        if not 0.0 <= ur_alpha <= 1.0:
            raise ValueError(f"ur_alpha must be in [0, 1], got {ur_alpha}")
        # fixed_subspaces: {param_id -> (U_r, V_r)} computed from T00 weights.
        # When provided, _init_state uses these instead of re-computing from
        # current weights, so the protected subspace never rotates across tasks.
        self._fixed_subspaces = fixed_subspaces or {}

        defaults = dict(
            lr=lr,
            energy_threshold=energy_threshold,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            inner_ns_steps=inner_ns_steps,
            n_dual_iter=n_dual_iter,
            dual_tol=dual_tol,
            dual_step_size=dual_step_size,
            vr_update_freq=vr_update_freq,
            residual_fallback_tol=residual_fallback_tol,
            vr_bypass_alpha=vr_bypass_alpha,
            ur_alpha=ur_alpha,
            eps=eps,
            debug=debug,
            fallback_lr=fallback_lr if fallback_lr is not None else lr,
            scale_to_proj_norm=scale_to_proj_norm,
        )
        super().__init__(params, defaults)

    # ------------------------------------------------------------------
    # State initialisation / V_r refresh
    # ------------------------------------------------------------------

    @staticmethod
    def compute_fixed_subspaces(params, energy_threshold: float) -> dict:
        """
        Compute {param_id -> (U_r, V_r)} from the current weights of `params`.
        Pass the result as `fixed_subspaces=` to freeze the protected subspace
        across all subsequent tasks (prevents the V_r rotation problem).

        Typical usage:
            # After training T00 with Adam:
            fixed = FeaturePreservingMuon.compute_fixed_subspaces(
                model.parameters(), energy_threshold=0.4)
            # Then for every subsequent task:
            fpm = FeaturePreservingMuon(..., fixed_subspaces=fixed)
        """
        fixed = {}
        for p in params:
            if p.ndim < 2:
                continue
            W = p.data.reshape(p.shape[0], -1).float()
            if min(W.shape) < 2:
                continue
            U_r, _, V_r, _, _ = energy_truncated_svd(W, energy_threshold)
            V_r, _ = torch.linalg.qr(V_r)
            U_r, _ = torch.linalg.qr(U_r)
            fixed[id(p)] = (U_r.to(p.device), V_r.to(p.device))
        return fixed

    def _compute_uvr(self, p: Tensor, energy_threshold: float) -> tuple[Tensor, Tensor, int, float]:
        """Compute QR-orthogonalized U_r and V_r from current weights of p."""
        W_mat = p.data.reshape(p.shape[0], -1).float()
        U_r, _, V_r, r, captured = energy_truncated_svd(W_mat, energy_threshold)
        V_r, _ = torch.linalg.qr(V_r)
        U_r, _ = torch.linalg.qr(U_r)
        dev = p.device
        return U_r.to(dev), V_r.to(dev), r, captured

    def _init_state(self, p: Tensor, state: dict, energy_threshold: float) -> None:
        if p.ndim < 2:
            state["skip"] = True
            return

        state["orig_shape"] = p.shape
        state["step"] = 0

        W_mat = p.data.reshape(p.shape[0], -1).float()  # [m, n]
        m, n = W_mat.shape

        if min(m, n) < 2:
            state["skip"] = True
            return

        state["skip"] = False

        if id(p) in self._fixed_subspaces:
            U_r, V_r = self._fixed_subspaces[id(p)]
            r = V_r.shape[1]
            captured = float("nan")  # energy not recomputed for fixed subspaces
        else:
            with torch.no_grad():
                U_r, V_r, r, captured = self._compute_uvr(p, energy_threshold)

        state["U_r"] = U_r          # [m, r]
        state["V_r"] = V_r          # [n, r]
        state["r"] = r
        state["captured_energy"] = captured

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        reset_msign_stats()

        for group in self.param_groups:
            lr             = group["lr"]
            energy_threshold = group["energy_threshold"]
            momentum       = group["momentum"]
            nesterov       = group["nesterov"]
            weight_decay   = group["weight_decay"]
            ns_steps       = group["ns_steps"]
            n_dual_iter           = group["n_dual_iter"]
            dual_tol              = group["dual_tol"]
            dual_step_size        = group["dual_step_size"]
            residual_fallback_tol = group["residual_fallback_tol"]
            eps                   = group["eps"]
            debug          = group["debug"]
            fallback_lr    = group["fallback_lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, state, energy_threshold)

                g = p.grad

                # ---- 1-D fallback: bias, LayerNorm, etc. --------------------
                if state["skip"]:
                    if weight_decay != 0.0:
                        g = g.add(p.data, alpha=weight_decay)
                    p.data.add_(g, alpha=-fallback_lr)
                    continue

                state["step"] += 1
                orig_shape = state["orig_shape"]

                # Refresh U_r/V_r from current weights every vr_update_freq steps.
                # Skipped when fixed_subspaces are in use (subspace is frozen to T00).
                vr_update_freq = group["vr_update_freq"]
                if vr_update_freq > 0 and state["step"] % vr_update_freq == 0 \
                        and id(p) not in self._fixed_subspaces:
                    with torch.no_grad():
                        U_r, V_r, r, captured = self._compute_uvr(p, energy_threshold)
                    state["U_r"] = U_r
                    state["V_r"] = V_r
                    state["r"] = r
                    state["captured_energy"] = captured

                U_r = state["U_r"]   # [m, r], float32
                V_r = state["V_r"]   # [n, r], float32
                vr_bypass_alpha = group["vr_bypass_alpha"]
                ur_alpha        = group["ur_alpha"]

                m = orig_shape[0]
                G_mat = g.reshape(m, -1).float()       # [m, n]
                W_mat = p.data.reshape(m, -1).float()  # [m, n]

                # ---- Weight decay -------------------------------------------
                if weight_decay != 0.0:
                    G_mat = G_mat.add(W_mat, alpha=weight_decay)

                # ---- Momentum -----------------------------------------------
                if momentum != 0.0:
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = G_mat.clone()
                        state["momentum_buffer"] = buf
                    else:
                        buf.mul_(momentum).add_(G_mat)
                    G_mat = G_mat.add(buf, alpha=momentum) if nesterov else buf

                # ---- Joint dual ascent for U_r and V_r constraints ----------
                #
                # min_D <G, D>  s.t.  ‖D‖_spectral ≤ 1,
                #                     D V_r = 0   (right: protect input directions)
                #                     U_r^T D = 0 (left:  protect output directions, ur_alpha>0)
                #
                # Lagrangian:
                #   L(D, Λ_R, Λ_L) = <G + Λ_R V_r^T + U_r Λ_L, D>
                # so  B = G + Λ_R V_r^T + U_r Λ_L
                #     D = -msign(B)
                # Dual updates:
                #   Λ_R ← Λ_R + ρ (D V_r)      [right violation, [m,r]]
                #   Λ_L ← Λ_L + ρ (U_r^T D)    [left violation,  [r,n]]
                #
                # Warm start (gives B₀ = (I-U_rU_r^T) G (I-V_rV_r^T)):
                #   Λ_R₀ = -(G V_r)
                #   Λ_L₀ = -(U_r^T G(I - V_r V_r^T))   [only when ur_alpha>0]
                #
                # vr_bypass_alpha: fraction of G's V_r component allowed through.

                G_vr = G_mat @ V_r  # [m, r]

                grad_norm_raw   = torch.linalg.norm(G_mat).item()
                grad_in_vr      = torch.linalg.norm(G_vr).item()
                grad_frac_in_vr = grad_in_vr / (grad_norm_raw + 1e-12)

                # Warm-start dual variables
                Lambda_R = -(1.0 - vr_bypass_alpha) * G_vr   # [m, r]
                if ur_alpha > 0.0:
                    G_perp_vr = G_mat - G_vr @ V_r.T          # G(I - V_r V_r^T)
                    Lambda_L  = -ur_alpha * (U_r.T @ G_perp_vr)  # [r, n]
                else:
                    Lambda_L = None

                def _make_B():
                    B = G_mat + Lambda_R @ V_r.T
                    if Lambda_L is not None:
                        B = B + U_r @ Lambda_L
                    return B

                if vr_bypass_alpha > 0.0:
                    # Bypass mode: single msign on warm-started B (no iteration).
                    B = _make_B()
                    D = -msign(B, ns_steps=ns_steps, eps=eps)
                    residual_right = torch.linalg.norm(D @ V_r).item()
                    used_fallback  = False
                else:
                    D = None
                    for _ in range(n_dual_iter):
                        B = _make_B()
                        D = -msign(B, ns_steps=ns_steps, eps=eps)

                        H_R = D @ V_r        # [m, r]
                        res_R = torch.linalg.norm(H_R).item()
                        if Lambda_L is not None:
                            H_L   = U_r.T @ D   # [r, n]
                            res_L = torch.linalg.norm(H_L).item()
                        else:
                            res_L = 0.0

                        if res_R < dual_tol and res_L < dual_tol:
                            break

                        Lambda_R = Lambda_R + dual_step_size * H_R
                        if Lambda_L is not None:
                            Lambda_L = Lambda_L + dual_step_size * H_L

                    if D is None:
                        B = _make_B()
                        D = -msign(B, ns_steps=ns_steps, eps=eps)

                    # ---- Exact-projection fallback --------------------------
                    # Fall back to analytic double projection if residuals are
                    # too large (dual ascent diverged or subspaces just refreshed).
                    residual_right = torch.linalg.norm(D @ V_r).item()
                    used_fallback  = False
                    if residual_right > residual_fallback_tol:
                        B = G_mat - G_vr @ V_r.T
                        if ur_alpha > 0.0:
                            B = B - ur_alpha * (U_r @ (U_r.T @ B))
                        D = -msign(B, ns_steps=ns_steps, eps=eps)
                        residual_right = torch.linalg.norm(D @ V_r).item()
                        used_fallback  = True

                # ---- Metrics ------------------------------------------------
                residual_left  = torch.linalg.norm(U_r.T @ D).item() if (ur_alpha > 0.0 and debug) else 0.0
                grad_norm_effective = torch.linalg.norm(B).item()
                grad_frac_surviving = grad_norm_effective / (grad_norm_raw + 1e-12)

                # ---- Gradient clipping: scale D by fraction surviving projection --
                # Without this, msign always produces unit-spectral-norm updates
                # regardless of how much gradient survived D@V_r=0 projection.
                # When ‖G_perp‖ ≪ ‖G‖ (task gradient mostly blocked), the residual
                # noise is amplified to full magnitude → adversarial update →
                # model drifts away from teacher → grad_norm grows → overflow.
                if group["scale_to_proj_norm"]:
                    D = D * grad_frac_surviving

                state["residual_right"]      = residual_right
                state["residual_left"]       = residual_left
                state["grad_norm_raw"]       = grad_norm_raw
                state["grad_norm_projected"] = grad_norm_effective
                state["grad_frac_surviving"] = grad_frac_surviving
                state["grad_frac_in_vr"]     = grad_frac_in_vr
                state["used_fallback"]       = used_fallback

                if debug:
                    left_str = f"  ‖U_r^T D‖={residual_left:.3e}" if ur_alpha > 0.0 else ""
                    print(
                        f"[FeaturePreservingMuon] step={state['step']}  "
                        f"‖D V_r‖={residual_right:.3e}{left_str}  "
                        f"‖G‖={grad_norm_raw:.3e}  "
                        f"‖B‖={grad_norm_effective:.3e}  "
                        f"frac={grad_frac_surviving:.3f}  "
                        f"frac_in_Vr={grad_frac_in_vr:.3f}  "
                        f"r={state['r']}  "
                        f"energy={state['captured_energy']:.3f}  "
                        f"fallback={used_fallback}  "
                        f"bypass={vr_bypass_alpha:.2f}  ur_alpha={ur_alpha:.2f}  "
                        f"shape={tuple(orig_shape)}"
                    )

                # ---- Parameter update ---------------------------------------
                update = D.to(dtype=p.dtype).reshape(orig_shape)
                p.data.add_(update, alpha=lr)

        self._last_msign_stats = get_msign_stats()
        return loss

    def get_constraint_metrics(self) -> dict:
        """
        Return summary constraint residuals for wandb logging.

        Keys:
            fp_muon/residual_right_mean  — mean ‖D V_r‖_F
            fp_muon/grad_norm_raw_mean   — mean ‖G‖_F (before projection)
            fp_muon/grad_norm_proj_mean  — mean ‖B‖_F (effective gradient)
            fp_muon/grad_frac_mean       — mean ‖B‖/‖G‖ (fraction surviving)
        """
        res_right, res_left, raw_norms, proj_norms, fracs, fracs_in_vr, fallbacks = [], [], [], [], [], [], []
        for group in self.param_groups:
            for p in group["params"]:
                s = self.state.get(p, {})
                if s.get("skip", True):
                    continue
                if "residual_right"      in s: res_right.append(s["residual_right"])
                if "residual_left"       in s and s["residual_left"] > 0.0:
                    res_left.append(s["residual_left"])
                if "grad_norm_raw"       in s: raw_norms.append(s["grad_norm_raw"])
                if "grad_norm_projected" in s: proj_norms.append(s["grad_norm_projected"])
                if "grad_frac_surviving" in s: fracs.append(s["grad_frac_surviving"])
                if "grad_frac_in_vr"     in s: fracs_in_vr.append(s["grad_frac_in_vr"])
                if "used_fallback"       in s: fallbacks.append(float(s["used_fallback"]))

        if not res_right:
            return {}

        def _agg(vals, name):
            return {
                f"fp_muon/{name}_mean": sum(vals) / len(vals),
                f"fp_muon/{name}_max":  max(vals),
                f"fp_muon/{name}_min":  min(vals),
            }

        metrics = {}
        metrics.update(_agg(res_right,  "residual_right"))
        if res_left:     metrics.update(_agg(res_left,     "residual_left"))
        if raw_norms:    metrics.update(_agg(raw_norms,    "grad_norm_raw"))
        if proj_norms:   metrics.update(_agg(proj_norms,   "grad_norm_proj"))
        if fracs:        metrics.update(_agg(fracs,        "grad_frac"))
        if fracs_in_vr:  metrics.update(_agg(fracs_in_vr,  "grad_frac_in_vr"))
        if fallbacks:
            metrics["fp_muon/fallback_rate"] = sum(fallbacks) / len(fallbacks)
        s = getattr(self, "_last_msign_stats", {})
        if s.get("total", 0) > 0:
            metrics["fp_muon/msign_total"]      = s["total"]
            metrics["fp_muon/msign_nan_input"]  = s["nan_input"]
            metrics["fp_muon/msign_ns_diverge"] = s["ns_diverge"]
            metrics["fp_muon/msign_nan_output"] = s["nan_output"]
        return metrics
