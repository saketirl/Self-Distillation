"""CoupledKeyHalfSplitMuon — half-split QK/OV with a Gram-coupled key leg.

Motivation (gated-DeltaNet / recurrent-state layers). In softmax attention the
key matrix is pure gauge beyond the composition M_qk = W_Q W_K^T: any invertible
R applied as (W_Q R^{-T}, W_K R) leaves every score unchanged. In a gated delta
net the key ALSO enters the state transition quadratically, (I - beta k k^T),
and unrolling the recurrence produces key-key inner products x^T (W_K W_K^T) x'.
The gauge group shrinks to O(d_k) and the key self-Gram

    M_kk = W_K W_K^T          (the ADDRESSING geometry of the associative state)

becomes functional: rotating it re-indexes every stored association even when
M_qk is perfectly preserved. So W_K answers to TWO atoms at once and its update
must satisfy both:

    min <G_K, dK>
    s.t. ||W_Q dK^T||_op <= eps/2                          (M_qk ball, k-leg)
         U_r^T (dK W_K^T + W_K dK^T) U_r = -gamma * E_r    (M_kk subspace
                                                            tangency+contraction)

with U_r the top-r eigenvectors of the TASK-START anchor M_kk* and
E_r = U_r^T (M_kk - M_kk*) U_r the accumulated leak in the protected subspace.

Solution (whitened reduction). With C_Q = (W_Q^T W_Q + lam I)^{1/2} the ball is
||dK C_Q||_op <= eps/2 exactly, and dualizing the equality with a small
symmetric multiplier Lam_r in Sym(r) gives the closed-form inner minimizer

    dK*(Lam_r) = -(eps/2) msign( (G_hat + 2 U_r Lam_r (U_r^T W_K)^T) C_Q^{-1} ) C_Q^{-1}

— the half-split's partner-whitened k-leg with GramFlow's multiplier INSIDE the
msign argument (so it is spectrally renormalized: the structural safety property
SoftCompPreservingMuon lacked). Lam_r is found by K warm-started projected
dual-ascent iterations (GramFlow discipline: normalized gradient, normalized
residuals, K=2 enforces softly across steps rather than exactly per step).

Limits:  r=0 (or gamma=0 & Lam=0)  ->  exact CompositionalHalfSplitMuon k-leg.
         C_Q = I, r = d            ->  GramFlow's msign(G + 2 W Lam) form.

Compute: everything stays thin — the multiplier term is (d,r)(r,r)(r,d_k), the
residual is (r,r); no d_model x d_model object is ever formed (U_r comes from a
d_k x d_k eigh of the SMALL Gram, once per task at construction). Per step the
k-leg costs K batched thin msigns instead of 1: ~+50% on the QK arithmetic.

Integration: replace CompositionalHalfSplitMuon with this class in
create_halfsplit_optimizer; OV and the q-leg are inherited unchanged.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .compositional_halfsplit_muon import (
    CompositionalHalfSplitMuon,
    msign_batched,
)


class CoupledKeyHalfSplitMuon(CompositionalHalfSplitMuon):
    """Half-split on QK/OV where the key leg additionally preserves the top-r
    subspace of the key self-Gram M_kk = W_K W_K^T (task-start anchor).

    New args (everything else mirrors CompositionalHalfSplitMuon):
        rank:        r, protected eigen-directions of M_kk per KV head.
                     0 disables the coupling (recovers the plain half-split).
        gamma:       contraction rate pulling U_r^T M_kk U_r back to its anchor.
        dual_steps:  K, warm-started dual-ascent iterations per step (default 2).
        dual_lr:     ascent step on the normalized residual (default 1.0).
        clamp_c:     feasibility clamp — the contraction demand is capped at
                     clamp_c of the residual scale so the equality can never
                     exceed what the eps/2 ball can deliver (default 0.5).
    """

    def __init__(self, named_params: List[Tuple[str, Tensor]], lr: float = 1e-3,
                 eps: float = 1.0, head_dim: int = 32, num_kv_heads: int = 4,
                 damping: float = 1e-6, ns_steps: int = 8,
                 whitening: str = "ns", track_every: int = 0,
                 attn_name_patterns: Optional[List[str]] = None,
                 rank: int = 8, gamma: float = 1e-3,
                 dual_steps: int = 2, dual_lr: float = 0.25,
                 clamp_c: float = 0.5):
        # Parent handles OV (and param parsing / apply / metrics); its QK path
        # is disabled — the coupled k-leg below replaces it together with the
        # (unchanged) q-leg.
        super().__init__(named_params, lr=lr, eps=eps, head_dim=head_dim,
                         num_kv_heads=num_kv_heads, damping=damping,
                         preserve_qk=False, preserve_ov=True, ns_steps=ns_steps,
                         whitening=whitening, track_every=track_every,
                         attn_name_patterns=attn_name_patterns)
        if rank < 0 or rank > head_dim:
            raise ValueError(f"rank must be in [0, head_dim={head_dim}], got {rank}")
        if dual_steps < 0:
            raise ValueError(f"dual_steps must be >= 0, got {dual_steps}")
        self.rank, self.gamma = rank, gamma
        self.dual_steps, self.dual_lr, self.clamp_c = dual_steps, dual_lr, clamp_c
        # per-layer anchors: {layer: {"Ur": (H_kv,d,r), "S2": (H_kv,r,r),
        #                             "Lam": (H_kv,r,r)}}
        self._coupled: Dict[str, Dict[str, Tensor]] = {}
        if rank > 0:
            self.snap_anchor()

    # ------------------------------------------------------------------ anchor
    @torch.no_grad()
    def snap_anchor(self) -> None:
        """Snap U_r and the anchored core from the CURRENT weights (task start).

        Small-Gram route — never forms d x d: eigh of W_K^T W_K (d_k x d_k,
        batched over KV heads), lift U_r = W_K V_r Sigma_r^{-1}. At the snap
        U_r^T M_kk U_r = diag(top-r eigenvalues) exactly.
        """
        d_h, H_kv, r = self.head_dim, self.num_kv_heads, self.rank
        self._coupled.clear()
        for lid, lp in self._layers.items():
            if "k" not in lp:
                continue
            p_K = lp["k"]
            WK = p_K.data.view(H_kv, d_h, -1).mT.float()          # (H_kv, d, d_h)
            G = WK.mT @ WK                                        # (H_kv, d_h, d_h)
            G = 0.5 * (G + G.mT)
            ev, V = torch.linalg.eigh(G)                          # ascending
            ev_r = ev[..., -r:].clamp_min(1e-12)                  # top-r
            V_r = V[..., -r:]                                     # (H_kv, d_h, r)
            Ur = (WK @ V_r) * ev_r.rsqrt().unsqueeze(-2)          # (H_kv, d, r)
            self._coupled[lid] = {
                "Ur": Ur,
                "S2": torch.diag_embed(ev_r),                     # anchored core
                "Lam": torch.zeros_like(torch.diag_embed(ev_r)),  # warm duals
            }

    # -------------------------------------------------------------------- step
    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)      # OV legs (parent QK disabled)
        lr = self.param_groups[0]["lr"]
        if lr == 0:
            return loss
        d_h, H_kv, half = self.head_dim, self.num_kv_heads, 0.5 * self.eps
        r = self.rank

        for lid, lp in self._layers.items():
            if not {"q", "k"} <= lp.keys() or lp["q"].grad is None:
                continue
            p_Q, p_K = lp["q"], lp["k"]
            H_q = p_Q.shape[0] // d_h
            grp = max(H_q // H_kv, 1)
            d = p_Q.shape[1]

            WQ = p_Q.data.view(H_q, d_h, d).mT.float()            # (H_q, d, d_h)
            GQ = p_Q.grad.view(H_q, d_h, d).mT.float()
            WK = p_K.data.view(H_kv, d_h, d).mT.float()           # (H_kv, d, d_h)
            GK = p_K.grad.view(H_kv, d_h, d).mT.float()

            # ---- q-leg: identical to the parent (whiten by the KEY Gram) ----
            CKi = self._isq(WK.mT @ WK).repeat_interleave(grp, 0)
            dQ = -half * (msign_batched(GQ @ CKi, self.ns_steps) @ CKi)

            # ---- coupled k-leg ----------------------------------------------
            # GQA NOTE (deliberate fix over the parent): averaging per-Q-head-
            # whitened K updates does NOT bound the applied leg — each term is
            # bounded w.r.t. its OWN Q head only (measured 4.4x violation on
            # random weights). Whitening by the GROUP-SUMMED Q Gram gives
            # W_Qh^T W_Qh <= sum_grp W_Q^T W_Q, hence ||W_Qh dK^T||_op <= eps/2
            # for EVERY head in the group, with ONE solve per KV head. For
            # grp == 1 (MHA) this is exactly the parent's k-leg.
            gramQ = (WQ.mT @ WQ).view(H_kv, grp, d_h, d_h).sum(1)  # (H_kv,d_h,d_h)
            CQi = self._isq(gramQ)                                 # (H_kv,d_h,d_h)
            # scale-invariant gradient (msign ignores scale; the Lam term must
            # live on a fixed scale for warm starts to be meaningful)
            gnorm = GK.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
            Ghat = GK / gnorm                                      # (H_kv, d, d_h)

            cd = self._coupled.get(lid) if r > 0 else None
            if cd is None:
                dK = -half * (msign_batched(Ghat @ CQi, self.ns_steps) @ CQi)
            else:
                Ur, S2, Lam = cd["Ur"], cd["S2"], cd["Lam"]
                UW = Ur.mT @ WK                                    # (H_kv, r, d_h)
                core = UW @ UW.mT                                  # (H_kv, r, r)
                E = core - S2                                      # leak
                # residual scale: the WHITENED update has ||dK|| ~ half*||CQi||,
                # so sym(U^T dK W^T U) ~ half*||CQi||*||U^T W_K|| — normalizing
                # by ||W_K|| alone under-scales by ||CQi|| and the ascent
                # overshoots/oscillates (measured: leak INCREASED).
                s = (half * CQi.norm(dim=(-2, -1)) * UW.norm(dim=(-2, -1)))
                s = s.clamp_min(1e-12)[:, None, None]
                # feasibility clamp: contraction demand <= clamp_c of scale
                e_norm = E.norm(dim=(-2, -1), keepdim=True)
                g_eff = self.gamma * torch.ones_like(e_norm)
                cap = self.clamp_c * s / e_norm.clamp_min(1e-12)
                g_eff = torch.minimum(g_eff, cap)

                # msign is nonsmooth in Lam (dual chattering, see the Buchanan
                # blog) — last-iterate ascent can oscillate, so apply the
                # BEST-scoring iterate's update, GramFlow-style.
                dK, P, best = None, None, None
                for _ in range(max(1, self.dual_steps)):
                    term = 2.0 * (Ur @ (Lam @ UW))                 # (H_kv, d, d_h)
                    Geff = Ghat + term
                    dK_i = -half * (msign_batched(Geff @ CQi, self.ns_steps) @ CQi)
                    P_i = (Ur.mT @ dK_i) @ UW.mT                   # (H_kv, r, r)
                    resid = (P_i + P_i.mT) + g_eff * E
                    score = float((resid.norm(dim=(-2, -1), keepdim=True) / s).mean())
                    if best is None or score < best:
                        best, dK, P = score, dK_i, P_i
                    if self.dual_steps == 0:
                        break                                      # frozen duals
                    Lam = Lam + self.dual_lr * (resid / s)
                    Lam = 0.5 * (Lam + Lam.mT)
                cd["Lam"] = torch.nan_to_num(Lam, nan=0.0, posinf=0.0, neginf=0.0)

                if self.track_every and self._nstep % self.track_every == 0:
                    leak = (P + P.mT).norm(dim=(-2, -1)) / s.squeeze(-1).squeeze(-1)
                    self._viol["coupled/k_leak_ratio_mean"] = float(leak.mean())
                    self._viol["coupled/k_leak_ratio_max"] = float(leak.max())
                    self._viol["coupled/E_fro_mean"] = float(E.norm(dim=(-2, -1)).mean())

            if self.track_every and self._nstep % self.track_every == 0:
                WK_rep = WK.repeat_interleave(grp, 0)
                dK_rep_m = dK.repeat_interleave(grp, 0)
                self._measure("qk", dQ, WK_rep, WQ, dK_rep_m, half)

            if torch.isfinite(dQ).all() and torch.isfinite(dK).all():
                self._apply(p_Q, dQ.mT.reshape(H_q * d_h, d), lr)
                self._apply(p_K, dK.mT.reshape(H_kv * d_h, d), lr)
            else:
                self.n_bad += 1
        # NOTE: parent step() already advanced self._nstep — do not increment.
        return loss
