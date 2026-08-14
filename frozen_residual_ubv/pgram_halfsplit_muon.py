"""PGramHalfSplitMuon — half-split with composed-Gram preservation (P-Gram).

Implements docs/halfsplit_constraint.md §7 on BOTH attention circuits. Per head
the atoms are M = W_Q W_K^T (QK) and N = W_O W_V (OV); in addition to the
half-split trust region on ||dM||_op, the update preserves the top-r right
singular subspace of the COMPOSED operator, i.e. the restriction of its Gram:

    V_r^T (M^T M) V_r  held at its task-start value (soft, contraction gamma)
    via   V_r^T sym(M^T dM) V_r = -gamma E_r .

Key identities (all thin — no d x d object is ever formed):
    M^T M = W_K C_Q^2 W_K^T,  C_Q^2 = W_Q^T W_Q      (query-metric key Gram)
    N^T N = W_V^T C_O^2 W_V,  C_O^2 = W_O^T W_O
Anchors V_r are the top-r eigvecs, computed from the small Gram of the thin
factor B = W_K chol(C_Q^2) (resp. B = W_V^T chol(C_O^2)) and lifted.

The tangency is GAUGE-INVARIANT (M^T M is unchanged by (W_Q,W_K) ->
(W_Q R^{-T}, W_K R)), which corrects CoupledKeyHalfSplitMuon's raw-M_kk anchor
(that class is the C_Q = I approximation of this one).

Dual game: one shared Lambda_r in Sym(r) per head per circuit tilts BOTH legs,

    G~_Q = G_Q + 2 W_Q A^T Lam A          A = V_r^T W_K   (r, d_h)
    G~_K = G_K + 2 V_r Lam A C_Q^2
    G~_O = G_O + 2 W_O A_ov Lam A_ov^T    A_ov = W_V V_r  (d_h, r)
    G~_V = G_V + 2 C_O^2 A_ov Lam V_r^T

then each tilted leg is the parent's whitened-msign closed form under eps/2,
and Lambda_r ascends on the r x r residual of the JOINT realized dM/dN
(K warm-started iterations, best-iterate selection — msign chattering).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .compositional_halfsplit_muon import (
    CompositionalHalfSplitMuon,
    msign_batched,
)


def _top_r_of_gram(B: Tensor, r: int) -> Tuple[Tensor, Tensor]:
    """Top-r eigvecs/eigvals of B B^T (B thin (H, d, k)) via the small Gram."""
    G = B.mT @ B                                       # (H, k, k)
    G = 0.5 * (G + G.mT)
    ev, V = torch.linalg.eigh(G)                       # ascending
    ev_r = ev[..., -r:].clamp_min(1e-12)
    V_r = V[..., -r:]
    U = (B @ V_r) * ev_r.rsqrt().unsqueeze(-2)         # (H, d, r), orthonormal
    return U, ev_r


class PGramHalfSplitMuon(CompositionalHalfSplitMuon):
    """Half-split + composed-Gram (M^T M, N^T N) subspace preservation.

    New args (rest mirror CompositionalHalfSplitMuon):
        rank:        r, protected right-singular directions of M (and N) per head.
                     0 disables the coupling -> plain half-split.
        gamma:       contraction pulling the protected core back to its anchor.
        dual_steps:  K warm-started ascent iterations per step (default 2).
        dual_lr:     ascent step on the normalized residual (default 0.25).
        clamp_c:     feasibility clamp on the contraction demand (default 0.5).
        pgram_qk / pgram_ov: enable the coupling per circuit (default both True).
    """

    def __init__(self, named_params: List[Tuple[str, Tensor]], lr: float = 1e-3,
                 eps: float = 1.0, head_dim: int = 32, num_kv_heads: int = 4,
                 damping: float = 1e-6, ns_steps: int = 8,
                 whitening: str = "ns", track_every: int = 0,
                 attn_name_patterns: Optional[List[str]] = None,
                 rank: int = 8, gamma: float = 1e-3,
                 dual_steps: int = 2, dual_lr: float = 0.25,
                 clamp_c: float = 0.5,
                 pgram_qk: bool = True, pgram_ov: bool = True):
        super().__init__(named_params, lr=lr, eps=eps, head_dim=head_dim,
                         num_kv_heads=num_kv_heads, damping=damping,
                         preserve_qk=False, preserve_ov=False, ns_steps=ns_steps,
                         whitening=whitening, track_every=track_every,
                         attn_name_patterns=attn_name_patterns)
        if rank < 0 or rank > head_dim:
            raise ValueError(f"rank must be in [0, head_dim={head_dim}]")
        self.rank, self.gamma = rank, gamma
        self.dual_steps, self.dual_lr, self.clamp_c = dual_steps, dual_lr, clamp_c
        self.pgram_qk, self.pgram_ov = pgram_qk, pgram_ov
        self._anchor: Dict[str, Dict[str, Tensor]] = {}
        if rank > 0:
            self.snap_anchor()

    # ------------------------------------------------------------------ views
    def _views(self, lp):
        d_h, H_kv = self.head_dim, self.num_kv_heads
        p_Q, p_K, p_V, p_O = lp["q"], lp["k"], lp["v"], lp["o"]
        H_q = p_Q.shape[0] // d_h
        grp = max(H_q // H_kv, 1)
        d = p_Q.shape[1]
        WQ = p_Q.data.view(H_q, d_h, d).mT.float()
        WK = p_K.data.view(H_kv, d_h, d).mT.float().repeat_interleave(grp, 0)
        WO = p_O.data.view(d, H_q, d_h).permute(1, 0, 2).float()
        WV = p_V.data.view(H_kv, d_h, d).float().repeat_interleave(grp, 0)
        return (p_Q, p_K, p_V, p_O, WQ, WK, WO, WV, H_q, H_kv, grp, d)

    # ----------------------------------------------------------------- anchor
    @torch.no_grad()
    def snap_anchor(self) -> None:
        """Snap V_r and anchored cores of M^T M and N^T N from CURRENT weights."""
        r = self.rank
        self._anchor.clear()
        for lid, lp in self._layers.items():
            if not {"q", "k", "v", "o"} <= lp.keys():
                continue
            (_, _, _, _, WQ, WK, WO, WV, *_ ) = self._views(lp)
            a: Dict[str, Tensor] = {}
            if self.pgram_qk:
                CQ2 = WQ.mT @ WQ                                     # (H,d_h,d_h)
                L = torch.linalg.cholesky(
                    0.5 * (CQ2 + CQ2.mT)
                    + 1e-9 * torch.eye(CQ2.shape[-1], device=CQ2.device,
                                       dtype=CQ2.dtype))
                Vr, ev = _top_r_of_gram(WK @ L, r)                    # (H,d,r)
                a["qk_Vr"], a["qk_S"] = Vr, torch.diag_embed(ev)
                a["qk_Lam"] = torch.zeros_like(a["qk_S"])
            if self.pgram_ov:
                CO2 = WO.mT @ WO
                L = torch.linalg.cholesky(
                    0.5 * (CO2 + CO2.mT)
                    + 1e-9 * torch.eye(CO2.shape[-1], device=CO2.device,
                                       dtype=CO2.dtype))
                Vr, ev = _top_r_of_gram(WV.mT @ L, r)                 # (H,d,r)
                a["ov_Vr"], a["ov_S"] = Vr, torch.diag_embed(ev)
                a["ov_Lam"] = torch.zeros_like(a["ov_S"])
            self._anchor[lid] = a

    # ------------------------------------------------------------------- step
    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        lr = self.param_groups[0]["lr"]
        if lr == 0:
            return loss
        half = 0.5 * self.eps
        r, K = self.rank, max(1, self.dual_steps)

        for lid, lp in self._layers.items():
            if not {"q", "k", "v", "o"} <= lp.keys() or lp["q"].grad is None:
                continue
            (p_Q, p_K, p_V, p_O, WQ, WK, WO, WV, H_q, H_kv, grp, d) = self._views(lp)
            d_h = self.head_dim
            GQ = p_Q.grad.view(H_q, d_h, d).mT.float()
            GK = p_K.grad.view(H_kv, d_h, d).mT.float().repeat_interleave(grp, 0)
            GO = p_O.grad.view(d, H_q, d_h).permute(1, 0, 2).float()
            GV = p_V.grad.view(H_kv, d_h, d).float().repeat_interleave(grp, 0)
            an = self._anchor.get(lid, {}) if r > 0 else {}

            # ---------------- QK circuit ----------------
            CKi = self._isq(WK.mT @ WK)
            CQi = self._isq(WQ.mT @ WQ)
            if "qk_Vr" in an:
                Vr, S, Lam = an["qk_Vr"], an["qk_S"], an["qk_Lam"]
                A = Vr.mT @ WK                                       # (H,r,d_h)
                CQ2 = WQ.mT @ WQ
                E = A @ CQ2 @ A.mT - S                               # (r,r) leak
                s = (half * (CQi.norm(dim=(-2, -1)) * (A @ CQ2).norm(dim=(-2, -1))
                             + CKi.norm(dim=(-2, -1)) * A.norm(dim=(-2, -1)).pow(2))
                     ).clamp_min(1e-12)[:, None, None]
                e_norm = E.norm(dim=(-2, -1), keepdim=True)
                g_eff = torch.minimum(self.gamma * torch.ones_like(e_norm),
                                      self.clamp_c * s / e_norm.clamp_min(1e-12))
                best = None
                for _ in range(K):
                    tQ = 2.0 * (WQ @ (A.mT @ (Lam @ A)))             # (H,d,d_h)
                    tK = 2.0 * (Vr @ (Lam @ (A @ CQ2)))              # (H,d,d_h)
                    dQ_i = -half * (msign_batched((GQ + tQ) @ CKi, self.ns_steps) @ CKi)
                    dK_i = -half * (msign_batched((GK + tK) @ CQi, self.ns_steps) @ CQi)
                    P = A @ (WQ.mT @ dQ_i) @ A.mT + (A @ CQ2) @ (Vr.mT @ dK_i).mT
                    resid = (P + P.mT) + g_eff * E
                    score = float((resid.norm(dim=(-2, -1), keepdim=True) / s).mean())
                    if best is None or score < best[0]:
                        best = (score, dQ_i, dK_i)
                    Lam = 0.5 * ((Lam + self.dual_lr * resid / s)
                                 + (Lam + self.dual_lr * resid / s).mT)
                an["qk_Lam"] = torch.nan_to_num(Lam, nan=0.0, posinf=0.0, neginf=0.0)
                dQ, dK_rep = best[1], best[2]
                if self.track_every and self._nstep % self.track_every == 0:
                    self._viol["pgram/qk_leak"] = best[0]
                    self._viol["pgram/qk_E_fro"] = float(E.norm(dim=(-2, -1)).mean())
            else:
                dQ = -half * (msign_batched(GQ @ CKi, self.ns_steps) @ CKi)
                dK_rep = -half * (msign_batched(GK @ CQi, self.ns_steps) @ CQi)
            dK = dK_rep.view(H_kv, grp, d, d_h).mean(1)

            # ---------------- OV circuit ----------------
            CVi = self._isq(WV @ WV.mT)
            COi = self._isq(WO.mT @ WO)
            if "ov_Vr" in an:
                Vr, S, Lam = an["ov_Vr"], an["ov_S"], an["ov_Lam"]
                Aov = WV @ Vr                                        # (H,d_h,r)
                CO2 = WO.mT @ WO
                E = Aov.mT @ CO2 @ Aov - S
                s = (half * (CVi.norm(dim=(-2, -1)) * (CO2 @ Aov).norm(dim=(-2, -1))
                             + COi.norm(dim=(-2, -1)) * Aov.norm(dim=(-2, -1)).pow(2))
                     ).clamp_min(1e-12)[:, None, None]
                e_norm = E.norm(dim=(-2, -1), keepdim=True)
                g_eff = torch.minimum(self.gamma * torch.ones_like(e_norm),
                                      self.clamp_c * s / e_norm.clamp_min(1e-12))
                best = None
                for _ in range(K):
                    tO = 2.0 * (WO @ (Aov @ (Lam @ Aov.mT)))         # (H,d,d_h)
                    tV = 2.0 * (CO2 @ (Aov @ (Lam @ Vr.mT)))         # (H,d_h,d)
                    dO_i = -half * (msign_batched((GO + tO) @ CVi, self.ns_steps) @ CVi)
                    dV_i = -half * (COi @ msign_batched(COi @ (GV + tV), self.ns_steps))
                    P = Aov.mT @ (WO.mT @ dO_i) @ Aov + Aov.mT @ CO2 @ (dV_i @ Vr)
                    resid = (P + P.mT) + g_eff * E
                    score = float((resid.norm(dim=(-2, -1), keepdim=True) / s).mean())
                    if best is None or score < best[0]:
                        best = (score, dO_i, dV_i)
                    Lam = 0.5 * ((Lam + self.dual_lr * resid / s)
                                 + (Lam + self.dual_lr * resid / s).mT)
                an["ov_Lam"] = torch.nan_to_num(Lam, nan=0.0, posinf=0.0, neginf=0.0)
                dO, dV_rep = best[1], best[2]
                if self.track_every and self._nstep % self.track_every == 0:
                    self._viol["pgram/ov_leak"] = best[0]
                    self._viol["pgram/ov_E_fro"] = float(E.norm(dim=(-2, -1)).mean())
            else:
                dO = -half * (msign_batched(GO @ CVi, self.ns_steps) @ CVi)
                dV_rep = -half * (COi @ msign_batched(COi @ GV, self.ns_steps))
            dV = dV_rep.view(H_kv, grp, d_h, d).mean(1)

            ok = all(torch.isfinite(x).all() for x in (dQ, dK, dO, dV))
            if ok:
                self._apply(p_Q, dQ.mT.reshape(H_q * d_h, d), lr)
                self._apply(p_K, dK.mT.reshape(H_kv * d_h, d), lr)
                self._apply(p_O, dO.permute(1, 0, 2).reshape(d, H_q * d_h), lr)
                self._apply(p_V, dV.reshape(H_kv * d_h, d), lr)
            else:
                self.n_bad += 1
        self._nstep += 1
        return loss
