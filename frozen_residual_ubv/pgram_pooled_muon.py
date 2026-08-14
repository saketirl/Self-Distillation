"""PGramPooledMuon — P-Gram half-split with GROUP-POOLED anchors for GQA.

The production (Qwen/Gemma-scale) form of PGramHalfSplitMuon. Differences from
the ICL class:

1. **Pooled atom per KV head.** Under GQA one W_K serves a group G of query
   heads; the protected object is the pooled composed Gram

       G_g = sum_{h in G} M_h^T M_h = W_K C_G^2 W_K^T,
       C_G^2 = sum_{h in G} W_{Q,h}^T W_{Q,h},

   one anchor V_r and one Sym(r) dual per KV head per circuit (not per Q head).
   Same pooling for OV: sum_h N_h^T N_h = W_V^T (sum_h C_{O,h}^2) W_V.

2. **Group-sum whitening for the shared K/V legs** (the CoupledKeyHalfSplit
   GQA fix): W_{Q,h}^T W_{Q,h} <= C_G^2 gives ||W_{Q,h} dK^T||_op <= eps/2 for
   EVERY sibling, with ONE solve per KV head — no repeat/average. The same
   C_G^2 defines the ball geometry and the preserved Gram.

3. Q/O legs stay per Q-head (unshared), whitened by their shared partner's
   Gram as in the parent.

Dual game: Lambda_r in Sym(r) per KV head per circuit tilts all its legs
(each sibling Q-leg once, the shared K-leg once); K warm-started iterations,
best-iterate selection. Tilts, residuals, anchors: all thin products.

Tilt algebra (QK; OV mirrors). With Vr (d,r), A = Vr^T W_K (r,d_h),
Lam in Sym(r), pooled constraint sum_h Vr^T sym(M_h^T dM_h) Vr = -gamma E_r:
    dW_{Q,h} tilt: 2 W_{Q,h} A^T Lam A            (per sibling)
    dW_K tilt:     2 Vr Lam A C_G^2               (once, pooled metric)
Residual from realized updates:
    P = sum_h A (W_{Q,h}^T dQ_h) A^T  +  (A C_G^2) (Vr^T dK).T
    resid = (P + P^T) + gamma_eff E_r,   E_r = A C_G^2 A^T - S_anchor.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .compositional_halfsplit_muon import (
    CompositionalHalfSplitMuon,
    msign_batched,
)
from .pgram_halfsplit_muon import _top_r_of_gram


def _chol_damped(G: Tensor) -> Tensor:
    G = 0.5 * (G + G.mT)
    eye = torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
    lam = 1e-9 + 1e-9 * G.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(0.0)
    return torch.linalg.cholesky(G + lam[..., None, None] * eye)


class PGramPooledMuon(CompositionalHalfSplitMuon):
    """Pooled-anchor P-Gram for GQA models (one anchor/dual per KV head)."""

    def __init__(self, named_params: List[Tuple[str, Tensor]], lr: float = 1e-3,
                 eps: float = 1.0, head_dim: int = 128, num_kv_heads: int = 8,
                 damping: float = 1e-3, ns_steps: int = 8,
                 whitening: str = "ns", track_every: int = 0,
                 attn_name_patterns: Optional[List[str]] = None,
                 rank: int = 8, gamma: float = 1e-3,
                 dual_steps: int = 2, dual_lr: float = 2.0,   # pooled dual is smooth — no chattering (see tests/debug_pooled_dual.py)
                 clamp_c: float = 0.5):
        super().__init__(named_params, lr=lr, eps=eps, head_dim=head_dim,
                         num_kv_heads=num_kv_heads, damping=damping,
                         preserve_qk=False, preserve_ov=False, ns_steps=ns_steps,
                         whitening=whitening, track_every=track_every,
                         attn_name_patterns=attn_name_patterns)
        if rank < 0 or rank > head_dim:
            raise ValueError(f"rank must be in [0, head_dim={head_dim}]")
        self.rank, self.gamma = rank, gamma
        self.dual_steps, self.dual_lr, self.clamp_c = dual_steps, dual_lr, clamp_c
        self._anchor: Dict[str, Dict[str, Tensor]] = {}
        if rank > 0:
            self.snap_anchor()

    # ------------------------------------------------------------------ views
    def _kv_views(self, lp):
        """KV-granular views. Q/O grouped as (H_kv, grp, d, d_h)."""
        d_h, H_kv = self.head_dim, self.num_kv_heads
        p_Q, p_K, p_V, p_O = lp["q"], lp["k"], lp["v"], lp["o"]
        H_q = p_Q.shape[0] // d_h
        grp = max(H_q // H_kv, 1)
        d = p_Q.shape[1]
        WQ = p_Q.data.view(H_kv, grp, d_h, d).mT.float()      # (H_kv,grp,d,d_h)
        WK = p_K.data.view(H_kv, d_h, d).mT.float()           # (H_kv,d,d_h)
        WO = p_O.data.view(d, H_kv, grp, d_h).permute(1, 2, 0, 3).float()
        WV = p_V.data.view(H_kv, d_h, d).float()              # (H_kv,d_h,d)
        return p_Q, p_K, p_V, p_O, WQ, WK, WO, WV, H_q, H_kv, grp, d

    # ----------------------------------------------------------------- anchor
    @torch.no_grad()
    def snap_anchor(self) -> None:
        r = self.rank
        self._anchor.clear()
        for lid, lp in self._layers.items():
            if not {"q", "k", "v", "o"} <= lp.keys():
                continue
            (_, _, _, _, WQ, WK, WO, WV, *_ ) = self._kv_views(lp)
            CG2 = (WQ.mT @ WQ).sum(1)                          # (H_kv,d_h,d_h)
            Vr, ev = _top_r_of_gram(WK @ _chol_damped(CG2), r)
            a = {"qk_Vr": Vr, "qk_S": torch.diag_embed(ev),
                 "qk_Lam": torch.zeros(WK.shape[0], r, r, device=WK.device,
                                       dtype=WK.dtype)}
            CO2 = (WO.mT @ WO).sum(1)
            Vr2, ev2 = _top_r_of_gram(WV.mT @ _chol_damped(CO2), r)
            a.update({"ov_Vr": Vr2, "ov_S": torch.diag_embed(ev2),
                      "ov_Lam": torch.zeros_like(a["qk_Lam"])})
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
            (p_Q, p_K, p_V, p_O, WQ, WK, WO, WV,
             H_q, H_kv, grp, d) = self._kv_views(lp)
            d_h = self.head_dim
            GQ = p_Q.grad.view(H_kv, grp, d_h, d).mT.float()
            GK = p_K.grad.view(H_kv, d_h, d).mT.float()
            GO = p_O.grad.view(d, H_kv, grp, d_h).permute(1, 2, 0, 3).float()
            GV = p_V.grad.view(H_kv, d_h, d).float()
            an = self._anchor.get(lid, {}) if r > 0 else {}

            # ================= QK circuit =================
            CG2 = (WQ.mT @ WQ).sum(1)                          # pooled Q metric
            CGi = self._isq(CG2)                               # (H_kv,d_h,d_h)
            CKi = self._isq(WK.mT @ WK)                        # for Q legs
            if "qk_Vr" in an and r > 0:
                Vr, S, Lam = an["qk_Vr"], an["qk_S"], an["qk_Lam"]
                A = Vr.mT @ WK                                 # (H_kv,r,d_h)
                ACG = A @ CG2
                E = ACG @ A.mT - S
                s = (half * (CGi.norm(dim=(-2, -1)) * ACG.norm(dim=(-2, -1))
                             + grp * CKi.norm(dim=(-2, -1))
                             * A.norm(dim=(-2, -1)).pow(2))).clamp_min(1e-12)[:, None, None]
                e_norm = E.norm(dim=(-2, -1), keepdim=True)
                g_eff = torch.minimum(self.gamma * torch.ones_like(e_norm),
                                      self.clamp_c * s / e_norm.clamp_min(1e-12))
                best = None
                for _ in range(K):
                    tQ = 2.0 * (WQ @ (A.mT @ (Lam @ A)).unsqueeze(1))
                    tK = 2.0 * (Vr @ (Lam @ ACG))
                    dQ_i = -half * (msign_batched(
                        (GQ + tQ).flatten(0, 1) @ CKi.repeat_interleave(grp, 0),
                        self.ns_steps) @ CKi.repeat_interleave(grp, 0)
                    ).view(H_kv, grp, d, d_h)
                    dK_i = -half * (msign_batched((GK + tK) @ CGi, self.ns_steps) @ CGi)
                    P = ((A.unsqueeze(1) @ (WQ.mT @ dQ_i) @ A.unsqueeze(1).mT).sum(1)
                         + ACG @ (Vr.mT @ dK_i).mT)
                    resid = (P + P.mT) + g_eff * E
                    score = float((resid.norm(dim=(-2, -1), keepdim=True) / s).mean())
                    if best is None or score < best[0]:
                        best = (score, dQ_i, dK_i)
                    Lam = 0.5 * ((Lam + self.dual_lr * resid / s)
                                 + (Lam + self.dual_lr * resid / s).mT)
                an["qk_Lam"] = torch.nan_to_num(Lam, nan=0.0, posinf=0.0, neginf=0.0)
                dQ, dK = best[1], best[2]
                if self.track_every and self._nstep % self.track_every == 0:
                    self._viol["pgram/qk_leak"] = best[0]
                    self._viol["pgram/qk_E_fro"] = float(E.norm(dim=(-2, -1)).mean())
            else:
                dQ = -half * (msign_batched(
                    GQ.flatten(0, 1) @ CKi.repeat_interleave(grp, 0), self.ns_steps
                ) @ CKi.repeat_interleave(grp, 0)).view(H_kv, grp, d, d_h)
                dK = -half * (msign_batched(GK @ CGi, self.ns_steps) @ CGi)

            # ================= OV circuit =================
            CO2 = (WO.mT @ WO).sum(1)                          # pooled O metric
            COg = self._isq(CO2)                               # for shared V leg
            CVi = self._isq(WV @ WV.mT)                        # for O legs
            if "ov_Vr" in an and r > 0:
                Vr, S, Lam = an["ov_Vr"], an["ov_S"], an["ov_Lam"]
                Aov = WV @ Vr                                  # (H_kv,d_h,r)
                CA = CO2 @ Aov
                E = Aov.mT @ CA - S
                s = (half * (COg.norm(dim=(-2, -1)) * CA.norm(dim=(-2, -1))
                             + grp * CVi.norm(dim=(-2, -1))
                             * Aov.norm(dim=(-2, -1)).pow(2))).clamp_min(1e-12)[:, None, None]
                e_norm = E.norm(dim=(-2, -1), keepdim=True)
                g_eff = torch.minimum(self.gamma * torch.ones_like(e_norm),
                                      self.clamp_c * s / e_norm.clamp_min(1e-12))
                best = None
                for _ in range(K):
                    tO = 2.0 * (WO @ (Aov @ (Lam @ Aov.mT)).unsqueeze(1))
                    tV = 2.0 * (CA @ (Lam @ Vr.mT))
                    dO_i = -half * (msign_batched(
                        (GO + tO).flatten(0, 1) @ CVi.repeat_interleave(grp, 0),
                        self.ns_steps) @ CVi.repeat_interleave(grp, 0)
                    ).view(H_kv, grp, d, d_h)
                    dV_i = -half * (COg @ msign_batched(COg @ (GV + tV), self.ns_steps))
                    P = ((Aov.unsqueeze(1).mT @ (WO.mT @ dO_i) @ Aov.unsqueeze(1)).sum(1)
                         + Aov.mT @ CO2 @ (dV_i @ Vr))
                    resid = (P + P.mT) + g_eff * E
                    score = float((resid.norm(dim=(-2, -1), keepdim=True) / s).mean())
                    if best is None or score < best[0]:
                        best = (score, dO_i, dV_i)
                    Lam = 0.5 * ((Lam + self.dual_lr * resid / s)
                                 + (Lam + self.dual_lr * resid / s).mT)
                an["ov_Lam"] = torch.nan_to_num(Lam, nan=0.0, posinf=0.0, neginf=0.0)
                dO, dV = best[1], best[2]
                if self.track_every and self._nstep % self.track_every == 0:
                    self._viol["pgram/ov_leak"] = best[0]
                    self._viol["pgram/ov_E_fro"] = float(E.norm(dim=(-2, -1)).mean())
            else:
                dO = -half * (msign_batched(
                    GO.flatten(0, 1) @ CVi.repeat_interleave(grp, 0), self.ns_steps
                ) @ CVi.repeat_interleave(grp, 0)).view(H_kv, grp, d, d_h)
                dV = -half * (COg @ msign_batched(COg @ GV, self.ns_steps))

            ok = all(torch.isfinite(x).all() for x in (dQ, dK, dO, dV))
            if ok:
                self._apply(p_Q, dQ.mT.reshape(H_q * d_h, d), lr)
                self._apply(p_K, dK.mT.reshape(H_kv * d_h, d), lr)
                self._apply(p_O, dO.permute(2, 0, 1, 3).reshape(d, H_q * d_h), lr)
                self._apply(p_V, dV.reshape(H_kv * d_h, d), lr)
            else:
                self.n_bad += 1
        self._nstep += 1
        return loss
