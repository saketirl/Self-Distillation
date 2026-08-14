import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent.parent))
from frozen_residual_ubv.compositional_halfsplit_muon import msign_batched, inv_sqrt_batched
from frozen_residual_ubv.pgram_halfsplit_muon import _top_r_of_gram

torch.manual_seed(0)
D, DH, HKV, GRP, R = 64, 16, 2, 4, 4
half = 0.5
g = torch.Generator().manual_seed(7)
WQ = 0.1 * torch.randn(HKV, GRP, D, DH, generator=g)
WK = 0.1 * torch.randn(HKV, D, DH, generator=g)
GQ = torch.randn(HKV, GRP, D, DH, generator=g)
GK = torch.randn(HKV, D, DH, generator=g)

CG2 = (WQ.mT @ WQ).sum(1)
CGi = inv_sqrt_batched(CG2, 1e-3)
CKi = inv_sqrt_batched(WK.mT @ WK, 1e-3)
L = torch.linalg.cholesky(0.5 * (CG2 + CG2.mT) + 1e-9 * torch.eye(DH))
Vr, ev = _top_r_of_gram(WK @ L, R)
A = Vr.mT @ WK
ACG = A @ CG2
s = (half * (CGi.norm(dim=(-2, -1)) * ACG.norm(dim=(-2, -1))
             + GRP * CKi.norm(dim=(-2, -1)) * A.norm(dim=(-2, -1)).pow(2))
     ).clamp_min(1e-12)[:, None, None]

for dl in (0.25, 1.0, 4.0):
    Lam = torch.zeros(HKV, R, R)
    scores = []
    for it in range(40):
        tQ = 2.0 * (WQ @ (A.mT @ (Lam @ A)).unsqueeze(1))
        tK = 2.0 * (Vr @ (Lam @ ACG))
        dQ = -half * (msign_batched((GQ + tQ).flatten(0, 1) @ CKi.repeat_interleave(GRP, 0), 8)
                      @ CKi.repeat_interleave(GRP, 0)).view(HKV, GRP, D, DH)
        dK = -half * (msign_batched((GK + tK) @ CGi, 8) @ CGi)
        P = ((A.unsqueeze(1) @ (WQ.mT @ dQ) @ A.unsqueeze(1).mT).sum(1)
             + ACG @ (Vr.mT @ dK).mT)
        resid = P + P.mT
        scores.append(float((resid.norm(dim=(-2, -1), keepdim=True) / s).mean()))
        Lam = Lam + dl * resid / s
        Lam = 0.5 * (Lam + Lam.mT)
    print(f"dual_lr={dl}: it0={scores[0]:.4f} it5={scores[5]:.4f} it10={scores[10]:.4f} "
          f"it20={scores[20]:.4f} it39={scores[39]:.4f} min={min(scores):.4f}@{scores.index(min(scores))}")
