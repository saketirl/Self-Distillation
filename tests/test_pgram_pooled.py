"""Unit tests for PGramPooledMuon (GQA, pooled anchors)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from frozen_residual_ubv.pgram_pooled_muon import PGramPooledMuon

D, DH, HQ, HKV, R = 64, 16, 8, 2, 4
GRP = HQ // HKV
torch.manual_seed(0)


def make_params(seed=0):
    g = torch.Generator().manual_seed(seed)
    def mat(rows, cols):
        return torch.nn.Parameter(0.1 * torch.randn(rows, cols, generator=g))
    return [
        ("model.layers.0.self_attn.q_proj.weight", mat(HQ * DH, D)),
        ("model.layers.0.self_attn.k_proj.weight", mat(HKV * DH, D)),
        ("model.layers.0.self_attn.v_proj.weight", mat(HKV * DH, D)),
        ("model.layers.0.self_attn.o_proj.weight", mat(D, HQ * DH)),
    ]


def set_grads(named, seed=1):
    g = torch.Generator().manual_seed(seed)
    for _, p in named:
        p.grad = torch.randn(p.shape, generator=g)


def make_opt(named, **kw):
    args = dict(lr=1e-2, eps=1.0, head_dim=DH, num_kv_heads=HKV,
                damping=1e-3, rank=R, gamma=1e-3, dual_steps=2, track_every=1)
    args.update(kw)
    return PGramPooledMuon(named, **args)


def test_reshape_roundtrip():
    """The (H_kv, grp) views and their inverse reshapes must be exact."""
    named = make_params()
    opt = make_opt(named, lr=0.0)
    lp = next(iter(opt._layers.values()))
    (p_Q, p_K, p_V, p_O, WQ, WK, WO, WV, H_q, H_kv, grp, d) = opt._kv_views(lp)
    assert torch.allclose(WQ.mT.reshape(H_q * DH, d), p_Q.data.float())
    assert torch.allclose(WK.mT.reshape(H_kv * DH, d), p_K.data.float())
    assert torch.allclose(WO.permute(2, 0, 1, 3).reshape(d, H_q * DH), p_O.data.float())
    assert torch.allclose(WV.reshape(H_kv * DH, d), p_V.data.float())


def test_step_moves_and_finite():
    named = make_params()
    opt = make_opt(named)
    before = [p.detach().clone() for _, p in named]
    set_grads(named)
    opt.step()
    moved = sum(float((p.detach() - b).abs().sum()) for (_, p), b in zip(named, before))
    assert moved > 1e-6
    for _, p in named:
        assert torch.isfinite(p.detach()).all()


def test_ball_holds_for_every_sibling():
    """Group-sum whitening: ||W_{Q,h} dK^T||_op <= eps/2 for EVERY Q head."""
    named = make_params()
    opt = make_opt(named, lr=1.0)
    WQ_pre = dict(named)["model.layers.0.self_attn.q_proj.weight"].detach().clone()
    before = dict(named)["model.layers.0.self_attn.k_proj.weight"].detach().clone()
    set_grads(named)
    opt.step()
    dK = (dict(named)["model.layers.0.self_attn.k_proj.weight"].detach() - before)
    WQh = WQ_pre.view(HQ, DH, D).mT.float()
    dKh = dK.view(HKV, DH, D).mT.float().repeat_interleave(GRP, 0)
    ratio = float(torch.linalg.matrix_norm(WQh @ dKh.mT, ord=2).max()) / 0.5
    assert ratio <= 1.10, ratio


def test_anchor_orthonormal_and_pooled():
    named = make_params()
    opt = make_opt(named)
    a = next(iter(opt._anchor.values()))
    for key in ("qk_Vr", "ov_Vr"):
        Vr = a[key]
        assert Vr.shape == (HKV, D, R)
        I = torch.eye(R).expand(HKV, R, R)
        assert torch.allclose(Vr.mT @ Vr, I, atol=1e-4)


def test_dual_reduces_pooled_leak():
    def leak_of(rank):
        named = make_params(seed=3)
        opt = make_opt(named, rank=rank, gamma=0.0, lr=1e-3)
        lp = next(iter(opt._layers.values()))
        (_, p_K, _, _, WQ0, WK0, *_ ) = opt._kv_views(lp)
        WQ0, WK0 = WQ0.clone(), WK0.clone()
        CG2 = (WQ0.mT @ WQ0).sum(1)
        before = p_K.detach().clone()
        qs = dict(named)["model.layers.0.self_attn.q_proj.weight"].detach().clone()
        for _ in range(6):
            set_grads(named, seed=5)
            opt.step()
        dK = (p_K.detach() - before).view(HKV, DH, D).mT.float()
        dQ = (dict(named)["model.layers.0.self_attn.q_proj.weight"].detach()
              - qs).view(HKV, GRP, DH, D).mT.float()
        # pooled D[G_g] = sum_h (dM_h^T M_h + M_h^T dM_h) restricted to ref Vr
        ref = PGramPooledMuon(make_params(seed=3), lr=1e-3, eps=1.0, head_dim=DH,
                              num_kv_heads=HKV, damping=1e-3, rank=R, gamma=0.0)
        Vr = next(iter(ref._anchor.values()))["qk_Vr"]
        A = Vr.mT @ WK0
        P = ((A.unsqueeze(1) @ (WQ0.mT @ dQ) @ A.unsqueeze(1).mT).sum(1)
             + (A @ CG2) @ (Vr.mT @ dK).mT)
        return float((P + P.mT).norm())

    leak_on = leak_of(R)
    leak_off = leak_of(0)
    assert leak_on < leak_off * 0.8, (leak_on, leak_off)


def test_degenerate_grads_no_nans():
    named = make_params()
    opt = make_opt(named)
    for bad in (torch.zeros, lambda s: torch.full(s, float("nan"))):
        for _, p in named:
            p.grad = bad(p.shape)
        opt.step()
        for _, p in named:
            assert torch.isfinite(p.detach()).all()


def test_lr0_noop():
    named = make_params()
    opt = make_opt(named, lr=0.0)
    before = [p.detach().clone() for _, p in named]
    set_grads(named)
    opt.step()
    for (_, p), b in zip(named, before):
        assert torch.equal(p.detach(), b)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
