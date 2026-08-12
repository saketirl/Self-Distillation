"""Unit tests for CoupledKeyHalfSplitMuon.

Mirrors the structure of tests/test_soft_comp_preserving_muon.py: build a tiny
GQA attention weight set, drive the optimizer with crafted gradients, and check
the constraint algebra, the recovery limits, and the numerical guards.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from frozen_residual_ubv.compositional_halfsplit_muon import CompositionalHalfSplitMuon
from frozen_residual_ubv.coupled_key_halfsplit_muon import CoupledKeyHalfSplitMuon

D, DH, HQ, HKV = 32, 8, 4, 2      # d_model, head_dim, q heads, kv heads
R = 3                             # protected rank per kv head
torch.manual_seed(0)


def make_params(seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    def mat(rows):
        return torch.nn.Parameter(0.1 * torch.randn(rows, D, generator=g, dtype=dtype))
    named = [
        (f"model.layers.0.self_attn.{n}_proj.weight",
         mat(HQ * DH if n in ("q", "o") else HKV * DH))
        for n in ("q", "k", "v", "o")
    ]
    # o_proj is (d, H_q*d_h) in nn.Linear; fix its shape
    named[3] = (named[3][0], torch.nn.Parameter(0.1 * torch.randn(D, HQ * DH, generator=g, dtype=dtype)))
    return named


def set_grads(named, seed=1):
    g = torch.Generator().manual_seed(seed)
    for _, p in named:
        p.grad = torch.randn(p.shape, generator=g, dtype=p.dtype)


def kv_heads(p):
    return p.data.view(HKV, DH, D).mT.float()          # (H_kv, d, d_h)


def make_opt(named, **kw):
    args = dict(lr=1e-2, eps=1.0, head_dim=DH, num_kv_heads=HKV,
                damping=1e-3, rank=R, gamma=1e-2, dual_steps=2, track_every=1)
    args.update(kw)
    return CoupledKeyHalfSplitMuon(named, **args)


# --------------------------------------------------------------------- basics

def test_step_moves_weights():
    named = make_params()
    opt = make_opt(named)
    before = [p.detach().clone() for _, p in named]
    set_grads(named)
    opt.step()
    moved = sum(float((p.detach() - b).abs().sum()) for (_, p), b in zip(named, before))
    assert moved > 1e-6


def test_lr_zero_is_noop():
    named = make_params()
    opt = make_opt(named, lr=0.0)
    before = [p.detach().clone() for _, p in named]
    set_grads(named)
    opt.step()
    for (_, p), b in zip(named, before):
        assert torch.equal(p.detach(), b)


def test_anchor_snap_is_orthonormal_and_diagonal():
    named = make_params()
    opt = make_opt(named)
    cd = next(iter(opt._coupled.values()))
    Ur, S2 = cd["Ur"], cd["S2"]
    I = torch.eye(R).expand(HKV, R, R)
    assert torch.allclose(Ur.mT @ Ur, I, atol=1e-4)
    # U_r^T M_kk U_r == anchored core (diagonal top-r eigenvalues) at the snap
    WK = kv_heads(dict(named)["model.layers.0.self_attn.k_proj.weight"])
    core = (Ur.mT @ WK) @ (Ur.mT @ WK).mT
    assert torch.allclose(core, S2, atol=1e-4)
    off = S2 - torch.diag_embed(S2.diagonal(dim1=-2, dim2=-1))
    assert float(off.abs().max()) < 1e-6


# --------------------------------------------------- constraint satisfaction

def leg_ratio(named, dK_per_param, eps=1.0):
    """max_h ||W_Q_h dK_h^T||_op / (eps/2) for the APPLIED update."""
    WQ = dict(named)["model.layers.0.self_attn.q_proj.weight"].data \
        .view(HQ, DH, D).mT.float()
    dK = dK_per_param.view(HKV, DH, D).mT.float().repeat_interleave(HQ // HKV, 0)
    comp = WQ @ dK.mT                                   # (H_q, d, d)
    return float(torch.linalg.matrix_norm(comp, ord=2).max()) / (eps / 2)


def test_k_leg_ball_holds():
    named = make_params()
    opt = make_opt(named, lr=1.0)     # lr=1 so the applied delta IS the solve
    # constraint is defined at the PRE-step point: snapshot W_Q before stepping
    WQ_pre = dict(named)["model.layers.0.self_attn.q_proj.weight"].detach().clone()
    before = dict(named)["model.layers.0.self_attn.k_proj.weight"].detach().clone()
    set_grads(named)
    opt.step()
    dK = dict(named)["model.layers.0.self_attn.k_proj.weight"].detach() - before
    WQ = WQ_pre.view(HQ, DH, D).mT.float()
    dKh = dK.view(HKV, DH, D).mT.float().repeat_interleave(HQ // HKV, 0)
    ratio = float(torch.linalg.matrix_norm(WQ @ dKh.mT, ord=2).max()) / 0.5
    # damping in C_Q makes the ball conservative; allow 10% slack for the
    # Polar-Express msign overshoot
    assert ratio <= 1.10, ratio


def test_dual_ascent_reduces_protected_leak():
    """With duals on, ||U_r^T D[dK] U_r|| must drop vs the uncoupled update.

    Small lr so the weights stay ~static and the cumulative delta reflects
    per-step solves at (approximately) one point; fixed gradient so the
    warm-started duals get a stationary target to converge against.
    """
    def leak_of(dual_steps, rank):
        named = make_params()
        # dual_lr default (0.25): 1.0 chatters — see debug trace in the PR/notes
        opt = make_opt(named, dual_steps=dual_steps, rank=rank, gamma=0.0,
                       lr=1e-3)
        cd = next(iter(opt._coupled.values())) if rank > 0 else None
        WK0 = kv_heads(dict(named)["model.layers.0.self_attn.k_proj.weight"]).clone()
        before = dict(named)["model.layers.0.self_attn.k_proj.weight"].detach().clone()
        for _ in range(6):
            set_grads(named, seed=3)
            opt.step()
        after = dict(named)["model.layers.0.self_attn.k_proj.weight"].detach()
        dK = (after - before).view(HKV, DH, D).mT.float()
        if cd is None:
            # measure leak in the subspace a fresh snap would protect
            ref = CoupledKeyHalfSplitMuon(make_params(), lr=1.0, eps=1.0,
                                          head_dim=DH, num_kv_heads=HKV,
                                          damping=1e-3, rank=R, gamma=0.0)
            Ur = next(iter(ref._coupled.values()))["Ur"]
        else:
            Ur = cd["Ur"]
        Dop = dK @ WK0.mT + WK0 @ dK.mT
        return float((Ur.mT @ Dop @ Ur).norm())

    leak_off = leak_of(dual_steps=0, rank=0)
    leak_on = leak_of(dual_steps=2, rank=R)
    assert leak_on < leak_off * 0.7, (leak_on, leak_off)


def test_contraction_bounds_gram_drift():
    """A Muon-class step has unit size regardless of gradient magnitude, so
    "zero gradient => pure contraction" is not a meaningful regime. The right
    property: under IDENTICAL gradients from a perturbed-anchor start, gamma>0
    must leave the protected core closer to the anchor than gamma=0."""
    def e_after(gamma):
        named = make_params()
        opt = make_opt(named, gamma=gamma, dual_steps=4, lr=2e-3)
        cd = next(iter(opt._coupled.values()))
        pK = dict(named)["model.layers.0.self_attn.k_proj.weight"]
        with torch.no_grad():                            # create leak E != 0
            g = torch.Generator().manual_seed(42)
            pK.add_(0.05 * torch.randn(pK.shape, generator=g))
        for i in range(20):
            set_grads(named, seed=100 + i)
            opt.step()
        WK = kv_heads(pK)
        UW = cd["Ur"].mT @ WK
        return float((UW @ UW.mT - cd["S2"]).norm())

    e_no, e_yes = e_after(0.0), e_after(0.5)
    assert e_yes < e_no, (e_yes, e_no)


# ------------------------------------------------------------- recovery limits

def make_params_mha(seed=0):
    """MHA config: H_q == H_kv, group size 1."""
    g = torch.Generator().manual_seed(seed)
    def mat(rows):
        return torch.nn.Parameter(0.1 * torch.randn(rows, D, generator=g))
    named = [(f"model.layers.0.self_attn.{n}_proj.weight", mat(HQ * DH))
             for n in ("q", "k", "v")]
    named.append(("model.layers.0.self_attn.o_proj.weight",
                  torch.nn.Parameter(0.1 * torch.randn(D, HQ * DH, generator=g))))
    return named


def test_rank0_recovers_plain_halfsplit_mha():
    """For MHA (group size 1) rank=0 must reproduce CompositionalHalfSplitMuon
    exactly. Under GQA the k-leg deliberately differs (group-sum whitening
    restores the per-head leg guarantee the parent's averaging violates), so
    exact recovery is asserted for grp == 1 only."""
    named_a, named_b = make_params_mha(seed=7), make_params_mha(seed=7)
    a = CoupledKeyHalfSplitMuon(named_a, lr=1e-2, eps=1.0, head_dim=DH,
                                num_kv_heads=HQ, damping=1e-3, rank=0)
    b = CompositionalHalfSplitMuon(named_b, lr=1e-2, eps=1.0, head_dim=DH,
                                   num_kv_heads=HQ, damping=1e-3)
    set_grads(named_a, seed=9), set_grads(named_b, seed=9)
    a.step(), b.step()
    for (_, pa), (_, pb) in zip(named_a, named_b):
        diff = float((pa.detach() - pb.detach()).abs().max())
        assert torch.allclose(pa.detach(), pb.detach(), atol=1e-6), f"mismatch {diff}"


def test_gqa_ball_holds_where_parent_violates():
    """The reason the k-leg diverges from the parent under GQA: the parent's
    group-averaged K update violates the per-head leg bound (measured >4x on
    random weights); the group-sum whitening must keep it <= eps/2."""
    named_p = make_params(seed=13)
    parent = CompositionalHalfSplitMuon(named_p, lr=1.0, eps=1.0, head_dim=DH,
                                        num_kv_heads=HKV, damping=1e-3)
    WQ_pre = dict(named_p)["model.layers.0.self_attn.q_proj.weight"].detach().clone()
    before = dict(named_p)["model.layers.0.self_attn.k_proj.weight"].detach().clone()
    set_grads(named_p, seed=14)
    parent.step()
    dK = dict(named_p)["model.layers.0.self_attn.k_proj.weight"].detach() - before
    WQ = WQ_pre.view(HQ, DH, D).mT.float()
    dKh = dK.view(HKV, DH, D).mT.float().repeat_interleave(HQ // HKV, 0)
    parent_ratio = float(torch.linalg.matrix_norm(WQ @ dKh.mT, ord=2).max()) / 0.5

    named_c = make_params(seed=13)
    ours = make_opt(named_c, lr=1.0)
    WQ_pre = dict(named_c)["model.layers.0.self_attn.q_proj.weight"].detach().clone()
    before = dict(named_c)["model.layers.0.self_attn.k_proj.weight"].detach().clone()
    set_grads(named_c, seed=14)
    ours.step()
    dK = dict(named_c)["model.layers.0.self_attn.k_proj.weight"].detach() - before
    WQ = WQ_pre.view(HQ, DH, D).mT.float()
    dKh = dK.view(HKV, DH, D).mT.float().repeat_interleave(HQ // HKV, 0)
    our_ratio = float(torch.linalg.matrix_norm(WQ @ dKh.mT, ord=2).max()) / 0.5

    assert our_ratio <= 1.10, our_ratio
    assert our_ratio < parent_ratio, (our_ratio, parent_ratio)


# ------------------------------------------------------------------- guards

def test_degenerate_gradients_no_nans():
    named = make_params()
    opt = make_opt(named)
    for bad in (torch.zeros, lambda s: 1e-15 * torch.ones(s),
                lambda s: torch.full(s, float("nan"))):
        for _, p in named:
            p.grad = bad(p.shape) if callable(bad) else bad(p.shape)
        opt.step()
        for _, p in named:
            assert torch.isfinite(p.detach()).all()


def test_warm_duals_persist_and_stay_finite():
    named = make_params()
    opt = make_opt(named)
    cd = next(iter(opt._coupled.values()))
    set_grads(named, seed=5)
    opt.step()
    lam1 = cd["Lam"].clone()
    assert float(lam1.abs().sum()) > 0                  # duals engaged
    for i in range(10):
        set_grads(named, seed=6 + i)
        opt.step()
    assert torch.isfinite(cd["Lam"]).all()
    # symmetric by construction
    assert torch.allclose(cd["Lam"], cd["Lam"].mT, atol=1e-6)


def test_metrics_exported():
    named = make_params()
    opt = make_opt(named, track_every=1)
    set_grads(named)
    opt.step()
    m = opt.get_metrics()
    assert "coupled/k_leak_ratio_mean" in m
    assert "halfsplit/n_nonfinite" in m
    assert m["halfsplit/n_nonfinite"] == 0.0


def test_bf16_updates_land():
    named = make_params(dtype=torch.bfloat16)
    opt = make_opt(named, lr=1e-3)
    before = [p.detach().float().clone() for _, p in named]
    torch.manual_seed(11)
    for i in range(20):
        set_grads(named, seed=20 + i)
        opt.step()
    moved = sum(float((p.detach().float() - b).abs().sum())
                for (_, p), b in zip(named, before))
    assert moved > 0, "updates lost to bf16 rounding — SR/master path broken"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
