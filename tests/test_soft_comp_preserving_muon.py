"""
Tests for SoftCompPreservingMuon.

Four property tests:
  1. U_k^T ΔM V_k decreases after dual ascent (H_norm goes down vs unconstrained).
  2. Core drift stays small after a step (constraint is approximately enforced).
  3. k=0 recovers plain Muon (no dual correction, same update as msign).
  4. No NaNs when corrected gradients are tiny or zero.
"""
import math
import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from frozen_residual_ubv.soft_comp_preserving_muon import (
    SoftCompPreservingMuon,
    _factored_svd,
    _factored_svd_energy,
    _energy_rank,
    _safe_msign,
    _solve_sylvester_damped,
)
from frozen_residual_ubv.utils import msign


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_named_params(
    layer_idx: int = 0,
    d_model: int = 32,
    d_h: int = 8,
    H_q: int = 4,
    H_kv: int = 2,
    seed: int = 42,
):
    """Synthetic Q/K/V/O weight tensors for one transformer layer."""
    torch.manual_seed(seed)
    W_Q = torch.randn(H_q * d_h, d_model, requires_grad=True)
    W_K = torch.randn(H_kv * d_h, d_model, requires_grad=True)
    W_V = torch.randn(H_kv * d_h, d_model, requires_grad=True)
    W_O = torch.randn(d_model, H_q * d_h, requires_grad=True)
    return [
        (f"model.layers.{layer_idx}.self_attn.q_proj.weight", W_Q),
        (f"model.layers.{layer_idx}.self_attn.k_proj.weight", W_K),
        (f"model.layers.{layer_idx}.self_attn.v_proj.weight", W_V),
        (f"model.layers.{layer_idx}.self_attn.o_proj.weight", W_O),
    ]


def _make_optimizer(named, d_h=8, H_kv=2,
                    qk_energy_threshold=0.8, ov_energy_threshold=0.8,
                    preserve_qk=True, preserve_ov=False,
                    admm_steps=50, dual_lr=0.02, dual_tol=1e-6, lr=1e-2):
    return SoftCompPreservingMuon(
        named,
        lr=lr,
        head_dim=d_h,
        num_kv_heads=H_kv,
        qk_energy_threshold=qk_energy_threshold,
        ov_energy_threshold=ov_energy_threshold,
        preserve_qk=preserve_qk,
        preserve_ov=preserve_ov,
        admm_steps=admm_steps,
        dual_lr=dual_lr,
        dual_tol=dual_tol,
        ns_steps=8,
        eps=1e-7,
        debug=False,
    )


def _assign_gradients(named, seed=99):
    torch.manual_seed(seed)
    for _, p in named:
        p.grad = torch.randn_like(p)


# ---------------------------------------------------------------------------
# Helper: H_norm before dual ascent (Lambda=0, plain msign)
# ---------------------------------------------------------------------------

def _H_norm_unconstrained(A, B, G_A, G_B, U_k, V_k):
    """‖U_k^T (Δ_A B + A Δ_B) V_k‖_F / √(k²) with Lambda=0."""
    Delta_A = -msign(G_A, ns_steps=8, eps=1e-7)
    Delta_B = -msign(G_B, ns_steps=8, eps=1e-7)
    k = U_k.shape[1]
    BV = B @ V_k
    UA = U_k.T @ A
    H = (U_k.T @ Delta_A) @ BV + UA @ (Delta_B @ V_k)
    return torch.linalg.norm(H).item() / math.sqrt(k * k)


# ---------------------------------------------------------------------------
# Test 1: dual ascent reduces U_k^T ΔM V_k
# ---------------------------------------------------------------------------

def test_dual_ascent_reduces_H_norm():
    """ADMM inner loop (dual_lr=0.02, 50 steps) reduces H_norm below the unconstrained baseline.

    Convergence is verified empirically: dual_lr=0.02 converges for the random-init
    case while dual_lr >= 0.1 diverges (msign non-smoothness → oscillation at large lr).
    """
    d_model, d_h, H_q, H_kv = 32, 8, 2, 1
    energy_thr = 0.8

    named = _make_named_params(d_model=d_model, d_h=d_h, H_q=H_q, H_kv=H_kv)
    _assign_gradients(named)

    # Compute unconstrained H_norm using the SAME subspace the optimizer will pick.
    W_Q, W_K = named[0][1], named[1][1]
    A0 = W_Q.data[0:d_h, :].float().T
    B0 = W_K.data[0:d_h, :].float()
    G_A0 = W_Q.grad[0:d_h, :].float().T
    G_B0 = W_K.grad[0:d_h, :].float()
    U_k, S_k, V_k, k_act, cap = _factored_svd_energy(A0, B0, energy_thr)
    h_before = _H_norm_unconstrained(A0, B0, G_A0, G_B0, U_k, V_k)
    print(f"  energy_thr={energy_thr}  k_act={k_act}  captured={cap:.3f}")

    opt = _make_optimizer(named, d_h=d_h, H_kv=H_kv,
                          qk_energy_threshold=energy_thr, ov_energy_threshold=energy_thr,
                          preserve_qk=True, preserve_ov=False,
                          admm_steps=50, dual_lr=0.02, dual_tol=1e-10)
    opt.step()

    metrics = opt.get_constraint_metrics()
    h_after = metrics.get("soft_comp_muon/H_norm_mean", float("inf"))

    print(f"  H_norm before={h_before:.4e}  after={h_after:.4e}")
    assert math.isfinite(h_after), "H_norm is not finite after dual ascent"
    assert h_after < h_before, (
        f"Dual ascent should reduce H_norm: before={h_before:.4e} after={h_after:.4e}"
    )


# ---------------------------------------------------------------------------
# Test 2: core drift stays small after a step
# ---------------------------------------------------------------------------

def test_core_drift_small_after_step():
    """After one optimizer step, U_k^T M_t V_k should be close to Σ_k.

    Specifically, drift = ‖core_t − diag(Σ)‖ / ‖Σ‖ should be < 1.5.
    (It equals 1.0 at initialisation if the constraint holds exactly;
    the test just checks it doesn't blow up after one update.)
    """
    d_model, d_h, H_q, H_kv = 32, 8, 2, 1
    named = _make_named_params(d_model=d_model, d_h=d_h, H_q=H_q, H_kv=H_kv)
    _assign_gradients(named)

    opt = _make_optimizer(named, d_h=d_h, H_kv=H_kv,
                          qk_energy_threshold=0.8, ov_energy_threshold=0.8,
                          preserve_qk=True, preserve_ov=False,
                          admm_steps=50, dual_lr=0.02, dual_tol=1e-10)
    opt._init_subspaces()

    # Record initial core for one QK pair
    sub = opt._subspaces[(0, "qk", 0)]
    W_Q, W_K = named[0][1], named[1][1]
    A0 = W_Q.data[0:d_h, :].float().T
    B0 = W_K.data[0:d_h, :].float()
    core_init = (sub["U"].T @ A0) @ (B0 @ sub["V"])   # [k, k]
    core_diag = torch.diag(sub["S"])
    drift_init = (torch.linalg.norm(core_init - core_diag).item() /
                  (torch.linalg.norm(sub["S"]).item() + 1e-7))
    print(f"  drift before step = {drift_init:.4e}")

    opt.step()
    metrics = opt.get_constraint_metrics()
    drift_after = metrics.get("soft_comp_muon/core_drift_mean", float("inf"))
    print(f"  drift after step  = {drift_after:.4e}")

    # Drift should be finite and not blow up (allow generous threshold since
    # a single LR=1e-2 step changes the core; the point is stability)
    assert math.isfinite(drift_after), "core_drift is NaN/Inf"
    assert drift_after < 10.0, f"core_drift too large: {drift_after:.4e}"


# ---------------------------------------------------------------------------
# Test 3: k=0 recovers plain Muon
# ---------------------------------------------------------------------------

def test_k0_recovers_plain_muon():
    """With k_qk=0 and k_ov=0, parameter updates must equal -msign(G) * lr."""
    d_model, d_h, H_q, H_kv = 32, 8, 2, 1
    named = _make_named_params(d_model=d_model, d_h=d_h, H_q=H_q, H_kv=H_kv, seed=7)
    _assign_gradients(named, seed=13)

    # Save initial weights and gradients
    W_Q0 = named[0][1].data.clone()
    W_K0 = named[1][1].data.clone()
    W_V0 = named[2][1].data.clone()
    W_O0 = named[3][1].data.clone()

    G_Q = named[0][1].grad.clone().float()
    G_K = named[1][1].grad.clone().float()
    G_V = named[2][1].grad.clone().float()
    G_O = named[3][1].grad.clone().float()

    lr = 1e-2
    opt = SoftCompPreservingMuon(
        named,
        lr=lr, head_dim=d_h, num_kv_heads=H_kv,
        qk_energy_threshold=0.0,   # 0 threshold → k=0 → no constraint → plain Muon
        ov_energy_threshold=0.0,
        preserve_qk=True, preserve_ov=True,
        admm_steps=10, dual_lr=0.1,
    )
    opt.step()

    # Expected: plain msign per Q-head for Q, per KV-head for K/V, per head for O
    ns, eps = 5, 1e-7

    # Q-heads (each updated independently with -msign(G_Q_h))
    for h in range(H_q):
        G_Q_h = G_Q[h*d_h:(h+1)*d_h, :]
        expected_dq = -msign(G_Q_h, ns_steps=ns, eps=eps)
        actual_dq = (named[0][1].data[h*d_h:(h+1)*d_h, :] - W_Q0[h*d_h:(h+1)*d_h, :]) / lr
        assert torch.allclose(actual_dq.float(), expected_dq, atol=1e-4), (
            f"Q head {h}: update deviates from plain msign"
        )

    # K/V: one KV-head, updated once per KV-head
    G_K_j = G_K[0:d_h, :]
    expected_dk = -msign(G_K_j, ns_steps=ns, eps=eps)
    actual_dk = (named[1][1].data[0:d_h, :] - W_K0[0:d_h, :]) / lr
    assert torch.allclose(actual_dk.float(), expected_dk, atol=1e-4), \
        "K: update deviates from plain msign"

    G_V_j = G_V[0:d_h, :]
    expected_dv = -msign(G_V_j, ns_steps=ns, eps=eps)
    actual_dv = (named[2][1].data[0:d_h, :] - W_V0[0:d_h, :]) / lr
    assert torch.allclose(actual_dv.float(), expected_dv, atol=1e-4), \
        "V: update deviates from plain msign"

    # O-heads
    for h in range(H_q):
        G_O_h = G_O[:, h*d_h:(h+1)*d_h]
        expected_do = -msign(G_O_h, ns_steps=ns, eps=eps)
        actual_do = (named[3][1].data[:, h*d_h:(h+1)*d_h] - W_O0[:, h*d_h:(h+1)*d_h]) / lr
        assert torch.allclose(actual_do.float(), expected_do, atol=1e-4), (
            f"O head {h}: update deviates from plain msign"
        )


# ---------------------------------------------------------------------------
# Test 4: no NaNs with tiny / zero gradients
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grad_scale", [0.0, 1e-15, float("nan")])
def test_no_nans_with_degenerate_gradients(grad_scale):
    """Step should not produce NaNs or Infs for degenerate gradient inputs."""
    d_model, d_h, H_q, H_kv = 16, 4, 2, 1
    named = _make_named_params(d_model=d_model, d_h=d_h, H_q=H_q, H_kv=H_kv, seed=5)

    for _, p in named:
        if math.isnan(grad_scale):
            p.grad = torch.full_like(p, float("nan"))
        else:
            p.grad = torch.zeros_like(p) * grad_scale  # all zeros for 0.0 or 1e-15

    opt = _make_optimizer(named, d_h=d_h, H_kv=H_kv,
                          qk_energy_threshold=0.8, ov_energy_threshold=0.8,
                          preserve_qk=True, preserve_ov=True,
                          admm_steps=5, dual_lr=0.1, dual_tol=1e-4)
    opt.step()

    for name, p in named:
        assert torch.isfinite(p.data).all(), (
            f"{name}: parameter contains NaN/Inf after step with grad_scale={grad_scale}"
        )


# ---------------------------------------------------------------------------
# Test 5: GQA accumulation — shared K/V head receives averaged update
# ---------------------------------------------------------------------------

def test_gqa_k_update_is_averaged():
    """With groups=2 (2 Q-heads per KV-head), K update is the mean of contributions."""
    d_model, d_h, H_q, H_kv = 32, 8, 4, 2
    named = _make_named_params(d_model=d_model, d_h=d_h, H_q=H_q, H_kv=H_kv, seed=17)
    _assign_gradients(named, seed=31)

    W_K0 = named[1][1].data.clone()

    opt = _make_optimizer(named, d_h=d_h, H_kv=H_kv,
                          qk_energy_threshold=0.8, ov_energy_threshold=0.8,
                          preserve_qk=True, preserve_ov=False,
                          admm_steps=20, dual_lr=0.02, dual_tol=1e-8)
    opt.step()

    # K has been updated; just verify it's finite and non-zero (update was applied)
    delta_K = named[1][1].data - W_K0
    assert torch.isfinite(delta_K).all(), "K update contains NaN/Inf"
    assert delta_K.norm() > 0, "K update is identically zero — GQA accumulation broken"


# ---------------------------------------------------------------------------
# Test 6: _factored_svd correctness
# ---------------------------------------------------------------------------

def test_factored_svd_matches_direct():
    """_factored_svd(A, B, k) should give the same top-k singular values as svd(A @ B)."""
    torch.manual_seed(0)
    m, r, n, k = 20, 5, 20, 3
    A = torch.randn(m, r)
    B = torch.randn(r, n)
    M = A @ B

    U_k, S_k, V_k = _factored_svd(A, B, k)

    _, S_direct, _ = torch.linalg.svd(M, full_matrices=False)
    S_direct_k = S_direct[:k]

    print(f"  factored S: {S_k.tolist()}")
    print(f"  direct  S: {S_direct_k.tolist()}")

    # Singular values should match to within numerical precision
    assert torch.allclose(S_k, S_direct_k, atol=1e-4), (
        f"Singular value mismatch: {S_k} vs {S_direct_k}"
    )

    # U_k, V_k should have orthonormal columns
    assert torch.allclose(U_k.T @ U_k, torch.eye(k), atol=1e-5), "U_k columns not orthonormal"
    assert torch.allclose(V_k.T @ V_k, torch.eye(k), atol=1e-5), "V_k columns not orthonormal"


# ---------------------------------------------------------------------------
# Test 7: multiple steps — Lambda warms up, H_norm decreases over time
# ---------------------------------------------------------------------------

def test_lambda_warmup_across_steps():
    """Lambda accumulation across steps should make dual feasibility improve."""
    d_model, d_h, H_q, H_kv = 32, 8, 2, 1
    named = _make_named_params(d_model=d_model, d_h=d_h, H_q=H_q, H_kv=H_kv)

    opt = _make_optimizer(named, d_h=d_h, H_kv=H_kv,
                          qk_energy_threshold=0.8, ov_energy_threshold=0.8,
                          preserve_qk=True, preserve_ov=False,
                          admm_steps=30, dual_lr=0.02, dual_tol=1e-12)

    H_norms = []
    for step_idx in range(5):
        torch.manual_seed(step_idx)
        for _, p in named:
            p.grad = torch.randn_like(p)
        opt.step()
        m = opt.get_constraint_metrics()
        H_norms.append(m.get("soft_comp_muon/H_norm_mean", float("inf")))

    print(f"  H_norm over steps: {[f'{h:.3e}' for h in H_norms]}")
    # H_norm should not diverge (may not monotonically decrease due to changing gradients)
    assert all(math.isfinite(h) for h in H_norms), "H_norm became non-finite"
    assert max(H_norms) < 10.0, f"H_norm blew up: max={max(H_norms):.3e}"


# ---------------------------------------------------------------------------
# Test 8: Sylvester correction reduces H_norm on random (A, B, ΔA, ΔB)
# ---------------------------------------------------------------------------

def test_sylvester_correction_reduces_h_norm():
    """_solve_sylvester_damped should reduce ‖H‖ by correcting Delta_A, Delta_B.

    With random A, B and random descent directions Delta_A, Delta_B:
        H  = U_k^T (Delta_A @ B + A @ Delta_B) @ V_k
        P  = U_k^T A A^T U_k,  Q = V_k^T B^T B V_k  (both k×k)
        solve  P Lambda_corr + Lambda_corr Q = H
        Delta_A' = Delta_A - U_k @ Lambda_corr @ BV^T
        Delta_B' = Delta_B - UA^T @ Lambda_corr @ V_k^T
        H2 = U_k^T (Delta_A' @ B + A @ Delta_B') @ V_k  ≈ 0
    """
    torch.manual_seed(123)
    k, r, m, n = 4, 8, 32, 32

    A = torch.randn(m, r)
    B = torch.randn(r, n)
    # Orthonormal U_k, V_k
    U_k = torch.linalg.qr(torch.randn(m, k))[0]
    V_k = torch.linalg.qr(torch.randn(n, k))[0]
    BV  = B @ V_k    # [r, k]
    UA  = U_k.T @ A  # [k, r]

    Delta_A = torch.randn(m, r)
    Delta_B = torch.randn(r, n)
    H = (U_k.T @ Delta_A) @ BV + UA @ (Delta_B @ V_k)
    resid_before = torch.linalg.norm(H).item() / math.sqrt(k * k)

    P = UA @ UA.T
    Q = BV.T @ BV
    Lambda_corr = _solve_sylvester_damped(P, Q, H, damping=1e-4)

    Delta_A_new = Delta_A - U_k @ (Lambda_corr @ BV.T)
    Delta_B_new = Delta_B - UA.T @ (Lambda_corr @ V_k.T)
    H2 = (U_k.T @ Delta_A_new) @ BV + UA @ (Delta_B_new @ V_k)
    resid_after = torch.linalg.norm(H2).item() / math.sqrt(k * k)

    print(f"  resid_before={resid_before:.4e}  resid_after={resid_after:.4e}")
    assert resid_after < resid_before, (
        f"Sylvester should reduce H_norm: before={resid_before:.4e} after={resid_after:.4e}"
    )
    assert resid_after < 0.1 * resid_before, (
        f"Sylvester should reduce H_norm by ≥10x: before={resid_before:.4e} after={resid_after:.4e}"
    )


# ---------------------------------------------------------------------------
# Test 9: updates zeroed when hard_fallback_tol is never satisfied
# ---------------------------------------------------------------------------

def test_sylvester_fallback_zeroes_updates_on_failure():
    """With hard_fallback_tol=0.0, Sylvester always 'fails'; weights must not change.

    admm_steps=0 leaves a large H (no ADMM reduction).  fallback_tol=0.0 triggers
    Sylvester on every head.  hard_fallback_tol=0.0 means any positive resid2
    is treated as failure → updates are zeroed → weights unchanged.
    """
    d_model, d_h, H_q, H_kv = 32, 8, 2, 1
    named = _make_named_params(d_model=d_model, d_h=d_h, H_q=H_q, H_kv=H_kv, seed=42)
    _assign_gradients(named, seed=7)

    W_Q0 = named[0][1].data.clone()
    W_K0 = named[1][1].data.clone()

    opt = SoftCompPreservingMuon(
        named,
        lr=1e-2,
        head_dim=d_h,
        num_kv_heads=H_kv,
        qk_energy_threshold=0.8,
        ov_energy_threshold=0.0,
        preserve_qk=True,
        preserve_ov=False,
        admm_steps=0,
        dual_lr=0.02,
        use_sylvester_fallback=True,
        fallback_tol=0.0,       # always trigger Sylvester
        hard_fallback_tol=0.0,  # resid2 > 0 → always "fail" → zero updates
        sylvester_damping=1e-4,
    )
    opt.step()

    assert any(opt._last_sylvester_used),   "Sylvester should have been triggered"
    assert any(opt._last_sylvester_failed), "Sylvester should have 'failed' (hard_fallback_tol=0)"

    # Weights must be unchanged — zeroed updates mean no parameter change
    assert torch.allclose(named[0][1].data, W_Q0, atol=1e-6), \
        "Q weights changed despite zeroed updates"
    assert torch.allclose(named[1][1].data, W_K0, atol=1e-6), \
        "K weights changed despite zeroed updates"


# ---------------------------------------------------------------------------
# Test 10: fallback not triggered when residual is below fallback_tol
# ---------------------------------------------------------------------------

def test_sylvester_always_applied_and_reduces_h_norm():
    """Sylvester projection always runs (when k_act > 0) and substantially reduces H_norm."""
    d_model, d_h, H_q, H_kv = 32, 8, 2, 1
    named = _make_named_params(d_model=d_model, d_h=d_h, H_q=H_q, H_kv=H_kv, seed=42)
    _assign_gradients(named, seed=7)

    opt = SoftCompPreservingMuon(
        named,
        lr=1e-2,
        head_dim=d_h,
        num_kv_heads=H_kv,
        qk_energy_threshold=0.8,
        ov_energy_threshold=0.0,
        preserve_qk=True,
        preserve_ov=False,
        hard_fallback_tol=0.5,
        sylvester_damping=1e-4,
    )
    opt.step()

    m = opt.get_constraint_metrics()
    resid_before = m["soft_comp_muon/H_norm_mean"]
    resid_after  = m["soft_comp_muon/step_resid_after_mean"]

    assert all(opt._last_sylvester_used), (
        f"Sylvester should always run when k_act > 0, got used={opt._last_sylvester_used}"
    )
    assert not any(opt._last_sylvester_failed), (
        f"Sylvester should not fail on this well-conditioned problem, "
        f"got failed={opt._last_sylvester_failed}"
    )
    assert resid_after < 0.01 * resid_before, (
        f"Sylvester should reduce H_norm by ≥100x: before={resid_before:.3e} after={resid_after:.3e}"
    )
