#!/usr/bin/env python3
"""Debug test for ProjectedGradientOptimizer - trace gradient flow."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from frozen_residual_ubv.projected_gradient_optimizer import ProjectedGradientOptimizer
from frozen_residual_ubv.utils import symmetric_part


def test_gradient_flow():
    """Test that gradients flow correctly through the optimizer."""
    torch.manual_seed(42)
    d = 128
    device = torch.device("cpu")

    # Simple model: f(x) = W @ x
    W = nn.Parameter(torch.eye(d, device=device))

    # Create optimizer
    optimizer = ProjectedGradientOptimizer(
        [{'params': [W], 'lr': 0.1}],
        lr=0.1,
        energy_threshold=0.9,
        resync_every=1000,  # Don't resync during test
        ns_steps=10,
        lambda_min=1e-4,
        dual_alpha=0.1,
        dual_max_iters=5,
        grad_clip=100.0,
    )

    # Generate a simple target
    x = torch.randn(32, d, device=device)
    W_target = torch.eye(d, device=device) * 2  # Target is 2*I
    y_target = x @ W_target.T

    print("Initial state:")
    print(f"  ||W|| = {torch.norm(W).item():.4f}")
    print(f"  W[0,0] = {W[0,0].item():.4f}")

    # Force initialization
    optimizer.zero_grad()
    y = x @ W.T
    loss = F.mse_loss(y, y_target)
    loss.backward()

    print(f"\nGradient info:")
    print(f"  ||grad|| = {torch.norm(W.grad).item():.4f}")
    print(f"  grad[0,0] = {W.grad[0,0].item():.4f}")

    # Do one step
    optimizer.step()

    # Check optimizer state
    state = optimizer.state[W]
    U = state['U']
    B = state['B']
    V = state['V']
    R = state['R']

    print(f"\nAfter init (step 1):")
    print(f"  U shape: {U.shape}")
    print(f"  B shape: {B.shape}")
    print(f"  V shape: {V.shape}")
    print(f"  R shape: {R.shape}")
    print(f"  ||U||_F = {torch.norm(U, 'fro').item():.4f}")
    print(f"  ||B||_F = {torch.norm(B, 'fro').item():.4f}")
    print(f"  ||V||_F = {torch.norm(V, 'fro').item():.4f}")
    print(f"  ||R||_F = {torch.norm(R, 'fro').item():.4f}")
    print(f"  ||W|| = {torch.norm(W).item():.4f}")
    print(f"  W[0,0] = {W[0,0].item():.4f}")

    # Check B eigenvalues
    B_eigs = torch.linalg.eigvalsh(symmetric_part(B))
    print(f"  B eigenvalues: min={B_eigs.min().item():.4f}, max={B_eigs.max().item():.4f}")

    # Check orthonormality
    U_orth_err = torch.norm(U.T @ U - torch.eye(U.shape[1], device=device)).item()
    V_orth_err = torch.norm(V.T @ V - torch.eye(V.shape[1], device=device)).item()
    print(f"  U orthonormality error: {U_orth_err:.6f}")
    print(f"  V orthonormality error: {V_orth_err:.6f}")

    # Verify reconstruction
    W_reconstructed = U @ B @ V.T + R
    recon_error = torch.norm(W - W_reconstructed).item()
    print(f"  ||W - (U @ B @ V.T + R)|| = {recon_error:.6f}")

    # Now do multiple steps and track changes
    print("\n" + "="*60)
    print("Training loop (10 steps):")
    print("="*60)

    W_norms = [torch.norm(W).item()]
    B_norms = [torch.norm(B, 'fro').item()]
    losses = []

    for step in range(10):
        optimizer.zero_grad()
        y = x @ W.T
        loss = F.mse_loss(y, y_target)
        loss.backward()

        # Store gradient info before step
        grad_norm = torch.norm(W.grad).item()

        # Get state before step
        U_before = state['U'].clone()
        B_before = state['B'].clone()
        V_before = state['V'].clone()
        W_before = W.data.clone()

        optimizer.step()

        # Compute changes
        U_change = torch.norm(state['U'] - U_before).item()
        B_change = torch.norm(state['B'] - B_before).item()
        V_change = torch.norm(state['V'] - V_before).item()
        W_change = torch.norm(W.data - W_before).item()

        losses.append(loss.item())
        W_norms.append(torch.norm(W).item())
        B_norms.append(torch.norm(state['B'], 'fro').item())

        print(f"Step {step+1}: loss={loss.item():.4f}, ||grad||={grad_norm:.4f}, "
              f"ΔU={U_change:.6f}, ΔB={B_change:.6f}, ΔV={V_change:.6f}, ΔW={W_change:.6f}")

    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    print(f"  ||W||: {W_norms[0]:.4f} → {W_norms[-1]:.4f}")
    print(f"  ||B||: {B_norms[0]:.4f} → {B_norms[-1]:.4f}")

    # Check if W actually changed
    total_W_change = abs(W_norms[-1] - W_norms[0])
    print(f"  Total ||W|| change: {total_W_change:.6f}")

    if total_W_change < 0.001:
        print("\n⚠️  WARNING: W barely changed! There might be a bug.")
    else:
        print("\n✓ W is being updated correctly.")

    return losses, W_norms


def test_simple_gradient():
    """Test that a simple gradient descent works without manifold constraints."""
    print("\n" + "="*60)
    print("TEST: Simple gradient descent (no manifold)")
    print("="*60)

    torch.manual_seed(42)
    d = 128

    # Simple W update: W = W - lr * grad
    W = torch.eye(d)
    W_target = torch.eye(d) * 2

    lr = 0.1
    for step in range(10):
        x = torch.randn(32, d)
        y = x @ W.T
        y_target = x @ W_target.T

        # Compute gradient manually
        # L = ||y - y_target||^2 = ||x @ W.T - x @ W_target.T||^2
        # dL/dW = 2 * (W - W_target) * (something)
        # Actually: dL/dW = (1/n) * (y - y_target).T @ x * 2
        diff = y - y_target
        grad = (2 / x.shape[0]) * diff.T @ x  # (d, d)

        W = W - lr * grad

        loss = F.mse_loss(y, y_target).item()
        print(f"Step {step+1}: loss={loss:.4f}, ||W||={torch.norm(W).item():.4f}, ||grad||={torch.norm(grad).item():.4f}")

    print(f"\nFinal W[0,0] = {W[0,0].item():.4f} (target: 2.0)")


if __name__ == "__main__":
    test_simple_gradient()
    print("\n")
    test_gradient_flow()
