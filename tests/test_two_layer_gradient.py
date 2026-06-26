#!/usr/bin/env python3
"""Test gradient flow in two-layer network: f(x) = U @ W @ x"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from frozen_residual_ubv.projected_gradient_optimizer import ProjectedGradientOptimizer


class TwoLayerNet(nn.Module):
    def __init__(self, d: int = 128):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)
        self.U = nn.Linear(d, 1, bias=False)
        nn.init.eye_(self.W.weight)
        nn.init.normal_(self.U.weight, std=0.01)

    def forward(self, x):
        h = self.W(x)  # x @ W.T
        return self.U(h).squeeze(-1)  # h @ U.T


def test_gradient_flow():
    torch.manual_seed(42)
    d = 128
    n = 32

    model = TwoLayerNet(d)

    # Generate data
    x = torch.randn(n, d)
    y = torch.randn(n)  # Random targets

    # Forward pass
    out = model(x)
    loss = F.mse_loss(out, y)

    print("Forward pass:")
    print(f"  x shape: {x.shape}")
    print(f"  out shape: {out.shape}")
    print(f"  loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    print("\nGradients:")
    print(f"  ||grad_W|| = {torch.norm(model.W.weight.grad).item():.4f}")
    print(f"  ||grad_U|| = {torch.norm(model.U.weight.grad).item():.4f}")
    print(f"  grad_W[0,0] = {model.W.weight.grad[0,0].item():.6f}")
    print(f"  grad_U[0,0] = {model.U.weight.grad[0,0].item():.6f}")

    # Check gradient magnitude relative to parameter magnitude
    print("\nRelative gradient magnitudes:")
    print(f"  ||grad_W|| / ||W|| = {torch.norm(model.W.weight.grad).item() / torch.norm(model.W.weight).item():.6f}")
    print(f"  ||grad_U|| / ||U|| = {torch.norm(model.U.weight.grad).item() / torch.norm(model.U.weight).item():.6f}")


def test_optimizer_update():
    """Test that the optimizer actually updates W."""
    torch.manual_seed(42)
    d = 128
    n = 100

    model = TwoLayerNet(d)

    # Create optimizers
    optimizer_W = ProjectedGradientOptimizer(
        [{'params': [model.W.weight], 'lr': 0.1}],
        lr=0.1,
        energy_threshold=0.9,
        resync_every=1000,
    )
    optimizer_U = torch.optim.Adam([model.U.weight], lr=0.01)

    # Generate data
    x = torch.randn(n, d)
    # Target: y = sign(x[:, 0]) to make it learnable
    y = (x[:, 0] > 0).float()

    print("\nTraining loop:")
    W_norms = []

    for epoch in range(20):
        optimizer_W.zero_grad()
        optimizer_U.zero_grad()

        out = model(x)
        loss = F.mse_loss(out, y)
        loss.backward()

        # Record gradient magnitudes
        grad_W_norm = torch.norm(model.W.weight.grad).item()
        grad_U_norm = torch.norm(model.U.weight.grad).item()

        W_before = model.W.weight.data.clone()
        optimizer_W.step()
        optimizer_U.step()
        W_after = model.W.weight.data

        W_change = torch.norm(W_after - W_before).item()
        W_norms.append(torch.norm(W_after).item())

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}, "
                  f"||grad_W||={grad_W_norm:.4f}, ||grad_U||={grad_U_norm:.4f}, "
                  f"ΔW={W_change:.4f}, ||W||={W_norms[-1]:.4f}")

    total_W_change = abs(W_norms[-1] - W_norms[0])
    print(f"\nTotal ||W|| change: {total_W_change:.4f}")

    if total_W_change < 0.01:
        print("⚠️  WARNING: W is not being updated significantly!")
    else:
        print("✓ W is being updated.")


def test_single_layer_baseline():
    """Baseline: train W directly without U layer."""
    torch.manual_seed(42)
    d = 128
    n = 100

    W = nn.Parameter(torch.eye(d))
    U = nn.Parameter(torch.randn(1, d) * 0.01)

    optimizer_W = ProjectedGradientOptimizer(
        [{'params': [W], 'lr': 0.1}],
        lr=0.1,
        energy_threshold=0.9,
        resync_every=1000,
    )
    optimizer_U = torch.optim.Adam([U], lr=0.01)

    # Generate data
    x = torch.randn(n, d)
    y = (x[:, 0] > 0).float()

    print("\nBaseline (direct W access):")

    for epoch in range(20):
        optimizer_W.zero_grad()
        optimizer_U.zero_grad()

        # f(x) = U @ W @ x
        h = x @ W.T  # (n, d)
        out = (h @ U.T).squeeze(-1)  # (n,)

        loss = F.mse_loss(out, y)
        loss.backward()

        grad_W_norm = torch.norm(W.grad).item()
        W_before = W.data.clone()

        optimizer_W.step()
        optimizer_U.step()

        W_change = torch.norm(W.data - W_before).item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}, "
                  f"||grad_W||={grad_W_norm:.4f}, ΔW={W_change:.4f}, "
                  f"||W||={torch.norm(W).item():.4f}")


if __name__ == "__main__":
    print("="*60)
    print("TEST: Gradient flow in two-layer network")
    print("="*60)
    test_gradient_flow()

    print("\n" + "="*60)
    print("TEST: Optimizer update in two-layer network")
    print("="*60)
    test_optimizer_update()

    print("\n" + "="*60)
    print("TEST: Baseline with direct parameter access")
    print("="*60)
    test_single_layer_baseline()
