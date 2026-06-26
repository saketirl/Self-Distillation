#!/usr/bin/env python3
"""Test UBV initialization for rectangular matrices."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from frozen_residual_ubv.projected_gradient_optimizer import ProjectedGradientOptimizer


def test_rectangular_matrices():
    """Test initialization for various rectangular matrix shapes."""
    torch.manual_seed(42)

    # Common transformer shapes
    shapes = [
        (4096, 4096),   # Square (attention)
        (11008, 4096),  # Tall (MLP up_proj/gate_proj)
        (4096, 11008),  # Wide (MLP down_proj)
        (896, 896),     # Small square (0.5B model)
        (2048, 896),    # Tall
        (896, 2048),    # Wide
    ]

    print("=" * 70)
    print("Testing rectangular matrix initialization")
    print("=" * 70)

    for m, n in shapes:
        print(f"\n--- Shape: ({m}, {n}) ---")

        # Create random weight matrix (simulating pretrained weights)
        W = nn.Parameter(torch.randn(m, n) * 0.02)  # Typical init scale

        # Create optimizer
        optimizer = ProjectedGradientOptimizer(
            [{'params': [W], 'lr': 0.01}],
            lr=0.01,
            energy_threshold=0.9,
            resync_every=1000,
            ns_steps=20,
        )

        # Trigger initialization by doing a forward/backward pass
        loss = (W ** 2).sum()
        loss.backward()
        optimizer.step()

        # Get state
        state = optimizer.state[W]
        U = state['U']
        B = state['B']
        V = state['V']
        R = state['R']
        r = state['r']

        print(f"  W shape: {W.shape}")
        print(f"  U shape: {U.shape}  (should be ({m}, {r}))")
        print(f"  B shape: {B.shape}  (should be ({r}, {r}))")
        print(f"  V shape: {V.shape}  (should be ({n}, {r}))")
        print(f"  R shape: {R.shape}  (should be ({m}, {n}))")
        print(f"  Rank r: {r} (from energy threshold 0.9)")

        # Check orthonormality
        I_r = torch.eye(r, device=U.device, dtype=U.dtype)
        U_orth_err = torch.linalg.norm(U.T @ U - I_r).item()
        V_orth_err = torch.linalg.norm(V.T @ V - I_r).item()

        print(f"  ||U.T @ U - I||: {U_orth_err:.6f}")
        print(f"  ||V.T @ V - I||: {V_orth_err:.6f}")

        # Check reconstruction
        W_reconstructed = U @ B @ V.T + R
        recon_err = torch.linalg.norm(W - W_reconstructed).item()
        print(f"  ||W - (U @ B @ V.T + R)||: {recon_err:.6f}")

        # Check B properties
        B_eigs = torch.linalg.eigvalsh(B)
        print(f"  B eigenvalues: min={B_eigs.min().item():.4f}, max={B_eigs.max().item():.4f}")

        # Verify shapes are correct
        assert U.shape == (m, r), f"U shape mismatch: {U.shape} vs ({m}, {r})"
        assert V.shape == (n, r), f"V shape mismatch: {V.shape} vs ({n}, {r})"
        assert B.shape == (r, r), f"B shape mismatch: {B.shape} vs ({r}, {r})"
        assert R.shape == (m, n), f"R shape mismatch: {R.shape} vs ({m}, {n})"

        # Verify orthonormality
        assert U_orth_err < 0.01, f"U not orthonormal: error = {U_orth_err}"
        assert V_orth_err < 0.01, f"V not orthonormal: error = {V_orth_err}"

        # Verify reconstruction
        assert recon_err < 1e-4, f"Reconstruction error too large: {recon_err}"

        print("  ✓ All checks passed")

        # Clean up
        del optimizer, W


def test_svd_rectangular():
    """Verify SVD gives orthonormal U and V for rectangular matrices."""
    print("\n" + "=" * 70)
    print("Verifying SVD orthonormality for rectangular matrices")
    print("=" * 70)

    shapes = [(1000, 500), (500, 1000), (1000, 1000)]

    for m, n in shapes:
        W = torch.randn(m, n)
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        V = Vt.T

        # U has shape (m, min(m,n)), V has shape (n, min(m,n))
        k = min(m, n)

        I_k = torch.eye(k)
        U_err = torch.linalg.norm(U.T @ U - I_k).item()
        V_err = torch.linalg.norm(V.T @ V - I_k).item()

        print(f"\nShape ({m}, {n}):")
        print(f"  U: {U.shape}, V: {V.shape}")
        print(f"  ||U.T @ U - I||: {U_err:.8f}")
        print(f"  ||V.T @ V - I||: {V_err:.8f}")

        # For SVD, these should be small (scales with sqrt(k) due to accumulation)
        # For k=500, expect ~sqrt(500) * machine_eps ≈ 1e-5 to 1e-4
        assert U_err < 1e-3, f"SVD U not orthonormal: {U_err}"
        assert V_err < 1e-3, f"SVD V not orthonormal: {V_err}"


if __name__ == "__main__":
    test_svd_rectangular()
    test_rectangular_matrices()
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
