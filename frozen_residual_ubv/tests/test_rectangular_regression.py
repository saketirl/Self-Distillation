"""
Test ProjectedGradientOptimizer on rectangular linear regression.

Tests that the optimizer:
1. Works correctly on non-square weight matrices (m != n)
2. Maintains Stiefel constraints (U.T @ U = I, V.T @ V = I)
3. Maintains SPD constraint on B (positive eigenvalues)
4. Decreases loss over training steps
5. Works with both "projected_qr" and "muon_admm" methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

import sys
sys.path.insert(0, '/Users/sakettiwari/Documents/continual/Self-Distillation')

from frozen_residual_ubv.projected_gradient_optimizer import (
    ProjectedGradientOptimizer,
    create_projected_optimizer,
)


def generate_regression_data(
    n_samples: int,
    input_dim: int,
    output_dim: int,
    noise_std: float = 0.1,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic linear regression data."""
    torch.manual_seed(seed)

    # True weight matrix
    W_true = torch.randn(output_dim, input_dim) * 0.5

    # Input data
    X = torch.randn(n_samples, input_dim)

    # Output with noise
    Y = X @ W_true.T + noise_std * torch.randn(n_samples, output_dim)

    return X, Y, W_true


def check_stiefel_constraint(U: torch.Tensor, tol: float = 1e-4) -> float:
    """Check orthonormality: ||U.T @ U - I||_F"""
    r = U.shape[1]
    I = torch.eye(r, device=U.device, dtype=U.dtype)
    return torch.linalg.norm(U.T @ U - I, 'fro').item()


def check_spd_constraint(B: torch.Tensor, tol: float = 1e-6) -> tuple[bool, float]:
    """Check B is symmetric positive definite. Returns (is_spd, min_eigenvalue)."""
    B_sym = (B + B.T) / 2
    eigs = torch.linalg.eigvalsh(B_sym)
    min_eig = eigs.min().item()
    return min_eig > -tol, min_eig


def test_rectangular_regression(
    input_dim: int = 64,
    output_dim: int = 32,
    n_samples: int = 256,
    n_steps: int = 50,
    lr: float = 1e-2,
    energy_threshold: float = 0.9,
    stiefel_update: str = "projected_qr",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Test optimizer on rectangular linear regression.

    Args:
        input_dim: Input dimension (columns of W)
        output_dim: Output dimension (rows of W)
        n_samples: Number of training samples
        n_steps: Number of optimization steps
        lr: Learning rate
        energy_threshold: Energy threshold for rank selection
        stiefel_update: "projected_qr" or "muon_admm"
        verbose: Print progress

    Returns:
        Dict with test results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing Rectangular Regression: {input_dim} -> {output_dim}")
        print(f"Stiefel update: {stiefel_update}")
        print(f"{'='*60}")

    # Generate data
    X, Y, W_true = generate_regression_data(n_samples, input_dim, output_dim)

    # Create model
    model = nn.Linear(input_dim, output_dim, bias=False)

    # Initialize with random weights (not the true weights)
    nn.init.xavier_normal_(model.weight)

    # Create optimizer
    optimizer = ProjectedGradientOptimizer(
        [{'params': model.parameters()}],
        lr=lr,
        energy_threshold=energy_threshold,
        resync_every=100,
        stiefel_update=stiefel_update,
        stiefel_max_rms=0.1,  # More permissive for faster convergence in test
        uv_lr_scale=1.0,
        b_lr_scale=1.0,
    )

    # Track metrics
    losses = []
    ortho_errors_U = []
    ortho_errors_V = []
    min_eigenvalues = []
    tangent_errors_before = []
    tangent_errors_after = []

    # Initial loss
    with torch.no_grad():
        Y_pred = model(X)
        initial_loss = F.mse_loss(Y_pred, Y).item()

    if verbose:
        print(f"Initial loss: {initial_loss:.6f}")

    # Training loop
    for step in range(n_steps):
        optimizer.zero_grad()

        # Forward pass
        Y_pred = model(X)
        loss = F.mse_loss(Y_pred, Y)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        losses.append(loss.item())

        # Check constraints
        state = optimizer.state[model.weight]
        U = state['U']
        V = state['V']
        B = state['B']

        ortho_U = check_stiefel_constraint(U)
        ortho_V = check_stiefel_constraint(V)
        is_spd, min_eig = check_spd_constraint(B)

        ortho_errors_U.append(ortho_U)
        ortho_errors_V.append(ortho_V)
        min_eigenvalues.append(min_eig)

        # Track tangent errors if available
        if 'tangent_err_U_before' in state:
            tangent_errors_before.append(state['tangent_err_U_before'])
        if 'tangent_err_U_after' in state:
            tangent_errors_after.append(state['tangent_err_U_after'])

        if verbose and (step + 1) % 10 == 0:
            print(f"Step {step+1:3d}: loss={loss.item():.6f}, "
                  f"ortho_U={ortho_U:.2e}, ortho_V={ortho_V:.2e}, "
                  f"min_eig={min_eig:.4f}")

    # Final metrics
    final_loss = losses[-1]
    max_ortho_U = max(ortho_errors_U)
    max_ortho_V = max(ortho_errors_V)
    min_min_eig = min(min_eigenvalues)

    # Get detailed metrics from optimizer
    detailed_metrics = optimizer.get_detailed_metrics()

    if verbose:
        print(f"\n{'='*60}")
        print("Results:")
        print(f"  Initial loss: {initial_loss:.6f}")
        print(f"  Final loss:   {final_loss:.6f}")
        print(f"  Loss reduction: {(1 - final_loss/initial_loss)*100:.1f}%")
        print(f"  Max ortho error U: {max_ortho_U:.2e}")
        print(f"  Max ortho error V: {max_ortho_V:.2e}")
        print(f"  Min eigenvalue B:  {min_min_eig:.6f}")
        if tangent_errors_before:
            print(f"  Max tangent err before: {max(tangent_errors_before):.2e}")
        if tangent_errors_after:
            print(f"  Max tangent err after:  {max(tangent_errors_after):.2e}")
        print(f"  Rank: {state['r']}")
        print(f"{'='*60}")

    # Assertions
    results = {
        'passed': True,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'loss_reduced': final_loss < initial_loss,
        'max_ortho_U': max_ortho_U,
        'max_ortho_V': max_ortho_V,
        'ortho_maintained': max_ortho_U < 1e-3 and max_ortho_V < 1e-3,
        'min_eigenvalue': min_min_eig,
        'spd_maintained': min_min_eig > 0,
        'rank': state['r'],
        'detailed_metrics': detailed_metrics,
    }

    # Check test conditions
    if not results['loss_reduced']:
        results['passed'] = False
        if verbose:
            print("FAILED: Loss did not decrease!")

    if not results['ortho_maintained']:
        results['passed'] = False
        if verbose:
            print("FAILED: Orthonormality constraint violated!")

    if not results['spd_maintained']:
        results['passed'] = False
        if verbose:
            print("FAILED: SPD constraint violated (negative eigenvalue)!")

    if results['passed'] and verbose:
        print("PASSED: All constraints maintained and loss decreased!")

    return results


def test_various_shapes():
    """Test optimizer on various rectangular shapes."""
    shapes = [
        (64, 32),   # Wide -> narrow (more rows than cols in W)
        (32, 64),   # Narrow -> wide (more cols than rows in W)
        (128, 64),  # Larger
        (64, 128),  # Larger, opposite
        (256, 32),  # Very tall
        (32, 256),  # Very wide
    ]

    results = {}
    all_passed = True

    for input_dim, output_dim in shapes:
        key = f"{input_dim}x{output_dim}"
        print(f"\n\nTesting shape: input={input_dim}, output={output_dim}")
        print(f"Weight matrix shape: ({output_dim}, {input_dim})")

        result = test_rectangular_regression(
            input_dim=input_dim,
            output_dim=output_dim,
            n_steps=30,
            lr=1e-2,
            verbose=True,
        )
        results[key] = result
        if not result['passed']:
            all_passed = False

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for key, result in results.items():
        status = "PASSED" if result['passed'] else "FAILED"
        print(f"  {key}: {status} (loss: {result['initial_loss']:.4f} -> {result['final_loss']:.4f})")

    print("="*60)
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("="*60)

    return all_passed


def test_both_stiefel_methods():
    """Test both projected_qr and muon_admm methods."""
    print("\n" + "="*60)
    print("Testing both Stiefel update methods")
    print("="*60)

    methods = ["projected_qr", "muon_admm"]
    results = {}
    all_passed = True

    for method in methods:
        print(f"\n--- Testing {method} ---")
        result = test_rectangular_regression(
            input_dim=64,
            output_dim=32,
            n_steps=30,
            lr=1e-2,
            stiefel_update=method,
            verbose=True,
        )
        results[method] = result
        if not result['passed']:
            all_passed = False

    print("\n" + "="*60)
    print("METHOD COMPARISON")
    print("="*60)
    for method, result in results.items():
        status = "PASSED" if result['passed'] else "FAILED"
        print(f"  {method}: {status}")
        print(f"    Loss: {result['initial_loss']:.4f} -> {result['final_loss']:.4f}")
        print(f"    Max ortho error: U={result['max_ortho_U']:.2e}, V={result['max_ortho_V']:.2e}")
        print(f"    Min eigenvalue: {result['min_eigenvalue']:.6f}")

    return all_passed


if __name__ == "__main__":
    print("="*60)
    print("Rectangular Linear Regression Tests")
    print("="*60)

    # Test 1: Basic rectangular regression
    print("\n\n" + "="*60)
    print("TEST 1: Basic Rectangular Regression (64 -> 32)")
    print("="*60)
    result = test_rectangular_regression(
        input_dim=64,
        output_dim=32,
        n_steps=50,
        verbose=True,
    )

    # Test 2: Various shapes
    print("\n\n" + "="*60)
    print("TEST 2: Various Rectangular Shapes")
    print("="*60)
    shapes_passed = test_various_shapes()

    # Test 3: Both Stiefel methods
    print("\n\n" + "="*60)
    print("TEST 3: Both Stiefel Update Methods")
    print("="*60)
    methods_passed = test_both_stiefel_methods()

    # Final summary
    print("\n\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"  Basic test:      {'PASSED' if result['passed'] else 'FAILED'}")
    print(f"  Various shapes:  {'PASSED' if shapes_passed else 'FAILED'}")
    print(f"  Both methods:    {'PASSED' if methods_passed else 'FAILED'}")
    print("="*60)

    all_passed = result['passed'] and shapes_passed and methods_passed
    exit(0 if all_passed else 1)
