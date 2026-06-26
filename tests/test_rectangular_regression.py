#!/usr/bin/env python3
"""
Test ProjectedGradientOptimizer on linear regression with RECTANGULAR weight matrix.

Network: f(x) = U @ W @ x
- U ∈ R^{1×d'}: trained with Adam (output layer, 1×64)
- W ∈ R^{d'×d}: trained with ProjectedGradientOptimizer (rectangular, 64×128)
- x ∈ R^d: input vector (128)

This tests the optimizer on non-square matrices where the intermediate dimension
d' differs from the input dimension d. Common in transformer MLPs where:
- up_proj: (hidden, embed) = (11008, 4096) - tall
- down_proj: (embed, hidden) = (4096, 11008) - wide

Task: Binary classification with MSE loss
- Class 0: x ~ N(μ_0, I)
- Class 1: x ~ N(μ_1, I)
- y ∈ {0, 1}
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Tuple

from frozen_residual_ubv.projected_gradient_optimizer import ProjectedGradientOptimizer


class RectangularLinearClassifier(nn.Module):
    """
    Two-layer linear network with rectangular intermediate: f(x) = U @ W @ x

    Architecture:
        x ∈ R^d → W ∈ R^{d' × d} → h ∈ R^{d'} → U ∈ R^{1 × d'} → y ∈ R

    This is a dimensionality reduction followed by a linear classifier.
    W compresses d dimensions to d' < d dimensions.
    """

    def __init__(self, d: int = 128, d_prime: int = 64):
        super().__init__()
        self.d = d
        self.d_prime = d_prime

        # W: d' x d rectangular matrix (trained with projected gradient)
        # This compresses from d to d' dimensions
        self.W = nn.Linear(d, d_prime, bias=False)

        # U: 1 x d' matrix (trained with Adam)
        self.U = nn.Linear(d_prime, 1, bias=False)

        # Initialize W as random orthonormal rows (for stability)
        # For d' < d, we want W @ W^T ≈ I (rows are orthonormal)
        nn.init.orthogonal_(self.W.weight)

        # Small random init for U
        nn.init.normal_(self.U.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: f(x) = U @ W @ x

        Args:
            x: Input tensor of shape (batch, d)

        Returns:
            Output tensor of shape (batch,)
        """
        # x: (batch, d)
        # W @ x: (batch, d')
        # U @ (W @ x): (batch, 1)
        h = self.W(x)  # (batch, d')
        out = self.U(h)  # (batch, 1)
        return out.squeeze(-1)  # (batch,)


def generate_data(
    n_samples: int,
    d: int,
    mu_0: torch.Tensor,
    mu_1: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate binary classification data."""
    n_per_class = n_samples // 2

    # Class 0: x ~ N(μ_0, I)
    x_0 = torch.randn(n_per_class, d, device=device) + mu_0
    y_0 = torch.zeros(n_per_class, device=device)

    # Class 1: x ~ N(μ_1, I)
    x_1 = torch.randn(n_per_class, d, device=device) + mu_1
    y_1 = torch.ones(n_per_class, device=device)

    # Concatenate and shuffle
    x = torch.cat([x_0, x_1], dim=0)
    y = torch.cat([y_0, y_1], dim=0)

    perm = torch.randperm(n_samples, device=device)
    return x[perm], y[perm]


def compute_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute classification accuracy."""
    model.eval()
    with torch.no_grad():
        preds = model(x)
        pred_labels = (preds > 0.5).float()
        accuracy = (pred_labels == y).float().mean().item()
    model.train()
    return accuracy


def test_rectangular_optimizer(
    d: int = 128,
    d_prime: int = 64,
    n_train: int = 1000,
    n_test: int = 200,
    n_epochs: int = 100,
    batch_size: int = 64,
    lr_W: float = 0.01,
    lr_U: float = 0.001,
    energy_threshold: float = 0.9,
    resync_every: int = 50,
    seed: int = 42,
    verbose: bool = True,
):
    """
    Test ProjectedGradientOptimizer on rectangular matrix regression.

    Args:
        d: Input dimension (128)
        d_prime: Hidden dimension (64), d' < d makes W a "wide" matrix
        n_train: Number of training samples
        n_test: Number of test samples
        n_epochs: Training epochs
        batch_size: Batch size
        lr_W: Learning rate for W (rectangular matrix)
        lr_U: Learning rate for U (output layer)
        energy_threshold: SVD energy threshold for rank selection
        resync_every: Steps between SVD resyncs
        seed: Random seed
        verbose: Print progress

    Returns:
        dict with training history and final metrics
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"Device: {device}")
        print(f"Architecture: x ∈ R^{d} → W ∈ R^{{{d_prime}×{d}}} → h ∈ R^{d_prime} → U ∈ R^{{1×{d_prime}}} → y")
        print(f"W is a WIDE matrix: {d_prime} rows × {d} cols (d' < d)")
        print(f"n_train={n_train}, n_test={n_test}")
        print(f"lr_W={lr_W}, lr_U={lr_U}")
        print(f"energy_threshold={energy_threshold}, resync_every={resync_every}")

    # Create means for the two classes (make them separable)
    # μ_0 and μ_1 differ in a few dimensions
    mu_0 = torch.zeros(d, device=device)
    mu_1 = torch.zeros(d, device=device)
    # Make first 10 dimensions different
    mu_0[:10] = -1.0
    mu_1[:10] = 1.0

    if verbose:
        print(f"Class separation: ||μ_1 - μ_0|| = {torch.norm(mu_1 - mu_0).item():.2f}")

    # Generate data
    x_train, y_train = generate_data(n_train, d, mu_0, mu_1, device)
    x_test, y_test = generate_data(n_test, d, mu_0, mu_1, device)

    if verbose:
        print(f"Train data: x={x_train.shape}, y={y_train.shape}")
        print(f"Test data: x={x_test.shape}, y={y_test.shape}")

    # Create model with rectangular W
    model = RectangularLinearClassifier(d=d, d_prime=d_prime).to(device)

    if verbose:
        print(f"\nModel W weight shape: {model.W.weight.shape}")
        print(f"Model U weight shape: {model.U.weight.shape}")

    # Create optimizers
    # W: ProjectedGradientOptimizer (rectangular d' x d matrix)
    optimizer_W = ProjectedGradientOptimizer(
        [{'params': [model.W.weight], 'lr': lr_W}],
        lr=lr_W,
        energy_threshold=energy_threshold,
        resync_every=resync_every,
        ns_steps=20,  # More iterations for rectangular matrices
        lambda_min=1e-4,
        dual_alpha=0.1,
        dual_max_iters=5,
        grad_clip=10.0,
    )

    # U: Adam
    optimizer_U = torch.optim.Adam([model.U.weight], lr=lr_U)

    # Training history
    history = {
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': [],
        'W_norm': [],
        'U_norm': [],
        'B_eigenvalues': [],
        'residual_ratio': [],
        'U_orth_error': [],
        'V_orth_error': [],
        'rank': [],
    }

    # Initial evaluation
    initial_train_acc = compute_accuracy(model, x_train, y_train)
    initial_test_acc = compute_accuracy(model, x_test, y_test)
    if verbose:
        print(f"\nInitial train acc: {initial_train_acc:.2%}")
        print(f"Initial test acc: {initial_test_acc:.2%}")
        print("\nTraining...")

    # Training loop
    n_batches = (n_train + batch_size - 1) // batch_size

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        # Shuffle data each epoch
        perm = torch.randperm(n_train, device=device)
        x_train_shuffled = x_train[perm]
        y_train_shuffled = y_train[perm]

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_train)

            x_batch = x_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]

            # Forward pass
            preds = model(x_batch)
            loss = F.mse_loss(preds, y_batch)

            # Backward pass
            optimizer_W.zero_grad()
            optimizer_U.zero_grad()
            loss.backward()

            # Update
            optimizer_W.step()
            optimizer_U.step()

            epoch_loss += loss.item()

        epoch_loss /= n_batches

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_preds = model(x_test)
            test_loss = F.mse_loss(test_preds, y_test).item()

        train_acc = compute_accuracy(model, x_train, y_train)
        test_acc = compute_accuracy(model, x_test, y_test)

        # Get optimizer stats
        W_norm = torch.norm(model.W.weight).item()
        U_norm = torch.norm(model.U.weight).item()

        # Get UBV stats from optimizer
        stats = optimizer_W.get_ubv_stats()
        if stats['eigenvalues']:
            B_eigs = stats['eigenvalues'][0].cpu().numpy()
            residual_ratio = stats['residual_ratios'][0] if stats['residual_ratios'] else 0.0
        else:
            B_eigs = []
            residual_ratio = 0.0

        # Get detailed metrics for orthonormality
        detailed_metrics = optimizer_W.get_detailed_metrics()
        U_orth_error = detailed_metrics.get('proj/mean_orthonormality_U', 0.0)
        V_orth_error = detailed_metrics.get('proj/mean_orthonormality_V', 0.0)
        rank = detailed_metrics.get('proj/mean_rank', 0)

        # Record history
        history['train_loss'].append(epoch_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['W_norm'].append(W_norm)
        history['U_norm'].append(U_norm)
        history['B_eigenvalues'].append(B_eigs)
        history['residual_ratio'].append(residual_ratio)
        history['U_orth_error'].append(U_orth_error)
        history['V_orth_error'].append(V_orth_error)
        history['rank'].append(rank)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: loss={epoch_loss:.4f}, "
                  f"train_acc={train_acc:.2%}, test_acc={test_acc:.2%}, "
                  f"||W||={W_norm:.2f}, rank={rank:.0f}, "
                  f"U_orth={U_orth_error:.6f}, V_orth={V_orth_error:.6f}")

    # Final evaluation
    final_train_acc = compute_accuracy(model, x_train, y_train)
    final_test_acc = compute_accuracy(model, x_test, y_test)

    if verbose:
        print(f"\nFinal train accuracy: {final_train_acc:.2%}")
        print(f"Final test accuracy: {final_test_acc:.2%}")
        print(f"Improvement: {final_test_acc - initial_test_acc:+.2%}")

    # Get detailed metrics
    detailed_metrics = optimizer_W.get_detailed_metrics()
    if verbose and detailed_metrics:
        print("\nOptimizer metrics (final):")
        for k, v in detailed_metrics.items():
            print(f"  {k}: {v:.6f}")

    # Verify shapes
    state = optimizer_W.state[model.W.weight]
    if verbose:
        print(f"\nUBV decomposition shapes for W ∈ R^{{{d_prime}×{d}}}:")
        print(f"  U shape: {state['U'].shape} (expected: ({d_prime}, r))")
        print(f"  B shape: {state['B'].shape} (expected: (r, r))")
        print(f"  V shape: {state['V'].shape} (expected: ({d}, r))")
        print(f"  R shape: {state['R'].shape} (expected: ({d_prime}, {d}))")
        print(f"  rank r: {state['r']}")

    return {
        'history': history,
        'initial_train_acc': initial_train_acc,
        'initial_test_acc': initial_test_acc,
        'final_train_acc': final_train_acc,
        'final_test_acc': final_test_acc,
        'model': model,
        'optimizer_W': optimizer_W,
        'detailed_metrics': detailed_metrics,
    }


def test_orthonormality_preservation():
    """
    Verify that U and V maintain orthonormality throughout training for rectangular W.

    For W ∈ R^{d' × d} with d' < d (wide matrix):
    - U ∈ St(r, d') where r ≤ d' → U.T @ U = I_r
    - V ∈ St(r, d) where r ≤ d' → V.T @ V = I_r

    The rank r is bounded by min(d', d) = d'.
    """
    print("=" * 70)
    print("Testing orthonormality preservation for rectangular matrices")
    print("=" * 70)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = 128
    d_prime = 64

    # Create model
    model = RectangularLinearClassifier(d=d, d_prime=d_prime).to(device)

    print(f"\nW shape: {model.W.weight.shape} (d'={d_prime} × d={d}, wide matrix)")

    # Create optimizer
    optimizer = ProjectedGradientOptimizer(
        [{'params': [model.W.weight], 'lr': 0.01}],
        lr=0.01,
        energy_threshold=0.9,
        resync_every=100,
        ns_steps=20,
    )

    # Do a forward/backward to initialize
    x = torch.randn(32, d, device=device)
    y = torch.randn(32, device=device)
    loss = F.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()

    # Check state
    state = optimizer.state[model.W.weight]
    U = state['U']
    B = state['B']
    V = state['V']
    R = state['R']
    r = state['r']

    print(f"\nDecomposition shapes:")
    print(f"  U: {U.shape} (should be ({d_prime}, r))")
    print(f"  B: {B.shape} (should be ({r}, {r}))")
    print(f"  V: {V.shape} (should be ({d}, r))")
    print(f"  R: {R.shape} (should be ({d_prime}, {d}))")
    print(f"  r: {r}")

    # Check shapes are correct
    assert U.shape == (d_prime, r), f"U shape wrong: {U.shape}"
    assert V.shape == (d, r), f"V shape wrong: {V.shape}"
    assert B.shape == (r, r), f"B shape wrong: {B.shape}"
    assert R.shape == (d_prime, d), f"R shape wrong: {R.shape}"

    # Check orthonormality
    I_r = torch.eye(r, device=device, dtype=U.dtype)
    U_orth_err = torch.linalg.norm(U.T @ U - I_r).item()
    V_orth_err = torch.linalg.norm(V.T @ V - I_r).item()

    print(f"\nOrthonormality check:")
    print(f"  ||U.T @ U - I_r||: {U_orth_err:.8f}")
    print(f"  ||V.T @ V - I_r||: {V_orth_err:.8f}")

    # Check reconstruction
    W_reconstructed = U @ B @ V.T + R
    recon_err = torch.linalg.norm(model.W.weight - W_reconstructed).item()
    print(f"\nReconstruction check:")
    print(f"  ||W - (U @ B @ V.T + R)||: {recon_err:.8f}")

    # Assertions
    assert U_orth_err < 0.01, f"U not orthonormal: {U_orth_err}"
    assert V_orth_err < 0.01, f"V not orthonormal: {V_orth_err}"
    assert recon_err < 1e-4, f"Reconstruction failed: {recon_err}"

    print("\n✓ All orthonormality checks passed for rectangular matrix")


def plot_results(history: dict, save_path: str = None):
    """Plot training results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Loss
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train')
    ax.plot(history['test_loss'], label='Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Loss')
    ax.legend()
    ax.grid(True)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(history['train_acc'], label='Train')
    ax.plot(history['test_acc'], label='Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy')
    ax.legend()
    ax.grid(True)
    ax.set_ylim([0, 1])

    # Norms
    ax = axes[0, 2]
    ax.plot(history['W_norm'], label='||W||')
    ax.plot(history['U_norm'], label='||U||')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Frobenius Norm')
    ax.set_title('Weight Norms')
    ax.legend()
    ax.grid(True)

    # Residual ratio
    ax = axes[1, 0]
    ax.plot(history['residual_ratio'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('||R|| / ||W||')
    ax.set_title('Frozen Residual Ratio')
    ax.grid(True)

    # Orthonormality errors
    ax = axes[1, 1]
    ax.plot(history['U_orth_error'], label='U orthonormality')
    ax.plot(history['V_orth_error'], label='V orthonormality')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('||X.T @ X - I||')
    ax.set_title('Orthonormality Errors')
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')

    # Rank
    ax = axes[1, 2]
    ax.plot(history['rank'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Rank')
    ax.set_title('Effective Rank')
    ax.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def test_different_aspect_ratios():
    """Test different aspect ratios for rectangular matrices."""
    print("=" * 70)
    print("Testing different aspect ratios for rectangular W")
    print("=" * 70)

    d = 128
    d_primes = [32, 64, 96, 128, 192, 256]  # Various aspect ratios
    results = {}

    for d_prime in d_primes:
        aspect = "wide" if d_prime < d else ("square" if d_prime == d else "tall")
        print(f"\n--- d'={d_prime}, d={d} ({aspect} matrix, ratio={d_prime/d:.2f}) ---")

        result = test_rectangular_optimizer(
            d=d,
            d_prime=d_prime,
            n_epochs=50,
            verbose=False,
        )
        results[d_prime] = result

        metrics = result['detailed_metrics']
        print(f"Final test acc: {result['final_test_acc']:.2%}")
        if metrics:
            print(f"  Mean rank: {metrics.get('proj/mean_rank', 'N/A'):.0f}")
            print(f"  U orthonormality: {metrics.get('proj/mean_orthonormality_U', 'N/A'):.6f}")
            print(f"  V orthonormality: {metrics.get('proj/mean_orthonormality_V', 'N/A'):.6f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test ProjectedGradientOptimizer with rectangular matrices")
    parser.add_argument("--test", type=str, default="basic",
                        choices=["basic", "orthonormality", "aspect_ratios", "all"],
                        help="Which test to run")
    parser.add_argument("--d", type=int, default=128, help="Input dimension")
    parser.add_argument("--d_prime", type=int, default=64, help="Hidden dimension (d' for W)")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr_W", type=float, default=0.1, help="Learning rate for W")
    parser.add_argument("--lr_U", type=float, default=0.01, help="Learning rate for U")
    parser.add_argument("--energy_threshold", type=float, default=0.9, help="Energy threshold")
    parser.add_argument("--resync_every", type=int, default=50, help="Resync interval")
    parser.add_argument("--plot", action="store_true", help="Plot results")
    parser.add_argument("--save_plot", type=str, default=None, help="Save plot to file")
    args = parser.parse_args()

    if args.test == "basic" or args.test == "all":
        print("=" * 70)
        print("BASIC RECTANGULAR MATRIX TEST")
        print("=" * 70)
        results = test_rectangular_optimizer(
            d=args.d,
            d_prime=args.d_prime,
            n_epochs=args.n_epochs,
            lr_W=args.lr_W,
            lr_U=args.lr_U,
            energy_threshold=args.energy_threshold,
            resync_every=args.resync_every,
            verbose=True,
        )

        if args.plot or args.save_plot:
            plot_results(results['history'], save_path=args.save_plot)

        # Assert learning happened
        assert results['final_test_acc'] > results['initial_test_acc'], \
            f"No learning! Initial: {results['initial_test_acc']:.2%}, Final: {results['final_test_acc']:.2%}"
        assert results['final_test_acc'] > 0.65, \
            f"Poor accuracy: {results['final_test_acc']:.2%} (expected > 65%)"
        print("\n✓ Basic rectangular test PASSED")

    if args.test == "orthonormality" or args.test == "all":
        test_orthonormality_preservation()

    if args.test == "aspect_ratios" or args.test == "all":
        test_different_aspect_ratios()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)
