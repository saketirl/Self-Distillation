#!/usr/bin/env python3
"""
Test ProjectedGradientOptimizer on simple linear regression.

Network: f(x) = U @ W @ x
- U ∈ R^{1×d}: trained with Adam
- W ∈ R^{d×d}: trained with ProjectedGradientOptimizer

Task: Binary classification with MSE loss
- Class 0: x ~ N(μ_0, I)
- Class 1: x ~ N(μ_1, I)
- y ∈ {0, 1}

This test verifies the optimizer correctly descends on a simple problem.
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


class LinearClassifier(nn.Module):
    """Simple two-layer linear network: f(x) = U @ W @ x"""

    def __init__(self, d: int = 128):
        super().__init__()
        self.d = d
        # W: d x d matrix (trained with projected gradient)
        self.W = nn.Linear(d, d, bias=False)
        # U: 1 x d matrix (trained with Adam)
        self.U = nn.Linear(d, 1, bias=False)

        # Initialize W close to identity for stability
        nn.init.eye_(self.W.weight)
        # Small random init for U
        nn.init.normal_(self.U.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # f(x) = U @ W @ x
        # x: (batch, d)
        # W @ x: (batch, d)
        # U @ (W @ x): (batch, 1)
        h = self.W(x)  # (batch, d)
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


def test_projected_optimizer(
    d: int = 128,
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
    Test ProjectedGradientOptimizer on linear regression.

    Returns:
        dict with training history and final metrics
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"Device: {device}")
        print(f"d={d}, n_train={n_train}, n_test={n_test}")
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

    # Create model
    model = LinearClassifier(d).to(device)

    # Create optimizers
    # W: ProjectedGradientOptimizer
    optimizer_W = ProjectedGradientOptimizer(
        [{'params': [model.W.weight], 'lr': lr_W}],
        lr=lr_W,
        energy_threshold=energy_threshold,
        resync_every=resync_every,
        ns_steps=10,
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

        # Record history
        history['train_loss'].append(epoch_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['W_norm'].append(W_norm)
        history['U_norm'].append(U_norm)
        history['B_eigenvalues'].append(B_eigs)
        history['residual_ratio'].append(residual_ratio)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: loss={epoch_loss:.4f}, "
                  f"train_acc={train_acc:.2%}, test_acc={test_acc:.2%}, "
                  f"||W||={W_norm:.2f}, ||U||={U_norm:.4f}, "
                  f"res_ratio={residual_ratio:.4f}")

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
        print("\nOptimizer metrics:")
        for k, v in detailed_metrics.items():
            print(f"  {k}: {v:.6f}")

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


def plot_results(history: dict, save_path: str = None):
    """Plot training results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

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
    ax = axes[1, 0]
    ax.plot(history['W_norm'], label='||W||')
    ax.plot(history['U_norm'], label='||U||')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Frobenius Norm')
    ax.set_title('Weight Norms')
    ax.legend()
    ax.grid(True)

    # Residual ratio
    ax = axes[1, 1]
    ax.plot(history['residual_ratio'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('||R|| / ||W||')
    ax.set_title('Frozen Residual Ratio')
    ax.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def test_comparison_with_adam(
    d: int = 128,
    n_train: int = 1000,
    n_test: int = 200,
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 0.1,
    seed: int = 42,
):
    """Compare ProjectedGradientOptimizer with Adam baseline."""
    print("=" * 60)
    print("COMPARISON: ProjectedGradientOptimizer vs Adam")
    print("=" * 60)

    # Test with ProjectedGradientOptimizer
    print("\n--- ProjectedGradientOptimizer ---")
    proj_results = test_projected_optimizer(
        d=d, n_train=n_train, n_test=n_test, n_epochs=n_epochs,
        batch_size=batch_size, lr_W=lr, lr_U=lr/10, seed=seed, verbose=True
    )

    # Test with Adam (baseline)
    print("\n--- Adam Baseline ---")
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Same data
    mu_0 = torch.zeros(d, device=device)
    mu_1 = torch.zeros(d, device=device)
    mu_0[:10] = -1.0
    mu_1[:10] = 1.0

    x_train, y_train = generate_data(n_train, d, mu_0, mu_1, device)
    x_test, y_test = generate_data(n_test, d, mu_0, mu_1, device)

    model_adam = LinearClassifier(d).to(device)
    optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=lr)

    adam_history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

    n_batches = (n_train + batch_size - 1) // batch_size

    for epoch in range(n_epochs):
        model_adam.train()
        epoch_loss = 0.0

        perm = torch.randperm(n_train, device=device)
        x_train_shuffled = x_train[perm]
        y_train_shuffled = y_train[perm]

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_train)

            x_batch = x_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]

            optimizer_adam.zero_grad()
            preds = model_adam(x_batch)
            loss = F.mse_loss(preds, y_batch)
            loss.backward()
            optimizer_adam.step()

            epoch_loss += loss.item()

        epoch_loss /= n_batches

        model_adam.eval()
        with torch.no_grad():
            test_loss = F.mse_loss(model_adam(x_test), y_test).item()

        train_acc = compute_accuracy(model_adam, x_train, y_train)
        test_acc = compute_accuracy(model_adam, x_test, y_test)

        adam_history['train_loss'].append(epoch_loss)
        adam_history['test_loss'].append(test_loss)
        adam_history['train_acc'].append(train_acc)
        adam_history['test_acc'].append(test_acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: loss={epoch_loss:.4f}, "
                  f"train_acc={train_acc:.2%}, test_acc={test_acc:.2%}")

    final_adam_acc = compute_accuracy(model_adam, x_test, y_test)
    print(f"\nFinal Adam test accuracy: {final_adam_acc:.2%}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"ProjectedGradient final test acc: {proj_results['final_test_acc']:.2%}")
    print(f"Adam final test acc: {final_adam_acc:.2%}")
    print(f"Difference: {proj_results['final_test_acc'] - final_adam_acc:+.2%}")

    return proj_results, adam_history


def test_different_energy_thresholds():
    """Test different energy thresholds to see their effect."""
    print("=" * 60)
    print("TESTING DIFFERENT ENERGY THRESHOLDS")
    print("=" * 60)

    thresholds = [0.5, 0.7, 0.9, 0.95, 0.99]
    results = {}

    for threshold in thresholds:
        print(f"\n--- Energy threshold: {threshold} ---")
        result = test_projected_optimizer(
            energy_threshold=threshold,
            n_epochs=50,
            verbose=False,
        )
        results[threshold] = result
        print(f"Final test acc: {result['final_test_acc']:.2%}")

        metrics = result['detailed_metrics']
        if metrics:
            print(f"  Mean rank: {metrics.get('proj/mean_rank', 'N/A')}")
            print(f"  Residual ratio: {metrics.get('proj/mean_residual_ratio', 'N/A'):.4f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test ProjectedGradientOptimizer")
    parser.add_argument("--test", type=str, default="basic",
                        choices=["basic", "comparison", "thresholds", "all"],
                        help="Which test to run")
    parser.add_argument("--d", type=int, default=128, help="Input dimension")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr_W", type=float, default=0.1, help="Learning rate for W")
    parser.add_argument("--lr_U", type=float, default=0.01, help="Learning rate for U")
    parser.add_argument("--energy_threshold", type=float, default=0.9, help="Energy threshold")
    parser.add_argument("--resync_every", type=int, default=50, help="Resync interval")
    parser.add_argument("--plot", action="store_true", help="Plot results")
    parser.add_argument("--save_plot", type=str, default=None, help="Save plot to file")
    args = parser.parse_args()

    if args.test == "basic" or args.test == "all":
        print("=" * 60)
        print("BASIC TEST")
        print("=" * 60)
        results = test_projected_optimizer(
            d=args.d,
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
        print("\n✓ Basic test PASSED")

    if args.test == "comparison" or args.test == "all":
        test_comparison_with_adam()

    if args.test == "thresholds" or args.test == "all":
        test_different_energy_thresholds()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
