"""
Demo script comparing BWRiemannianSGD vs vanilla SGD on a linear layer fitting task.

Creates a nn.Linear(128, 64) layer and trains it to match a random target
linear function using low-rank constraints. Logs loss and singular value
spectrum every 10 steps.
"""

import torch
import torch.nn as nn
import numpy as np
from bw_riemannian_sgd import BWRiemannianSGD


def create_low_rank_target(in_features: int, out_features: int, rank: int) -> torch.Tensor:
    """Create an ill-conditioned low-rank target matrix."""
    # Create orthonormal bases
    U, _ = torch.linalg.qr(torch.randn(out_features, rank))
    V, _ = torch.linalg.qr(torch.randn(in_features, rank))

    # Ill-conditioned singular values (condition number ~ 100)
    singular_values = torch.logspace(2, 0, rank)  # [100, ..., 1]

    # Target = U @ diag(σ) @ V^T
    target = U @ torch.diag(singular_values) @ V.T
    return target


def get_singular_values(weight: torch.Tensor, top_k: int = 8) -> np.ndarray:
    """Get top-k singular values of weight matrix."""
    with torch.no_grad():
        _, S, _ = torch.linalg.svd(weight, full_matrices=False)
        return S[:top_k].cpu().numpy()


def train_with_optimizer(
    optimizer_class,
    optimizer_kwargs: dict,
    target: torch.Tensor,
    in_features: int,
    out_features: int,
    num_steps: int,
    log_interval: int = 10,
    seed: int = 42,
    project_rank: int = None,  # If set, project to this rank after each step
) -> dict:
    """Train a linear layer with the given optimizer."""
    torch.manual_seed(seed)

    # Create linear layer (no bias for simplicity)
    layer = nn.Linear(in_features, out_features, bias=False)

    # Initialize with small random weights
    nn.init.normal_(layer.weight, std=0.1)

    # Create optimizer
    optimizer = optimizer_class([layer.weight], **optimizer_kwargs)

    losses = []
    singular_values_history = []

    for step in range(num_steps):
        optimizer.zero_grad()

        # Loss: ||W - W_target||_F^2
        loss = ((layer.weight - target) ** 2).sum()
        loss.backward()

        optimizer.step()

        # Project to rank-r if specified (for projected SGD baseline)
        if project_rank is not None:
            with torch.no_grad():
                U, S, Vh = torch.linalg.svd(layer.weight.data, full_matrices=False)
                layer.weight.data = U[:, :project_rank] @ torch.diag(S[:project_rank]) @ Vh[:project_rank, :]

        losses.append(loss.item())

        if step % log_interval == 0 or step == num_steps - 1:
            sv = get_singular_values(layer.weight)
            singular_values_history.append((step, sv))

    return {
        'losses': losses,
        'singular_values': singular_values_history,
        'final_weight': layer.weight.detach().clone(),
    }


def main():
    print("=" * 70)
    print("BWRiemannianSGD vs Vanilla SGD Demo")
    print("=" * 70)

    # Setup
    in_features = 128
    out_features = 64
    target_rank = 8
    optimizer_rank = 8
    num_steps = 500
    lr = 0.01
    seed = 42

    print(f"\nConfiguration:")
    print(f"  Layer: Linear({in_features}, {out_features})")
    print(f"  Target rank: {target_rank}")
    print(f"  Optimizer rank: {optimizer_rank}")
    print(f"  Learning rate: {lr}")
    print(f"  Steps: {num_steps}")

    # Create target
    torch.manual_seed(seed)
    target = create_low_rank_target(in_features, out_features, target_rank)

    # Check condition number
    _, S_target, _ = torch.linalg.svd(target)
    cond_number = S_target[0] / S_target[target_rank - 1]
    print(f"  Target condition number: {cond_number:.2f}")

    # Train with BWRiemannianSGD
    print("\n" + "-" * 70)
    print("Training with BWRiemannianSGD...")
    print("-" * 70)

    results_bw = train_with_optimizer(
        BWRiemannianSGD,
        {'lr': lr, 'rank': optimizer_rank},
        target,
        in_features,
        out_features,
        num_steps,
        seed=seed,
    )

    print(f"\nBWRiemannianSGD Loss progression:")
    for step, sv in results_bw['singular_values']:
        loss = results_bw['losses'][step]
        sv_str = ', '.join([f'{v:.3f}' for v in sv[:4]])
        print(f"  Step {step:4d}: loss={loss:12.4f}, σ=[{sv_str}, ...]")

    # Train with Projected SGD (SGD + rank truncation - fair comparison)
    print("\n" + "-" * 70)
    print("Training with Projected SGD (SGD + rank truncation)...")
    print("-" * 70)

    results_sgd = train_with_optimizer(
        torch.optim.SGD,
        {'lr': lr},
        target,
        in_features,
        out_features,
        num_steps,
        seed=seed,
        project_rank=optimizer_rank,  # Project to same rank as BW
    )

    print(f"\nProjected SGD Loss progression:")
    for step, sv in results_sgd['singular_values']:
        loss = results_sgd['losses'][step]
        sv_str = ', '.join([f'{v:.3f}' for v in sv[:4]])
        print(f"  Step {step:4d}: loss={loss:12.4f}, σ=[{sv_str}, ...]")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    final_loss_bw = results_bw['losses'][-1]
    final_loss_sgd = results_sgd['losses'][-1]

    print(f"\nFinal losses:")
    print(f"  BWRiemannianSGD: {final_loss_bw:.6f}")
    print(f"  Projected SGD:   {final_loss_sgd:.6f}")

    if final_loss_bw < final_loss_sgd:
        improvement = (final_loss_sgd - final_loss_bw) / final_loss_sgd * 100
        print(f"\n  BWRiemannianSGD achieves {improvement:.1f}% lower loss!")
    else:
        diff = (final_loss_bw - final_loss_sgd) / final_loss_sgd * 100
        print(f"\n  Projected SGD achieved {diff:.1f}% lower loss")
        print(f"  (Both methods stay on rank-{optimizer_rank} manifold)")

    # Compare singular value spectra
    print(f"\nFinal singular value spectra (top 8):")
    print(f"  Target:          {', '.join([f'{v:.3f}' for v in S_target[:8].numpy()])}")

    sv_bw = get_singular_values(results_bw['final_weight'])
    sv_sgd = get_singular_values(results_sgd['final_weight'])

    print(f"  BWRiemannianSGD: {', '.join([f'{v:.3f}' for v in sv_bw])}")
    print(f"  Projected SGD:   {', '.join([f'{v:.3f}' for v in sv_sgd])}")

    # Check rank of final solutions
    rank_bw = torch.linalg.matrix_rank(results_bw['final_weight']).item()
    rank_sgd = torch.linalg.matrix_rank(results_sgd['final_weight']).item()

    print(f"\nNumerical ranks:")
    print(f"  Target:          {target_rank}")
    print(f"  BWRiemannianSGD: {rank_bw}")
    print(f"  Projected SGD:   {rank_sgd}")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
