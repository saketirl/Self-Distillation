"""
Tests for spectrum preservation during training.

This is the key regression test for the rank-preservation claim. We verify:
1. Smallest singular value never drops below lambda_min
2. Effective rank stays within ±20% of initial value
3. No NaNs or Infs during training
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from frozen_residual_ubv.factored_linear import FactoredLinear
from frozen_residual_ubv.ubv_optimizer import UBVOptimizer


class TestSpectrumPreservation:
    """Test spectrum preservation during extended training."""

    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute_b_stats(self, B: torch.Tensor) -> dict:
        """Compute spectrum statistics for B matrix."""
        B_sym = (B + B.T) / 2
        eigenvalues = torch.linalg.eigvalsh(B_sym.float())

        # Effective rank: tr(B) / ||B||_op = sum(lambda) / max(lambda)
        trace = eigenvalues.sum()
        spectral_norm = eigenvalues.max()
        effective_rank = trace / spectral_norm if spectral_norm > 0 else 0

        return {
            'min_eigenvalue': eigenvalues.min().item(),
            'max_eigenvalue': eigenvalues.max().item(),
            'effective_rank': effective_rank.item(),
            'trace': trace.item(),
            'condition_number': (spectral_norm / eigenvalues.min()).item() if eigenvalues.min() > 0 else float('inf'),
        }

    def test_spectrum_floor_maintained(self, device):
        """
        Run 1000 steps and verify smallest eigenvalue never drops below lambda_min.
        """
        torch.manual_seed(42)

        m, n, r = 32, 64, 8
        lambda_min = 1e-4
        num_steps = 1000

        W = torch.randn(m, n, device=device)
        factored = FactoredLinear(W, r_star=r)
        factored.to(device)

        optimizer = UBVOptimizer(
            factored.parameters(),
            lr_U=0.02,
            lr_B=0.01,
            lr_V=0.02,
            lambda_min=lambda_min,
        )

        min_eigenvalue_seen = float('inf')

        for step in range(num_steps):
            optimizer.zero_grad()

            # Random training targets (simulating diverse gradients)
            x = torch.randn(16, n, device=device)
            target = torch.randn(16, m, device=device)

            y = factored(x)
            loss = ((y - target) ** 2).mean()
            loss.backward()
            optimizer.step()

            # Track minimum eigenvalue
            stats = self.compute_b_stats(factored.B.data)
            min_eigenvalue_seen = min(min_eigenvalue_seen, stats['min_eigenvalue'])

            # Early failure detection
            assert stats['min_eigenvalue'] >= lambda_min * 0.99, (
                f"Step {step}: min eigenvalue {stats['min_eigenvalue']} < {lambda_min}"
            )

        # Final check
        assert min_eigenvalue_seen >= lambda_min * 0.99, (
            f"Min eigenvalue seen: {min_eigenvalue_seen}, expected >= {lambda_min}"
        )

    def test_effective_rank_preserved(self, device):
        """
        Run 1000 steps and verify effective rank stays within ±20% of initial.

        Effective rank = tr(B) / ||B||_op = sum(eigenvalues) / max(eigenvalue)
        """
        torch.manual_seed(42)

        m, n, r = 32, 64, 8
        num_steps = 1000

        W = torch.randn(m, n, device=device)
        factored = FactoredLinear(W, r_star=r)
        factored.to(device)

        optimizer = UBVOptimizer(
            factored.parameters(),
            lr_U=0.01,
            lr_B=0.005,
            lr_V=0.01,
            lambda_min=1e-4,
        )

        # Record initial effective rank
        initial_stats = self.compute_b_stats(factored.B.data)
        initial_eff_rank = initial_stats['effective_rank']

        effective_ranks = [initial_eff_rank]

        for step in range(num_steps):
            optimizer.zero_grad()

            x = torch.randn(16, n, device=device)
            target = torch.randn(16, m, device=device)

            y = factored(x)
            loss = ((y - target) ** 2).mean()
            loss.backward()
            optimizer.step()

            # Track effective rank every 100 steps
            if (step + 1) % 100 == 0:
                stats = self.compute_b_stats(factored.B.data)
                effective_ranks.append(stats['effective_rank'])

        # Verify effective rank stayed within ±20%
        final_eff_rank = effective_ranks[-1]
        ratio = final_eff_rank / initial_eff_rank

        assert 0.8 <= ratio <= 1.2, (
            f"Effective rank changed by more than 20%: {initial_eff_rank} -> {final_eff_rank} (ratio: {ratio})"
        )

    def test_no_nans_or_infs(self, device):
        """
        Run 1000 steps and verify no NaNs or Infs appear in parameters.
        """
        torch.manual_seed(42)

        m, n, r = 32, 64, 8
        num_steps = 1000

        W = torch.randn(m, n, device=device)
        factored = FactoredLinear(W, r_star=r)
        factored.to(device)

        optimizer = UBVOptimizer(
            factored.parameters(),
            lr_U=0.02,
            lr_B=0.01,
            lr_V=0.02,
        )

        for step in range(num_steps):
            optimizer.zero_grad()

            x = torch.randn(16, n, device=device)
            target = torch.randn(16, m, device=device)

            y = factored(x)
            loss = ((y - target) ** 2).mean()

            # Check loss is finite
            assert torch.isfinite(loss), f"Step {step}: Loss is not finite: {loss}"

            loss.backward()
            optimizer.step()

            # Check all parameters are finite
            for name, param in factored.named_parameters():
                assert torch.isfinite(param).all(), (
                    f"Step {step}: {name} contains NaN or Inf"
                )

    def test_spectrum_tracking(self, device):
        """
        Track full spectrum over training and verify stability.
        """
        torch.manual_seed(42)

        m, n, r = 32, 64, 8
        num_steps = 500

        W = torch.randn(m, n, device=device)
        factored = FactoredLinear(W, r_star=r)
        factored.to(device)

        optimizer = UBVOptimizer(
            factored.parameters(),
            lr_U=0.01,
            lr_B=0.005,
        )

        spectrum_history = []

        for step in range(num_steps):
            optimizer.zero_grad()

            x = torch.randn(16, n, device=device)
            target = torch.randn(16, m, device=device)

            y = factored(x)
            loss = ((y - target) ** 2).mean()
            loss.backward()
            optimizer.step()

            # Record spectrum every 50 steps
            if step % 50 == 0:
                B_sym = (factored.B.data.float() + factored.B.data.float().T) / 2
                eigenvalues = torch.linalg.eigvalsh(B_sym)
                spectrum_history.append({
                    'step': step,
                    'eigenvalues': eigenvalues.cpu().tolist(),
                    'min': eigenvalues.min().item(),
                    'max': eigenvalues.max().item(),
                })

        # Verify no collapse: ratio of min/max eigenvalues shouldn't explode
        initial_ratio = spectrum_history[0]['min'] / spectrum_history[0]['max']
        final_ratio = spectrum_history[-1]['min'] / spectrum_history[-1]['max']

        # Allow up to 10x change in condition number
        assert final_ratio > initial_ratio / 10, (
            f"Spectrum collapsed: initial ratio {initial_ratio}, final {final_ratio}"
        )


class TestMultipleFactoredLayers:
    """Test spectrum preservation with multiple FactoredLinear layers."""

    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_multiple_layers_stability(self, device):
        """Test stability with a small network of FactoredLinear layers."""
        torch.manual_seed(42)

        # Create a small network
        class FactoredNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = FactoredLinear(torch.randn(64, 32), r_star=8)
                self.fc2 = FactoredLinear(torch.randn(64, 64), r_star=16)
                self.fc3 = FactoredLinear(torch.randn(32, 64), r_star=8)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return self.fc3(x)

        net = FactoredNet().to(device)
        optimizer = UBVOptimizer(
            net.parameters(),
            lr_U=0.01,
            lr_B=0.005,
            lambda_min=1e-4,
        )

        num_steps = 200

        for step in range(num_steps):
            optimizer.zero_grad()

            x = torch.randn(16, 32, device=device)
            target = torch.randn(16, 32, device=device)

            y = net(x)
            loss = ((y - target) ** 2).mean()
            loss.backward()
            optimizer.step()

        # Verify all B matrices are valid
        for name, module in net.named_modules():
            if isinstance(module, FactoredLinear):
                B = module.B.data.float()
                B_sym = (B + B.T) / 2
                eigenvalues = torch.linalg.eigvalsh(B_sym)

                assert eigenvalues.min() >= 1e-4 * 0.99, (
                    f"{name}: min eigenvalue {eigenvalues.min()} < 1e-4"
                )
                assert torch.isfinite(eigenvalues).all(), (
                    f"{name}: eigenvalues contain NaN/Inf"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
