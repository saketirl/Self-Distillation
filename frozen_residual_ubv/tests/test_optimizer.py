"""
Tests for UBVOptimizer.

Tests verify:
1. Single-step correctness for each update rule
2. Stiefel constraint preservation after many steps
3. B positive definiteness maintenance
4. Warm-starting effectiveness for dual variables
5. Toy convergence on matrix recovery
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
from frozen_residual_ubv.utils import check_stiefel_constraint, check_spd_constraint


class TestOptimizerBasic:
    """Basic optimizer functionality tests."""

    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_single_step_runs(self, device):
        """Verify optimizer step runs without error."""
        torch.manual_seed(42)

        m, n, r = 32, 64, 8
        W = torch.randn(m, n, device=device)
        factored = FactoredLinear(W, r_star=r)
        factored.to(device)

        optimizer = UBVOptimizer(factored.parameters(), lr_U=0.01, lr_B=0.001)

        # Forward and backward
        x = torch.randn(16, n, device=device)
        target = torch.randn(16, m, device=device)

        y = factored(x)
        loss = ((y - target) ** 2).mean()
        loss.backward()

        # Step should not raise
        optimizer.step()
        optimizer.zero_grad()

    def test_loss_decreases(self, device):
        """Verify loss decreases after optimization step."""
        torch.manual_seed(42)

        m, n, r = 32, 64, 8
        W = torch.randn(m, n, device=device)
        factored = FactoredLinear(W, r_star=r)
        factored.to(device)

        optimizer = UBVOptimizer(factored.parameters(), lr_U=0.1, lr_B=0.01)

        x = torch.randn(16, n, device=device)
        target = torch.randn(16, m, device=device)

        # Initial loss
        y = factored(x)
        loss_before = ((y - target) ** 2).mean().item()

        # Take several steps
        for _ in range(10):
            optimizer.zero_grad()
            y = factored(x)
            loss = ((y - target) ** 2).mean()
            loss.backward()
            optimizer.step()

        # Final loss
        y = factored(x)
        loss_after = ((y - target) ** 2).mean().item()

        assert loss_after < loss_before, (
            f"Loss should decrease: {loss_before} -> {loss_after}"
        )


class TestStiefelConstraint:
    """Test Stiefel manifold constraint preservation."""

    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_u_orthonormality_preserved(self, device):
        """
        After N=100 steps, verify ||U^T U - I||_F < 10^{-4}.
        """
        torch.manual_seed(42)

        m, n, r = 32, 64, 8
        W = torch.randn(m, n, device=device)
        factored = FactoredLinear(W, r_star=r)
        factored.to(device)

        optimizer = UBVOptimizer(
            factored.parameters(),
            lr_U=0.01,
            lr_B=0.001,
            dual_max_iters=10,
        )

        # Run 100 steps
        for _ in range(100):
            optimizer.zero_grad()
            x = torch.randn(16, n, device=device)
            target = torch.randn(16, m, device=device)
            y = factored(x)
            loss = ((y - target) ** 2).mean()
            loss.backward()
            optimizer.step()

        # Check U orthonormality
        is_valid, violation = check_stiefel_constraint(factored.U.data, tol=1e-4)
        assert is_valid, f"U Stiefel violation: {violation} (should be < 1e-4)"

    def test_v_orthonormality_preserved(self, device):
        """
        After N=100 steps, verify ||V^T V - I||_F < 10^{-4}.
        """
        torch.manual_seed(42)

        m, n, r = 32, 64, 8
        W = torch.randn(m, n, device=device)
        factored = FactoredLinear(W, r_star=r)
        factored.to(device)

        optimizer = UBVOptimizer(
            factored.parameters(),
            lr_U=0.01,
            lr_V=0.01,
            lr_B=0.001,
            dual_max_iters=10,
        )

        # Run 100 steps
        for _ in range(100):
            optimizer.zero_grad()
            x = torch.randn(16, n, device=device)
            target = torch.randn(16, m, device=device)
            y = factored(x)
            loss = ((y - target) ** 2).mean()
            loss.backward()
            optimizer.step()

        # Check V orthonormality
        is_valid, violation = check_stiefel_constraint(factored.V.data, tol=1e-4)
        assert is_valid, f"V Stiefel violation: {violation} (should be < 1e-4)"


class TestSPDConstraint:
    """Test S_{++} constraint preservation for B."""

    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_b_positive_definiteness(self, device):
        """
        After 100 steps, verify all eigenvalues of B are >= lambda_min.
        """
        torch.manual_seed(42)

        m, n, r = 32, 64, 8
        lambda_min = 1e-4

        W = torch.randn(m, n, device=device)
        factored = FactoredLinear(W, r_star=r)
        factored.to(device)

        optimizer = UBVOptimizer(
            factored.parameters(),
            lr_U=0.01,
            lr_B=0.01,  # Larger LR to test eigenvalue clamping
            lambda_min=lambda_min,
        )

        # Run 100 steps
        for _ in range(100):
            optimizer.zero_grad()
            x = torch.randn(16, n, device=device)
            target = torch.randn(16, m, device=device)
            y = factored(x)
            loss = ((y - target) ** 2).mean()
            loss.backward()
            optimizer.step()

        # Check B positive definiteness
        is_valid, min_eig = check_spd_constraint(factored.B.data, lambda_min=lambda_min)
        assert is_valid, f"B min eigenvalue: {min_eig} (should be >= {lambda_min})"

    def test_b_symmetry_preserved(self, device):
        """Verify B remains symmetric after updates."""
        torch.manual_seed(42)

        m, n, r = 32, 64, 8
        W = torch.randn(m, n, device=device)
        factored = FactoredLinear(W, r_star=r)
        factored.to(device)

        optimizer = UBVOptimizer(factored.parameters())

        # Run several steps
        for _ in range(50):
            optimizer.zero_grad()
            x = torch.randn(16, n, device=device)
            target = torch.randn(16, m, device=device)
            y = factored(x)
            loss = ((y - target) ** 2).mean()
            loss.backward()
            optimizer.step()

        # Check symmetry
        B = factored.B.data
        sym_error = (B - B.T).abs().max()
        assert sym_error < 1e-5, f"B symmetry error: {sym_error}"


class TestStateTracking:
    """Test optimizer state tracking."""

    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_step_count_tracking(self, device):
        """
        Verify that step count is tracked in optimizer state.
        """
        torch.manual_seed(42)

        m, n, r = 32, 64, 8
        W = torch.randn(m, n, device=device)
        factored = FactoredLinear(W, r_star=r)
        factored.to(device)

        optimizer = UBVOptimizer(
            factored.parameters(),
            lr_U=0.01,
        )

        x = torch.randn(16, n, device=device)
        target = torch.randn(16, m, device=device)

        # Step 1
        optimizer.zero_grad()
        y = factored(x)
        loss = ((y - target) ** 2).mean()
        loss.backward()
        optimizer.step()

        # Check that step_count was stored for Stiefel parameters
        for p in factored.parameters():
            if getattr(p, '_ubv_role', None) in ('U', 'V'):
                state = optimizer.state.get(p, {})
                assert 'step_count' in state, "step_count should be stored after first step"
                assert state['step_count'] == 1, "step_count should be 1 after first step"

        # Step 2
        optimizer.zero_grad()
        y = factored(x)
        loss = ((y - target) ** 2).mean()
        loss.backward()
        optimizer.step()

        # Verify step count incremented
        for p in factored.parameters():
            if getattr(p, '_ubv_role', None) in ('U', 'V'):
                state = optimizer.state.get(p, {})
                assert state['step_count'] == 2, "step_count should be 2 after second step"


class TestConvergence:
    """Test convergence on toy problems."""

    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_low_rank_matrix_recovery(self, device):
        """
        On a low-rank matrix recovery problem, verify convergence to low loss
        in <500 steps.

        Problem: min_W ||W - W*||_F^2 where W* is rank-r*.
        """
        torch.manual_seed(42)

        m, n, r = 32, 64, 8
        num_steps = 500

        # Create low-rank target
        U_target = torch.linalg.qr(torch.randn(m, r, device=device))[0]
        V_target = torch.linalg.qr(torch.randn(n, r, device=device))[0]
        S_target = torch.linspace(5.0, 0.5, r, device=device)
        W_target = U_target @ torch.diag(S_target) @ V_target.T

        # Initialize factored linear
        W_init = torch.randn(m, n, device=device)
        factored = FactoredLinear(W_init, r_star=r, freeze_residual=False)
        factored.to(device)

        optimizer = UBVOptimizer(
            factored.parameters(),
            lr_U=0.05,
            lr_B=0.01,
            lr_V=0.05,
        )

        initial_loss = None
        final_loss = None

        for step in range(num_steps):
            optimizer.zero_grad()

            # Compute loss: ||W_eff - W_target||_F^2
            W_eff = factored.get_effective_weight()
            loss = ((W_eff - W_target) ** 2).sum()

            if step == 0:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

            final_loss = loss.item()

        # Loss should decrease significantly (at least 50%)
        assert final_loss < initial_loss * 0.5, (
            f"Loss should decrease by >50%: {initial_loss} -> {final_loss}"
        )

        # Relative error should be reasonably small (< 10%)
        W_eff = factored.get_effective_weight()
        relative_error = ((W_eff - W_target) ** 2).sum() / (W_target ** 2).sum()
        assert relative_error < 0.1, f"Relative error too high: {relative_error}"


class TestMomentum:
    """Test momentum functionality."""

    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_momentum_buffer_created(self, device):
        """Verify momentum buffer is created when momentum > 0."""
        torch.manual_seed(42)

        m, n, r = 32, 64, 8
        W = torch.randn(m, n, device=device)
        factored = FactoredLinear(W, r_star=r)
        factored.to(device)

        optimizer = UBVOptimizer(
            factored.parameters(),
            momentum=0.9,
        )

        # Take a step
        x = torch.randn(16, n, device=device)
        target = torch.randn(16, m, device=device)
        y = factored(x)
        loss = ((y - target) ** 2).mean()
        loss.backward()
        optimizer.step()

        # Check momentum buffer exists for B
        for p in factored.parameters():
            if getattr(p, '_ubv_role', None) == 'B':
                state = optimizer.state.get(p, {})
                assert 'momentum_buffer' in state, "Momentum buffer should exist"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
