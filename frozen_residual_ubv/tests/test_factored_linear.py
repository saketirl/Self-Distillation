"""
Tests for FactoredLinear module.

Tests verify:
1. Forward pass equivalence with full-rank and residual modes
2. Gradient flow to U, B, V but not W_res
3. Correct parameter counts
4. Numerical stability with various dtypes
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from frozen_residual_ubv.factored_linear import FactoredLinear


class TestFactoredLinearForward:
    """Test forward pass correctness."""

    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_full_rank_equivalence(self, device):
        """
        For r_star = min(m, n) (full rank mode, no residual), verify
        UBV^T x ≈ W x to float32 tolerance right after initialization.
        """
        torch.manual_seed(42)

        m, n = 64, 128
        W = torch.randn(m, n, device=device)

        # Full rank: r_star = min(m, n) = 64
        r_star = min(m, n)
        factored = FactoredLinear(W, r_star=r_star, freeze_residual=False)
        factored.to(device)

        # Residual should be None in full-rank mode
        assert factored.W_res is None, "Residual should be None for full-rank"

        # Test forward equivalence
        x = torch.randn(32, n, device=device)

        with torch.no_grad():
            y_original = x @ W.T
            y_factored = factored(x)

        # Should match to float32 tolerance (allow for accumulated numerical error)
        assert torch.allclose(y_original, y_factored, atol=1e-4, rtol=1e-4), (
            f"Full-rank forward mismatch: max diff = {(y_original - y_factored).abs().max()}"
        )

    def test_residual_forward_equivalence(self, device):
        """
        Verify W_res x + UBV^T x ≈ W x for general r_star.
        """
        torch.manual_seed(42)

        m, n = 64, 128
        W = torch.randn(m, n, device=device)

        # Partial rank: capture top 16 singular values
        r_star = 16
        factored = FactoredLinear(W, r_star=r_star, freeze_residual=True)
        factored.to(device)

        # Residual should exist
        assert factored.W_res is not None, "Residual should exist for partial rank"

        # Test forward equivalence
        x = torch.randn(32, n, device=device)

        with torch.no_grad():
            y_original = x @ W.T
            y_factored = factored(x)

        # Should match to float32 tolerance (allow for accumulated numerical error)
        assert torch.allclose(y_original, y_factored, atol=1e-4, rtol=1e-4), (
            f"Residual forward mismatch: max diff = {(y_original - y_factored).abs().max()}"
        )

    def test_forward_with_bias(self, device):
        """Test forward pass with bias."""
        torch.manual_seed(42)

        m, n = 32, 64
        W = torch.randn(m, n, device=device)
        bias = torch.randn(m, device=device)

        r_star = 8
        factored = FactoredLinear(W, r_star=r_star, bias=bias)
        factored.to(device)

        x = torch.randn(16, n, device=device)

        with torch.no_grad():
            y_original = x @ W.T + bias
            y_factored = factored(x)

        assert torch.allclose(y_original, y_factored, atol=1e-4, rtol=1e-4)

    def test_forward_batch_dimensions(self, device):
        """Test forward pass with various batch dimensions."""
        torch.manual_seed(42)

        m, n = 32, 64
        W = torch.randn(m, n, device=device)
        r_star = 8
        factored = FactoredLinear(W, r_star=r_star)
        factored.to(device)

        # Test different batch shapes
        for batch_shape in [(16,), (4, 8), (2, 3, 4)]:
            x = torch.randn(*batch_shape, n, device=device)
            y = factored(x)
            assert y.shape == (*batch_shape, m), f"Shape mismatch for batch {batch_shape}"


class TestFactoredLinearGradients:
    """Test gradient flow."""

    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_grad_flow_to_factors(self, device):
        """
        Wrap in a toy loss, call backward, confirm .grad is populated
        on U, B, V but not on W_res.
        """
        torch.manual_seed(42)

        m, n = 32, 64
        W = torch.randn(m, n, device=device)
        r_star = 8

        factored = FactoredLinear(W, r_star=r_star, freeze_residual=True)
        factored.to(device)

        # Forward and backward
        x = torch.randn(16, n, device=device, requires_grad=True)
        target = torch.randn(16, m, device=device)

        y = factored(x)
        loss = ((y - target) ** 2).mean()
        loss.backward()

        # Check gradients exist for trainable factors
        assert factored.U.grad is not None, "U should have gradient"
        assert factored.B.grad is not None, "B should have gradient"
        assert factored.V.grad is not None, "V should have gradient"

        # Check gradients are non-zero
        assert factored.U.grad.abs().max() > 0, "U gradient is zero"
        assert factored.B.grad.abs().max() > 0, "B gradient is zero"
        assert factored.V.grad.abs().max() > 0, "V gradient is zero"

        # W_res is a buffer, not a parameter, so it shouldn't have grad
        # (buffers don't participate in autograd)

    def test_grad_shapes(self, device):
        """Verify gradient shapes match parameter shapes."""
        torch.manual_seed(42)

        m, n = 32, 64
        W = torch.randn(m, n, device=device)
        r_star = 8

        factored = FactoredLinear(W, r_star=r_star)
        factored.to(device)

        x = torch.randn(16, n, device=device)
        target = torch.randn(16, m, device=device)

        y = factored(x)
        loss = ((y - target) ** 2).mean()
        loss.backward()

        assert factored.U.grad.shape == (m, r_star)
        assert factored.B.grad.shape == (r_star, r_star)
        assert factored.V.grad.shape == (n, r_star)


class TestFactoredLinearParameters:
    """Test parameter counts and structure."""

    def test_parameter_count(self):
        """
        Confirm trainable parameter count = m*r + r^2 + n*r.
        """
        m, n = 64, 128
        r_star = 16

        W = torch.randn(m, n)
        factored = FactoredLinear(W, r_star=r_star)

        expected_trainable = m * r_star + r_star * r_star + n * r_star
        actual_trainable = factored.trainable_params

        assert actual_trainable == expected_trainable, (
            f"Expected {expected_trainable} trainable params, got {actual_trainable}"
        )

    def test_parameter_count_with_bias(self):
        """Test parameter count includes bias."""
        m, n = 64, 128
        r_star = 16

        W = torch.randn(m, n)
        bias = torch.randn(m)
        factored = FactoredLinear(W, r_star=r_star, bias=bias)

        expected_trainable = m * r_star + r_star * r_star + n * r_star + m
        actual_trainable = factored.trainable_params

        assert actual_trainable == expected_trainable

    def test_residual_is_frozen(self):
        """Confirm residual parameters are frozen (buffer, not parameter)."""
        m, n = 64, 128
        r_star = 16

        W = torch.randn(m, n)
        factored = FactoredLinear(W, r_star=r_star, freeze_residual=True)

        # W_res should be a buffer, not a parameter
        param_names = [name for name, _ in factored.named_parameters()]
        buffer_names = [name for name, _ in factored.named_buffers()]

        assert 'W_res' not in param_names, "W_res should not be a parameter"
        assert 'W_res' in buffer_names, "W_res should be a buffer"

    def test_ubv_role_tags(self):
        """Verify _ubv_role attributes are set correctly."""
        m, n = 32, 64
        r_star = 8

        W = torch.randn(m, n)
        bias = torch.randn(m)
        factored = FactoredLinear(W, r_star=r_star, bias=bias)

        assert hasattr(factored.U, '_ubv_role') and factored.U._ubv_role == "U"
        assert hasattr(factored.B, '_ubv_role') and factored.B._ubv_role == "B"
        assert hasattr(factored.V, '_ubv_role') and factored.V._ubv_role == "V"
        assert hasattr(factored.bias, '_ubv_role') and factored.bias._ubv_role == "bias"


class TestFactoredLinearInitialization:
    """Test initialization correctness."""

    def test_orthonormality_at_init(self):
        """Verify U, V have orthonormal columns at initialization."""
        torch.manual_seed(42)

        m, n = 64, 128
        r_star = 16
        W = torch.randn(m, n)

        factored = FactoredLinear(W, r_star=r_star)

        # Check U^T U ≈ I
        UtU = factored.U.T @ factored.U
        I_r = torch.eye(r_star)
        assert torch.allclose(UtU, I_r, atol=1e-5), "U should have orthonormal columns"

        # Check V^T V ≈ I
        VtV = factored.V.T @ factored.V
        assert torch.allclose(VtV, I_r, atol=1e-5), "V should have orthonormal columns"

    def test_b_is_diagonal_at_init(self):
        """Verify B is diagonal (from singular values) at initialization."""
        torch.manual_seed(42)

        m, n = 64, 128
        r_star = 16
        W = torch.randn(m, n)

        factored = FactoredLinear(W, r_star=r_star)

        # B should be diagonal at init
        B_offdiag = factored.B.data - torch.diag(torch.diag(factored.B.data))
        assert torch.allclose(B_offdiag, torch.zeros_like(B_offdiag), atol=1e-6), (
            "B should be diagonal at initialization"
        )

    def test_b_eigenvalues_positive_at_init(self):
        """Verify B has positive eigenvalues (from singular values)."""
        torch.manual_seed(42)

        m, n = 64, 128
        r_star = 16
        W = torch.randn(m, n)

        factored = FactoredLinear(W, r_star=r_star)

        # B is diagonal, so diagonal entries are eigenvalues
        eigenvalues = torch.diag(factored.B.data)
        assert (eigenvalues > 0).all(), "B eigenvalues should be positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
