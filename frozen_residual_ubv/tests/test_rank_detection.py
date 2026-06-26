"""
Tests for energy-based rank detection.

Tests verify:
1. Energy-based rank detection captures target energy fraction
2. Tolerance-based rank detection finds significant singular values
3. create_rank_fn works with different rank specifications
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from frozen_residual_ubv.model_surgery import (
    compute_energy_rank,
    compute_tolerance_rank,
    create_rank_fn,
)


class TestEnergyRank:
    """Test energy-based rank detection."""

    def test_energy_rank_low_rank_matrix(self):
        """Test energy rank detection on a clearly low-rank matrix."""
        torch.manual_seed(42)

        # Create a rank-4 matrix (64x128)
        r_true = 4
        m, n = 64, 128
        U = torch.randn(m, r_true)
        V = torch.randn(n, r_true)
        W = U @ V.T  # Exactly rank 4

        # With 80% energy threshold, should detect rank ~4
        rank = compute_energy_rank(W, energy_threshold=0.8)
        assert rank <= r_true + 2, f"Expected rank ~{r_true}, got {rank}"
        assert rank >= 1, f"Rank should be at least 1"

    def test_energy_rank_captures_target_energy(self):
        """Verify detected rank captures at least target energy."""
        torch.manual_seed(42)

        m, n = 64, 128
        W = torch.randn(m, n)

        for threshold in [0.5, 0.8, 0.9, 0.95]:
            rank = compute_energy_rank(W, energy_threshold=threshold)

            # Compute actual energy captured
            S = torch.linalg.svdvals(W.float())
            S_sq = S ** 2
            total_energy = S_sq.sum()
            captured_energy = S_sq[:rank].sum()
            actual_ratio = (captured_energy / total_energy).item()

            assert actual_ratio >= threshold - 0.01, (
                f"threshold={threshold}: rank={rank} only captures {actual_ratio:.3f}"
            )

    def test_energy_rank_respects_bounds(self):
        """Test that min_rank and max_rank are respected."""
        torch.manual_seed(42)

        m, n = 64, 128
        W = torch.randn(m, n)

        # Test min_rank
        rank = compute_energy_rank(W, energy_threshold=0.01, min_rank=10)
        assert rank >= 10, f"min_rank not respected: got {rank}"

        # Test max_rank
        rank = compute_energy_rank(W, energy_threshold=0.99, max_rank=5)
        assert rank <= 5, f"max_rank not respected: got {rank}"

    def test_energy_rank_full_rank_matrix(self):
        """Test on a full-rank random matrix."""
        torch.manual_seed(42)

        m, n = 32, 64
        W = torch.randn(m, n)

        # With 99% threshold, should need many singular values
        rank_99 = compute_energy_rank(W, energy_threshold=0.99)
        rank_50 = compute_energy_rank(W, energy_threshold=0.5)

        assert rank_99 > rank_50, "Higher threshold should require higher rank"


class TestToleranceRank:
    """Test tolerance-based rank detection."""

    def test_tolerance_rank_low_rank_matrix(self):
        """Test tolerance rank on a low-rank matrix with small noise."""
        torch.manual_seed(42)

        # Rank-4 matrix with tiny noise
        r_true = 4
        m, n = 64, 128
        U = torch.randn(m, r_true)
        V = torch.randn(n, r_true)
        W = U @ V.T + 1e-8 * torch.randn(m, n)  # Tiny noise

        # With reasonable tolerance, should detect ~4
        rank = compute_tolerance_rank(W, rank_tol=1e-5)
        assert rank <= r_true + 2, f"Expected rank ~{r_true}, got {rank}"

    def test_tolerance_rank_respects_bounds(self):
        """Test that min_rank and max_rank are respected."""
        torch.manual_seed(42)

        m, n = 64, 128
        W = torch.randn(m, n)

        # Test min_rank
        rank = compute_tolerance_rank(W, rank_tol=0.5, min_rank=10)
        assert rank >= 10, f"min_rank not respected: got {rank}"

        # Test max_rank
        rank = compute_tolerance_rank(W, rank_tol=1e-10, max_rank=5)
        assert rank <= 5, f"max_rank not respected: got {rank}"


class TestCreateRankFn:
    """Test create_rank_fn factory."""

    def test_fixed_rank(self):
        """Test fixed rank function."""
        rank_fn = create_rank_fn(rank=16)

        # Create a mock module
        module = nn.Linear(128, 64)

        rank = rank_fn("layer.0", module)
        assert rank == 16, f"Expected fixed rank 16, got {rank}"

    def test_fixed_rank_clamped(self):
        """Test fixed rank is clamped to matrix size."""
        rank_fn = create_rank_fn(rank=100)

        # Small module where rank > min(m, n)
        module = nn.Linear(32, 16)

        rank = rank_fn("layer.0", module)
        assert rank == 16, f"Rank should be clamped to min(m,n)=16, got {rank}"

    def test_full_rank(self):
        """Test full rank function."""
        rank_fn = create_rank_fn(rank="full")

        module = nn.Linear(128, 64)
        rank = rank_fn("layer.0", module)
        assert rank == 64, f"Expected full rank 64, got {rank}"

    def test_auto_rank_energy(self):
        """Test auto rank with energy threshold."""
        rank_fn = create_rank_fn(rank="auto", energy_threshold=0.8)

        # Create module with known rank structure
        module = nn.Linear(128, 64)
        # Initialize with low-rank weight
        r_true = 8
        U = torch.randn(64, r_true)
        V = torch.randn(128, r_true)
        module.weight.data = U @ V.T

        rank = rank_fn("layer.0", module)
        # Should detect approximately r_true
        assert rank <= r_true + 4, f"Expected rank ~{r_true}, got {rank}"
        assert rank >= 1, f"Rank should be at least 1"

    def test_auto_rank_tolerance(self):
        """Test auto rank with tolerance-based detection."""
        rank_fn = create_rank_fn(rank="auto", energy_threshold=0)

        module = nn.Linear(128, 64)
        rank = rank_fn("layer.0", module)
        assert 1 <= rank <= 64, f"Invalid rank: {rank}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
