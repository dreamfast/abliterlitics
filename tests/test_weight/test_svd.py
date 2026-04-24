"""Tests for src/weight/svd_analysis.py — SVD computation functions."""
from __future__ import annotations

import pytest
import torch

from src.weight.svd_analysis import compute_svd_stats


class TestComputeSVDStats:
    def test_rank1_matrix_eff_rank_1(self):
        """Rank-1 matrix should have effective_rank_90pct_energy = 1."""
        u = torch.tensor([[1.0], [2.0], [3.0]])
        v = torch.tensor([[4.0, 5.0, 6.0]])
        delta = u @ v  # rank 1, shape (3,3)
        result = compute_svd_stats(delta, top_k=5, full_svd=True)
        assert result["effective_rank_90pct_energy"] == 1
        assert result["energy_top1_pct"] == pytest.approx(100.0, abs=0.1)

    def test_rank2_matrix_eff_rank_2(self):
        """Rank-2 matrix should need rank 2 for 90% energy."""
        a = torch.zeros(4, 4)
        a[0, 0] = 10.0
        a[1, 1] = 5.0
        result = compute_svd_stats(a, top_k=5, full_svd=True)
        assert result["effective_rank_90pct_energy"] == 2

    def test_zero_matrix(self):
        """All-zero tensor should return basic stats without crashing."""
        delta = torch.zeros(4, 4)
        result = compute_svd_stats(delta, top_k=5, full_svd=True)
        assert result["frobenius_norm"] == 0.0
        assert "effective_rank_90pct_energy" not in result

    def test_1d_tensor(self):
        """1D tensor should return rank 1."""
        delta = torch.tensor([3.0, 4.0])
        result = compute_svd_stats(delta, top_k=5, full_svd=True)
        assert result["effective_rank_90pct_energy"] == 1

    def test_scalar(self):
        """Scalar should return basic stats."""
        delta = torch.tensor(5.0)
        result = compute_svd_stats(delta, top_k=5, full_svd=True)
        assert result["frobenius_norm"] == 5.0

    def test_frobenius_norm_correct(self):
        """Frobenius norm should match torch.norm."""
        delta = torch.randn(10, 8)
        result = compute_svd_stats(delta, top_k=5, full_svd=True)
        expected = float(delta.norm())
        assert result["frobenius_norm"] == pytest.approx(expected, abs=1e-4)

    def test_lowrank_vs_full_svd(self):
        """Low-rank should capture most energy for low-rank matrix."""
        # Create rank-2 matrix
        u = torch.randn(10, 2)
        v = torch.randn(2, 8)
        delta = u @ v
        result = compute_svd_stats(delta, top_k=5, full_svd=False)
        assert "lowrank_captured_energy_pct" in result
        assert result["lowrank_captured_energy_pct"] > 95.0

    def test_sv_ratio_top2(self):
        """SV ratio for rank-1 matrix should be very large (near-inf)."""
        u = torch.tensor([[1.0], [2.0], [3.0]])
        v = torch.tensor([[1.0, 1.0, 1.0]])
        delta = u @ v
        result = compute_svd_stats(delta, top_k=5, full_svd=True)
        # Due to float precision, S2 won't be exactly 0, but ratio should be very large
        assert result["sv_ratio_top2"] > 1000  # very large, effectively inf

    def test_result_has_required_keys(self):
        delta = torch.randn(6, 4)
        result = compute_svd_stats(delta, top_k=5, full_svd=True)
        for key in ("frobenius_norm", "mean_abs", "max_abs",
                     "effective_rank_90pct_energy", "top_singular_values"):
            assert key in result, f"Missing key: {key}"
