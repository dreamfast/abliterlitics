"""Tests for src/weight/lowrank_reconstruction.py — Low-rank reconstruction math."""
from __future__ import annotations

import pytest
import torch

from src.weight.lowrank_reconstruction import _svd_for_ranks


class TestSVDForRanks:
    def test_returns_three_tensors(self):
        delta = torch.randn(8, 6)
        U, S, Vh = _svd_for_ranks(delta, ranks=[1, 2, 5])
        assert isinstance(U, torch.Tensor)
        assert isinstance(S, torch.Tensor)
        assert isinstance(Vh, torch.Tensor)

    def test_shapes_consistent(self):
        delta = torch.randn(8, 6)
        U, S, Vh = _svd_for_ranks(delta, ranks=[1, 2, 5])
        # U: (8, k), S: (k,), Vh: (k, 6)
        assert U.shape[0] == 8
        assert Vh.shape[1] == 6
        assert U.shape[1] == S.shape[0]
        assert S.shape[0] == Vh.shape[0]

    def test_rank1_reconstruction_perfect(self):
        """Rank-1 matrix should be perfectly reconstructed at rank 1."""
        u = torch.tensor([[1.0, 2.0, 3.0]]).T
        v = torch.tensor([[4.0, 5.0, 6.0, 7.0]])
        delta = u @ v  # (3, 4)
        U, S, Vh = _svd_for_ranks(delta, ranks=[1])
        recon = U[:, :1] @ torch.diag(S[:1]) @ Vh[:1, :]
        error = (delta - recon).norm().item()
        assert error < 1e-4

    def test_large_matrix_uses_lowrank(self):
        """Matrix with large min_dim should use svd_lowrank."""
        # min_dim = 6, max_rank = 5, 6 > 5*2+10=20 is False -> full SVD
        # min_dim = 50, max_rank = 5, 50 > 5*2+10=20 is True -> lowrank
        delta = torch.randn(100, 50)
        U, S, Vh = _svd_for_ranks(delta, ranks=[1, 2, 5])
        assert S.shape[0] >= 5  # at least max_rank singular values
