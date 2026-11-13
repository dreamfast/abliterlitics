"""Tests for src/weight/lowrank_reconstruction.py — Low-rank reconstruction math."""

from __future__ import annotations

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


class TestLowrankRunAnalysis:
    """Audit deletions + emission: no total_orig_energy_*, shape_skipped emitted."""

    def test_report_has_no_orig_energy_and_emits_shape_skipped(self, tmp_path):
        import json

        from src.weight.lowrank_reconstruction import run_analysis
        from tests.conftest import create_mock_model
        from tests.test_weight._arch_extending import make_arch_extending_variant

        base_dir = create_mock_model(tmp_path / "base", arch="qwen3", seed=42)
        var_a = create_mock_model(tmp_path / "va", arch="qwen3", seed=43)
        var_b_ext = make_arch_extending_variant(
            create_mock_model(tmp_path / "vb", arch="qwen3", seed=44),
            tmp_path / "vb_ext",
            "model.layers.0.mlp.gate_proj.weight",
        )
        output = tmp_path / "lowrank.json"

        run_analysis(
            base_path=str(base_dir),
            variant_a_path=str(var_a),
            variant_b_path=str(var_b_ext),
            label_a="a",
            label_b="b",
            output_path=str(output),
            ranks=[1, 2],
        )

        text = output.read_text()
        data = json.loads(text)
        assert "total_orig_energy_a" not in text
        assert "total_orig_energy_b" not in text
        assert data["shape_skipped"] == 1
        assert data["ranks_tested"] == [1, 2]
