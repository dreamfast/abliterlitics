"""Tests for src/weight/subspace_alignment.py — Principal angles computation."""

from __future__ import annotations

import torch

from src.weight.subspace_alignment import _safe_device_for_stack, principal_angles


class TestPrincipalAngles:
    def test_identical_subspaces_angles_near_one(self):
        """Identical subspaces should have all principal angles ≈ 1.0."""
        mat_a = torch.randn(20, 3)
        mat_b = mat_a.clone()
        angles = principal_angles(mat_a, mat_b, top_k=3)
        assert len(angles) == 3
        for a in angles:
            assert a > 0.99

    def test_orthogonal_subspaces_angles_near_zero(self):
        """Orthogonal subspaces should have principal angles ≈ 0.0."""
        mat_a = torch.eye(6, 3)
        mat_b = torch.zeros(6, 3)
        mat_b[3:, :] = torch.eye(3)
        angles = principal_angles(mat_a, mat_b, top_k=3)
        for a in angles:
            assert a < 0.1

    def test_angles_bounded(self):
        """All angles should be in [0, 1]."""
        mat_a = torch.randn(20, 5)
        mat_b = torch.randn(20, 5)
        angles = principal_angles(mat_a, mat_b, top_k=5)
        for a in angles:
            assert 0.0 <= a <= 1.0

    def test_top_k_limits_output(self):
        """top_k should limit number of returned angles."""
        mat_a = torch.randn(20, 5)
        mat_b = torch.randn(20, 5)
        angles = principal_angles(mat_a, mat_b, top_k=2)
        assert len(angles) == 2


class TestSafeDeviceForStack:
    def test_small_vectors_returns_requested_device(self):
        vecs = [torch.randn(100) for _ in range(10)]
        dev = _safe_device_for_stack(vecs, torch.device("cpu"))
        assert dev == torch.device("cpu")

    def test_large_vectors_fallback_to_cpu(self):
        """Vectors that would exceed MAX_GPU_BYTES should fall back to CPU."""
        from unittest.mock import patch

        from src.weight.subspace_alignment import _safe_device_for_stack

        # Create small vectors, but mock MAX_GPU_BYTES to a tiny value
        # so the fallback logic triggers without needing huge allocations
        vecs = [torch.randn(100) for _ in range(10)]

        # Normal: should not trigger fallback
        dev = _safe_device_for_stack(vecs, torch.device("cpu"))
        assert dev == torch.device("cpu")

        # With tiny threshold: 100 * 10 * 4 = 4000 bytes, set threshold to 1000
        with patch("src.weight.subspace_alignment.MAX_GPU_BYTES", 1000):
            dev = _safe_device_for_stack(vecs, torch.device("cuda"))
            assert dev == torch.device("cpu")  # fallback triggered


class TestSubspaceShapeSkipped:
    """Lesson #79 observability: emitted shape_skipped on both report paths."""

    def test_normal_path_emits_shape_skipped(self, tmp_path):
        import json

        from src.weight.subspace_alignment import run_analysis
        from tests.conftest import create_mock_model
        from tests.test_weight._arch_extending import make_arch_extending_variant

        base_dir = create_mock_model(tmp_path / "base", arch="qwen3", seed=42)
        var_a = create_mock_model(tmp_path / "va", arch="qwen3", seed=43)
        var_b_ext = make_arch_extending_variant(
            create_mock_model(tmp_path / "vb", arch="qwen3", seed=44),
            tmp_path / "vb_ext",
            "model.layers.0.mlp.gate_proj.weight",
        )
        output = tmp_path / "subspace.json"

        run_analysis(
            base_path=str(base_dir),
            variant_a_path=str(var_a),
            variant_b_path=str(var_b_ext),
            label_a="a",
            label_b="b",
            output_path=str(output),
        )

        data = json.loads(output.read_text())
        assert "error" not in data
        assert data["shape_skipped"] == 1

    def test_insufficient_overlap_path_emits_shape_skipped(self, tmp_path):
        import json

        from src.weight.subspace_alignment import run_analysis
        from tests.conftest import create_mock_model

        base_dir = create_mock_model(tmp_path / "base", arch="qwen3", seed=42)
        output = tmp_path / "subspace.json"

        # variant identical to base -> zero both-changed -> error report
        run_analysis(
            base_path=str(base_dir),
            variant_a_path=str(base_dir),
            variant_b_path=str(base_dir),
            label_a="a",
            label_b="b",
            output_path=str(output),
        )

        data = json.loads(output.read_text())
        assert data["error"] == "insufficient overlap"
        assert data["shape_skipped"] == 0
