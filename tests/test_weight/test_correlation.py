"""Tests for src/weight/technique_correlation.py — Cosine similarity and projection math."""

from __future__ import annotations

import pytest
import torch

from src.weight.technique_correlation import cosine_sim, project_out


class TestCosineSim:
    def test_identical(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        assert cosine_sim(a, a) == pytest.approx(1.0, abs=1e-6)

    def test_opposite(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        assert cosine_sim(a, -a) == pytest.approx(-1.0, abs=1e-6)

    def test_orthogonal(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        assert cosine_sim(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_zero_vec(self):
        assert cosine_sim(torch.zeros(5), torch.ones(5)) == 0.0


class TestProjectOut:
    def test_removes_component(self):
        """After projecting out a direction, the residual should be orthogonal."""
        vec = torch.tensor([3.0, 4.0, 0.0])
        direction = torch.tensor([1.0, 0.0, 0.0])
        residual = project_out(vec, direction)
        # Residual should be orthogonal to direction
        dot = (residual.flatten() * direction.float().flatten()).sum()
        assert dot.abs().item() < 1e-6

    def test_preserves_orthogonal_component(self):
        """Projecting out a direction should preserve orthogonal parts."""
        vec = torch.tensor([3.0, 4.0])
        direction = torch.tensor([1.0, 0.0])
        residual = project_out(vec, direction)
        assert residual.flatten()[1].item() == pytest.approx(4.0, abs=1e-6)

    def test_zero_direction(self):
        """Zero direction should return original vector."""
        vec = torch.tensor([3.0, 4.0])
        direction = torch.zeros(2)
        residual = project_out(vec, direction)
        assert (residual - vec.float()).abs().max().item() < 1e-6


class TestCorrelationRunAnalysis:
    """Lesson #79 observability: emitted shape_skipped (counted via skeleton)."""

    def test_emits_shape_skipped(self, tmp_path):
        import json

        from src.weight.technique_correlation import run_analysis
        from tests.conftest import create_mock_model
        from tests.test_weight._arch_extending import make_arch_extending_variant

        base_dir = create_mock_model(tmp_path / "base", arch="qwen3", seed=42)
        var_a = create_mock_model(tmp_path / "va", arch="qwen3", seed=43)
        var_b_ext = make_arch_extending_variant(
            create_mock_model(tmp_path / "vb", arch="qwen3", seed=44),
            tmp_path / "vb_ext",
            "model.layers.0.mlp.gate_proj.weight",
        )
        output = tmp_path / "correlation.json"

        run_analysis(
            base_path=str(base_dir),
            variant_a_path=str(var_a),
            variant_b_path=str(var_b_ext),
            label_a="a",
            label_b="b",
            output_path=str(output),
        )

        data = json.loads(output.read_text())
        assert data["shape_skipped"] == 1
        assert data["total_keys"] == len(data["per_key_details"]) + 1
