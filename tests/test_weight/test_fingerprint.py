"""Tests for src/weight/technique_fingerprint.py — Layer entropy and fingerprint math."""

from __future__ import annotations

import pytest

from src.weight.technique_fingerprint import _layer_entropy


class TestLayerEntropy:
    def test_uniform_distribution_max_entropy(self):
        """Equal counts across all layers -> 100% normalized entropy."""
        counts = {0: 10, 1: 10, 2: 10, 3: 10}
        result = _layer_entropy(counts, total_layers=4)
        assert result == pytest.approx(100.0, abs=0.1)

    def test_single_layer_zero_entropy(self):
        """All changes in one layer -> 0% normalized entropy."""
        counts = {0: 100, 1: 0, 2: 0, 3: 0}
        result = _layer_entropy(counts, total_layers=4)
        assert result == pytest.approx(0.0, abs=0.1)

    def test_empty_counts(self):
        counts = {}
        result = _layer_entropy(counts, total_layers=4)
        assert result == 0.0

    def test_zero_total_layers(self):
        counts = {0: 5}
        result = _layer_entropy(counts, total_layers=0)
        assert result == 0.0

    def test_single_layer_total(self):
        """Single layer with all changes — entropy formula returns 0 when max_entropy=0."""
        counts = {0: 50}
        result = _layer_entropy(counts, total_layers=1)
        # With 1 layer, max_entropy = log2(1) = 0, so division guard returns 0
        assert result == 0.0


class TestRunAnalysisShapeSkipped:
    """Lesson #79 observability: scope.shape_skipped must be emitted."""

    def test_emits_shape_skipped(self, tmp_path):
        import json

        from src.weight.technique_fingerprint import run_analysis
        from tests.conftest import create_mock_model
        from tests.test_weight._arch_extending import make_arch_extending_variant

        base_dir = create_mock_model(tmp_path / "base", arch="qwen3", seed=42)
        var_dir = create_mock_model(tmp_path / "variant", arch="qwen3", seed=99)
        ext_dir = make_arch_extending_variant(var_dir, tmp_path / "ext", "model.layers.0.mlp.gate_proj.weight")
        output = tmp_path / "fingerprint.json"

        run_analysis(base_path=str(base_dir), variant_path=str(ext_dir), label="v", output_path=str(output))

        data = json.loads(output.read_text())
        assert data["scope"]["shape_skipped"] == 1
        assert len(data["per_key"]) == len(
            [e for e in data["per_key"] if e["key"] != "model.layers.0.mlp.gate_proj.weight"]
        )
