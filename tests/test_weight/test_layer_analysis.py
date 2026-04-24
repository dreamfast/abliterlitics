"""Tests for src/weight/layer_analysis.py — Output structure validation (integration)."""
from __future__ import annotations

import json

import pytest

from src.model_config import clear_shard_map_cache
from src.weight.layer_analysis import run_analysis
from tests.conftest import create_mock_model


@pytest.fixture(autouse=True)
def clear_cache():
    clear_shard_map_cache()
    yield
    clear_shard_map_cache()


class TestLayerAnalysis:
    def test_produces_valid_json_with_results_version(self, tmp_path):
        base_dir = create_mock_model(tmp_path / "base", arch="qwen3", seed=42)
        var_dir = create_mock_model(tmp_path / "variant", arch="qwen3", seed=99)
        output = tmp_path / "layer_output.json"

        run_analysis(
            base_path=str(base_dir),
            variant_path=str(var_dir),
            variant_b_path=None,
            label="test_variant",
            label_b=None,
            output_path=str(output),
        )

        assert output.exists()
        data = json.loads(output.read_text())
        assert data["results_version"] == 1
        # The actual output key is "layer_progression" not "layer_analysis"
        assert "layer_progression" in data
