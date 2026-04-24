"""Tests for src/weight/expert_analysis.py — MoE expert analysis."""
from __future__ import annotations

import json

import pytest

from src.model_config import clear_shard_map_cache
from src.weight.expert_analysis import run_analysis
from tests.conftest import create_mock_model


@pytest.fixture(autouse=True)
def clear_cache():
    clear_shard_map_cache()
    yield
    clear_shard_map_cache()


class TestExpertAnalysis:
    def test_no_experts_reports_error(self, tmp_path):
        """Non-MoE model should produce error JSON."""
        base_dir = create_mock_model(tmp_path / "base", arch="qwen3", seed=42)
        var_dir = create_mock_model(tmp_path / "variant", arch="qwen3", seed=99)
        output = tmp_path / "expert_output.json"

        run_analysis(
            base_path=str(base_dir),
            variant_a_path=str(var_dir),
            variant_b_path=None,
            variant_c_path=None,
            label_a="test",
            label_b="b",
            label_c="c",
            output_path=str(output),
        )

        assert output.exists()
        data = json.loads(output.read_text())
        assert "error" in data

    def test_glm_model_with_experts(self, tmp_path):
        """GLM model with experts should produce valid analysis."""
        base_dir = create_mock_model(tmp_path / "base", arch="glm", seed=42)
        var_dir = create_mock_model(tmp_path / "variant", arch="glm", seed=99)
        output = tmp_path / "expert_output.json"

        run_analysis(
            base_path=str(base_dir),
            variant_a_path=str(var_dir),
            variant_b_path=None,
            variant_c_path=None,
            label_a="test",
            label_b="b",
            label_c="c",
            output_path=str(output),
        )

        assert output.exists()
        data = json.loads(output.read_text())
        assert data["results_version"] == 1
