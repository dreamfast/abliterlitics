"""Tests for src/weight/edit_vector_analysis.py — Pure math functions."""

from __future__ import annotations

import math

import pytest
import torch

from src.weight.edit_vector_analysis import (
    cosine_sim,
    linear_r2,
    null_cosine_expected,
    pearson_corr,
    safe_mean,
    safe_median,
    safe_std,
    summary,
)


class TestCosineSim:
    def test_identical_vectors(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        assert cosine_sim(a, a) == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([-1.0, -2.0, -3.0])
        assert cosine_sim(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        a = torch.tensor([1.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0])
        assert cosine_sim(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_zero_vector(self):
        a = torch.zeros(10)
        b = torch.ones(10)
        assert cosine_sim(a, b) == 0.0

    def test_2d_tensor(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert cosine_sim(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_result_bounded(self):
        a = torch.randn(100)
        b = torch.randn(100)
        result = cosine_sim(a, b)
        assert -1.0 <= result <= 1.0


class TestPearsonCorr:
    def test_perfect_positive(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        b = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
        assert pearson_corr(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_perfect_negative(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        b = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0])
        assert pearson_corr(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_no_correlation_constant(self):
        a = torch.tensor([1.0, 1.0, 1.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        # a is constant -> zero-centered a is zero -> norm=0 -> return 0
        assert pearson_corr(a, b) == 0.0

    def test_symmetric(self):
        a = torch.randn(50)
        b = torch.randn(50)
        assert pearson_corr(a, b) == pytest.approx(pearson_corr(b, a), abs=1e-6)


class TestLinearR2:
    def test_perfect_linear(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * x + 1.0
        assert linear_r2(x, y) == pytest.approx(1.0, abs=1e-4)

    def test_poor_fit(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y = torch.tensor([10.0, -5.0, 20.0, -10.0, 15.0])
        r2 = linear_r2(x, y)
        assert r2 < 0.5

    def test_single_element(self):
        x = torch.tensor([1.0])
        y = torch.tensor([2.0])
        assert linear_r2(x, y) == 0.0  # < 2 elements


class TestNullCosineExpected:
    def test_dimensions(self):
        result = null_cosine_expected(100)
        assert result["dimensionality"] == 100
        assert result["expected_mean"] == 0.0
        assert result["expected_std"] == pytest.approx(1.0 / math.sqrt(100), abs=1e-6)
        assert result["three_sigma"] == pytest.approx(3.0 / math.sqrt(100), abs=1e-6)

    def test_zero_dim(self):
        result = null_cosine_expected(0)
        assert result["expected_std"] == 0.0


class TestSummaryHelpers:
    def test_safe_mean(self):
        assert safe_mean([1.0, 2.0, 3.0]) == 2.0
        assert safe_mean([]) is None

    def test_safe_median_odd(self):
        assert safe_median([1.0, 3.0, 5.0]) == 3.0

    def test_safe_median_even(self):
        assert safe_median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_safe_median_empty(self):
        assert safe_median([]) is None

    def test_safe_std(self):
        vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        s = safe_std(vals)
        assert s is not None
        assert s == pytest.approx(2.0, abs=0.1)

    def test_safe_std_too_few(self):
        assert safe_std([]) is None
        assert safe_std([1.0]) is None

    def test_summary(self):
        s = summary([1.0, 2.0, 3.0, 4.0, 5.0])
        assert s["count"] == 5
        assert s["mean"] == 3.0
        assert s["min"] == 1.0
        assert s["max"] == 5.0

    def test_summary_empty(self):
        s = summary([])
        assert s["count"] == 0
        assert s["mean"] is None


class TestRunAnalysisShapeGuards:
    """Lesson #79: architecture-extending variants must be skipped + counted."""

    EXTENDED_KEY = "model.layers.0.mlp.gate_proj.weight"

    def test_single_variant_emits_shape_skipped(self, tmp_path):
        import json

        from src.weight.edit_vector_analysis import run_analysis
        from tests.conftest import create_mock_model
        from tests.test_weight._arch_extending import make_arch_extending_variant

        base_dir = create_mock_model(tmp_path / "base", arch="qwen3", seed=42)
        var_dir = create_mock_model(tmp_path / "variant", arch="qwen3", seed=99)
        ext_dir = make_arch_extending_variant(var_dir, tmp_path / "ext", self.EXTENDED_KEY)
        output = tmp_path / "edit_vector.json"

        run_analysis(
            base_path=str(base_dir),
            variant_a_path=str(ext_dir),
            variant_b_path=None,
            label_a="v",
            label_b="b",
            output_path=str(output),
        )

        data = json.loads(output.read_text())
        assert data["shape_skipped"] == 1
        assert data["total_keys"] == len(data["per_key_details"])
        assert all(r["key"] != self.EXTENDED_KEY for r in data["per_key_details"])

    def test_two_variant_shape_mismatch_skipped_not_crashed(self, tmp_path):
        """Regression: analyze_two_variants had no shape guard (audit bug #1)."""
        import json

        from src.weight.edit_vector_analysis import run_analysis
        from tests.conftest import create_mock_model
        from tests.test_weight._arch_extending import make_arch_extending_variant

        base_dir = create_mock_model(tmp_path / "base", arch="qwen3", seed=42)
        var_a = create_mock_model(tmp_path / "va", arch="qwen3", seed=43)
        var_b_ext = make_arch_extending_variant(
            create_mock_model(tmp_path / "vb", arch="qwen3", seed=44),
            tmp_path / "vb_ext",
            self.EXTENDED_KEY,
        )
        output = tmp_path / "edit_vector_ab.json"

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
        assert "overlap_count" in data
        assert data["total_keys"] == len(data["per_key_details"])
        assert all(r["key"] != self.EXTENDED_KEY for r in data["per_key_details"])
