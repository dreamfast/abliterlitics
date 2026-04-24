"""Tests for src/weight/edit_vector_analysis.py — Pure math functions."""
from __future__ import annotations

import math

import pytest
import torch

from src.weight.edit_vector_analysis import (
    TRIVIAL_THRESHOLD,
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
