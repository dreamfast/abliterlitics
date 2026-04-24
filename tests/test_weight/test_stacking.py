"""Tests for src/weight/stacking_analysis.py — Math helper functions."""
from __future__ import annotations

import pytest
import torch

from src.weight.stacking_analysis import (
    cosine_sim,
    explained_variance,
    regression_slope,
    safe_mean,
    safe_median,
    safe_std,
)


class TestCosineSim:
    def test_identical(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        assert cosine_sim(a, a) == pytest.approx(1.0, abs=1e-6)

    def test_opposite(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        assert cosine_sim(a, -a) == pytest.approx(-1.0, abs=1e-6)


class TestRegressionSlope:
    def test_perfect_positive(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        y = 2.0 * x + 1.0
        assert regression_slope(x, y) == pytest.approx(2.0, abs=1e-4)

    def test_perfect_negative(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        y = -3.0 * x
        assert regression_slope(x, y) == pytest.approx(-3.0, abs=1e-4)

    def test_constant_x(self):
        x = torch.tensor([5.0, 5.0, 5.0])
        y = torch.tensor([1.0, 2.0, 3.0])
        assert regression_slope(x, y) == 0.0


class TestExplainedVariance:
    def test_perfect_linear(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 3.0 * x + 7.0
        assert explained_variance(x, y) == pytest.approx(1.0, abs=1e-4)

    def test_poor_fit(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([10.0, -5.0, 20.0])
        r2 = explained_variance(x, y)
        assert r2 < 0.5
