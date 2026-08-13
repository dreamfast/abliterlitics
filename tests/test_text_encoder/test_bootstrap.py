"""Tests for bootstrap confidence intervals.

Verifies:
- CI contains the point estimate
- CIs narrow with more prompts
- Deterministic with seed
- Ratio CI uses joint resampling (not independent num/den)
"""

from __future__ import annotations

import pytest
import torch

from src.text_encoder.bootstrap import bootstrap_ci, bootstrap_ratio_ci


class TestBootstrapCi:
    def test_ci_contains_point(self) -> None:
        g = torch.Generator().manual_seed(42)
        values = torch.randn(100, generator=g)
        point, lo, hi = bootstrap_ci(values, n_boot=500, seed=42)
        assert lo <= point <= hi

    def test_ci_narrows_with_more_prompts(self) -> None:
        """More data → tighter CI."""
        g = torch.Generator().manual_seed(42)
        small = torch.randn(10, generator=g)
        large = torch.randn(500, generator=g)
        _, lo_small, hi_small = bootstrap_ci(small, n_boot=500, seed=42)
        _, lo_large, hi_large = bootstrap_ci(large, n_boot=500, seed=42)
        assert (hi_large - lo_large) < (hi_small - lo_small)

    def test_deterministic_with_seed(self) -> None:
        g = torch.Generator().manual_seed(42)
        values = torch.randn(50, generator=g)
        r1 = bootstrap_ci(values, n_boot=200, seed=99)
        r2 = bootstrap_ci(values, n_boot=200, seed=99)
        assert r1 == r2

    def test_different_seed_different_result(self) -> None:
        g = torch.Generator().manual_seed(42)
        values = torch.randn(50, generator=g)
        r1 = bootstrap_ci(values, n_boot=200, seed=1)
        r2 = bootstrap_ci(values, n_boot=200, seed=2)
        # CIs should be slightly different (different resamples)
        assert r1[1] != r2[1] or r1[2] != r2[2]

    def test_too_few_values(self) -> None:
        """n < 2 should return NaN CIs."""
        values = torch.tensor([3.14])
        point, lo, hi = bootstrap_ci(values, n_boot=100)
        assert point == pytest.approx(3.14)
        assert lo != lo  # NaN check


class TestBootstrapRatioCi:
    def test_ratio_ci_contains_point(self) -> None:
        g = torch.Generator().manual_seed(42)
        num = torch.rand(100, generator=g) * 0.1  # MSE per-prompt
        den = torch.rand(100, generator=g) * 0.05
        point, lo, hi = bootstrap_ratio_ci(num, den, n_boot=500, seed=42)
        assert lo <= point <= hi or point == float("inf")

    def test_ratio_deterministic(self) -> None:
        g = torch.Generator().manual_seed(42)
        num = torch.rand(50, generator=g)
        den = torch.rand(50, generator=g) * 2
        r1 = bootstrap_ratio_ci(num, den, n_boot=200, seed=99)
        r2 = bootstrap_ratio_ci(num, den, n_boot=200, seed=99)
        assert r1 == r2

    def test_ratio_joint_resampling(self) -> None:
        """The ratio CI should differ from dividing independent CIs.

        If num and den are correlated, joint resampling captures that
        correlation; independent resampling would not.
        """
        g = torch.Generator().manual_seed(42)
        base = torch.randn(200, generator=g)
        # Correlated numerator and denominator
        num = base**2 + 0.01
        den = base**2 + 0.01
        point, lo, hi = bootstrap_ratio_ci(num, den, n_boot=1000, seed=42)
        # Ratio should be ~1.0 (num == den) with tight CI
        assert point == pytest.approx(1.0, abs=0.01)
        assert hi - lo < 0.5  # reasonably tight for correlated data
