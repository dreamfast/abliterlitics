"""Tests for CKA metrics — the correctness gate.

These known-answer tests validate that CKA has the mathematical properties
the methodology depends on:
- Identity: cka(X, X) = 1.0
- Orthogonal invariance: cka(X, X@R) ~ 1.0 for orthogonal R
- Isotropic scale invariance: cka(X, s*X) = 1.0
- Independence: cka(X, Y) ~ 0 for independent random X, Y (with large n)
- Symmetry: cka(X, Y) = cka(Y, X)
- Bounded: result in [0, 1]

Also tests padding invariance (same content, different pad lengths -> identical
CKA when properly masked).
"""

from __future__ import annotations

import pytest
import torch

from src.text_encoder.metrics import MetricResult, cka_linear, cka_rbf, pool_last, pool_mean

# ---------------------------------------------------------------------------
# MetricResult
# ---------------------------------------------------------------------------


class TestMetricResult:
    def test_no_ci(self) -> None:
        r = MetricResult(value=0.5)
        assert "0.500000" in repr(r)

    def test_with_ci(self) -> None:
        r = MetricResult(value=0.5, ci_low=0.3, ci_high=0.7)
        assert "0.500000" in repr(r)
        assert "0.300000" in repr(r)
        assert "0.700000" in repr(r)


# ---------------------------------------------------------------------------
# Pooling helpers
# ---------------------------------------------------------------------------


class TestPoolMean:
    def test_no_mask(self) -> None:
        """Mean pooling without mask = simple mean over seq dim."""
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # [1, 2, 2]
        result = pool_mean(x, None)
        assert torch.allclose(result, torch.tensor([[2.0, 3.0]]))

    def test_with_mask(self) -> None:
        """Mean pooling with mask excludes pad tokens."""
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]]])  # [1, 3, 2]
        mask = torch.tensor([[1, 1, 0]])  # third position is pad
        result = pool_mean(x, mask)
        assert torch.allclose(result, torch.tensor([[2.0, 3.0]]))

    def test_all_pad_clamped(self) -> None:
        """All-pad row should not produce NaN (clamped to min 1 count)."""
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        mask = torch.tensor([[0, 0]])
        result = pool_mean(x, mask)
        assert not torch.any(torch.isnan(result))


class TestPoolLast:
    def test_no_mask(self) -> None:
        """Last token without mask = last position."""
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])  # [1, 3, 2]
        result = pool_last(x, None)
        assert torch.allclose(result, torch.tensor([[5.0, 6.0]]))

    def test_with_mask(self) -> None:
        """Last NON-PAD token with mask."""
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]]])  # [1, 3, 2]
        mask = torch.tensor([[1, 1, 0]])  # third position is pad
        result = pool_last(x, mask)
        assert torch.allclose(result, torch.tensor([[3.0, 4.0]]))


# ---------------------------------------------------------------------------
# CKA linear — known-answer correctness gate
# ---------------------------------------------------------------------------


class TestCkaLinearIdentity:
    def test_identity_exact(self) -> None:
        """cka(X, X) should be exactly 1.0."""
        g = torch.Generator().manual_seed(42)
        x = torch.randn(50, 32, generator=g)
        assert cka_linear(x, x) == pytest.approx(1.0, abs=1e-6)


class TestCkaLinearOrthogonalInvariance:
    def test_orthogonal_rotation(self) -> None:
        """cka(X, X@R) ~ 1.0 for random orthogonal R."""
        g = torch.Generator().manual_seed(42)
        x = torch.randn(50, 32, generator=g)
        # Generate orthogonal matrix via QR
        a_mat = torch.randn(32, 32, generator=g)
        q_mat, _ = torch.linalg.qr(a_mat)
        x_rotated = x @ q_mat
        assert cka_linear(x, x_rotated) == pytest.approx(1.0, abs=1e-4)

    def test_different_orthogonal_rotation(self) -> None:
        """Same test with different seed and dimensions."""
        g = torch.Generator().manual_seed(99)
        x = torch.randn(100, 64, generator=g)
        a_mat = torch.randn(64, 64, generator=g)
        q_mat, _ = torch.linalg.qr(a_mat)
        assert cka_linear(x, x @ q_mat) == pytest.approx(1.0, abs=1e-4)


class TestCkaLinearScaleInvariance:
    def test_isotropic_scale(self) -> None:
        """cka(X, s*X) = 1.0 for any scalar s."""
        g = torch.Generator().manual_seed(42)
        x = torch.randn(50, 32, generator=g)
        for s in [0.001, 0.5, 3.14, 1000.0, -2.0]:
            assert cka_linear(x, s * x) == pytest.approx(1.0, abs=1e-4), f"Failed for s={s}"


class TestCkaLinearIndependence:
    def test_independent_random(self) -> None:
        """cka(X, Y) should be low for independent random X, Y with large n.

        Note: CKA has positive bias when n < d. With n=200 and d=32, the
        bias is small enough that the result should be well below 0.5.
        For higher dims, use more samples or relax the threshold.
        """
        g = torch.Generator().manual_seed(42)
        x = torch.randn(200, 32, generator=g)
        y = torch.randn(200, 32, generator=g)
        result = cka_linear(x, y)
        assert result < 0.2, f"Expected cka < 0.2 for independent data, got {result}"

    def test_independent_random_high_dim(self) -> None:
        """CKA with n << d has strong positive bias — verify it's still below identity.

        With n=50 and d=512, linear CKA of independent random data can be ~0.9
        due to subspace overlap (both random matrices span most of R^50 after
        centering). This is a known mathematical property, not a bug.
        The damage formula uses MSE (T1.4) which is not affected by this bias.
        """
        g = torch.Generator().manual_seed(42)
        x = torch.randn(50, 512, generator=g)
        y = torch.randn(50, 512, generator=g)
        result = cka_linear(x, y)
        # Still well below identity (1.0)
        assert result < 0.95, f"Expected cka < 0.95 for independent high-dim data, got {result}"


class TestCkaLinearSymmetry:
    def test_symmetric(self) -> None:
        """cka(X, Y) == cka(Y, X)."""
        g = torch.Generator().manual_seed(42)
        x = torch.randn(50, 32, generator=g)
        y = torch.randn(50, 48, generator=g)
        assert cka_linear(x, y) == pytest.approx(cka_linear(y, x), abs=1e-6)


class TestCkaLinearBounded:
    def test_bounded_0_1(self) -> None:
        """Result should be in [0, 1]."""
        g = torch.Generator().manual_seed(42)
        x = torch.randn(50, 32, generator=g)
        y = torch.randn(50, 32, generator=g)
        result = cka_linear(x, y)
        assert 0.0 <= result <= 1.0


class TestCkaLinearErrorHandling:
    def test_mismatched_n(self) -> None:
        x = torch.randn(50, 32)
        y = torch.randn(40, 32)
        with pytest.raises(ValueError, match="same n"):
            cka_linear(x, y)

    def test_too_few_samples(self) -> None:
        x = torch.randn(1, 32)
        y = torch.randn(1, 32)
        with pytest.raises(ValueError, match="n >= 2"):
            cka_linear(x, y)


class TestCkaLinearPaddingInvariance:
    """Same content, different pad lengths -> identical CKA when properly pooled+masked.

    This is THE critical mask test: if padding is not correctly excluded,
    the metric silently corrupts.
    """

    def test_padding_invariance(self) -> None:
        """Two identical-content tensors with different pad lengths should
        produce the same pooled CKA."""
        g = torch.Generator().manual_seed(42)
        # Base content: 5 real tokens per prompt, d=32
        real_content = torch.randn(20, 5, 32, generator=g)
        mask_5 = torch.ones(20, 5, dtype=torch.int)

        # Padded version: same real content + 10 pad tokens (zeros)
        padded = torch.zeros(20, 15, 32)
        padded[:, :5, :] = real_content
        mask_15 = torch.zeros(20, 15, dtype=torch.int)
        mask_15[:, :5] = 1

        # Pool both identically
        pooled_real = pool_mean(real_content, mask_5)
        pooled_padded = pool_mean(padded, mask_15)

        # CKA of each with itself (should be 1.0)
        assert cka_linear(pooled_real, pooled_padded) == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# CKA RBF
# ---------------------------------------------------------------------------


class TestCkaRbf:
    def test_identity(self) -> None:
        g = torch.Generator().manual_seed(42)
        x = torch.randn(50, 32, generator=g)
        assert cka_rbf(x, x) == pytest.approx(1.0, abs=1e-4)

    def test_identity_high_dim(self) -> None:
        g = torch.Generator().manual_seed(42)
        x = torch.randn(30, 128, generator=g)
        assert cka_rbf(x, x) == pytest.approx(1.0, abs=1e-4)

    def test_independent_low(self) -> None:
        g = torch.Generator().manual_seed(42)
        x = torch.randn(100, 32, generator=g)
        y = torch.randn(100, 32, generator=g)
        result = cka_rbf(x, y)
        # RBF CKA has slightly higher bias than linear on independent data
        assert result < 0.5, f"Expected cka_rbf < 0.5 for independent data, got {result}"

    def test_bounded(self) -> None:
        g = torch.Generator().manual_seed(42)
        x = torch.randn(50, 32, generator=g)
        y = torch.randn(50, 32, generator=g)
        result = cka_rbf(x, y)
        assert 0.0 <= result <= 1.0

    def test_sigma_determinism(self) -> None:
        """Same sigma should produce identical results."""
        g = torch.Generator().manual_seed(42)
        x = torch.randn(50, 32, generator=g)
        y = torch.randn(50, 32, generator=g)
        r1 = cka_rbf(x, y, sigma=1.0)
        r2 = cka_rbf(x, y, sigma=1.0)
        assert r1 == r2

    def test_too_few_samples(self) -> None:
        x = torch.randn(3, 32)
        y = torch.randn(3, 32)
        with pytest.raises(ValueError, match="n >= 4"):
            cka_rbf(x, y)

    def test_symmetric(self) -> None:
        """cka_rbf(X, Y) == cka_rbf(Y, X)."""
        g = torch.Generator().manual_seed(42)
        x = torch.randn(50, 32, generator=g)
        y = torch.randn(50, 48, generator=g)
        assert cka_rbf(x, y) == pytest.approx(cka_rbf(y, x), abs=1e-4)

    def test_padding_invariance(self) -> None:
        """Same content, different pad lengths -> identical RBF CKA when properly pooled."""
        g = torch.Generator().manual_seed(42)
        real_content = torch.randn(20, 5, 32, generator=g)
        mask_5 = torch.ones(20, 5, dtype=torch.int)
        padded = torch.zeros(20, 15, 32)
        padded[:, :5, :] = real_content
        mask_15 = torch.zeros(20, 15, dtype=torch.int)
        mask_15[:, :5] = 1
        pooled_real = pool_mean(real_content, mask_5)
        pooled_padded = pool_mean(padded, mask_15)
        assert cka_rbf(pooled_real, pooled_padded) == pytest.approx(1.0, abs=1e-4)


class TestCkaLinearVsRbf:
    def test_linear_rbf_close_for_linear_data(self) -> None:
        """For data with only linear structure, linear and RBF CKA should
        produce similar rankings."""
        g = torch.Generator().manual_seed(42)
        x = torch.randn(100, 32, generator=g)
        # Slight linear perturbation
        y = x + 0.1 * torch.randn(100, 32, generator=g)
        z = torch.randn(100, 32, generator=g)
        # Both metrics should rank (x,y) as more similar than (x,z)
        lin_xy = cka_linear(x, y)
        lin_xz = cka_linear(x, z)
        rbf_xy = cka_rbf(x, y)
        rbf_xz = cka_rbf(x, z)
        assert lin_xy > lin_xz
        assert rbf_xy > rbf_xz
