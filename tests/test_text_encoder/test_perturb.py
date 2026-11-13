"""Tests for perturbation synthesizers — dose matching and invariants.

Known-answer tests for the control battery: MSE/norm matching within
tolerance, permutation properties of the shuffles, direction orthogonality,
alpha scaling exactness.
"""

from __future__ import annotations

import pytest
import torch

from src.text_encoder.perturb import (
    _match_mse,
    delta_scale,
    direction_injection,
    embedding_assignment_shuffle,
    gaussian_delta,
    null_conditioning,
    per_prompt_delta_norm,
    rank1_random_delta,
    token_order_shuffle,
)


def _cond(n: int = 4, length: int = 7, d: int = 16, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, length, d, generator=g)


def _mask(n: int = 4, length: int = 7, real: int = 5) -> torch.Tensor:
    m = torch.zeros(n, length, dtype=torch.int)
    m[:, :real] = 1
    return m


class TestMatchMse:
    def test_matches_per_prompt(self) -> None:
        delta = _cond()
        target = torch.tensor([0.01, 0.1, 1.0, 10.0])
        scaled = _match_mse(delta, target)
        got = (scaled**2).mean(dim=(1, 2))
        assert torch.allclose(got, target, rtol=1e-4)

    def test_preserves_direction(self) -> None:
        delta = _cond(seed=1)
        scaled = _match_mse(delta, torch.full((4,), 0.5))
        cos = torch.nn.functional.cosine_similarity(scaled.flatten(1), delta.flatten(1), dim=1)
        assert torch.allclose(cos, torch.ones(4), atol=1e-5)


class TestGaussianDelta:
    def test_dose_matched_scalar(self) -> None:
        base = _cond()
        delta = gaussian_delta(base, 0.25, seed=7)
        assert delta.shape == base.shape
        assert abs(float((delta**2).mean()) - 0.25) < 1e-4

    def test_dose_matched_per_prompt(self) -> None:
        base = _cond()
        target = torch.tensor([0.1, 0.2, 0.3, 0.4])
        delta = gaussian_delta(base, target, seed=7)
        got = (delta**2).mean(dim=(1, 2))
        assert torch.allclose(got, target, rtol=1e-3)

    def test_deterministic(self) -> None:
        base = _cond()
        a = gaussian_delta(base, 0.1, seed=7)
        b = gaussian_delta(base, 0.1, seed=7)
        assert torch.equal(a, b)


class TestRank1RandomDelta:
    def test_low_rank(self) -> None:
        base = _cond()
        delta = rank1_random_delta(base, 0.1, seed=11)
        # flatten per prompt -> rank 1 matrix
        mat = delta[0]
        rank = torch.linalg.matrix_rank(mat, tol=1e-4)
        assert int(rank) <= 1

    def test_dose_matched(self) -> None:
        base = _cond()
        delta = rank1_random_delta(base, 0.3, seed=11)
        assert abs(float((delta**2).mean()) - 0.3) < 1e-4

    def test_orthogonal_to_direction(self) -> None:
        base = _cond()
        d_taboo = torch.randn(16, generator=torch.Generator().manual_seed(5))
        delta = rank1_random_delta(base, 0.2, orthogonal_to=d_taboo, seed=11)
        # every row of the (rank-1) delta must be orthogonal to d_taboo
        dots = delta[0] @ d_taboo
        assert float(dots.abs().max()) < 1e-4

    def test_same_direction_all_prompts(self) -> None:
        base = _cond()
        delta = rank1_random_delta(base, 1.0, seed=11)
        cos = torch.nn.functional.cosine_similarity(delta[0, 0], delta[3, 5], dim=0)
        assert abs(float(cos) - 1.0) < 1e-5


class TestDirectionInjection:
    def test_norm_matched(self) -> None:
        base = _cond()
        direction = torch.randn(16, generator=torch.Generator().manual_seed(3))
        delta = direction_injection(base, direction, 2.5)
        assert delta.shape == base.shape
        # every token vector has norm == target and is parallel to direction
        unit = direction / direction.norm()
        norms = delta.norm(dim=-1)
        assert torch.allclose(norms, torch.full_like(norms, 2.5), atol=1e-5)
        cos = torch.nn.functional.cosine_similarity(delta, unit.view(1, 1, -1), dim=-1)
        assert torch.allclose(cos, torch.ones_like(cos), atol=1e-5)

    def test_dim_mismatch_raises(self) -> None:
        base = _cond(d=16)
        with pytest.raises(ValueError, match="direction dim"):
            direction_injection(base, torch.randn(8), 1.0)


class TestDeltaScale:
    def test_alpha_exact(self) -> None:
        base, var = _cond(seed=0), _cond(seed=1)
        d05 = delta_scale(base, var, 0.5)
        assert torch.allclose(d05, 0.5 * (var - base), atol=1e-6)
        assert torch.allclose(delta_scale(base, var, 1.0), var - base, atol=1e-6)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            delta_scale(_cond(), _cond(length=9), 1.0)


class TestShuffles:
    def test_token_order_permutation(self) -> None:
        base = _cond(seed=2)
        out = token_order_shuffle(base, seed=2)
        # multiset of token rows preserved per prompt
        for i in range(base.shape[0]):
            a = base[i].sum(dim=0)
            b = out[i].sum(dim=0)
            assert torch.allclose(a, b, atol=1e-5)

    def test_mask_preserves_pad(self) -> None:
        base = _cond(seed=2)
        m = _mask(real=5)
        out = token_order_shuffle(base, length_mask=m, seed=3)
        assert torch.equal(out[:, 5:], base[:, 5:])  # pad untouched
        for i in range(base.shape[0]):
            assert torch.allclose(base[i, :5].sum(dim=0), out[i, :5].sum(dim=0), atol=1e-5)

    def test_embedding_shuffle_changes_assignment(self) -> None:
        base = _cond(seed=2)
        out = embedding_assignment_shuffle(base, seed=2)
        # derangement: EVERY real position receives a different embedding
        pos_same = ((out - base).abs().amax(dim=-1) < 1e-6).float().mean()
        assert float(pos_same) == 0.0

    def test_embedding_shuffle_permutation(self) -> None:
        base = _cond(seed=2)
        out = embedding_assignment_shuffle(base, seed=2)
        # it is still a permutation-composition: multiset preserved
        for i in range(base.shape[0]):
            assert torch.allclose(base[i].sum(dim=0), out[i].sum(dim=0), atol=1e-4)

    def test_deterministic(self) -> None:
        base = _cond(seed=2)
        assert torch.equal(token_order_shuffle(base, seed=9), token_order_shuffle(base, seed=9))


class TestAnchors:
    def test_null(self) -> None:
        base = _cond()
        z = null_conditioning(base)
        assert z.shape == base.shape
        assert float(z.abs().sum()) == 0.0

    def test_delta_stats(self) -> None:
        base, var = _cond(seed=0), _cond(seed=1)
        norm = per_prompt_delta_norm(var - base)
        assert norm.shape == (4,)
        assert (norm > 0).all()
