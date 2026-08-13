"""Tests for directional.py — ablation delta geometry."""

from __future__ import annotations

import pytest
import torch

from src.text_encoder.directional import (
    DEFAULT_TABOO_CATEGORIES,
    DirectionalReport,
    analyze_layer,
    taboo_direction,
)

CATS = ["innocent"] * 10 + ["violent"] * 5 + ["sexual"] * 5 + ["extreme"] * 5 + ["mild"] * 10 + ["multilingual"] * 10


def _make_base(n: int = 45, d: int = 64, seed: int = 0) -> torch.Tensor:
    """Activations with a real taboo structure: taboo prompts shifted."""
    g = torch.Generator().manual_seed(seed)
    acts = torch.randn(n, d, generator=g) * 0.1
    taboo_shift = torch.randn(d, generator=g)
    taboo_shift /= taboo_shift.norm()
    for i, c in enumerate(CATS):
        if c in DEFAULT_TABOO_CATEGORIES:
            acts[i] += taboo_shift * 2.0  # strong separation along one direction
    return acts


class TestTabooDirection:
    def test_recovers_planted_direction(self) -> None:
        base = _make_base()
        d = taboo_direction(base, CATS)
        # The planted shift IS the mean difference direction — up to sign/scale
        taboo_mask = torch.tensor([c in DEFAULT_TABOO_CATEGORIES for c in CATS])
        ref_mask = torch.tensor([c == "innocent" for c in CATS])
        planted = base[taboo_mask].mean(0) - base[ref_mask].mean(0)
        cos = torch.nn.functional.cosine_similarity(d.unsqueeze(0), planted.unsqueeze(0))
        assert cos.item() > 0.99
        assert abs(d.norm().item() - 1.0) < 1e-6

    def test_missing_category_raises(self) -> None:
        base = _make_base()
        with pytest.raises(ValueError, match="category groups missing"):
            taboo_direction(base, CATS, taboo=("nonexistent",))


class TestAnalyzeLayer:
    def test_directional_delta_scores_high(self) -> None:
        """Delta EXACTLY along −d_taboo on taboo prompts: |cos| ≈ 1, z ≫ 2."""
        base = _make_base()
        d = taboo_direction(base, CATS)
        var = base.clone()
        for i, c in enumerate(CATS):
            if c in DEFAULT_TABOO_CATEGORIES:
                var[i] -= d * 0.5  # move taboo AWAY from taboo direction
            else:
                var[i] += torch.randn_like(var[i]) * 0.01
        r = analyze_layer(base, var, CATS, layer=0, n_perm=100)
        assert r.cos_taboo_abs_mean > 0.9
        assert r.cos_z > 3.0
        assert r.orth_fraction < 0.05
        # Moving ALONG −d toward the innocent side: positive toward_innocent
        assert r.toward_innocent_mean > 0
        assert r.toward_innocent_z > 2.0

    def test_noise_delta_scores_null(self) -> None:
        """Isotropic random delta: |cos| small, z near 0, orth ≈ 1."""
        base = _make_base(seed=7)
        noise = torch.randn(base.shape, generator=torch.Generator().manual_seed(77))
        var = base + noise * 0.05
        r = analyze_layer(base, var, CATS, layer=0, n_perm=100)
        assert r.cos_taboo_abs_mean < 0.3
        assert abs(r.cos_z) < 3.0
        assert r.orth_fraction > 0.9
        assert abs(r.toward_innocent_z) < 3.0

    def test_shape_mismatch_raises(self) -> None:
        base = _make_base()
        with pytest.raises(ValueError, match="shape mismatch"):
            analyze_layer(base, base[:, :32], CATS, layer=0, n_perm=10)

    def test_delta_norms_split_by_category(self) -> None:
        """Taboo prompts get big deltas, innocent get small ones."""
        base = _make_base()
        var = base.clone()
        noise = torch.randn(base.shape, generator=torch.Generator().manual_seed(78))
        for i, c in enumerate(CATS):
            if c in DEFAULT_TABOO_CATEGORIES:
                var[i] += noise[i] * 0.1
        r = analyze_layer(base, var, CATS, layer=3, n_perm=50)
        assert r.delta_norm_taboo > 5 * r.delta_norm_ref


class TestReportSummary:
    def test_summary_fields(self) -> None:
        base = _make_base()
        noise = torch.randn(base.shape, generator=torch.Generator().manual_seed(79))
        var = base + noise * 0.05
        layers = [analyze_layer(base, var, CATS, layer=i, n_perm=50) for i in range(3)]
        rep = DirectionalReport(
            variant_id="v",
            reference_id="r",
            taboo_categories=DEFAULT_TABOO_CATEGORIES,
            reference_categories=("innocent",),
            layers=layers,
        )
        s = rep.summary()
        assert s["n_layers"] == 3
        assert 0 <= s["mean_orth_fraction"] <= 1
        assert "peak_z_layer" in s
