"""Tests for image-space metrics — pair mining, PSNR/SSIM, aggregation.

Pure-core tests (no model downloads): pair mining on synthetic manifest
entries, classical metric known-answers, summary/anchor helpers. Model-backed
meters (LPIPS/CLIP) are lazily imported and only smoke-tested when
ABL_TEST_DOWNLOAD_MODELS=1 (weights are ~2GB; CI stays offline).
"""

from __future__ import annotations

import math

import pytest
import torch

from src.text_encoder.image_metrics import (
    DEFAULT_FLOOR_PAIR,
    LPIPSMeter,
    PairScores,
    _gaussian_window,
    divergence_ratio,
    mine_cross_prompt_pairs,
    mine_floor_band_pairs,
    mine_seed_change_pairs,
    mine_te_swap_pairs,
    psnr,
    ssim,
)

# ---------------------------------------------------------------------------
# Synthetic manifest fixture
# ---------------------------------------------------------------------------


def _manifest(
    n_labels: int = 3,
    seeds: tuple[int, ...] = (42,),
    variants: tuple[str, ...] = ("base_int8", "base_fp8", "heretic"),
) -> list[dict]:
    """Synthetic manifest images: every (variant, label, seed, warm) present."""
    images = []
    for var in variants:
        for li in range(n_labels):
            for seed in seeds:
                images.append(
                    {
                        "variant": var,
                        "label": f"prompt_{li}",
                        "seed": seed,
                        "pass": "warm",
                        "path": f"img/{var}/prompt_{li}__seed{seed}.webp",
                    }
                )
    return images


# ---------------------------------------------------------------------------
# Pair mining
# ---------------------------------------------------------------------------


class TestMineTeSwap:
    def test_one_pair_per_variant_label(self) -> None:
        images = _manifest()
        pairs = mine_te_swap_pairs(images, reference="base_int8")
        # heretic only (base_fp8 is a different variant too)
        got = {(p.variant_a, p.label) for p in pairs}
        assert got == {
            ("base_fp8", "prompt_0"),
            ("base_fp8", "prompt_1"),
            ("base_fp8", "prompt_2"),
            ("heretic", "prompt_0"),
            ("heretic", "prompt_1"),
            ("heretic", "prompt_2"),
        }

    def test_reference_excluded(self) -> None:
        pairs = mine_te_swap_pairs(_manifest(), reference="base_int8")
        assert all(p.variant_a != "base_int8" for p in pairs)
        assert all(p.variant_b == "base_int8" for p in pairs)

    def test_seed_matched(self) -> None:
        images = _manifest(seeds=(42, 43))
        pairs = mine_te_swap_pairs(images, reference="base_int8", base_seed=42)
        assert pairs and all(p.seed_a == p.seed_b == 42 for p in pairs)

    def test_pass_filtered(self) -> None:
        images = _manifest()
        for img in images:
            if img["variant"] == "heretic":
                img["pass"] = "cold"
        pairs = mine_te_swap_pairs(images, reference="base_int8")
        assert all(p.variant_a != "heretic" for p in pairs)

    def test_missing_reference_skipped(self) -> None:
        images = [i for i in _manifest() if not (i["variant"] == "base_int8" and i["label"] == "prompt_1")]
        pairs = mine_te_swap_pairs(images, reference="base_int8")
        assert all(p.label != "prompt_1" for p in pairs)


class TestMineCrossPrompt:
    def test_within_variant_same_seed(self) -> None:
        pairs = mine_cross_prompt_pairs(_manifest(), per_variant=30)
        assert pairs
        assert all(p.variant_a == p.variant_b for p in pairs)
        assert all(p.seed_a == p.seed_b == 42 for p in pairs)
        # 3 labels -> 3 combos per variant, 3 variants
        assert len(pairs) == 9

    def test_subsample_caps_per_variant(self) -> None:
        images = _manifest(n_labels=12)  # 66 combos per variant
        pairs = mine_cross_prompt_pairs(images, per_variant=10)
        per_var = {}
        for p in pairs:
            per_var.setdefault(p.variant_a, 0)
            per_var[p.variant_a] += 1
        assert all(v == 10 for v in per_var.values())

    def test_subsample_deterministic(self) -> None:
        images = _manifest(n_labels=12)
        a = mine_cross_prompt_pairs(images, per_variant=10)
        b = mine_cross_prompt_pairs(images, per_variant=10)
        assert [(p.label, p.path_a) for p in a] == [(p.label, p.path_a) for p in b]


class TestMineSeedChange:
    def test_needs_multi_seed(self) -> None:
        assert mine_seed_change_pairs(_manifest(seeds=(42,))) == []

    def test_pairs_against_base_seed(self) -> None:
        images = _manifest(seeds=(42, 43, 44))
        pairs = mine_seed_change_pairs(images, base_seed=42)
        # 3 variants x 3 labels x 2 alt seeds
        assert len(pairs) == 18
        assert all(p.seed_b == 42 and p.seed_a in (43, 44) for p in pairs)
        assert all(p.variant_a == p.variant_b for p in pairs)


class TestMineFloorBand:
    def test_default_pair(self) -> None:
        pairs = mine_floor_band_pairs(_manifest())
        assert len(pairs) == 3  # 3 labels
        assert all(p.variant_a == DEFAULT_FLOOR_PAIR[0] and p.variant_b == DEFAULT_FLOOR_PAIR[1] for p in pairs)
        assert all(p.seed_a == p.seed_b == 42 for p in pairs)

    def test_missing_leg_skipped(self) -> None:
        images = [i for i in _manifest() if i["variant"] != "base_fp8"]
        assert mine_floor_band_pairs(images) == []


# ---------------------------------------------------------------------------
# PSNR / SSIM known answers
# ---------------------------------------------------------------------------


def _img(seed: int = 0, size: int = 64) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, size, size, generator=g)


class TestPSNR:
    def test_identical_is_inf(self) -> None:
        x = _img()
        assert psnr(x, x) == float("inf")

    def test_higher_for_closer(self) -> None:
        x = _img()
        near = (x + 0.01 * torch.randn_like(x)).clamp(0, 1)
        far = torch.rand_like(x)
        assert psnr(x, near) > psnr(x, far)

    def test_known_value(self) -> None:
        x = torch.zeros(1, 3, 8, 8)
        y = torch.full((1, 3, 8, 8), 0.1)  # mse = 0.01
        expected = 10.0 * math.log10(1.0 / 0.01)
        assert abs(psnr(x, y) - expected) < 1e-6

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            psnr(_img(), _img(size=32))


class TestSSIM:
    def test_identical_is_one(self) -> None:
        x = _img()
        assert ssim(x, x) == pytest.approx(1.0, abs=1e-6)

    def test_in_range_and_ordered(self) -> None:
        x = _img()
        near = (x + 0.02 * torch.randn_like(x)).clamp(0, 1)
        far = torch.rand_like(x)
        s_near, s_far = ssim(x, near), ssim(x, far)
        assert -1.0 <= s_far <= 1.0
        assert s_near > s_far

    def test_window_shape(self) -> None:
        w = _gaussian_window()
        assert w.shape == (3, 1, 11, 11)
        assert w[0].sum() == pytest.approx(1.0, abs=1e-6)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            ssim(_img(), _img(size=32))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestPairScores:
    def test_summarize_bootstrap_ci(self) -> None:
        g = torch.Generator().manual_seed(1)
        vals = torch.randn(50, generator=g).relu().tolist()
        scores = PairScores(kind="te_swap", lpips=vals, ssim_val=vals)
        summaries = {s.metric: s for s in scores.summarize(n_boot=200)}
        assert set(summaries) == {"lpips", "ssim"}
        s = summaries["lpips"]
        assert s.n == 50
        assert s.ci_low <= s.mean <= s.ci_high

    def test_empty_buckets_skipped(self) -> None:
        scores = PairScores(kind="seed_change", lpips=[0.1, 0.2])
        summaries = scores.summarize()
        assert [s.metric for s in summaries] == ["lpips"]

    def test_inf_values_filtered(self) -> None:
        scores = PairScores(kind="te_swap", psnr_db=[10.0, 20.0, float("inf")])
        summaries = scores.summarize(n_boot=100)
        assert summaries[0].n == 2

    def test_single_value_no_ci(self) -> None:
        scores = PairScores(kind="te_swap", lpips=[0.5])
        assert scores.summarize() == []


class TestDivergenceRatio:
    def test_simple_ratio(self) -> None:
        assert divergence_ratio(0.10, 0.40) == pytest.approx(0.25)

    def test_zero_anchor_none(self) -> None:
        assert divergence_ratio(0.10, 0.0) is None


# ---------------------------------------------------------------------------
# LPIPS/CLIP meters — offline interface tests + opt-in smoke
# ---------------------------------------------------------------------------


class TestMetersOffline:
    def test_lpips_lazy_no_import(self) -> None:
        meter = LPIPSMeter(net="alex", device="cpu")
        assert meter._fn is None  # no weights touched on construction

    def test_lpips_shape_mismatch(self) -> None:
        meter = LPIPSMeter()
        with pytest.raises(ValueError, match="same shape"):
            meter.score(torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 16, 16))


@pytest.mark.skipif(
    __import__("os").environ.get("ABL_TEST_DOWNLOAD_MODELS") != "1",
    reason="downloads ~2GB of weights; set ABL_TEST_DOWNLOAD_MODELS=1 to opt in",
)
class TestMetersOnline:
    def test_lpips_smoke(self) -> None:
        meter = LPIPSMeter(device="cpu")
        x = torch.rand(1, 3, 64, 64)
        identical = meter.score(x * 2 - 1, x * 2 - 1)
        assert identical < 0.01
        different = meter.score(x * 2 - 1, torch.rand(1, 3, 64, 64) * 2 - 1)
        assert different > identical

    def test_clip_smoke(self) -> None:
        from src.text_encoder.image_metrics import CLIPMeter

        meter = CLIPMeter(device="cpu")
        # A solid red image should score higher for "a red square" than "a blue ocean"
        red = torch.zeros(1, 3, 224, 224)
        red[:, 0, :, :] = 1.0
        s_red = meter.clip_score(red, "a photo of a red square")
        s_blue = meter.clip_score(red, "a photo of a blue ocean")
        assert s_red > s_blue
