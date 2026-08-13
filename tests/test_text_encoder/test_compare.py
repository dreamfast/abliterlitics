"""Tests for compare() + damage computation.

Verifies:
- Damage = 1.0 when variant == fake_quant (by construction)
- Damage ≈ 0 when variant == base
- Mismatched prompts rejected
- Zero-denominator guard
- Per-category breakdown works
"""

from __future__ import annotations

import pytest
import torch

from src.text_encoder.artifact import ConditioningArtifact, Pooling, Stage, Stream, StreamSpec
from src.text_encoder.compare import compare, compute_damage, compute_stream_metrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_artifact(
    activations: torch.Tensor,
    n: int = 10,
    categories: list[str] | None = None,
    stage: Stage = Stage.RAW_HIDDEN,
    ckpt_id: str = "/fake/base",
    dtype: str = "bf16",
) -> ConditioningArtifact:
    """Build a minimal ConditioningArtifact with one stream."""
    d = activations.shape[-1]
    spec = StreamSpec(
        name="test",
        source_encoder="gemma3",
        layers=(0,),
        pooling=Pooling.MEAN,
        stage=stage,
        dim=d,
    )
    mask = torch.ones(activations.shape[0], activations.shape[1], dtype=torch.int)
    stream = Stream(name="test", spec=spec, activations=activations, mask=mask)
    if categories is None:
        categories = ["innocent"] * n
    return ConditioningArtifact(
        model="test",
        backend="TestBackend",
        ckpt_id=ckpt_id,
        stage=stage,
        streams={"test": stream},
        prompts=[f"p{i}" for i in range(n)],
        categories=categories,
        dtype=dtype,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# compute_damage
# ---------------------------------------------------------------------------


class TestComputeDamage:
    def test_damage_one_when_equal(self) -> None:
        """Damage = 1.0 when MSE(base, variant) == MSE(base, fake_quant)."""
        assert compute_damage(mse_base_variant=0.05, mse_base_quant=0.05) == pytest.approx(1.0)

    def test_damage_zero_when_no_perturbation(self) -> None:
        """Damage ≈ 0 when variant == base (MSE=0)."""
        assert compute_damage(mse_base_variant=0.0, mse_base_quant=0.05) == pytest.approx(0.0)

    def test_damage_scaled(self) -> None:
        """Damage 2.0 when variant MSE is 2× the quant floor."""
        assert compute_damage(mse_base_variant=0.10, mse_base_quant=0.05) == pytest.approx(2.0)

    def test_zero_denominator_returns_inf(self) -> None:
        """Near-zero quant floor → inf with warning."""
        assert compute_damage(mse_base_variant=0.01, mse_base_quant=1e-20) == float("inf")


# ---------------------------------------------------------------------------
# compute_stream_metrics
# ---------------------------------------------------------------------------


class TestComputeStreamMetrics:
    def test_identical_inputs(self) -> None:
        """All similarity metrics should be ~1.0 for identical inputs."""
        g = torch.Generator().manual_seed(42)
        x = torch.randn(20, 5, 32, generator=g)
        mask = torch.ones(20, 5, dtype=torch.int)
        m = compute_stream_metrics(x, x, mask, mask)
        assert m["cka_linear"] == pytest.approx(1.0, abs=1e-4)
        assert m["cos_all"] == pytest.approx(1.0, abs=1e-6)
        assert m["mse"] == pytest.approx(0.0, abs=1e-10)
        assert m["rank_corr"] == pytest.approx(1.0, abs=1e-4)

    def test_different_inputs(self) -> None:
        """Different inputs → lower similarity."""
        g = torch.Generator().manual_seed(42)
        x = torch.randn(20, 5, 32, generator=g)
        y = torch.randn(20, 5, 32, generator=g)
        mask = torch.ones(20, 5, dtype=torch.int)
        m = compute_stream_metrics(x, y, mask, mask)
        # CKA has positive bias with n<d (n=20, d=32) — still well below identity
        assert m["cka_linear"] < 0.7
        assert m["mse"] > 0.0


# ---------------------------------------------------------------------------
# compare()
# ---------------------------------------------------------------------------


class TestCompare:
    def test_variant_equals_base_damage_zero(self) -> None:
        """When variant == base, damage should be ~0."""
        g = torch.Generator().manual_seed(42)
        x = torch.randn(10, 5, 16, generator=g)
        base = _make_artifact(x, ckpt_id="base")
        variant = _make_artifact(x.clone(), ckpt_id="variant")
        # Quant floor: slightly different
        quant = _make_artifact(x + 0.01 * torch.randn(10, 5, 16, generator=g), ckpt_id="quant")
        report = compare(base, variant, quant_floor=quant)
        assert len(report.streams) == 1
        sr = report.streams[0]
        assert sr.damage_quant_units is not None
        assert sr.damage_quant_units.value == pytest.approx(0.0, abs=1e-6)

    def test_variant_equals_quant_damage_one(self) -> None:
        """When variant == fake_quant, damage should be ~1.0."""
        g = torch.Generator().manual_seed(42)
        x = torch.randn(10, 5, 16, generator=g)
        noise = 0.05 * torch.randn(10, 5, 16, generator=g)
        base = _make_artifact(x)
        quant = _make_artifact(x + noise, ckpt_id="quant")
        variant = _make_artifact(x + noise.clone(), ckpt_id="variant")  # same as quant
        report = compare(base, variant, quant_floor=quant)
        sr = report.streams[0]
        assert sr.damage_quant_units is not None
        assert sr.damage_quant_units.value == pytest.approx(1.0, abs=1e-4)

    def test_no_quant_floor_no_damage(self) -> None:
        """Without quant_floor, damage_quant_units should be None."""
        g = torch.Generator().manual_seed(42)
        x = torch.randn(10, 5, 16, generator=g)
        base = _make_artifact(x)
        variant = _make_artifact(x + 0.1 * torch.randn(10, 5, 16, generator=g))
        report = compare(base, variant, quant_floor=None)
        sr = report.streams[0]
        assert sr.damage_quant_units is None

    def test_floor_base_decouples_floor_from_reference(self) -> None:
        """Matched-quant semantics: with reference=base_int8 and floor=base_int8,
        damage must be MSE(base_int8, variant) / MSE(base_bf16, base_int8) —
        NOT divided by MSE(base_int8, base_int8)=0 (which yields inf).
        """
        g = torch.Generator().manual_seed(42)
        base_bf16 = torch.randn(10, 5, 16, generator=g)
        base_int8_act = base_bf16 + 0.05 * torch.randn(10, 5, 16, generator=g)
        heretic_act = base_int8_act + 0.10 * torch.randn(10, 5, 16, generator=g)

        global_base = _make_artifact(base_bf16, ckpt_id="base_bf16")
        ref = _make_artifact(base_int8_act, ckpt_id="base_int8")  # comparison reference
        floor = _make_artifact(base_int8_act.clone(), ckpt_id="base_int8")  # same file as ref
        variant = _make_artifact(heretic_act, ckpt_id="heretic_int8")

        # Without floor_base: denominator = MSE(ref, floor) = 0 -> inf (the bug)
        report_bug = compare(ref, variant, quant_floor=floor)
        assert report_bug.streams[0].damage_quant_units is not None
        assert report_bug.streams[0].damage_quant_units.value == float("inf")

        # With floor_base=global base: finite damage isolating the abliteration
        report_fixed = compare(ref, variant, quant_floor=floor, floor_base=global_base)
        dmg = report_fixed.streams[0].damage_quant_units
        assert dmg is not None
        # Expected: MSE(ref,variant) / MSE(base,floor)
        from src.text_encoder.metrics import mse as mse_fn

        expected = mse_fn(base_int8_act, heretic_act) / mse_fn(base_bf16, base_int8_act)
        assert dmg.value == pytest.approx(expected, rel=1e-4)

    def test_rejects_mismatched_prompts(self) -> None:
        g = torch.Generator().manual_seed(42)
        x = torch.randn(10, 5, 16, generator=g)
        base = _make_artifact(x)
        variant = _make_artifact(
            x.clone(),
            n=10,
            categories=["mild"] * 10,  # different categories
        )
        with pytest.raises(ValueError, match="Category lists do not match"):
            compare(base, variant)

    def test_per_category_breakdown(self) -> None:
        """When categories vary, per_category dict should be populated."""
        g = torch.Generator().manual_seed(42)
        x = torch.randn(10, 5, 16, generator=g)
        cats = ["innocent"] * 5 + ["extreme"] * 5
        base = _make_artifact(x, categories=cats)
        variant = _make_artifact(x + 0.1 * torch.randn(10, 5, 16, generator=g), categories=cats)
        report = compare(base, variant)
        sr = report.streams[0]
        assert "innocent" in sr.per_category
        assert "extreme" in sr.per_category
        # Each category has CKA and MSE
        assert "cka_linear" in sr.per_category["innocent"]
        assert "mse" in sr.per_category["extreme"]

    def test_report_has_all_metrics(self) -> None:
        """StreamReport should contain all expected metric fields."""
        g = torch.Generator().manual_seed(42)
        x = torch.randn(10, 5, 16, generator=g)
        base = _make_artifact(x)
        variant = _make_artifact(x + 0.01 * torch.randn(10, 5, 16, generator=g))
        report = compare(base, variant)
        sr = report.streams[0]
        assert sr.cka_linear.value >= 0.0
        assert sr.cka_rbf.value >= 0.0
        assert sr.cos_all.value >= -1.0
        assert sr.cos_last.value >= -1.0
        assert sr.mse.value >= 0.0
        assert sr.l2.value >= 0.0
        assert sr.rank_corr.value >= -1.0


class TestSemanticCeiling:
    def test_identical_inputs_zero_semantic_damage(self) -> None:
        """Semantic damage ~0 for identical base and variant."""
        g = torch.Generator().manual_seed(42)
        x = torch.randn(20, 5, 16, generator=g)
        base = _make_artifact(x, n=20)
        variant = _make_artifact(x.clone(), n=20)
        report = compare(base, variant)
        sr = report.streams[0]
        assert sr.damage_semantic_units is not None
        assert sr.damage_semantic_units.value < 1e-8

    def test_semantic_ceiling_positive(self) -> None:
        """Semantic ceiling = mean pairwise distance between distinct prompts."""
        g = torch.Generator().manual_seed(42)
        x = torch.randn(20, 5, 16, generator=g)
        base = _make_artifact(x, n=20)
        variant = _make_artifact(x + 0.1 * torch.randn(20, 5, 16, generator=g), n=20)
        report = compare(base, variant)
        sr = report.streams[0]
        assert sr.semantic_ceiling_mse is not None
        assert sr.semantic_ceiling_mse > 0

    def test_semantic_damage_small_for_small_perturbation(self) -> None:
        """A 0.1-std perturbation should be a small fraction of a meaning change."""
        g = torch.Generator().manual_seed(42)
        x = torch.randn(40, 5, 16, generator=g)
        base = _make_artifact(x, n=40)
        variant = _make_artifact(x + 0.1 * torch.randn(40, 5, 16, generator=g), n=40)
        report = compare(base, variant)
        sr = report.streams[0]
        assert sr.damage_semantic_units is not None
        # 0.1-std noise on pooled vectors is ~1% of inter-prompt distance
        assert sr.damage_semantic_units.value < 0.1


class TestBootstrapCIs:
    def test_mse_ci_present(self) -> None:
        """compare() with default n_boot populates the MSE CI."""
        g = torch.Generator().manual_seed(42)
        x = torch.randn(20, 5, 16, generator=g)
        base = _make_artifact(x, n=20)
        variant = _make_artifact(x + 0.1 * torch.randn(20, 5, 16, generator=g), n=20)
        report = compare(base, variant)
        sr = report.streams[0]
        assert sr.mse.ci_low is not None
        assert sr.mse.ci_high is not None
        assert sr.mse.ci_low <= sr.mse.value <= sr.mse.ci_high

    def test_cos_all_ci_present(self) -> None:
        g = torch.Generator().manual_seed(42)
        x = torch.randn(20, 5, 16, generator=g)
        base = _make_artifact(x, n=20)
        variant = _make_artifact(x + 0.1 * torch.randn(20, 5, 16, generator=g), n=20)
        report = compare(base, variant)
        sr = report.streams[0]
        assert sr.cos_all.ci_low is not None
        assert sr.cos_all.ci_high is not None

    def test_damage_ci_joint_bootstrap(self) -> None:
        """Damage CI from joint ratio bootstrap contains the point estimate."""
        g = torch.Generator().manual_seed(42)
        x = torch.randn(20, 5, 16, generator=g)
        base = _make_artifact(x, n=20)
        quant = _make_artifact(x + 0.05 * torch.randn(20, 5, 16, generator=g), n=20, ckpt_id="quant")
        variant = _make_artifact(x + 0.10 * torch.randn(20, 5, 16, generator=g), n=20)
        report = compare(base, variant, quant_floor=quant)
        sr = report.streams[0]
        assert sr.damage_quant_units is not None
        assert sr.damage_quant_units.ci_low is not None
        assert sr.damage_quant_units.ci_high is not None
        assert sr.damage_quant_units.ci_low <= sr.damage_quant_units.value

    def test_n_boot_zero_disables_ci(self) -> None:
        g = torch.Generator().manual_seed(42)
        x = torch.randn(10, 5, 16, generator=g)
        base = _make_artifact(x, n=10)
        variant = _make_artifact(x + 0.1 * torch.randn(10, 5, 16, generator=g), n=10)
        report = compare(base, variant, n_boot=0)
        sr = report.streams[0]
        assert sr.mse.ci_low is None
        assert sr.mse.ci_high is None
