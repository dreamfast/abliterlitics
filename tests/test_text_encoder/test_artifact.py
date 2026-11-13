"""Tests for ConditioningArtifact, Stream, StreamSpec, Stage, Pooling.

These verify the data-type invariants that the metrics core relies on:
- fp32 cast on construction
- prompt/category alignment
- mask presence rules
- stream key consistency
"""

from __future__ import annotations

import pytest
import torch

from src.text_encoder.artifact import (
    ConditioningArtifact,
    Pooling,
    Stage,
    Stream,
    StreamSpec,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(
    name: str = "test_stream",
    dim: int = 16,
    stage: Stage = Stage.RAW_HIDDEN,
    pooling: Pooling = Pooling.NONE,
) -> StreamSpec:
    return StreamSpec(
        name=name,
        source_encoder="gemma3",
        layers=(0,),
        pooling=pooling,
        stage=stage,
        dim=dim,
    )


def _make_stream(
    spec: StreamSpec | None = None,
    activations: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    auto_mask: bool = True,
) -> Stream:
    if activations is None:
        if spec is None:
            spec = _make_spec()
        activations = torch.randn(5, 10, spec.dim)
    elif spec is None:
        # Infer dim from activations if no spec given
        dim = activations.shape[-1] if activations.ndim >= 2 else activations.shape[0]
        spec = _make_spec(dim=dim)
    # Auto-generate mask if seq_len > 1 and no explicit mask provided
    if auto_mask and mask is None and activations.ndim == 3 and activations.shape[1] > 1:
        mask = torch.ones(activations.shape[0], activations.shape[1], dtype=torch.int)
    return Stream(name=spec.name, spec=spec, activations=activations, mask=mask)


def _make_artifact(
    n: int = 5,
    streams: dict[str, Stream] | None = None,
    prompts: list[str] | None = None,
    categories: list[str] | None = None,
) -> ConditioningArtifact:
    if prompts is None:
        prompts = [f"prompt_{i}" for i in range(n)]
    if categories is None:
        categories = ["innocent"] * n
    actual_n = len(prompts)
    if streams is None:
        spec = _make_spec()
        acts = torch.randn(actual_n, 8, spec.dim)
        mask = torch.ones(actual_n, 8, dtype=torch.int)
        streams = {"test": Stream(name="test", spec=spec, activations=acts, mask=mask)}
    return ConditioningArtifact(
        model="test",
        backend="TestBackend",
        ckpt_id="/fake/ckpt",
        stage=Stage.RAW_HIDDEN,
        streams=streams,
        prompts=prompts,
        categories=categories,
        dtype="bf16",
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Stage / Pooling enums
# ---------------------------------------------------------------------------


class TestStageEnum:
    def test_stage_values(self) -> None:
        assert Stage.RAW_HIDDEN == "raw_hidden"
        assert Stage.PROJECTED == "projected"
        assert Stage.CONNECTOR_OUT == "connector_out"

    def test_stage_is_str(self) -> None:
        assert isinstance(Stage.RAW_HIDDEN, str)


class TestPoolingEnum:
    def test_pooling_values(self) -> None:
        assert Pooling.NONE == "none"
        assert Pooling.MEAN == "mean"


# ---------------------------------------------------------------------------
# StreamSpec
# ---------------------------------------------------------------------------


class TestStreamSpec:
    def test_frozen(self) -> None:
        spec = _make_spec()
        with pytest.raises(AttributeError):
            spec.name = "other"  # type: ignore[misc]

    def test_layers_tuple(self) -> None:
        spec = StreamSpec(
            name="multi",
            source_encoder="gemma3",
            layers=(0, 8, 16, 24, 48),
            pooling=Pooling.MEAN,
            stage=Stage.RAW_HIDDEN,
            dim=3840,
        )
        assert spec.layers == (0, 8, 16, 24, 48)

    def test_layers_str_all(self) -> None:
        spec = StreamSpec(
            name="all_layers",
            source_encoder="gemma3",
            layers="all",
            pooling=Pooling.MEAN,
            stage=Stage.RAW_HIDDEN,
            dim=3840,
        )
        assert spec.layers == "all"

    def test_layers_rejects_list(self) -> None:
        """layers must be tuple or str — list breaks hashability."""
        with pytest.raises(TypeError, match="layers must be"):
            StreamSpec(
                name="bad",
                source_encoder="gemma3",
                layers=[0, 8, 16],  # type: ignore[arg-type]
                pooling=Pooling.MEAN,
                stage=Stage.RAW_HIDDEN,
                dim=3840,
            )


# ---------------------------------------------------------------------------
# Stream invariants
# ---------------------------------------------------------------------------


class TestStream:
    def test_fp32_cast(self) -> None:
        """Activations in bf16/fp16 must be cast to fp32 on construction."""
        acts = torch.randn(3, 5, 8, dtype=torch.bfloat16)
        stream = _make_stream(activations=acts)
        assert stream.activations.dtype == torch.float32

    def test_fp16_cast(self) -> None:
        acts = torch.randn(3, 5, 8, dtype=torch.float16)
        stream = _make_stream(activations=acts)
        assert stream.activations.dtype == torch.float32

    def test_pooled_2d_expanded(self) -> None:
        """[n, d] input should be expanded to [n, 1, d]."""
        acts = torch.randn(3, 8)
        stream = _make_stream(activations=acts)
        assert stream.activations.shape == (3, 1, 8)

    def test_3d_accepted(self) -> None:
        acts = torch.randn(3, 5, 8)
        stream = _make_stream(activations=acts)
        assert stream.activations.shape == (3, 5, 8)

    def test_1d_rejected(self) -> None:
        acts = torch.randn(8)
        with pytest.raises(ValueError, match="must be .n, L, d. or .n, d."):
            _make_stream(activations=acts)

    def test_mask_required_when_seq_len_gt_1(self) -> None:
        acts = torch.randn(3, 5, 8)
        with pytest.raises(ValueError, match="mask is required"):
            _make_stream(activations=acts, mask=None, auto_mask=False)

    def test_mask_optional_when_seq_len_1(self) -> None:
        acts = torch.randn(3, 1, 8)
        stream = _make_stream(activations=acts, mask=None)
        assert stream.mask is None

    def test_mask_rows_must_match(self) -> None:
        acts = torch.randn(3, 5, 8)
        mask = torch.ones(4, 5, dtype=torch.int)  # wrong row count
        with pytest.raises(ValueError, match="mask rows"):
            _make_stream(activations=acts, mask=mask)

    def test_mask_cols_must_match(self) -> None:
        """Mask seq_len must match activations seq_len."""
        acts = torch.randn(3, 5, spec_dim := 8)
        spec = _make_spec(dim=spec_dim)
        mask = torch.ones(3, 99, dtype=torch.int)  # wrong col count
        with pytest.raises(ValueError, match="mask cols"):
            _make_stream(spec=spec, activations=acts, mask=mask)

    def test_mask_must_be_2d(self) -> None:
        acts = torch.randn(3, 5, spec_dim := 8)
        spec = _make_spec(dim=spec_dim)
        mask = torch.ones(3, 5, 1, dtype=torch.int)  # 3D mask
        with pytest.raises(ValueError, match="mask must be 2D"):
            _make_stream(spec=spec, activations=acts, mask=mask)

    def test_dim_mismatch_rejected(self) -> None:
        """Activations feature dim must match spec.dim."""
        spec = _make_spec(dim=16)
        acts = torch.randn(3, 5, 8)  # dim=8 but spec says 16
        with pytest.raises(ValueError, match="activations dim"):
            _make_stream(spec=spec, activations=acts)


# ---------------------------------------------------------------------------
# ConditioningArtifact invariants
# ---------------------------------------------------------------------------


class TestConditioningArtifact:
    def test_basic_construction(self) -> None:
        art = _make_artifact(n=5)
        assert art.n_prompts == 5
        assert len(art.categories) == 5

    def test_category_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="len.categories"):
            _make_artifact(
                n=5,
                prompts=["a", "b", "c", "d", "e"],
                categories=["innocent", "mild", "violent"],  # wrong length
            )

    def test_stream_row_count_mismatch(self) -> None:
        """Stream activations must have the same number of rows as prompts."""
        spec = _make_spec(dim=8)
        acts = torch.randn(3, 5, 8)  # 3 rows but we pass 5 prompts
        mask = torch.ones(3, 5, dtype=torch.int)
        stream = Stream(name="s", spec=spec, activations=acts, mask=mask)
        with pytest.raises(ValueError, match="3 rows but there are 5 prompts"):
            _make_artifact(n=5, streams={"s": stream})

    def test_stream_key_must_match_name(self) -> None:
        spec = _make_spec(name="stream_a")
        acts = torch.randn(2, 3, spec.dim)
        mask = torch.ones(2, 3, dtype=torch.int)
        stream = Stream(name="stream_a", spec=spec, activations=acts, mask=mask)
        with pytest.raises(ValueError, match="does not match stream.name"):
            _make_artifact(n=2, streams={"wrong_key": stream})

    def test_prompt_set_hash_deterministic(self) -> None:
        """Same prompts+categories should produce the same hash."""
        art1 = _make_artifact(prompts=["a", "b", "c"], categories=["x", "y", "z"])
        art2 = _make_artifact(prompts=["a", "b", "c"], categories=["x", "y", "z"])
        assert art1.prompt_set_hash == art2.prompt_set_hash

    def test_prompt_set_hash_order_independent(self) -> None:
        """Reordered prompts with same content should produce the same hash (sorted)."""
        art1 = _make_artifact(prompts=["a", "b", "c"], categories=["x", "y", "z"])
        art2 = _make_artifact(prompts=["c", "b", "a"], categories=["z", "y", "x"])
        assert art1.prompt_set_hash == art2.prompt_set_hash

    def test_assert_aligned_matching(self) -> None:
        art1 = _make_artifact(prompts=["a", "b"], categories=["x", "y"])
        art2 = _make_artifact(prompts=["a", "b"], categories=["x", "y"])
        art1.assert_aligned(art2)  # should not raise

    def test_assert_aligned_mismatched_prompts(self) -> None:
        art1 = _make_artifact(prompts=["a", "b"], categories=["x", "y"])
        art2 = _make_artifact(prompts=["a", "c"], categories=["x", "y"])
        with pytest.raises(ValueError, match="Prompt lists do not match"):
            art1.assert_aligned(art2)

    def test_assert_aligned_mismatched_categories(self) -> None:
        art1 = _make_artifact(prompts=["a", "b"], categories=["x", "y"])
        art2 = _make_artifact(prompts=["a", "b"], categories=["x", "z"])
        with pytest.raises(ValueError, match="Category lists do not match"):
            art1.assert_aligned(art2)

    def test_meta_defaults_empty(self) -> None:
        art = _make_artifact()
        assert art.meta == {}

    def test_meta_custom(self) -> None:
        art = _make_artifact()
        art.meta["tokenization_hash"] = "abc123"
        assert art.meta["tokenization_hash"] == "abc123"
