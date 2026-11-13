"""Tests for activation cache.

Verifies:
- Round-trip save/load preserves artifact data
- Cache key is deterministic
- Distinct transforms produce distinct keys (C1 regression)
- Different prompts produce different keys
"""

from __future__ import annotations

import torch

from src.text_encoder.artifact import ConditioningArtifact, Pooling, Stage, Stream, StreamSpec
from src.text_encoder.cache import WILDCARD_DTYPE, cache_key, load_artifact, save_artifact


def _make_test_artifact(
    n: int = 5,
    ckpt_id: str = "/fake/base",
    dtype: str = "bf16",
    meta: dict | None = None,
) -> ConditioningArtifact:
    spec = StreamSpec(
        name="test",
        source_encoder="gemma3",
        layers=(0,),
        pooling=Pooling.MEAN,
        stage=Stage.RAW_HIDDEN,
        dim=8,
    )
    acts = torch.randn(n, 4, 8)
    mask = torch.ones(n, 4, dtype=torch.int)
    stream = Stream(name="test", spec=spec, activations=acts, mask=mask)
    return ConditioningArtifact(
        model="test",
        backend="TestBackend",
        ckpt_id=ckpt_id,
        stage=Stage.RAW_HIDDEN,
        streams={"test": stream},
        prompts=[f"p{i}" for i in range(n)],
        categories=["innocent"] * n,
        dtype=dtype,
        device="cpu",
        meta=meta or {},
    )


class TestCacheKey:
    def test_deterministic(self) -> None:
        k1 = cache_key("/ckpt", "hash123", "bf16", "cpu", "raw_hidden", "orig")
        k2 = cache_key("/ckpt", "hash123", "bf16", "cpu", "raw_hidden", "orig")
        assert k1 == k2

    def test_different_transforms_different_keys(self) -> None:
        """C1: distinct transforms must produce distinct keys."""
        k_orig = cache_key("/ckpt", "hash", "bf16", "cpu", "raw_hidden", "orig")
        k_fp16 = cache_key("/ckpt", "hash", "bf16", "cpu", "raw_hidden", "fp16_ref")
        k_fp8 = cache_key("/ckpt", "hash", "bf16", "cpu", "raw_hidden", "fakequant_fp8")
        k_int8 = cache_key("/ckpt", "hash", "bf16", "cpu", "raw_hidden", "fakequant_int8")
        keys = {k_orig, k_fp16, k_fp8, k_int8}
        assert len(keys) == 4, f"Transform keys must be distinct, got {keys}"

    def test_different_prompts_different_keys(self) -> None:
        k1 = cache_key("/ckpt", "hash1", "bf16", "cpu", "raw_hidden", "orig")
        k2 = cache_key("/ckpt", "hash2", "bf16", "cpu", "raw_hidden", "orig")
        assert k1 != k2

    def test_different_stages_different_keys(self) -> None:
        k1 = cache_key("/ckpt", "hash", "bf16", "cpu", "raw_hidden", "orig")
        k2 = cache_key("/ckpt", "hash", "bf16", "cpu", "projected", "orig")
        assert k1 != k2


class TestSaveLoad:
    def test_round_trip(self, tmp_path) -> None:
        """Save and load should preserve all data."""
        art = _make_test_artifact(meta={"transform": "orig"})
        save_artifact(art, tmp_path)
        # Compute key and load
        from src.text_encoder.cache import cache_key

        key = cache_key(art.ckpt_id, art.prompt_set_hash, WILDCARD_DTYPE, art.device, art.stage.value, "orig")
        loaded = load_artifact(tmp_path, key)
        assert loaded is not None
        assert loaded.ckpt_id == art.ckpt_id
        assert loaded.n_prompts == art.n_prompts
        assert "test" in loaded.streams
        assert torch.allclose(loaded.streams["test"].activations, art.streams["test"].activations)

    def test_load_missing_returns_none(self, tmp_path) -> None:
        loaded = load_artifact(tmp_path, "nonexistent_key")
        assert loaded is None

    def test_different_transforms_cached_separately(self, tmp_path) -> None:
        """C1 regression: orig vs fakequant must cache separately."""
        art_orig = _make_test_artifact(meta={"transform": "orig"})
        art_quant = _make_test_artifact(meta={"transform": "fakequant_fp8"})
        save_artifact(art_orig, tmp_path)
        save_artifact(art_quant, tmp_path)
        # Both should be in cache
        from src.text_encoder.cache import cache_key

        key_orig = cache_key(
            art_orig.ckpt_id,
            art_orig.prompt_set_hash,
            WILDCARD_DTYPE,
            art_orig.device,
            art_orig.stage.value,
            "orig",
        )
        key_quant = cache_key(
            art_quant.ckpt_id,
            art_quant.prompt_set_hash,
            WILDCARD_DTYPE,
            art_quant.device,
            art_quant.stage.value,
            "fakequant_fp8",
        )
        assert key_orig != key_quant
        assert load_artifact(tmp_path, key_orig) is not None
        assert load_artifact(tmp_path, key_quant) is not None
