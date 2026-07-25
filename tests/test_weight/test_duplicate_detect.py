"""Tests for src/weight/duplicate_detect.py — CPU-only SHA256 fingerprinting."""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from src.weight.duplicate_detect import (
    _hash_tensor,
    detect_family,
    fingerprint_model,
    group_duplicates,
    scan_models,
)
from tests.conftest import create_mock_model


class TestHashTensor:
    def test_identical_tensors_hash_equal(self):
        a = torch.randn(16, 8, generator=torch.Generator().manual_seed(1))
        assert _hash_tensor(a) == _hash_tensor(a.clone())

    def test_different_values_hash_differently(self):
        a = torch.randn(16, 8, generator=torch.Generator().manual_seed(1))
        b = torch.randn(16, 8, generator=torch.Generator().manual_seed(2))
        assert _hash_tensor(a) != _hash_tensor(b)

    def test_scalar_tensor_does_not_raise(self):
        # 0-dim tensors must be handled (Gemma4 has layer_scalar scalars).
        h = _hash_tensor(torch.tensor(3.14))
        assert isinstance(h, str) and len(h) == 64

    def test_dtype_independent_raw_bytes(self):
        # A float32 tensor and its byte-identical int32 view hash the same,
        # proving the hash is over raw bytes, not the typed values.
        f32 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        assert _hash_tensor(f32) == _hash_tensor(f32.view(torch.int32))


class TestDetectFamily:
    def test_gemma4(self):
        assert detect_family({"model.language_model.layers.0.layer_scalar"}) == "gemma4"
        assert detect_family({"model.language_model.layers.0.per_layer_projection.weight"}) == "gemma4"

    def test_qwen35(self):
        assert detect_family({"model.language_model.embed_tokens.weight"}) == "qwen3.5"

    def test_glm(self):
        assert detect_family({"model.layers.0.mlp.experts.0.weight"}) == "glm"

    def test_qwen3_default(self):
        assert detect_family({"model.layers.0.self_attn.q_proj.weight"}) == "qwen3"


class TestFingerprintModel:
    def test_identical_seeds_produce_identical_sha256(self, tmp_path):
        a = create_mock_model(tmp_path / "a", arch="gemma4", seed=42)
        b = create_mock_model(tmp_path / "b", arch="gemma4", seed=42)
        assert fingerprint_model(a).sha256 == fingerprint_model(b).sha256

    def test_different_seeds_produce_different_sha256(self, tmp_path):
        a = create_mock_model(tmp_path / "a", arch="gemma4", seed=42)
        b = create_mock_model(tmp_path / "b", arch="gemma4", seed=43)
        assert fingerprint_model(a).sha256 != fingerprint_model(b).sha256

    def test_sharding_independent_sha256(self, tmp_path):
        # The core value: splitting the same weights across different shard
        # boundaries must NOT change the SHA256.
        single = create_mock_model(tmp_path / "single", arch="qwen3", seed=42)
        fp_single = fingerprint_model(single)

        from src.model_config import build_shard_map, detect_architecture, load_tensor

        cfg = detect_architecture(str(single))
        smap = build_shard_map(str(single), cfg)
        keys = sorted(smap)
        half = len(keys) // 2

        multi_dir = tmp_path / "multi"
        multi_dir.mkdir()
        save_file(
            {ck: load_tensor(smap, ck) for ck in keys[:half]},
            str(multi_dir / "model-00001-of-00002.safetensors"),
        )
        save_file(
            {ck: load_tensor(smap, ck) for ck in keys[half:]},
            str(multi_dir / "model-00002-of-00002.safetensors"),
        )
        (multi_dir / "config.json").write_text((single / "config.json").read_text())

        assert fingerprint_model(multi_dir).sha256 == fp_single.sha256

    def test_raises_when_no_safetensors(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            fingerprint_model(empty)

    def test_counts_and_bytes(self, tmp_path):
        d = create_mock_model(tmp_path / "m", arch="qwen3", seed=1)
        fp = fingerprint_model(d)
        assert fp.tensor_count > 0
        assert fp.total_bytes > 0
        assert fp.architecture == "qwen3"
        assert len(fp.sha256) == 64


class TestGroupDuplicates:
    def test_identical_models_grouped(self, tmp_path):
        models = [create_mock_model(tmp_path / f"m{i}", arch="qwen3", seed=42) for i in range(3)]
        fps = [fingerprint_model(m, name=f"m{i}") for i, m in enumerate(models)]
        report = group_duplicates(fps)
        assert report["duplicate_group_count"] == 1
        assert report["duplicates"] == [["m0", "m1", "m2"]]

    def test_no_duplicates(self, tmp_path):
        a = create_mock_model(tmp_path / "a", arch="qwen3", seed=1)
        b = create_mock_model(tmp_path / "b", arch="qwen3", seed=2)
        fps = [fingerprint_model(a, name="a"), fingerprint_model(b, name="b")]
        report = group_duplicates(fps)
        assert report["duplicate_group_count"] == 0
        assert report["duplicates"] == []


class TestScanModels:
    def test_scans_all_children_with_safetensors(self, tmp_path):
        models_root = tmp_path / "models"
        create_mock_model(models_root / "a", arch="qwen3", seed=1)
        create_mock_model(models_root / "b", arch="qwen3", seed=1)  # duplicate of a
        (models_root / "empty").mkdir()  # no safetensors -> skipped
        fps = scan_models(models_root)
        assert [fp.name for fp in fps] == ["a", "b"]
        assert group_duplicates(fps)["duplicate_group_count"] == 1

    def test_subset_filter(self, tmp_path):
        models_root = tmp_path / "models"
        create_mock_model(models_root / "a", arch="qwen3", seed=1)
        create_mock_model(models_root / "b", arch="qwen3", seed=2)
        create_mock_model(models_root / "c", arch="qwen3", seed=3)
        fps = scan_models(models_root, subset=["a", "c"])
        assert sorted(fp.name for fp in fps) == ["a", "c"]
