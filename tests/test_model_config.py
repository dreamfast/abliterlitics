"""Tests for src/model_config.py — Architecture detection, shard maps, tensor loading."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from src.model_config import (
    ArchitectureConfig,
    build_shard_map,
    clear_shard_map_cache,
    compute_changed_keys,
    detect_architecture,
    get_all_canonical_keys,
    load_tensor,
)
from tests.conftest import create_mock_model


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear shard map cache between tests."""
    clear_shard_map_cache()
    yield
    clear_shard_map_cache()


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

class TestDetectArchitecture:
    def test_qwen3_family(self, tmp_model_dir):
        cfg = detect_architecture(str(tmp_model_dir))
        assert cfg.family == "qwen3"
        assert cfg.layer_count == 1
        assert not cfg.has_experts
        assert not cfg.has_mamba
        assert cfg.total_keys > 0

    def test_qwen35_family(self, tmp_model_dir_qwen35):
        cfg = detect_architecture(str(tmp_model_dir_qwen35))
        assert cfg.family == "qwen35"
        assert cfg.has_mamba is False

    def test_glm_family_with_experts(self, tmp_model_dir_glm):
        cfg = detect_architecture(str(tmp_model_dir_glm))
        assert cfg.family == "glm"
        assert cfg.has_experts is True
        assert cfg.expert_count == 2
        assert cfg.has_shared_experts is True

    def test_qwen35_mamba_detected(self, tmp_path):
        model_dir = tmp_path / "qwen35_mamba"
        create_mock_model(model_dir, arch="qwen35_mamba", seed=42)
        cfg = detect_architecture(str(model_dir))
        assert cfg.family == "qwen35"
        assert cfg.has_mamba is True

    def test_gemma4_family(self, tmp_model_dir_gemma4):
        cfg = detect_architecture(str(tmp_model_dir_gemma4))
        assert cfg.family == "gemma4"
        assert cfg.layer_count == 1
        assert cfg.has_multimodal_towers is True
        assert not cfg.has_experts
        assert not cfg.has_mamba

    def test_gemma4_not_misdetected_as_qwen35(self, tmp_model_dir_gemma4):
        cfg = detect_architecture(str(tmp_model_dir_gemma4))
        assert cfg.family != "qwen35"

    def test_lm_keys_populated(self, tmp_model_dir):
        cfg = detect_architecture(str(tmp_model_dir))
        assert len(cfg.lm_keys) > 0


# ---------------------------------------------------------------------------
# ArchitectureConfig methods
# ---------------------------------------------------------------------------

class TestArchitectureConfig:
    @pytest.fixture
    def qwen3_cfg(self, tmp_model_dir):
        return detect_architecture(str(tmp_model_dir))

    @pytest.fixture
    def glm_cfg(self, tmp_model_dir_glm):
        return detect_architecture(str(tmp_model_dir_glm))

    @pytest.fixture
    def gemma4_cfg(self, tmp_model_dir_gemma4):
        return detect_architecture(str(tmp_model_dir_gemma4))

    def test_get_layer_index(self, qwen3_cfg):
        idx = qwen3_cfg.get_layer_index("model.layers.0.self_attn.q_proj.weight")
        assert idx == 0

    def test_get_layer_index_no_layer(self, qwen3_cfg):
        idx = qwen3_cfg.get_layer_index("model.embed_tokens.weight")
        assert idx is None

    def test_get_tensor_type_qwen3(self, qwen3_cfg):
        assert qwen3_cfg.get_tensor_type("model.layers.0.self_attn.q_proj.weight") == "self_attn.q_proj.weight"
        assert qwen3_cfg.get_tensor_type("model.embed_tokens.weight") == "embed_tokens.weight"

    def test_get_tensor_type_qwen35(self, tmp_model_dir_qwen35):
        cfg = detect_architecture(str(tmp_model_dir_qwen35))
        ttype = cfg.get_tensor_type("model.language_model.layers.0.self_attn.q_proj.weight")
        assert ttype == "self_attn.q_proj.weight"

    def test_get_expert_id(self, glm_cfg):
        assert glm_cfg.get_expert_id("model.layers.0.mlp.experts.1.gate_proj.weight") == 1

    def test_get_expert_id_non_expert(self, qwen3_cfg):
        assert qwen3_cfg.get_expert_id("model.layers.0.self_attn.q_proj.weight") is None

    def test_is_expert_tensor(self, glm_cfg):
        assert glm_cfg.is_expert_tensor("model.layers.0.mlp.experts.0.gate_proj.weight")
        assert not glm_cfg.is_expert_tensor("model.layers.0.self_attn.q_proj.weight")

    def test_is_shared_expert_tensor(self, glm_cfg):
        assert glm_cfg.is_shared_expert_tensor("model.layers.0.mlp.shared_expert.gate_proj.weight")
        assert not glm_cfg.is_shared_expert_tensor("model.layers.0.mlp.experts.0.gate_proj.weight")

    def test_tensor_category(self, qwen3_cfg):
        assert qwen3_cfg.tensor_category("model.layers.0.self_attn.q_proj.weight") == "attention"
        assert qwen3_cfg.tensor_category("model.embed_tokens.weight") == "embedding"
        assert qwen3_cfg.tensor_category("lm_head.weight") == "lm_head"
        assert qwen3_cfg.tensor_category("model.layers.0.input_layernorm.weight") == "norm"

    def test_tensor_category_experts(self, glm_cfg):
        assert glm_cfg.tensor_category("model.layers.0.mlp.experts.0.gate_proj.weight") == "expert"
        assert glm_cfg.tensor_category("model.layers.0.mlp.shared_expert.gate_proj.weight") == "shared_expert"

    def test_gemma4_is_lm_key_excludes_towers(self, gemma4_cfg):
        assert gemma4_cfg.is_lm_key(
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.self_attn.q_proj.weight",
        )
        assert not gemma4_cfg.is_lm_key(
            "model.audio_tower.layers.0.self_attn.q_proj.weight",
            "model.audio_tower.layers.0.self_attn.q_proj.weight",
        )
        assert not gemma4_cfg.is_lm_key(
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.weight",
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.weight",
        )
        assert not gemma4_cfg.is_lm_key(
            "model.embed_audio.embedding_projection.weight",
            "model.embed_audio.embedding_projection.weight",
        )

    def test_gemma4_get_tensor_type(self, gemma4_cfg):
        assert gemma4_cfg.get_tensor_type("model.language_model.layers.0.self_attn.q_proj.weight") == "self_attn.q_proj.weight"
        assert gemma4_cfg.get_tensor_type("model.language_model.layers.0.mlp.gate_proj.weight") == "mlp.gate_proj.weight"
        assert gemma4_cfg.get_tensor_type("model.language_model.layers.0.layer_scalar") == "layer_scalar"
        assert gemma4_cfg.get_tensor_type("model.language_model.layers.0.per_layer_projection.weight") == "per_layer_projection.weight"
        assert gemma4_cfg.get_tensor_type("model.language_model.embed_tokens.weight") == "embed_tokens.weight"
        assert gemma4_cfg.get_tensor_type("model.language_model.norm.weight") == "norm.weight"

    def test_gemma4_tensor_categories(self, gemma4_cfg):
        assert gemma4_cfg.tensor_category("model.language_model.layers.0.self_attn.q_proj.weight") == "attention"
        assert gemma4_cfg.tensor_category("model.language_model.layers.0.input_layernorm.weight") == "norm"
        assert gemma4_cfg.tensor_category("model.language_model.layers.0.post_per_layer_input_norm.weight") == "norm"
        assert gemma4_cfg.tensor_category("model.language_model.layers.0.pre_feedforward_layernorm.weight") == "norm"
        assert gemma4_cfg.tensor_category("model.language_model.layers.0.post_feedforward_layernorm.weight") == "norm"
        assert gemma4_cfg.tensor_category("model.language_model.layers.0.per_layer_projection.weight") == "per_layer"
        assert gemma4_cfg.tensor_category("model.language_model.layers.0.per_layer_input_gate.weight") == "per_layer"
        assert gemma4_cfg.tensor_category("model.language_model.layers.0.layer_scalar") == "scalar"
        assert gemma4_cfg.tensor_category("model.language_model.layers.0.mlp.gate_proj.weight") == "mlp"
        assert gemma4_cfg.tensor_category("model.language_model.embed_tokens.weight") == "embedding"

    def test_gemma4_is_norm_tensor(self, gemma4_cfg):
        assert gemma4_cfg.is_norm_tensor("model.language_model.layers.0.input_layernorm.weight")
        assert gemma4_cfg.is_norm_tensor("model.language_model.layers.0.post_attention_layernorm.weight")
        assert gemma4_cfg.is_norm_tensor("model.language_model.layers.0.pre_feedforward_layernorm.weight")
        assert gemma4_cfg.is_norm_tensor("model.language_model.layers.0.post_feedforward_layernorm.weight")
        assert gemma4_cfg.is_norm_tensor("model.language_model.layers.0.post_per_layer_input_norm.weight")
        assert gemma4_cfg.is_norm_tensor("model.language_model.layers.0.self_attn.q_norm.weight")
        assert gemma4_cfg.is_norm_tensor("model.language_model.layers.0.self_attn.k_norm.weight")
        assert gemma4_cfg.is_norm_tensor("model.language_model.norm.weight")

    def test_canonicalize_triple_prefix(self, tmp_path):
        cfg = ArchitectureConfig(family="qwen35", prefix="model.language_model.")
        raw = "model.language_model.language_model.language_model.layers.0.self_attn.q_proj.weight"
        result = cfg.canonicalize(raw)
        assert result == "model.language_model.layers.0.self_attn.q_proj.weight"

    def test_canonicalize_no_change(self, tmp_path):
        cfg = ArchitectureConfig(family="qwen3", prefix="model.")
        key = "model.layers.0.self_attn.q_proj.weight"
        assert cfg.canonicalize(key) == key


# ---------------------------------------------------------------------------
# Shard map
# ---------------------------------------------------------------------------

class TestShardMap:
    def test_build_shard_map_keys(self, tmp_model_dir):
        cfg = detect_architecture(str(tmp_model_dir))
        smap = build_shard_map(str(tmp_model_dir), cfg)
        assert len(smap) > 0
        # Each value is (raw_key, shard_file_path)
        for raw_key, shard_path in smap.values():
            assert isinstance(raw_key, str)
            assert isinstance(shard_path, str)
            assert Path(shard_path).exists()

    def test_shard_map_caching(self, tmp_model_dir):
        cfg = detect_architecture(str(tmp_model_dir))
        smap1 = build_shard_map(str(tmp_model_dir), cfg)
        smap2 = build_shard_map(str(tmp_model_dir), cfg)
        assert smap1 is smap2  # same object from cache

    def test_get_all_canonical_keys(self, tmp_model_dir):
        cfg = detect_architecture(str(tmp_model_dir))
        keys = get_all_canonical_keys(str(tmp_model_dir), cfg)
        assert len(keys) > 0
        assert keys == sorted(keys)

    def test_load_tensor(self, tmp_model_dir):
        cfg = detect_architecture(str(tmp_model_dir))
        smap = build_shard_map(str(tmp_model_dir), cfg)
        first_key = sorted(smap.keys())[0]
        tensor = load_tensor(smap, first_key)
        assert tensor is not None
        assert isinstance(tensor, torch.Tensor)
        assert tensor.numel() > 0

    def test_load_tensor_missing_key(self, tmp_model_dir):
        cfg = detect_architecture(str(tmp_model_dir))
        smap = build_shard_map(str(tmp_model_dir), cfg)
        result = load_tensor(smap, "nonexistent.key.weight")
        assert result is None


# ---------------------------------------------------------------------------
# compute_changed_keys
# ---------------------------------------------------------------------------

class TestComputeChangedKeys:
    def test_detects_changed_keys(self, tmp_path):
        """Two models with different seeds should have changed keys."""
        from tests.conftest import create_mock_model

        base_dir = create_mock_model(tmp_path / "base", arch="qwen3", seed=42)
        variant_dir = create_mock_model(tmp_path / "variant", arch="qwen3", seed=99)

        cfg = detect_architecture(str(base_dir))
        base_map = build_shard_map(str(base_dir), cfg)
        variant_map = build_shard_map(str(variant_dir), cfg)

        changed = compute_changed_keys(base_map, variant_map)
        assert len(changed) > 0
        for entry in changed:
            assert "canonical" in entry
            assert "mean_abs_diff" in entry
            assert entry["mean_abs_diff"] > 0

    def test_identical_models_no_changes(self, tmp_path):
        """Same seed = same weights = no changes."""
        from tests.conftest import create_mock_model

        base_dir = create_mock_model(tmp_path / "base", arch="qwen3", seed=42)
        copy_dir = create_mock_model(tmp_path / "copy", arch="qwen3", seed=42)

        cfg = detect_architecture(str(base_dir))
        base_map = build_shard_map(str(base_dir), cfg)
        copy_map = build_shard_map(str(copy_dir), cfg)

        changed = compute_changed_keys(base_map, copy_map)
        assert len(changed) == 0

    def test_gemma4_reduced_keys_intersection(self, tmp_path):
        """Full Gemma4 base vs reduced-key variant: shard map intersection works."""
        base_dir = create_mock_model(tmp_path / "base", arch="gemma4", seed=42)
        reduced_dir = create_mock_model(tmp_path / "reduced", arch="gemma4_reduced", seed=43)

        cfg = detect_architecture(str(base_dir))
        base_map = build_shard_map(str(base_dir), cfg)
        reduced_map = build_shard_map(str(reduced_dir), cfg)

        common_keys = set(base_map) & set(reduced_map)
        assert len(common_keys) > 0
        assert "model.language_model.layers.0.self_attn.q_proj.weight" in common_keys
        assert "model.language_model.layers.0.self_attn.k_proj.weight" not in common_keys
        assert "model.language_model.layers.0.layer_scalar" in common_keys

        changed = compute_changed_keys(base_map, reduced_map)
        assert len(changed) > 0
