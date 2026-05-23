"""
Shared test fixtures for abliterlitics test suite.

Creates temporary model directories with mock safetensors and config.json
for testing architecture detection, config loading, and weight analysis.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


# ---------------------------------------------------------------------------
# Minimal safetensors data generators
# ---------------------------------------------------------------------------

def _qwen35_keys() -> dict[str, torch.Tensor]:
    """Tiny Qwen3.5-style keys (language_model prefix, no experts)."""
    return {
        "model.language_model.embed_tokens.weight": torch.randn(32, 8),
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8),
        "model.language_model.layers.0.self_attn.k_proj.weight": torch.randn(8, 8),
        "model.language_model.layers.0.self_attn.v_proj.weight": torch.randn(8, 8),
        "model.language_model.layers.0.self_attn.o_proj.weight": torch.randn(8, 8),
        "model.language_model.layers.0.mlp.gate_proj.weight": torch.randn(16, 8),
        "model.language_model.layers.0.mlp.up_proj.weight": torch.randn(16, 8),
        "model.language_model.layers.0.mlp.down_proj.weight": torch.randn(8, 16),
        "model.language_model.layers.0.input_layernorm.weight": torch.randn(8),
        "model.language_model.layers.0.post_attention_layernorm.weight": torch.randn(8),
        "model.language_model.norm.weight": torch.randn(8),
        "lm_head.weight": torch.randn(32, 8),
    }


def _qwen35_mamba_keys() -> dict[str, torch.Tensor]:
    """Qwen3.5-style keys with Mamba (linear_attn + conv1d)."""
    keys = _qwen35_keys()
    keys["model.language_model.layers.0.linear_attn.weight"] = torch.randn(8, 8)
    keys["model.language_model.layers.0.conv1d.weight"] = torch.randn(8, 4)
    return keys


def _qwen3_keys() -> dict[str, torch.Tensor]:
    """Tiny Qwen3-style keys (standard model.layers, no language_model prefix)."""
    return {
        "model.embed_tokens.weight": torch.randn(32, 8),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(8, 8),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(8, 8),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(8, 8),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(16, 8),
        "model.layers.0.mlp.up_proj.weight": torch.randn(16, 8),
        "model.layers.0.mlp.down_proj.weight": torch.randn(8, 16),
        "model.layers.0.input_layernorm.weight": torch.randn(8),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(8),
        "model.norm.weight": torch.randn(8),
        "lm_head.weight": torch.randn(32, 8),
    }


def _glm_keys() -> dict[str, torch.Tensor]:
    """GLM-style keys with experts."""
    return {
        "model.embed_tokens.weight": torch.randn(32, 8),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(8, 8),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(8, 8),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(8, 8),
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(16, 8),
        "model.layers.0.mlp.experts.0.up_proj.weight": torch.randn(16, 8),
        "model.layers.0.mlp.experts.0.down_proj.weight": torch.randn(8, 16),
        "model.layers.0.mlp.experts.1.gate_proj.weight": torch.randn(16, 8),
        "model.layers.0.mlp.experts.1.up_proj.weight": torch.randn(16, 8),
        "model.layers.0.mlp.experts.1.down_proj.weight": torch.randn(8, 16),
        "model.layers.0.mlp.shared_expert.gate_proj.weight": torch.randn(16, 8),
        "model.layers.0.input_layernorm.weight": torch.randn(8),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(8),
        "model.norm.weight": torch.randn(8),
        "lm_head.weight": torch.randn(32, 8),
    }


def _gemma4_keys() -> dict[str, torch.Tensor]:
    """Gemma4-E2B-style keys with language_model prefix, audio/vision towers, per-layer projections."""
    return {
        "model.language_model.embed_tokens.weight": torch.randn(32, 8),
        "model.language_model.embed_tokens_per_layer.weight": torch.randn(32, 24),
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8),
        "model.language_model.layers.0.self_attn.k_proj.weight": torch.randn(4, 8),
        "model.language_model.layers.0.self_attn.v_proj.weight": torch.randn(4, 8),
        "model.language_model.layers.0.self_attn.o_proj.weight": torch.randn(8, 8),
        "model.language_model.layers.0.self_attn.q_norm.weight": torch.randn(4),
        "model.language_model.layers.0.self_attn.k_norm.weight": torch.randn(4),
        "model.language_model.layers.0.mlp.gate_proj.weight": torch.randn(16, 8),
        "model.language_model.layers.0.mlp.up_proj.weight": torch.randn(16, 8),
        "model.language_model.layers.0.mlp.down_proj.weight": torch.randn(8, 16),
        "model.language_model.layers.0.input_layernorm.weight": torch.randn(8),
        "model.language_model.layers.0.post_attention_layernorm.weight": torch.randn(8),
        "model.language_model.layers.0.pre_feedforward_layernorm.weight": torch.randn(8),
        "model.language_model.layers.0.post_feedforward_layernorm.weight": torch.randn(8),
        "model.language_model.layers.0.post_per_layer_input_norm.weight": torch.randn(8),
        "model.language_model.layers.0.per_layer_projection.weight": torch.randn(8, 4),
        "model.language_model.layers.0.per_layer_input_gate.weight": torch.randn(4, 8),
        "model.language_model.layers.0.layer_scalar": torch.randn(1),
        "model.language_model.norm.weight": torch.randn(8),
        "model.language_model.per_layer_model_projection.weight": torch.randn(24, 8),
        "model.language_model.per_layer_projection_norm.weight": torch.randn(4),
        "model.audio_tower.layers.0.self_attn.q_proj.weight": torch.randn(4, 4),
        "model.vision_tower.encoder.layers.0.self_attn.q_proj.weight": torch.randn(4, 4),
        "model.embed_audio.embedding_projection.weight": torch.randn(8, 8),
        "model.embed_vision.embedding_projection.weight": torch.randn(8, 4),
    }


def _gemma4_keys_reduced() -> dict[str, torch.Tensor]:
    """Gemma4-E2B reduced-key variant (missing k_proj/k_norm/v_proj for shared-KV layers)."""
    keys = _gemma4_keys()
    del keys["model.language_model.layers.0.self_attn.k_proj.weight"]
    del keys["model.language_model.layers.0.self_attn.k_norm.weight"]
    del keys["model.language_model.layers.0.self_attn.v_proj.weight"]
    return keys


KEY_GENERATORS = {
    "qwen35": _qwen35_keys,
    "qwen35_mamba": _qwen35_mamba_keys,
    "qwen3": _qwen3_keys,
    "glm": _glm_keys,
    "gemma4": _gemma4_keys,
    "gemma4_reduced": _gemma4_keys_reduced,
}

CONFIG_TEMPLATES = {
    "qwen35": {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": 16,
        "vocab_size": 32,
    },
    "qwen35_mamba": {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": 16,
        "vocab_size": 32,
    },
    "qwen3": {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": 16,
        "vocab_size": 32,
    },
    "glm": {
        "architectures": ["ChatGLMModel"],
        "model_type": "chatglm",
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": 16,
        "vocab_size": 32,
    },
    "gemma4": {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": {
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "intermediate_size": 16,
            "vocab_size": 32,
        },
    },
    "gemma4_reduced": {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": {
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "intermediate_size": 16,
            "vocab_size": 32,
        },
    },
}


def create_mock_model(
    model_dir: Path,
    arch: str = "qwen3",
    seed: int = 42,
) -> Path:
    """Create a minimal mock model directory with safetensors + config.json.

    Returns the model directory path.
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    gen = torch.Generator().manual_seed(seed)
    keys = KEY_GENERATORS[arch]()
    tensors = {k: v.clone().to(torch.bfloat16) for k, v in keys.items()}
    # Re-seed so different seeds produce different weights
    for k in tensors:
        tensors[k] = torch.empty_like(tensors[k]).random_(generator=gen)
    save_file(tensors, str(model_dir / "model-00001-of-00001.safetensors"))

    config = CONFIG_TEMPLATES[arch].copy()
    (model_dir / "config.json").write_text(json.dumps(config))

    return model_dir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_model_dir(tmp_path):
    """Create a single mock model dir (Qwen3 style, seed=42)."""
    return create_mock_model(tmp_path / "model_base", arch="qwen3", seed=42)


@pytest.fixture
def tmp_model_dir_qwen35(tmp_path):
    """Create a mock Qwen3.5 model dir."""
    return create_mock_model(tmp_path / "model_qwen35", arch="qwen35", seed=42)


@pytest.fixture
def tmp_model_dir_glm(tmp_path):
    """Create a mock GLM model dir."""
    return create_mock_model(tmp_path / "model_glm", arch="glm", seed=42)


@pytest.fixture
def tmp_comparison_dir(tmp_path):
    """Create a temporary comparison directory with mock models.

    Structure:
        test-comp/
        ├── comparison.json
        ├── base/           (Qwen3, seed=42)
        ├── variant_a/      (Qwen3, seed=43 — slightly different)
        └── variant_b/      (Qwen3, seed=44 — slightly different)
    """
    comp_dir = tmp_path / "test-comp"
    comp_dir.mkdir()

    comparison = {
        "name": "test-comp",
        "base": "base",
        "variants": {
            "variant_a": {"path": "variant_a"},
            "variant_b": {"path": "variant_b"},
        },
    }
    (comp_dir / "comparison.json").write_text(json.dumps(comparison))

    create_mock_model(comp_dir / "base", arch="qwen3", seed=42)
    create_mock_model(comp_dir / "variant_a", arch="qwen3", seed=43)
    create_mock_model(comp_dir / "variant_b", arch="qwen3", seed=44)

    return comp_dir


@pytest.fixture
def tmp_comparison_dir_3variants(tmp_path):
    """Comparison dir with 3 variants (for multi-variant tests)."""
    comp_dir = tmp_path / "test-3var"
    comp_dir.mkdir()

    comparison = {
        "name": "test-3var",
        "base": "base",
        "variants": {
            "heretic": {"path": "heretic"},
            "hauhau": {"path": "hauhau"},
            "huihui": {"path": "huihui"},
        },
    }
    (comp_dir / "comparison.json").write_text(json.dumps(comparison))

    create_mock_model(comp_dir / "base", arch="qwen3", seed=42)
    create_mock_model(comp_dir / "heretic", arch="qwen3", seed=43)
    create_mock_model(comp_dir / "hauhau", arch="qwen3", seed=44)
    create_mock_model(comp_dir / "huihui", arch="qwen3", seed=45)

    return comp_dir


@pytest.fixture
def tmp_model_dir_gemma4(tmp_path):
    """Create a mock Gemma4 model dir."""
    return create_mock_model(tmp_path / "model_gemma4", arch="gemma4", seed=42)


@pytest.fixture
def tmp_comparison_dir_gemma4(tmp_path):
    """Create a temporary comparison directory with Gemma4 mock models.

    Structure:
        test-gemma4/
        ├── comparison.json
        ├── base/               (Gemma4, seed=42)
        ├── variant_full/       (Gemma4, seed=43 — full 600-key style)
        └── variant_reduced/    (Gemma4 reduced, seed=44 — 540-key style)
    """
    comp_dir = tmp_path / "test-gemma4"
    comp_dir.mkdir()

    comparison = {
        "name": "test-gemma4",
        "base": "base",
        "variants": {
            "variant_full": {"path": "variant_full"},
            "variant_reduced": {"path": "variant_reduced"},
        },
    }
    (comp_dir / "comparison.json").write_text(json.dumps(comparison))

    create_mock_model(comp_dir / "base", arch="gemma4", seed=42)
    create_mock_model(comp_dir / "variant_full", arch="gemma4", seed=43)
    create_mock_model(comp_dir / "variant_reduced", arch="gemma4_reduced", seed=44)

    return comp_dir
