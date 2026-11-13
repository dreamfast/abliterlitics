"""
Architecture auto-detection for Qwen3, Qwen3.5, Gemma4, GLM models.

Builds safetensors shard maps, provides lazy tensor loading.
This is the foundation for all weight analysis scripts.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

log = logging.getLogger(__name__)

TRIPLE_PREFIX = "model.language_model.language_model.language_model."

# Module-level shard map cache: (model_dir_path_str,) -> shard_map
_shard_map_cache: dict[tuple[str, str], dict[str, tuple[str, str]]] = {}


@dataclass
class ArchitectureConfig:
    """Detected architecture configuration for a model."""

    family: str
    prefix: str
    has_experts: bool = False
    expert_count: int = 0
    has_shared_experts: bool = False
    has_mamba: bool = False
    has_multimodal_towers: bool = False
    layer_count: int = 0
    total_keys: int = 0
    lm_keys: list[str] = field(default_factory=list)

    _LANGUAGE_MODEL_FAMILIES = frozenset({"qwen35", "gemma4"})

    def is_lm_key(self, raw_key: str, canonical_key: str) -> bool:
        if self.family in self._LANGUAGE_MODEL_FAMILIES:
            return "language_model" in canonical_key and "visual" not in raw_key and "mtp" not in raw_key
        return True

    def canonicalize(self, raw_key: str) -> str:
        if raw_key.startswith(TRIPLE_PREFIX):
            return "model.language_model." + raw_key[len(TRIPLE_PREFIX) :]
        return raw_key

    def get_layer_index(self, canonical_key: str) -> int | None:
        m = re.search(r"layers\.(\d+)", canonical_key)
        return int(m.group(1)) if m else None

    def get_tensor_type(self, canonical_key: str) -> str:
        if self.family in self._LANGUAGE_MODEL_FAMILIES:
            for prefix in ("model.language_model.layers.",):
                if canonical_key.startswith(prefix):
                    rest = canonical_key[len(prefix) :]
                    parts = rest.split(".")
                    if parts[0].isdigit():
                        return ".".join(parts[1:])
            if canonical_key.startswith("model.language_model."):
                return canonical_key[len("model.language_model.") :]
            return canonical_key
        else:
            for prefix in ("model.layers.",):
                if canonical_key.startswith(prefix):
                    rest = canonical_key[len(prefix) :]
                    parts = rest.split(".")
                    if parts[0].isdigit():
                        return ".".join(parts[1:])
            if canonical_key.startswith("model."):
                return canonical_key[len("model.") :]
            return canonical_key

    def get_expert_id(self, canonical_key: str) -> int | None:
        m = re.search(r"mlp\.experts\.(\d+)\.", canonical_key)
        return int(m.group(1)) if m else None

    def is_expert_tensor(self, canonical_key: str) -> bool:
        return ".experts." in canonical_key and "shared_expert" not in canonical_key

    def is_shared_expert_tensor(self, canonical_key: str) -> bool:
        return "shared_expert" in canonical_key

    def is_router_tensor(self, canonical_key: str) -> bool:
        return "mlp.gate." in canonical_key

    def is_attention_tensor(self, canonical_key: str) -> bool:
        return "self_attn." in canonical_key

    def is_norm_tensor(self, canonical_key: str) -> bool:
        return any(
            n in canonical_key
            for n in (
                "layernorm",
                "input_layernorm",
                "post_attention_layernorm",
                "post_per_layer_input_norm",
                "q_norm",
                "k_norm",
                "kv_a_layernorm",
                "q_a_layernorm",
                "model.norm",
                "enorm",
                "hnorm",
            )
        )

    def tensor_category(self, canonical_key: str) -> str:
        if self.is_expert_tensor(canonical_key):
            return "expert"
        if self.is_shared_expert_tensor(canonical_key):
            return "shared_expert"
        if self.is_router_tensor(canonical_key):
            return "router"
        if self.is_attention_tensor(canonical_key):
            return "attention"
        if self.is_norm_tensor(canonical_key):
            return "norm"
        if "embed_tokens" in canonical_key:
            return "embedding"
        if "lm_head" in canonical_key:
            return "lm_head"
        if "per_layer_projection" in canonical_key or "per_layer_input_gate" in canonical_key:
            return "per_layer"
        if "layer_scalar" in canonical_key:
            return "scalar"
        if "mlp." in canonical_key:
            return "mlp"
        if self.has_mamba and ("linear_attn." in canonical_key or "conv1d" in canonical_key):
            return "mamba"
        return "other"


def _scan_keys(model_dir: str | Path) -> list[str]:
    """Scan all safetensors files in a directory and return all keys."""
    all_keys: list[str] = []
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:  # type: ignore[no-untyped-call]
            all_keys.extend(sf.keys())
    return all_keys


def detect_architecture(model_dir: str | Path) -> ArchitectureConfig:
    """Detect architecture from safetensors key patterns."""
    keys = _scan_keys(model_dir)
    has_language_model = any("language_model" in k for k in keys)
    has_experts = any(".experts." in k for k in keys)
    has_shared_experts = any("shared_expert" in k for k in keys)
    has_mamba = any("linear_attn" in k or "conv1d" in k for k in keys)
    has_gemma4 = any("layer_scalar" in k or "per_layer_projection" in k for k in keys)
    has_multimodal = any("audio_tower" in k or "vision_tower" in k for k in keys)

    if has_gemma4:
        family = "gemma4"
        prefix = "model.language_model."
    elif has_language_model:
        family = "qwen35"
        prefix = "model.language_model."
    elif has_experts:
        family = "glm"
        prefix = "model."
    else:
        family = "qwen3"
        prefix = "model."

    layer_nums: set[int] = set()
    for k in keys:
        m = re.search(r"layers\.(\d+)", k)
        if m:
            layer_nums.add(int(m.group(1)))

    expert_count = 0
    if has_experts:
        expert_ids: set[int] = set()
        for k in keys:
            m = re.search(r"experts\.(\d+)\.", k)
            if m:
                expert_ids.add(int(m.group(1)))
        expert_count = len(expert_ids)

    cfg = ArchitectureConfig(
        family=family,
        prefix=prefix,
        has_experts=has_experts,
        expert_count=expert_count,
        has_shared_experts=has_shared_experts,
        has_mamba=has_mamba,
        has_multimodal_towers=has_multimodal,
        layer_count=len(layer_nums),
        total_keys=len(keys),
    )
    cfg.lm_keys = sorted(keys)
    log.info(
        "Detected architecture: family=%s layers=%d keys=%d experts=%d mamba=%s multimodal=%s",
        family,
        len(layer_nums),
        len(keys),
        expert_count,
        has_mamba,
        has_multimodal,
    )
    return cfg


def build_shard_map(model_dir: str | Path, cfg: ArchitectureConfig) -> dict[str, tuple[str, str]]:
    """Build a map from canonical key -> (raw_key, shard_file_path).

    Results are cached per (model_dir, family) pair.
    """
    model_dir_str = str(Path(model_dir).resolve())
    cache_key = (model_dir_str, cfg.family)
    if cache_key in _shard_map_cache:
        return _shard_map_cache[cache_key]

    keymap: dict[str, tuple[str, str]] = {}
    p = Path(model_dir)
    for f in sorted(p.glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:  # type: ignore[no-untyped-call]
            for raw_key in sf.keys():
                ck = cfg.canonicalize(raw_key)
                if cfg.is_lm_key(raw_key, ck):
                    keymap[ck] = (raw_key, str(f))

    _shard_map_cache[cache_key] = keymap
    log.debug("Built shard map for %s: %d keys", model_dir_str, len(keymap))
    return keymap


def clear_shard_map_cache() -> None:
    """Clear the shard map cache (useful for testing)."""
    _shard_map_cache.clear()


def load_tensor(shard_map: dict[str, tuple[str, str]], canonical_key: str) -> torch.Tensor | None:
    """Load a single tensor from the correct shard file."""
    entry = shard_map.get(canonical_key)
    if entry is None:
        return None
    raw_key, shard_path = entry
    with safe_open(shard_path, framework="pt", device="cpu") as sf:  # type: ignore[no-untyped-call]
        tensor: torch.Tensor = sf.get_tensor(raw_key)
        return tensor


def get_all_canonical_keys(model_dir: str | Path, cfg: ArchitectureConfig) -> list[str]:
    """Return sorted list of all canonical tensor keys.

    Delegates to build_shard_map (which caches) and returns its keys.
    """
    return sorted(build_shard_map(model_dir, cfg).keys())


def compute_changed_keys(
    base_shard_map: dict[str, tuple[str, str]],
    other_shard_map: dict[str, tuple[str, str]],
    threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """Find tensors that differ between base and variant."""
    base_keys = set(base_shard_map)
    other_keys = set(other_shard_map)
    common = sorted(base_keys & other_keys)
    changed: list[dict[str, Any]] = []
    for ck in common:
        bt = load_tensor(base_shard_map, ck)
        ot = load_tensor(other_shard_map, ck)
        if bt is None or ot is None:
            continue
        if bt.shape != ot.shape:
            changed.append({"canonical": ck, "mean_abs_diff": float("inf")})
            continue
        diff = (ot.float() - bt.float()).abs().mean().item()
        if diff > threshold:
            changed.append({"canonical": ck, "mean_abs_diff": diff})
    changed.sort(key=lambda x: x["mean_abs_diff"], reverse=True)
    return changed
