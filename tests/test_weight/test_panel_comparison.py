"""Tests for src/weight/panel_comparison.py — Changed keys detection."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.model_config import ArchitectureConfig, build_shard_map, detect_architecture
from src.weight.panel_comparison import changed_keys_lazy
from tests.conftest import create_mock_model


@pytest.fixture(autouse=True)
def clear_cache():
    from src.model_config import clear_shard_map_cache
    clear_shard_map_cache()
    yield
    clear_shard_map_cache()


class TestChangedKeysLazy:
    def test_detects_changes_between_different_seeds(self, tmp_path):
        base_dir = create_mock_model(tmp_path / "base", arch="qwen3", seed=42)
        variant_dir = create_mock_model(tmp_path / "variant", arch="qwen3", seed=99)

        cfg = detect_architecture(str(base_dir))
        base_map = build_shard_map(str(base_dir), cfg)
        var_map = build_shard_map(str(variant_dir), cfg)

        changed = changed_keys_lazy(base_map, var_map, cfg, "test")
        assert len(changed) > 0
        for item in changed:
            assert "canonical" in item
            assert "mean_abs_diff" in item
            assert item["mean_abs_diff"] > 0

    def test_no_changes_identical_models(self, tmp_path):
        base_dir = create_mock_model(tmp_path / "base", arch="qwen3", seed=42)
        copy_dir = create_mock_model(tmp_path / "copy", arch="qwen3", seed=42)

        cfg = detect_architecture(str(base_dir))
        base_map = build_shard_map(str(base_dir), cfg)
        copy_map = build_shard_map(str(copy_dir), cfg)

        changed = changed_keys_lazy(base_map, copy_map, cfg, "test")
        assert len(changed) == 0

    def test_sorted_by_magnitude_descending(self, tmp_path):
        base_dir = create_mock_model(tmp_path / "base", arch="qwen3", seed=42)
        variant_dir = create_mock_model(tmp_path / "variant", arch="qwen3", seed=99)

        cfg = detect_architecture(str(base_dir))
        base_map = build_shard_map(str(base_dir), cfg)
        var_map = build_shard_map(str(variant_dir), cfg)

        changed = changed_keys_lazy(base_map, var_map, cfg, "test")
        diffs = [item["mean_abs_diff"] for item in changed]
        assert diffs == sorted(diffs, reverse=True)
