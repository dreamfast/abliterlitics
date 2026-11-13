"""Tests for src/weight/_common.py — shared streaming/CLI helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

from src.config import ComparisonConfig
from src.model_config import build_shard_map, detect_architecture
from src.weight._common import (
    StreamStats,
    cosine_sim,
    iter_common_tensors,
    resolve_results_dir,
    safe_mean,
    safe_median,
    safe_std,
    variant_output_path,
)
from tests.conftest import create_mock_model


class TestIterCommonTensors:
    def _maps(self, tmp_path):
        base_dir = create_mock_model(tmp_path / "base", arch="qwen3", seed=42)
        var_dir = create_mock_model(tmp_path / "var", arch="qwen3", seed=99)
        cfg = detect_architecture(str(base_dir))
        return build_shard_map(str(base_dir), cfg), build_shard_map(str(var_dir), cfg)

    def test_yields_sorted_common_keys_with_tensor_lists(self, tmp_path):
        base_map, var_map = self._maps(tmp_path)
        keys = []
        # The yielded list is cleared on resume — inspect tensors inside the loop.
        for _, k, tensors in iter_common_tensors([base_map, var_map]):
            keys.append(k)
            assert len(tensors) == 2
            assert all(isinstance(t, torch.Tensor) for t in tensors)
            assert tensors[0].shape == tensors[1].shape
        assert keys == sorted(set(base_map) & set(var_map))

    def test_shape_mismatch_counted_and_skipped(self, tmp_path):
        from tests.test_weight._arch_extending import make_arch_extending_variant

        base_map, _ = self._maps(tmp_path)
        ext_dir = make_arch_extending_variant(tmp_path / "var", tmp_path / "ext", "model.layers.0.mlp.gate_proj.weight")
        cfg = detect_architecture(str(tmp_path / "base"))
        ext_map = build_shard_map(str(ext_dir), cfg)

        stats = StreamStats()
        keys = [k for _, k, _ in iter_common_tensors([base_map, ext_map], stats=stats)]
        assert stats.shape_skipped == 1
        assert "model.layers.0.mlp.gate_proj.weight" not in keys

    def test_missing_key_skipped_silently(self, tmp_path):
        base_map, var_map = self._maps(tmp_path)
        trimmed = dict(var_map)
        del trimmed["model.norm.weight"]

        stats = StreamStats()
        keys = [k for _, k, _ in iter_common_tensors([base_map, trimmed], stats=stats)]
        assert "model.norm.weight" not in keys
        assert stats.shape_skipped == 0  # missing != shape-mismatch

    def test_explicit_keys_respected(self, tmp_path):
        base_map, var_map = self._maps(tmp_path)
        subset = sorted(set(base_map))[:3]
        keys = [k for _, k, _ in iter_common_tensors([base_map, var_map], keys=subset)]
        assert keys == subset

    def test_clears_yielded_list_on_resume(self, tmp_path):
        base_map, var_map = self._maps(tmp_path)
        gen = iter_common_tensors([base_map, var_map])
        _, _, first = next(gen)
        assert len(first) == 2
        next(gen)  # resume the generator -> yielded list must be cleared
        assert first == []

    def test_progress_callback_fires_at_gc_interval(self, tmp_path):
        base_map, var_map = self._maps(tmp_path)
        total = len(set(base_map) & set(var_map))
        calls: list[tuple[int, int]] = []
        list(iter_common_tensors([base_map, var_map], gc_interval=2, progress=lambda i, n: calls.append((i, n))))
        # one call per completed interval: indices 1, 3, 5, ...
        assert calls[0] == (1, total)
        assert [i for i, _ in calls] == [i for i in range(total) if (i + 1) % 2 == 0]

    def test_gc_interval_zero_disables_collection_branch(self, tmp_path):
        base_map, var_map = self._maps(tmp_path)
        calls: list[tuple[int, int]] = []
        list(iter_common_tensors([base_map, var_map], gc_interval=0, progress=lambda i, n: calls.append((i, n))))
        assert calls == []


class TestSharedStats:
    def test_safe_mean_median_std_empty(self):
        assert safe_mean([]) is None
        assert safe_median([]) is None
        assert safe_std([]) is None
        assert safe_std([1.0]) is None

    def test_safe_std_matches_manual(self):
        import math

        vals = [2.0, 4.0, 9.0]
        m = sum(vals) / len(vals)

        assert safe_std(vals) == pytest.approx(math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals)))

    def test_cosine_sim_bounds_and_zero(self):
        assert cosine_sim(torch.zeros(4), torch.ones(4)) == 0.0
        assert cosine_sim(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0])) == 1.0


class TestBatchCliHelpers:
    def test_variant_output_path(self):
        p = variant_output_path(Path("results/weight"), "heretic", "edit_vector")
        assert p == Path("results/weight/heretic/edit_vector_heretic.json")

    def test_resolve_results_dir_prefers_flag(self, tmp_comparison_dir):
        config = ComparisonConfig.from_dir(tmp_comparison_dir)
        args = argparse.Namespace(results_dir="/results/override")
        assert resolve_results_dir(args, config) == Path("/results/override")

    def test_resolve_results_dir_falls_back_to_config(self, tmp_comparison_dir):
        config = ComparisonConfig.from_dir(tmp_comparison_dir)
        args = argparse.Namespace(results_dir=None)
        assert resolve_results_dir(args, config) == config.weight_results_dir(Path("results"))
