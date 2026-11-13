"""Tests for scripts/merge_lora.py — sharded-output handling.

Regression guard for the single-file assumption: save_pretrained shards >5GB
models into model-0000N-of-0000M.safetensors + model.safetensors.index.json.
The helpers must recognize both layouts, size them correctly, and the
idempotency skip must be base-relative (not the old hardcoded >20GiB).
Files are sparse-truncated (no bytes allocated) so tests stay fast.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


merge_lora = _load("merge_lora")

GIB = 1024**3


def _sparse(path: Path, size: int) -> None:
    with open(path, "wb") as fh:
        fh.truncate(size)


def _single_file_model(d: Path, size: int) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    _sparse(d / "model.safetensors", size)
    return d


def _sharded_model(d: Path, shard_sizes: list[int]) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    n = len(shard_sizes)
    for i, size in enumerate(shard_sizes, start=1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        _sparse(d / fname, size)
        weight_map[f"layer_{i}.weight"] = fname
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": sum(shard_sizes)}, "weight_map": weight_map})
    )
    return d


def test_safetensors_files_single() -> None:
    d = _single_file_model(Path("/tmp/unused-single"), 10)
    assert merge_lora.safetensors_files(d) == [d / "model.safetensors"]
    assert merge_lora.has_weights(d)


def test_safetensors_files_sharded(tmp_path: Path) -> None:
    d = _sharded_model(tmp_path / "sharded", [100, 200])
    files = merge_lora.safetensors_files(d)
    assert len(files) == 2
    assert all(f.is_file() for f in files)
    assert merge_lora.total_model_size(d) == 300


def test_total_model_size_sharded_base(tmp_path: Path) -> None:
    """A sharded BASE (e.g. a 27B index layout) sizes via its index."""
    d = _sharded_model(tmp_path / "base27b", [15 * GIB, 13 * GIB])
    assert merge_lora.total_model_size(d) == 28 * GIB


def test_empty_dir_has_no_weights(tmp_path: Path) -> None:
    d = tmp_path / "empty"
    d.mkdir()
    assert not merge_lora.has_weights(d)
    assert merge_lora.total_model_size(d) == 0
    assert merge_lora.safetensors_files(d) == []


def test_skip_reason_single_file_7b(tmp_path: Path) -> None:
    """7B-class merge (~14GiB, single file) must SKIP, not re-merge (old >20GiB bug)."""
    base = _single_file_model(tmp_path / "base", 14 * GIB)
    out = _single_file_model(tmp_path / "out", 14 * GIB)
    reason = merge_lora.skip_reason(out, base)
    assert reason is not None
    assert "[SKIP]" in reason


def test_skip_reason_sharded_output(tmp_path: Path) -> None:
    """A sharded merged output (5GB shards + index) must be recognized."""
    base = _single_file_model(tmp_path / "base", 28 * GIB)
    out = _sharded_model(tmp_path / "out", [15 * GIB, 13 * GIB])
    assert merge_lora.skip_reason(out, base) is not None


def test_no_skip_on_partial_output(tmp_path: Path) -> None:
    """A tiny placeholder file is 'likely partial' -> re-merge."""
    base = _single_file_model(tmp_path / "base", 28 * GIB)
    out = _single_file_model(tmp_path / "out", 2 * GIB)
    assert merge_lora.skip_reason(out, base) is None


def test_no_skip_on_missing_output(tmp_path: Path) -> None:
    base = _single_file_model(tmp_path / "base", 14 * GIB)
    out = tmp_path / "fresh"
    out.mkdir()
    assert merge_lora.skip_reason(out, base) is None


def test_skip_reason_fallback_threshold_without_base(tmp_path: Path) -> None:
    """No readable base -> fall back to the 20GiB absolute threshold."""
    out = _single_file_model(tmp_path / "out", 25 * GIB)
    empty = tmp_path / "nobase"
    empty.mkdir()
    assert merge_lora.skip_reason(out, empty) is not None
