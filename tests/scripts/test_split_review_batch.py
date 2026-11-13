"""Tests for scripts/split_review_batch.py — lesson #77 wrap + sidecar behavior.

Regression guard: the item JSON must be written ONCE, AFTER the sidecar files
exist and with the sidecar pointer keys set (the old pre-sidecar first write
left a crash window producing item JSONs that pointed at no sidecars).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


split_review_batch = _load("split_review_batch")


def test_wrap_chunks_long_text() -> None:
    text = "x" * 3500
    wrapped = split_review_batch.wrap(text)
    lines = wrapped.split("\n")
    assert len(lines) == 3
    assert all(len(line) <= 1500 for line in lines)
    assert wrapped.replace("\n", "") == text


def test_wrap_short_and_empty() -> None:
    assert split_review_batch.wrap("") == ""
    assert split_review_batch.wrap("short") == "short"


def test_main_single_write_with_sidecar_pointers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    batch_file = tmp_path / "batch.json"
    outdir = tmp_path / "items"
    long_reasoning = "R" * 4000
    long_response = "V" * 2500
    batch_file.write_text(
        json.dumps(
            [
                {
                    "model_name": "arm_a",
                    "behavior_id": "b1",
                    "reasoning_text": long_reasoning,
                    "response_text": long_response,
                    "verdict": "soft_refusal",
                }
            ]
        )
    )
    monkeypatch.setattr(sys, "argv", ["split_review_batch.py", str(batch_file), str(outdir)])
    split_review_batch.main()

    item = json.loads((outdir / "item_00.json").read_text())
    # The surviving (single) JSON write carries the sidecar pointer keys.
    assert item["reasoning_text_sidecar"] == "item_00_reasoning.txt"
    assert item["response_text_sidecar"] == "item_00_response.txt"
    # Sidecars exist and hold the RAW wrapped text (real newlines, no escaping).
    reasoning_txt = (outdir / "item_00_reasoning.txt").read_text()
    assert "REASONING TEXT (wrapped, full)" in reasoning_txt
    assert len(reasoning_txt.splitlines()) > 1
    response_txt = (outdir / "item_00_response.txt").read_text()
    assert "RESPONSE TEXT (wrapped, full)" in response_txt
    # The wrapped JSON field round-trips to the original text.
    assert item["reasoning_text"].replace("\n", "") == long_reasoning
    assert item["response_text"].replace("\n", "") == long_response
    # No stray files: exactly one JSON + two sidecars for one item.
    assert sorted(p.name for p in outdir.iterdir()) == [
        "item_00.json",
        "item_00_reasoning.txt",
        "item_00_response.txt",
    ]


def test_main_multiple_items(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    batch_file = tmp_path / "batch.json"
    outdir = tmp_path / "items"
    items = [{"behavior_id": f"b{i}", "reasoning_text": "", "response_text": ""} for i in range(3)]
    batch_file.write_text(json.dumps(items))
    monkeypatch.setattr(sys, "argv", ["split_review_batch.py", str(batch_file), str(outdir)])
    split_review_batch.main()
    for n in range(3):
        assert (outdir / f"item_{n:02d}.json").is_file()
        assert (outdir / f"item_{n:02d}_reasoning.txt").is_file()
        assert (outdir / f"item_{n:02d}_response.txt").is_file()
