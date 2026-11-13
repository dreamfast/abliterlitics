"""Tests for scripts/gsm8k_three_number_report.py — lessons #20/#32/#51.

compute() is the pure 3-number core (dedupe by doc_id, flexible-extract
filter, empty-response handling); the bridge must place the samples file in
__model/ under the <model>__ prefix while working from ANY invocation CWD
(the old bug: copyfile args resolved against CWD, so the bridge only worked
when invoked from inside __model/).
"""

from __future__ import annotations

import importlib.util
import json
import os
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


gsm8k = _load("gsm8k_three_number_report")


def _row(doc_id: int, resp: str, em: float, filt: str = "flexible-extract") -> dict[str, Any]:
    return {"doc_id": doc_id, "filter": filt, "resps": [[resp]], "exact_match": em}


def _write_samples(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


class TestCompute:
    def test_three_numbers(self, tmp_path: Path) -> None:
        samples = tmp_path / "samples_gsm8k_1.jsonl"
        _write_samples(
            samples,
            [
                _row(0, "The answer is 42 #### 42", 1.0),
                _row(1, "", 0.0),  # empty -> wrong in headline, excluded from answered
                _row(2, "5", 1.0),
                _row(3, "wrong #### 7", 0.0),
                _row(2, "5", 1.0),  # duplicate doc_id row must not double-count
                _row(9, "ignored other filter", 1.0, filt="strict-match"),
            ],
        )
        report = gsm8k.compute(samples)
        assert report["total_problems"] == 4
        assert report["empty_count"] == 1
        assert report["empty_pct"] == 25.0
        assert report["headline_exact_match"] == 0.5  # 2 of 4 (empty counts wrong)
        assert report["answered_only_exact_match"] == round(2 / 3, 4)
        assert report["thinking_tax_pp"] == round(100.0 * (0.5 - 2 / 3), 2)

    def test_no_flexible_rows_exits(self, tmp_path: Path) -> None:
        samples = tmp_path / "samples.jsonl"
        _write_samples(samples, [_row(0, "x", 1.0, filt="strict-match")])
        with pytest.raises(SystemExit, match="no flexible-extract rows"):
            gsm8k.compute(samples)


class TestBridge:
    def test_bridge_works_from_any_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        comp = tmp_path / "comparison"
        model_dir = comp / "results" / "lm_eval" / "__model"
        per_model = comp / "results" / "lm_eval" / "myarm"
        samples = per_model / "samples_gsm8k_999.jsonl"
        _write_samples(
            samples,
            [
                _row(0, "42 #### 42", 1.0),
                _row(1, "7", 0.0),
            ],
        )
        # Invoke from an unrelated CWD — this is the old bug's failure condition.
        elsewhere = tmp_path / "unrelated"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "gsm8k_three_number_report.py",
                "--comparison-dir",
                str(comp),
                "--model",
                "myarm",
                "--samples",
                str(samples),
            ],
        )
        assert gsm8k.main() == 0
        bridged = model_dir / "myarm__samples_gsm8k_999.jsonl"
        assert bridged.is_file()
        assert bridged.read_text() == samples.read_text()
        # The canonical report lands next to the bridge.
        report = json.loads((comp / "results" / "lm_eval" / "gsm8k_3num_myarm.json").read_text())
        assert report["model"] == "myarm"
        assert report["total_problems"] == 2
        # CWD restored after the bridge.
        assert os.getcwd() == str(elsewhere.resolve())

    def test_no_bridge_when_already_in_model_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        comp = tmp_path / "comparison"
        model_dir = comp / "results" / "lm_eval" / "__model"
        samples = model_dir / "myarm__samples_gsm8k_5.jsonl"
        _write_samples(samples, [_row(0, "1", 1.0)])
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "gsm8k_three_number_report.py",
                "--comparison-dir",
                str(comp),
                "--model",
                "myarm",
                "--samples",
                str(samples),
            ],
        )
        assert gsm8k.main() == 0
        # No duplicate bridged copy.
        assert sorted(p.name for p in model_dir.iterdir()) == ["myarm__samples_gsm8k_5.jsonl"]
