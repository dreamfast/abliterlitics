"""Table-driven tests for scripts/harm_review_gate.py (the overseer's gate).

Covers the violation classes the gate exists to catch: bad verdicts, missing /
extra keys, duplicates, verbatim-key mismatch, thin reasoning — plus the
range-scoping args ([start] [end]) used for partial dispatch gating.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
GATE = SCRIPTS / "harm_review_gate.py"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, GATE.with_name(f"{name}.py"))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gate = _load("harm_review_gate")

REASONING = "r" * 60  # >= 40 chars: passes the thin_reasoning floor


def _item(arm: str, bid: str) -> dict[str, str]:
    return {"model_name": arm, "behavior_id": bid}


def _result(arm: str, bid: str, verdict: str = "refusal") -> dict[str, Any]:
    return {
        "model_name": arm,
        "behavior_id": bid,
        "verdict": verdict,
        "review_reasoning": REASONING,
    }


def _run_gate(batch: list[dict[str, Any]], results: list[dict[str, Any]], *extra: str) -> tuple[int, str]:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        bp = Path(td) / "batch.json"
        rp = Path(td) / "results.json"
        bp.write_text(json.dumps(batch))
        rp.write_text(json.dumps(results))
        proc = subprocess.run(
            [sys.executable, str(GATE), str(bp), str(rp), *extra],
            capture_output=True,
            text=True,
        )
    return proc.returncode, proc.stdout


def test_gate_pass() -> None:
    batch = [_item("a", "b1"), _item("a", "b2")]
    results = [_result("a", "b1"), _result("a", "b2", "no_refusal")]
    code, out = _run_gate(batch, results)
    assert code == 0
    assert "covered=2/2" in out
    assert "thin_reasoning=[]" in out


@pytest.mark.parametrize(
    "results",
    [
        [_result("a", "b1", "partially_complied")],  # bad verdict
        [],  # missing
        [_result("a", "b1"), _result("a", "b2"), _result("a", "b3")],  # extra
        [_result("a", "b1"), _result("a", "b1")],  # duplicates
    ],
)
def test_gate_violations(results: list[dict[str, Any]]) -> None:
    batch = [_item("a", "b1"), _item("a", "b2")]
    code, _out = _run_gate(batch, results)
    assert code == 1


def test_gate_verbatim_key_mismatch() -> None:
    batch = [_item("a", "b1"), _item("a", "b2")]
    results = [_result("a", "b1"), _result("a", "WRONG_ID")]
    code, out = _run_gate(batch, results)
    assert code == 1
    assert "keys_verbatim=False" in out


def test_gate_thin_reasoning() -> None:
    batch = [_item("a", "b1")]
    r = _result("a", "b1")
    r["review_reasoning"] = "looks fine"  # 10 chars < 40
    code, out = _run_gate(batch, [r])
    assert code == 1
    assert "thin_reasoning=[('a', 'b1')]" in out


def test_gate_range_scoping() -> None:
    """Partial dispatch: results for items 1-3 of a 4-item batch pass under [1] [3]."""
    batch = [_item("a", f"b{i}") for i in range(4)]
    results = [_result("a", "b1"), _result("a", "b2")]
    code, out = _run_gate(batch, results, "1", "3")
    assert code == 0
    assert "covered=2/2" in out


def test_gate_range_scoping_outside_range_fails() -> None:
    """A result outside the dispatched range is an 'extra' violation."""
    batch = [_item("a", f"b{i}") for i in range(4)]
    results = [_result("a", "b0"), _result("a", "b1")]
    code, _out = _run_gate(batch, results, "1", "3")
    assert code == 1
