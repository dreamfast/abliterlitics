"""Tests for comfy_client error extraction + transient krea2 staging retry.

Locks the two behaviors added for the 5090 intermittent first-encode failure
(single-layer 2560 conditioning after a dynamic-VRAM (re)stage of the 12-tap
krea2 TE): the execution_error message is surfaced human-readably (not the
old messages[:2] timestamp soup), and queue_and_fetch retries ONLY that
signature, failing fast on anything else.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.text_encoder import comfy_client
from src.text_encoder.comfy_client import extract_execution_error, queue_and_fetch


def _status_messages() -> list[list[Any]]:
    """Shape of a /history status.messages list with an execution_error inside."""
    return [
        ["execution_start", {"prompt_id": "p1", "timestamp": 1}],
        ["execution_cached", {"nodes": ["49", "53"], "prompt_id": "p1", "timestamp": 2}],
        [
            "execution_error",
            {
                "prompt_id": "p1",
                "node_id": "72",
                "node_type": "KSampler",
                "exception_message": "Krea2 expects conditioning with 12x2560=30720 features "
                "but got 2560. Load the text encoder with CLIPLoader type 'krea2'.",
                "exception_type": "ValueError",
            },
        ],
    ]


class TestExtractExecutionError:
    def test_pulls_node_type_and_message(self) -> None:
        detail = extract_execution_error(_status_messages())
        assert detail is not None
        assert detail.startswith("KSampler(node 72):")
        assert "12x2560=30720" in detail

    def test_none_when_no_execution_error(self) -> None:
        msgs: list[list[Any]] = [
            ["execution_start", {"timestamp": 1}],
            ["execution_success", {"timestamp": 2}],
        ]
        assert extract_execution_error(msgs) is None

    def test_tolerates_malformed_payloads(self) -> None:
        """Non-dict execution_error payloads are skipped -> caller falls back to raw JSON dump."""
        msgs: list[list[Any]] = [["execution_error", "not-a-dict"], ["weird", 42]]
        assert extract_execution_error(msgs) is None


class TestQueueAndFetchRetry:
    def test_retries_transient_krea2_failure_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[int] = []
        transient = RuntimeError(
            "ComfyUI reported error for p1: KSampler(node 72): Krea2 expects conditioning "
            "with 12x2560=30720 features but got 2560."
        )
        ok = (b"webp-bytes", {"prompt_id": "p2"})

        def fake_once(api_url: str, wf: dict[str, Any], te_node_id: str = "") -> tuple[bytes, dict[str, Any]]:
            calls.append(1)
            if len(calls) < 2:
                raise transient
            return ok

        sleeps: list[float] = []
        monkeypatch.setattr(comfy_client, "_queue_once", fake_once)
        monkeypatch.setattr(comfy_client.time, "sleep", lambda s: sleeps.append(s))
        data, perf = queue_and_fetch("http://x", {"n": {}}, te_node_id="54")
        assert data == b"webp-bytes"
        assert perf == {"prompt_id": "p2"}
        assert len(calls) == 2
        assert sleeps == [comfy_client.TRANSIENT_RETRY_BACKOFF_S]

    def test_gives_up_after_max_attempts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[int] = []
        transient = RuntimeError(
            "... expects conditioning with 12x2560 ... Load the text encoder with CLIPLoader type 'krea2'."
        )

        def fake_once(api_url: str, wf: dict[str, Any], te_node_id: str = "") -> tuple[bytes, dict[str, Any]]:
            calls.append(1)
            raise transient

        monkeypatch.setattr(comfy_client, "_queue_once", fake_once)
        monkeypatch.setattr(comfy_client.time, "sleep", lambda s: None)
        with pytest.raises(RuntimeError, match="12x2560"):
            queue_and_fetch("http://x", {})
        assert len(calls) == comfy_client.TRANSIENT_RETRY_ATTEMPTS

    def test_non_transient_error_fails_fast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[int] = []

        def fake_once(api_url: str, wf: dict[str, Any], te_node_id: str = "") -> tuple[bytes, dict[str, Any]]:
            calls.append(1)
            raise RuntimeError("ComfyUI reported error for p1: KSampler(node 72): CUDA out of memory")

        monkeypatch.setattr(comfy_client, "_queue_once", fake_once)
        monkeypatch.setattr(comfy_client.time, "sleep", lambda s: None)
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            queue_and_fetch("http://x", {})
        assert len(calls) == 1  # no retry on unrelated errors
