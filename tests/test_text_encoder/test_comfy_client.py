"""Tests for comfy_client error extraction + transient krea2 staging retry.

Locks the two behaviors added for the 5090 intermittent first-encode failure
(single-layer 2560 conditioning after a dynamic-VRAM (re)stage of the 12-tap
krea2 TE): the execution_error message is surfaced human-readably (not the
old messages[:2] timestamp soup), and queue_and_fetch retries ONLY that
signature, failing fast on anything else.

Also locks the HTTP surface error paths (lesson-#39/#45 semantics): the
server-assigned prompt_id contract, node_errors rejection, /view subfolder
params, /free eviction waiting, and the flush-between-retries behavior.
"""

from __future__ import annotations

import json
import urllib.error
from typing import Any

import pytest

from src.text_encoder import comfy_client
from src.text_encoder.comfy_client import (
    _queue_once,
    api,
    comfy_te_name,
    extract_execution_error,
    flush_loaded_models,
    free_models,
    query_vram,
    queue_and_fetch,
    sanitize,
)


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


# ---------------------------------------------------------------------------
# HTTP surface: helpers
# ---------------------------------------------------------------------------


class _FakeResp:
    """Minimal urlopen() context-manager stand-in."""

    def __init__(self, body: bytes = b"{}", status: int = 200) -> None:
        self.body = body
        self.status = status

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *args: Any) -> bool:
        return False


def _history_entry(
    status_str: str = "success",
    completed: bool = True,
    outputs: dict[str, Any] | None = None,
    messages: list[list[Any]] | None = None,
) -> dict[str, Any]:
    return {
        "status": {
            "status_str": status_str,
            "completed": completed,
            "messages": messages if messages is not None else [],
        },
        "outputs": outputs if outputs is not None else {},
        "prompt": [3, "p9"],
    }


class TestApi:
    def test_get_no_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def fake_urlopen(req: Any, timeout: float | None = None) -> _FakeResp:
            captured["req"] = req
            return _FakeResp(b'{"ok": 1}')

        monkeypatch.setattr(comfy_client.urllib.request, "urlopen", fake_urlopen)
        assert api("http://x:8188/", "/system_stats") == {"ok": 1}
        assert captured["req"].full_url == "http://x:8188/system_stats"
        assert captured["req"].data is None

    def test_post_json_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def fake_urlopen(req: Any, timeout: float | None = None) -> _FakeResp:
            captured["req"] = req
            return _FakeResp(b'{"queued": true}')

        monkeypatch.setattr(comfy_client.urllib.request, "urlopen", fake_urlopen)
        assert api("http://x", "/prompt", {"a": 1}) == {"queued": True}
        assert captured["req"].data == json.dumps({"a": 1}).encode()
        assert captured["req"].headers.get("Content-type") == "application/json"

    def test_network_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_urlopen(req: Any, timeout: float | None = None) -> _FakeResp:
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(comfy_client.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(urllib.error.URLError):
            api("http://x", "/system_stats")


class TestQueryVram:
    def test_parses_system_stats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            comfy_client,
            "api",
            lambda *a, **k: {
                "devices": [{"vram_total": 32 * 2**30, "vram_free": 30 * 2**30, "torch_vram_total": 4 * 2**30}]
            },
        )
        v = query_vram("http://x")
        assert v["vram_used_mb"] == 2048.0
        assert v["torch_vram_mb"] == 4096.0

    def test_missing_devices_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(comfy_client, "api", lambda *a, **k: {})
        v = query_vram("http://x")
        assert v["vram_used_mb"] == 0.0


class TestFreeModels:
    def test_eviction_confirmed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        paths: list[str] = []

        def fake_api(url: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
            paths.append(path)
            return {"devices": [{"vram_total": 32 * 2**30, "vram_free": 31 * 2**30}]}  # ~1GB used

        monkeypatch.setattr(comfy_client, "api", fake_api)
        monkeypatch.setattr(comfy_client.time, "sleep", lambda s: None)
        assert free_models("http://x") is True
        assert paths[0] == "/free"  # unload-only: free_memory purge is never sent
        assert "/system_stats" in paths

    def test_free_endpoint_failure_is_best_effort_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_api(url: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
            if path == "/free":
                raise OSError("connection refused")
            raise AssertionError("should not poll after /free failure")

        monkeypatch.setattr(comfy_client, "api", fake_api)
        assert free_models("http://x") is False

    def test_eviction_not_confirmed_within_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # VRAM stays high forever; timeout_s=0 skips the wait loop entirely.
        monkeypatch.setattr(
            comfy_client,
            "api",
            lambda *a, **k: {"devices": [{"vram_total": 32 * 2**30, "vram_free": 2**30}]},
        )
        assert free_models("http://x", timeout_s=0.0) is False

    def test_transient_stats_errors_tolerated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        polls = {"n": 0}

        def fake_api(url: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
            if path == "/free":
                return {}
            polls["n"] += 1
            if polls["n"] == 1:
                raise OSError("transient stats failure")
            return {"devices": [{"vram_total": 0, "vram_free": 0}]}

        monkeypatch.setattr(comfy_client, "api", fake_api)
        monkeypatch.setattr(comfy_client.time, "sleep", lambda s: None)
        assert free_models("http://x") is True
        assert polls["n"] == 2


class TestFlushLoadedModels:
    def test_flush_posts_unload_and_returns_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def fake_urlopen(req: Any, timeout: float | None = None) -> _FakeResp:
            captured["req"] = req
            return _FakeResp(status=200)

        monkeypatch.setattr(comfy_client.urllib.request, "urlopen", fake_urlopen)
        assert flush_loaded_models("http://x") is True
        assert captured["req"].full_url.endswith("/free")
        assert json.loads(captured["req"].data) == {"unload_models": True}

    def test_flush_failure_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_urlopen(req: Any, timeout: float | None = None) -> _FakeResp:
            raise urllib.error.URLError("down")

        monkeypatch.setattr(comfy_client.urllib.request, "urlopen", fake_urlopen)
        assert flush_loaded_models("http://x") is False


class TestQueueOnce:
    """_queue_one error paths + the lesson-#39/#45 happy-path contracts."""

    def test_missing_prompt_id_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(comfy_client, "api", lambda *a, **k: {"node_errors": {}})
        with pytest.raises(RuntimeError, match="No prompt_id"):
            _queue_once("http://x", {})

    def test_node_errors_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # /prompt's node_errors must be checked — a rejected workflow queues nothing.
        monkeypatch.setattr(
            comfy_client, "api", lambda *a, **k: {"prompt_id": "p1", "node_errors": {"3": {"errors": ["bad"]}}}
        )
        with pytest.raises(RuntimeError, match="rejected the workflow"):
            _queue_once("http://x", {})

    def test_history_error_status_raises_with_detail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_api(url: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
            if path == "/prompt":
                return {"prompt_id": "p1", "node_errors": {}}
            return {"p1": _history_entry(status_str="error", messages=_status_messages())}

        monkeypatch.setattr(comfy_client, "api", fake_api)
        monkeypatch.setattr(comfy_client.time, "sleep", lambda s: None)
        with pytest.raises(RuntimeError, match="KSampler\\(node 72\\)"):
            _queue_once("http://x", {})

    def test_completed_without_outputs_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_api(url: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
            if path == "/prompt":
                return {"prompt_id": "p1", "node_errors": {}}
            return {"p1": _history_entry(outputs={})}

        monkeypatch.setattr(comfy_client, "api", fake_api)
        monkeypatch.setattr(comfy_client.time, "sleep", lambda s: None)
        with pytest.raises(RuntimeError, match="completed without"):
            _queue_once("http://x", {})

    def test_poll_timeout_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_api(url: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
            if path == "/prompt":
                return {"prompt_id": "p1", "node_errors": {}}
            return {}  # prompt never appears in history

        monkeypatch.setattr(comfy_client, "api", fake_api)
        monkeypatch.setattr(comfy_client.time, "sleep", lambda s: None)
        monkeypatch.setattr(comfy_client, "POLL_TIMEOUT", 0.0)
        with pytest.raises(TimeoutError):
            _queue_once("http://x", {})

    def test_success_returns_bytes_and_perf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Happy path: server-assigned prompt_id polled, /view fetched with
        separate filename/subfolder/type params (lesson #39), perf extracted
        from status messages."""
        paths: list[str] = []
        view_calls: list[str] = []

        def fake_api(url: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
            paths.append(path)
            if path == "/prompt":
                assert payload == {"prompt": {"3": {}}, "client_id": comfy_client.CLIENT_ID}
                return {"prompt_id": "p9", "node_errors": {}}
            assert path == "/history/p9"
            messages = [
                ["execution_start", {"timestamp": 100.0}],
                ["execution_success", {"timestamp": 103.5}],
                ["execution_cached", {"nodes": ["7", "54"]}],
            ]
            outputs = {"55": {"images": [{"filename": "img_00001.webp", "subfolder": "abl", "type": "output"}]}}
            return {"p9": _history_entry(outputs=outputs, messages=messages)}

        def fake_urlopen(url: Any, timeout: float | None = None) -> _FakeResp:
            view_calls.append(str(url))
            return _FakeResp(b"png-bytes")

        monkeypatch.setattr(comfy_client, "api", fake_api)
        monkeypatch.setattr(comfy_client.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(comfy_client.time, "sleep", lambda s: None)

        data, perf = _queue_once("http://x:8188", {"3": {}}, te_node_id="54")
        assert data == b"png-bytes"
        assert perf["prompt_id"] == "p9"
        assert perf["queue_number"] == 3
        # shipped semantics: server_gen_ms = execution_success - execution_start (raw delta)
        assert perf["server_gen_ms"] == 3.5
        assert perf["te_cached"] is True  # TE loader node 54 in execution_cached
        assert perf["cached_nodes"] == ["7", "54"]
        assert perf["status_str"] == "success"
        # /view needs separate query params, NOT a slash-path filename
        assert len(view_calls) == 1
        assert "filename=img_00001.webp" in view_calls[0]
        assert "subfolder=abl" in view_calls[0]
        assert "type=output" in view_calls[0]


class TestNamingHelpers:
    def test_sanitize(self) -> None:
        assert sanitize("Hello, World! 3x") == "hello_world_3x"
        assert sanitize("   ") == "prompt"  # empty stem falls back

    def test_comfy_te_name_strips_subdirs(self) -> None:
        p = "/m/text_encoders/qwen3-4b/te.safetensors"
        assert comfy_te_name(p, "/m") == "qwen3-4b/te.safetensors"

    def test_comfy_te_name_foreign_path_kept(self) -> None:
        assert comfy_te_name("relative.safetensors", "/m") == "relative.safetensors"
