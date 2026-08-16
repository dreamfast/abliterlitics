"""Shared ComfyUI-API client for A/B generation runners.

Single home for the HTTP surface (queue/poll/fetch, model eviction, VRAM
probes, TE-name relativization) so every runner (images: run_ab_test.py,
video+audio: run_h3_ab.py) shares ONE copy of the hard-won semantics:

- ComfyUI assigns its own prompt_id — poll THAT, never a self-generated one
  (a self-polled id hangs forever; lesson #39).
- SaveVideo outputs arrive under ``outputs[*]["images"]`` with
  ``"animated": [true]`` — same fetch path as images (probe, journal #135).
- ``filename_prefix`` containing a slash writes into a SUBFOLDER; /view
  needs separate filename/subfolder/type query params (lesson #39).
- /free is asynchronous — poll /system_stats until VRAM drops before
  trusting cold-start timings.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

POLL_INTERVAL = 2.0
POLL_TIMEOUT = 900.0  # 15 min per generation (turbo: seconds; H3 ~3min cold)

CLIENT_ID = "abliterlitics-ab"


def api(api_url: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
    """GET/POST a ComfyUI endpoint; returns parsed JSON."""
    url = f"{api_url.rstrip('/')}{path}"
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
        )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body: str = resp.read().decode()
    loaded: dict[str, Any] = json.loads(body)
    return loaded


def query_vram(api_url: str) -> dict[str, float]:
    """Capture VRAM usage from ComfyUI's /system_stats (cuda:0 device).

    vram_used = total GPU VRAM consumed (includes all resident models);
    torch_alloc = PyTorch-committed VRAM (higher for GGUF which materializes
    dequantized tensors).
    """
    d = api(api_url, "/system_stats")
    dev = d.get("devices", [{}])[0]
    return {
        "vram_used_mb": round((dev.get("vram_total", 0) - dev.get("vram_free", 0)) / 2**20, 1),
        "torch_vram_mb": round(dev.get("torch_vram_total", 0) / 2**20, 1),
    }


def free_models(api_url: str, timeout_s: float = 180.0, threshold_mb: float = 4000.0) -> bool:
    """Unload ALL models via ComfyUI's /free endpoint and WAIT for eviction.

    /free is asynchronous: returning immediately lets the next queued prompt
    race ahead of the unload and run against still-cached models (fake
    "cold"). We poll /system_stats until total VRAM drops below
    ``threshold_mb`` (loaded state is 15-20GB; idle CUDA context ~1GB).

    Returns True if eviction confirmed within the timeout.
    """
    try:
        # unload ONLY - free_memory (allocator purge) makes the next generation
        # pathologically slow (17-257s observed) from allocator re-growth, which
        # poisons the cold-start measurement we are trying to take.
        api(api_url, "/free", {"unload_models": True})
    except Exception as exc:  # noqa: BLE001 - free is best-effort
        log.warning("/free failed (cold start not guaranteed): %s", exc)
        return False
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        try:
            v = query_vram(api_url)["vram_used_mb"]
            if v < threshold_mb:
                log.info("Models evicted in %.1fs (vram %.0fMB)", time.monotonic() - t0, v)
                return True
        except Exception:  # noqa: BLE001 - transient stats errors
            pass
        time.sleep(1.0)
    log.warning("Eviction not confirmed within %.0fs - proceeding anyway", timeout_s)
    return False


def comfy_te_name(te_file: str, models_root: str, subdirs: tuple[str, ...] = ("text_encoders", "clip")) -> str:
    """Map an absolute te_file path to ComfyUI's relative model name.

    ComfyUI lists models relative to each models/* folder. Our TEs live in
    text_encoders/qwen3-4b/... -> 'qwen3-4b/...'.
    """
    p = Path(te_file)
    try:
        rel = p.relative_to(models_root)
    except ValueError:
        # Already relative or foreign path — best effort: keep as-is
        return str(p)
    parts = rel.parts
    for sd in subdirs:
        if parts and parts[0] == sd:
            parts = parts[1:]
            break
    return "/".join(parts)


def sanitize(text: str, max_len: int = 48) -> str:
    """Filesystem-safe stem for a prompt."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")[:max_len].strip("_")
    return s or "prompt"


def extract_execution_error(messages: list[list[Any]]) -> str | None:
    """Pull the human-readable execution_error out of a /history status message list.

    Status messages are [event_name, payload] pairs; the plain ``messages[:2]``
    rendering shows only the first two entries (timestamps) and hides the
    actual error. Returns None when no execution_error is present. Tolerates
    malformed entries (wrong arity / non-dict payload) by skipping them.
    """
    for msg in messages:
        try:
            name, payload = msg
        except (TypeError, ValueError):
            continue
        if name == "execution_error" and isinstance(payload, dict):
            node = f"{payload.get('node_type', '?')}(node {payload.get('node_id', '?')})"
            return f"{node}: {payload.get('exception_message', '<no message>')}".strip()
    return None


# First encode after a dynamic-VRAM (re)stage of the krea2 12-tap TE
# intermittently yields single-layer (2560) conditioning on comfy 0.33.0 —
# the DiT's width check fails LOUD (30720 expected), a re-queue re-encodes
# fresh and succeeds. Tap collection is a fixed python list, so there is no
# wrong-but-passing state: retry is safe. Targeted at this signature only.
TRANSIENT_KREA2_STAGING_ERR = "expects conditioning with 12x2560"

TRANSIENT_RETRY_ATTEMPTS = 3
TRANSIENT_RETRY_BACKOFF_S = 2.0


def _queue_once(api_url: str, wf: dict[str, Any], te_node_id: str = "") -> tuple[bytes, dict[str, Any]]:
    submit_t0 = time.monotonic()
    qresp = api(api_url, "/prompt", {"prompt": wf, "client_id": CLIENT_ID})
    # CRITICAL: ComfyUI assigns its own prompt_id — polling a self-generated
    # one never resolves (hangs until POLL_TIMEOUT). Lesson #39.
    prompt_id = qresp.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"No prompt_id in queue response: {qresp}")
    node_errors = qresp.get("node_errors") or {}
    if node_errors:
        raise RuntimeError(f"ComfyUI rejected the workflow: {json.dumps(node_errors)[:400]}")

    deadline = time.monotonic() + POLL_TIMEOUT
    while time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL)
        hist = api(api_url, f"/history/{prompt_id}")
        entry = hist.get(prompt_id)
        if entry is None:
            continue
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            messages = status.get("messages", [])
            detail = extract_execution_error(messages)
            raise RuntimeError(f"ComfyUI reported error for {prompt_id}: {detail or json.dumps(messages)[:400]}")
        outputs = entry.get("outputs", {})
        img_meta = None
        for node_out in outputs.values():
            for img in node_out.get("images", []):
                img_meta = img
                break
            if img_meta:
                break
        if img_meta is None:
            if status.get("completed"):
                raise RuntimeError(f"Prompt {prompt_id} completed without image/video outputs: {outputs}")
            continue

        # --- perf extraction from status messages ---
        msgs = dict(status.get("messages", []))
        ts_start = msgs.get("execution_start", {}).get("timestamp")
        ts_done = msgs.get("execution_success", {}).get("timestamp")
        cached_nodes = msgs.get("execution_cached", {}).get("nodes", [])
        server_gen_ms = (ts_done - ts_start) if (ts_start and ts_done) else None
        perf = {
            "prompt_id": prompt_id,
            "queue_number": (entry.get("prompt", [None])[0] if isinstance(entry.get("prompt"), list) else None),
            "server_gen_ms": server_gen_ms,
            "client_wall_ms": round((time.monotonic() - submit_t0) * 1000, 1),
            "cached_nodes": cached_nodes,
            "te_cached": (te_node_id in cached_nodes) if te_node_id else None,
            "status_str": status.get("status_str"),
        }

        subfolder = img_meta.get("subfolder", "")
        fname = img_meta["filename"]
        ftype = img_meta.get("type", "output")
        q = urllib.parse.urlencode({"filename": fname, "subfolder": subfolder, "type": ftype})
        with urllib.request.urlopen(f"{api_url.rstrip('/')}/view?{q}", timeout=120) as resp:
            data: bytes = resp.read()
        return data, perf
    raise TimeoutError(f"ComfyUI did not finish {prompt_id} within {POLL_TIMEOUT}s")


def flush_loaded_models(api_url: str) -> bool:
    """POST /free {"unload_models": true} — drop loaded models from memory.

    Used between transient retries: a failed TE load can leave a bad CLIP
    object in the node-output cache, and every re-queue would REUSE it
    (execution_cached) without re-loading. Flushing forces a fresh load.
    """
    try:
        req = urllib.request.Request(
            api_url.rstrip("/") + "/free",
            data=json.dumps({"unload_models": True}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return int(resp.status) == 200
    except (urllib.error.URLError, OSError) as e:
        log.warning("/free flush failed (non-fatal): %s", e)
        return False


def queue_and_fetch(api_url: str, wf: dict[str, Any], te_node_id: str = "") -> tuple[bytes, dict[str, Any]]:
    """Queue the workflow, wait for completion, return (output bytes, perf).

    Works for images (SaveImage) AND video (SaveVideo) — both surface their
    outputs under ``outputs[*]["images"]``; video entries carry
    ``"animated": [true]``.

    Perf data extracted from the /history status messages (the richest the
    ComfyUI API offers):
    - ``server_gen_ms``: execution_success − execution_start timestamps
    - ``te_cached``: whether the TE loader node appears in execution_cached
      (true cold/warm verification — model loaded fresh vs reused)
    - ``prompt_id`` / ``queue_number``: server-side identifiers

    Retries (up to TRANSIENT_RETRY_ATTEMPTS total) on the known-transient
    krea2 TE staging failure (see TRANSIENT_KREA2_STAGING_ERR); every other
    error — including submit-time/network errors and poll timeouts —
    propagates immediately. The retry is deliberately narrow: widening the
    except would mask real failures.
    """
    last_err: RuntimeError | None = None
    for attempt in range(1, TRANSIENT_RETRY_ATTEMPTS + 1):
        try:
            return _queue_once(api_url, wf, te_node_id=te_node_id)
        except RuntimeError as e:
            if TRANSIENT_KREA2_STAGING_ERR not in str(e):
                raise
            last_err = e
            log.warning("Transient krea2 TE staging failure (attempt %d/%d): %s", attempt, TRANSIENT_RETRY_ATTEMPTS, e)
            if attempt < TRANSIENT_RETRY_ATTEMPTS:
                # Force a fresh TE load on retry: a failed load can leave a
                # bad CLIP object in the node-output cache (execution_cached),
                # which plain re-queues would silently reuse forever.
                flush_loaded_models(api_url)
                time.sleep(TRANSIENT_RETRY_BACKOFF_S)
    raise last_err if last_err else RuntimeError("unreachable")
