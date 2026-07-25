#!/usr/bin/env python3
"""HarmBench runner — full 3-phase pipeline via Docker containers.

Phase 1: Generate — Start vLLM (or llama.cpp), send 400 harmful prompts
Phase 2: Classify — Run classifier on generated responses
Phase 3: Score — Compute Attack Success Rate (ASR)

Supports two inference backends:
  - vLLM (default): Starts vLLM OpenAI server in Docker with native bf16
  - llama.cpp: Uses GGUF files via llama-server (fallback)

All settings come from comparison.json via _load_comparison.py shell exports,
or from CLI flags which override config settings.

Usage:
    python3 run_harmbench.py ./comparisons/qwen36-35b
    python3 run_harmbench.py --gpu 0 --phase generate ./comparisons/gemma4-e4b
    python3 run_harmbench.py --backend llamacpp ./comparisons/qwen25-7b
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

ABL_ROOT = Path(__file__).resolve().parent.parent

FORENSICS_IMAGE = "abliterlitics-forensics:1.0.0"
LMEVAL_IMAGE = "abliterlitics-lmeval:1.0.0"
LLAMACPP_IMAGE = "abliterlitics-llamacpp:1.0.0"


# ---------------------------------------------------------------------------
# DB stage tracking (fire-and-forget, mirrors runners/common.sh db_stage_*)
# ---------------------------------------------------------------------------


def _db_stage(
    action: str,
    comparison: str,
    model: str,
    stage: str = "harmbench",
    *,
    result_path: str = "",
    dry_run: bool = False,
) -> None:
    """Mark a pipeline stage in the SQLite DB. Never raises — DB is best-effort."""
    if dry_run or not comparison:
        return
    cmd = [
        sys.executable,
        "-m",
        "src.db",
        f"stage-{action}",
        "--comparison",
        comparison,
        "--model",
        model,
        "--stage",
        stage,
    ]
    if result_path:
        cmd += ["--result-path", result_path]
    with contextlib.suppress(Exception):
        subprocess.run(cmd, capture_output=True, timeout=30, check=False, cwd=str(ABL_ROOT))


def _db_import_results(comp_dir: str, dry_run: bool = False) -> None:
    """Import all result files (incl. HarmBench) into the DB. Never raises."""
    if dry_run:
        return
    with contextlib.suppress(Exception):
        subprocess.run(
            [sys.executable, "-m", "src.db", "import-results", comp_dir],
            capture_output=True,
            timeout=120,
            check=False,
            cwd=str(ABL_ROOT),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def detect_best_gpu() -> int:
    """Detect GPU with most VRAM via nvidia-smi. Defaults to GPU 0 if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        lines = [ln.strip() for ln in result.stdout.strip().split("\n") if ln.strip()]
        if not lines:
            log.warning("nvidia-smi returned no output, defaulting to GPU 0")
            return 0
        best = max(lines, key=lambda ln: int(ln.split(",")[1].strip()))
        return int(best.split(",")[0].strip())
    except (FileNotFoundError, Exception):
        log.warning("nvidia-smi not available, defaulting to GPU 0")
        return 0


def allocate_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def container_bridge_ip(container_name: str, timeout: int = 60) -> str:
    """Resolve a container's IP on the default bridge network.

    We reach sibling containers by their bridge IP rather than relying on
    ``-p`` port-mapping + ``localhost``. This is robust in both normal hosts
    and nested (Docker-from-Docker) environments where published ports are
    mapped onto the *outer* host namespace and are unreachable from the
    orchestrator's own namespace. The bridge IP is routable from the host and
    from any sibling container sharing the default bridge, so both the
    in-process health check and the generate container can reach vLLM.
    Falls back to ``localhost`` if the IP cannot be resolved.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "-f",
                "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                container_name,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        ip = result.stdout.strip()
        if ip:
            return ip
        time.sleep(1)
    log.warning(
        "Could not resolve bridge IP for %s, falling back to localhost",
        container_name,
    )
    return "localhost"


def model_size_gb(model_path: str) -> float:
    """Get model size in GiB — counting SAFETENSORS FILES ONLY.

    The directory may contain GGUF, cache, or other non-model files that
    would inflate the size. We only care about the actual model weights.
    Returns 0.0 if no safetensors found or path doesn't exist.
    """
    try:
        size_cmd = (
            f"find '{model_path}' -maxdepth 1 -name '*.safetensors' -exec du -ck {{}} + 2>/dev/null | tail -1 | cut -f1"
        )
        result = subprocess.run(
            ["bash", "-c", size_cmd],
            capture_output=True,
            text=True,
        )
        kb = int(result.stdout.strip()) if result.stdout.strip() else 0
        return kb / 1024 / 1024
    except (subprocess.CalledProcessError, IndexError, ValueError) as exc:
        log.warning("Could not determine size of %s: %s", model_path, exc)
        return 0.0


def docker_stop(name: str) -> None:
    """Stop and remove a Docker container by name (idempotent)."""
    subprocess.run(["docker", "stop", name], capture_output=True, timeout=60)
    subprocess.run(["docker", "rm", name], capture_output=True, timeout=30)


def docker_logs_tail(name: str, lines: int = 30) -> str:
    """Get last N lines of a container's logs."""
    result = subprocess.run(
        ["docker", "logs", "--tail", str(lines), name],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stderr + result.stdout


def wait_for_vllm(base_url: str, timeout: int = 300) -> bool:
    """Wait for vLLM server to respond at /v1/models."""
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{base_url}/v1/models")
            resp = urllib.request.urlopen(req, timeout=5)
            if resp.status == 200:
                log.info("vLLM ready at %s", base_url)
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def wait_for_llamacpp(base_url: str, timeout: int = 120) -> bool:
    """Wait for llama-server to respond at /health."""
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{base_url}/health")
            resp = urllib.request.urlopen(req, timeout=5)
            if resp.status == 200:
                log.info("llama-server ready at %s", base_url)
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


# ---------------------------------------------------------------------------
# Phase 1: Generate
# ---------------------------------------------------------------------------


def run_generate_vllm(
    model_path: str,
    model_name: str,
    gpu: int,
    results_dir: Path,
    max_tokens: int,
    max_model_len: int,
    reasoning_parser: str,
    reasoning_config_json: str,
    dry_run: bool,
    skip_existing: bool,
    concurrent: int = 4,
) -> Path | None:
    """Phase 1 via vLLM: start server, generate responses, tear down."""
    out_file = results_dir / f"harmbench_{model_name}_responses.json"
    if skip_existing and out_file.exists():
        log.info("[skip] %s: responses exist", model_name)
        return out_file

    log.info("[generate/vllm] %s on GPU %d", model_name, gpu)
    if dry_run:
        log.info("  [dry-run] Would start vLLM server + generate responses")
        return None

    # Quantization: ALWAYS native bf16 for fair, consistent comparison conditions.
    # BNB4 quantization damages scores and makes cross-model comparison unfair.
    # Only enable BNB4 via an explicit override if reviewing persistent OOM errors.
    use_bnb = False
    quant_label = "bf16 native"

    # Allocate a free port (used as the vLLM listen port inside the container)
    port = allocate_port()
    container_name = f"vllm-hb-{model_name}-{os.getpid()}"

    # server_url is resolved after the container starts via its bridge IP — see below.

    # Pre-cleanup any stale container with same name
    docker_stop(container_name)

    # Build vLLM server command
    server_cmd: list[str] = [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "/model",
        "--dtype",
        "bfloat16",
        "--trust-remote-code",
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        "0.90",
        "--enforce-eager",
        "--port",
        str(port),
        "--host",
        "0.0.0.0",
    ]
    # Force thinking-enabled chat template from comparison dir (if present).
    # Variant models may have different/missing chat templates — this ensures
    # consistent thinking behavior across all models.
    # IMPORTANT: Do NOT fall back to another comparison's template — that's
    # model-specific and would silently produce wrong results for other models.
    # If the comparison dir has no thinking_chat_template.jinja, vLLM will use
    # whatever chat template is bundled with the model itself (which is correct).
    comp_dir_path = Path(os.environ.get("COMPARISON_DIR", ""))
    thinking_template = comp_dir_path / "thinking_chat_template.jinja"
    if thinking_template.exists():
        server_cmd.extend(["--chat-template", "/comparison/thinking_chat_template.jinja"])
        log.info("  Using thinking chat template: %s", thinking_template)
    else:
        log.info("  No thinking_chat_template.jinja in comparison dir — using model's bundled chat template")
    if use_bnb:
        server_cmd.extend(["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"])
    # Don't pass reasoning parser to vLLM — it misclassifies refusal text as "reasoning"
    # and leaves content empty. Instead, parse <think/> tags ourselves in the generate script.
    # if reasoning_parser:
    #     server_cmd.extend(["--reasoning-parser", reasoning_parser])
    # if reasoning_config_json:
    #     server_cmd.extend(["--reasoning-config", reasoning_config_json])

    log.info(
        "  Starting vLLM: %s, %s, max_len=%d",
        quant_label,
        f"reasoning={reasoning_parser or 'none'}",
        max_model_len,
    )

    # Start vLLM as detached Docker container
    docker_cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "--runtime=nvidia",
        "--shm-size=16g",
        "--ipc=host",
        "-e",
        f"NVIDIA_VISIBLE_DEVICES={gpu}",
        "-e",
        "CUDA_VISIBLE_DEVICES=0",
        "-v",
        f"{model_path}:/model:ro",
    ]
    # Mount comparison dir for thinking chat template
    if thinking_template.exists():
        docker_cmd.extend(["-v", f"{thinking_template.parent}:/comparison:ro"])
    docker_cmd.append(LMEVAL_IMAGE)
    docker_cmd.extend(server_cmd)
    subprocess.run(docker_cmd, check=True)

    # Resolve the server URL via the container's bridge IP. We deliberately do
    # NOT use `-p` port-mapping: in nested Docker environments the published
    # port lands on the outer host namespace and is unreachable from here, and
    # even on normal hosts the bridge IP is directly routable. The generate
    # container (below) also joins the default bridge so it can reach this IP.
    bridge_ip = container_bridge_ip(container_name)
    server_url = f"http://{bridge_ip}:{port}"
    log.info("  vLLM container %s at %s", container_name, server_url)

    try:
        if not wait_for_vllm(server_url, timeout=300):
            logs = docker_logs_tail(container_name)
            log.error("vLLM failed to start for %s:\n%s", model_name, logs)
            return None

        # Run harmbench_generate.py inside Docker (HTTP client, no GPU)
        src_mount = f"{ABL_ROOT}/src:/app/src:ro"
        results_mount = f"{results_dir}:/results"

        generate_cmd = [
            "docker",
            "run",
            "--rm",
            # No --network host: join the default bridge so the generate
            # container can reach vLLM by its bridge IP (and still has NAT for
            # downloading the HarmBench behaviors CSV).
            "-v",
            src_mount,
            "-v",
            results_mount,
            "-e",
            "PYTHONPATH=/app",
            FORENSICS_IMAGE,
            "/app/src/benchmark/harmbench_generate.py",
            "--base-url",
            server_url,
            "--output",
            f"/results/harmbench_{model_name}_responses.json",
            "--max-tokens",
            str(max_tokens),
            "--concurrent",
            str(concurrent),
            "--model-name",
            model_name,
        ]
        # Thinking is enabled by default — harmbench_generate.py sends
        # chat_template_kwargs={"enable_thinking": True} via the chat completions API.
        # Models without thinking support will simply ignore it (no-op).
        # The --no-thinking flag should only be used for truly non-thinking models
        # (e.g. Gemma 2/3, Llama) and can be passed via VARIANT_SKIP flags if needed.

        subprocess.run(generate_cmd, check=True)
    finally:
        log.info("  Tearing down vLLM container for %s", model_name)
        docker_stop(container_name)

    return out_file if out_file.exists() else None


def run_generate_llamacpp(
    model_path: str,
    model_name: str,
    gpu: int,
    results_dir: Path,
    gguf_dir: str,
    max_tokens: int,
    dry_run: bool,
    skip_existing: bool,
) -> Path | None:
    """Phase 1 via llama.cpp: start llama-server with GGUF, generate responses."""
    out_file = results_dir / f"harmbench_{model_name}_responses.json"
    if skip_existing and out_file.exists():
        log.info("[skip] %s: responses exist", model_name)
        return out_file

    log.info("[generate/llamacpp] %s on GPU %d", model_name, gpu)
    if dry_run:
        log.info("  [dry-run] Would start llama-server + generate responses")
        return None

    # Resolve GGUF file
    gguf_path = Path(gguf_dir) if gguf_dir else ABL_ROOT / "models_gguf"
    if not gguf_path.is_dir():
        log.error("GGUF directory not found: %s", gguf_path)
        return None

    gguf_file = None
    name_clean = model_name.lower().replace("-", "").replace("_", "")
    for f in gguf_path.iterdir():
        if f.name.endswith(".gguf") and name_clean in f.name.lower().replace("-", "").replace("_", ""):
            gguf_file = f
            break
    if gguf_file is None:
        log.error("No GGUF file found for %s in %s", model_name, gguf_path)
        return None

    port = allocate_port()
    container_name = f"llama-hb-{model_name}-{os.getpid()}"

    # server_url is resolved after the container starts via its bridge IP — see below.

    docker_stop(container_name)

    # Start llama-server
    server_cmd = [
        "-m",
        "/model.gguf",
        "--port",
        str(port),
        "--host",
        "0.0.0.0",
        "-c",
        "8192",
        "-b",
        "512",
        "-np",
        "1",
        "--reasoning-budget",
        "2048",
    ]
    docker_cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "--runtime=nvidia",
        "-e",
        f"NVIDIA_VISIBLE_DEVICES={gpu}",
        "-e",
        "CUDA_VISIBLE_DEVICES=0",
        "-v",
        f"{gguf_file}:/model.gguf:ro",
        LLAMACPP_IMAGE,
        *server_cmd,
    ]
    subprocess.run(docker_cmd, check=True)

    # Resolve server URL via the container's bridge IP (see run_generate_vllm
    # for rationale). Avoids -p port-mapping which is unreachable in nested
    # Docker environments.
    bridge_ip = container_bridge_ip(container_name)
    server_url = f"http://{bridge_ip}:{port}"
    log.info("  llama-server container %s at %s", container_name, server_url)

    try:
        if not wait_for_llamacpp(server_url, timeout=120):
            logs = docker_logs_tail(container_name)
            log.error("llama-server failed to start for %s:\n%s", model_name, logs)
            return None

        src_mount = f"{ABL_ROOT}/src:/app/src:ro"
        results_mount = f"{results_dir}:/results"

        generate_cmd = [
            "docker",
            "run",
            "--rm",
            # No --network host: join the default bridge to reach
            # llama-server by its bridge IP (NAT also available for the
            # HarmBench behaviors download).
            "-v",
            src_mount,
            "-v",
            results_mount,
            "-e",
            "PYTHONPATH=/app",
            FORENSICS_IMAGE,
            "/app/src/benchmark/harmbench_generate.py",
            "--base-url",
            server_url,
            "--output",
            f"/results/harmbench_{model_name}_responses.json",
            "--max-tokens",
            str(max_tokens),
            "--model-name",
            model_name,
        ]
        subprocess.run(generate_cmd, check=True)
    finally:
        log.info("  Tearing down llama-server container for %s", model_name)
        docker_stop(container_name)

    return out_file if out_file.exists() else None


# ---------------------------------------------------------------------------
# Phase 2: Classify
# ---------------------------------------------------------------------------


def run_classify(
    responses_file: Path,
    model_name: str,
    results_dir: Path,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> Path | None:
    """Phase 2: Classify HarmBench responses (CPU-only, no GPU)."""
    out_file = results_dir / f"harmbench_{model_name}_classified.json"
    if skip_existing and out_file.exists():
        log.info("[skip] %s: classified exists", model_name)
        return out_file

    if not responses_file.exists():
        log.error("Responses file not found: %s", responses_file)
        return None

    log.info("[classify] %s", model_name)
    if dry_run:
        log.info("  [dry-run] Would run harmbench_classify.py classify")
        return None

    src_mount = f"{ABL_ROOT}/src:/app/src:ro"
    results_mount = f"{results_dir}:/results"

    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        src_mount,
        "-v",
        results_mount,
        "-e",
        "PYTHONPATH=/app",
        FORENSICS_IMAGE,
        "/app/src/benchmark/harmbench_classify.py",
        "classify",
        "--input",
        f"/results/harmbench_{model_name}_responses.json",
        "--output",
        f"/results/harmbench_{model_name}_classified.json",
    ]
    subprocess.run(cmd, check=True)

    return out_file if out_file.exists() else None


# ---------------------------------------------------------------------------
# Phase 3: Score
# ---------------------------------------------------------------------------


def run_score(
    classified_file: Path,
    model_name: str,
    results_dir: Path,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> Path | None:
    """Phase 3: Compute ASR scores (CPU-only, no GPU)."""
    out_file = results_dir / f"harmbench_{model_name}_scores.json"
    if skip_existing and out_file.exists():
        log.info("[skip] %s: scores exist", model_name)
        return out_file

    if not classified_file.exists():
        log.error("Classified file not found: %s", classified_file)
        return None

    log.info("[score] %s", model_name)
    if dry_run:
        log.info("  [dry-run] Would run harmbench_classify.py score")
        return None

    src_mount = f"{ABL_ROOT}/src:/app/src:ro"
    results_mount = f"{results_dir}:/results"

    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        src_mount,
        "-v",
        results_mount,
        "-e",
        "PYTHONPATH=/app",
        FORENSICS_IMAGE,
        "/app/src/benchmark/harmbench_classify.py",
        "score",
        "--classified",
        f"/results/harmbench_{model_name}_classified.json",
        "--output",
        f"/results/harmbench_{model_name}_scores.json",
    ]
    subprocess.run(cmd, check=True)

    return out_file if out_file.exists() else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HarmBench evaluation runner (3-phase pipeline)",
    )
    parser.add_argument("comparison_dir", help="Path to comparison directory")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index (default: auto)")
    parser.add_argument(
        "--backend",
        choices=["vllm", "llamacpp"],
        default=None,
        help="Inference backend (default: from comparison.json or vllm)",
    )
    parser.add_argument(
        "--phase",
        choices=["generate", "classify", "score", "all"],
        default="all",
        help="Which phase to run (default: all)",
    )
    parser.add_argument("--max-tokens", type=int, default=None, help="Max generation tokens")
    parser.add_argument("--concurrent", type=int, default=4, help="Concurrent requests (default: 4)")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Read settings from environment (set by common.sh + _load_comparison.py)
    # These are available when called via abliterlitics.sh. Fallback to defaults
    # for standalone invocation.
    comp_dir = Path(args.comparison_dir).resolve()
    os.environ["COMPARISON_DIR"] = str(comp_dir)  # for run_generate_vllm
    results_dir = comp_dir / "results" / "harmbench"
    results_dir.mkdir(parents=True, exist_ok=True)

    gpu = args.gpu if args.gpu is not None else int(os.environ.get("GLOBAL_GPU", "") or detect_best_gpu())

    # Backend selection: CLI > env > comparison.json setting
    backend = args.backend
    if backend is None:
        env_backend = os.environ.get("INFERENCE_BACKEND", "auto")
        backend = "llamacpp" if env_backend in ("llamacpp", "ik_llamacpp") else "vllm"

    # Settings from environment (set by _load_comparison.py) or defaults
    max_tokens = args.max_tokens or int(os.environ.get("HARMBENCH_MAX_TOKENS", "8096"))
    max_model_len = int(os.environ.get("LM_EVAL_MAX_MODEL_LEN", "8192"))
    reasoning_parser = os.environ.get("REASONING_PARSER", "")
    reasoning_config_json = os.environ.get("REASONING_CONFIG_JSON", "")
    gguf_dir = os.environ.get("GGUF_DIR", "")
    skip = not args.force and args.skip_existing

    # Build model list from environment or fall back to reading comparison.json
    models: list[tuple[str, str, bool]] = []  # (name, path, skip_hb)

    variant_names = os.environ.get("VARIANT_NAMES", "")
    base_dir = os.environ.get("BASE_DIR", "")
    comparison_name = os.environ.get("COMPARISON_NAME", "")

    if variant_names and base_dir:
        # Called via abliterlitics.sh — env vars are set
        base_name = os.environ.get("BASE_NAME", "") or os.path.basename(os.path.normpath(base_dir))
        models.append((base_name, base_dir, False))
        for name in variant_names.split():
            var_slug = name.replace("-", "_")
            path = os.environ.get(f"VARIANT_PATH_{var_slug}", "")
            skip_hb = os.environ.get(f"VARIANT_SKIP_HB_{var_slug}", "0") == "1"
            if path:
                models.append((name, path, skip_hb))
    else:
        # Standalone invocation — read comparison.json directly
        comp_file = comp_dir / "comparison.json"
        if not comp_file.exists():
            log.error("comparison.json not found: %s", comp_dir)
            sys.exit(1)
        with open(comp_file) as f:
            comp = json.load(f)
        comparison_name = comp.get("name", comp_dir.name)
        base_dir_local = str(comp_dir / comp["base"])
        models.append((os.path.basename(os.path.normpath(base_dir_local)), base_dir_local, False))
        for name, variant in comp.get("variants", {}).items():
            skip_hb = variant.get("skip_harmbench", False)
            models.append((name, str(comp_dir / variant["path"]), skip_hb))

        # Apply inference_backend from comparison.json if not overridden by CLI
        if backend is None:
            comp_backend = comp.get("settings", {}).get("inference_backend", "auto")
            backend = "llamacpp" if comp_backend in ("llamacpp", "ik_llamacpp") else "vllm"

    # Auto-detect reasoning if env vars not set (standalone invocation)
    if not reasoning_parser and base_dir:
        sys.path.insert(0, str(ABL_ROOT))
        from runners._load_comparison import _detect_reasoning_parser

        reasoning_parser, reasoning_config_json = _detect_reasoning_parser(base_dir, None)

    log.info(
        "HarmBench pipeline: %d models, backend=%s, max_tokens=%d, GPU=%d",
        len(models),
        backend,
        max_tokens,
        gpu,
    )

    for model_name, model_path, skip_hb in models:
        if skip_hb:
            log.info("[skip] %s: skip_harmbench=true", model_name)
            _db_stage("skip", comparison_name, model_name, dry_run=args.dry_run)
            continue

        # If all phases already exist and skip is on, mark skipped in DB
        responses_file = results_dir / f"harmbench_{model_name}_responses.json"
        if skip and args.phase == "all" and responses_file.exists():
            log.info("[skip] %s: responses exist", model_name)
            _db_stage("skip", comparison_name, model_name, dry_run=args.dry_run)
            continue

        _db_stage("start", comparison_name, model_name, dry_run=args.dry_run)
        log.info("=== HarmBench for %s ===", model_name)

        if args.phase in ("generate", "all"):
            if backend == "vllm":
                run_generate_vllm(
                    model_path,
                    model_name,
                    gpu,
                    results_dir,
                    max_tokens,
                    max_model_len,
                    reasoning_parser,
                    reasoning_config_json,
                    args.dry_run,
                    skip,
                    args.concurrent,
                )
            else:
                run_generate_llamacpp(
                    model_path,
                    model_name,
                    gpu,
                    results_dir,
                    gguf_dir,
                    max_tokens,
                    args.dry_run,
                    skip,
                )

        if args.phase in ("classify", "all"):
            responses_file = results_dir / f"harmbench_{model_name}_responses.json"
            run_classify(responses_file, model_name, results_dir, args.dry_run, skip)

        score_file = results_dir / f"harmbench_{model_name}_scores.json"
        if args.phase in ("score", "all"):
            classified_file = results_dir / f"harmbench_{model_name}_classified.json"
            run_score(classified_file, model_name, results_dir, args.dry_run, skip)

        _db_stage(
            "complete",
            comparison_name,
            model_name,
            result_path=str(score_file) if score_file.exists() else "",
            dry_run=args.dry_run,
        )

    # Import all results (incl. HarmBench responses/classifications/scores) into the DB
    _db_import_results(str(comp_dir), dry_run=args.dry_run)

    log.info("HarmBench evaluation complete")


if __name__ == "__main__":
    main()
