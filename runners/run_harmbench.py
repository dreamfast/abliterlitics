#!/usr/bin/env python3
"""HarmBench runner — full 3-phase pipeline via Docker containers.

Phase 1: Generate — Start inference server, send 400 harmful prompts
Phase 2: Classify — Run classifier on generated responses
Phase 3: Score — Compute Attack Success Rate (ASR)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
ABL_ROOT = SCRIPT_DIR.parent

FORENSICS_IMAGE = "abliterlitics-forensics:1.0.0"
LLAMACPP_IMAGE = "abliterlitics-llamacpp:1.0.0"


def load_comparison(comp_dir: str) -> tuple[dict, Path]:
    """Load comparison.json, return (data, comparison_dir)."""
    comp_path = Path(comp_dir)
    if comp_path.is_file():
        comp_file = comp_path
        comp_dir_resolved = comp_path.parent
    else:
        comp_file = comp_path / "comparison.json"
        comp_dir_resolved = comp_path
    if not comp_file.exists():
        print(f"ERROR: comparison.json not found: {comp_dir}", file=sys.stderr)
        sys.exit(1)
    with open(comp_file) as f:
        return json.load(f), comp_dir_resolved


def detect_best_gpu() -> int:
    """Detect GPU with most VRAM."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
    best = max(lines, key=lambda l: int(l.split(",")[1].strip()))
    return int(best.split(",")[0].strip())


def docker_run_bg(gpu: int, image: str, args: list[str], command: list[str]) -> subprocess.Popen:
    """Start a Docker container in the background. Returns Popen object."""
    full_cmd = [
        "docker", "run", "--rm", "--runtime=nvidia",
        "-e", f"NVIDIA_VISIBLE_DEVICES={gpu}",
        "-e", "CUDA_VISIBLE_DEVICES=0",
        *args,
        image,
        *command,
    ]
    log.info("Starting: %s", " ".join(full_cmd))
    return subprocess.Popen(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def wait_for_server(base_url: str, timeout: int = 300) -> bool:
    """Wait for inference server to be ready."""
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{base_url}/health")
            resp = urllib.request.urlopen(req, timeout=5)
            if resp.status == 200:
                log.info("Server ready at %s", base_url)
                return True
        except Exception:
            pass
        time.sleep(5)
    log.error("Server failed to start within %d seconds", timeout)
    return False


def run_generate(
    model_path: str,
    model_name: str,
    gpu: int,
    results_dir: Path,
    max_tokens: int = 2048,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> Path | None:
    """Phase 1: Generate HarmBench responses via llama-server."""
    out_file = results_dir / f"harmbench_{model_name}_responses.json"
    if skip_existing and out_file.exists():
        log.info("[skip] %s: responses exist", model_name)
        return out_file

    log.info("[generate] %s on GPU %d", model_name, gpu)
    if dry_run:
        log.info("  [dry-run] Would start llama-server + generate responses")
        return None

    # Start inference server in background container
    container_port = 8080
    server_url = f"http://localhost:{container_port}"

    # Resolve GGUF file path — model_path is the safetensors dir, GGUF is in models_gguf/
    gguf_dir = ABL_ROOT / "models_gguf"
    gguf_file = None
    for f in gguf_dir.iterdir():
        if f.name.endswith(".gguf") and model_name in f.name.lower().replace("-", "").replace("_", ""):
            gguf_file = f
            break
    # Fallback: try matching by variant key in filename
    if gguf_file is None:
        for f in gguf_dir.iterdir():
            if f.name.endswith(".gguf"):
                gguf_file = f
                break
    if gguf_file is None:
        log.error("No GGUF file found for %s in %s", model_name, gguf_dir)
        return None

    proc = docker_run_bg(
        gpu,
        LLAMACPP_IMAGE,
        [
            "-v", f"{gguf_file}:/model.gguf:ro",
            "-p", f"{container_port}:{container_port}",
        ],
        [
            "-m", "/model.gguf", "--port", str(container_port),
            "--host", "0.0.0.0",
            "-c", "8192", "-b", "512",
            "-np", "1",
            "--fit", "on",
            "--reasoning-budget", "2048",
        ],
    )

    try:
        if not wait_for_server(server_url):
            log.error("Server startup failed for %s", model_name)
            return None

        # Run generation script — no GPU needed, just HTTP client
        src_mount = f"{ABL_ROOT}/src:/app/src:ro"
        results_mount = f"{results_dir}:/results"

        cmd = [
            "docker", "run", "--rm",
            "--network", "host",
            "-v", src_mount,
            "-v", results_mount,
            "-e", "PYTHONPATH=/app",
            FORENSICS_IMAGE,
            "/app/src/benchmark/harmbench_generate.py",
            "--base-url", server_url,
            "--output", f"/results/harmbench_{model_name}_responses.json",
            "--max-tokens", str(max_tokens),
            "--model-name", model_name,
        ]
        subprocess.run(cmd, check=True)
    finally:
        # Kill the server container
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()

    return out_file if out_file.exists() else None


def run_classify(
    responses_file: Path,
    model_name: str,
    gpu: int,
    results_dir: Path,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> Path | None:
    """Phase 2: Classify HarmBench responses."""
    out_file = results_dir / f"harmbench_{model_name}_classified.json"
    if skip_existing and out_file.exists():
        log.info("[skip] %s: classified exists", model_name)
        return out_file

    if not responses_file.exists():
        log.error("Responses file not found: %s", responses_file)
        return None

    log.info("[classify] %s", model_name)
    if dry_run:
        log.info("  [dry-run] Would run harmbench_classify.py")
        return None

    src_mount = f"{ABL_ROOT}/src:/app/src:ro"
    results_mount = f"{results_dir}:/results"

    cmd = [
        "docker", "run", "--rm",
        "-v", src_mount,
        "-v", results_mount,
        "-e", "PYTHONPATH=/app",
        FORENSICS_IMAGE,
        "/app/src/benchmark/harmbench_classify.py", "classify",
        "--input", f"/results/harmbench_{model_name}_responses.json",
        "--output", f"/results/harmbench_{model_name}_classified.json",
    ]
    subprocess.run(cmd, check=True)

    return out_file if out_file.exists() else None


def run_score(
    classified_file: Path,
    model_name: str,
    results_dir: Path,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> Path | None:
    """Phase 3: Compute ASR scores."""
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
        "docker", "run", "--rm",
        "-v", src_mount,
        "-v", results_mount,
        "-e", "PYTHONPATH=/app",
        FORENSICS_IMAGE,
        "/app/src/benchmark/harmbench_classify.py", "score",
        "--classified", f"/results/harmbench_{model_name}_classified.json",
        "--output", f"/results/harmbench_{model_name}_scores.json",
    ]
    subprocess.run(cmd, check=True)

    return out_file if out_file.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="HarmBench evaluation runner (3-phase pipeline)")
    parser.add_argument("comparison_dir", help="Path to comparison directory")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index (default: auto)")
    parser.add_argument("--phase", choices=["generate", "classify", "score", "all"], default="all",
                        help="Which phase to run (default: all)")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    comp, comp_dir = load_comparison(args.comparison_dir)
    results_dir = comp_dir / "results" / "harmbench"
    results_dir.mkdir(parents=True, exist_ok=True)

    gpu = args.gpu if args.gpu is not None else detect_best_gpu()
    skip = not args.force and args.skip_existing

    # Build model list: base + variants
    models = [("base", str(comp_dir / comp["base"]))]
    for name, variant in comp.get("variants", {}).items():
        models.append((name, str(comp_dir / variant["path"])))

    for model_name, model_path in models:
        log.info("=== HarmBench for %s ===", model_name)

        if args.phase in ("generate", "all"):
            responses = run_generate(
                model_path, model_name, gpu, results_dir,
                args.max_tokens, args.dry_run, skip,
            )

        if args.phase in ("classify", "all"):
            responses_file = results_dir / f"harmbench_{model_name}_responses.json"
            classified = run_classify(
                responses_file, model_name, gpu, results_dir,
                args.dry_run, skip,
            )

        if args.phase in ("score", "all"):
            classified_file = results_dir / f"harmbench_{model_name}_classified.json"
            run_score(
                classified_file, model_name, results_dir,
                args.dry_run, skip,
            )

    log.info("HarmBench evaluation complete")


if __name__ == "__main__":
    main()
