"""Docker command builder for Abliterlitics runners."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

# Image names (versioned)
IMAGES: dict[str, str] = {
    "forensics": "abliterlitics-forensics:1.0.0",
    "lmeval": "abliterlitics-lmeval:1.0.0",
    "llamacpp": "abliterlitics-llamacpp:1.0.0",
    "ik_llamacpp": "abliterlitics-ik-llamacpp:1.0.0",
}

# Environment variable keys that should be redacted in logs
SENSITIVE_ENV_KEYS = frozenset(
    {
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "API_KEY",
        "SECRET",
        "PASSWORD",
        "TOKEN",
    }
)


def _redact_cmd(cmd: list[str]) -> list[str]:
    """Redact sensitive env var values from command list for logging."""
    redacted = []
    i = 0
    while i < len(cmd):
        if cmd[i] == "-e" and i + 1 < len(cmd):
            env_val = cmd[i + 1]
            key = env_val.split("=", 1)[0] if "=" in env_val else env_val
            if any(s in key.upper() for s in SENSITIVE_ENV_KEYS):
                redacted.extend([cmd[i], f"{key}=***"])
            else:
                redacted.extend([cmd[i], cmd[i + 1]])
            i += 2
        else:
            redacted.append(cmd[i])
            i += 1
    return redacted


def build_docker_run_cmd(
    image: str,
    gpu_args: list[str],
    mounts: list[tuple[str, str, str]],  # (host_path, container_path, mode)
    env: dict[str, str] | None = None,
    command: list[str] | None = None,
    ipc: str = "",  # "host" for lm-eval
    shm_size: str = "",  # "16g" for lm-eval
    workdir: str = "/app",
) -> list[str]:
    """Build a complete docker run command as a list (for subprocess).

    All model dirs mounted :ro. Results dir mounted :rw.
    Uses --runtime=nvidia (never --gpus).
    """
    cmd: list[str] = [
        "docker",
        "run",
        "--rm",
        "--runtime=nvidia",
    ]

    # GPU environment variables
    for arg in gpu_args:
        cmd.extend(["-e", arg])

    # Mounts
    for host_path, container_path, mode in mounts:
        cmd.extend(["-v", f"{host_path}:{container_path}:{mode}"])

    # Environment variables
    if env:
        for key, value in env.items():
            cmd.extend(["-e", f"{key}={value}"])

    # IPC mode (needed for lm-eval with vLLM)
    if ipc:
        cmd.extend(["--ipc", ipc])

    # Shared memory (needed for vLLM/lm-eval)
    if shm_size:
        cmd.extend(["--shm-size", shm_size])

    # Working directory
    if workdir:
        cmd.extend(["-w", workdir])

    # Image
    cmd.append(image)

    # Command
    if command:
        cmd.extend(command)

    return cmd


def run_docker(
    image: str,
    gpu_args: list[str],
    mounts: list[tuple[str, str, str]],
    env: dict[str, str] | None = None,
    command: list[str] | None = None,
    ipc: str = "",
    shm_size: str = "",
    workdir: str = "/app",
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Build and execute a docker run command."""
    cmd = build_docker_run_cmd(
        image=image,
        gpu_args=gpu_args,
        mounts=mounts,
        env=env,
        command=command,
        ipc=ipc,
        shm_size=shm_size,
        workdir=workdir,
    )
    # Redact sensitive env vars from log output
    safe_cmd = _redact_cmd(cmd)
    log.info("Running: %s", " ".join(safe_cmd))
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def build_image(
    image_name: str,
    dockerfile_path: Path,
    build_context: Path | None = None,
) -> None:
    """Build a Docker image from its Dockerfile.

    Streams build output to the terminal (no buffering).
    Raises CalledProcessError on failure.
    """
    if build_context is None:
        build_context = dockerfile_path.parent

    cmd = [
        "docker",
        "build",
        "-t",
        image_name,
        "-f",
        str(dockerfile_path),
        str(build_context),
    ]
    log.info("Building image: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def image_exists(image_name: str) -> bool:
    """Check if a Docker image exists locally."""
    result = subprocess.run(
        ["docker", "image", "inspect", image_name],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def ensure_image(image_key: str) -> str:
    """Get image name, warn if not built yet."""
    image_name = IMAGES.get(image_key, image_key)
    if not image_exists(image_name):
        log.warning("Docker image not found: %s. Run './abliterlitics.sh build' first.", image_name)
    return image_name
