"""
GPU topology detection: query nvidia-smi, map host <-> Docker indices,
decide execution strategy based on model size and available hardware.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about a single GPU."""

    host_index: int  # Host nvidia-smi index
    name: str  # e.g. "NVIDIA GeForce RTX 5090"
    vram_mb: int  # Total VRAM in MB
    docker_index: int  # Index inside Docker (host order reversed)
    compute_capability: str  # e.g. "12.0", "8.9"


def detect_gpus() -> list[GPUInfo]:
    """Query nvidia-smi, return GPU list with host <-> Docker mapping.

    Docker reverses the CUDA device order when using --runtime=nvidia
    with NVIDIA_VISIBLE_DEVICES=all. So host GPU 0 = Docker cuda:1
    and host GPU 1 = Docker cuda:0.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        log.warning("nvidia-smi not found — no GPUs detected")
        return []

    if result.returncode != 0:
        log.warning("nvidia-smi failed: %s", result.stderr.strip())
        return []

    gpus: list[GPUInfo] = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        host_idx = int(parts[0])
        gpus.append(
            GPUInfo(
                host_index=host_idx,
                name=parts[1],
                vram_mb=int(parts[2]),
                docker_index=-1,  # filled below
                compute_capability=parts[3],
            )
        )

    # Docker reverses the order
    n = len(gpus)
    for gpu in gpus:
        gpu.docker_index = n - 1 - gpu.host_index

    for gpu in gpus:
        log.info(
            "GPU %d: %s (%d MB, cc %s) -> Docker cuda:%d",
            gpu.host_index,
            gpu.name,
            gpu.vram_mb,
            gpu.compute_capability,
            gpu.docker_index,
        )

    return gpus


def select_strategy(
    model_size_gb: float,
    gpus: list[GPUInfo],
    runner: str,  # "weight" | "kl" | "lm_eval" | "harmbench"
) -> dict[str, Any]:
    """Decide execution strategy based on model size and GPU availability.

    Returns dict with:
      - single_gpu: bool
      - gpu: GPUInfo | None (selected GPU for single-GPU)
      - use_gguf: bool
      - docker_gpu_args: list[str]
      - gpu_memory_utilization: float
      - tensor_parallel: int
      - nccl_flags: dict[str, str]
      - needs_conversion: bool
    """
    if not gpus:
        raise RuntimeError("No GPUs detected. Cannot select strategy.")

    total_vram_mb = sum(g.vram_mb for g in gpus)
    largest_gpu = max(gpus, key=lambda g: g.vram_mb)
    model_size_mb = model_size_gb * 1024

    # Sort GPUs by host index descending so largest VRAM GPU maps to cuda:0 in Docker
    # With --runtime=nvidia, Docker reverses CUDA order, so we need to set
    # CUDA_VISIBLE_DEVICES explicitly to get the right GPU as cuda:0
    host_indices_desc = ",".join(str(g.host_index) for g in sorted(gpus, key=lambda g: g.vram_mb, reverse=True))

    # Default NCCL flags for multi-GPU
    nccl_flags: dict[str, str] = {}
    if len(gpus) > 1:
        nccl_flags = {"NCCL_P2P_DISABLE": "1", "NCCL_IB_DISABLE": "1"}

    # Strategy selection based on runner and model size
    if runner in ("weight", "kl"):
        # Weight analysis and KL use transformers (not vLLM)
        # They handle their own device mapping
        if largest_gpu and model_size_mb < largest_gpu.vram_mb * 0.85:
            # Single GPU
            return {
                "single_gpu": True,
                "gpu": largest_gpu,
                "use_gguf": False,
                "docker_gpu_args": [
                    f"NVIDIA_VISIBLE_DEVICES={largest_gpu.host_index}",
                    "CUDA_VISIBLE_DEVICES=0",
                ],
                "gpu_memory_utilization": 0.9,
                "tensor_parallel": 1,
                "nccl_flags": {},
                "needs_conversion": False,
            }
        elif len(gpus) > 1 and model_size_mb < total_vram_mb * 0.7:
            # Multi-GPU
            return {
                "single_gpu": False,
                "gpu": None,
                "use_gguf": False,
                "docker_gpu_args": [
                    "NVIDIA_VISIBLE_DEVICES=all",
                    f"CUDA_VISIBLE_DEVICES={host_indices_desc}",
                ],
                "gpu_memory_utilization": 0.9,
                "tensor_parallel": len(gpus),
                "nccl_flags": nccl_flags,
                "needs_conversion": False,
            }
        else:
            # CPU offload (large models)
            return {
                "single_gpu": True,
                "gpu": largest_gpu,
                "use_gguf": False,
                "docker_gpu_args": [
                    f"NVIDIA_VISIBLE_DEVICES={largest_gpu.host_index}",
                    "CUDA_VISIBLE_DEVICES=0",
                ],
                "gpu_memory_utilization": 0.9,
                "tensor_parallel": 1,
                "nccl_flags": {},
                "needs_conversion": False,
            }

    # Benchmarks (lm_eval, harmbench) — use vLLM or llama.cpp
    # BNB4 can fit ~52GB on a single 32GB GPU
    bnb4_threshold_gb = largest_gpu.vram_mb / 1024 * 1.6 if largest_gpu else 30

    if model_size_gb < 15:
        # Small model — single GPU, vLLM
        return {
            "single_gpu": True,
            "gpu": largest_gpu,
            "use_gguf": False,
            "docker_gpu_args": [
                f"NVIDIA_VISIBLE_DEVICES={largest_gpu.host_index}",
                "CUDA_VISIBLE_DEVICES=0",
            ],
            "gpu_memory_utilization": 0.85,
            "tensor_parallel": 1,
            "nccl_flags": {},
            "needs_conversion": False,
        }
    elif model_size_gb < bnb4_threshold_gb:
        # Medium-large — single GPU with BNB4 or high utilization
        return {
            "single_gpu": True,
            "gpu": largest_gpu,
            "use_gguf": False,
            "docker_gpu_args": [
                f"NVIDIA_VISIBLE_DEVICES={largest_gpu.host_index}",
                "CUDA_VISIBLE_DEVICES=0",
            ],
            "gpu_memory_utilization": 0.9,
            "tensor_parallel": 1,
            "nccl_flags": {},
            "needs_conversion": False,
        }
    elif model_size_mb < total_vram_mb * 0.7 and len(gpus) > 1:
        # Multi-GPU with tensor parallel
        return {
            "single_gpu": False,
            "gpu": None,
            "use_gguf": False,
            "docker_gpu_args": [
                "NVIDIA_VISIBLE_DEVICES=all",
                f"CUDA_VISIBLE_DEVICES={host_indices_desc}",
            ],
            "gpu_memory_utilization": 0.9,
            "tensor_parallel": len(gpus),
            "nccl_flags": nccl_flags,
            "needs_conversion": False,
        }
    else:
        # Too large — needs GGUF conversion and llama.cpp
        return {
            "single_gpu": True,
            "gpu": largest_gpu,
            "use_gguf": True,
            "docker_gpu_args": [
                f"NVIDIA_VISIBLE_DEVICES={largest_gpu.host_index}",
                "CUDA_VISIBLE_DEVICES=0",
            ],
            "gpu_memory_utilization": 0.95,
            "tensor_parallel": 1,
            "nccl_flags": {},
            "needs_conversion": True,
        }


def get_gpu_info_by_host(gpus: list[GPUInfo], host_index: int) -> GPUInfo | None:
    """Look up GPU info by host index."""
    for gpu in gpus:
        if gpu.host_index == host_index:
            return gpu
    return None


def get_gpu_info_by_docker(gpus: list[GPUInfo], docker_index: int) -> GPUInfo | None:
    """Look up GPU info by Docker index."""
    for gpu in gpus:
        if gpu.docker_index == docker_index:
            return gpu
    return None
