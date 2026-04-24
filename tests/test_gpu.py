"""Tests for src/gpu.py — GPU detection, Docker index mapping, strategy selection."""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.gpu import GPUInfo, detect_gpus, get_gpu_info_by_docker, get_gpu_info_by_host, select_strategy


# ---------------------------------------------------------------------------
# GPUInfo
# ---------------------------------------------------------------------------

class TestGPUInfo:
    def test_gpuinfo_creation(self):
        gpu = GPUInfo(host_index=0, name="RTX 5090", vram_mb=32768, docker_index=1, compute_capability="12.0")
        assert gpu.host_index == 0
        assert gpu.docker_index == 1
        assert gpu.vram_mb == 32768


# ---------------------------------------------------------------------------
# detect_gpus — mock nvidia-smi
# ---------------------------------------------------------------------------

DUAL_GPU_SMI = "0, NVIDIA GeForce RTX 5090, 32768, 12.0\n1, NVIDIA GeForce RTX 4090, 24576, 8.9"
SINGLE_GPU_SMI = "0, NVIDIA GeForce RTX 5090, 32768, 12.0"


class TestDetectGpus:
    @patch("src.gpu.subprocess.run")
    def test_dual_gpu_detection(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=DUAL_GPU_SMI,
            stderr="",
        )
        gpus = detect_gpus()
        assert len(gpus) == 2

        # Host 0 (5090) -> Docker 1
        assert gpus[0].host_index == 0
        assert gpus[0].docker_index == 1
        assert gpus[0].vram_mb == 32768

        # Host 1 (4090) -> Docker 0
        assert gpus[1].host_index == 1
        assert gpus[1].docker_index == 0
        assert gpus[1].vram_mb == 24576

    @patch("src.gpu.subprocess.run")
    def test_single_gpu_detection(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=SINGLE_GPU_SMI,
            stderr="",
        )
        gpus = detect_gpus()
        assert len(gpus) == 1
        assert gpus[0].docker_index == 0  # single GPU: docker = host

    @patch("src.gpu.subprocess.run", side_effect=FileNotFoundError)
    def test_no_nvidia_smi(self, mock_run):
        gpus = detect_gpus()
        assert gpus == []

    @patch("src.gpu.subprocess.run")
    def test_nvidia_smi_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        gpus = detect_gpus()
        assert gpus == []


# ---------------------------------------------------------------------------
# Helper lookups
# ---------------------------------------------------------------------------

def _make_gpus():
    return [
        GPUInfo(host_index=0, name="RTX 5090", vram_mb=32768, docker_index=1, compute_capability="12.0"),
        GPUInfo(host_index=1, name="RTX 4090", vram_mb=24576, docker_index=0, compute_capability="8.9"),
    ]


class TestGetGpuInfo:
    def test_by_host(self):
        gpus = _make_gpus()
        gpu = get_gpu_info_by_host(gpus, 0)
        assert gpu is not None
        assert gpu.name == "RTX 5090"

    def test_by_host_missing(self):
        gpus = _make_gpus()
        assert get_gpu_info_by_host(gpus, 99) is None

    def test_by_docker(self):
        gpus = _make_gpus()
        gpu = get_gpu_info_by_docker(gpus, 0)
        assert gpu is not None
        assert gpu.name == "RTX 4090"

    def test_by_docker_missing(self):
        gpus = _make_gpus()
        assert get_gpu_info_by_docker(gpus, 99) is None


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------

class TestSelectStrategy:
    gpus = _make_gpus()

    def test_no_gpus_raises(self):
        with pytest.raises(RuntimeError, match="No GPUs"):
            select_strategy(10.0, [], "weight")

    def test_small_model_weight_single_gpu(self):
        """4GB model on weight runner -> single GPU."""
        s = select_strategy(4.0, self.gpus, "weight")
        assert s["single_gpu"] is True
        assert s["use_gguf"] is False
        assert s["tensor_parallel"] == 1
        assert s["gpu"].name == "RTX 5090"  # largest GPU

    def test_small_model_lm_eval_single_gpu(self):
        """10GB model on lm_eval -> single GPU."""
        s = select_strategy(10.0, self.gpus, "lm_eval")
        assert s["single_gpu"] is True
        assert s["use_gguf"] is False
        assert s["tensor_parallel"] == 1

    def test_large_model_weight_cpu_offload(self):
        """Huge model on weight runner -> single GPU (CPU offload)."""
        s = select_strategy(60.0, self.gpus, "weight")
        assert s["single_gpu"] is True
        assert s["use_gguf"] is False

    def test_huge_model_gguf_needed(self):
        """80GB model on lm_eval -> GGUF conversion."""
        s = select_strategy(80.0, self.gpus, "lm_eval")
        assert s["use_gguf"] is True
        assert s["needs_conversion"] is True

    def test_nccl_flags_on_multi_gpu(self):
        """Multi-GPU strategies include NCCL flags."""
        s = select_strategy(40.0, self.gpus, "lm_eval")
        if not s["single_gpu"]:
            assert "NCCL_P2P_DISABLE" in s["nccl_flags"]
            assert s["nccl_flags"]["NCCL_P2P_DISABLE"] == "1"

    def test_single_gpu_no_nccl(self):
        """Single GPU strategies have empty NCCL flags."""
        s = select_strategy(4.0, self.gpus, "kl")
        assert s["nccl_flags"] == {}

    def test_docker_gpu_args_format(self):
        """Docker GPU args are properly formatted."""
        s = select_strategy(4.0, self.gpus, "weight")
        args = s["docker_gpu_args"]
        assert any("NVIDIA_VISIBLE_DEVICES" in a for a in args)
        assert any("CUDA_VISIBLE_DEVICES" in a for a in args)
