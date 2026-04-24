# GPU Setup

GPU configuration, topology detection, and multi-GPU strategies for Abliterlitics.

---

## Table of Contents

1. [Requirements](#1-requirements)
2. [GPU Topology Detection](#2-gpu-topology-detection)
3. [Docker GPU Index Mapping](#3-docker-gpu-index-mapping)
4. [Execution Strategies](#4-execution-strategies)
5. [Multi-GPU Configuration](#5-multi-gpu-configuration)
6. [Single-GPU Fallback](#6-single-gpu-fallback)
7. [CPU Offloading and GGUF](#7-cpu-offloading-and-gguf)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Requirements

### Minimum

- 1x NVIDIA GPU with ≥16 GB VRAM (for 2B-4B models)
- Docker with NVIDIA Container Toolkit
- CUDA 12.x compatible driver

### Recommended

- 1x NVIDIA GPU with ≥32 GB VRAM (for up to 14B models)
- 2x GPUs with combined ≥48 GB VRAM (for up to 27B models with TP=2 or BNB4)

### Large Model (27B+, MoE)

- 2x GPUs with combined ≥56 GB VRAM
- Or single GPU with BitsAndBytes 4-bit quantization
- Or GGUF + llama.cpp fallback (works with CPU+GPU hybrid)

### Software

- **Docker** with NVIDIA Container Toolkit (`nvidia-container-toolkit` package)
- **NVIDIA driver** 535+ (for CUDA 12.x)
- **Python 3.10+** on host (for shell orchestration only)

Verify NVIDIA runtime:
```bash
docker info | grep -i runtime
# Should show: nvidia
```

---

## 2. GPU Topology Detection

Abliterlitics auto-detects GPU topology via `nvidia-smi`:

```
$ python3 -c "from src.gpu import detect_gpus; [print(g) for g in detect_gpus()]"
GPUInfo(index=0, name='NVIDIA GeForce RTX 5090', vram_mb=32768, ...)
GPUInfo(index=1, name='NVIDIA GeForce RTX 4090', vram_mb=24576, ...)
```

### Detection Logic

1. Run `nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits`
2. Parse results into `GPUInfo` dataclass
3. Map host GPU indices to Docker indices (see below)

### What Gets Detected

- GPU count and model names
- Total VRAM per GPU
- GPU topology (PCIe vs. NVLink) — currently PCIe (PHB) only

---

## 3. Docker GPU Index Mapping

### The Problem

With `--runtime=nvidia`, Docker **reverses** the GPU index order compared to the host:

| Host Index | GPU | Docker `cuda:0` | Docker `cuda:1` |
|---|---|---|---|
| 0 | RTX 5090 (32 GB) | — | RTX 5090 |
| 1 | RTX 4090 (24 GB) | RTX 4090 | — |

This means without correction, the smaller GPU gets `cuda:0` (the default device).

### The Solution

Abliterlitics uses `CUDA_VISIBLE_DEVICES` to remap indices:

```bash
# To make the 5090 = cuda:0 inside Docker:
CUDA_VISIBLE_DEVICES=0,1  # Host index 0 first → Docker cuda:0

# To make the 4090 = cuda:0 inside Docker:
CUDA_VISIBLE_DEVICES=1,0  # Host index 1 first → Docker cuda:0
```

The `select_strategy()` function in `gpu.py` automatically:
1. Sorts GPUs by VRAM descending
2. Sets `CUDA_VISIBLE_DEVICES` so the largest GPU becomes `cuda:0`
3. Passes `NVIDIA_VISIBLE_DEVICES` for the Docker GPU filter

### Manual Override

```bash
# Force specific GPU
./abliterlitics.sh --gpu 0 kl ./my-comparison/   # Uses host GPU 0
./abliterlitics.sh --gpu 1 lm-eval ./my-comparison/  # Uses host GPU 1
```

---

## 4. Execution Strategies

The `select_strategy()` function decides how to run based on model size and available VRAM:

### Decision Matrix

| Model Size | Available VRAM | Strategy |
|---|---|---|
| < 8 GB | ≥ 16 GB single GPU | Single GPU, BF16 |
| 8-16 GB | ≥ 16 GB single GPU | Single GPU, BF16 |
| 16-30 GB | ≥ 32 GB single GPU | Single GPU, BF16 |
| 16-30 GB | 16-31 GB single GPU | Single GPU, BNB4 quantization |
| 16-52 GB | ≥ 48 GB combined | TP=2 across both GPUs |
| > 52 GB | < 48 GB combined | GGUF fallback (llama.cpp) |

### Strategy Output

```python
{
    "single_gpu": True,
    "gpu": GPUInfo(index=0, name='RTX 5090', vram_mb=32768),
    "use_gguf": False,
    "docker_gpu_args": ["--runtime=nvidia", "-e", "NVIDIA_VISIBLE_DEVICES=0", ...],
    "gpu_memory_utilization": 0.9,
    "tensor_parallel": 1,
    "nccl_flags": {},
    "needs_conversion": False,
}
```

---

## 5. Multi-GPU Configuration

### Tensor Parallelism (TP=2)

When a model exceeds single-GPU VRAM but fits in combined VRAM:

```bash
docker run --rm --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=0,1 \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e NCCL_P2P_DISABLE=1 \
  -e NCCL_IB_DISABLE=1 \
  ...
```

### NCCL Flags (Required for PCIe GPUs)

On PCIe-connected GPUs (PHB topology) without NVLink:

- **`NCCL_P2P_DISABLE=1`**: Disables peer-to-peer transfer (not available on PCIe)
- **`NCCL_IB_DISABLE=1`**: Disables InfiniBand (not available on consumer GPUs)

These flags prevent NCCL hangs during multi-GPU communication. Without them, TP=2 will hang indefinitely on consumer hardware.

### TP=2 Performance

Expect ~30% throughput penalty compared to single-GPU due to:
- PCIe communication overhead
- No P2P or NVLink
- Gradient synchronization latency

---

## 6. Single-GPU Fallback

### BitsAndBytes 4-bit Quantization

For models that don't quite fit in a single GPU:

```python
from transformers import BitsAndBytesConfig
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

**Impact:**
- Reduces VRAM usage by ~4x
- Minimal accuracy impact on benchmarks (<1% on MMLU)
- Required for 27B models on 32 GB GPUs

**When auto-selected:**
- Model size > single GPU VRAM * 0.85
- Model size ≤ single GPU VRAM * 4 (BNB4 can compress ~4x)

### GPU Memory Utilization

The `gpu_memory_utilization` parameter (default 0.9) controls how much VRAM vLLM allocates:

- **0.85**: Conservative, leaves room for system processes
- **0.9**: Default, good for dedicated GPU
- **0.95**: Aggressive, for maximizing throughput on single-GPU setups

---

## 7. CPU Offloading and GGUF

### When GGUF is Needed

- Model exceeds combined GPU VRAM even with BNB4
- Model has compatibility issues with vLLM (e.g., specific architecture support)
- Running on limited hardware

### GGUF Conversion

Abliterlitics can auto-convert safetensors to GGUF:

```bash
# Manual conversion
docker run --rm --runtime=nvidia \
  -v /path/to/model:/model:ro \
  -v /path/to/gguf-output:/output \
  abliterlitics-llamacpp:1.0.0 \
  python3 /opt/llama.cpp/convert_hf_to_gguf.py /model --outfile /output/model-Q4_K_M.gguf --outtype q4_k_m
```

### Inference Backends

| Backend | Image | Speed | Notes |
|---|---|---|---|
| llama.cpp | `abliterlitics-llamacpp:1.0.0` | Baseline | Official pre-built binary |
| ik_llama.cpp | `abliterlitics-ik-llamacpp:1.0.0` | ~16% faster | Custom build, pinned commit |

### GGUF Limitations

- **lm-eval**: Requires logprobs proxy for loglikelihood tasks (may slightly affect accuracy)
- **KL divergence**: Not supported (requires direct model access for logits)
- **HarmBench**: Fully supported via OpenAI-compatible API

---

## 8. Troubleshooting

### "No GPUs detected"

```
RuntimeError: No GPUs detected. Cannot select strategy.
```

**Fix:** Verify NVIDIA Container Toolkit is installed:
```bash
nvidia-container-cli --version
docker run --rm --runtime=nvidia ubuntu nvidia-smi
```

### OOM (Out of Memory)

If a run crashes with CUDA OOM:
1. Check `results_version` in the output — incomplete results are flagged
2. Try `--backend llamacpp` to use GGUF quantization
3. Reduce `gpu_memory_utilization` in `comparison.json` settings
4. For lm-eval: reduce `max_model_len` (e.g., 2048 instead of 4096)

### GPU Index Mismatch

If the wrong GPU is selected:
1. Check `./abliterlitics.sh status` to see detected topology
2. Use `--gpu N` to force a specific host GPU index
3. Check Docker GPU mapping with:
   ```bash
   docker run --rm --runtime=nvidia -e CUDA_VISIBLE_DEVICES=0,1 \
     ubuntu bash -c "nvidia-smi --query-gpu=index,name --format=csv"
   ```

### NCCL Hang

If TP=2 hangs indefinitely:
1. Verify `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` are set
2. Check GPU topology: `nvidia-smi topo -m` — should show PHB (PCIe)
3. If topology shows NVLink, remove the NCCL flags (they're only for PCIe)

### Slow GGUF Conversion

GGUF conversion for large models can take 10-30 minutes. The conversion is cached — subsequent runs reuse the existing GGUF file.
