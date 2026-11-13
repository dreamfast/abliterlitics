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
- 2x GPUs with combined ≥48 GB VRAM (for up to 27B models with TP=2, or single-GPU with FP8)

### Large Model (27B+, MoE)

- 2x GPUs with combined ≥56 GB VRAM
- Or single GPU with dynamic FP8 quantization
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

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits
# 0, NVIDIA GeForce RTX 5090, 32607
# 1, NVIDIA GeForce RTX 4090, 24564
```

### Detection Logic

1. Run `nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits`
2. Parse GPU count, names, and VRAM in the shell runners
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

GPU selection happens in the runners (no separate Python GPU module):

- `runners/run_lm_eval.sh` accepts `--gpus single|dual|0|1|0,1` (or the
  `settings.lm_eval_gpus` comparison.json key) and picks the largest-VRAM GPU by
  default, pinning it via `NVIDIA_VISIBLE_DEVICES`.
- Weight/KL/HarmBench runners accept the global `--gpu N` flag (`./abliterlitics.sh --gpu 0 ...`).

### Manual Override

```bash
# Force specific GPU
./abliterlitics.sh --gpu 0 kl ./my-comparison/   # Uses host GPU 0
./abliterlitics.sh --gpu 1 lm-eval ./my-comparison/  # Uses host GPU 1
```

---

## 4. Execution Strategies

### Decision Matrix (lm-eval runner)

| Condition | Strategy |
|---|---|
| Model fits one GPU (default) | Single GPU, BF16, `gpu_memory_utilization: 0.9` |
| `--gpus dual` (or `settings.lm_eval_gpus: "dual"`) | TP=2 across both GPUs with NCCL flags |
| Model > 50 GB on disk and no FP8 setting | Dual TP2 is auto-selected — pass `--gpus 0` to pin single-GPU explicitly |
| `settings.lm_eval_quantization: "fp8"` | Dynamic online FP8 (~half the weight footprint; effective on-disk size is halved for the GPU heuristic, so large models stay single-GPU) |
| Model exceeds available VRAM / unsupported arch | GGUF fallback (llama.cpp) |

**GPU-1 policy:** in the standard two-GPU dev setup, GPU 1 is reserved for other work.
Pin `--gpus 0` (or the comparison setting) unless the user explicitly authorizes GPU 1.

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

## 6. Quantization and GPU Memory Utilization

### Dynamic FP8 (recommended for large models)

For models that don't fit a single GPU, enable dynamic online FP8 via the comparison
settings:

```json
{
  "settings": {
    "lm_eval_quantization": "fp8",
    "lm_eval_gpus": "0"
  }
}
```

- ~2x weight reduction vs BF16 with the default online quantization path (applies
  uniformly to base and variants, so deltas stay comparable)
- Keeps mamba/norm internals unquantized; only linear layers quantize
- On tight VRAM, cap `--max-num-batched-tokens` via `settings.lm_eval_extra_vllm_flags`
  (prompt_logprobs transients scale with batched tokens × vocab)

### BitsAndBytes 4-bit (legacy)

BNB4 is legacy and no longer auto-selected — it measurably damages benchmark scores.
Prefer FP8 (above) or GGUF.

### GPU Memory Utilization

The `gpu_memory_utilization` parameter (default 0.9) controls how much VRAM vLLM allocates:

- **0.85**: Conservative, leaves room for system processes
- **0.9**: Default, good for dedicated GPU
- **0.95**: Aggressive, for maximizing throughput on single-GPU setups

---

## 7. CPU Offloading and GGUF

### When GGUF is Needed

- Model exceeds combined GPU VRAM even with FP8
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
