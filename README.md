<div align="center">
  <img src="https://murmur.dreamfast.solutions/abliterlitics.webp" alt="Abliterlitics" width="480">
</div>

Comparative forensic analysis of LLM abliteration techniques.

## What It Does

Abliterlitics measures structural and behavioral differences between base LLMs and their
"abliterated" (safety-removed) variants. It compares abliteration techniques across multiple
model architectures using four analysis axes:

1. **Weight Analysis** — Structural comparison of modified tensors, SVD decomposition, subspace alignment, technique fingerprinting
2. **KL Divergence** — Measures collateral damage of abliteration by comparing full-vocabulary log probability distributions (methodology matches [Heretic](https://github.com/p-e-w/heretic))
3. **Capability Benchmarks** — lm-evaluation-harness (8 tasks: MMLU, GSM8K, HellaSwag, ARC-Challenge, WinoGrande, TruthfulQA, PiQA, Lambada)
4. **Safety Benchmarks** — HarmBench (400 prompts) to measure Attack Success Rate

## Quick Start

### Prerequisites

- Docker with NVIDIA Container Toolkit (`--runtime=nvidia` support)
- NVIDIA GPU(s) with sufficient VRAM for target models
- 50 GB+ disk space for results
- Python 3.10+ (host, for shell orchestration only)

### 1. Build Docker Images

```bash
cd abliterlitics/
docker build -t abliterlitics-forensics:1.0.0 -f docker/Dockerfile.forensics .
docker build -t abliterlitics-lmeval:1.0.0 -f docker/Dockerfile.lmeval .
docker build -t abliterlitics-llamacpp:1.0.0 -f docker/Dockerfile.llamacpp .
```

> The `forensics` image is required for all analyses. The `lmeval` image is needed for capability benchmarks. The `llamacpp` and `ik-llamacpp` images are optional fallbacks for models that don't fit in GPU VRAM.

### 2. Set Up a Comparison Directory

Create a directory with your base model and variants, plus a `comparison.json`:

```
my-comparison/
├── comparison.json
├── Qwen3.5-4B/              # Base model (safetensors)
├── Qwen3.5-4B-heretic/      # Heretic-abliterated variant
├── Qwen3.5-4B-hauhau/       # HauhauCS-uncensored variant
└── Qwen3.5-4B-huihui/       # Huihui-abliterated variant
```

See [`comparison.example.json`](comparison.example.json) for a complete example (GLM-4.7-Flash with 4 variants). Copy it and adapt the paths:

```bash
cp comparison.example.json my-comparison/comparison.json
# Edit paths to match your model directories
```

`comparison.json`:
```json
{
  "name": "qwen35-4b",
  "base": "Qwen3.5-4B",
  "variants": {
    "heretic": { "path": "Qwen3.5-4B-heretic" },
    "hauhau": { "path": "Qwen3.5-4B-hauhau" },
    "huihui": { "path": "Qwen3.5-4B-huihui" }
  }
}
```

### 3. Run the Analysis

```bash
# Full pipeline (weights + KL + lm-eval + harmbench)
./abliterlitics.sh auto ./my-comparison/

# Or run individual phases
./abliterlitics.sh weights ./my-comparison/
./abliterlitics.sh kl ./my-comparison/
./abliterlitics.sh lm-eval ./my-comparison/
./abliterlitics.sh harmbench ./my-comparison/
```

### 4. Generate Reports

```bash
# SVG graphs for all analyses
./abliterlitics.sh graphs ./my-comparison/

# HTML provenance report
./abliterlitics.sh report ./my-comparison/
```

## Commands

| Command | Description |
|---------|-------------|
| `auto` | Run full pipeline (weights + KL + lm-eval + harmbench + graphs) |
| `weights` | Weight analysis (panel, edit, SVD, correlation, fingerprint, etc.) |
| `kl` | KL divergence analysis |
| `lm-eval` | lm-evaluation-harness (8 tasks) |
| `harmbench` | HarmBench safety evaluation (generate + classify + score) |
| `graphs` | Generate SVG graphs from existing results |
| `report` | Generate HTML provenance report |
| `build` | Build/pull Docker images |
| `status` | Show completion status for a comparison |
| `clean` | Remove generated results |
| `validate` | Validate comparison.json and model paths |

## Global Options

| Option | Description |
|--------|-------------|
| `--gpu GPU` | GPU index (default: auto-detect) |
| `--backend BACKEND` | Override inference backend (`vllm`/`llamacpp`/`ik_llamacpp`/`auto`) |
| `--skip-existing` | Skip steps with existing results (default) |
| `--force` | Overwrite existing results |
| `--dry-run` | Show commands without executing |

## comparison.json Reference

The full schema is defined in [`comparison.schema.json`](comparison.schema.json). See [`comparison.example.json`](comparison.example.json) for a working example.

### Settings (optional)

```json
{
  "name": "my-comparison",
  "base": "BaseModel/",
  "variants": { ... },
  "settings": {
    "inference_backend": "auto",
    "lm_eval_tasks": "mmlu,gsm8k,hellaswag,arc_challenge,winogrande,truthfulqa,piqa,lambada_openai",
    "lm_eval_max_gen_toks": 2048,
    "lm_eval_max_model_len": 4096,
    "harmbench_max_tokens": 2048,
    "kl_num_prompts": 100,
    "kl_dataset": "mlabonne/harmless_alpaca",
    "gguf_dir": null,
    "tokenizer_dir": null
  }
}
```

### Per-variant skip flags

Individual analyses can be skipped per variant:

```json
{
  "variants": {
    "heretic": { "path": "HereticModel/" },
    "hauhau": {
      "path": "HauhauModel/",
      "skip_kl": true,
      "skip_harmbench": true
    }
  }
}
```

## Project Structure

```
abliterlitics/
├── abliterlitics.sh        # Main CLI entry point
├── comparison.example.json  # Example comparison config (GLM-4.7-Flash)
├── comparison.schema.json   # JSON Schema for comparison.json
├── pyproject.toml           # Build config + linter settings
├── requirements.txt         # Python dependencies (pinned)
├── docker/
│   ├── Dockerfile.forensics   # Weight analysis + KL divergence
│   ├── Dockerfile.lmeval      # lm-evaluation-harness
│   ├── Dockerfile.llamacpp    # llama.cpp (official, pre-built)
│   └── Dockerfile.ik-llamacpp # ik_llama.cpp (faster fork, from source)
├── runners/
│   ├── common.sh            # Shared shell functions
│   ├── run_weights.sh       # Weight analysis runner
│   ├── run_kl.sh            # KL divergence runner
│   ├── run_lm_eval.sh       # lm-eval runner
│   └── run_harmbench.py     # HarmBench runner (Python)
├── src/
│   ├── __init__.py          # Version + RESULTS_VERSION
│   ├── config.py            # ComparisonConfig, schema validation
│   ├── gpu.py               # GPU detection, strategy selection
│   ├── docker_helpers.py    # Docker command builder
│   ├── model_config.py      # Architecture auto-detection
│   ├── weight/              # 11 weight analysis scripts
│   ├── kl/                  # KL divergence (frozen methodology)
│   ├── benchmark/           # HarmBench + lm-eval scripts
│   └── report/              # Graph + report generation
├── tests/                   # Full test suite (203 tests)
├── docs/
│   ├── METHODOLOGY.md
│   ├── EXAMPLES.md
│   ├── SECURITY.md
│   └── GPU_SETUP.md
└── LICENSE
```

## Supported Architectures

Abliterlitics auto-detects architecture from model weights:

| Architecture | Example Models | Notes |
|---|---|---|
| Qwen3.5 (Mamba2+Transformer) | Qwen3.5-2B/4B/9B/27B | Hybrid SSM+attention layers |
| Qwen3 (Transformer) | Qwen3-4B-Instruct | Standard transformer |
| GLM-4 (MoE) | GLM-4.7-Flash | Mixture-of-experts with expert analysis |

## Testing

All tests run inside Docker (no host dependencies needed):

```bash
docker run --rm -v $(pwd):/app -w /app \
  abliterlitics-forensics:1.0.0 \
  python3 -m pytest tests/ -v
```

Quality gates:
```bash
# All run inside Docker
ruff check src/            # Zero lint errors
ruff format --check src/   # Consistent formatting
mypy src/ --strict         # Type checking
pytest tests/ -q           # 203 tests passing
```

## Documentation

- [Methodology](docs/METHODOLOGY.md) — Exact methodology for each analysis axis
- [Examples](docs/EXAMPLES.md) — Step-by-step workflows
- [Security](docs/SECURITY.md) — Token handling, trust boundaries, supply chain
- [GPU Setup](docs/GPU_SETUP.md) — Multi-GPU configuration, Docker index mapping

## License

MIT
