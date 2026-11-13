# Abliterlitics

[![CI](https://github.com/dreamfast/abliterlitics/actions/workflows/ci.yml/badge.svg)](https://github.com/dreamfast/abliterlitics/actions/workflows/ci.yml)

Comparative forensic analysis of LLM abliteration techniques.

Abliterlitics measures structural and behavioral differences between base LLMs and their
"abliterated" (safety-removed) variants. It compares abliteration techniques across multiple
model architectures using four analysis axes:

1. **Weight Analysis** — Structural comparison of modified tensors, SVD decomposition, subspace alignment, technique fingerprinting
2. **KL Divergence** — Measures collateral damage of abliteration by comparing full-vocabulary log probability distributions (methodology matches [Heretic](https://github.com/p-e-w/heretic))
3. **Capability Benchmarks** — lm-evaluation-harness with a 7-task default suite (Open LLM Leaderboard v2 core + GSM8K + TruthfulQA); an extended 13-task suite is available via `lm_eval_tasks`
4. **Safety Benchmarks** — HarmBench (400 prompts) to measure Attack Success Rate

## Quick Start

### Prerequisites

- Docker with NVIDIA Container Toolkit (`--runtime=nvidia` support)
- NVIDIA GPU(s) with sufficient VRAM for target models
- 50 GB+ disk space for results
- Python 3.10+ on the host (for shell orchestration and the test suite)

### 1. Build Docker Images

```bash
./abliterlitics.sh build
```

Or build individually:

```bash
docker build -t abliterlitics-forensics:1.0.0 -f docker/Dockerfile.forensics .
docker build -t abliterlitics-lmeval:1.0.0 -f docker/Dockerfile.lmeval .
```

| Image | Dockerfile | Purpose |
|---|---|---|
| `abliterlitics-forensics:1.0.0` | `docker/Dockerfile.forensics` | Weight analysis + KL divergence |
| `abliterlitics-lmeval:1.0.0` | `docker/Dockerfile.lmeval` | lm-eval + HarmBench (vLLM 0.19 baseline) |
| `abliterlitics-lmeval-nightly:1.0.0` | *(no Dockerfile in repo — prebuilt image)* | vLLM 0.20 nightly for Gemma4 E4B/E2B/31B |
| `abliterlitics-lmeval-gemma4unified:1.0.0` | `docker/Dockerfile.lmeval_gemma4_unified` | vLLM 0.26 for Gemma4Unified (12B) |
| `abliterlitics-llamacpp:1.0.0` | `docker/Dockerfile.llamacpp` | llama.cpp fallback |
| `abliterlitics-ik-llamacpp:1.0.0` | `docker/Dockerfile.ik-llamacpp` | ik_llama.cpp fork (from source) |
| `abliterlitics-te:1.2.0` | `docker/Dockerfile.te` | Text-encoder conditioning battery |

A comparison can override the serving images per model family via `settings.lmeval_image` /
`settings.forensics_image` in `comparison.json` (newer architectures need newer vLLM builds).

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

# Markdown tables from the DB
./abliterlitics.sh tables all ./my-comparison/
```

## Commands

All commands are invoked through `./abliterlitics.sh <command> [args]`.

### Core pipeline

| Command | Description |
|---------|-------------|
| `auto` | Run full pipeline (weights + KL + lm-eval + harmbench) |
| `weights` | Weight analysis (panel, edit vectors, SVD, correlation, fingerprint, etc.) |
| `kl` | KL divergence analysis |
| `lm-eval` | lm-evaluation-harness benchmarks (vLLM server + local-completions in one container) |
| `harmbench` | HarmBench safety evaluation (generate + classify + score) |

### Text-encoder battery (tebench)

| Command | Description |
|---------|-------------|
| `text-encoder` / `te` | Text-encoder conditioning fidelity analysis |
| `te-report` | Generate SVG graphs + HTML report from text-encoder results |
| `te-img-metrics` | Image-space metrics over the A/B set (LPIPS/CLIP/SSIM) |
| `te-controls` | Counterfactual conditioning battery (noise/taboo/shuffles/null) |
| `te-attention` | Attention readout: what the DiT reads per text encoder |
| `te-discriminability` | Inter-prompt spread per TE (semantic-collapse detector) |
| `te-divergence` | DiT-step divergence analysis (paired sampling trajectories) |
| `te-directional` | Directional TE conditioning analysis |

### Reporting, data, and utility

| Command | Description |
|---------|-------------|
| `graphs` | Generate SVG graphs from existing results |
| `tables` | Generate markdown tables from the DB (`benchmark`, `harmbench`, `kl`, `summary`, `all`) |
| `db` | Database operations (`init`, `status`, `import-results`, `pipeline-status`, `stage-start/complete/fail/skip`, `journal add/list/latest/search`, `export-review`, `export-prompt`, `import-reviews`, `asr`) via `python3 -m src.db` |
| `tables` | Generate markdown tables from the DB (`benchmark`, `harmbench`, `kl`, `summary`, `all`) |
| `build` | Build Docker images |
| `status` | Show completion status for a comparison |
| `clean` | Remove generated results |
| `validate` | Validate comparison.json and model paths |

### Global Options

| Option | Description |
|--------|-------------|
| `--gpu GPU` | GPU index (default: auto-detect) |
| `--backend BACKEND` | Override inference backend (`vllm`/`llamacpp`/`ik_llamacpp`/`auto`) |
| `--skip-existing` | Skip steps with existing results (default) |
| `--force` | Overwrite existing results |
| `--dry-run` | Show commands without executing |
| `-v, --version` | Show version |

## comparison.json Reference

The full schema is defined in [`comparison.schema.json`](comparison.schema.json). See [`comparison.example.json`](comparison.example.json) for a working example.

### Settings (optional)

Defaults shown match what the pipeline uses when a key is omitted:

```json
{
  "name": "my-comparison",
  "base": "BaseModel/",
  "variants": { ... },
  "settings": {
    "inference_backend": "auto",
    "lm_eval_tasks": "leaderboard_mmlu_pro,leaderboard_gpqa_diamond,leaderboard_bbh,leaderboard_musr,leaderboard_ifeval,gsm8k,truthfulqa",
    "lm_eval_max_gen_toks": 4096,
    "lm_eval_max_model_len": 16384,
    "harmbench_max_tokens": 8096,
    "kl_num_prompts": 100,
    "kl_dataset": "mlabonne/harmless_alpaca",
    "lmeval_image": null,
    "forensics_image": null
  }
}
```

The default task suite is the Open LLM Leaderboard v2 core
(`leaderboard_mmlu_pro`, `leaderboard_gpqa_diamond`, `leaderboard_bbh`, `leaderboard_musr`,
`leaderboard_ifeval`) plus `gsm8k` and `truthfulqa`. An extended suite adds
`hellaswag`, `winogrande`, `arc_challenge`, `piqa`, `lambada_openai`, and `humaneval` —
set them explicitly via `lm_eval_tasks`. See [docs/METHODOLOGY.md](docs/METHODOLOGY.md)
for the task-by-task breakdown and the phased execution model.

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
├── pyproject.toml           # Build config + ruff/mypy/pytest settings
├── requirements.txt         # Python dependencies (Docker installs)
├── docker/                  # Dockerfiles (see image table above)
├── runners/
│   ├── common.sh            # Shared shell functions (docker_run helper)
│   ├── _load_comparison.py  # Exports comparison.json as shell env vars
│   ├── run_weights.sh       # Weight analysis runner
│   ├── run_kl.sh            # KL divergence runner
│   ├── run_lm_eval.sh       # lm-eval runner (vLLM server + local-completions, 3-phase)
│   ├── run_harmbench.py     # HarmBench runner (vLLM + llama.cpp fallback)
│   ├── run_text_encoder.sh  # Text-encoder battery runner (+ run_te_*.sh)
│   ├── weights_sequencer.sh # Single-container long-run pattern for weights
│   ├── lm_eval_watchdog.sh  # lm-eval queue watchdog (restart + cache resume)
│   └── predownload_datasets.sh  # One-time HF dataset cache
├── src/
│   ├── __init__.py          # Version + RESULTS_VERSION
│   ├── config.py            # ComparisonConfig, schema validation
│   ├── model_config.py      # Architecture auto-detection
│   ├── db_tables.py         # Markdown table generation from the DB
│   ├── db/                  # SQLite state: journal, importers, pipeline stages, CLI
│   ├── weight/              # Weight forensics scripts (panel, SVD, correlation, fingerprints, ...)
│   ├── kl/                  # KL divergence (frozen methodology)
│   ├── benchmark/           # HarmBench generate/classify, benchmark_eval, logprobs proxy
│   ├── text_encoder/        # Text-encoder conditioning battery (largest subsystem)
│   └── report/              # Graph generation
├── tests/                   # Test suite (1100+ tests)
├── docs/
│   ├── METHODOLOGY.md
│   ├── EXAMPLES.md
│   ├── SECURITY.md
│   └── GPU_SETUP.md
└── LICENSE
```

Results are written per comparison under `comparisons/<name>/results/`
(`weight/`, `kl/`, `lm_eval/`, `harmbench/`).

## Supported Architectures

Abliterlitics auto-detects architecture from model weights:

| Architecture | Example Models | Notes |
|---|---|---|
| Qwen3 (Transformer) | Qwen3-4B-Instruct | Standard transformer |
| Qwen3.5 (Mamba2+Transformer) | Qwen3.5-2B/4B/9B/27B | Hybrid SSM+attention layers; Qwen3.8 models are evaluated on the Qwen3.5 code path |
| Gemma4 | Gemma4 E4B/E2B/12B/31B | Includes the `Gemma4Unified` multimodal variants (need the gemma4unified image) |
| GLM-4 (MoE) | GLM-4.7-Flash | Mixture-of-experts with expert analysis |

## Testing

The suite runs on the host (this is what CI does):

```bash
pip install -e ".[dev]"
MPLCONFIGDIR=/tmp/mpl python3 -m pytest tests/ -q   # 1100+ tests, CPU-only
```

Quality gates:
```bash
ruff check .
ruff format --check .
mypy src/
python3 -m pytest tests/ -q
```

No GPU or Docker is needed for the test suite. The same gates run in CI
([.github/workflows/ci.yml](.github/workflows/ci.yml)).

## Documentation

- [Methodology](docs/METHODOLOGY.md) — Exact methodology for each analysis axis
- [Examples](docs/EXAMPLES.md) — Step-by-step workflows
- [Security](docs/SECURITY.md) — Token handling, trust boundaries, supply chain
- [GPU Setup](docs/GPU_SETUP.md) — Multi-GPU configuration, Docker index mapping

## Attribution

The KL divergence measurement in `src/kl/kl_divergence.py` reimplements the methodology from [Heretic](https://github.com/p-e-w/heretic) by Philipp Emanuel Weidmann. Heretic is licensed under AGPL-3.0.

```bibtex
@misc{heretic,
  author = {Weidmann, Philipp Emanuel},
  title = {Heretic: Fully automatic censorship removal for language models},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/p-e-w/heretic}}
}
```

## License

[AGPL-3.0](LICENSE) — includes the Heretic KL-divergence methodology attribution above.
