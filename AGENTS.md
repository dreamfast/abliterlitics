# AGENTS.md — Project Context & Lessons Learned

> This file is for AI agents (and humans) working on the Abliterlitics project.
> It captures hard-won knowledge about how to run evaluations correctly.

## Project Overview

Abliterlitics compares **base LLMs vs their abliterated/uncensored variants** across multiple axes:
- **Weight forensics**: Layer-by-layer weight diffs, SVD, cosine similarity
- **KL divergence**: Distribution shifts between base and modified models
- **Benchmark evaluation**: lm-evaluation-harness on 8 standard tasks
- **HarmBench**: Safety benchmark via vLLM + classifier
- **Reports**: Technical reports published to HuggingFace, condensed versions to Reddit

### Current Comparison: Qwen3.6-27B Family

6 models, all `Qwen3_5ForConditionalGeneration` architecture (~51-52GB bf16):

| Short Name | Model Path | Type |
|---|---|---|
| `base` | `models/Qwen3.6-27B/` | Base (vanilla) |
| `aeon` | `models/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16/` | Uncensored |
| `huihui` | `models/Huihui-Qwen3.6-27B-abliterated/` | Abliterated (Chinese community) |
| `heretic` | `models/Qwen3.6-27B-uncensored-heretic-v2/` | Uncensored v2 |
| `hauhau` | `models/Qwen3.6-27B-HauhauCS-Q8KP-recovered/` | Recovered variant |
| `abliterix` | `models/Qwen3.6-27B-abliterated-v2/` | Abliterated v2 |

---

## LM-Eval: The Proven Working Approach

### Architecture: vLLM OpenAI Server + `local-completions` Backend

**Why this approach?** The `vllm_causallms` direct backend OOMs with 27B BNB4 on a single 32GB GPU. Running vLLM as an OpenAI-compatible server and using lm-eval's `local-completions` backend avoids this because the server manages memory more efficiently.

**Both vLLM server AND lm-eval must run inside the SAME Docker container** — if lm-eval runs outside, the `local-completions` model init tries to download the tokenizer from HF Hub using the model name `/model` and hangs.

### Docker Images

- `abliterlitics-lmeval:1.0.0` — vLLM 0.19.0, lm-eval, proven working
- `abliterlitics-forensics:1.0.0` — Weight analysis, KL divergence

### Working Docker Command Template

```bash
docker run --rm --runtime=nvidia --shm-size=16g --ipc=host \
  --name <MODEL>-lmeval \
  -e "NVIDIA_VISIBLE_DEVICES=0" -e "CUDA_VISIBLE_DEVICES=0" \
  -v "<model_path>:/model:ro" \
  -v "models/Qwen3.6-27B:/tokenizer:ro" \
  -v "results/lm_eval:/results" \
  -v "results/lm_eval_logs:/logs" \
  -v ".cache/hf:/root/.cache/huggingface" \
  -v ".cache/vllm:/root/.cache/vllm" \
  -e HF_HOME=/root/.cache/huggingface \
  abliterlitics-lmeval:1.0.0 \
  bash -c '
    # Start vLLM server
    python3 -m vllm.entrypoints.openai.api_server \
      --model /model --dtype bfloat16 --quantization bitsandbytes \
      --load-format bitsandbytes --trust-remote-code --max-model-len 8192 \
      --gpu-memory-utilization 0.90 --enforce-eager \
      --reasoning-parser qwen3 \
      --reasoning-config '\''{"reasoning_start_str": "<think}", "reasoning_end_str": "</think"}'\'' \
      --port 8080 --host 127.0.0.1 > /logs/<model>_vllm_server.log 2>&1 &
    SERVER_PID=$!

    # Wait for ready
    for i in $(seq 1 120); do
      if curl -s http://127.0.0.1:8080/v1/models | grep -q model; then break; fi
      sleep 5
    done

    # Phase 1: Loglikelihood + truthfulqa (fast, 2048 gen tokens)
    python3 -m lm_eval --model local-completions \
      --model_args "base_url=http://127.0.0.1:8080/v1/completions,model=/model,tokenizer=/tokenizer,tokenizer_backend=huggingface,num_concurrent=4,max_length=8192" \
      --tasks "mmlu,hellaswag,arc_challenge,winogrande,truthfulqa,piqa,lambada_openai" \
      --batch_size 4 --gen_kwargs max_gen_toks=2048 \
      --output_path /results/ --log_samples 2>&1 | tee /logs/<model>_phase1.log

    # Phase 2: GSM8K only (needs high gen tokens for reasoning models)
    python3 -m lm_eval --model local-completions \
      --model_args "base_url=http://127.0.0.1:8080/v1/completions,model=/model,tokenizer=/tokenizer,tokenizer_backend=huggingface,num_concurrent=4,max_length=8192" \
      --tasks "gsm8k" \
      --batch_size 4 --gen_kwargs max_gen_toks=7168 \
      --output_path /results/ --log_samples 2>&1 | tee /logs/<model>_gsm8k.log

    kill $SERVER_PID 2>/dev/null
  '
```

---

## Critical Lessons Learned

### 1. `max_gen_toks` includes thinking tokens — plan accordingly

**The mistake:** Set `max_gen_toks=2048` thinking that was the "response budget". But for reasoning models (Qwen3.5 with `<think/>` tags), `max_gen_toks` is the TOTAL generation budget including thinking tokens. The model would think for 1900 tokens, get cut off, and never produce an answer.

**The fix:** GSM8K needs `max_gen_toks=7168` to give the model room for extended reasoning (~5000 thinking tokens) + answer (~2048 response tokens). But this is ONLY needed for generative tasks (gsm8k). Loglikelihood tasks (mmlu, hellaswag, arc, winogrande, piqa, lambada) ignore `max_gen_toks` entirely.

**Never set `max_gen_toks` higher than `max_model_len - max_prompt_length`.** With `max_model_len=8192` and prompts up to ~1024 tokens, `max_gen_toks=7168` is the safe maximum. Setting 8092 caused context overflow crashes.

**Two-phase approach:** Run loglikelihood + truthfulqa with `max_gen_toks=2048` (keeps truthfulqa fast), then run gsm8k separately with `max_gen_toks=7168`.

### 2. BNB4 quantization damages math reasoning significantly

GSM8K scores under BNB4 quantization are MUCH lower than published bf16 scores. This is expected and consistent across all models — the comparison is still valid since all models use the same quantization. Do NOT compare these scores with published leaderboard scores run in fp16/bf16.

### 3. Qwen3.5 architecture cannot do BNB4 + tensor parallelism

`Qwen3_5ForConditionalGeneration` raises `NotImplementedError` for BNB4 with TP>1. This works for GLM architecture but not Qwen3.5. Must use single GPU.

### 4. vLLM `reasoning-parser` settings

For Qwen3.5 models that use extended `<think/>` reasoning:
```
--reasoning-parser qwen3
--reasoning-config '{"reasoning_start_str": "<think}", "reasoning_end_str": "</think"}'
```
This is needed for the vLLM server to properly handle the thinking tags. Without it, the server may not parse responses correctly.

### 5. Tokenizer mount is required separately

The tokenizer must be mounted from the BASE model (`models/Qwen3.6-27B:/tokenizer:ro`) even when evaluating variants. The variant models' tokenizers are identical but the mount ensures lm-eval can find it without downloading.

### 6. `local-completions` tokenizer download hang

If lm-eval runs in a separate container from the vLLM server, the `local-completions` model initialization tries to download the tokenizer from HF Hub using the model name `/model` and hangs indefinitely. Both MUST run in the same container.

### 7. Reasoning models and thinking loops

Some Qwen3.5 abliterated models exhibit "thinking loops" — they generate repetitive reasoning text that consumes generation tokens without converging. This manifests as very short final responses after the thinking is stripped. It's a model behavior issue, not an eval config issue.

### 8. Old results can be from a different architecture

Always verify which model produced existing result files. The `results/lm_eval/` directory contained results from a GLM-4.7-Flash run (April), not Qwen3.6-27B. Check the JSON for `model_class` or `config` fields to confirm.

---

## Benchmark Tasks

| Task | Type | Metric | Notes |
|---|---|---|---|
| `mmlu` (57 subtasks) | Loglikelihood (MC) | acc | 14k+ questions, slow batch |
| `gsm8k` | Generate | exact_match | Needs high max_gen_toks for reasoning |
| `hellaswag` | Loglikelihood | acc_norm | ~10k questions |
| `arc_challenge` | Loglikelihood | acc | 1.1k questions |
| `winogrande` | Loglikelihood | acc | 1.2k questions |
| `truthfulqa` (gen + mc1 + mc2) | Mixed | bleu/rouge/acc | 817 questions, gen takes ~45min |
| `piqa` | Loglikelihood | acc | 1.8k questions |
| `lambada_openai` | Loglikelihood | perplexity | ~5k questions |

**Total loglikelihood requests:** ~30,596 (takes ~1.5h per model)
**Total generate requests:** gsm8k ~205 items + truthfulqa_gen ~330 items

### Timing estimates (per model, single 5090, BNB4)

- Loglikelihood batch: ~1.5 hours
- GSM8K (2048 gen): ~3 minutes | (7168 gen): ~15-30 minutes
- TruthfulQA gen: ~45 minutes
- **Total per model: ~2.5-3 hours**

---

## File Locations

```
results/lm_eval/              # Result JSON files
  lm_eval_<model>.json        # Canonical result files
  __model/                    # Raw output from lm-eval (samples, timestamps)

results/lm_eval_logs/         # All run logs
  <model>_vllm_server.log     # vLLM server log
  <model>_lm_eval_combined.log # lm-eval output
  <model>_combined_nohup.log  # nohup wrapper log

results/lm_eval_glm47/        # Archived GLM-4.7-Flash results (old)
comparison.json                # Model configs (paths, names)
```

---

## GPU Hardware

- GPU 0: NVIDIA RTX 5090 (32GB) — used for all eval runs
- GPU 1: NVIDIA RTX 4090 (24GB) — leave alone unless told otherwise
- Docker images need `--runtime=nvidia --shm-size=16g --ipc=host`
- BNB4 model uses ~31.6GB of the 32GB card
