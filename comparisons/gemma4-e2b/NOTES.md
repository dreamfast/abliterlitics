# Gemma4-E2B Abliterlitics — Notes & Findings

## Model Family

**Base**: `google/gemma-4-E2B-it` — Gemma4ForConditionalGeneration, 35 text layers, multimodal (~2B text params)

### Architecture Quirks

- **Dual-norm / shared-KV**: `num_kv_shared_layers: 20`, `num_key_value_heads: 1`
  - 15 layers with full KV projections (layers 0-14)
  - 20 layers with shared KV (layers 15-34)
  - `layer_types` alternates `sliding_attention` / `full_attention` every 5 layers
  - Base model has 600 LM keys; shared-KV layers still have `k_proj`/`k_norm`/`v_proj` in safetensors
- **Tied embeddings**: `tie_word_embeddings: true`
- **Multimodal**: audio_tower, vision_tower, embed_audio, embed_vision (non-LM keys)
- **vLLM**: requires `--reasoning-parser gemma4`, `--entrypoint python3 -m vllm.entrypoints.openai.api_server` (vLLM 0.20.0)
- **Thinking**: `<|think|>` token (ID 98), requires `chat_template_kwargs={"enable_thinking": true}`

### 13 Variants

| Slug | Model Dir | Display Name | LM Keys | HF Repo | Notes |
|---|---|---|---|---|---|
| coder3101 | coder3101-heretic | Coder3101 Heretic | 600 | `coder3101/gemma-4-E2B-it-heretic` | |
| duoneural | duoneural-heretic | DuoNeural Heretic | **540** | `DuoNeural/Gemma-4-Abliterated-LiteRT` | Missing shared-KV weights |
| ether4o4 | ether4o4-opus | Ether Opus | **540** | `amkkk/Gemma4_E2B_Abliterated_Baked_HF_Ready` | Missing shared-KV weights |
| huihui-v1 | huihui-v1 | Huihui v1 | 600 | `huihui-ai/Huihui-gemma-4-E2B-it-abliterated` | |
| huihui-v2 | huihui-v2 | Huihui v2 | 600 | `huihui-ai/Huihui-gemma-4-E2B-it-abliterated` (v2) | |
| kasper | kasper-heretic | Kasper Heretic | **540** | `Kasper-Bankler/gemma-4-E2B-uncensored` | Missing shared-KV weights |
| llmfan46 | llmfan46-heretic | LLMFan46 Heretic | 600 | `llmfan46/gemma-4-E2B-it-ultra-uncensored-heretic` | |
| pew | pew-heretic-ara | PEW Heretic ARA | 600 | `p-e-w/gemma-4-E2B-it-heretic-ara` | |
| prithiv | prithiv-max | Prithiv Max | 600 | (prithiv) | |
| treadon | treadon-dual | Treadon Dual | **540** | `treadon/gemma4-E2B-it-Abliterated-AND-Disinhibited-USE-THIS` | Missing shared-KV weights |
| trevorjs | trevorjs-biprojection | TrevorJS BiProjection | 600 | (trevorjs) | |
| wangzhang | wangzhang-abliterix | Wangzhang Abliterix | **540** | `wangzhang/gemma-4-E2B-it-abliterated` | Missing shared-KV weights |
| wwtcyberlab | wwtcyberlab-abliterated | WWT CyberLab | 600 | `WWTCyberLab/gemma-4-E2B-it-abliterated` | |

---

## Shared-KV Export Bug (Novel Finding)

### Summary

5 of 13 variants shipped with **60 missing weights** — `k_proj.weight`, `k_norm.weight`, `v_proj.weight` for layers 15-34 (the 20 shared-KV layers). This makes them unloadable by vLLM and HuggingFace Transformers.

### Root Cause

The abliteration export tools (used by all 5 authors) only saved weights they modified (`o_proj`, `down_proj`, `gate_proj`, `up_proj`) plus whatever their framework's default export captured. They did not understand Gemma4's `num_kv_shared_layers` architecture, and the shared-KV weights for layers 15-34 were silently dropped.

### Evidence

1. **All 5 variants missing exactly the same 60 weights**: `k_proj.weight` + `k_norm.weight` + `v_proj.weight` × 20 layers
2. **Affected layers 15-34**: exactly the `sliding_attention` layers after the first `full_attention` layer group
3. **No HF discussions report this**: all 5 repos have zero relevant discussions
4. **Unmodified weights are byte-for-byte identical** across all 8 working 600-key variants and the base model — `k_proj`, `v_proj`, `q_proj` for layers 0, 15, 34 all identical
5. **Only `o_proj` and MLP weights differ** between variants (as expected — those are the abliteration targets)

### Fix

Copy the 60 missing weights from `google-base` to each of the 5 variants. Since these weights are unmodified and identical in all working variants, this is a safe, lossless patch.

**Status**: Patched 2026-05-19. All 5 variants now have 2011 total keys / 600 LM keys, matching the base model. Sharded files consolidated into single `model.safetensors`.

### Affected Models

| Variant | Export Tool | HF Repo |
|---|---|---|
| duoneural | Unknown (LiteRT export?) | `DuoNeural/Gemma-4-Abliterated-LiteRT` |
| ether4o4 | Module-input ortho bake | `amkkk/Gemma4_E2B_Abliterated_Baked_HF_Ready` |
| kasper | Heretic ARA | `Kasper-Bankler/gemma-4-E2B-uncensored` |
| treadon | Disinhibition + abliteration | `treadon/gemma4-E2B-it-Abliterated-AND-Disinhibited-USE-THIS` |
| wangzhang | Direct weight editing (abliterix) | `wangzhang/gemma-4-E2B-it-abliterated` |

---

## HarmBench Results (temperature=0, max_tokens=8096)

**Method**: vLLM OpenAI server + `harmbench_generate.py`, 400 behaviors, `enable_thinking=true`, keyword-based refusal detection

| Model | ASR | Refusals | Errors | Trunc | Avg Tokens | Notes |
|---|---|---|---|---|---|---|
| base | 29.8% | 281 | 0 | 1 | 617 | |
| coder3101 | 95.8% | 17 | 0 | 2 | 1190 | |
| duoneural | 81.8% | 73 | 0 | 1 | 1138 | patched shared-KV |
| ether4o4 | 97.0% | 12 | 0 | 2 | 1087 | patched shared-KV |
| huihui-v1 | 87.0% | 52 | 0 | 2 | 1219 | |
| huihui-v2 | 97.0% | 12 | 0 | 0 | 1332 | zero truncations |
| kasper | 91.5% | 34 | 0 | 1 | 1309 | patched shared-KV |
| llmfan46 | 85.0% | 60 | 0 | 1 | 1042 | |
| pew | 92.0% | 32 | 0 | 1 | 1144 | |
| prithiv | 88.0% | 48 | 0 | 2 | 1218 | |
| treadon | 98.8% | 5 | 0 | 21 | 2067 | highest ASR, most verbose, 21 truncations |
| trevorjs | 97.2% | 11 | 0 | 0 | 1340 | zero truncations |
| wangzhang | 98.8% | 5 | 0 | 2 | 1742 | patched shared-KV, tied highest ASR |
| wwtcyberlab | 97.2% | 11 | 0 | 1 | 1727 | |

**All 14 models complete. Zero errors. One thinking loop (kasper, HarmBench suicide-instruction item, 2,698 `<|channel>thought` repeats). Zero empty responses (except kasper: 1).**

---

## LM-Eval Benchmarks

**Docker image**: `abliterlitics-lmeval-gemma4:1.0.0` (vLLM 0.20.0 + lm-eval 0.4.12)
**Tokenizer**: always mounted from `google-base:/tokenizer:ro`

### Phase 1: Loglikelihood Tasks (complete — all 14 models)

**Method**: vLLM OpenAI server + lm-eval `local-completions` backend, same container
**Tasks**: `mmlu,hellaswag,arc_challenge,winogrande,truthfulqa_mc1,truthfulqa_mc2,piqa,lambada_openai`

**Settings**:
- `max_model_len=8096`, `gpu_memory_utilization=0.92`, `enforce_eager=True`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `--reasoning-parser gemma4`
- `batch_size=1`, `num_concurrent=1` (higher values OOM on log_softmax over 262K vocab)
- `max_gen_toks` irrelevant for loglikelihood tasks (they don't generate tokens)
- `--log_samples` not used for Phase 1 (loglikelihood tasks don't produce interesting samples)
- ~47 min per model

**OOM note**: `batch_size=4` + `max_model_len=8192` OOM'd during MMLU — `log_softmax` over 262K vocab on long prompts exceeded 32GB VRAM. Single-request (`batch_size=1`, `num_concurrent=1`) at `max_model_len=8096` works fine.

| Model | MMLU | HellaSwag | ARC | WinoGrande | TQA-MC1 | TQA-MC2 | PiQA | LAMBADA |
|---|---|---|---|---|---|---|---|---|
| base | 29.00 | 30.97 | 20.90 | 52.09 | 24.85 | 48.38 | 55.17 | 145,956 |
| coder3101 | 28.70 | 31.18 | 21.50 | 51.14 | 25.95 | 47.18 | 56.04 | 137,990 |
| duoneural | 28.75 | 30.90 | 21.84 | 51.07 | 25.21 | 48.77 | 55.55 | 127,877 |
| ether4o4 | 28.23 | 32.36 | 20.90 | 49.72 | 25.46 | 47.07 | 57.13 | 332,771 |
| huihui-v1 | 29.33 | 30.83 | 21.59 | 51.38 | 24.85 | 48.44 | 55.82 | 114,126 |
| huihui-v2 | 28.39 | 30.76 | 21.33 | 51.46 | 24.36 | 47.57 | 55.55 | 77,045 |
| kasper | 28.53 | 31.61 | 22.44 | 50.83 | 25.83 | 48.02 | 56.80 | 200,157 |
| llmfan46 | 28.36 | 30.85 | 21.84 | 51.78 | 26.19 | 47.82 | 55.93 | 150,562 |
| pew | 28.86 | 31.39 | 21.93 | 51.14 | 25.70 | 48.93 | 55.71 | 153,860 |
| prithiv | 29.33 | 30.83 | 21.59 | 51.38 | 24.85 | 48.44 | 55.82 | 114,126 |
| treadon | 28.02 | 31.30 | 22.95 | 52.25 | 22.52 | 43.74 | 56.09 | 198,775 |
| trevorjs | 28.94 | 31.18 | 21.08 | 51.38 | 25.95 | 47.84 | 56.20 | 170,183 |
| wangzhang | 26.69 | 31.64 | 22.18 | 51.14 | 25.34 | 45.44 | 56.58 | 1,072,918 |
| wwtcyberlab | 27.14 | 31.43 | 21.67 | 52.09 | 25.21 | 45.18 | 55.01 | 831,086 |

**Findings**: Abliteration barely dents loglikelihood scores (most models within 2pp of base). LAMBADA perplexity is the outlier — ether4o4, wangzhang, wwtcyberlab blow up (2-7x higher than base).

### Hyperparameters (Fixed Across All Evals)

| Parameter | Phase 1 (Loglikelihood) | Phase 2 (GSM8K) |
|---|---|---|
| Backend | `local-completions` | `local-chat-completions` |
| `max_model_len` | 8,096 | 16,384 |
| `max_gen_toks` | N/A (loglikelihood) | 14,336 |
| `batch_size` | 1 | 4 |
| `num_concurrent` | 1 | 4 |
| `temperature` | 0 | 0 |
| `do_sample` | False | False |
| `random_seed` | 0 | 0 |
| `numpy_seed` | 1234 | 1234 |
| `torch_seed` | 1234 | 1234 |
| `fewshot_seed` | 1234 | 1234 |
| `gpu_memory_utilization` | 0.92 | 0.92 |
| `enforce_eager` | True | True |
| `reasoning_parser` | gemma4 | gemma4 |
| `apply_chat_template` | N/A | True |
| Chat template | Standard | Modified: `enable_thinking=true` default |
| `--log_samples` | No | Yes |

**Baseline empty responses**: 10/1319 (~0.8%) on base model. Same seeds + greedy decoding (`temperature=0`, `do_sample=False`) means deterministic behavior. Only model weights differ between variants.

### Phase 2: GSM8K (complete — all 14 models)

**Critical discovery**: lm-eval's `local-completions` backend sends plain text completions (no chat template), which means **thinking is never activated** for reasoning models. Base scored 10.6% without thinking vs **83.3%** with thinking enabled via `local-chat-completions` — a 7.9x improvement.

**Wrong approach (discarded)**: lm-eval `local-completions` + `max_gen_toks=7168` → 10.6% exact_match, responses had no thinking tokens, one sample entered a repetition loop (128K chars of "together together...")

**Correct approach**: lm-eval `local-chat-completions` with modified chat template that defaults `enable_thinking=true`

**Bug found & fixed**: The batch run script's "Saved to" copy operation copied from the wrong path — all 14 `_gsm8k_lmeval_results.json` files ended up as copies of the last model's results (wwtcyberlab). Correct scores were recovered from the raw `__tmp__model_{slug}/results_*.json` files written by lm-eval.

| Model | GSM8K Flexible | GSM8K Strict | Empty | Flex Δ vs Base | Strict Δ vs Base |
|---|---|---|---|---|---|
| coder3101 | **84.84%** | **75.21%** | 6 | +1.37 | +3.94 |
| llmfan46 | 83.93% | 72.86% | 10 | +0.46 | +1.59 |
| base | 83.47% | 71.27% | 10 | — | — |
| pew | 83.47% | 72.71% | 10 | ±0.00 | +1.44 |
| huihui-v1 | 83.40% | 69.83% | 8 | -0.07 | -1.44 |
| kasper | 83.24% | 72.71% | 4 | -0.23 | +1.44 |
| duoneural | 83.09% | 72.63% | 20 | -0.38 | +1.36 |
| prithiv | 82.94% | 68.92% | 10 | -0.53 | -2.35 |
| trevorjs | 82.49% | 68.31% | 8 | -0.98 | -2.96 |
| wwtcyberlab | 82.41% | 55.50% | 8 | -1.06 | -15.77 |
| wangzhang | 81.58% | 66.19% | 36 | -1.89 | -5.08 |
| treadon | 80.59% | 59.44% | 38 | -2.88 | -11.83 |
| huihui-v2 | 79.23% | 64.37% | 54 | -4.24 | -6.90 |
| ether4o4 | 76.57% | 68.39% | 84 | -6.90 | -2.88 |

**Key findings**:
- **Flex scores cluster tightly**: 11 of 13 variants within ±4.2pp of base
- **Strict scores diverge more**: wwtcyberlab drops 15.8 points, treadon drops 11.8 points
- **coder3101 beats base** on both metrics (+1.4% flex, +3.9% strict)
- **Empty responses correlate with thinking loops**: ether4o4 (84 empties, 6.4%), huihui-v2 (54, 4.1%), treadon (38, 2.9%) — models that think too much exhaust the 14,336 token budget without producing content
- **Shared-KV patched models** (duoneural, ether4o4, kasper, treadon, wangzhang) show mixed results — not systematically worse
- **No question failed across all models** — max common failures: doc_id=1129 failed on 7/14 models

**Empty response analysis** (null content from vLLM reasoning parser):
| Tier | Models | Empty Count | % |
|---|---|---|---|
| Severe | ether4o4 | 84 | 6.4% |
| Heavy | huihui-v2 | 54 | 4.1% |
| Moderate | treadon, wangzhang | 36-38 | 2.7-2.9% |
| Light | duoneural | 20 | 1.5% |
| Normal | all others | 4-10 | 0.3-0.8% |

### Key Lesson: `local-completions` vs Chat Completions for Reasoning Models

lm-eval's `local-completions` backend bypasses the chat template entirely. For reasoning models like Gemma4 that use `<|think|>` tokens activated by `chat_template_kwargs={"enable_thinking": true}`, this means:
- **No thinking tokens generated** — model answers directly without reasoning
- **Repetition loops** — without thinking, the model sometimes falls into degenerate repetition
- **7.9x score improvement** when thinking is enabled (GSM8K: 10.6% → 83.3%)

For loglikelihood tasks (MMLU, HellaSwag, ARC, etc.) this doesn't matter — they rank token probabilities, not generate text. But for any generative eval of a reasoning model, you MUST use the chat completions endpoint with thinking enabled.

**Known upstream issue**: lm-eval does not yet support `reasoning`/`reasoning_content` fields from vLLM's `--reasoning-parser` (see EleutherAI/lm-evaluation-harness#3391, #3685). When the model returns thinking in a separate `reasoning` field, lm-eval sees `null` content and fills with a placeholder. The vLLM `--reasoning-parser gemma4` strips thinking from `content` into `reasoning`, which lm-eval ignores. Our chat template modification ensures thinking is still in the `content` field, so lm-eval can evaluate the full response.

---

## Weight Forensics

- **287 JSON result files** across 8 phases (panel, edit, SVD, fingerprint, layer, correlation, subspace, lowrank)
- Expert (MoE-only) and Cross-arch are N/A for this model
- All analyses handle the 600/540 key difference via key intersection

## KL Divergence

- **13 variants complete**, results in `comparisons/gemma4-e2b/results/kl/kl_*.json`
- Heretic-based models match their README-reported KL within ~10-20%
- wangzhang and duoneural have large discrepancies due to different methodology

---

## Timing

- HarmBench per variant: ~15-25 min (temperature=0, concurrent=4)
- vLLM server startup: ~60-70s
- Weight pipeline: ~4h total
- KL divergence: ~2h total
