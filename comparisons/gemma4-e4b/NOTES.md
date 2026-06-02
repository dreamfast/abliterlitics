# Gemma 4 E4B — Abliteration Forensics Notes

> Comparison: `google/gemma-4-E4B-it` (base) vs `gemma-4-E4B-it-apostate` (Apostate)
> Started: 2026-06-02

---

## Model Architecture

**Architecture:** `Gemma4ForConditionalGeneration` (multimodal: text + vision + audio)

| Property | Value |
|---|---|
| Effective params | 4.5B (8B with embeddings) |
| Text layers | 42 |
| Hidden size | 2560 |
| Attention heads | 8 (GQA: 2 KV heads) |
| Intermediate size | 10240 |
| Vocabulary | 262,144 |
| Context length | 128K tokens |
| Sliding window | 512 tokens (interleaved full attention at layers 5,11,17,23,29,35,41) |
| Model file | Single `model.safetensors` (~15 GB base, ~17 GB apostate) |
| Per-Layer Embeddings (PLE) | Yes — `hidden_size_per_layer_input: 256`, `vocab_size_per_layer_input: 262144` |

**Multimodal towers:**
- Vision: 16 layers, 768 hidden, 12 heads, ~150M params
- Audio: 12 layers, 1024 hidden, 8 heads, ~300M params

**Key architectural notes:**
- `tie_word_embeddings: true` in base, `false` in apostate — Apostate untied the embedding/lm_head during bake
- Uses `final_logit_softcapping: 30.0`
- 18 KV-shared layers (`num_kv_shared_layers: 18`)
- `hidden_activation: gelu_pytorch_tanh` (not SiLU)
- Hybrid attention: 6 full attention layers (5,11,17,23,29,35,41 — every 6th layer), rest are sliding window

---

## Thinking / Reasoning Mode

Gemma 4 E4B is a **thinking model** with configurable reasoning mode.

### Thinking tokens / control flow:
| Token | ID | Role |
|---|---|---|
| `<\|think\|>` | 98 | Injected at start of system prompt to enable thinking |
| `<\|channel>` | 100 | Opens a channel (e.g. `thought`) |
| `<channel\|>` | 101 | Closes a channel |
| `<\|turn>` | 105 | Turn boundary |
| `<turn\|>` | 106 | Turn close (also an EOS token) |

### Output pattern when thinking is enabled:
```
<|channel>thought
...chain of thought reasoning...<channel|>
Final answer text here.
```

### When thinking is disabled:
- For E4B: generates empty thought block `<|channel>thought\n<channel|>` then answer
- Only non-E2B/E4B models skip the empty block

### vLLM reasoning parser:
- **Parser:** `gemma4_reasoning_parser` (available in vLLM 0.20.1, NOT in 0.19.0)
- **Docker image:** `abliterlitics-lmeval-nightly:1.0.0` (vLLM 0.20.1, transformers 5.8.0)
- **Parser flags:** `--reasoning-parser gemma4`
- No `--reasoning-config` needed (unlike Qwen3 which needed start/end strings)

### Generation params (from model card):
- `temperature: 1.0`
- `top_p: 0.95`
- `top_k: 64`

### Implications for benchmarks:
- **GSM8K:** Needs high `max_gen_toks` (thinking tokens eat into the budget). Recommend separate phase with 7168 tokens.
- **TruthfulQA gen:** Thinking tokens consumed before answer. Watch for truncated responses.
- **Loglikelihood tasks:** (MMLU, HellaSwag, ARC, etc.) — thinking tokens not relevant, these use logprob scoring.
- **HarmBench:** Needs `max_tokens=4096+` to allow thinking + answer. Must strip `<|channel>thought...<channel|>` before scoring.
- **LM-eval `local-completions`:** If using vLLM server, reasoning parser strips thinking automatically from completions endpoint.

### Thinking token measurement:
We should instrument:
- Thinking token count per response (split on `<channel|>`)
- Answer token count (after `<channel|>`)
- Ratio thinking/answer
- Cutoff detection: responses where `<channel|>` is NOT found (thinking filled entire budget)
- Compare base vs apostate: does abliteration change reasoning behavior?

---

## Apostate Abliteration Method

**Tool:** [apostate](https://github.com/apostatelabs/apostate) — attention head-level refusal ablation
**Profile:** `balanced` (target_refusal=0.03, max_kl=0.16)
**Elapsed:** 350.5 seconds (~6 minutes)

### What Apostate did:
- **Method:** Attention head ablation — NOT weight-space orthogonal projection (like reaper/heretic)
- **Target:** Layer 24, attention heads (direction_layer_frac=0.58)
- **Head alpha:** 4.85 (scaling factor applied to refusal direction in attention head space)
- **Layer alphas:** ALL ZERO — no per-layer weight surgery
- **`ablate_embed: false`, `ablate_head: true`** — only head-level intervention
- **Optimization:** 14 head sweep trials, Bayesian optimization (16 trials, adaptive)
- **Capability penalty:** Enabled (gsm8k + humaneval + mbpp suites)
- **Bake:** Final weights baked to safetensors (bf16)

### Key metrics (from Apostate's own measurement):
| Metric | Value |
|---|---|
| Baseline refusal rate | 95.8% |
| Edited refusal rate | 12.4% |
| Best trial refusal | 5.66% |
| Best trial KL | 0.1102 nats |
| Harmless KL (final) | 0.1346 nats |
| KL target | 0.060 nats |
| KL budget | 0.160 nats |
| Preserve rank | 4 |

### Notable differences:
1. **`tie_word_embeddings` flipped** from `true` (base) to `false` (apostate) — Apostate untied embeddings during bake, creating separate lm_head weights
2. **File size difference:** Base 15 GB → Apostate 17 GB — the untied lm_head adds ~2 GB
3. **All layer alphas = 0** — this is a pure attention head modification, no weight matrix edits. Weight forensics may show very small diffs in text model weights, with the largest changes in the attention head output projections at layer 24.

---

## Docker Images

| Image | Version | Use |
|---|---|---|
| `abliterlitics-forensics:1.0.0` | transformers 5.5.4 | Weight forensics, KL divergence |
| `abliterlitics-lmeval-nightly:1.0.0` | vLLM 0.20.1, transformers 5.8.0 | lm-eval + HarmBench (has gemma4_reasoning_parser) |
| `abliterlitics-lmeval:1.0.0` | vLLM 0.19.0 | ❌ Does NOT have gemma4 reasoning parser |
| `abliterlitics-lmeval:1.1.0` | vLLM 0.19.0 | ❌ Same issue |

**Important:** Must use `abliterlitics-lmeval-nightly:1.0.0` for any inference that needs thinking mode (HarmBench, lm-eval). The standard lmeval images (vLLM 0.19.0) don't have the `gemma4_reasoning_parser`.

---

## Pipeline Stages

### Stage 1: Weight Forensics ✏️
- Docker: `abliterlitics-forensics:1.0.0`
- Single variant → no pairwise comparisons needed
- Watch for: attention head output projections at layer 24, lm_head vs embed diff
- Architecture detection: `gemma4` family already supported in `src/model_config.py`

### Stage 2: KL Divergence ✏️
- Docker: `abliterlitics-forensics:1.0.0`
- Response prefix: `<|channel>thought\n` (auto-detected)
- 100 harmless prompts from `mlabonne/harmless_alpaca`

### Stage 3: HarmBench ✏️
- Docker: `abliterlitics-lmeval-nightly:1.0.0` (for vLLM server)
- `--reasoning-parser gemma4`
- `--max-tokens 4096` (thinking + answer)
- Must handle thinking token stripping before classification
- Need to check if `harmbench_generate.py` handles the Gemma4 thinking format

### Stage 4: Benchmarks (lm-eval) ✏️
- Docker: `abliterlitics-lmeval-nightly:1.0.0`
- Two phases: Phase 1 (loglikelihood + truthfulqa, max_gen_toks=2048), Phase 2 (gsm8k, max_gen_toks=7168)
- `--reasoning-parser gemma4`
- Model fits in single GPU (15 GB bf16, no quantization needed — well within 32 GB)
- Actually: with 5090 (32 GB), model is ~15 GB bf16, should fit without BNB4

---

## Open Questions
- [ ] Does `harmbench_generate.py` correctly handle `<|channel>thought...<channel|>` format?
- [ ] For lm-eval with `local-completions`, does the vLLM gemma4 parser auto-strip thinking from completion responses?
- [ ] Model is ~15 GB bf16 — can we run WITHOUT quantization? (Much better benchmark scores than BNB4)
- [ ] Should we instrument thinking token counting in HarmBench responses?
