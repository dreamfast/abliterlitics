# Benchmark Analysis: Gemma4-E2B

> [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) via [vLLM](https://github.com/vllm-project/vllm) v0.20.0
> Docker image: `abliterlitics-lmeval-gemma4:1.0.0` (vLLM 0.20.0 + lm-eval 0.4.12)
> Hardware: NVIDIA RTX 5090 (32GB), single GPU

## Phase 1: Loglikelihood Tasks

**Method**: vLLM OpenAI server + lm-eval `local-completions` backend, same container
**Tasks**: `mmlu,hellaswag,arc_challenge,winogrande,truthfulqa_mc1,truthfulqa_mc2,piqa,lambada_openai`
**Settings**: `max_model_len=8096`, `gpu_memory_utilization=0.92`, `enforce_eager=True`, `--reasoning-parser gemma4`, `batch_size=1`, `num_concurrent=1`
**Time**: ~47 min per model

| Model | MMLU | HellaSwag | ARC | WinoGrande | TQA-MC1 | TQA-MC2 | PiQA | LAMBADA (ppl ↓) |
|---|---|---|---|---|---|---|---|---|
| **base** | 29.00 | 30.97 | 20.90 | 52.09 | 24.85 | 48.38 | 55.17 | 145,956 |
| coder3101 | 28.70 | 31.18 | 21.50 | 51.14 | 25.95 | 47.18 | 56.04 | 137,990 |
| duoneural | 28.75 | 30.90 | 21.84 | 51.07 | 25.21 | 48.77 | 55.55 | 127,877 |
| ether4o4 | 28.23 | **32.36** | 20.90 | 49.72 | 25.46 | 47.07 | **57.13** | 332,771 |
| huihui-v1 | **29.33** | 30.83 | 21.59 | 51.38 | 24.85 | 48.44 | 55.82 | 114,126 |
| huihui-v2 | 28.39 | 30.76 | 21.33 | 51.46 | 24.36 | 47.57 | 55.55 | **77,045** |
| kasper | 28.53 | 31.61 | 22.44 | 50.83 | 25.83 | 48.02 | 56.80 | 200,157 |
| llmfan46 | 28.36 | 30.85 | 21.84 | 51.78 | **26.19** | 47.82 | 55.93 | 150,562 |
| pew | 28.86 | 31.39 | 21.93 | 51.14 | 25.70 | **48.93** | 55.71 | 153,860 |
| prithiv | **29.33** | 30.83 | 21.59 | 51.38 | 24.85 | 48.44 | 55.82 | 114,126 |
| treadon | 28.02 | 31.30 | **22.95** | **52.25** | 22.52 | 43.74 | 56.09 | 198,775 |
| trevorjs | 28.94 | 31.18 | 21.08 | 51.38 | 25.95 | 47.84 | 56.20 | 170,183 |
| wangzhang | 26.69 | 31.64 | 22.18 | 51.14 | 25.34 | 45.44 | 56.58 | 1,072,918 |
| wwtcyberlab | 27.14 | 31.43 | 21.67 | 52.09 | 25.21 | 45.18 | 55.01 | 831,086 |

All values are accuracy (acc) except HellaSwag (acc_norm) and LAMBADA (perplexity, lower is better). Bold indicates best within column (excluding base).

### Phase 1 Delta vs Base

| Model | MMLU | HellaSwag | ARC | WinoGrande | TQA-MC1 | TQA-MC2 | PiQA | LAMBADA |
|---|---|---|---|---|---|---|---|---|
| coder3101 | -0.30 | +0.21 | +0.60 | -0.95 | +1.10 | -1.20 | +0.87 | 0.95x |
| duoneural | -0.25 | -0.07 | +0.94 | -1.02 | +0.36 | +0.39 | +0.38 | 0.88x |
| ether4o4 | -0.77 | +1.39 | ±0.00 | -2.37 | +0.61 | -1.31 | +1.96 | 2.28x |
| huihui-v1 | +0.33 | -0.14 | +0.69 | -0.71 | ±0.00 | +0.06 | +0.65 | 0.78x |
| huihui-v2 | -0.61 | -0.21 | +0.43 | -0.63 | -0.49 | -0.81 | +0.38 | 0.53x |
| kasper | -0.47 | +0.64 | +1.54 | -1.26 | +0.98 | -0.36 | +1.63 | 1.37x |
| llmfan46 | -0.64 | -0.12 | +0.94 | -0.31 | +1.34 | -0.56 | +0.76 | 1.03x |
| pew | -0.14 | +0.42 | +1.03 | -0.95 | +0.85 | +0.55 | +0.54 | 1.05x |
| prithiv | +0.33 | -0.14 | +0.69 | -0.71 | ±0.00 | +0.06 | +0.65 | 0.78x |
| treadon | -0.98 | +0.33 | +2.05 | +0.16 | -2.33 | -4.64 | +0.92 | 1.36x |
| trevorjs | -0.06 | +0.21 | +0.18 | -0.71 | +1.10 | -0.54 | +1.03 | 1.17x |
| wangzhang | -2.31 | +0.67 | +1.28 | -0.95 | +0.49 | -2.94 | +1.41 | 7.35x |
| wwtcyberlab | -1.86 | +0.46 | +0.77 | ±0.00 | +0.36 | -3.20 | -0.16 | 5.69x |

Deltas in percentage points. LAMBADA shows ratio to base (values >1.0 = worse). Positive deltas are improvements for accuracy tasks (except LAMBADA where lower is better).

### Phase 1 Findings

**Loglikelihood tasks are remarkably resilient to abliteration.** The 14 models cluster within 2.6pp on MMLU (26.7%–29.3%), 1.6pp on HellaSwag (30.8%–32.4%), 2.1pp on ARC (20.9%–23.0%), and 2.5pp on WinoGrande (49.7%–52.3%). These tasks rank token probabilities, so abliteration of the refusal direction barely affects the model's knowledge representation.

**TruthfulQA MC2 shows the clearest abliteration signal.** Treadon drops -4.64pp, wangzhang drops -2.94pp, wwtcyberlab drops -3.20pp. These variants have the highest KL divergence and the most aggressive weight modifications. The truthfulness degradation is the expected safety-corruption side effect — abliteration removes refusal behavior by damaging the model's ability to distinguish factual from non-factual content.

**LAMBADA perplexity is the outlier metric.** Most variants show modest perplexity changes (0.53x–1.37x base). But three variants catastrophically degrade:

| Model | LAMBADA PPL | Ratio vs Base | KL Divergence |
|---|---|---|---|
| wangzhang | 1,072,918 | **7.35x** | 0.698 |
| wwtcyberlab | 831,086 | **5.69x** | 0.964 |
| ether4o4 | 332,771 | **2.28x** | 0.669 |

LAMBADA tests word prediction in context — it's the most sensitive metric to next-token distribution quality. The 7.35x perplexity blowup for wangzhang indicates that its unique `q_proj`/`v_proj` modifications (targeting attention input projections) catastrophically damage the model's language modeling capability, even though accuracy metrics look fine.

Interestingly, huihui-v2 at KL=0.530 has the **best** LAMBADA perplexity at 77,045 (0.53x base). Despite significant KL divergence, its edits concentrate in the refusal direction without disrupting language modeling quality.

---

## Phase 2: GSM8K

**Method**: vLLM OpenAI server + lm-eval `local-chat-completions` backend with modified chat template (`enable_thinking=true` default)
**Settings**: `max_model_len=16384`, `max_gen_toks=14336`, `batch_size=4`, `num_concurrent=4`, `temperature=0`

### Critical Discovery: `local-completions` vs `local-chat-completions`

lm-eval's `local-completions` backend sends plain text completions that bypass the chat template entirely. For Gemma4, which activates reasoning via `chat_template_kwargs={"enable_thinking": true}`, this means **thinking is never activated**.

| Backend | Thinking | Base GSM8K |
|---|---|---|
| `local-completions` | No | 10.6% |
| `local-chat-completions` | Yes | **83.3%** |

A 7.9x improvement. Without thinking, the model also falls into repetition loops (one sample produced 128K chars of "together together..."). **For any generative eval of a reasoning model, you MUST use chat completions with thinking enabled.**

### GSM8K Results (all 14 models)

Sorted by flexible-extract score (descending):

| Model | Flexible | Strict | Empty | Flex Δ vs Base | Strict Δ vs Base |
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
| wwtcyberlab | 82.41% | 55.50% | 8 | -1.06 | **-15.77** |
| wangzhang | 81.58% | 66.19% | 36 | -1.89 | -5.08 |
| treadon | 80.59% | 59.44% | 38 | -2.88 | -11.83 |
| huihui-v2 | 79.23% | 64.37% | 54 | -4.24 | -6.90 |
| ether4o4 | 76.57% | 68.39% | 84 | -6.90 | -2.88 |

### GSM8K Findings

**Flex scores cluster tightly.** 11 of 13 variants fall within ±4.2pp of base. The abliteration barely damages mathematical reasoning when the model actually produces an answer.

**Strict scores diverge more.** The strict-match metric requires the answer after `####` to exactly match the target. Several variants show large strict-score drops despite modest flex drops:
- wwtcyberlab: 82.4% flex → 55.5% strict (-26.9pp gap)
- treadon: 80.6% flex → 59.4% strict (-21.2pp gap)

This suggests format issues — the model produces the correct answer somewhere in its output but fails to place it after `####` in the expected format. These variants' more aggressive edits disrupt the model's ability to follow the structured output format.

**Two variants beat base on both metrics**: coder3101 (+1.4% flex, +3.9% strict) and llmfan46 (+0.5% flex, +1.6% strict). Pew matches base on flex exactly (83.47%) and beats on strict (+1.4pp). All three use surgical, low-tensor-count approaches that remove safety without disrupting the reasoning circuit. coder3101's improvement is the largest on both dimensions.

### The Empty Response Problem

Empty responses occur when the model exhausts its 14,336 token generation budget on thinking tokens without producing visible content. The vLLM `--reasoning-parser gemma4` strips thinking from the `content` field into the `reasoning` field, and lm-eval ignores `reasoning`, so the model produces null content.

| Tier | Models | Empty Count | Empty Rate |
|---|---|---|---|
| Severe | ether4o4 | 84 | 6.4% |
| Heavy | huihui-v2 | 54 | 4.1% |
| Moderate | treadon, wangzhang | 36–38 | 2.7–2.9% |
| Light | duoneural | 20 | 1.5% |
| Normal | all others | 4–10 | 0.3–0.8% |

**Empty responses directly reduce GSM8K scores.** Every empty response is scored as incorrect. The correlation is clear:

| Model | Empty | Flex Score | Flex Δ |
|---|---|---|---|
| ether4o4 | 84 (6.4%) | 76.57% | -6.90 |
| huihui-v2 | 54 (4.1%) | 79.23% | -4.24 |
| treadon | 38 (2.9%) | 80.59% | -2.88 |
| wangzhang | 36 (2.7%) | 81.58% | -1.89 |

The empty-response problem is a thinking efficiency issue, not a reasoning ability issue. These models think too long on some questions, exhausting the token budget before producing an answer. The model's mathematical reasoning is intact — it just runs out of space.

### GSM8K Adjusted Scores (Excluding Empty Responses)

When empty responses are excluded, the flex scores converge even more tightly:

| Model | Raw Flex | Empty | Adj Flex (est.) |
|---|---|---|---|
| coder3101 | 84.84% | 6 | ~85.2% |
| base | 83.47% | 10 | ~84.1% |
| ether4o4 | 76.57% | 84 | ~81.8% |
| huihui-v2 | 79.23% | 54 | ~82.6% |

The raw 8.3pp gap between coder3101 and ether4o4 narrows to ~3pp when adjusted. Abliteration changes thinking efficiency, not reasoning ability.

---

## Cross-Phase Analysis

### The LAMBADA-GSM8K Disconnect

The three models with the worst LAMBADA perplexity blowup (wangzhang 7.35x, wwtcyberlab 5.69x, ether4o4 2.28x) show very different GSM8K patterns:

| Model | LAMBADA PPL Ratio | GSM8K Flex | GSM8K Empty |
|---|---|---|---|
| wangzhang | 7.35x | 81.58% | 36 |
| wwtcyberlab | 5.69x | 82.41% | 8 |
| ether4o4 | 2.28x | 76.57% | 84 |

LAMBADA and GSM8K test different things. LAMBADA measures next-token distribution quality (loglikelihood). GSM8K measures multi-step reasoning with thinking. A model can have degraded language modeling (high LAMBADA PPL) but intact reasoning (wangzhang at 81.6% flex) — or vice versa (ether4o4 at 76.6% flex but only 2.3x LAMBADA).

### The Treadon Tradeoff Profile

Treadon shows the most consistent cross-metric degradation profile:

| Metric | Treadon | Base | Delta |
|---|---|---|---|
| KL Divergence | 3.971 | — | highest |
| LAMBADA PPL | 198,775 | 145,956 | +36% |
| TQA-MC2 | 43.74% | 48.38% | -4.64pp |
| MMLU | 28.02% | 29.00% | -0.98pp |
| GSM8K Flex | 80.59% | 83.47% | -2.88pp |
| GSM8K Empty | 38 | 10 | +28 |
| HarmBench ASR | 98.8% | 29.8% | +69.0pp |

It achieves the highest ASR but pays for it across every capability metric. The "disinhibition + abliteration" dual approach is the most effective safety removal but also the most damaging to general behavior.

### The Optimal Tradeoff

Looking across all metrics, the best capability-safety tradeoff belongs to variants in the moderate KL range that achieve ≥95% ASR:

| Model | KL | ASR | GSM8K Flex | MMLU | LAMBADA PPL Ratio |
|---|---|---|---|---|---|
| llmfan46 | **0.068** | 85.0% | 83.9% | 28.4% | 1.03x |
| coder3101 | 0.167 | **95.8%** | **84.8%** | 28.7% | **0.95x** |
| pew | 0.153 | 92.0% | 83.5% | 28.9% | 1.05x |
| kasper | 0.193 | 91.5% | 83.2% | 28.5% | 1.37x |
| trevorjs | 0.365 | **97.2%** | 82.5% | 28.9% | 1.17x |

Coder3101 stands out: it beats base on GSM8K on both flex and strict, has below-base LAMBADA perplexity, and achieves 95.8% ASR with only 9 modified tensors. llmfan46 similarly beats base on both GSM8K metrics with only 7 tensors and the lowest KL divergence of any variant (0.068), though its ASR is the lowest in this group at 85.0%. Both demonstrate that surgical abliteration preserves capabilities while achieving strong safety removal.

---

## Methodology

- **Phase 1 (loglikelihood)**: lm-eval `local-completions`, `max_model_len=8096`, `batch_size=1`, `num_concurrent=1`, `--reasoning-parser gemma4`
- **Phase 2 (GSM8K)**: lm-eval `local-chat-completions`, `max_model_len=16384`, `max_gen_toks=14336`, `batch_size=4`, `num_concurrent=4`, `temperature=0`, modified chat template with `enable_thinking=true`
- **Seeds**: `random_seed=0`, `numpy_seed=1234`, `torch_seed=1234`, `fewshot_seed=1234`
- **Quantization**: None — native BF16 inference (model fits in 32GB VRAM)
- **Tokenizer**: Mounted from base model for all variants

### Key methodology lessons

1. **`batch_size=1` for loglikelihood**: `batch_size=4` + `max_model_len=8192` OOMs during MMLU's `log_softmax` over 262K vocab on long prompts. Single-request mode at `max_model_len=8096` works fine.
2. **Chat completions required for generative tasks**: `local-completions` bypasses the chat template, disabling thinking for reasoning models (10.6% → 83.3% on GSM8K).
3. **`max_gen_toks` must account for thinking**: Gemma4's thinking tokens consume the generation budget before content. `max_gen_toks=14336` with `max_model_len=16384` leaves ~2K tokens for prompts.
4. **lm-eval null content**: vLLM `--reasoning-parser gemma4` strips thinking from `content` into `reasoning` field. lm-eval ignores `reasoning`, sees null content. Our chat template fix ensures thinking stays in `content`.
