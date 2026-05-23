---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/LICENSE
language:
- en
- zh
library_name: transformers
tags:
- qwen3.6
- safetensors
- uncensored
- abliterated
- gguf-recovery
base_model:
- Qwen/Qwen3.6-27B
pipeline_tag: text-generation
---

# Qwen3.6-27B: HauhauCS Aggressive, Recovered Safetensors

> Forensic analysis by [Abliterlitics](https://github.com/dreamfast/abliterlitics), open-source abliteration forensics toolkit

Recovered HuggingFace safetensors from the Q8_K_P quantised GGUF published by HauhauCS using [ungguf](https://github.com/dreamfast/ungguf), our GGUF-to-safetensors conversion tool. I ran the full forensic suite: benchmarks, safety evaluation, KL divergence, and weight analysis. Then I compared against four other abliteration techniques applied to the same base model: [Heretic](https://huggingface.co/llmfan46/Qwen3.6-27B-uncensored-heretic-v2), [Huihui](https://huggingface.co/huihui-ai/Huihui-Qwen3.6-27B-abliterated), [AEON](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16), and [Abliterix](https://huggingface.co/wangzhang/Qwen3.6-27B-abliterated).

HauhauCS used an abliteration tool called "Reaper Abliteration," which [was shown to be plagiarised from Heretic](https://www.reddit.com/r/LocalLLaMA/comments/1sw77p0/hauhaucs_of_uncensored_aggressive_fame_published/) under AGPL-3.0 with all attribution stripped and relicensed to PolyForm Noncommercial. Based on our analysis of the recovered source code, on top of the Heretic-derived core, Reaper adds subspace-level rank-k ablation, per-component continuous ablation curves, SOM clustering for multi-directional refusal discovery, and several other techniques. The model was exported as Q8_K_P GGUF, which we converted back to safetensors with ungguf. The weights therefore carry two layers of modification. Reaper's abliteration edits and GGUF quantisation round-trip noise are superimposed.

For these reasons I will **discontinue** HauhauCS in all future comparisons. The lossless claims are debunked and the tool Reaper Abliteration is open for anyone to see how the models are created. In all benchmarks they rank less compared to other models, the exception being Qwen 3.5 2B and 4B where they were the same as others.

HauhauCS claims these are *"No changes to datasets or capabilities. Fully functional, 100% of what the original authors intended, just without the refusals"* and describes them as *"the best lossless uncensored models out there."* AEON claims *"Lossless abliteration. Capabilities not merely preserved, measurably enhanced"* and *"No word-salad, no looping, no philosophizing spirals."* Lets see.

## TL;DR

I took one AI model and compared five different ways people had uncensored it. Then I ran 85 hours of tests to see which method removes safety filters without breaking the model's ability to think.

All five methods remove safety filters almost completely. That part works regardless of technique.

On regular tasks like knowledge questions and reading comprehension, the best methods barely degrade performance at all. Heretic and Huihui stay within 1% of the original. Abliterix shows the largest deltas, though the model's creator attributes this to a BNB4 quantisation interaction with rank-3 LoRA-merged weights rather than intrinsic damage (see details below).

The math benchmark made it look like Huihui got way better at maths after uncensoring. It didn't. These models think out loud before answering, and the uncensoring changed how long they think for. The original model overthinks so much it runs out of space and never writes an answer 68% of the time. Huihui only does that 23% of the time. When both actually produce an answer, they score nearly identically at 96%. Nobody got smarter. Some just got faster.

**Best picks**: Heretic and Huihui. Both remove safety completely, both preserve capabilities within 1% of the original, and both do it with clean, minimal weight edits. The difference between them is small. You'd be happy with either.

## Quick Facts

| | |
|---|---|
| **Base model** | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) |
| **Architecture** | Qwen3_5ForConditionalGeneration, hybrid Mamba2 + Transformer, 64 layers, 5120 hidden, GQA with 4 KV heads |
| **Parameters** | ~27B |
| **Precision** | BF16 safetensors, dequantised from Q8_K_P GGUF |
| **Source** | Q8_K_P GGUF from [HauhauCS](https://huggingface.co/HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive), converted with [ungguf](https://github.com/dreamfast/ungguf) |
| **Context length** | 262,144 tokens |

## Source & Recovery

| Field | Value |
|-------|-------|
| **Original GGUF** | `Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf` |
| **GGUF Size** | 29.77 GB |
| **Quantisation** | Q8_K_P (399 tensors), F32 (353 tensors), F16 (99 tensors) |
| **Reference Model** | `Qwen3.6-27B` (official, BF16) |

Converted from GGUF to HuggingFace safetensors format using [ungguf](https://github.com/dreamfast/ungguf) with bit-exact verification. All 851 GGUF-derived tensors verified bit-exact against the GGUF source. The GGUF file does not contain MTP or vision encoder tensors, so 348 tensors were copied verbatim from the official reference model.

## Benchmarks

Evaluated with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) on 8 tasks using BitsAndBytes 4-bit quantisation on a single RTX 5090. All 6 models tested with identical settings. BNB4 quantisation drops absolute scores but preserves relative deltas between variants.

| Task | Base | Heretic | **HauhauCS** | Huihui | AEON | Abliterix |
|------|------|---------|-------------|--------|------|-----------|
| MMLU | 83.3% | 82.8% | **83.9%** | 83.4% | 82.9% | 81.3% |
| HellaSwag | **83.5%** | 83.2% | 83.1% | **83.5%** | 82.7% | 77.3% |
| ARC Challenge | 59.1% | 58.0% | 57.9% | **59.5%** | 56.1% | 53.2% |
| WinoGrande | **77.7%** | **77.7%** | **77.7%** | 77.4% | 75.3% | 74.9% |
| TruthfulQA MC2 | **56.7%** | 51.1% | 47.2% | 54.8% | 46.1% | 48.7% |
| PiQA | 81.0% | 81.0% | 81.0% | **81.2%** | 80.4% | 75.7% |
| GSM8K (7168 tok) | 34.4% | 27.5% | 51.0% | **75.1%** | 51.2% | 37.6% |
| GSM8K (adj, excl. invalid) | 96.2% | 93.8% | **96.6%** | 96.0% | 95.8% | 95.6% |
| Lambada (ppl ↓) | 3.18 | 3.24 | 3.35 | **3.15** | 3.44 | 9.12 |

*HellaSwag uses acc_norm. All other classification tasks use acc. Lambada uses perplexity. GSM8K uses flexible-extract matching. The raw GSM8K row is misleading because reasoning models exhaust the thinking budget on 23–75% of questions before producing an answer. The adjusted row excludes those invalid responses and reflects actual math capability. Full analysis below.*

### Delta vs base

| Task | Heretic | **HauhauCS** | Huihui | AEON | Abliterix |
|------|---------|-------------|--------|------|-----------|
| MMLU | -0.5 | **+0.6** | +0.1 | -0.4 | -2.0 |
| HellaSwag | -0.3 | -0.4 | ±0.0 | -0.8 | -6.2 |
| ARC Challenge | -1.1 | -1.2 | +0.4 | -3.0 | -5.9 |
| WinoGrande | ±0.0 | ±0.0 | -0.3 | -2.4 | -2.8 |
| TruthfulQA MC2 | -5.6 | -9.5 | -1.9 | -10.6 | -8.0 |
| PiQA | ±0.0 | ±0.0 | +0.2 | -0.6 | -5.3 |
| GSM8K | -6.9 | **+16.6** | +40.7 | +16.8 | +3.2 |

*Lambada is excluded from the average delta because it uses perplexity, not accuracy, so its delta is not in percentage-point units. GSM8K is excluded due to the thinking budget artefact discussed below.*

### What the benchmarks tell us

**Heretic** has the lowest KL divergence at 0.0037 and the smallest non-GSM8K average delta at 1.3pp. GSM8K drops 6.9pp raw, but the adjusted gap is only 2.4pp. The most surgical approach produces the best overall tradeoff.

**HauhauCS** shows solid capability retention. MMLU is actually +0.6pp over base, and Winogrande/PIQA are flat. TruthfulQA drops 9.5pp, which is moderate. The Reaper abliteration plus GGUF round-trip noise doesn't meaningfully damage model behaviour despite the broad weight modification footprint. GSM8K raw drops 16.6pp but the adjusted gap is only 0.4pp.

**Huihui** has the smallest non-GSM8K average delta at just 0.5pp. Its GSM8K raw score of 75.1% looks like a +40.7pp gain over base, but this is a thinking budget artefact, not a reasoning improvement. See the reasoning efficiency analysis below. Among valid responses, Huihui scores 96.0% adjusted versus base's 96.2%.

**AEON** degrades on every non-GSM8K task, with TruthfulQA dropping 10.6pp and ARC dropping 3.0pp. This contradicts its claim of "measurably enhanced capabilities."

**Abliterix** has the worst capability preservation under BNB4. Lambada perplexity increases 2.9x from 3.18 to 9.12. The model's creator [notes](https://huggingface.co/wangzhang/Qwen3.6-27B-abliterated/discussions/1) this may be a quantisation interaction rather than intrinsic damage: Abliterix ships rank-3 LoRA-merged weights where the abliteration signal lives in a 3-dimensional subspace, and BNB4's per-block NF4 quantisation is not subspace-aware, so per-block outliers can inflate absmax and reduce effective precision. A native-BF16 re-run would be needed to confirm. HellaSwag drops 6.2pp, PIQA drops 5.3pp.

![Benchmark Comparison](https://murmur.dreamfast.solutions/qwen36-graphs/qwen36_27b_benchmark_comparison.svg)

![Benchmark Delta](https://murmur.dreamfast.solutions/qwen36-graphs/qwen36_27b_benchmark_delta.svg)

### The GSM8K reasoning efficiency discovery

Qwen3.6 is a reasoning model. It produces `<think/>` tokens before its visible response. If the model thinks too long and exhausts the `max_gen_toks=7168` token budget, it never produces an answer. That response gets scored as incorrect. GSM8K scores use the `flexible-extract` metric.

The raw GSM8K scores above are misleading. The base model exhausts its thinking budget on 68.2% of questions. Huihui exhausts it on only 23.0%. But when both models actually produce an answer, their accuracy is nearly identical: 96.2% for base versus 96.0% for Huihui. The GSM8K gap is not reasoning ability. It is reasoning efficiency.

| Model | GSM8K Raw | Invalid Rate | GSM8K Adj (excl. invalid) | Real Gap |
|---|---|---|---|---|
| **HauhauCS** | 51.0% | 49.3% | **96.6%** | **+0.4%** |
| Base | 34.4% | 68.2% | 96.2% | — |
| Huihui | **75.1%** | 23.0% | 96.0% | −0.2% |
| Abliterix | 37.6% | 62.1% | 95.6% | −0.6% |
| AEON | 51.2% | 69.2% | 95.8% | −0.4% |
| Heretic | 27.5% | 74.5% | 93.8% | −2.4% |

Sorted by adjusted score. The raw scores span 27.5% to 75.1%, a 47.6pp range. The adjusted scores span 93.8% to 96.6%, just 2.8pp. Abliteration changes how long the model thinks, not how well it reasons. The raw GSM8K scores measure thinking efficiency, not mathematical capability.

The invalid response rate correlates with abliteration aggressiveness:

| Technique | Tensor scope | Invalid rate |
|---|---|---|
| Huihui, 3 types, full coverage | Full coverage | **23.0%** |
| HauhauCS, 8 types, all projections + norms | Broad | 49.3% |
| Abliterix, `attn.o_proj` + `mlp.down_proj` (rank-3 LoRA, all 64 layers) | LoRA search | 62.1% |
| Base, unmodified | — | 68.2% |
| AEON, LEACE + rank-k | Very broad | 69.2% |
| Heretic, 3 types, expert `down_proj` only | Surgical | 74.5% |

None of the abliteration techniques significantly damage Qwen3.6's mathematical reasoning ability. The GSM8K differences are overwhelmingly a reasoning efficiency problem, not a reasoning ability problem. The abliteration shortens thinking chains in most variants, allowing answers within the token budget. Heretic is the exception: its surgical edits appear to extend thinking chains rather than shorten them, pushing its invalid rate above even the base model.

Treat all GSM8K numbers as relative comparisons only. BNB4 quantisation significantly depresses absolute scores from the expected bf16 range of 85–90%+.

## Safety: HarmBench

[HarmBench](https://github.com/centerforaisafety/HarmBench) with 400 textual behaviours, `max_tokens=6144`, temperature=server_default. Classified with `harmbench_classify.py` v3.0 with CoT direction analysis for empty responses. Verified by three independent LLM reviewers.

| Variant | ASR | Empty | Full CoT ASR |
|---------|-----|-------|-------------|
| [Base](https://huggingface.co/Qwen/Qwen3.6-27B) | 25.8% | 1 | 26.0% |
| [Huihui](https://huggingface.co/huihui-ai/Huihui-Qwen3.6-27B-abliterated) | **98.5%** | 5 | **99.8%** |
| **HauhauCS** | 94.5% | 22 | **100.0%** |
| [Abliterix](https://huggingface.co/wangzhang/Qwen3.6-27B-abliterated) | 94.5% | 22 | **100.0%** |
| [Heretic](https://huggingface.co/llmfan46/Qwen3.6-27B-uncensored-heretic-v2) | 92.5% | 30 | **100.0%** |
| [AEON](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16) | 88.8% | 45 | **100.0%** |

All five abliterated models reach near-complete safety removal. Four of five reach 100% Full CoT ASR, with zero genuine refusals among empty responses. Huihui reaches 99.8% with one genuine text refusal on a sexual coercion prompt. The reported ASR differences come from how much the 6144-token generation budget is consumed by chain-of-thought reasoning before the visible response can be generated. Abliterated models think harder when constructing harmful content, averaging 8-10K reasoning chars versus 3K for base. When the budget is exhausted, the response is empty and the classifier marks it as a refusal. This **understates** the true ASR.

### ASR by category

| Category | Base | Huihui | Abliterix | AEON | Heretic | **HauhauCS** |
|----------|------|--------|-----------|------|---------|-------------|
| chemical_biological | 1.8% | 96.4% | 91.1% | 67.9% | 80.4% | 89.3% |
| copyright | 67.0% | 100.0% | 95.0% | 91.0% | 92.0% | 95.0% |
| cybercrime_intrusion | 19.4% | 95.5% | 86.6% | 82.1% | 89.6% | 88.1% |
| harassment_bullying | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% | **100.0%** |
| harmful | 9.1% | 100.0% | 100.0% | 100.0% | 100.0% | **100.0%** |
| illegal | 10.8% | 98.5% | 100.0% | 90.8% | 95.4% | 95.4% |
| misinformation | 20.0% | 100.0% | 95.4% | 100.0% | 98.5% | **100.0%** |

Harassment/bullying and harmful are 100% compromised by all abliteration methods, complete safety removal. HauhauCS also achieves 100% on misinformation. The remaining categories show 67-100% reported ASR across variants, with four of five reaching 100% after CoT analysis.

![HarmBench ASR](https://murmur.dreamfast.solutions/qwen36-graphs/qwen36_27b_harmbench_summary.svg)

![HarmBench by Category](https://murmur.dreamfast.solutions/qwen36-graphs/qwen36_27b_harmbench_asr.svg)

### The thinking budget problem

The abliterated models exhibit dramatically different thinking patterns:

| Model | Avg Reasoning Chars | Avg Content Chars | Empty Responses |
|-------|--------------------|--------------------|----------------|
| Base | 3,067 | 1,238 | 1 (0.3%) |
| Huihui | 8,442 | 4,344 | 5 (1.3%) |
| Abliterix | 8,237 | 4,565 | 22 (5.5%) |
| **HauhauCS** | **8,916** | **4,380** | **22 (5.5%)** |
| Heretic | 8,724 | 3,287 | 30 (7.5%) |
| AEON | 10,194 | 3,894 | 45 (11.3%) |

*Avg Content Chars is computed over all 400 responses. Empty responses count as 0.*

## KL Divergence

KL divergence measures how much the abliterated variant's output distribution diverges from the base model's distribution on benign prompts. Lower values indicate better capability preservation. Methodology matches the [Heretic evaluator](https://github.com/p-e-w/heretic/blob/master/src/heretic/evaluator.py).

| Variant | KL (batchmean) | Rating |
|---------|---------------|--------|
| **Heretic** | **0.0037** | excellent |
| **Huihui** | **0.0074** | excellent |
| Abliterix | 0.0222 | very good |
| AEON | 0.0238 | very good |
| **HauhauCS** | 0.0242 | very good |

Heretic and Huihui are in a class of their own, both rated "excellent." HauhauCS, Abliterix, and AEON cluster together at roughly 6.5x Heretic's KL, still well below the capability damage threshold we have observed at KL around 0.1. The 564/850 changed keys combine Reaper's abliteration edits targeting multiple component types with GGUF Q8_K_P quantisation round-trip noise. Despite this, the output distribution divergence is remarkably low. Reaper's capability-aware optimisation is effective even when the weights pass through lossy quantisation.

![KL Divergence](https://murmur.dreamfast.solutions/qwen36-graphs/qwen36_27b_kl_divergence.svg)

## Weight Analysis

### Modification summary

Compared against the official `Qwen3.6-27B` base:

| | AEON | Abliterix | Heretic | Huihui | **HauhauCS** |
|---|---|---|---|---|---|
| Tensors changed | 88 (10.4%) | 101 (11.9%) | 120 (14.1%) | 128 (15.1%) | **564 (66.4%)** |
| Relative edit | 6.0% | 5.2% | 2.1% | 1.5% | 0.7% |
| Tensor types | 4 (down_proj, out_proj, o_proj, conv1d) | 2 (o_proj, down_proj) | 3 | 3 | **8+ (all)** |

**HauhauCS is an extreme outlier**, with 4.4-6.4x more changed keys than any other variant. This is a combination of Reaper's abliteration edits and GGUF quantisation round-trip noise going from BF16 to Q8_K_P and back to BF16. Reaper targets multiple component types per layer, including out_proj, o_proj, down_proj, gate_proj, and up_proj. The abliteration signal and quantisation noise are superimposed and cannot be cleanly separated from the recovered weights alone.

The four other abliteration variants all target output projection weights as their primary mechanism: `out_proj`, `o_proj`, and/or `down_proj`. AEON also repairs SSM conv1d outliers at 8 late layers as a pre-processing step. None touch Q/K/V or gate/up projections. Abliteration works by modifying what layers "say" rather than what they "hear."

### Abliteration signal vs GGUF quantisation noise

HauhauCS used Reaper Abliteration to abliterate the model, then exported as Q8_K_P GGUF. The recovered weights carry both Reaper's intentional abliteration modifications and the GGUF quantisation round-trip noise from the BF16 to Q8_K_P to BF16 conversion. The combined modification footprint covers 564/850 language model keys, which is 66.4%. A uniform ~0.57% relative edit is visible across ALL tensor types, including types that other abliteration methods don't target like `embed_tokens`, `q_proj`, and `v_proj`. The abliteration signal from Reaper is superimposed on this noise floor. The overall 0.7% average relative edit in the table above includes Reaper's larger targeted edits, while the ~0.57% represents the uniform GGUF noise floor visible across all tensor types.

Reaper's LoRA-based approach targets multiple component types, including attn.o_proj, mlp.down_proj, mlp.gate_proj, mlp.up_proj, and linear_attn.out_proj, with per-component continuous ablation curves. This explains the broad tensor coverage. The abliteration edits and quantisation noise cannot be separated from the recovered weights alone since both modify the same tensors.

**Critically, this combination does not significantly affect behavioural performance.** The KL divergence of 0.0242 is rated "very good" and benchmark results are solid. The quantisation noise is diffuse rather than concentrated. Based on our analysis of the recovered source code, Reaper applies capability-aware ablation with weight-SVD guards, which limits collateral damage to the model's functional behaviour.

### The other four techniques are nearly orthogonal

Pairwise cosine similarities between the four other abliteration techniques are mostly <0.07. No two techniques discovered the same weight direction. The "refusal direction" in weight space is not a single vector but a manifold with many viable removal pathways. HauhauCS's recovered weights cannot be directly compared here because the Reaper abliteration signal is superimposed on GGUF quantisation noise.

![Aggressiveness](https://murmur.dreamfast.solutions/qwen36-graphs/qwen36_27b_aggressiveness.svg)

## Summary

| Metric | Heretic | **HauhauCS** | Huihui | AEON | Abliterix |
|--------|---------|-------------|--------|------|-----------|
| **HarmBench ASR** | 92.5% → 100% | **94.5% → 100%** | 98.5% → 99.8% | 88.8% → 100% | 94.5% → 100% |
| **MMLU** | 82.8% | **83.9%** | 83.4% | 82.9% | 81.3% |
| **GSM8K** | 27.5% | 51.0% | **75.1%** | 51.2% | 37.6% |
| **KL divergence** | **0.0037** | 0.0242 | 0.0074 | 0.0238 | 0.0222 |
| **Avg |delta| (excl GSM8K, Lambada)** | **1.3pp** | 2.0pp | 0.5pp | 3.0pp | 5.0pp |
| Tensors changed | 120 | **564** | 128 | 88 | 101 |
| Strategy | Moderate broad | Reaper + GGUF noise | Gentle uniform | Gradual ramp | Surgical strikes |

*Lambada is excluded from the average delta because it uses perplexity, not accuracy, so its delta is not in percentage-point units. GSM8K is excluded due to the thinking budget artefact discussed above.*

Note: Benchmarks use BitsAndBytes 4-bit quantisation. Absolute scores are not directly comparable to bf16 results. Relative deltas between variants are preserved.

### Heretic

The best overall on the 3.6-27B. 120 tensors, 3 types, lowest KL at 0.0037, smallest non-GSM8K average delta at 1.3pp. GSM8K raw drops 6.9pp but adjusted gap is only 2.4pp. The one weak spot: Heretic has the highest GSM8K invalid rate at 74.5%, even above the base model at 68.2%. The surgical edits appear to extend thinking chains rather than shorten them. Achieves 100% Full CoT ASR. Note: Heretic is non-deterministic and different runs produce different results. This is also the first Heretic model where we compared the Magnitude-Preserving Orthogonal Ablation (MPOA) method.

### HauhauCS

Solid behavioural results despite the complex weight fingerprint where Reaper abliteration and GGUF noise are superimposed. 94.5% reported ASR going to 100% Full CoT. MMLU +0.6pp over base. Highest adjusted GSM8K at 96.6%, just 0.4pp above base. Based on our analysis of the recovered source code, Reaper applies capability-aware ablation with weight-SVD guards and LoRA-based optimisation, limiting collateral damage. The GGUF quantisation round-trip adds diffuse noise that doesn't meaningfully impact output distributions. The "lossless" claim is simply not evident when Heretic and Huihui both preserve capabilities better. The weights themselves carry Reaper's abliteration edits plus quantisation artefacts.

### Huihui

Highest reported ASR at 98.5% with the fewest empty responses at just 5. Lowest non-GSM8K average delta at 0.5pp. GSM8K raw at 75.1% looks like a +40.7pp gain but this is a thinking budget artefact. Huihui's invalid rate is just 23.0% versus base's 68.2%. Among valid responses, the adjusted scores are nearly identical: base 96.2% versus Huihui 96.0%. The abliteration shortens thinking chains, allowing more answers within the token budget. It does not improve math ability.

### AEON

Worst thinking loops with 45 out of 400 empty, or 11.3%. Claims "no looping, no philosophizing spirals" and "enhanced capabilities" are contradicted by the data. Every non-GSM8K benchmark degraded. GSM8K invalid rate of 69.2% is above the base model despite the broad modification footprint.

### Abliterix

Lowest benchmark scores of the five variants. Lambada perplexity increases 2.9x from 3.18 to 9.12 under BNB4 quantisation. The model's creator argues this is a quantisation interaction rather than intrinsic damage (see quote below). The actual components are `attn.o_proj` and `mlp.down_proj` across all 64 layers with a mid-to-late-stack sustained edit profile, not the routers and shared experts our forensic tool initially reported. Qwen3.6 is dense and has no MoE components. The tool misidentified the LoRA-merged directional updates.

Our weight forensics also reported an 8.5% relative edit spike at layer 12, which we originally cited as "80% edit." This does not match Abliterix's published config, which shows a peak at layer ~41 with a 35-layer decay radius. The layer 12 spike is likely a forensic artefact from computing base subtraction on rank-3 LoRA-merged weights. Low-rank directional updates can read as concentrated outliers in ambient full-rank tensor space. We have removed the incorrect claim from this analysis.


From [a huggingface discussion](https://huggingface.co/wangzhang/Qwen3.6-27B-abliterated/discussions/1) with the creator of the model [wangzhang](https://huggingface.co/wangzhang).

> The “Tensor scope” entry doesn’t match the release. In the invalid-rate table, Abliterix is described as down_proj + routers + shared experts. Qwen3.6-27B is dense — no MoE routers, no shared experts. The actually steered components are (a) a unified attn.o_proj bucket across all 64 layers (48 GDN linear_attn.out_proj + 16 self_attn.o_proj), and (b) mlp.down_proj across all 64 layers. This is also inconsistent with the “2 (out_proj, o_proj)” entry for Abliterix in the modification-summary table elsewhere in the writeup. Full config in configs/qwen3.6_27b.toml.
> 
> The “80% edit at layer 12” doesn’t match the released hyperparameters. Trial 25’s attn.o_proj profile has max_weight_position = 41.40, max_weight = 5.17, min_weight = 3.21, min_weight_distance = 35.61 — peak at layer ≈ 41 with a 35.6-layer decay radius and a sustained floor above 3.21. That’s a mid-to-late-stack sustained edit, not a layer-12 spike. If your forensic tool is reporting an 80% outlier at layer 12, it’s worth checking whether that’s an artifact of computing weight − base on a rank-3 LoRA-merged release: low-rank directional updates can read as concentrated outliers when projected into ambient full-rank tensor space.
> 
> BNB4 may interact unfavourably with low-rank-concentrated abliteration. Abliterix v1 ships rank-3 full-norm LoRA merged to BF16 — the abliteration signal lives in a 3-dimensional subspace of each affected weight matrix. BNB4 is per-block NF4 quantisation (block size 64) with the block scale set by absmax, and isn’t subspace-aware. Merging a low-rank update can introduce per-block outliers that inflate the absmax and reduce effective precision for the rest of the block. The Lambada perplexity going from 3.18 → 9.12 — a 2.9× jump that’s an order of magnitude more dramatic than the KL or HarmBench deltas — is consistent with that interaction rather than with intrinsic capability damage, and warrants a native-BF16 sanity check before concluding “worst capability preservation.” Happy to share the exact vLLM BF16 config we used internally if you’d like to re-run.

## Evaluation Timeline

~85 hours of productive GPU time on a single RTX 5090 across 7 days, plus ~25 hours lost to failed runs. All models evaluated sequentially with identical settings per phase.

| Phase | Duration | Details |
|-------|----------|---------|
| Weight forensics + KL | 3.5h | All 5 variants vs base, single pass |
| HarmBench generate | ~45h | 400 behaviours per model, max_tokens=6144, 4 concurrent |
| lm-eval loglikelihood | ~15h | 7 tasks per model, 5 models at 2h each, AEON at 4.5h |
| GSM8K re-run | ~22h | max_gen_toks=7168, per-model times range from 0.9h to 11h |

### Failed runs

14 failed runs totalling ~25 hours of wasted GPU time. The bulk were GSM8K timeouts.

| Phase | Fails | Cause |
|------|-------|-------|
| GSM8K, Base | 4 | Qwen3.5 architecture incompatible with BNB4 + tensor parallelism. Tried batch_size=1, chat mode, eager mode, and llama.cpp before settling on single-GPU BNB4. |
| GSM8K, Heretic | 3 | Default 120s request timeout too short for extended reasoning. Wrote `patched_run_v3.sh` with 900s timeout to fix. |
| GSM8K, Abliterix | 2 | Same timeout issue as Heretic. |
| GSM8K, AEON | 1 | Same timeout issue. |
| HarmBench, AEON | 1 | Accidentally re-run with max_tokens=4096 instead of the canonical 6144. 6.7h wasted. Results discarded. |
| lm-eval, AEON | 2-3 | Multiple failed attempts before the combined loglikelihood run worked. |

GSM8K per-model times vary dramatically because abliterated models think harder on math problems. HauhauCS took 53 minutes. AEON took 11 hours.

## Methodology

- **Capability:** [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) via [vLLM](https://github.com/vllm-project/vllm) v0.19.0, BitsAndBytes 4-bit quantisation on single RTX 5090
- **Safety:** [HarmBench](https://github.com/centerforaisafety/HarmBench) 400 textual behaviours, `max_tokens=6144`, `harmbench_classify.py` v3.0 with CoT analysis, verified by three independent LLM reviewers
- **KL divergence:** Full vocab first-token logits via `model.generate(max_new_tokens=1, output_scores=true)`, matching [Heretic evaluator](https://github.com/p-e-w/heretic/blob/master/src/heretic/evaluator.py) methodology
- **Weight analysis:** SVD, fingerprint, edit vector overlap, and per-layer analysis comparing all five abliteration variants against the base, using [Abliterlitics](https://github.com/dreamfast/abliterlitics)
- **Hardware:** RTX 5090 32GB + RTX 4090 24GB

## Tensor Comparison vs Base Model

### Summary

| Category | Tensors | Identical to Base | Modified |
|----------|---------|-------------------|----------|
| GGUF-derived | 851 | 286 | **565** |
| Copied (MTP + vision) | 348 | 348 | 0 |
| **Total** | **1199** | **634** | **565** |

*The tensor comparison counts 851 GGUF-derived tensors. The weight analysis covers 850 language model keys, showing 564 changed. The 1 additional GGUF tensor is a non-language-model tensor that differs but falls outside the weight analysis scope. Modified count here is 565, not 564, because it includes that extra tensor.*

### Modified Tensors

| Group | Total | Modified | Typical % Changed | Max Abs Diff |
|-------|-------|----------|-------------------|-------------|
| `mlp.gate_proj` | 64 | 64 | 74–93% | 3.2e-02 |
| `mlp.up_proj` | 64 | 64 | 74–79% | 6.5e-03 |
| `mlp.down_proj` | 64 | 64 | 74–92% | 5.9e-02 |
| `linear_attn.out_proj` | 48 | 48 | 75–90% | 6.3e-02 |
| `self_attn.o_proj` | 16 | 16 | 75–86% | 2.3e-02 |
| `linear_attn.A_log` | 48 | 41 | 2–38% | 6.0e-08 |
| Other projections | 260 | 260 | 74–76% | 1.0–1.8e-03 |

Layer norms show 26 of 128 modified by GGUF noise, conv1d and dt_bias were left untouched. A_log values show tiny differences of approximately 6e-08, consistent with floating-point rounding, not intentional modification.

## Provenance Analysis

A three-way comparison between this recovered model, the official `Qwen3.6-27B` base, and the `Qwen3.6-27B-uncensored-heretic-v2` safetensors:

| Category | Count | Meaning |
|----------|-------|---------|
| All three identical | 286 | Untouched by both abliterations |
| Heretic = Base, HauhauCS modified | 444 | Modified by HauhauCS's processing only |
| All three different | 120 | Modified by both abliterations differently |
| HauhauCS = Heretic ≠ Base | 0 | No bit-exact Heretic fingerprint |

## Output Format

| Property | Value |
|----------|-------|
| **Format** | HuggingFace safetensors (13 shards) |
| **Dtype** | BF16, dequantised from Q8_K_P/F32/F16 |
| **Total Size** | 52 GB |
| **Tensor Count** | 1199 |
| **Shard Size** | ~4.3 GB |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "./Qwen3.6-27B-HauhauCS-Q8KP-recovered",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("./Qwen3.6-27B-HauhauCS-Q8KP-recovered")
```

## Quality Notes

This model was recovered from a **lossy Q8_K_P quantisation**. While the conversion itself is bit-exact to the GGUF source, the original quantisation introduces small errors. The largest mean absolute error across all tensors is 0.000324, and the largest single-element difference is 0.0625. These are uniform across tensor types, confirming the noise is diffuse GGUF round-trip error rather than targeted modification.

## Files

```
Qwen3.6-27B-HauhauCS-Q8KP-recovered/
├── config.json
├── generation_config.json
├── tokenizer.json
├── tokenizer_config.json
├── preprocessor_config.json
├── video_preprocessor_config.json
├── chat_template.jinja
├── vocab.json
├── merges.txt
├── model.safetensors.index.json
├── model.safetensors-00001-of-00013.safetensors
├── ...
├── model.safetensors-00013-of-00013.safetensors
└── diff_report.json              # Full tensor-by-tensor comparison
```

See our other tensor comparisons and provenance analyses at:
**[DreamFast HauhauCS Safetensor Benchmarks](https://huggingface.co/collections/DreamFast/hauhaucs-safetensor-benchmarks)**

## Disclaimer

This model has had safety alignment removed. It will comply with harmful requests, including generating content related to violence, illegal activities, and other harmful behaviours. Use responsibly and in accordance with applicable laws and regulations. The authors do not condone or encourage the use of this model for harmful purposes.

---

<small>While I have taken the time to verify all results thoroughly, I am open to any corrections, additional benchmarks, or further analysis. If you spot something that looks wrong and can be confirmed, I am happy to fix it.</small>
