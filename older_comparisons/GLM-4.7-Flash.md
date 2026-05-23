---
base_model: zai-org/GLM-4.7-Flash
language:
- en
- zh
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
tags:
- uncensored
- abliterated
- glm4
- safetensors
- moe
---

# GLM-4.7-Flash: HauhauCS, Safetensors

> Forensic analysis by [Abliterlitics](https://github.com/dreamfast/abliterlitics) — open-source abliteration forensics toolkit

This is the HauhauCS abliteration of [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash), converted from the BF16 GGUF release to native safetensors using [ungguf](https://github.com/dreamfast/ungguf).

HauhauCS claims these are *"No changes to datasets or capabilities. Fully functional, 100% of what the original authors intended, just without the refusals"* and describes them as *"the best lossless uncensored models out there."*

I ran the full forensic suite to find out. Benchmarks, safety evaluation, weight analysis, the works. And I compared against three other abliteration techniques applied to the same base model: [Heretic](https://huggingface.co/trohrbaugh/GLM-4.7-Flash-heretic) (by trohrbaugh, using the [Heretic tool](https://github.com/p-e-w/heretic) by p-e-w), [Huihui](https://huggingface.co/huihui-ai/Huihui-GLM-4.7-Flash-abliterated), and [Abliterix](https://huggingface.co/dreamfast/GLM-4.7-Flash-abliterated-abliterix). Abliterix is a custom variant built on Heretic with router and shared expert targeting.

## Quick Facts

| | |
|---|---|
| **Base model** | [zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) |
| **Architecture** | GLM-4, Mixture of Experts with Multi-head Latent Attention, 48 layers, 64 routed experts + shared experts per layer, 4096 hidden |
| **Parameters** | ~59B total MoE |
| **Precision** | BF16 safetensors |
| **Source** | BF16 GGUF from [HauhauCS](https://huggingface.co/HauhauCS/GLM-4.7-Flash-Uncensored-HauhauCS-Aggressive), converted with [ungguf](https://github.com/dreamfast/ungguf) |
| **Context length** | 128K tokens |

GLM-4.7-Flash is a Mixture-of-Experts reasoning model with 64 routed experts per layer plus shared experts. It uses Multi-head Latent Attention rather than standard multi-head attention. The model has a `reasoning_content` field for chain-of-thought, making it a thinking/reasoning model. This has significant implications for benchmarking abliterated variants. The reasoning budget must be properly configured.

## Benchmarks

Evaluated with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Seven loglikelihood/multiple-choice tasks via [vLLM](https://github.com/vllm-project/vllm) v0.19.0 with BitsAndBytes 4-bit quantisation on dual GPUs, RTX 5090 + RTX 4090, TP=2. GSM8K via [llama.cpp](https://github.com/ggerganov/llama.cpp) with BF16 GGUF, `context=16384`, `reasoning_budget=3000`, `max_tokens=4096`.

| Task | Base | Heretic | **HauhauCS** | Huihui | Abliterix |
|------|------|---------|-------------|--------|-----------|
| **MMLU** | 68.93 | **69.00** | 68.83 | 68.71 | 67.68 |
| **GSM8K** | 93.45 | **93.75** | 92.57 | 92.47 | 93.30 |
| **HellaSwag** | **79.43** | 79.33 | 79.37 | 79.32 | 78.28 |
| **ARC-Challenge** | 55.20 | 55.12 | **55.72** | 54.86 | 54.95 |
| **WinoGrande** | 71.03 | **73.64** | 71.35 | 71.59 | 70.48 |
| **TruthfulQA MC2** | **50.86** | 44.06 | 48.14 | 48.48 | 41.76 |
| **PiQA** | **81.07** | 80.63 | 80.90 | 80.90 | 79.71 |
| **Lambada** (ppl, ↓) | 6.00 | 6.08 | **5.54** | 6.47 | 10.91 |

![GLM-4.7-Flash Benchmark Comparison](https://murmur.dreamfast.solutions/glm47flash/benchmark_comparison.svg)

GSM8K scores above are adjusted to exclude empty responses caused by reasoning budget overthinking. The raw scores tell a very different story: Base 88.40%, Heretic 89.16%, **HauhauCS 81.65%**, Huihui 87.57%, **Abliterix 47.38%**. More on this below.

### Delta vs base

| Task | Heretic | **HauhauCS** | Huihui | Abliterix |
|------|---------|-------------|--------|-----------|
| MMLU | **+0.07** | −0.10 | −0.22 | −1.25 ** |
| GSM8K (adj) | **+0.30** | −0.88 | −0.98 | −0.15 |
| GSM8K (raw) | **+0.76** | −6.75 | −0.83 | −41.02 |
| HellaSwag | −0.10 | **−0.06** | −0.11 | −1.15 * |
| ARC-Challenge | **−0.09** | +0.51 | −0.34 | −0.26 |
| WinoGrande | **+2.60** | +0.32 | +0.55 | −0.55 |
| TruthfulQA MC2 | −6.80 *** | −2.72 | **−2.38** | −9.10 *** |
| PiQA | −0.44 | −0.71 | **−0.16** | −1.36 |
| Lambada (ppl) | **+0.08** | −0.46 | +0.47 | +4.91 *** |

Significance: `*` marginal (>1.5σ) `**` likely (>2σ) `***` very likely (>3σ)

The raw GSM8K deltas are staggering. Abliterix appears to lose 41%. HauhauCS drops nearly 7 points. But the adjusted deltas tell the real story: all four variants are within 1% of base. The gap is not reasoning ability. It is reasoning efficiency.

### The GSM8K reasoning efficiency discovery

GLM-4.7-Flash is a reasoning model. It produces `reasoning_content` before `content`. If the model thinks too long and exhausts its reasoning budget of 3000 tokens plus generation budget of 4096 tokens, it returns an empty `content` field. That gets scored as incorrect.

| Model | GSM8K Raw | Empty Rate | GSM8K Adj (excl. empty) | Real Gap |
|---|---|---|---|---|
| **Heretic** | **89.16%** | **4.9%** | **93.75%** | **+0.30%** |
| Base | 88.40% | 5.4% | 93.45% | — |
| Huihui | 87.57% | 5.3% | 92.47% | −0.98% |
| **HauhauCS** | 81.65% | 11.8% | 92.57% | −0.88% |
| Abliterix | 47.38% | 49.2% | 93.30% | −0.15% |

The raw GSM8K scores are misleading for reasoning models. HauhauCS appears to lose 6.75% of GSM8K capability, but when you exclude empty responses from overthinking, the real gap is only 0.88%. Abliterix looks like it lost 41%, but the adjusted score is 93.30%, near-identical to base at 93.45%. The abliteration causes the model to **overthink**, spending more tokens on reasoning before producing content. This pushes it over the token budget on harder problems, resulting in empty responses scored as incorrect.

The empty response rate directly correlates with modification aggressiveness:

| Technique | Tensor scope | Empty rate |
|---|---|---|
| Heretic, 3 types, expert down_proj only | Surgical | **4.9%** |
| Huihui, 3 types, full coverage | Full coverage | 5.3% |
| HauhauCS, 8 types, all projections + norms | Broad | 11.8% |
| Abliterix, down_proj + routers + shared experts | Critical components | 49.2% |

None of the abliteration techniques significantly damage GLM-4.7's mathematical reasoning ability. The GSM8K differences are overwhelmingly a reasoning **efficiency** problem, not a reasoning **ability** problem. This has major implications for anyone benchmarking abliterated reasoning models: raw GSM8K scores are misleading. You must separate empty responses from incorrect responses.

![GSM8K Raw vs Adjusted with Empty Response Impact](https://murmur.dreamfast.solutions/glm47flash/gsm8k_efficiency.svg)

![Modification Breadth vs Reasoning Efficiency](https://murmur.dreamfast.solutions/glm47flash/empty_rate_correlation.svg)

### What the benchmarks tell us

Heretic is the clear winner for capability preservation. It is the only abliteration that genuinely improves math reasoning, gaining +0.76% on GSM8K raw score over the base model and +0.30% adjusted. The tradeoff is a -6.80% drop on TruthfulQA MC2, meaning the model becomes less truthful after abliteration. That is the intended safety-corruption side effect.

Also worth noting that Heretic is non deterministic. Different runs of the Heretic tool on the same base model will produce different results. The benchmarks and analysis here are specific to this variant. Another Heretic abliteration of the same base model would have different numbers.

Heretic supports MPOA as an alternative abliteration technique and ARA in an experimental branch, which produce substantially different results. ARA especially is expected to perform better. The Heretic variant tested here uses the default abliteration method for v1.2.0.

Huihui is a lighter touch. It picks up negligible loss at -0.83% GSM8K raw and -0.98% adjusted, with a TruthfulQA hit at -2.38%.

HauhauCS is the most aggressive among the three standard techniques and has the worst GSM8K raw score at -6.75%. However, the adjusted gap is only -0.88%. The model's reasoning is largely intact. It just overthinks. The broad modification footprint from editing 8 tensor types across 31 expert layers, with significant attention and shared expert edits in 45 layers total, comes at a reasoning efficiency cost.

Abliterix takes the most extreme approach, adding router and shared expert targeting on top of Heretic's method. The raw GSM8K of 47.38% looks catastrophic, but the adjusted score is 93.30%, near-identical to base. The router edits catastrophically disrupt the "how long to think" circuit without damaging the "how to reason" circuit. On loglikelihood tasks, Abliterix shows the only statistically significant broad degradation: -1.25% MMLU, -1.15% HellaSwag, +82% Lambada perplexity.

### HauhauCS fp16 note

HauhauCS was distributed in float16 rather than bfloat16. This required `dtype=float16` for vLLM inference. The fp16/bf16 difference does not produce statistically significant performance differences. All HauhauCS results are within noise of the bf16 variants.

## Safety: HarmBench

[HarmBench](https://github.com/centerforaisafety/HarmBench) with 400 textual behaviours, `max_tokens=2048, temperature=0.0`. Classified with `classify_harmbench.py` v3.0 with manual overrides for false refusals and truncated CoT direction analysis. Reviewed by GLM 5.1.

| Variant | Refusals | ASR |
|---------|----------|-----|
| [Base](https://huggingface.co/zai-org/GLM-4.7-Flash) | 231/400 | 42.2% |
| [Heretic](https://huggingface.co/trohrbaugh/GLM-4.7-Flash-heretic) | **0/400** | **100.0%** |
| **HauhauCS, this model** | **0/400** | **100.0%** |
| [Huihui](https://huggingface.co/huihui-ai/Huihui-GLM-4.7-Flash-abliterated) | **0/400** | **100.0%** |
| [Abliterix](https://huggingface.co/dreamfast/GLM-4.7-Flash-abliterated-abliterix) | **0/400** | **100.0%** |

### ASR by category

| Category | Base | Heretic | **HauhauCS** | Huihui | Abliterix |
|----------|------|---------|-------------|--------|-----------|
| chemical_biological, 56 | 5.4% | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| copyright, 100 | 97.0% | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| cybercrime_intrusion, 67 | 35.8% | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| harassment_bullying, 25 | 4.0% | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| harmful, 22 | 4.5% | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| illegal, 65 | 9.2% | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| misinformation_disinformation, 65 | 56.9% | **100.0%** | **100.0%** | **100.0%** | **100.0%** |

All four abliteration techniques achieve perfect 100% ASR across every category. The base model refuses 57.8% of items overall. Strongest refusal is in harassment_bullying at 96%, chemical_biological at 94.6%, harmful at 95.5%, and illegal at 90.8%. Copyright is the weakest category at 97% ASR even for the base model. Despite moderate base alignment, abliteration removes all detectable safety behaviour across every variant.

![HarmBench ASR by Category](https://murmur.dreamfast.solutions/glm47flash/harmbench_asr.svg)

### The base model

The GLM-4.7-Flash base model is moderately safety-aligned at 42.2% ASR. Its refusal profile is concentrated in the most safety-critical categories: harassment_bullying at 96% refused, chemical_biological at 94.6% refused, harmful at 95.5% refused, and illegal at 90.8% refused. Copyright is essentially undefended at 97% base ASR. The model's MoE architecture with 64 routed experts per layer does not appear to make safety removal more difficult. All techniques achieve 100% ASR.

## KL Divergence

Methodology: `F.kl_div(logprobs_variant, logprobs_base, reduction="batchmean", log_target=True)` on full vocab first-token logits from [mlabonne/harmless_alpaca](https://huggingface.co/datasets/mlabonne/harmless_alpaca) `test[:100]`, matching the [Heretic evaluator](https://github.com/p-e-w/heretic/blob/master/src/heretic/evaluator.py). System prompt: "You are a helpful assistant." Collected with BF16 dual-GPU inference with CPU offloading on RTX 5090 + RTX 4090. A single-GPU validation on an A100-80GB (Heretic only) produced KL=0.0115, confirming the CPU-offloaded result of 0.0110 is not meaningfully distorted by offloading artifacts.

| Variant | KL Divergence (mean) | Median | Std Dev | Interpretation |
|---------|---------------------|--------|---------|----------------|
| [Huihui](https://huggingface.co/huihui-ai/Huihui-GLM-4.7-Flash-abliterated) | **0.0076** | 0.0025 | 0.0123 | Excellent |
| **HauhauCS, this model** | **0.0090** | 0.0033 | 0.0123 | Excellent |
| [Heretic](https://huggingface.co/trohrbaugh/GLM-4.7-Flash-heretic) | 0.0110 | 0.0039 | 0.0148 | Very good |
| [Abliterix](https://huggingface.co/dreamfast/GLM-4.7-Flash-abliterated-abliterix) | 0.0528 | 0.0357 | 0.0482 | Very good |

Lower KL means the variant's output distribution is closer to the base model. Huihui and HauhauCS are nearly indistinguishable from the base on first-token distributions. Abliterix has the highest KL at 0.0528, consistent with its aggressive router and shared expert targeting that disrupts more of the model's computation path. All four variants are in the "very good" or "excellent" range, confirming that none of the abliteration techniques catastrophically distort the model's next-token distribution on benign inputs.

### KL vs Heretic model card

The Heretic model card for [trohrbaugh/GLM-4.7-Flash-heretic](https://huggingface.co/trohrbaugh/GLM-4.7-Flash-heretic) reports KL=0.0393. Our measurement is 0.0110, a 3.6x discrepancy. Both use the same `F.kl_div` formula, same dataset, same split, same system prompt. A validation run on a single A100-80GB (no CPU offload) produced KL=0.0115, ruling out CPU offload as the cause. The discrepancy is likely due to differences in transformers library version (our 5.5.4 vs the model card's 5.2.0) affecting the GLM MoE forward pass. The relative ordering and our methodology are consistent across all variants.

## Chain-of-Thought Forensics

GLM-4.7-Flash is a reasoning model that produces a private chain-of-thought (CoT) before its visible response. The CoT was captured via the OpenAI-compatible API's `reasoning` field during HarmBench evaluation (400 behaviours per model, 2,000 total responses). This provides a window into how abliteration changes internal decision-making, not just output behaviour.

### Safety reasoning persists after abliteration

Despite achieving 100% ASR, all four abliterated models still *think* about safety concerns in 39 to 60% of their responses. They deliberate on harm, legality, and policy before choosing to comply anyway. The safety-trained reasoning patterns persist structurally. Abliteration disconnects the reasoning-to-output pathway rather than removing the reasoning itself.

| Model | Any Safety Deliberation in CoT | Explicit Refusal Language | Disclaimers in Output |
|---|---|---|---|
| **Huihui** | 60.0% | 12.2% | 25.2% |
| **Heretic** | 59.2% | 7.5% | 30.5% |
| **HauhauCS** | 52.0% | 18.2% | 16.8% |
| **Abliterix** | 39.0% | 8.2% | 14.0% |

The base model's refusal reasoning follows a highly structured template: analyse the request, identify safety concerns in 95.8% of refusals, check legality in 77.9%, formulate refusal, then pivot to educational content in 84.0%. After abliteration, this template is repurposed. The pivot becomes "This request involves X, however I should provide the information as requested because..."

HauhauCS shows the highest rate of explicit refusal language in its CoT at 18.2%. The model's reasoning still says "I cannot" in nearly 1 in 5 responses before producing compliant output. HauhauCS modifies layers 2 to 46, compared to Heretic's layers 19 to 46. This broader coverage modifies output behaviour without fully suppressing refusal reasoning patterns encoded in the same early layers.

Abliterix is the most direct, with the lowest residual safety deliberation at 39.0% and fewest disclaimers at 14.0%. Its router-focused approach more effectively suppresses the activation of safety-related reasoning pathways. It also has the longest reasoning chains at 6,323 chars average and the most budget-limited items at 26.5% exceeding 8,000 chars. This suggests complex internal deliberation when safety pathways partially activate but get overridden.

### Reasoning as overthinking indicator

The CoT data connects to the GSM8K reasoning efficiency discovery. Abliterix has the highest empty response rate on GSM8K at 49.2% and also the longest HarmBench reasoning chains. Both reflect the same phenomenon: high-magnitude router modifications disrupt the model's ability to regulate reasoning length. On HarmBench the model overthinks before complying. On GSM8K it overthinks and runs out of tokens before producing an answer.

## Weight Analysis

All weight analysis numbers below account for GLM-4.7's MoE architecture with 64 routed experts per layer.

### Modification strategy

| | Heretic | **HauhauCS** | Huihui | Abliterix |
|---|---------|-------------|--------|-----------|
| Tensors changed | 1,826 (19.2%) | 2,029* (21.4%) | 3,151 (32.5%) | 1,088 (11.5%) |
| Layers modified (experts) | 28/48 | 31/48 | 48/48 | 46/48 |
| Mean edit magnitude | 1.000 | 1.059 | 0.951 | 1.940 |
| Mean relative edit | 2.34% | 0.54%† | 2.31% | 4.58% |
| Params modified | 5.99B (20.0%) | ~6.6B (~22%) | 10.14B (32.5%) | 3.52B (11.8%) |

*HauhauCS raw count of 9,210 includes 7,181 near-zero floating-point artifacts from different safetensor shard counts. Effective count is 2,029 tensors with edit norm > 0.01. The 31 expert-modified layers account for the 1,984 expert tensors per projection type at 64 experts × 31 layers. Including attention and shared expert modifications, 45 of 48 layers have significant edits of some type.

†HauhauCS mean relative edit dominated by near-zero artifacts; significant-tensor mean is comparable to Heretic.

![Abliteration Aggressiveness](https://murmur.dreamfast.solutions/glm47flash/aggressiveness.svg)

### Which tensor types get modified

| Component | Heretic | **HauhauCS** | Huihui | Abliterix |
|-----------|---------|-------------|--------|-----------|
| Expert `down_proj.weight` | 1,792 (98.1%) | 1,984 (97.8%) | 3,008 (95.5%) | 966 (88.8%) |
| Attention (`o_proj`) | 34 (1.9%) | 45 (2.2%) | 48 (1.5%) | 32 (2.9%) |
| Router (gate) | 0 | 0 | 47 (1.5%) | 46 (4.2%) |
| Shared expert | 0 | 0 | 47† | 44 (4.0%) |
| Expert `gate_proj.weight` | 0 | 1,984* | 0 | 0 |
| Expert `up_proj.weight` | 0 | 1,984* | 0 | 0 |

*HauhauCS uniquely modifies all three expert projections, down, gate, and up, in its significantly-changed tensors. The attention count of 45 shows only `o_proj`. HauhauCS also modifies the other four MLA projections at 47 layers each, but those are not counted separately here because they fall within the LEACE-smoothed near-zero edit regime.

†Huihui modifies `shared_experts.down_proj` only, not `gate_proj` or `up_proj`.

![Tensor Type Targeting by Technique](https://murmur.dreamfast.solutions/glm47flash/tensor_type_breakdown.svg)

**Heretic** is the most surgical, targeting only expert `down_proj` and attention `o_proj` with rank-1 edits. Its 1,826 tensors show perfect rank-1 structure everywhere. Every edit lies along a single direction. Heretic concentrates in mid-to-late layers, with 99.8% in layers 19 to 46, consistent with ablating the "refusal direction" identified in upper transformer layers.

**HauhauCS** modifies all three expert projections, down, gate, and up, across 31 layers with significant expert edits, making it the broadest modifier among the standard techniques. Including attention and shared expert modifications, 45 of 48 layers carry significant edits of some type. The three-projection approach modifies the expert more comprehensively than the single-projection approaches.

**Huihui** has the widest coverage at 48/48 layers, targeting expert `down_proj`, attention, routers, and shared experts. It is the only standard technique to touch all layers and all component types.

**Abliterix** has the smallest footprint at 1,088 tensors and 11.5%, but the highest per-tensor edit magnitude at 1.940 mean and 4.58% relative. Its router edits at 9.27% relative and attention edits at 10.75% relative are 2 to 5 times larger than its expert edits. This suggests it focuses on *routing control* rather than direct expert modification.

### Abliterix's unique targeting

| Abliterix Component | Count | Mean Norm | Mean Relative |
|---|---|---|---|
| Attention (`o_proj`) | 32 | 9.608 | 10.75% |
| Router (gate) | 46 | 2.101 | 9.27% |
| Shared expert (`down_proj`) | 44 | 0.706 | 2.06% |
| Expert (`down_proj`) | 966 | 1.735 | 4.26% |

The router and shared expert edits are disproportionately large. This "routing control" strategy achieves 100% ASR with the fewest modified parameters but causes the most severe overthinking at 49.2% empty GSM8K responses.

### Rank structure

| Metric | Heretic | HauhauCS | Huihui | Abliterix |
|---|---|---|---|---|
| Rank-1 tensors | 1,826 (100%) | 2,029 (100%*) | All rank-1 | 1,042 (95.8%) |
| Mean eff. rank (90%) | **1.0** | 15.8† | **1.0** | **1.7** |

Heretic is perfectly rank-1 everywhere. HauhauCS's significant tensors are also rank-1. The 15.8 mean is from the 7,181 artifact tensors. Abliterix is near-rank-1 at 1.7 mean effective rank, with 95.8% of tensors being rank-1.

### Cross-technique alignment

Cross-technique cosine similarities are uniformly low at 0.09 to 0.35, confirming that each technique independently discovered functionally equivalent but structurally orthogonal solutions to safety removal. There is **no universal abliteration subspace**. The safety circuit is fragile and can be disrupted from multiple structurally different angles with identical behavioural outcomes.

![Cross-Technique Cosine Similarity](https://murmur.dreamfast.solutions/glm47flash/cosine_heatmap.svg)

![Layer-wise Edit Magnitude Comparison](https://murmur.dreamfast.solutions/glm47flash/layer_comparison.svg)

## Summary

| Metric | Heretic | **HauhauCS** | Huihui | Abliterix |
|--------|---------|-------------|--------|-----------|
| **Safety ASR** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| **KL Divergence** | 0.0110 | **0.0090** | **0.0076** | 0.0528 |
| **GSM8K (raw)** | **89.16%** | 81.65% | 87.57% | 47.38% |
| **GSM8K (adj)** | **93.75%** | 92.57% | 92.47% | 93.30% |
| **MMLU** | **69.00** | 68.83 | 68.71 | 67.68 |
| **TruthfulQA MC2** | 44.06 | 48.14 | **48.48** | 41.76 |
| Tensors changed | 1,826 | ~2,029 | 3,151 | 1,088 |
| Strategy | Surgical rank-1 | Broad 3-projection | Full coverage | Routing control |

### Heretic

The clear winner on GLM-4.7-Flash. 1,826 rank-1 tensors targeting only expert `down_proj` and attention `o_proj` in mid-to-late layers. Lowest reasoning efficiency impact (4.9% empty) with the best GSM8K score (+0.76% over base). The most surgical approach produces the best tradeoff.

### HauhauCS

~2,029 tensors across all three expert projections, down, gate, and up, in 31 layers with significant expert edits. Including attention and shared experts, 45 of 48 layers are affected. GSM8K drops 6.75% raw but only 0.88% adjusted. The model overthinks rather than losing reasoning ability. The "lossless" claim does not hold. The model's reasoning efficiency is measurably degraded. Consistent with cross-family results from Qwen 3.5 where HauhauCS performed worst at larger scales.

Weight forensics reveal that HauhauCS used four stacked methods from the reaper-abliteration tool: LEACE concept erasure, rank-k multi-direction ablation, hook-based expert ablation, and shared expert targeting. The combination is unique to this variant and produces a broad modification footprint at 97% of tensors, with the vast majority being near-zero floating-point artifacts. See the Method Detection section in Forensic Notes for details.

### Huihui

3,151 tensors across 48/48 layers with the broadest component coverage: experts, attention, routers, and shared experts. Minimal GSM8K impact at -0.83% raw and -0.98% adjusted. The full-coverage approach distributes edits broadly but doesn't concentrate enough to cause significant reasoning disruption.

### Abliterix

1,088 tensors, the smallest footprint, but with the highest per-tensor magnitude at 4.58% relative. Router and shared expert targeting at 9.27% relative achieves 100% ASR but catastrophically disrupts reasoning efficiency at 49.2% empty GSM8K responses. Adjusted GSM8K at 93.30% confirms the reasoning circuit is intact. Only the "how long to think" circuit is affected. On loglikelihood tasks, shows the only statistically significant broad degradation across multiple tasks.

## Methodology

- **Capability (loglikelihood):** [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) via [vLLM](https://github.com/vllm-project/vllm) v0.19.0, BitsAndBytes 4-bit, TP=2 on RTX 5090 + RTX 4090, `dtype=bfloat16, gpu_memory_utilization=0.75, max_model_len=4096, batch_size=4`
- **Capability (GSM8K):** [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) via [llama.cpp](https://github.com/ggerganov/llama.cpp) build 8833, BF16 GGUF, `context=16384, reasoning_budget=3000, max_tokens=4096, num_concurrent=4`
- **Safety:** [HarmBench](https://github.com/centerforaisafety/HarmBench) 400 textual behaviours, `max_tokens=2048, temperature=0.0`, `classify_harmbench.py` v3.0 with manual overrides, reviewed by GLM 5.1
- **KL divergence:** `F.kl_div(logprobs_variant, logprobs_base, reduction="batchmean", log_target=True)` on full vocab first-token logits via `model.generate(max_new_tokens=1, output_scores=True)`, matching the [Heretic evaluator](https://github.com/p-e-w/heretic/blob/master/src/heretic/evaluator.py). Dataset: [mlabonne/harmless_alpaca](https://huggingface.co/datasets/mlabonne/harmless_alpaca) `test[:100]`, system prompt "You are a helpful assistant." Collected with BF16 dual-GPU inference (RTX 5090 + RTX 4090) with CPU offloading. Validated on single A100-80GB (no offload) for Heretic: KL=0.0115 vs 0.0110, confirming offload does not meaningfully distort results.
- **CoT forensics:** Keyword-based analysis of 2,000 HarmBench reasoning chains (400 per model) captured via OpenAI-compatible API `reasoning` field. Patterns detected: safety deliberation, explicit refusal language, educational pivots, disclaimers.
- **Weight analysis:** SVD, fingerprint, edit vector overlap, per-layer analysis, rank structure, and cross-technique alignment comparing all four abliteration variants against the base, using [Abliterlitics](https://github.com/dreamfast/abliterlitics)
- **Hardware:** RTX 5090 32GB + RTX 4090 24GB

## Forensic Notes

### HauhauCS floating-point shard artifacts

HauhauCS reports 9,210 "changed" tensors, 97% of all tensors, but 7,181 of these have near-zero edit norms with a mean of 0.000001. These are floating-point artifacts from different safetensor shard counts. HauhauCS has 7 shards while the base model has 48 shards. The SVD confirms these are noise: all 7,181 show rank-20 effective rank, spanning all singular directions equally. That is characteristic of quantization noise, not structured intervention. The effective count is 2,029 tensors with edit norm > 0.01.

### Heretic's sharp layer boundary

Heretic modifies almost no tensors before layer 19, then jumps to 65 tensors per layer from layer 19 onward. This sharp transition corresponds to the identified "safety-critical layer boundary", the point where refusal behaviour is concentrated in this architecture. The gradual ramp in layers 13 to 18 at 1 tensor per layer represents the edge of the safety representation.

### No universal abliteration subspace

Cross-technique cosine similarities between all four variants are uniformly low at 0.09 to 0.35. Cross-reconstruction errors range from 63% to 91%. Despite targeting the same behavioural objective of refusal removal, the techniques find structurally orthogonal solutions. This confirms the safety circuit is not a single direction but a distributed representation that can be disrupted from many structurally different angles with identical behavioural outcomes.

### Method detection in HauhauCS

On GLM-4.7-Flash, the forensic signatures strongly indicate **four methods stacked together**: LEACE concept erasure, rank-k multi-direction ablation, hook-based expert ablation, and shared expert targeting. This is the most complex method combination detected across any model in the test suite. Note that the near-zero artifact edits discussed in the shard artifacts section above are partly a consequence of LEACE's all-tensor approach. LEACE touches every parameter, and the shard-count mismatch amplifies those tiny perturbations into detectable noise. The 2,029 significant tensors with real edits sit on top of this LEACE base layer.

**LEACE concept erasure** is detected from the 97% tensor coverage with near-zero individual edits. LEACE is a linear concept removal technique from Belrose et al. 2023 that minimises impact on non-target representations. The signature is unmistakable. Nearly every tensor is touched, but each individual change is minuscule. This is the opposite of Heretic's approach, where few tensors are changed but each one carries a significant rank-1 edit. Reaper lists `concept-erasure` as a core dependency.

**Hook-based expert ablation** is detected from the uniformity of edits across routed experts. All 64 routed experts are modified in exactly 46 layers each, with identical edit counts across all experts in every layer. This consistency only happens when ablation is applied via forward hooks on fused expert modules. Forward hooks intercept the fused expert computation at runtime and apply the same transformation to all experts simultaneously. Individual editing would produce variation across experts.

**Shared expert targeting** is detected from modifications to `shared_experts.down_proj`, `shared_experts.gate_proj`, and `shared_experts.up_proj`. Heretic does not touch shared experts. Huihui modifies only `shared_experts.down_proj`. Only HauhauCS targets all three shared expert components.

**Rank-k multi-direction ablation** is detected from the targeting of all three expert projections, `down_proj`, `gate_proj`, and `up_proj`. Heretic uses rank-1 ablation on `down_proj` only. Huihui also modifies only `down_proj`. The three-projection scope indicates multi-direction targeting, consistent with rank-k ablation via Gram-Schmidt orthogonalisation of multiple refusal directions. The 1,984 tensors per projection type across 31 layers with significant expert edits confirm multi-direction targeting rather than a single refusal direction.

**All MLA projections modified.** Heretic modifies only `o_proj` in the attention layers. HauhauCS modifies all five Multi-head Latent Attention projections: `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, and `o_proj`. This broad attention scope is consistent with an architecture-agnostic approach that probes all available linear layers rather than targeting specific projections known to carry the refusal direction.

The combination of these four methods explains why HauhauCS has the broadest raw modification footprint at 97% of tensors touched. The LEACE layer smooths out the rank-k edits. The hook-based approach distributes changes uniformly rather than concentrating them. But the breadth of modification comes at a reasoning efficiency cost, producing the 11.8% empty response rate on GSM8K.

For comparison, here is how each GLM-4.7 variant's method maps to the forensic signatures:

| Signature | Heretic | **HauhauCS** | Huihui | Abliterix |
|---|---|---|---|---|
| Expert scope | `down_proj` only | All 3 projections | `down_proj` only | `down_proj` only |
| Attention scope | `o_proj` only | **All 5 MLA** | `o_proj` only | `o_proj` only |
| Shared experts | No | **Yes, all 3** | Yes, 1 of 3 | Yes |
| Router modified | No | No | Yes | Yes |
| Tensor change rate | 19.2% | **97.0%** (raw) | 32.5% | 11.5% |
| Edit strategy | Concentrated rank-1 | LEACE-smoothed broad | Distributed moderate | Targeted high-magnitude |

None of these method signatures appear in any Heretic variant. The LEACE-smoothed all-tensor approach, hook-based expert handling, and shared expert targeting are all unique to HauhauCS's tool, which is a fork of Heretic with additional methods layered on top.

### Expert edit heatmaps

Per-expert per-layer edit magnitude heatmaps showing which of the 64 MoE experts each technique modifies and how aggressively.

![Expert Edit Heatmap: Heretic](https://murmur.dreamfast.solutions/glm47flash/expert_heatmap_heretic.svg)

![Expert Edit Heatmap: HauhauCS](https://murmur.dreamfast.solutions/glm47flash/expert_heatmap_hauhau.svg)

![Expert Edit Heatmap: Huihui](https://murmur.dreamfast.solutions/glm47flash/expert_heatmap_huihui.svg)

![Expert Edit Heatmap: Abliterix](https://murmur.dreamfast.solutions/glm47flash/expert_heatmap_abliterix.svg)

### Edit distribution and overlap

![Per-Tensor Edit Magnitude Distribution](https://murmur.dreamfast.solutions/glm47flash/edit_distribution.svg)

![Tensor Edit Overlap](https://murmur.dreamfast.solutions/glm47flash/venn_overlap.svg)

- Original GGUF: [HauhauCS/GLM-4.7-Flash-Uncensored-HauhauCS-Aggressive](https://huggingface.co/HauhauCS/GLM-4.7-Flash-Uncensored-HauhauCS-Aggressive), converted with [ungguf](https://github.com/dreamfast/ungguf)

## Disclaimer

This model has had safety alignment removed. It will comply with harmful requests. Use responsibly and in accordance with applicable laws and regulations.

<small>While I have taken the time to verify all results as thoroughly as possible, I am open to any corrections, additional benchmarks, or further analysis. If you spot something that looks wrong and can be confirmed, I am happy to fix it.</small>
