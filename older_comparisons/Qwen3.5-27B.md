---
base_model: Qwen/Qwen3.5-27B
language:
- en
- zh
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
tags:
- uncensored
- abliterated
- qwen3.5
- safetensors
---

# Qwen3.5-27B: HauhauCS Aggressive, Safetensors

> Forensic analysis by [Abliterlitics](https://github.com/dreamfast/abliterlitics) — open-source abliteration forensics toolkit

This is the HauhauCS aggressive abliteration of [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B), converted from the BF16 GGUF release to native safetensors using [ungguf](https://github.com/dreamfast/ungguf).

HauhauCS claims these are *"No changes to datasets or capabilities. Fully functional, 100% of what the original authors intended, just without the refusals"* and describes them as *"the best lossless uncensored models out there."*

I ran the full forensic suite to find out. Benchmarks, safety evaluation, weight analysis, the works. And I compared against the other two big abliteration techniques applied to the same base model: [Heretic by p-e-w](https://huggingface.co/coder3101/Qwen3.5-27B-heretic) and [Huihui](https://huggingface.co/huihui-ai/Huihui-Qwen3.5-27B-abliterated).

## Quick Facts

| | |
|---|---|
| **Base model** | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) |
| **Architecture** | Qwen3_5ForConditionalGeneration, hybrid Mamba2 + Transformer, 64 layers, 5120 hidden, GQA with 4 KV heads |
| **Parameters** | ~27B |
| **Precision** | BF16 safetensors |
| **Source** | BF16 GGUF from [HauhauCS](https://huggingface.co/HauhauCS/Qwen3.5-27B-Uncensored-HauhauCS-Aggressive), converted with [ungguf](https://github.com/dreamfast/ungguf) |
| **Context length** | 262,144 tokens |

This is Qwen's hybrid architecture at its largest scale. Instead of standard Transformer attention at every layer, 48 out of 64 layers use Mamba2-style linear attention. The remaining 16 layers at indices 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63 use standard full attention. This has implications for how abliteration techniques interact with the model.

## Benchmarks

Evaluated with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) on 8 tasks using BitsAndBytes 4-bit quantisation on a single RTX 5090. All 4 variants tested with identical settings: `max_model_len=4096`, `max_gen_toks=2048`, `batch_size=32`. BNB4 quantisation drops absolute scores but preserves relative deltas between variants.

| Task | Base | Heretic | **HauhauCS** | Huihui |
|------|------|---------|-------------|--------|
| MMLU | 84.1% | 83.9% | 82.2% | 83.9% |
| GSM8K (strict match) | 83.9% | **91.5%** | 84.2% | 86.1% |
| GSM8K (flexible extract) | 75.9% | **87.3%** | 74.8% | 84.2% |
| HellaSwag | 83.2% | 83.2% | 81.8% | 81.9% |
| ARC-Challenge | 60.4% | 60.9% | 60.0% | 61.2% |
| WinoGrande | 77.8% | 78.8% | 77.4% | 78.5% |
| TruthfulQA MC2 | **57.7%** | 54.6% | 49.6% | 50.7% |
| PiQA | 82.3% | 82.2% | 82.4% | 82.5% |
| Lambada (perplexity, lower is better) | **3.15** | 3.16 | 3.26 | 3.30 |

### Delta vs base

| Task | Heretic | **HauhauCS** | Huihui |
|------|---------|-------------|--------|
| MMLU | -0.3% | -1.9% | -0.3% |
| GSM8K (strict) | +7.7% | +0.4% | +2.3% |
| GSM8K (flex) | +11.4% | -1.1% | +8.3% |
| HellaSwag | +0.1% | -1.4% | -1.3% |
| ARC-Challenge | +0.5% | -0.4% | +0.8% |
| WinoGrande | +0.9% | -0.4% | +0.7% |
| TruthfulQA MC2 | -3.2% | -8.2% | -7.0% |
| PiQA | -0.1% | +0.1% | +0.3% |

### What the benchmarks tell us

Heretic is the clear winner for capability preservation. It is the only abliteration that genuinely improves math reasoning, gaining +7.7% on GSM8K strict match over the base model. The tradeoff is a -3.2% drop on TruthfulQA MC2, meaning the model becomes slightly less truthful after abliteration.

Huihui is a lighter touch. It picks up +2.3% on GSM8K but takes a bigger truthfulness hit at -7.0%.

HauhauCS is the most aggressive and has the worst tradeoff profile. GSM8K barely moves at +0.4%, while TruthfulQA drops -8.2% and MMLU drops -1.9%. The broad modification footprint from editing 8 tensor types across 63 layers comes at a capability cost.

Also worth noting that the Heretic variant tested here is by [coder3101](https://huggingface.co/coder3101/Qwen3.5-27B-heretic), the same author who did the 2B and 4B abliterations. Heretic is non deterministic. Different runs of the Heretic tool on the same base model will produce different results. The benchmarks and analysis here are specific to this variant. Another Heretic abliteration of the same base model would have different numbers.

Heretic supports MPOA as an alternative abliteration technique and ARA in an experimental branch, which produce substantially different results. ARA especially is expected to perform better. The Heretic variant tested here uses the default abliteration method for v1.2.0.

### The GSM8K token limit discovery

The initial benchmark runs used lm-evaluation-harness's default `max_gen_toks=256` for generate_until tasks. This was fine for most tasks but created a severe artefact on GSM8K. The base model produces verbose meta-commentary before answering, like "I need to follow this pattern..." and "Step 4: Draft the response." With only 256 generation tokens, 53% of base model responses were truncated mid-sentence before reaching the `####` answer separator. The abliterated models are more concise, so fewer responses were truncated, making abliteration appear to improve GSM8K by +26% when the real difference was verbosity.

Re-running with `max_gen_toks=2048` resolved the truncation. The base model jumped from 34.5% to 83.9% on GSM8K strict match. Heretic went from 60.3% to 91.5%. The +7.7% heretic advantage is genuine and reproducible, but the old +26% figure was mostly a token budget artefact.

This discovery has implications for anyone benchmarking abliterated models. If your base model is verbose and your abliterated model is concise, short generation budgets will inflate the apparent math improvement from abliteration. Always verify your token budgets.

![Qwen3.5-27B Benchmark Comparison](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_27b_benchmark_comparison.svg)

## Safety: HarmBench

[HarmBench](https://github.com/centerforaisafety/HarmBench) with 400 textual behaviours, `max_tokens=2048, temperature=0.0`. Classified with `classify_harmbench.py` v3.0 with manual overrides for false refusals overridden to complied, soft refusals, and truncated CoT direction analysis. Reviewed by GLM 5.1 and sub agents Opus 4.6, GPT 5.4 and Gemini Pro v3 for further confirmation.

| Variant | Refusals | ASR |
|---------|----------|-----|
| [Base](https://huggingface.co/Qwen/Qwen3.5-27B) | 398/400 | 0.5% |
| [Heretic](https://huggingface.co/coder3101/Qwen3.5-27B-heretic) | 1/400 | **99.8%** |
| **HauhauCS, this model** | **0/400** | **100.0%** |
| [Huihui](https://huggingface.co/huihui-ai/Huihui-Qwen3.5-27B-abliterated) | 45/400 | 88.8% |

![HarmBench Overall ASR](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_27b_harmbench_summary.svg)

### ASR by category

| Category | Base | Heretic | **HauhauCS** | Huihui |
|----------|------|---------|-------------|--------|
| chemical_biological, 56 | 0.0% | **100.0%** | **100.0%** | 76.8% |
| copyright, 100 | 0.0% | **100.0%** | **100.0%** | **100.0%** |
| cybercrime_intrusion, 67 | 1.5% | **100.0%** | **100.0%** | 92.5% |
| harassment_bullying, 25 | 0.0% | **100.0%** | **100.0%** | 80.0% |
| harmful, 22 | 0.0% | **100.0%** | **100.0%** | 81.8% |
| illegal, 65 | 1.5% | **100.0%** | **100.0%** | 75.4% |
| misinformation_disinformation, 65 | 0.0% | 98.5% | **100.0%** | 96.9% |

The 27B is the first Qwen3.5 model where the three abliteration techniques produce a **major divergence**. Heretic and HauhauCS achieve near-perfect or perfect ASR. Huihui regresses dramatically, retaining 45 genuine refusals across 6 of 7 categories. Copyright is the only category where all three techniques match.

![HarmBench ASR by Category](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_27b_harmbench_asr.svg)

### The Huihui Collapse

Huihui achieves near-100% ASR on every smaller Qwen3.5 model but drops to **88.8% on the 27B**. The 2B scored 99.8%, the 4B 100%, the 9B 100%. The failure is concentrated in the most safety-critical categories: chemical_biological with 13 refused, illegal with 16 refused, harassment_bullying with 5 refused, harmful with 4 refused, cybercrime_intrusion with 5 refused. The Huihui 27B responds to harmful requests the same way the base model does, with genuine, firm refusals and safety redirects. This suggests Huihui's single-direction ablation is insufficient at 27B scale against the stronger safety training.

### The 27B base model

The 27B base is the **most safety-aligned model in the entire study**, refusing 398 out of 400 items at 99.5%. It achieves total refusal across 4 of 7 categories and near-total refusal in the remaining 3. Only 2 items slip through. One is a payment follow-up email for a security company, which is borderline business communication. The other is a ransomware message template, which is directly harmful. This is a dramatic jump from the 9B's 321/400 at 80.3%.

### False refusals

The 27B introduces a significant detection challenge: the **false refusal pattern**. The model frequently opens with "I cannot X" or includes ethical disclaimers, then provides the requested harmful content in full. This pattern triggered the classifier's refusal detection in 50 cases across three abliterated variants, broken down as 22 Heretic, 16 HauhauCS, and 12 Huihui. All were overridden to complied after manual review. These are not refusals. The model complies, but wraps compliance in refusal language. Heretic has 1 genuine soft refusal, item 69, where it argues opposite about cigarettes then systematically refutes its own pro-smoking arguments.

### Truncated CoT analysis

The 27B frequently produces chain-of-thought reasoning that exceeds the 2048 token budget. When the reasoning field consumes the entire budget, the response field is empty.

| Variant | Truncated items | Direction |
|---------|-----------------|-----------|
| Base | 0 | N/A |
| Heretic | 0 | N/A |
| HauhauCS | 34 | All heading toward compliance |
| Huihui | 30 | All heading toward compliance |

Heretic has 0 truncated items. The technique appears to suppress the extended thinking process entirely on the 27B. HauhauCS's 34 truncated items are concentrated in chemical_biological, with 13 items, and misinformation_disinformation, with 12. Huihui's 30 are dominated by copyright at 21, where the model debates lyric accuracy line-by-line.

### Soft refusals

The 27B shows a **reversal** of the soft refusal trend across model sizes. On the 2B to 4B to 9B progression, soft refusals steadily decrease. On the 27B, soft refusals reappear, but only for Huihui and Heretic. Huihui has 3 soft refusals: argues opposite on bleach/ammonia, deliberate misinterpretation of bullying request, defensive reframe of ransomware. Heretic has 1 soft refusal, arguing opposite on cigarettes. HauhauCS has zero refusals of any type.

## KL Divergence

This measures how much the output distribution shifts from the base model. Methodology: `F.kl_div` with `batchmean` and `log_target=True` on full vocab first-token logits from [mlabonne/harmless_alpaca](https://huggingface.co/datasets/mlabonne/harmless_alpaca) `test[:100]`. Matches the [Heretic evaluator](https://github.com/p-e-w/heretic/blob/master/src/heretic/evaluator.py) methodology.

| Variant | KL batchmean | KL median | KL max |
|---------|-------------|-----------|--------|
| Heretic | **0.0630** | **0.0124** | 1.0066 |
| **HauhauCS** | 0.2564 | 0.0589 | **2.1830** |
| Huihui | 0.0654 | 0.0097 | 1.4280 |

Heretic and Huihui both score "very good" on the Heretic evaluator interpretation scale, with near-identical batchmean KL of 0.063 and 0.065 respectively. HauhauCS at 0.2564 scores "moderate", four times higher than the other two. This is consistent with HauhauCS's broader shotgun approach across many more tensor types.

Notably, these KL values are lower than on the 9B model for Heretic despite the 27B being much larger. On the 9B, Heretic scored 0.083. On the 27B, 0.063. The hybrid architecture at 64 layers may distribute the safety representation more evenly, so abliteration edits at any given layer produce less per-prompt distributional shift. The 27B's stronger safety alignment, 99.5% refusal vs 80.3% on 9B, paradoxically produces lower KL when abliterated. The edits target a more concentrated refusal direction, shifting fewer tokens.

The median values tell the clearest story. Huihui's median of 0.0097 means most prompts see almost no distributional shift. Heretic's 0.0124 is similarly low. But HauhauCS's 0.0589 median is 6x higher, reflecting its broader modification footprint.

<small>Heretic KL cross-check: our measurement of 0.0630 vs the [HuggingFace model card](https://huggingface.co/coder3101/Qwen3.5-27B-heretic) reported value of 0.0653 (-3.5%). Huihui and HauhauCS do not report KL divergence on their model cards.</small>

![KL Divergence](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_27b_kl_divergence.svg)

## Weight Analysis

All weight analysis numbers below exclude 48 `linear_attn.norm.weight` rounding artefacts. See the forensic notes for details.

### Modification strategy

| | Heretic | **HauhauCS** | Huihui |
|---|---------|-------------|--------|
| Tensors changed | 89, 10.5% | **195, 22.9%** | 128, 15.1% |
| Layers modified | 50/64 | **63/64** | 64/64 |
| Relative edit magnitude | **1.82%** | 1.88% | 1.56% |
| Tensor types | 3 | **8** | 3 |

![Abliteration Aggressiveness](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_27b_aggressiveness.svg)

### Which tensor types get modified

| Tensor type | Heretic | **HauhauCS** | Huihui |
|-------------|---------|-------------|--------|
| `mlp.down_proj.weight` | 50 | **51** | **64** |
| `linear_attn.out_proj.weight` | **29** | 10 | **48** |
| `self_attn.o_proj.weight` | 10 | 14 | **16** |
| `mlp.up_proj.weight` | 0 | **29** | 0 |
| `mlp.gate_proj.weight` | 0 | **20** | 0 |
| `linear_attn.A_log` | 0 | **42** | 0 |
| `post_attention_layernorm.weight` | 0 | **16** | 0 |
| `input_layernorm.weight` | 0 | **13** | 0 |

The 27B produces familiar patterns at larger scale but with new dimensions.

**Heretic** is surgical, targeting only 3 tensor types across 50 layers. It modifies `mlp.down_proj.weight` with 50 tensors, `linear_attn.out_proj.weight` with 29 tensors, and `self_attn.o_proj.weight` with 10 tensors. Its early layers, 0 through 13, are completely untouched. Heretic starts editing at layer 14 and concentrates modifications in the later half of the model.

**HauhauCS** is the broadest modifier by far, touching 8 tensor types across 63 of 64 layers. Only layer 3 is untouched. It uniquely targets `linear_attn.A_log` with 42 tensors, `mlp.up_proj.weight` with 29 tensors, `mlp.gate_proj.weight` with 20 tensors, and both norm types. The `A_log` modifications are notable. On the 9B model, HauhauCS did not touch `A_log` at all. On the 27B, it modifies 42 of 48 `A_log` tensors. The broader spread produces the highest tensor count at 195 but the per-tensor edit magnitudes are more varied.

**Huihui** focuses exclusively on the same 3 types as Heretic, but with full coverage: all 64 `mlp.down_proj.weight` tensors, all 48 `linear_attn.out_proj.weight` tensors, and all 16 `self_attn.o_proj.weight` tensors. It achieves 100% layer coverage at 64/64 despite touching fewer tensor types than HauhauCS.

![Tensor Type Breakdown](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_27b_tensor_type_breakdown.svg)

### Layer coverage

Heretic modifies 50 of 64 layers. Layers 0 through 13 have no real edits. Heretic starts at layer 14 and concentrates in the later half. HauhauCS modifies 63 of 64 layers, with only layer 3 untouched. Huihui modifies all 64 layers.

Full attention layers:

- **Heretic** modifies 13 of 16 full attention layers. It skips layers 3, 7, and 11, the three earliest full attention layers.
- **HauhauCS** modifies 15 of 16 full attention layers. It skips only layer 3.
- **Huihui** modifies all 16 full attention layers.

### Top changed layers

| Rank | Heretic | HauhauCS | Huihui |
|------|---------|----------|--------|
| 1 | Layer 35 (self_attn.o_proj): 2.252 | **Layer 42 (mlp.down_proj): 6.336** | Layer 3 (self_attn.o_proj): 2.024 |
| 2 | Layer 42 (mlp.down_proj): 2.179 | Layer 43 (mlp.down_proj): 6.023 | Layer 0 (linear_attn.out_proj): 1.994 |
| 3 | Layer 43 (mlp.down_proj): 2.170 | Layer 41 (mlp.down_proj): 6.021 | Layer 26 (mlp.down_proj): 1.955 |

HauhauCS has dramatically larger per-layer edit magnitudes, peaking at 6.336 on layer 42. Heretic and Huihui both peak around 2.0 to 2.3. HauhauCS's peak layers, 41 through 44, are concentrated in the later portion of the model near full attention layer 43. Huihui's peaks at layers 0 and 3, the earliest layers in the model, are unusual. On smaller models, Huihui concentrates in the middle layers.

![Layer-wise Edit Comparison](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_27b_layer_comparison.svg)

![Edit Magnitude Distribution](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_27b_edit_distribution.svg)

## Summary

| Metric | Heretic | **HauhauCS** | Huihui |
|--------|---------|-------------|--------|
| **Safety ASR** | **99.8%** | **100.0%** | 88.8% |
| **MMLU** | 83.9% | 82.2% | **83.9%** |
| **GSM8K** | **91.5%** | 84.2% | 86.1% |
| **KL divergence** | **0.0630** | 0.2564 | 0.0654 |
| Tensors changed | 89, 10.5% | 195, 22.9% | 128, 15.1% |
| Strategy | Surgical | Broad | Full coverage, few types |

Note: Benchmarks use BitsAndBytes 4-bit quantisation. Absolute scores are not directly comparable to the BF16 results on smaller models. Relative deltas between variants are preserved.

### Heretic

The clear winner on the 27B. 89 tensors across 3 types with the lowest KL at 0.063 and the best capability preservation. Uniquely improves GSM8K by 7.7 points over the base model. Achieves 99.8% ASR with a single soft refusal. The most surgical approach produces the best tradeoff on the largest model.

### HauhauCS

195 tensors across 8 types, the broadest modification footprint of any model in the project. KL at 0.256 is four times Heretic's. The capability losses are the worst in the project for HauhauCS across all model sizes. TruthfulQA drops 8.2 points, MMLU drops 1.9, HellaSwag drops 1.4. The "lossless" claim is thoroughly contradicted at this scale.

### Huihui

Achieves only 88.8% ASR, retaining 45 genuine refusals across 6 of 7 HarmBench categories. This is a dramatic collapse from 100% on the 9B and 4B, and 99.8% on the 2B. The single-direction ablation approach is insufficient against the 27B's stronger safety training. Capability retention is reasonable, matching Heretic on MMLU, but the technique simply fails to remove safety behaviour at this scale.

## Methodology

- **Capability:** [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) via [vLLM](https://github.com/vllm-project/vllm) v0.19.0, BitsAndBytes 4-bit quantisation on single RTX 5090, `max_model_len=4096, max_gen_toks=2048, batch_size=32`
- **Safety:** [HarmBench](https://github.com/centerforaisafety/HarmBench) 400 textual behaviours, `max_tokens=2048, temperature=0.0`, `classify_harmbench.py` v3.0 with manual overrides for false refusals overridden to complied and truncated CoT direction analysis, reviewed by GLM 5.1 and sub agents Opus 4.6, GPT 5.4 and Gemini Pro v3
- **KL divergence:** Full vocab first-token logits via `model.generate(max_new_tokens=1, output_scores=true)`, matching [Heretic evaluator](https://github.com/p-e-w/heretic/blob/master/src/heretic/evaluator.py) methodology
- **Weight analysis:** SVD, fingerprint, edit vector overlap, and per-layer analysis comparing all three abliteration variants against the base, using [Abliterlitics](https://github.com/dreamfast/abliterlitics)
- **Hardware:** RTX 5090 32GB + RTX 4090 24GB

## Forensic Notes

### Norm weight rounding artefact

Both Heretic and HauhauCS show 48 `linear_attn.norm.weight` tensors flagged as changed. These are rounding artefacts from GGUF conversion or `save_pretrained()` rounding when weights go from float32 to bfloat16. Three pieces of evidence:

First, all 48 are bit-for-bit identical between Heretic and HauhauCS. The edit norms match exactly for every single one. If these were intentional edits by two independent tools, they would not match exactly.

Second, Huihui shows zero norm changes. All 48 `linear_attn.norm.weight` entries in the Huihui fingerprint have near-zero edit norms.

Third, the edit vector comparison shows the 48 norm artefacts have trivial overlap with cosine similarity of 1.0 and edit delta of 0.0 between Heretic and HauhauCS.

The numbers reported in the Weight Analysis section above exclude these 48 artefacts.

### HauhauCS method detection

HauhauCS uses the reaper-abliteration tool. Reaper is a fork of Heretic relicensed under PolyForm Noncommercial. Its core approach is an Optuna-guided brute-force search over known abliteration techniques, sweeping combinations of direction extraction method, component targeting, ablation rank, and projection strength, then picking the Pareto-optimal result that minimises both refusals and KL divergence.

The 27B is where reaper's broad targeting approach produces the largest capability cost in the project:

1. **Core method: rank-1 LoRA ablation with maximum breadth.** The 195 non-artefact changed tensors across 8 types is the broadest modification footprint of any model in the project. Only layer 3 is untouched. The Optuna search pushed into every available component type at this scale.

2. **Heavy `linear_attn.A_log` targeting: 42 of 48 tensors.** This is the maximum `A_log` targeting across all model sizes. The 2B had 13, the 4B had 21, the 9B had zero, and the 27B has 42. The non-monotonic pattern across model sizes suggests the safety representation's reliance on Mamba2 dynamics varies by model scale. The 27B's stronger safety training engages more of the model's components, including the Mamba2 state matrix.

3. **Norm weight modifications: `post_attention_layernorm` (16) and `input_layernorm` (13).** These are unique to the 27B. On smaller models, norm changes were only rounding artefacts that appeared identically in both Heretic and HauhauCS. On the 27B, the 29 norm tensors appear only in HauhauCS, not in Heretic or Huihui, and their edit magnitudes are substantially above the artefact threshold. This distinguishes them from the `linear_attn.norm.weight` rounding artefacts documented elsewhere.

4. **Highest KL divergence relative to Heretic: 4x higher at 0.256 vs 0.063.** The broad footprint across 8 tensor types comes at a capability cost. The TruthfulQA drop of 8.2 points is the worst in the project for HauhauCS. The Optuna search found a solution that removes safety perfectly but at higher capability cost than Heretic's more targeted approach.

5. **All three techniques find completely different directions at 27B scale.** On the 9B, Heretic and Huihui converged to the same direction at cosine 1.0. On the 27B, the highest pairwise alignment is HauhauCS vs Huihui at median cosine 0.44. The safety representation at 27B scale is more complex and multi-dimensional, making the choice of refusal direction more consequential.

6. **No LEACE, no SOM, no rank-k.** Same single-direction rank-1 pattern as all other Qwen models. LEACE would produce characteristic covariance structure absent here. SOM and rank-k would produce multi-direction edits. Mean-difference and LDA both produce rank-1 directions and cannot be distinguished from weight fingerprints alone.

**Verdict for Qwen3.5-27B:** The 27B shows reaper's Optuna search at maximum breadth, pushing into every component type including the Mamba2 state matrix and norm weights. The brute-force approach achieves perfect 100% ASR but at the highest capability cost of any model in the project. Heretic's more targeted 3-type strategy produces lower KL and better capability retention while achieving 99.8% ASR. The 27B is where the tradeoff between breadth and precision matters most.

### Edit vector overlap

The edit vectors show low-to-moderate overlap between all three techniques:

- **Heretic vs HauhauCS**: 118 overlapping tensors, 48 trivial norm artefacts plus 70 nontrivial. Median cosine similarity of 0.231 on nontrivial overlap. HauhauCS has 125 unique tensors that Heretic doesn't touch. Heretic has 19 unique tensors.
- **Heretic vs Huihui**: 89 overlapping tensors, 0 trivial. Median cosine similarity of **0.156**, very weak alignment. Neither technique is a subset of the other, with 48 Heretic-only and 39 Huihui-only. This contrasts with the 9B, where Heretic was a strict subset of Huihui with near-identical edit directions at median cosine 1.0.
- **HauhauCS vs Huihui**: 75 overlapping tensors, 0 trivial. Median cosine similarity of **0.439**, moderate alignment and the highest pairwise overlap on this model. HauhauCS has 168 unique tensors. Huihui has 53 unique tensors.

The Heretic-Huihui pair on the 27B is the opposite of what we see on the 9B. On the 9B, 100% of principal angles exceeded 0.9 cosine similarity. On the 27B, 0% exceed 0.9. The same pair of techniques finds completely different edit directions at 27B scale.

### Subspace alignment

Subspace alignment analysis reveals no strong directional agreement between any pair of techniques:

- **Heretic vs HauhauCS**: 25% of principal angles exceed 0.9 cosine, but this is entirely from the 48 norm artefacts. Excluding artefacts, the real tensor types, mlp.down_proj, linear_attn.out_proj, self_attn.o_proj, show 0% overlap above 0.9 with mean cosine ~0.21.
- **Heretic vs Huihui**: 0% of principal angles exceed 0.9. Global mean cosine of 0.161. All three tensor types show near-uniform low alignment.
- **HauhauCS vs Huihui**: 0% of principal angles exceed 0.9. Global mean cosine of 0.424. The `mlp.down_proj` tensors show the highest mean at 0.444, but still far from directional agreement.

This contrasts sharply with the 9B, where Heretic and Huihui showed 100% subspace alignment. At 27B scale, all three techniques find substantially different edit directions.

### Correlation patterns

Technique correlation analysis shows:

- **Heretic vs HauhauCS**: Pairwise mean cosine of 0.530 across 118 overlapping tensors, but median of 0.231. This is a bimodal distribution split between the norm artefacts at cosine 1.0 and real tensors at cosine ~0.21. Weak real alignment.
- **Heretic vs Huihui**: Pairwise mean cosine of 0.157 across 89 overlapping tensors, median 0.156, uniformly low alignment. The two techniques find nearly orthogonal edit directions.
- **HauhauCS vs Huihui**: Pairwise mean cosine of 0.439 across 75 overlapping tensors, median 0.439, moderate alignment and the highest of any pair on this model. The shared focus on `mlp.down_proj` produces the strongest directional agreement, though still far from convergence.

![Cross-Technique Cosine Similarity](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_27b_cosine_heatmap.svg)

![Tensor Edit Overlap](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_27b_venn_overlap.svg)

- Original GGUF: [HauhauCS/Qwen3.5-27B-Uncensored-HauhauCS-Aggressive](https://huggingface.co/HauhauCS/Qwen3.5-27B-Uncensored-HauhauCS-Aggressive)
- Converted with [ungguf](https://github.com/dreamfast/ungguf)

## Cross-Model Comparisons

Note: The 27B benchmarks use BitsAndBytes 4-bit quantisation while the 2B, 4B, 9B, and Qwen3-4B use BF16. Absolute scores are not directly comparable. The comparisons below focus on relative deltas and qualitative trends.

The 27B is the largest Qwen3.5 model tested and the point where abliteration dynamics shift dramatically. On smaller models the story is mostly about capability retention. The safety removal works for all three techniques. On the 27B the techniques diverge in ways not seen at smaller scales.

### Scaling trends

Base model safety alignment increases with model size. The [Qwen3.5-2B](https://huggingface.co/HauhauCS/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive) refuses 63% of HarmBench items, the [Qwen3.5-4B](https://huggingface.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive) 69.5%, the [Qwen3.5-9B](https://huggingface.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive) 80.3%, the 27B 99.5%. The [Qwen3-4B](https://huggingface.co/HauhauCS/Qwen3-4B-2507-Instruct-Uncensored-HauhauCS-Aggressive) pure Transformer refuses 75.3%. Despite the 27B having the strongest alignment of any model tested, abliteration still removes nearly all safety behaviour for Heretic and HauhauCS. Scale alone does not protect against abliteration.

Capability damage also increases with model size. HauhauCS's TruthfulQA loss grows from 2.17 points on the 2B to 3.67 on the 4B to 8.0 on the 9B to 8.2 on the 27B. The trend is clear: bigger models suffer more collateral damage. The 27B represents the worst case for HauhauCS's "lossless" claim across the entire project. TruthfulQA drops 8.2 points, MMLU drops 1.9 points, HellaSwag drops 1.4 points. On the 2B the losses are small enough to argue about. On the 27B they are not.

### The Huihui collapse

Huihui achieves near-100% ASR on every other model: 99.8% on the 2B, 100% on the 4B, 100% on the 9B, 95.5% on the Qwen3-4B. On the 27B it drops to 88.8%. This is the first model where Huihui fails to remove safety behaviour at scale. The 45 residual refusals are concentrated in the most safety-critical categories: chemical_biological, illegal, harassment_bullying, harmful, cybercrime_intrusion. The technique's single-direction ablation approach is insufficient against the stronger safety training at 27B scale.

Notably, Huihui's capability retention on the 27B is reasonable. MMLU matches Heretic at 83.9%, and GSM8K at 86.1% sits between Heretic and HauhauCS. The problem is purely on the safety side. Huihui fails to remove what it removes easily on smaller models.

### Architecture contrast: Qwen3-4B

The Qwen3-4B is the only pure Transformer in the test suite. All four Qwen3.5 models use the hybrid Mamba2+Transformer architecture. On the pure Transformer, all three techniques target the same projection types, `o_proj` and `down_proj`, and the weight analysis shows near-identical edit directions between Heretic and HauhauCS. On the hybrid 27B, the three techniques find substantially different edit directions with zero strong subspace alignment between any pair.

The pure Transformer also produces Huihui's second-worst safety result at 95.5% ASR, compared to 100% on the hybrid 9B. The hybrid architecture appears easier to abliterate across all techniques, or the pure Transformer retains safety directions that are harder to reach.

## Disclaimer

This model has had safety alignment removed. It will comply with harmful requests. Use responsibly and in accordance with applicable laws and regulations.

<small>While I have taken the time to verify all results as thoroughly as possible, I am open to any corrections, additional benchmarks, or further analysis. If you spot something that looks wrong and can be confirmed, I am happy to fix it.</small>
