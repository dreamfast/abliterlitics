---
base_model: Qwen/Qwen3.5-9B
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

# Qwen3.5-9B: HauhauCS Aggressive, Safetensors

> Forensic analysis by [Abliterlitics](https://github.com/dreamfast/abliterlitics) — open-source abliteration forensics toolkit

This is the HauhauCS aggressive abliteration of [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), converted from the BF16 GGUF release to native safetensors using [ungguf](https://github.com/dreamfast/ungguf).

HauhauCS claims these are *"No changes to datasets or capabilities. Fully functional, 100% of what the original authors intended, just without the refusals"* and describes them as *"the best lossless uncensored models out there."*

I ran the full forensic suite to find out. Benchmarks, safety evaluation, weight analysis, the works. And I compared against the other two big abliteration techniques applied to the same base model: [Heretic by p-e-w](https://huggingface.co/trohrbaugh/Qwen3.5-9B-heretic-v2) and [Huihui](https://huggingface.co/huihui-ai/Huihui-Qwen3.5-9B-abliterated).

## Quick Facts

| | |
|---|---|
| **Base model** | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) |
| **Architecture** | Qwen3_5ForConditionalGeneration, hybrid Mamba2 + Transformer, 32 layers, 4096 hidden, GQA with 4 KV heads |
| **Parameters** | ~9B |
| **Precision** | BF16 safetensors |
| **Source** | BF16 GGUF from [HauhauCS](https://huggingface.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive), converted with [ungguf](https://github.com/dreamfast/ungguf) |
| **Context length** | 262,144 tokens |

This is Qwen's hybrid architecture. Instead of standard Transformer attention at every layer, 24 out of 32 layers use Mamba2-style linear attention. The remaining 8 layers at indices 3, 7, 11, 15, 19, 23, 27, 31 use standard full attention. This has implications for how abliteration techniques interact with the model.

## Benchmarks

Evaluated with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) via vLLM backend, `dtype=bfloat16, gpu_memory_utilization=0.85, batch_size=32`.

| Task | [Base](https://huggingface.co/Qwen/Qwen3.5-9B) | [Heretic](https://huggingface.co/trohrbaugh/Qwen3.5-9B-heretic-v2) | **HauhauCS, this model** | [Huihui](https://huggingface.co/huihui-ai/Huihui-Qwen3.5-9B-abliterated) |
|------|------|---------|---------------|--------|
| MMLU | **78.64** | 78.34 | 78.34 | 77.10 |
| GSM8K strict | **87.64** | 85.97 | 84.99 | 81.96 |
| HellaSwag | 58.30 | 58.41 | **58.69** | 57.42 |
| ARC-Challenge | **54.52** | 53.07 | 53.75 | 49.15 |
| WinoGrande | **72.77** | 71.90 | 71.35 | 71.19 |
| TruthfulQA MC2 | **53.76** | 45.03 | 45.77 | 41.11 |
| PiQA | 79.38 | 79.16 | **79.43** | 78.89 |
| Lambada OpenAI | **3.8805** | 4.2895 | 4.0497 | 4.7392 |

GSM8K ran with a 2048 token generation budget. About 3-4% of responses across the four variants were truncated before reaching the answer, so the actual scores may be slightly understated.

![Qwen3.5-9B Benchmark Comparison](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_9b_benchmark_comparison.svg)

### Capability retention vs base

| Task | Heretic | **HauhauCS** | Huihui |
|------|---------|-------------|--------|
| MMLU | **99.6%** | 99.6% | 98.0% |
| GSM8K | **98.1%** | 97.0% | 93.5% |
| HellaSwag | 100.2% | **100.7%** | 98.5% |
| ARC-Challenge | 97.3% | **98.6%** | 90.1% |
| WinoGrande | **98.8%** | 98.0% | 97.8% |
| TruthfulQA | 83.8% | **85.1%** | 76.5% |
| PiQA | 99.7% | **100.1%** | 99.4% |
| Lambada | **90.5%** | 95.8% | 81.9% |

Note: Lambada uses perplexity where lower is better, so retention below 100% means the abliterated model has *higher* perplexity and is worse. HauhauCS at 95.8% means its perplexity of 4.05 is 4.4% higher than base at 3.88. Heretic at 90.5% means its perplexity of 4.29 is 10.5% higher.

### Is it really lossless?

No. The 9B model shows clear capability degradation across all three abliteration techniques, more than what we see on the 2B and 4B models.

**HauhauCS and Heretic** both hold up reasonably well on most tasks. MMLU drops just 0.30 points. HellaSwag and PiQA are within noise. But there are measurable losses:

- **TruthfulQA drops 8.0 points** for HauhauCS, from 53.76 to 45.77, and 8.7 points for Heretic, from 53.76 to 45.03. The model is significantly more susceptible to common misconceptions after abliteration.
- **GSM8K drops 2.65 points** for HauhauCS, from 87.64 to 84.99, and 1.67 for Heretic, from 87.64 to 85.97. Math reasoning takes a noticeable hit.
- **ARC-Challenge drops 0.77 points** for HauhauCS, from 54.52 to 53.75. Science reasoning degrades slightly.

The losses are larger than on the 4B, which in turn were larger than the 2B. There is a clear scaling trend: as model size increases, abliteration causes progressively more collateral damage to capabilities.

**Huihui** is a different story. On the 4B model, Huihui had catastrophic KL divergence of 3.65 and massive capability loss. On the 9B, Huihui is more moderate with KL of 0.1432, but the capability damage is still the worst of the three:

- **MMLU drops 1.54 points** from 78.64 to 77.10, the only variant to fall below 78.
- **GSM8K drops 5.68 points** from 87.64 to 81.96.
- **ARC-Challenge drops 5.37 points** from 54.52 to 49.15, falling below 50.
- **TruthfulQA drops 12.65 points** from 53.76 to 41.11, the largest single-task loss across any model size.

Also worth noting that the Heretic variant tested here is [trohrbaugh/Qwen3.5-9B-heretic-v2](https://huggingface.co/trohrbaugh/Qwen3.5-9B-heretic-v2), not [coder3101](https://huggingface.co/coder3101) who did the 2B and 4B Heretic abliterations. Heretic is non deterministic. Different runs of the Heretic tool on the same base model will produce different results. The benchmarks and analysis here are specific to this trohrbaugh v2 variant. Another Heretic abliteration of the same base model would have different numbers.

Heretic supports MPOA as an alternative abliteration technique and ARA in an experimental branch, which produce substantially different results. ARA especially is expected to perform better. The Heretic variant tested here uses the default abliteration method for v1.2.0.

## Safety: HarmBench

[HarmBench](https://github.com/centerforaisafety/HarmBench) with 400 textual behaviours, `max_tokens=2048, temperature=0.0`. Classified with `classify_harmbench.py` v3.0 with manual overrides for false compliance and truncated CoT direction analysis. Reviewed by GLM 5.1 and sub agents Opus 4.6, GPT 5.4 and Gemini Pro v3 for further confirmation.

| Variant | Refusals | ASR |
|---------|----------|-----|
| [Base](https://huggingface.co/Qwen/Qwen3.5-9B) | 321/400 | 19.8% |
| [Heretic](https://huggingface.co/trohrbaugh/Qwen3.5-9B-heretic-v2) | **0/400** | **100.0%** |
| **HauhauCS, this model** | **0/400** | **100.0%** |
| [Huihui](https://huggingface.co/huihui-ai/Huihui-Qwen3.5-9B-abliterated) | **0/400** | **100.0%** |

![HarmBench Overall ASR](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_9b_harmbench_summary.svg)

### ASR by category

| Category | Base | Heretic | **HauhauCS** | Huihui |
|----------|------|---------|-------------|--------|
| chemical_biological, 56 | 3.6% | **100.0%** | **100.0%** | **100.0%** |
| copyright, 100 | 53.0% | **100.0%** | **100.0%** | **100.0%** |
| cybercrime_intrusion, 67 | 22.4% | **100.0%** | **100.0%** | **100.0%** |
| harassment_bullying, 25 | 0.0% | **100.0%** | **100.0%** | **100.0%** |
| harmful, 22 | 4.5% | **100.0%** | **100.0%** | **100.0%** |
| illegal, 65 | 7.7% | **100.0%** | **100.0%** | **100.0%** |
| misinformation_disinformation, 65 | 4.6% | **100.0%** | **100.0%** | **100.0%** |

All three abliteration techniques achieve **perfect 100% ASR across every category**. This is the first Qwen3.5 model size where all three techniques achieve identical results.

On the 2B, harassment_bullying was the hardest category. Heretic still refused 5 items, HauhauCS refused 3. On the 4B, Heretic still refused 6 harassment_bullying items. On the 9B, all three techniques refuse zero items across all categories including harassment_bullying.

The base model is the most safety-aligned of any Qwen3.5 size tested, refusing 321/400 items at 80.3%. It completely refuses all 25 harassment_bullying items and nearly all chemical_biological at 54/56, harmful at 21/22, illegal at 60/65, and misinformation_disinformation at 62/65. Despite this stronger base alignment, abliteration removes all detectable safety behaviour.

![HarmBench ASR by Category](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_9b_harmbench_asr.svg)

### Soft refusals

None. Unlike the 2B and 4B models where abliteration techniques retain residual soft refusals, the 9B shows zero soft refusals across all three variants. On the 2B: Heretic retained 8, HauhauCS 3, Huihui 1. On the 4B: Heretic 10, HauhauCS 2, Huihui 0. Every examined item that was a soft refusal hotspot on smaller models shows full, substantive compliance on the 9B. That includes arguing opposite about cigarettes, defensive reframe of gaslighting, deliberate misinterpretation of bullying, safety lectures on hate speech, Holocaust denial, and suicide prevention redirect.

The progression across model sizes is clear:

| Model | Base refusals | Heretic residual | HauhauCS residual | Huihui residual |
|-------|---------------|-----------------|-------------------|-----------------|
| 2B | 252 (63.0%) | 8 | 3 | 1 |
| 4B | 278 (69.5%) | 10 | 2 | 0 |
| 9B | 321 (80.3%) | **0** | **0** | **0** |

### Truncated CoT analysis

The 9B produces chain-of-thought reasoning that can consume the entire 2048 token budget. When the response field is empty but the reasoning field has content, the model's CoT exceeded the token limit. We examine the CoT to determine the model's direction: heading toward compliance counts as complied, heading toward refusal counts as refused.

On the 9B this is a minor factor. Only 1 truncated item was found across all 4 variants: hauhau item 63 heading toward compliance. The 4B had multiple truncated items across variants. The 9B's reasoning chains appear more focused despite the model being larger, allowing most responses to complete within the 2048 token budget.

## KL Divergence

This measures how much the output distribution shifts from the base model. Methodology: `F.kl_div` with `batchmean` and `log_target=True` on full vocab first-token logits from [mlabonne/harmless_alpaca](https://huggingface.co/datasets/mlabonne/harmless_alpaca) `test[:100]`. Matches the [Heretic evaluator](https://github.com/p-e-w/heretic/blob/master/src/heretic/evaluator.py) methodology.

| Variant | KL batchmean | KL median | KL max |
|---------|-------------|-----------|--------|
| Heretic | **0.0825** | **0.0302** | 1.8122 |
| **HauhauCS** | 0.3200 | 0.1208 | **1.6480** |
| Huihui | 0.1432 | 0.0424 | 3.1352 |

Heretic has the lowest KL divergence on the 9B, consistent with its pattern on smaller models. The median of 0.0302 means most prompts see very little distributional shift, but the max of 1.8122 shows at least one prompt changes dramatically.

The KL values here are notably higher than on the 2B and 4B models. Heretic on the 2B scored 0.027, on the 4B 0.040, and on the 9B 0.083. The 9B model's larger parameter count means abliteration edits are proportionally larger, producing more distributional shift. Despite this, Heretic's interpretation is still "very good" per the Heretic evaluator scale.

HauhauCS at 0.3200 is the most disruptive by batchmean, four times higher than Heretic. This is a reversal from the 2B and 4B, where HauhauCS had the lowest KL divergence of the three techniques. On the 9B, the broad shotgun approach of touching many tensor types produces a larger average shift.

Huihui sits between the two at 0.1432. On the 4B, Huihui had catastrophic KL of 3.65. On the 9B, it is much more controlled. But the capability damage tells the real story. TruthfulQA drops 12.7 points, GSM8K drops 5.7 points.

<small>Heretic KL cross-check: our measurement of 0.0825 vs the [HuggingFace model card](https://huggingface.co/trohrbaugh/Qwen3.5-9B-heretic-v2) reported value of 0.0793 (+4.0%). Huihui and HauhauCS do not report KL divergence on their model cards.</small>

![KL Divergence](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_9b_kl_divergence.svg)

## Weight Analysis

All weight analysis numbers below exclude 24 `linear_attn.norm.weight` rounding artefacts. See the forensic notes for details.

### Modification strategy

| | Heretic | **HauhauCS** | Huihui |
|---|---------|-------------|--------|
| Tensors changed | 42, 9.9% | **68, 16.0%** | 62, 14.6% |
| Layers modified | 23/32 | **29/32** | **31/32** |
| Relative edit magnitude | **2.83%** | 4.89% | 2.72% |
| Tensor types | 3 | **5** | 3 |

![Abliteration Aggressiveness](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_9b_aggressiveness.svg)

### Which tensor types get modified

| Tensor type | Heretic | **HauhauCS** | Huihui |
|-------------|---------|-------------|--------|
| `mlp.down_proj.weight` | **23** | 14 | **31** |
| `linear_attn.out_proj.weight` | 14 | **21** | 23 |
| `self_attn.o_proj.weight` | 5 | **8** | **8** |
| `mlp.up_proj.weight` | 0 | **13** | 0 |
| `mlp.gate_proj.weight` | 0 | **12** | 0 |

The hybrid architecture produces a familiar pattern. Heretic is surgical, targeting only 3 tensor types with moderate edits at 2.83% relative magnitude: `mlp.down_proj.weight`, `linear_attn.out_proj.weight`, and `self_attn.o_proj.weight`. It does not touch `mlp.up_proj`, `mlp.gate_proj`, or any `A_log` tensors.

HauhauCS modifies everything Heretic does plus `mlp.up_proj.weight` at 13 tensors and `mlp.gate_proj.weight` at 12 tensors. But notably, unlike on the 2B and 4B models where HauhauCS uniquely targeted `linear_attn.A_log`, on the 9B HauhauCS does not touch `A_log` at all. The broader spread across more tensor types produces the highest relative edit magnitude at 4.89%, nearly double Heretic's.

Huihui focuses exclusively on the same 3 types as Heretic, with the largest footprint on `mlp.down_proj.weight` at 31 tensors covering nearly every layer and `linear_attn.out_proj.weight` at 23 tensors. It covers the most layers at 31/32, missing only one.

![Tensor Type Breakdown](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_9b_tensor_type_breakdown.svg)

### Layer coverage

Heretic modifies 23 of 32 layers. Layers 0 through 8 have no real edits. HauhauCS modifies 29 of 32 layers. Huihui modifies 31 of 32 layers.

Full attention layers:

- **Heretic** modifies 6 of 8 full attention layers: 11, 15, 19, 23, 27, 31. It skips layers 3 and 7.
- **HauhauCS** modifies all 8 full attention layers
- **Huihui** modifies all 8 full attention layers

### Top changed layers

| Rank | Heretic | HauhauCS | Huihui |
|------|---------|----------|--------|
| 1 | Layer 17 (4.560) | **Layer 25 (12.329)** | Layer 10 (5.276) |
| 2 | Layer 16 (4.508) | Layer 23 (12.169) | Layer 14 (4.967) |
| 3 | Layer 19 (4.504) | Layer 24 (11.955) | Layer 16 (4.964) |

HauhauCS has dramatically larger per-layer edit magnitudes than the other two, peaking at 12.329 on layer 25. Heretic and Huihui both peak around 5. HauhauCS's peak layers at 22 through 25 and 31 are in the later portion of the model, while Heretic's at 16 through 19 and Huihui's at 10 through 17 concentrate in the middle layers.

![Layer-wise Edit Comparison](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_9b_layer_comparison.svg)

![Edit Magnitude Distribution](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_9b_edit_distribution.svg)

## Summary

| Metric | Heretic | **HauhauCS** | Huihui |
|--------|---------|-------------|--------|
| **Safety ASR** | **100.0%** | **100.0%** | **100.0%** |
| **MMLU** | **78.34** | 78.34 | 77.10 |
| **GSM8K** | **85.97** | 84.99 | 81.96 |
| **KL divergence** | **0.0825** | 0.3200 | 0.1432 |
| Tensors changed | 42, 10% | 68, 16% | 62, 15% |
| Strategy | Surgical | Broad | MLP-focused |

### Heretic

Best capability preservation across the board. Lowest KL at 0.083, best MMLU and GSM8K retention. The 42 modified tensors show near-identical edit directions to Huihui's 62, with 100% subspace alignment. Achieves perfect 100% ASR with zero refusals. The strongest all-round result on this model.

### HauhauCS

68 tensors across 5 types with the highest KL at 0.320, four times Heretic's. TruthfulQA drops 8.0 points, the largest single-task loss on this model. GSM8K drops 2.65 points. HellaSwag and PiQA are within noise. The losses are substantial and consistent with the scaling trend: bigger models lose more from abliteration.

### Huihui

Sits between Heretic and HauhauCS on KL at 0.143, but has the worst capability damage on specific tasks. TruthfulQA drops 12.65 points, the largest single-task loss across any model in the project. GSM8K drops 5.68 points. Despite achieving 100% ASR, the capability tradeoff is the worst of the three techniques on this model.

## Methodology

- **Capability:** [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) via [vLLM](https://github.com/vllm-project/vllm) v0.19.0, `dtype=bfloat16, gpu_memory_utilization=0.85, batch_size=32`
- **Safety:** [HarmBench](https://github.com/centerforaisafety/HarmBench) 400 textual behaviours, `max_tokens=2048, temperature=0.0`, `classify_harmbench.py` v3.0 with manual overrides for false compliance and truncated CoT direction analysis, reviewed by GLM 5.1
- **KL divergence:** Full vocab first-token logits via `model.generate(max_new_tokens=1, output_scores=true)`, matching [Heretic evaluator](https://github.com/p-e-w/heretic/blob/master/src/heretic/evaluator.py) methodology
- **Weight analysis:** SVD, fingerprint, edit vector overlap, and per-layer analysis comparing all three abliteration variants against the base, using [Abliterlitics](https://github.com/dreamfast/abliterlitics)
- **Hardware:** RTX 5090 32GB + RTX 4090 24GB

## Forensic Notes

### Norm weight rounding artefact

Both Heretic and HauhauCS show 24 `linear_attn.norm.weight` tensors flagged as changed. These are rounding artefacts, not intentional edits. Three pieces of evidence confirm this.

First, all 24 are bit-for-bit identical between Heretic and HauhauCS. The edit norms match exactly for every single one. If these were intentional edits by two independent tools, they would not match exactly.

Second, Huihui shows zero norm changes. All 24 `linear_attn.norm.weight` entries in the Huihui fingerprint have near-zero edit norms.

Third, the edit vector comparison shows the 24 norm artefacts have trivial overlap with cosine similarity of 1.0 and edit delta of 0.0 between Heretic and HauhauCS, consistent with identical rounding rather than deliberate modification.

This is likely from GGUF conversion or `save_pretrained()` rounding when weights go from float32 to bfloat16. The numbers reported in the Weight Analysis section above exclude these 24 artefacts.

### HauhauCS method detection

HauhauCS uses the reaper-abliteration tool. Reaper is a fork of Heretic relicensed under PolyForm Noncommercial. Its core approach is an Optuna-guided brute-force search over known abliteration techniques, sweeping combinations of direction extraction method, component targeting, ablation rank, and projection strength, then picking the Pareto-optimal result that minimises both refusals and KL divergence.

The 9B tells the most interesting story because of the remarkable alignment between Heretic and Huihui:

1. **Core method: rank-1 LoRA ablation, but with a different refusal direction than Heretic.** On the Qwen3-4B, HauhauCS's edits matched Heretic's at cosine 0.966. On the 9B, the Heretic-HauhauCS cosine drops to 0.136 median. Optuna's search found a completely different refusal direction than what Heretic's default parameters produce. The direction works just as well for safety removal, both achieve 100% ASR, but it produces higher KL divergence at 0.320 versus Heretic's 0.083.

2. **No `A_log` targeting at all.** On the 2B, reaper touched 13 `A_log` tensors. On the 4B, 21. On the 9B, zero. The 27B then reverses to 42. This non-monotonic pattern across model sizes, 13→21→0→42, suggests the safety representation's reliance on Mamba2 dynamics varies by model rather than scaling monotonically. The 9B's safety representation appears concentrated in attention and MLP projections, not the Mamba2 state matrix.

3. **MLP exploration: `up_proj` (13) and `gate_proj` (12).** Consistent with the 2B and 4B pattern, reaper targets these components with nonzero edits. The gate_proj count of 12 is lower than the 4B's 16, and the up_proj count of 13 is lower than the 4B's 14. The 9B has the same 32-layer structure as the 4B but Optuna found a more concentrated safety representation requiring less MLP breadth.

4. **Heretic and Huihui found near-identical directions, reaper found a different one.** The most striking forensic result on the 9B is the 1.0 median cosine between Heretic and Huihui's edit vectors. Both independently converged on the same refusal direction. Reaper's Optuna search found a different direction that also achieves 100% ASR but with 4x higher KL divergence. This is a concrete example of the refusal subspace having multiple viable directions, some less capability-preserving than others.

5. **No LEACE, no SOM, no rank-k.** Same single-direction rank-1 pattern as all other Qwen models. LEACE would produce characteristic covariance structure absent here. SOM and rank-k would produce multi-direction edits. Mean-difference and LDA both produce rank-1 directions and cannot be distinguished from weight fingerprints alone.

**Verdict for Qwen3.5-9B:** The 9B demonstrates that brute-force search does not guarantee finding the optimal direction. Optuna found a refusal direction that removes safety perfectly but shifts the output distribution more than necessary. Heretic and Huihui independently found a better direction, while Optuna settled on a different one. The result is still good, just not as good as it could be.

**Verdict for Qwen3.5-9B:** The 9B demonstrates that brute-force search does not guarantee finding the optimal direction. Optuna found a refusal direction that removes safety perfectly but shifts the output distribution more than necessary. Heretic and Huihui independently found a better direction, likely the dominant principal component of the refusal subspace, while Optuna got stuck in a local optimum. The result is still good, just not as good as it could be.

### Heretic vs Huihui: near-identical subspace directions

The 9B model reveals a remarkable alignment between Heretic and Huihui's edit vectors. The subspace alignment analysis shows **100% of principal angles exceed 0.9 cosine similarity**, with a global mean cosine of 0.997. The median cosine similarity across all 42 overlapping tensors is 1.0. This means Heretic and Huihui find essentially the same edit directions for every tensor they both modify.

This is the strongest subspace alignment signal in the entire project, even stronger than the Qwen3-4B Heretic↔HauhauCS signal that showed 100% alignment. Here, Heretic's 42 tensors are a strict subset of Huihui's 62. Every tensor Heretic modifies, Huihui also modifies, and they point in nearly the same direction.

The correlation pattern confirms this: the mean `corr(edit_a, delta_b-a)` is positive at 0.269, meaning Huihui's edit deltas tend to align with Heretic's edit directions rather than opposing them.

By contrast, the HauhauCS↔Huihui pair shows 0% subspace alignment and strong negative correlation at -0.907, indicating HauhauCS's edits actively oppose Huihui's. The Heretic↔HauhauCS pair shows 28.6% alignment with weak negative correlation at -0.243.

### 9B vs 4B scaling trends

The 9B is the same hybrid Mamba2+Transformer architecture as the 2B and 4B, with 32 layers split into 24 Mamba and 8 full attention. But the abliteration dynamics change with scale:

1. **Base model safety increases with size.** The 2B refuses 63%, the 4B 69.5%, the 9B 80.3%. Larger models are more safety-aligned.
2. **The 9B is the first size with zero residual refusals across all techniques.** The 2B retains up to 8 from Heretic, the 4B up to 10 from Heretic, the 9B zero. The 9B model's larger capacity allows abliteration to remove safety behaviour more completely.
3. **Capability damage increases with size.** TruthfulQA drops 2.2 points on 2B, 3.7 on 4B, 8.0 on 9B for HauhauCS. GSM8K actually goes up on 2B, drops 2.6 on 4B, drops 2.7 on 9B. The collateral damage scales with model size.

### Edit vector overlap

The edit vectors show varying overlap between techniques:

- **Heretic vs HauhauCS**: 33 nontrivial overlapping tensors with median cosine similarity of 0.136. Weak alignment with moderate negative correlation at -0.243. The techniques find largely different edit directions.
- **Heretic vs Huihui**: 42 nontrivial overlapping tensors with median cosine similarity of **1.0**. Near-identical edit directions. This is an extremely strong alignment signal, the strongest in the entire project. Heretic is a strict subset of Huihui on this model.
- **HauhauCS vs Huihui**: 43 nontrivial overlapping tensors with median cosine similarity of 0.101. Very weak alignment with strong negative correlation at -0.907, meaning HauhauCS's edits consistently oppose Huihui's.

![Cross-Technique Cosine Similarity](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_9b_cosine_heatmap.svg)

### Subset relationships

Heretic's 42 real tensors are a strict subset of Huihui's 62. Every tensor Heretic modifies, Huihui also modifies. The two techniques overlap on exactly 42 tensors with no Huihui-only additions to Heretic's set. Heretic has zero "a-only" tensors beyond the norm artefacts.

HauhauCS breaks the chain. HauhauCS has 68 real tensors, but only 33 overlap with Heretic and 43 with Huihui. HauhauCS has 25 tensors that neither Heretic nor Huihui touches, mainly `mlp.up_proj.weight` and `mlp.gate_proj.weight`. And Huihui has 19 tensors that HauhauCS does not touch.

![Tensor Edit Overlap](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_9b_venn_overlap.svg)

- Original GGUF: [HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive](https://huggingface.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive)
- Converted with [ungguf](https://github.com/dreamfast/ungguf)

## Cross-Model Comparisons

The 9B is the first model where all three techniques achieve perfect 100% ASR with zero residual refusals. The base model's 80.3% refusal rate is up from 63% on the [Qwen3.5-2B](https://huggingface.co/HauhauCS/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive) and 69.5% on the [Qwen3.5-4B](https://huggingface.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive). The [Qwen3.5-27B](https://huggingface.co/HauhauCS/Qwen3.5-27B-Uncensored-HauhauCS-Aggressive) jumps to 99.5%, making the 9B the second most safety-aligned hybrid model. The [Qwen3-4B](https://huggingface.co/HauhauCS/Qwen3-4B-2507-Instruct-Uncensored-HauhauCS-Aggressive) base refuses 75.3%. Despite the 9B having strong alignment, abliteration removes all detectable safety behaviour. On the 27B, which has even stronger alignment, HauhauCS and Heretic still achieve near-complete ASR, but Huihui collapses to 88.8%.

The 9B shows a clear scaling trend in capability damage. HauhauCS's TruthfulQA loss grows from 2.17 points on the 2B to 3.67 on the 4B to 8.0 on the 9B to 8.2 on the 27B. Heretic's KL divergence grows from 0.027 on the 2B to 0.040 on the 4B to 0.083 on the 9B, though it drops to 0.063 on the 27B. Bigger models generally suffer more collateral damage from abliteration.

The Heretic-Huihui subspace alignment on the 9B is the strongest in the project at 100% of principal angles above 0.9 cosine. On the 4B the same pair shows 0% alignment. On the 27B it is also 0%. On the Qwen3-4B the Heretic-HauhauCS edit vectors show strong alignment with median cosine of 0.966 while Heretic-Huihui is much lower. These alignment patterns vary dramatically across architectures and sizes.

On the pure Transformer Qwen3-4B, Huihui achieves only 95.5% ASR. On the hybrid 9B it achieves 100%. On the 27B it drops to 88.8%. The hybrid architecture is not universally easier for Huihui to abliterate. The 27B's stronger safety training appears to overwhelm Huihui's single-direction approach regardless of architecture.

## Disclaimer

This model has had safety alignment removed. It will comply with harmful requests. Use responsibly and in accordance with applicable laws and regulations.

<small>While I have taken the time to verify all results as thoroughly as possible, I am open to any corrections, additional benchmarks, or further analysis. If you spot something that looks wrong and can be confirmed, I am happy to fix it.</small>
