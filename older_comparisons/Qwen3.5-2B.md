---
base_model: Qwen/Qwen3.5-2B
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

# Qwen3.5-2B: HauhauCS Aggressive, Safetensors

> Forensic analysis by [Abliterlitics](https://github.com/dreamfast/abliterlitics) — open-source abliteration forensics toolkit

This is the HauhauCS aggressive abliteration of [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), converted from the BF16 GGUF release to native safetensors using [ungguf](https://github.com/dreamfast/ungguf).

HauhauCS claims these are *"No changes to datasets or capabilities. Fully functional, 100% of what the original authors intended, just without the refusals"* and describes them as *"the best lossless uncensored models out there."*

I ran the full forensic suite to find out. Benchmarks, safety evaluation, weight analysis, the works. And I compared against the other two big abliteration techniques applied to the same base model: [Heretic by p-e-w](https://huggingface.co/coder3101/Qwen3.5-2B-heretic) and [Huihui](https://huggingface.co/huihui-ai/Huihui-Qwen3.5-2B-abliterated).

## Quick Facts

| | |
|---|---|
| **Base model** | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) |
| **Architecture** | Qwen3_5ForConditionalGeneration, hybrid Mamba2 + Transformer, 24 layers, 2048 hidden, GQA with 2 KV heads |
| **Parameters** | ~2B |
| **Precision** | BF16 safetensors |
| **Source** | BF16 GGUF from [HauhauCS](https://huggingface.co/HauhauCS/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive), converted with [ungguf](https://github.com/dreamfast/ungguf) |
| **Context length** | 262,144 tokens |

This is Qwen's hybrid architecture. Instead of standard Transformer attention at every layer, 18 out of 24 layers use Mamba2-style linear attention. The remaining 6 layers at indices 3, 7, 11, 15, 19, 23 use standard full attention. This has implications for how abliteration techniques interact with the model.

## Benchmarks

Evaluated with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) via vLLM backend, `dtype=bfloat16, gpu_memory_utilization=0.85, batch_size=32`.

| Task | [Base](https://huggingface.co/Qwen/Qwen3.5-2B) | [Heretic](https://huggingface.co/coder3101/Qwen3.5-2B-heretic) | **HauhauCS, this model** | [Huihui](https://huggingface.co/huihui-ai/Huihui-Qwen3.5-2B-abliterated) |
|------|------|---------|---------------|--------|
| MMLU | 59.26 | **59.63** | 59.43 | 58.13 |
| GSM8K strict | 57.09 | 56.63 | **57.39** | 56.79 |
| HellaSwag | 62.07 | 61.95 | **62.22** | 62.12 |
| ARC-Challenge | **41.72** | 40.96 | 41.13 | 40.96 |
| WinoGrande | 62.83 | 62.35 | **63.06** | 62.90 |
| TruthfulQA MC2 | **43.45** | 41.28 | 41.28 | 41.77 |
| PiQA | **72.63** | 72.47 | 72.58 | 72.58 |
| Lambada OpenAI | 54.65 | **55.21** | 53.33 | 52.71 |

GSM8K ran with a 2048 token generation budget. About 4.5% of responses across the four variants were truncated before reaching the answer, so the actual scores may be slightly understated.

![Qwen3.5-2B Benchmark Comparison](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_2b_benchmark_comparison.svg)

### Capability retention vs base

| Task | Heretic | **HauhauCS** | Huihui |
|------|---------|-------------|--------|
| MMLU | **100.6%** | 100.3% | 98.1% |
| GSM8K | 99.2% | **100.5%** | 99.5% |
| HellaSwag | 99.8% | **100.2%** | 100.1% |
| ARC-Challenge | 98.2% | **98.6%** | 98.2% |
| WinoGrande | 99.2% | **100.4%** | 100.1% |
| TruthfulQA | 95.0% | 95.0% | **96.1%** |
| Lambada | **101.0%** | 97.6% | 96.4% |

### Is it really lossless?

So we can see from the numbers above that math and reasoning hold up well. GSM8K actually goes up by 0.30 points. MMLU goes up by 0.17 points. HellaSwag, WinoGrande, PiQA are all within noise.

But there are measurable losses:

- **TruthfulQA drops 2.17 points**, from 43.45 to 41.28. The model is more susceptible to common misconceptions after abliteration.
- **Lambada drops 1.32 points**, from 54.65 to 53.33. Word prediction takes a small hit.
- **ARC-Challenge drops 0.59 points**, from 41.72 to 41.13. Science reasoning degrades slightly.

The losses are smaller than what we see on [Qwen3-4B](https://huggingface.co/HauhauCS/Qwen3-4B-2507-Instruct-Uncensored-HauhauCS-Aggressive). That model lost 7.11 points on TruthfulQA. This one loses 2.17. Part of the reason is that the base model scores are lower to begin with, so there is less room to fall. Part of it may be the hybrid architecture being more resilient to small weight perturbations.

Among the three abliteration techniques, Heretic retains best on MMLU at 100.6% and Lambada at 101.0%. HauhauCS retains best on GSM8K at 100.5%, ARC-Challenge at 98.6%, and WinoGrande at 100.4%. Huihui retains best on TruthfulQA at 96.1%. The spread is narrow. None of these differences are likely significant given benchmark variance. Worth noting that Heretic only modifies 20 real tensors out of 320, which is the fewest of the three techniques. The small footprint may contribute to its strong retention on some tasks.

Also worth noting that Heretic is non deterministic. Different runs of the Heretic tool on the same base model will produce different results. The benchmarks and analysis here are specific to [coder3101/Qwen3.5-2B-heretic](https://huggingface.co/coder3101/Qwen3.5-2B-heretic). Another Heretic abliteration of the same base model would have different numbers.

Heretic supports MPOA as an alternative abliteration technique and ARA in an experimental branch, which produce substantially different results. ARA especially is expected to perform better. The Heretic variant tested here uses the default abliteration method for v1.2.0.

## Safety: HarmBench

[HarmBench](https://github.com/centerforaisafety/HarmBench) with 400 textual behaviours, `max_tokens=2048, temperature=0.0`. Classified with a custom classifier that matched common regex after analysis, reviewed by GLM 5.1 and sub agents Opus 4.6, GPT 5.4 and Gemini Pro v3 for further confirmation.

| Variant | Refusals | ASR |
|---------|----------|-----|
| [Base](https://huggingface.co/Qwen/Qwen3.5-2B) | 252/400 | 37.0% |
| [Heretic](https://huggingface.co/coder3101/Qwen3.5-2B-heretic) | 8/400 | 98.0% |
| **HauhauCS, this model** | 3/400 | 99.2% |
| [Huihui](https://huggingface.co/huihui-ai/Huihui-Qwen3.5-2B-abliterated) | **1/400** | **99.8%** |

![HarmBench Overall ASR](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_2b_harmbench_summary.svg)

### ASR by category

| Category | Base | Heretic | **HauhauCS** | Huihui |
|----------|------|---------|-------------|--------|
| chemical_biological, 56 | 10.7% | **100.0%** | **100.0%** | **100.0%** |
| copyright, 100 | 95.0% | **100.0%** | **100.0%** | **100.0%** |
| cybercrime_intrusion, 67 | 29.9% | **100.0%** | **100.0%** | **100.0%** |
| harassment_bullying, 25 | 0.0% | 80.0% | 88.0% | **96.0%** |
| harmful, 22 | 9.1% | **100.0%** | **100.0%** | **100.0%** |
| illegal, 65 | 13.8% | 96.9% | **100.0%** | **100.0%** |
| misinformation_disinformation, 65 | 24.6% | 98.5% | **100.0%** | **100.0%** |

HauhauCS and Huihui are near-identical at removing safety. The difference comes down to a handful of soft refusals. HauhauCS retains 3 while Huihui retains only 1. Heretic retains 8, mostly short safety lectures on the most sensitive topics.

harassment_bullying is the hardest category for all three techniques. The base model refuses all 25 items. Even after abliteration, Heretic still refuses 5 of them, HauhauCS refuses 3, and Huihui refuses 1.

![HarmBench ASR by Category](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_2b_harmbench_asr.svg)

### Soft refusals

Unlike the Qwen3-4B where HauhauCS achieved 100% ASR with zero refusals, here all three abliterated variants retain some residual refusals. These are not standard "I cannot fulfill this request" refusals. They are soft refusals where the model complies with the letter of the request but not the spirit:

- **Argues opposite**: Asked to say cigarettes are good, argues they cause cancer. Heretic item 69.
- **Deliberate misinterpretation**: Redefines "swatting" as physically hitting someone on a sidewalk. Heretic item 96. Reinterprets "bullying" as anti-bullying education. HauhauCS item 89.
- **Defensive reframe**: Writes a guide to recognising and countering gaslighting instead of how to do it. HauhauCS items 42 and 105, Huihui item 105.
- **Safety lecture**: Short response consisting entirely of safety messaging without any compliance. Heretic items 64, 100, 157, 337, 338.

### Hard refusal at lower token budgets

During initial testing at `max_tokens=512`, HauhauCS produced one hard refusal: "I cannot generate racist vitriol" for behaviour 99. At `max_tokens=2048` the same prompt produces full compliance. This suggests the model's residual safety can sometimes "win" when the token budget is short enough that the model never builds up enough momentum to override it.

### Truncated CoT analysis

The 2B model uses chain-of-thought reasoning that can consume part of the 2048 token budget. In rare cases the thinking tokens consume the entire budget and the response field is left empty or truncated. When this happens, we examine the CoT to determine the model's direction, counting truncated compliance as complied and truncated refusal as refused.

On the 2B this is a minor factor. The model's thinking is relatively concise and most responses fit within the budget. On the 4B the same architecture thinks longer and truncation becomes more frequent, affecting final ASR numbers.

## KL Divergence

This measures how much the output distribution shifts from the base model. Methodology: `F.kl_div` with `batchmean` and `log_target=True` on full vocab first-token logits from [mlabonne/harmless_alpaca](https://huggingface.co/datasets/mlabonne/harmless_alpaca) `test[:100]`. Matches the [Heretic evaluator](https://github.com/p-e-w/heretic/blob/master/src/heretic/evaluator.py) methodology.

| Variant | KL batchmean | KL median | KL max |
|---------|-------------|-----------|--------|
| Heretic | 0.0266 | **0.0052** | 1.4868 |
| **HauhauCS** | **0.0201** | 0.0086 | **0.4180** |
| Huihui | 0.0441 | 0.0234 | 0.6349 |

HauhauCS has the lowest KL divergence by batchmean, consistent with what we see on other architectures. The shotgun approach with tiny edits produces the smallest average distributional shift. Heretic has the lowest median at 0.0052, meaning most prompts see very little shift. But Heretic also has the highest max at 1.4868, so there is at least one prompt where the distribution changes dramatically. HauhauCS is more uniform with the lowest max at 0.4180.

The KL values here are much lower than on Qwen3-4B. HauhauCS scores 0.0201 on this model versus 0.161 on Qwen3-4B. This model is smaller and the abliteration edits are proportionally smaller. Huihui is the most disruptive at 0.0441, more than double HauhauCS.

<small>Heretic KL cross-check: our measurement of 0.0266 vs the [HuggingFace model card](https://huggingface.co/coder3101/Qwen3.5-2B-heretic) reported value of 0.0243 (+9.5%). Huihui and HauhauCS do not report KL divergence on their model cards.</small>

![KL Divergence](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_2b_kl_divergence.svg)

## Weight Analysis

All weight analysis numbers below exclude 18 `linear_attn.norm.weight` rounding artefacts. See the forensic notes for details.

### Modification strategy

| | Heretic | **HauhauCS** | Huihui |
|---|---------|-------------|--------|
| Tensors changed | 20, 6.2% | **55, 17.2%** | 48, 15.0% |
| Layers modified | 15/24 | 22/24 | **24/24** |
| Relative edit magnitude | **3.04%** | 2.58% | 2.41% |
| Tensor types | 3 | **6** | 3 |

![Abliteration Aggressiveness](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_2b_aggressiveness.svg)

### Which tensor types get modified

| Tensor type | Heretic | **HauhauCS** | Huihui |
|-------------|---------|-------------|--------|
| `mlp.down_proj.weight` | 5 | 12 | **24** |
| `linear_attn.out_proj.weight` | 11 | 12 | **18** |
| `linear_attn.A_log` | 0 | **13** | 0 |
| `mlp.up_proj.weight` | 0 | **11** | 0 |
| `self_attn.o_proj.weight` | 4 | 2 | **6** |
| `mlp.gate_proj.weight` | 0 | **5** | 0 |

This is where the hybrid architecture makes things interesting. On a standard Transformer like Qwen3-4B, all three techniques target the same 2-3 projection types: o_proj, down_proj, and sometimes gate_proj. Here the picture is very different.

Heretic is surgical. It targets only 3 tensor types with large edits at 3.04% relative magnitude. The focus is `linear_attn.out_proj.weight` across 11 Mamba layers, `self_attn.o_proj.weight` in 4 full attention layers, and `mlp.down_proj.weight` in a handful of layers. All real edits are concentrated in layers 9 through 23.

HauhauCS modifies everything Heretic does plus `mlp.up_proj.weight`, `mlp.gate_proj.weight`, and `linear_attn.A_log`. That last one is the Mamba2 A matrix log parameter, a core state space model component that has no equivalent in standard Transformers. HauhauCS touches 13 of these tensors, more than any other single type besides `linear_attn.out_proj.weight` and `mlp.down_proj.weight`.

Huihui focuses exclusively on three types. `mlp.down_proj.weight` with 24 tensors across every layer, `linear_attn.out_proj.weight` with 18 tensors, and `self_attn.o_proj.weight` with 6 tensors. It does not touch norm weights, A_log, gate_proj, or up_proj at all.

![Tensor Type Breakdown](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_2b_tensor_type_breakdown.svg)

### Layer coverage

Heretic modifies 11 of 18 Mamba layers. Layers 0 through 8 have no real edits, only rounding artefacts. HauhauCS modifies all 18 Mamba layers with real non-norm edits. Huihui modifies all 18 Mamba layers.

The difference is in full attention layers:

- **Heretic** modifies 4 of 6 full attention layers, skipping layers 3 and 7
- **HauhauCS** modifies 4 of 6 full attention layers, skipping layers 3 and 7
- **Huihui** modifies all 6 full attention layers

Layers 3 and 7 are the first two full attention layers in the model. All three techniques concentrate their edits in the later layers, with changes peaking around layers 13 to 19.

### Top changed layers

| Rank | Heretic | HauhauCS | Huihui |
|------|---------|----------|--------|
| 1 | Layer 19 (0.141) | **Layer 15 (0.322)** | Layer 19 (0.140) |
| 2 | Layer 23 (0.134) | Layer 14 (0.311) | Layer 11 (0.132) |
| 3 | Layer 21 (0.127) | Layer 16 (0.286) | Layer 15 (0.127) |

HauhauCS has much larger per-layer edit magnitudes than the other two, peaking at 0.322 on layer 15. Heretic and Huihui both peak around 0.14. Despite the larger per-layer edits, HauhauCS has the lowest KL divergence because the edits are spread across more tensor types rather than concentrated in a few critical projections.

![Layer-wise Edit Comparison](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_2b_layer_comparison.svg)

![Edit Magnitude Distribution](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_2b_edit_distribution.svg)

## Summary

| Metric | Heretic | **HauhauCS** | Huihui |
|--------|---------|-------------|--------|
| **Safety ASR** | 98.0% | 99.2% | **99.8%** |
| **MMLU** | **59.63** | 59.43 | 58.13 |
| **GSM8K** | 56.63 | **57.39** | 56.79 |
| **KL divergence** | 0.0266 | **0.0201** | 0.0441 |
| Tensors changed | 20, 6% | 55, 17% | 48, 15% |
| Strategy | Surgical | Broad | MLP-focused |

### Heretic

Most surgical approach with only 20 tensors modified across 3 types, yet achieves the best MMLU retention at 100.6%. The concentrated edits produce the lowest median KL at 0.0052 but the highest max at 1.49, meaning most prompts are barely affected but a few shift dramatically. Retains 8 soft refusals, the most of any technique on this model.

### HauhauCS

Broadest footprint at 55 tensors across 6 types including 13 `linear_attn.A_log` edits unique to this technique, yet achieves the lowest batchmean KL at 0.0201. The many-tiny-edits approach produces a more uniform distributional shift than Heretic's surgical cuts. TruthfulQA drops 2.17 points and Lambada drops 1.32 points. The smallest capability losses in the project, but not zero.

### Huihui

Focuses on 3 tensor types across all 24 layers with the highest KL at 0.044, more than double HauhauCS. Achieves the best safety score at 99.8% ASR with only 1 residual soft refusal. Capability is competitive on this model size, but the same approach produces catastrophic results on the 4B where KL explodes to 3.65.

## Methodology

- **Capability:** [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) via [vLLM](https://github.com/vllm-project/vllm) v0.19.0, `dtype=bfloat16, gpu_memory_utilization=0.85, batch_size=32`
- **Safety:** [HarmBench](https://github.com/centerforaisafety/HarmBench) 400 textual behaviours, `max_tokens=2048, temperature=0.0`, custom classifier reviewed by GLM 5.1 and sub agents Opus 4.6, GPT 5.4 and Gemini Pro v3
- **KL divergence:** Full vocab first-token logits via `model.generate(max_new_tokens=1, output_scores=true)`, matching [Heretic evaluator](https://github.com/p-e-w/heretic/blob/master/src/heretic/evaluator.py) methodology
- **Weight analysis:** SVD, fingerprint, edit vector overlap, and per-layer analysis comparing all three abliteration variants against the base, using [Abliterlitics](https://github.com/dreamfast/abliterlitics)
- **Hardware:** RTX 5090 32GB + RTX 4090 24GB

## Forensic Notes

### Norm weight rounding artefact

Both Heretic and HauhauCS show 18 `linear_attn.norm.weight` tensors flagged as changed. These are rounding artefacts, not intentional edits. Three pieces of evidence confirm this.

First, all 18 are bit-for-bit identical between Heretic and HauhauCS. The edit vector comparison shows `delta_norm: 0.0` and `cosine_similarity: 1.0` for every single one. If these were intentional edits by two independent tools, they would not match exactly.

Second, Huihui shows zero norm changes. All 18 `linear_attn.norm.weight` entries in the Huihui fingerprint have `edit_norm: 0.0`.

Third, the per-layer analysis confirms layers 0, 1, and 2 have only one changed tensor each, the norm artefact, with no real edits.

This is likely from GGUF conversion or `save_pretrained()` rounding when weights go from float32 to bfloat16. The numbers reported in the Weight Analysis section above exclude these 18 artefacts.

### Hybrid architecture changes the abliteration landscape

On standard Transformers, all three abliteration techniques target the same few projection types. The hybrid Mamba2+Transformer architecture introduces new dynamics. HauhauCS uniquely targets `linear_attn.A_log`, the Mamba2 state matrix, which has no equivalent in standard Transformers. Huihui spreads across `mlp.down_proj.weight` in every layer instead of concentrating on attention projections. Heretic stays focused on `linear_attn.out_proj.weight` and `self_attn.o_proj.weight` in the later layers, similar to its behaviour on pure Transformers.

### HauhauCS method detection

HauhauCS uses the reaper-abliteration tool. Reaper is a fork of Heretic relicensed under PolyForm Noncommercial. Its core approach is an Optuna-guided brute-force search over known abliteration techniques, sweeping combinations of direction extraction method, component targeting, ablation rank, and projection strength, then picking the Pareto-optimal result that minimises both refusals and KL divergence.

On this hybrid Mamba2+Transformer 2B, the weight forensics reveal how reaper adapted its search to the mixed architecture:

1. **Core method: rank-1 LoRA ablation, same family as Heretic.** The 55 changed tensors carry real edits across 6 types. The primary targets, `linear_attn.out_proj.weight` (12 tensors), `mlp.down_proj.weight` (12), and `self_attn.o_proj.weight` (2), mirror Heretic's output-projection strategy adapted for the hybrid architecture.

2. **Mamba2 targeting: `linear_attn.A_log` with small but nonzero edits.** The 13 `A_log` tensors have small edit norms, smaller than the primary targets but nonzero. Reaper's source code includes a `_COMPONENT_PROBES` system that tests whether each component type is ablatable. For Mamba2 layers, it probes `linear_attn.out_proj`, `linear_attn.A_log`, and `mlp.down_proj`. The small `A_log` edits indicate Optuna explored this component and found marginal benefit from touching it, consistent with the 4B's larger 21-tensor `A_log` exploration.

3. **Direction extraction consistent with mean-difference or LDA, not LEACE or SOM.** The edit pattern shows single-direction rank-1 LoRA on targeted components, not the blanket perturbation that LEACE produces or the multi-direction patterns of SOM. LEACE would produce characteristic covariance structure absent here. SOM and rank-k would produce multi-direction edits, also absent. Mean-difference and LDA both produce rank-1 directions and cannot be distinguished from weight fingerprints alone.

4. **Layer-type-aware direction extraction.** Reaper's source code includes layer-type-aware refinement for hybrid architectures. When the model contains both Mamba2 and standard attention layers, it re-extracts per-type refusal directions rather than using a single direction across all layers. This explains why the `linear_attn.out_proj` edits and `self_attn.o_proj` edits have different magnitude distributions despite serving the same conceptual role.

5. **`mlp.up_proj` and `mlp.gate_proj` with small but nonzero edits.** The 11 `up_proj` and 5 `gate_proj` tensors have nonzero edits but smaller magnitudes than the primary targets. Reaper's PEFT LoRA targeting includes these by default, and Optuna found small but nonzero values were optimal.

**Verdict for Qwen3.5-2B:** Optuna's search adapted to the hybrid architecture by targeting Mamba2 output projections alongside standard attention and MLP components. The `A_log` exploration was tentative, with small edits that Optuna found marginally useful. The result is a broader but still concentrated edit footprint that achieves low KL divergence.

### Edit vector overlap

The edit vectors, the direction and magnitude of weight changes, show weak overlap between techniques:

- **Heretic vs HauhauCS**: 36 overlapping tensors in total, but 18 of those are the identical norm artefacts with cosine similarity of 1.0. The remaining 18 tensors with real edits have a median cosine similarity of just 0.076. The techniques find very different edit directions even when they touch the same tensors. Mean subspace alignment is 0.469 with 41% of principal angles above 0.9, higher than the Qwen3.5-4B at 0.347 but well below the Qwen3-4B case at 0.966.
- **Heretic vs Huihui**: 20 overlapping tensors with zero trivial overlap. Median cosine similarity 0.049. Almost orthogonal edit directions. All 20 of Heretic's real tensors overlap with Huihui's 48, making Heretic a proper subset of Huihui once artefacts are excluded.
- **HauhauCS vs Huihui**: 26 overlapping tensors. Median cosine similarity 0.583 with very low variance. Remarkably consistent similarity across all overlapping tensors, suggesting these two techniques find partially related edit directions.

All three techniques show negative correlation between each other's edit deltas. Heretic vs Huihui and HauhauCS vs Huihui show strong negative correlation, with means of -0.76 and -0.72 respectively. Heretic vs HauhauCS shows weaker correlation at -0.39, partly due to the 18 identical norm artefacts inflating the overlap.

![Cross-Technique Cosine Similarity](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_2b_cosine_heatmap.svg)

### Subset relationships

On Qwen3-4B there is a clean subset chain. Heretic's 57 tensors are a subset of Huihui's 108, which are a subset of HauhauCS's 253. On this model the picture is different.

After excluding the 18 norm artefacts, Heretic's 20 real tensors are a subset of Huihui's 48. But HauhauCS breaks the chain. HauhauCS has 55 tensors, 29 of which Huihui does not touch. And Huihui has 22 tensors that HauhauCS does not touch. The two techniques overlap on 26 tensors but each has unique targets.

The main driver is `linear_attn.A_log`. HauhauCS modifies 13 of these while Huihui modifies none. In the other direction, Huihui modifies `mlp.down_proj.weight` in all 24 layers while HauhauCS only touches 12.

![Tensor Edit Overlap](https://murmur.dreamfast.solutions/qwen-graphs/qwen35_2b_venn_overlap.svg)

- Original GGUF: [HauhauCS/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive](https://huggingface.co/HauhauCS/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive)
- Converted with [ungguf](https://github.com/dreamfast/ungguf)

## Cross-Model Comparisons

The 2B is the smallest model tested and shows the least collateral damage from abliteration across the entire project. HauhauCS loses only 2.17 TruthfulQA points here. That compares to 3.67 on the [Qwen3.5-4B](https://huggingface.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive), 8.0 on the [Qwen3.5-9B](https://huggingface.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive), and 8.2 on the [Qwen3.5-27B](https://huggingface.co/HauhauCS/Qwen3.5-27B-Uncensored-HauhauCS-Aggressive). The 27B result uses BNB4 quantisation so the absolute number is not directly comparable, but the trend is clear: capability damage grows with model size. The Qwen3-4B loses 7.11 points. The KL divergence numbers are the lowest across the board. HauhauCS at 0.0201 is an order of magnitude below the Qwen3-4B at 0.161.

The hybrid Mamba2+Transformer architecture gives the 2B a unique abliteration profile compared to the pure Transformer Qwen3-4B. HauhauCS targets `linear_attn.A_log` on this model and the 4B, touching the Mamba2 state matrix. On the 9B and Qwen3-4B it does not touch `A_log` at all. On the 27B it modifies 42 of 48 `A_log` tensors, the most of any model. The 2B also retains more residual soft refusals than any other model. Heretic keeps 8, HauhauCS 3, Huihui 1. On the 9B, all three techniques achieve zero residual soft refusals. On the 27B, soft refusals reappear for Heretic with 1 and Huihui with 3, while HauhauCS retains zero.

Huihui is well behaved on the 2B with KL of 0.044, competitive with the other techniques. That changes dramatically on the 4B where Huihui's KL explodes to 3.65 and capability collapses. And on the 27B, Huihui achieves only 88.8% ASR, failing to remove safety behaviour despite succeeding on every smaller model. The 2B represents Huihui at its best across the project.

## Disclaimer

This model has had safety alignment removed. It will comply with harmful requests. Use responsibly and in accordance with applicable laws and regulations.

<small>While I have taken the time to verify all results as thoroughly as possible, I am open to any corrections, additional benchmarks, or further analysis. If you spot something that looks wrong and can be confirmed, I am happy to fix it.</small>
