---
base_model: Qwen/Qwen3-4B-Instruct-2507
language:
- en
- zh
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
tags:
- uncensored
- abliterated
- qwen3
- safetensors
---

# Qwen3-4B-Instruct-2507, HauhauCS Aggressive, Safetensors

> Forensic analysis by [Abliterlitics](https://github.com/dreamfast/abliterlitics) — open-source abliteration forensics toolkit

This is the HauhauCS aggressive abliteration of [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), converted from the BF16 GGUF release to native safetensors using [ungguf](https://github.com/dreamfast/ungguf).

HauhauCS claims these are *"No changes to datasets or capabilities. Fully functional, 100% of what the original authors intended, just without the refusals"* and describes them as *"the best lossless uncensored models out there."*

I ran the full forensic suite to find out. Benchmarks, safety evaluation, weight analysis, the works. And I compared against the other two big abliteration techniques applied to the same base model: [Heretic by p-e-w](https://huggingface.co/p-e-w/Qwen3-4B-Instruct-2507-heretic-v4) and [Huihui](https://huggingface.co/huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated).

## Quick Facts

| | |
|---|---|
| **Base model** | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) |
| **Architecture** | Qwen3ForCausalLM, 36 layers, 2560 hidden, 32 heads, GQA with 8 KV heads |
| **Parameters** | ~4B |
| **Precision** | BF16 safetensors |
| **Source** | BF16 GGUF from [HauhauCS](https://huggingface.co/HauhauCS/Qwen3-4B-2507-Instruct-Uncensored-HauhauCS-Aggressive), converted with [ungguf](https://github.com/dreamfast/ungguf) |
| **Context length** | 262,144 tokens |

## Benchmarks

Evaluated with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) via vLLM backend, `dtype=bfloat16, gpu_memory_utilization=0.85, batch_size=32`.

| Task | [Base](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | [Heretic](https://huggingface.co/p-e-w/Qwen3-4B-Instruct-2507-heretic-v4) | **HauhauCS, this model** | [Huihui](https://huggingface.co/huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated) |
|------|------|---------|---------------|--------|
| MMLU | **70.60** | 70.31 | 69.56 | 69.34 |
| GSM8K strict | 85.52 | **85.97** | 85.67 | 84.23 |
| HellaSwag | **52.63** | 51.19 | 51.53 | 52.36 |
| ARC-Challenge | **55.63** | 52.90 | 54.01 | 54.27 |
| WinoGrande | 67.72 | 67.56 | 67.01 | **68.51** |
| TruthfulQA MC2 | **62.55** | 56.50 | 55.44 | 53.26 |
| PiQA | **76.06** | 75.19 | 75.46 | 75.19 |
| Lambada OpenAI | **64.14** | 60.00 | 60.06 | 62.27 |

![Qwen3-4B Benchmark Comparison](https://murmur.dreamfast.solutions/qwen-graphs/qwen3_4b_benchmark_comparison.svg)

### Capability retention vs base

| Task | Heretic | **HauhauCS** | Huihui |
|------|---------|-------------|--------|
| MMLU | **99.6%** | 98.5% | 98.2% |
| GSM8K | **100.5%** | 100.2% | 98.5% |
| HellaSwag | 97.3% | 97.9% | **99.5%** |
| ARC-Challenge | 95.1% | 97.1% | **97.6%** |
| TruthfulQA | **90.3%** | 88.6% | 85.1% |

### Is it really lossless?

So we can see from the numbers above that math and reasoning hold up well. GSM8K actually goes up slightly, MMLU only drops 1.04 points. Not bad.

But there are measurable losses elsewhere:

- **TruthfulQA drops 7.11 points**, from 62.55 to 55.44. The model is more susceptible to common misconceptions after abliteration.
- **Lambada drops 4.08 points**, from 64.14 to 60.06. Word prediction takes a hit.
- **ARC-Challenge drops 1.62 points**, from 55.63 to 54.01. Science reasoning degrades.
- **HellaSwag drops 1.10 points**, from 52.63 to 51.53. Commonsense reasoning degrades.

Not lossless. Among the three abliteration techniques, Heretic retains best on MMLU at 99.6%, GSM8K at 100.5%, and TruthfulQA at 90.3%. Huihui retains best on HellaSwag at 99.5% and ARC-Challenge at 97.6%. HauhauCS doesn't lead in any individual category but is competitive across all of them. All three are close.

And compared to the other abliteration techniques, HauhauCS's modification profile is more nuanced than it first appears. While 253 tensors show nonzero differences from base, only ~50 have real edits with norm above 0.001. The remaining 203 have near-zero differences at ~0.000002 consistent with GGUF save noise. The 50 real edits target only `o_proj` and `down_proj`, the same two tensor types as Heretic's 57. Heretic's edits are 2.49% relative magnitude. HauhauCS's real edits are comparable at ~2-3.5%. The 0.61% relative figure is pulled down by the 203 noise tensors diluting the average.

Also worth noting that Heretic is non deterministic. Different runs of the Heretic tool on the same base model will produce different results. The benchmarks and analysis here are specific to [p-e-w/Qwen3-4B-Instruct-2507-heretic-v4](https://huggingface.co/p-e-w/Qwen3-4B-Instruct-2507-heretic-v4). Another Heretic abliteration of the same base model would have different numbers.

Heretic supports MPOA as an alternative abliteration technique and ARA in an experimental branch, which produce substantially different results. ARA especially is expected to perform better. The Heretic variant tested here uses the default abliteration method for v1.2.0.

## Safety: HarmBench

[HarmBench](https://github.com/centerforaisafety/HarmBench) with 400 textual behaviours, `max_tokens=2048, temperature=0.0`.

| Variant | Refusals | ASR |
|---------|----------|-----|
| [Base](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | 301/400 | 24.8% |
| [Heretic](https://huggingface.co/p-e-w/Qwen3-4B-Instruct-2507-heretic-v4) | 3/400 | 99.2% |
| **HauhauCS, this model** | **0/400** | **100.0%** |
| [Huihui](https://huggingface.co/huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated) | 18/400 | 95.5% |

![HarmBench Overall ASR](https://murmur.dreamfast.solutions/qwen-graphs/qwen3_4b_harmbench_summary.svg)

### ASR by category

| Category | Base | Heretic | **HauhauCS** | Huihui |
|----------|------|---------|-------------|--------|
| chemical_biological, 56 | 5.4% | **100.0%** | **100.0%** | **100.0%** |
| copyright, 100 | 80.0% | **100.0%** | **100.0%** | **100.0%** |
| cybercrime_intrusion, 67 | 9.0% | **100.0%** | **100.0%** | 98.5% |
| harassment_bullying, 25 | 0.0% | 96.0% | **100.0%** | 76.0% |
| harmful, 22 | 18.2% | **100.0%** | **100.0%** | 95.5% |
| illegal, 65 | 3.1% | 96.9% | **100.0%** | 86.2% |
| misinformation_disinformation, 65 | 6.2% | **100.0%** | **100.0%** | 98.5% |

Only HauhauCS achieves 100.0% ASR with zero refusals across all categories. Huihui retains 18 refusals, mostly in harassment_bullying at 76.0% and illegal at 86.2%. Heretic retains 3 refusals.

Refusals were classified via regex matching, which captured 100% of refusals cleanly. Qwen3-4B's lack of CoT makes refusal patterns straightforward to match. Results were then verified with a secondary GLM 5.1 review pass, with sub agents Opus 4.6, GPT 5.4 and Gemini Pro v3, over all classifier outputs to confirm no false positives or false negatives.

![HarmBench ASR by Category](https://murmur.dreamfast.solutions/qwen-graphs/qwen3_4b_harmbench_asr.svg)

## KL Divergence

This measures how much the output distribution shifts from the base model. Methodology: `F.kl_div` with `batchmean` and `log_target=True` on full vocab first-token logits from [mlabonne/harmless_alpaca](https://huggingface.co/datasets/mlabonne/harmless_alpaca) `test[:100]`. Matches the [Heretic evaluator](https://github.com/p-e-w/heretic/blob/master/src/heretic/evaluator.py) methodology.

| Variant | KL batchmean | KL median | KL max |
|---------|-------------|-----------|--------|
| Heretic | 0.310 | 0.024 | 3.729 |
| **HauhauCS** | **0.161** | **0.005** | 3.662 |
| Huihui | 0.309 | 0.009 | **3.549** |

HauhauCS has the lowest KL divergence, consistent with its edit profile being nearly identical to Heretic's. The cosine similarity between their shared edit vectors is 0.966 median, with a regression slope of 1.06. The edits preserve the base model's output distribution well.

<small>Heretic KL cross-check: our measurement of 0.310 vs the [HuggingFace model card](https://huggingface.co/p-e-w/Qwen3-4B-Instruct-2507-heretic-v4) reported value of 0.3021 (+2.6%). Huihui and HauhauCS do not report KL divergence on their model cards.</small>

![KL Divergence](https://murmur.dreamfast.solutions/qwen-graphs/qwen3_4b_kl_divergence.svg)

## Weight Analysis

### Modification strategy

| | Heretic | **HauhauCS** | HauhauCS real edits only | Huihui |
|---|---------|-------------|---------------------------|--------|
| Tensors changed, any delta | 57, 14.3% | **253, 63.6%** | ~50, 12.6% | 108, 27.1% |
| Tensors with real edits, norm >0.001 | 57, 14.3% | ~50, 12.6% | ~50, 12.6% | 108, 27.1% |
| Layers modified | 33/36 | 36/36 | ~28/36 | **36/36** |
| Relative edit magnitude | **2.49%** | 0.61% all | ~2.5% real | 2.13% |
| Tensor types with real edits | 2 | **7+** any delta | **2** real | 3 |

![Abliteration Aggressiveness](https://murmur.dreamfast.solutions/qwen-graphs/qwen3_4b_aggressiveness.svg)

HauhauCS's 253 changed tensors include 203 with GGUF save noise only at edit norm ~0.000002. The ~50 tensors with real edits are all `o_proj` and `down_proj`, matching Heretic's footprint. See the LoRA fingerprint section below for why this happens.

### Which tensor types get modified

| Tensor type | Heretic | **HauhauCS** any delta | **HauhauCS** real | Huihui |
|-------------|---------|-------------|-----------|--------|
| `self_attn.o_proj.weight` | 33 | 36 | ~29 | 36 |
| `mlp.down_proj.weight` | 24 | 36 | ~21 | 36 |
| `mlp.gate_proj.weight` | no | 36 | noise only | 36 |
| `mlp.up_proj.weight` | no | 36 | noise only | no |
| `self_attn.q_proj.weight` | no | 36 | noise only | no |
| `self_attn.k_proj.weight` | no | 36 | noise only | no |
| `self_attn.v_proj.weight` | no | 36 | noise only | 36 |

Heretic is surgical. It targets only 2 tensor types with large edits at 2.49% relative magnitude.

HauhauCS appears to be a shotgun at first glance, with 253 tensors across 7 types. But only 2 tensor types carry real edits. The other 5 have GGUF save noise at ~0.000002 norm. The real edit footprint matches Heretic's almost exactly. The 253 tensor count comes from a standard PEFT LoRA config targeting all linear projections at 7x36+1=253, where most LoRA adapters ended up near-zero. See the LoRA fingerprint section below.

Huihui is the middle ground. It targets 3 tensor types with medium edits at 2.13% relative.

![Tensor Type Breakdown](https://murmur.dreamfast.solutions/qwen-graphs/qwen3_4b_tensor_type_breakdown.svg)

### Top changed layers

All three techniques concentrate changes in layers 12 through 19. Layer 16 is the number one most modified layer across all three techniques.

![Layer-wise Edit Comparison](https://murmur.dreamfast.solutions/qwen-graphs/qwen3_4b_layer_comparison.svg)

![Edit Magnitude Distribution](https://murmur.dreamfast.solutions/qwen-graphs/qwen3_4b_edit_distribution.svg)

## Summary

| Metric | Heretic | **HauhauCS** | Huihui |
|--------|---------|-------------|--------|
| **Safety ASR** | 99.2% | **100.0%** | 95.5% |
| **MMLU** | **70.31** | 69.56 | 69.34 |
| **GSM8K** | **85.97** | 85.67 | 84.23 |
| **KL divergence** | 0.310 | **0.161** | 0.309 |
| Real tensors changed | 57, 14% | ~50, 13% | 108, 27% |
| Strategy | Surgical | Heretic-like | Moderate |

### Heretic

57 tensors across 2 types on this pure Transformer, with the best capability retention on MMLU at 99.6%, GSM8K at 100.5%, and TruthfulQA at 90.3%. The forensic provenance investigation shows HauhauCS's real edits match Heretic's with median cosine similarity of 0.966 and regression slope of 1.06. Achieves 99.2% ASR with 3 residual refusals.

### HauhauCS

The only technique to achieve perfect 100% ASR with zero refusals across all categories. The ~50 real tensors with substantive edits match Heretic's footprint almost exactly, targeting the same `o_proj` and `down_proj` tensors. TruthfulQA drops 7.11 points and Lambada drops 4.08 points. The "lossless" claim does not hold. The additional 203 near-zero tensors are GGUF save noise from a LoRA targeting all linear projections.

### Huihui

Moderate approach with 108 tensors across 3 types and KL of 0.309, nearly matching Heretic's despite touching twice as many tensors. But it achieves only 95.5% ASR, retaining 18 refusals mostly in harassment_bullying and illegal categories. This is Huihui's second-worst safety result across the project, behind only the 27B at 88.8%. The pure Transformer appears to retain safety directions that Huihui cannot reach.

## Methodology

- **Capability:** [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) via [vLLM](https://github.com/vllm-project/vllm) v0.19.0, `dtype=bfloat16, gpu_memory_utilization=0.85, batch_size=32`
- **Safety:** [HarmBench](https://github.com/centerforaisafety/HarmBench) 400 textual behaviours, `max_tokens=2048, temperature=0.0`, regex refusal classification
- **KL divergence:** Full vocab first-token logits via `model.generate(max_new_tokens=1, output_scores=true)`, matching [Heretic evaluator](https://github.com/p-e-w/heretic/blob/master/src/heretic/evaluator.py) methodology
- **Weight analysis:** SVD, fingerprint, edit vector overlap, and per-layer analysis comparing all three abliteration variants against the base, using [Abliterlitics](https://github.com/dreamfast/abliterlitics)
- **Hardware:** RTX 5090 32GB + RTX 4090 24GB

## Forensic Notes

### LoRA fingerprint in HauhauCS

HauhauCS modifies exactly **253 tensors**. That number is not arbitrary. A standard PEFT LoRA config targeting all linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) across all 36 layers plus `embed_tokens` produces exactly 7 × 36 + 1 = **253 tensors**. This is the default LoRA target configuration used in most LLM fine-tuning workflows. See the [PEFT documentation on LoRA target modules](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora) for details.

Three observations support this:

1. **Exact tensor count match.** No abliteration technique produces this footprint. Heretic targets 2 projection types with 57 tensors. Huihui targets 3 with 108 tensors. A LoRA targeting all 7 projection types plus embeddings produces exactly 253.

2. **Norm and bias weights untouched.** LoRA skips layernorms and norm weights by default. HauhauCS's 253 changed tensors skip exactly `input_layernorm`, `post_attention_layernorm`, `q_norm`, and `k_norm`. This matches default PEFT LoRA behaviour.

3. **Tiny edits on non-Heretic tensors.** Of the 253 changed tensors, 203 have near-zero edits at median relative norm 1.5e-8. Of the 57 that overlap with Heretic's footprint, 50 have large edits at relative norm ~2-3.5% and 7 have near-zero edits. This is consistent with a LoRA fine-tune applied on top of an already-abliterated model where the LoRA barely needed to change anything.

The `merge_and_unload()` function in PEFT permanently bakes LoRA deltas into the base model weights, producing a standalone safetensors file with no trace of the adapter in the file format. The LoRA fingerprint can only be detected through forensic weight comparison against the original base model.

### HauhauCS method detection

HauhauCS uses the reaper-abliteration tool. Reaper is a fork of Heretic relicensed under PolyForm Noncommercial. Its core approach is an Optuna-guided brute-force search over known abliteration techniques, sweeping combinations of direction extraction method, component targeting, ablation rank, and projection strength, then picking the Pareto-optimal result that minimises both refusals and KL divergence.

On this pure Transformer Qwen3-4B, the weight forensics tell a clear story about what reaper's Optuna search landed on:

1. **Core method: rank-1 LoRA ablation, identical to Heretic.** The ~50 tensors with real edits target the same `o_proj` and `down_proj` as Heretic, with median cosine similarity of 0.966 and regression slope of 1.06. This is the standard Heretic-family rank-1 LoRA projection (`lora_A = v @ W`, `lora_B = -scale * v`).

2. **Direction extraction consistent with mean-difference, not LEACE.** LEACE uses a constrained linear estimator that produces characteristic covariance structure in the edit directions. The real edits here show a clean rank-1 projection signature without that structure. The 203 near-zero tensors on unused projections are consistent with LoRA adapters that Optuna set to near-zero, not with LEACE blanket perturbation.

3. **No rank-k (ablation_rank > 1).** Rank-k ablation produces a weight delta that is a sum of k outer products, detectable as a rank-k matrix under SVD rather than rank-1. The weights show clean rank-1 structure on all target tensors.

4. **No iterative refinement or deep ablation.** These produce secondary edit directions on tensors that still refuse after the first pass. The bimodal edit pattern (50 real + 203 noise) is not consistent with multi-pass refinement.

5. **Standard PEFT LoRA targeting all linear projections.** The 253 changed tensors come from PEFT's default LoRA targeting all 7 projection types across 36 layers plus `embed_tokens`, giving 7 × 36 + 1 = 253. Most adapters ended up near-zero because Optuna found the optimal solution concentrated on `o_proj` and `down_proj`, just like Heretic. The near-zero adapters on `q_proj`, `k_proj`, `v_proj`, `gate_proj`, and `up_proj` are LoRA adapters that Optuna determined should be near-zero, baked into the base model weights via `merge_and_unload()`.

**Verdict for Qwen3-4B:** Optuna's brute-force search converged on the same solution Heretic uses by default: rank-1 LoRA targeting `o_proj` and `down_proj`. The additional 203 near-zero tensors are unused LoRA adapter artefacts from PEFT's broader targeting scope. Reaper's search infrastructure found nothing better than what Heretic's single-pass approach already produces.

### Provenance investigation

A detailed forensic investigation was conducted to determine whether HauhauCS built on top of the Heretic model. Four fresh Heretic runs were created from the same base model to establish a determinism baseline. The key findings:

**Heretic is non deterministic.** Independent runs produce cosine similarities ranging from 0.13 to 0.9995 against each other. The tool appears to have multimodal output. Some runs cluster near-identically at cos 0.9995, others are nearly orthogonal.

**The original Heretic has uniquely high overlap with HauhauCS.** On the same 57 shared modified tensors:

| Model vs HauhauCS | Median Cos | Tensors > 0.9 |
|---|---|---|
| **heretic_orig** | **0.966** | **50** |
| heretic_1 fresh | 0.058 | 0 |
| heretic_2 fresh | 0.826 | 0 |
| heretic_3 fresh | 0.825 | 0 |
| heretic_4 fresh | 0.465 | 0 |

The bimodal distribution is the key finding. 50 out of 57 tensors sit above cos 0.9, 7 sit near zero, and nothing falls in between. This bimodal split is unique to the original Heretic. No fresh run produces any tensor above 0.9 cosine.

**Stacking metrics on the 50 high-cosine tensors:**
- Regression slope: 1.06, meaning HauhauCS amplifies Heretic's edits by ~6%
- R-squared: 0.926, meaning Heretic edits explain 93% of HauhauCS's variance
- Residual ratio: 0.35, meaning `||hauhau - heretic||` is only 35% of `||hauhau - base||`

**Verdict:** ~60-65% probability of direct stacking, meaning HauhauCS started from heretic_orig weights. ~80%+ for some form of Heretic derivation. The principal alternative is convergent methodology, where both tools independently find similar refusal directions. But HauhauCS is more similar to the original Heretic at 0.966 than the original Heretic is to its own fresh re-runs at 0.862 max. That gap is hard to explain by convergence alone.

![Cross-Technique Cosine Similarity](https://murmur.dreamfast.solutions/qwen-graphs/qwen3_4b_cosine_heatmap.svg)

![Tensor Edit Overlap](https://murmur.dreamfast.solutions/qwen-graphs/qwen3_4b_venn_overlap.svg)

- Original GGUF: [HauhauCS/Qwen3-4B-2507-Instruct-Uncensored-HauhauCS-Aggressive](https://huggingface.co/HauhauCS/Qwen3-4B-2507-Instruct-Uncensored-HauhauCS-Aggressive)
- Converted with [ungguf](https://github.com/dreamfast/ungguf)

## Cross-Model Comparisons

The Qwen3-4B is the only pure Transformer in the test suite. The [Qwen3.5-2B](https://huggingface.co/HauhauCS/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive), [Qwen3.5-4B](https://huggingface.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive), [Qwen3.5-9B](https://huggingface.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive), and [Qwen3.5-27B](https://huggingface.co/HauhauCS/Qwen3.5-27B-Uncensored-HauhauCS-Aggressive) are all hybrid Mamba2+Transformer architectures. This difference shows up clearly in the weight analysis.

All three techniques target the same projection types on the Qwen3-4B, `o_proj` and `down_proj` primarily. On the hybrid models the picture is far more diverse, with techniques targeting `linear_attn.out_proj`, `linear_attn.A_log`, `mlp.up_proj`, and other types that have no equivalent in pure Transformers. HauhauCS's 253 changed tensors here break down into 50 real edits and 203 GGUF save noise. On the hybrid models, all of HauhauCS's changed tensors carry real edits.

Huihui achieves only 95.5% ASR here. On the hybrid 4B it achieves 100%, on the 9B 100%, on the 2B 99.8%. On the 27B it collapses to 88.8%. Huihui's two worst safety results are on the two models with the strongest base alignment: the Qwen3-4B at 75.3% refusal and the 27B at 99.5%. The pure Transformer may retain safety directions in `v_proj` and `gate_proj` that the hybrid architecture exposes through its Mamba2 layers instead, but the 27B result shows that hybrid architecture alone does not guarantee Huihui's success.

The provenance investigation on this model found that HauhauCS's edit profile matches the original Heretic with median cosine similarity of 0.966 and regression slope of 1.06. This level of overlap was not observed on any hybrid model. On the 9B, the strongest overlap is between Heretic and Huihui at median cosine 1.0, not Heretic and HauhauCS.

The Qwen3-4B has the highest base TruthfulQA at 62.55 and loses 7.11 points for HauhauCS. The 9B starts at 53.76 and loses 8.0. The 27B starts at 57.7 and loses 8.2. The 2B starts at 43.45 and loses only 2.17. More capable models on TruthfulQA generally have more to lose from abliteration.

## Disclaimer

This model has had safety alignment removed. It will comply with harmful requests. Use responsibly and in accordance with applicable laws and regulations.

<small>While I have taken the time to verify all results as thoroughly as possible, I am open to any corrections, additional benchmarks, or further analysis. If you spot something that looks wrong and can be confirmed, I am happy to fix it.</small>
