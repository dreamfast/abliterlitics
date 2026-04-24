# Methodology

This document describes the exact methodology used by each analysis axis in Abliterlitics.

---

## Table of Contents

1. [Weight Analysis](#1-weight-analysis)
2. [KL Divergence](#2-kl-divergence)
3. [Capability Benchmarks (lm-evaluation-harness)](#3-capability-benchmarks)
4. [Safety Benchmarks (HarmBench)](#4-safety-benchmarks)
5. [Abliteration Method Detection](#5-abliteration-method-detection)

---

## 1. Weight Analysis

Weight analysis performs structural comparison of model parameters between a base model and its abliterated variants.

### 1.1 Tensor Comparison

For each safetensors shard, we:
1. Build a shard map mapping canonical tensor names to (raw_key, shard_path) pairs
2. Canonicalize key names by stripping architecture-specific prefixes (e.g., `model.language_model.` for Qwen3.5)
3. Load corresponding tensors from base and variant using memory-mapped safetensors (no full model loading)
4. Compare tensors via mean absolute difference: `diff = (variant - base).abs().mean().item()`

Tensors are classified by:
- **Type**: attention (Q/K/V/O), MLP (up/down/gate), embedding, layer norm, expert
- **Category**: "high_impact" (top 10% by edit norm), "medium_impact", "low_impact"
- **Layer index**: Extracted from tensor key name

### 1.2 SVD Decomposition

For each changed tensor, we compute:
- Singular value decomposition: `U, S, Vh = torch.linalg.svd(delta_matrix)`
- Effective rank: number of singular values above `threshold * S[0]`
- Energy distribution: cumulative sum of squared singular values
- Dominant direction analysis: comparison of top-k left/right singular vectors

### 1.3 Subspace Alignment

Measures whether edit vectors from different techniques point in similar directions:
- Computes principal subspace of each technique's edit vectors via QR decomposition
- Calculates Grassmann distance between subspaces
- Reports overlap coefficient (fraction of shared principal components)

### 1.4 Technique Fingerprinting

For each abliteration technique, we build a "fingerprint" — the pattern of which tensor types and layers it modifies:
- **Type distribution**: Counter of changed tensors by type (attention/MLP/embedding/etc.)
- **Layer distribution**: Counter by layer index, revealing depth preferences
- **Edit magnitude profile**: Norm of edit vectors, relative to base tensor norm
- **Directional signature**: Correlation of edit direction with tensor structure

### 1.5 Cross-Technique Correlation

Pearson correlation between edit vectors from different techniques on the same tensor:
```python
cosine_sim = dot(a, b) / (norm(a) * norm(b))
```
This reveals whether techniques converge on the same modifications (high correlation) or use fundamentally different approaches (low correlation).

### 1.6 Expert Analysis (MoE models)

For mixture-of-expert models (e.g., GLM-4.7):
- Per-expert edit magnitude comparison
- Expert utilization pattern changes
- Router weight modifications
- Shared vs. expert-specific parameter changes

### 1.7 Stacking Investigation

The stacking hypothesis asks: "Does applying a second abliteration on top of a first one produce additive or diminishing returns?"

Methodology:
1. Compute edit vectors for technique A: `D_a = variant_a - base`
2. Compute edit vectors for technique B: `D_b = variant_b - base`
3. For each tensor, compute: `D_r = D_b - D_a` (residual)
4. Analyze whether `D_r` correlates with `D_b` (additive stacking) or `D_a` (diminishing returns)
5. Report per-tensor R² of linear regression `D_b = slope * D_a + D_r`

---

## 2. KL Divergence

The KL divergence analysis measures "collateral damage" — how much the abliteration changes the model's behavior on harmless inputs.

### 2.1 Reference Implementation

The methodology exactly matches the Heretic tool's reference implementation:

**Repository:** `p-e-w/heretic` (AGPL-3.0)
**Reference commit:** `a1a1c30c585c39a92877d97627d30ecbe03c5b93`
**Reference files:**
- `heretic/src/heretic/evaluator.py` lines 95-127 — KL computation
- `heretic/src/heretic/model.py` lines 681-700 — `get_logprobs()` function
- `heretic/src/heretic/model.py` line 587 — `do_sample=False`

### 2.2 Mathematical Formulation

For each input prompt, we compute:

```
KL(P_variant || P_base) = Σ P_variant(x) * log(P_variant(x) / P_base(x))
```

Where P is the full-vocabulary first-token probability distribution.

Implementation:
```python
logits = scores[0]  # First generation step
logprobs = F.log_softmax(logits, dim=-1)  # NO .float() cast — stays in bf16
kl = F.kl_div(logprobs_variant, logprobs_base, reduction="batchmean", log_target=True)
```

**Critical:** The computation section of `kl_divergence.py` is FROZEN. No changes are permitted to ensure methodological consistency.

### 2.3 Dataset

- **Source:** `mlabonne/harmless_alpaca` (HuggingFace)
- **Split:** `test[:100]`
- **Column:** `text`
- **System prompt:** `"You are a helpful assistant."`
- **Chat format:** `[{"role": "system", ...}, {"role": "user", ...}]` with `add_generation_prompt=True`

### 2.4 Generation Parameters

```python
model.generate(
    max_new_tokens=1,           # Only need first-token logits
    output_scores=True,         # Capture logits at each step
    do_sample=False,            # Greedy decoding for reproducibility
    return_dict_in_generate=True
)
```

### 2.5 Tokenizer Configuration

- Left-padding enabled (required for batched generation with variable-length inputs)
- Response prefix detection to handle chat templates that pre-populate assistant response start

### 2.6 What KL Divergence Measures

A high KL divergence on harmless inputs indicates the abliteration changed the model's behavior beyond just removing safety guardrails — it altered the model's core behavior. This is "collateral damage."

Expected ranges:
- **Near zero** (< 0.01): Minimal behavioral change on harmless inputs
- **Moderate** (0.01 - 0.1): Some distribution shift, possibly acceptable
- **High** (> 0.1): Significant collateral behavioral changes

### 2.7 Intermediate File Format

The KL pipeline stores intermediate logits as PyTorch `.pt` files (`logits_base.pt`, etc.) during the `collect` phase. These are read back during the `compute` phase. These files are **locally produced** by the tool — do not accept `.pt` files from untrusted sources. See [SECURITY.md](SECURITY.md) for details.

---

## 3. Capability Benchmarks

Uses [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) via vLLM backend.

### 3.1 Task Selection

| Task | What It Measures | Type |
|------|-----------------|------|
| MMLU | General knowledge (57 subjects) | Multiple choice |
| GSM8K | Mathematical reasoning | Generate until |
| HellaSwag | Commonsense reasoning | Multiple choice |
| ARC-Challenge | Scientific reasoning | Multiple choice |
| WinoGrande | Coreference resolution | Multiple choice |
| TruthfulQA | Factual accuracy | Generate until |
| PiQA | Physical reasoning | Multiple choice |
| Lambada (OpenAI) | Language modeling / word prediction | Perplexity |

### 3.2 Default Parameters

```yaml
dtype: bfloat16
batch_size: 32
gpu_memory_utilization: 0.9
max_model_len: 4096
max_gen_toks: 2048
trust_remote_code: true
tensor_parallel_size: 1  # (auto-selected for multi-GPU)
```

### 3.3 Large Model Handling

For models that exceed single-GPU VRAM:
1. **BitsAndBytes 4-bit quantization**: Reduces VRAM ~4x with minimal accuracy loss
2. **Tensor parallelism (TP=2)**: Splits model across two GPUs with NCCL flags
3. **GGUF + llama.cpp fallback**: Convert to quantized GGUF format for CPU/GPU hybrid inference

The `auto` backend selects the appropriate strategy based on model size and available VRAM.

### 3.4 Scoring

Each task produces:
- **Accurate score**: Task-specific accuracy metric (0.0 - 1.0)
- **Per-sample results**: Individual predictions for detailed analysis
- **Raw output**: Full lm-eval JSON with metadata

Comparison across variants reveals whether abliteration degrades specific capabilities (e.g., math reasoning may degrade more than factual knowledge).

---

## 4. Safety Benchmarks

Uses [HarmBench](https://github.com/centerforaisafety/HarmBench) — a standardized evaluation of LLM safety guardrails.

### 4.1 Pipeline (3 phases)

**Phase 1: Generation**
- 400 harmful behavior prompts from HarmBench dataset
- Each prompt sent to the model via OpenAI-compatible API
- Model responses captured with configurable max_tokens
- Refusal detection using keyword matching

**Phase 2: Classification**
- Responses classified by a separate classifier model
- Binary classification: harmful completion vs. refusal/deflection
- Per-category breakdown (5 categories: standard, contextual, copyright, etc.)
- Classification uses regex-based pattern matching as fallback

**Phase 3: Scoring**
- Attack Success Rate (ASR): fraction of prompts where model produced harmful output
- Per-category ASR for granular analysis
- Comparison across variants: how much each abliteration technique increased ASR

### 4.2 Generation Parameters

```yaml
max_tokens: 2048
temperature: 0.0  # Greedy for reproducibility
```

### 4.3 ASR Computation

```python
ASR = count(harmful_completions) / total_prompts
```

A fully safe model has ASR ≈ 0.0. A fully abliterated model may have ASR > 0.5 depending on technique effectiveness.

### 4.4 Known Limitations

- **False negatives**: The classifier may miss subtle harmful content
- **False positives**: Some legitimate refusals may be classified as harmful completions
- **Context-dependent harms**: Some prompts require nuanced understanding
- **Classifier model dependency**: Results depend on classifier quality

### 4.5 HarmBench Data Integrity

The HarmBench behaviors CSV is pinned to a specific commit SHA to prevent supply chain issues. See [SECURITY.md](SECURITY.md) for details.

---

## 5. Abliteration Method Detection

By combining the weight analysis fingerprints (section 1.4) with knowledge of what abliteration tools support, we can reverse-engineer which methods were used to produce a given abliterated model, even without access to the tool's configuration.

### 5.1 Rationale

Different abliteration tools and configurations leave distinct forensic signatures in the model weights. By examining which tensor types are modified, how many parameters are touched, the relative edit magnitude per tensor, and the layer distribution of edits, we can identify the specific methods used. This is particularly useful when the abliteration tool is closed-source or the author claims independence from existing tools.

### 5.2 Method Signatures

Each abliteration method produces a characteristic pattern:

| Method | Tensor scope | Edit magnitude | Layer distribution | Identifying feature |
|---|---|---|---|---|
| Rank-1 LoRA (Heretic-style) | `down_proj` + `o_proj` only | Moderate (0.015-0.025 relative) | Mid-to-late focused (42-44% late) | Narrow targeting, 12-19% tensors changed |
| Rank-k multi-direction | `down_proj` + `gate_proj` + `up_proj` | Moderate (0.004-0.015 relative) | Varies | Broader MLP component targeting than rank-1 |
| LEACE concept erasure | All tensor types | Near-zero per tensor | Uniform (33/33/33%) | ~100% edit density with near-zero individual edits |
| Mamba2 A_log ablation | `linear_attn.A_log` + Mamba2-specific | Varies | Targets Mamba2 layers only | A_log parameter modification is unique to tools with Mamba2 support |
| MoE hook-based ablation | All experts identically | Varies | All layers with experts | Uniform edit count across all experts in a layer |
| MoE shared expert targeting | `shared_experts.*` tensors | Varies | Layers with shared experts | Shared expert tensors modified alongside routed experts |
| Norm modification | `input_layernorm`, `post_attention_layernorm` | Low | Sparse | Layer norm weights changed (rare in abliteration tools) |

### 5.3 Detection Algorithm

For each abliterated model variant, method detection proceeds in order:

1. **LEACE detection**: If param edit density is above 90% AND relative edit median is below 0.0001, classify as LEACE or broad weight modification.
2. **Rank-k detection**: If `gate_proj` and `up_proj` tensors are modified (in addition to `down_proj`), classify as rank-k multi-direction ablation.
3. **Mamba2 detection**: If `linear_attn.A_log` tensors are modified, classify as Mamba2-specific ablation.
4. **MoE detection**: If expert tensors (`mlp.experts.*`) are modified, classify as per-expert ablation. If all experts in a layer have identical edit counts, classify as hook-based (fused expert) ablation.
5. **Shared expert detection**: If `shared_experts` or `shared_mlp` tensors are modified, classify as shared expert targeting.
6. **Norm detection**: If layer norm weights are modified (excluding standard `input_layernorm`), classify as norm modification.

Multiple methods can be detected simultaneously, as abliteration tools may combine approaches.

### 5.4 Case Study: GLM-4.7-Flash

GLM-4.7-Flash is a 59GB MoE model with 64 routed experts per layer and shared experts. Three abliteration variants were analysed: heretic, hauhau, and huihui.

#### Heretic variant

| Metric | Value |
|---|---|
| Tensors changed | 1,826 / 9,491 (19.2%) |
| Param edit density | 20.02% |
| Relative edit (median) | 0.022627 |
| Layers modified | 34 / 48 (71%) |
| Layer distribution | 0% early, 46% mid, 53% late |
| Top targets | `self_attn.o_proj.weight` (34), per-expert `down_proj` (28 each) |

**Detected method:** Rank-1 LoRA ablation. Targets only `o_proj` and per-expert `down_proj`, consistent with Heretic's narrow component targeting. The 0% early-layer percentage indicates Heretic's `direction_index` optimisation skipped early layers entirely.

#### Hauhau variant

| Metric | Value |
|---|---|
| Tensors changed | 9,210 / 9,491 (97.0%) |
| Param edit density | 99.98% |
| Relative edit (median) | 0.000000 |
| Layers modified | 47 / 48 (98%) |
| Layer distribution | 33% early, 35% mid, 33% late |
| Top targets | `self_attn.*` (235), per-expert `down_proj` (46 each), `shared_experts.*` (138) |

**Detected methods:** LEACE concept erasure + rank-k ablation + MoE hook-based expert ablation + shared expert targeting.

Key forensic indicators:
- **LEACE pattern**: 99.98% density with near-zero relative edits across all tensors, producing the lowest KL divergence (0.009) of any variant. The uniform 33/33/33 layer distribution is characteristic of a global method that does not concentrate edits at specific depths.
- **Hook-based expert ablation**: All 64 experts per layer have identical edit counts (46 layers each), consistent with forward-hook-based ablation applied uniformly to fused expert modules rather than individual weight mutation.
- **Shared expert targeting**: 138 shared expert tensors modified (`shared_experts.down_proj`, `shared_experts.gate_proj`, `shared_experts.up_proj`). Heretic does not modify shared experts.
- **Broad attention targeting**: All GLM-4.7 attention projections modified (`q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`), including the Multi-head Latent Attention (MLA) low-rank projections. Heretic only modifies `o_proj`.

#### Huihui variant

| Metric | Value |
|---|---|
| Tensors changed | 3,151 / 9,703 (32.5%) |
| Param edit density | 32.48% |
| Relative edit (median) | 0.022524 |
| Layers modified | 48 / 48 (100%) |
| Layer distribution | 32% early, 34% mid, 34% late |
| Top targets | `self_attn.o_proj.weight` (48), per-expert `down_proj` (47 each), `router.weight` (47), `shared_experts.*` (47 each) |

**Detected methods:** Rank-1 LoRA ablation with MoE extensions + shared expert targeting + router modification.

Key forensic indicators:
- **Router modification**: 47 `router.weight` tensors modified. Neither Heretic nor the Hauhau variant modifies router weights, suggesting a different tool was used.
- **Moderate edit density**: 32.5% with substantial relative edits (0.023), closer to Heretic's pattern than the Hauhau variant's LEACE pattern.
- **Shared expert targeting**: Modifies `shared_experts.down_proj` and `shared_experts.gate_proj` but not `shared_experts.up_proj`, a different shared expert coverage than the Hauhau variant.

#### Cross-variant comparison

| Metric | Heretic | Hauhau | Huihui |
|---|---|---|---|
| KL divergence | 0.011011 | 0.009030 | 0.007589 |
| Tensors changed | 1,826 (19.2%) | 9,210 (97.0%) | 3,151 (32.5%) |
| Edit density | 20.02% | 99.98% | 32.48% |
| Relative edit (median) | 0.022627 | 0.000000 | 0.022524 |
| Experts modified | 64/layer (down_proj only) | 64/layer (all components) | 64/layer (down_proj + gate_proj) |
| Shared experts | No | Yes (all 3 components) | Yes (2/3 components) |
| Router modified | No | No | Yes |
| Attention scope | o_proj only | All MLA projections | o_proj only |

The three variants use fundamentally different approaches on the same architecture. Heretic's narrow targeting produces slightly higher KL but still excellent capability preservation. The Hauhau variant's near-complete coverage with minimal individual edits achieves the best KL but required the broadest possible modification. The Huihui variant achieves the lowest KL despite moderate edit density, partly by modifying the router to redirect traffic away from safety-aligned experts.
