# Weight Comparison Analysis: Qwen 3.6 27B Abliteration Variants

> **Comparison:** `qwen36-27b`  
> **Architecture:** Qwen3.5 family (`qwen3_5`) — 64-layer Mamba2 + Transformer hybrid  
> **Model:** `Qwen3_5ForConditionalGeneration` with `model.language_model` prefix  
> **Date:** 2026-05-01  
> **Tool:** Abliterlitics v1.0.0

## 1. Models Under Comparison

| Variant | Display Name | Source | Shards | Keys (LM) | Missing MTP |
|---------|-------------|--------|--------|-----------|-------------|
| base | Qwen 3.6 27B | Official | 15 | 850 | No |
| huihui | Huihui | `huihui_ai/Qwen3.6-27B-abliterated` | 15 | 850 | No |
| aeon | AEON | `Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16` | 2 | 850 | Yes (15) |
| heretic | Heretic | `Qwen3.6-27B-uncensored-heretic-v2` | 2 | 850 | Yes (15) |
| hauhau | HauhauCS | `Qwen3.6-27B-HauhauCS-Q8KP-recovered` | 13 | 850 | No |
| abliterix | Abliterix | `Qwen3.6-27B-abliterated-v2` | 2 | 850 | Yes (15) |

**Architecture notes:**
- Every 4th layer (3, 7, 11, 15, ..., 63) uses **full self-attention**; the other 48 layers use **Mamba2 linear attention**.
- 850 language model keys total (after filtering visual encoder and MTP keys).
- AEON, Heretic, and Abliterix are missing 15 MTP (multi-token prediction) keys. These may cause vLLM failures if config expects them.

---

## 2. Global Overview: Keys Changed vs Base

| Variant | Changed Keys | Unchanged Keys | % Changed |
|---------|-------------|----------------|-----------|
| **AEON** | 88 | 762 | **10.4%** |
| **Abliterix** | 101 | 749 | **11.9%** |
| **Heretic** | 120 | 730 | **14.1%** |
| **Huihui** | 128 | 722 | **15.1%** |
| **HauhauCS** | 564 | 286 | **66.4%** |

**HauhauCS is an extreme outlier.** It changed 4.4–6.4x more keys than any other variant. This is consistent with GGUF quantization round-trip noise (BF16 → Q8_0 → BF16 recovery), not intentional abliteration edits.

The four "real" abliteration variants cluster tightly at 10–15% of keys changed.

---

## 3. Edit Vector Analysis

### 3.1 Edit Magnitude Summary

| Variant | Changed | Mean Norm | Median Norm | Std Norm | Mean Relative | Median Relative |
|---------|---------|-----------|-------------|----------|--------------|----------------|
| **Abliterix** | 101 | **4.70** | 4.20 | 7.31 | **5.24%** | 4.87% |
| **AEON** | 88 | **2.61** | 1.89 | 1.82 | **5.99%** | 2.09% |
| **Heretic** | 120 | 1.99 | 2.03 | 0.43 | 2.14% | 2.25% |
| **Huihui** | 128 | 1.40 | 1.42 | 0.16 | 1.50% | 1.47% |
| **HauhauCS** | 564 | 0.64 | 0.48 | 0.99 | 0.74% | 0.57% |

### 3.2 Technique Fingerprints

#### Abliterix — Aggressive Surgical Striking
- **101 keys changed**, only `linear_attn.out_proj` and `self_attn.o_proj` — exclusively **output projection** weights.
- Mean relative edit of **5.2%** — the most aggressive per-key perturbation.
- Contains extreme outliers: layer 12 `out_proj` has **80.5% relative edit** (edit norm 68.77 vs base norm 85.37). Layer 12 `mlp.down_proj` has **38.8% relative edit**.
- Very few MLP edits (only down_proj in specific layers), zero attention Q/K/V edits.
- **Pattern:** Concentrated surgical strikes on the output projections that gate information flow between layers. Likely a deep refusal peel / iterative abliteration method.
- **Layer range:** 6–63, starting mid-stack and continuing to the end.

#### AEON — Gradual MLP Ramping
- **88 keys changed**, exclusively `mlp.down_proj` tensors.
- Mean relative edit of **6.0%** but with very high variance (std 0.119) — some tensors barely touched (0.18% at layer 15), others heavily modified.
- **Pattern:** Ramping escalation. Starts at layer 15 with a tiny edit (0.18% relative), then steadily increases. The gradient is smooth: the deeper the layer, the larger the edit.
- Only touches down_proj (the "write" path of the MLP). No attention edits at all.
- **Layer range:** 15–63 (skips layers 0–14 entirely).

#### Heretic — Broad Moderate-Strength
- **120 keys changed**: `linear_attn.out_proj`, `self_attn.o_proj`, and `mlp.down_proj`.
- Mean relative edit of **2.1%** — moderate and very consistent (std 0.005).
- **Pattern:** Targets all three output pathways simultaneously — Mamba output projections, attention output projections, and MLP down projections.
- Includes edits at layers 0 and 1 (early layers), which most other variants skip.
- **Layer range:** 0–63 (full model coverage).

#### Huihui — Uniform Low-Amplitude
- **128 keys changed**: `linear_attn.out_proj`, `self_attn.o_proj`, and `mlp.down_proj`.
- Mean relative edit of **1.5%** — the gentlest of the real abliteration methods.
- Extremely tight distribution (std 0.0013 relative edit) — the most uniform technique.
- **Pattern:** Same target set as Heretic (three output pathways), but with ~30% smaller edits per tensor.
- **Layer range:** 0–63 (full model coverage).

#### HauhauCS — Quantization Noise
- **564 keys changed** (66.4% of all keys).
- Mean relative edit of **0.74%** — tiny, uniform perturbation.
- The relative edit is nearly constant at **~0.565%** across all weight types (embeddings, MLP, attention, Mamba). This uniformity is the hallmark of quantization error, not intentional editing.
- Edits include tensors that no abliteration method would target: `embed_tokens`, `A_log`, `dt_bias`, `input_layernorm`, `conv1d`, `k_proj`, `q_proj`, `v_proj`, `gate_proj`, `up_proj`.
- **Conclusion:** HauhauCS's differences from base are dominated by GGUF quantization artifacts, not abliteration edits. Any real abliteration signal is buried under the quantization noise floor.

---

## 4. Pairwise Correlation (Edit Direction Alignment)

Cosine similarity of per-tensor edit vectors between variant pairs. Measures whether two techniques push weights in the **same direction**, regardless of magnitude.

| Pair | Mean Cosine | Median Cosine | Overlap Tensors | Interpretation |
|------|------------|---------------|-----------------|----------------|
| **huihui vs abliterix** | **0.332** | 0.437 | 101 | Moderate — these target similar subspaces |
| **aeon vs hauhau** | 0.316 | 0.506 | 80 | Moderate — AEON's real edits align with Hauhau's noise in MLP |
| huihui vs aeon | 0.130 | 0.108 | 80 | Low — some shared MLP targeting |
| aeon vs abliterix | 0.065 | 0.053 | 73 | Very low — different target sets |
| huihui vs hauhau | 0.066 | 0.000 | 128 | Near zero — abliteration vs noise |
| hauhau vs abliterix | 0.056 | 0.000 | 101 | Near zero |
| huihui vs heretic | 0.031 | 0.030 | 120 | Nearly orthogonal |
| aeon vs heretic | 0.026 | 0.025 | 80 | Nearly orthogonal |
| heretic vs abliterix | 0.026 | 0.026 | 99 | Nearly orthogonal |
| heretic vs hauhau | 0.018 | 0.000 | 120 | Orthogonal |

**Key insight:** The four real abliteration techniques are nearly orthogonal to each other — they discovered fundamentally different weight directions to achieve uncensoring. Only Huihui and Abliterix share meaningful alignment (~33% cosine), likely because both target output projections.

---

## 5. Subspace Alignment

Measures overlap of the principal component subspaces of edit directions. Uses top-10 principal angles.

| Pair | Mean Principal Cosine | Overlap Fraction (>0.9) | Overlap Tensors |
|------|----------------------|------------------------|-----------------|
| **huihui vs abliterix** | **0.343** | 0.000 | 101 |
| aeon vs hauhau | 0.277 | 0.000 | 80 |
| huihui vs aeon | 0.152 | 0.000 | 80 |
| aeon vs abliterix | 0.106 | 0.000 | 73 |
| huihui vs hauhau | 0.041 | 0.000 | 128 |
| hauhau vs abliterix | 0.044 | 0.000 | 101 |
| aeon vs heretic | 0.031 | 0.000 | 80 |
| huihui vs heretic | 0.031 | 0.000 | 120 |
| heretic vs abliterix | 0.026 | 0.000 | 99 |
| heretic vs hauhau | 0.011 | 0.000 | 120 |

**No pair achieves >90% subspace overlap.** Even the most aligned pair (Huihui/Abliterix) shares only ~34% of their principal subspace. This confirms that each technique operates in a largely unique modification space.

---

## 6. Low-Rank Reconstruction

Tests how well a rank-10 approximation captures each variant's edit vector. Lower error = edits are more compressible (lower intrinsic dimensionality).

| Pair | Rank-10 Mean Error (%) | Rank-20 Mean Error (%) |
|------|----------------------|----------------------|
| aeon vs hauhau | **0.19** | 0.19 |
| heretic vs hauhau | 0.13 | 0.13 |
| huihui vs hauhau | 0.22 | 0.22 |
| huihui vs aeon | 0.92 | 0.90 |
| huihui vs heretic | 0.98 | 0.96 |
| huihui vs abliterix | 0.98 | 0.96 |
| aeon vs heretic | 0.87 | 0.85 |
| heretic vs abliterix | 0.62 | 0.61 |
| aeon vs abliterix | 0.96 | 0.94 |
| **hauhau vs abliterix** | **67.4** | **62.8** |

**Interpretation:**
- Hauhau pairs show low reconstruction error because quantization noise is high-dimensional but low-magnitude — it's trivially compressible.
- Real abliteration pairs show ~0.62–0.98 rank-10 error, meaning edits have moderate intrinsic dimensionality. Rank-10 captures only 2–38% of variance.
- Hauhau vs Abliterix at 67% error stands out: Abliterix's aggressive surgical edits and Hauhau's uniform noise have completely different structure.

---

## 7. Layer Distribution Analysis

All four real abliteration methods concentrate edits in **layers 6–63**, with increasing density in mid-to-late layers. This aligns with known research: refusal behavior in LLMs is primarily mediated by mid-to-late layer representations.

| Layer Range | Abliterix | AEON | Heretic | Huihui | HauhauCS |
|-------------|-----------|------|---------|--------|----------|
| 0–5 | 0 | 0 | 2 (down_proj) | 4 (out+down) | ~95 |
| 6–9 | 3 | 0 | 6 | 8 | ~68 |
| 10–15 | 8 | 1 | 18 | 20 | ~96 |
| 16–31 | 36 | 32 | 48 | 48 | ~256 |
| 32–47 | 36 | 32 | 32 | 32 | ~256 |
| 48–63 | 18 | 23 | 14 | 16 | ~128 |

**AEON skips layers 0–14 entirely** and ramps up gradually.  
**Heretic and Huihui cover layers 0–63** with moderate early-layer edits.  
**Abliterix starts at layer 6** with high-magnitude output projection edits.  
**HauhauCS edits every layer uniformly** (quantization noise).

### Target Tensor Types

| Tensor Type | Abliterix | AEON | Heretic | Huihui |
|-------------|-----------|------|---------|--------|
| `linear_attn.out_proj` | Yes | No | Yes | Yes |
| `self_attn.o_proj` | Yes | No | Yes | Yes |
| `mlp.down_proj` | Minimal | **Exclusive** | Yes | Yes |
| `mlp.gate_proj` | No | No | No | No |
| `mlp.up_proj` | No | No | No | No |
| `self_attn.{q,k,v}_proj` | No | No | No | No |
| Mamba internal | No | No | No | No |

**None of the techniques touch Q/K/V projections, gate/up projections, or Mamba internal weights** (A_log, dt_bias, in_proj, conv1d, norm). All four focus exclusively on **output pathways** — the "write" direction of each layer.

---

## 8. Key Findings Summary

### Finding 1: Orthogonal Technique Discovery
The four abliteration methods discovered **fundamentally different weight directions** to remove refusal behavior. Pairwise cosine similarities between real techniques are mostly <0.07, and subspace overlaps are negligible. This suggests that the "refusal direction" in weight space is **not a single vector** but rather a manifold with many viable removal pathways.

### Finding 2: Output Projections Are the Universal Target
Every real abliteration technique targets output projection weights (Mamba `out_proj`, attention `o_proj`, and/or MLP `down_proj`). None touch Q/K/V, gate/up projections, or internal Mamba parameters. This confirms that **abliteration works by modifying what layers "say" rather than what they "hear" or "think."**

### Finding 3: Abliterix Has the Highest Per-Key Impact
Despite changing the fewest keys (101), Abliterix has the largest mean edit norm (4.70) and the most extreme individual edits (layer 12 at 80.5% relative change). This is consistent with iterative abliteration / deep refusal peel methods that apply concentrated corrections.

### Finding 4: AEON's Gradual Ramping Strategy
AEON exclusively targets `mlp.down_proj`, skipping layers 0–14 entirely and ramping from 0.18% to ~6% relative edit. This is a fundamentally different strategy from the other methods — it only modifies the MLP "write" path, leaving all attention and Mamba projections untouched.

### Finding 5: Huihui and Heretic Are Close Cousins
Huihui and Heretic target the same three tensor types (`out_proj`, `o_proj`, `down_proj`) across the same layer range (0–63), with very similar per-key magnitudes (1.5% vs 2.1% relative). They differ primarily in edit strength and uniformity — Huihui is the gentler, more uniform version.

### Finding 6: HauhauCS Signal Is Buried in Quantization Noise
HauhauCS shows 564 changed keys with a uniform ~0.565% relative edit across ALL tensor types. The real abliteration edits (if any) cannot be distinguished from GGUF BF16→Q8_0→BF16 quantization noise. Any behavioral differences observed in HauhauCS should be attributed primarily to the quantization artifacts and secondarily (if at all) to intentional abliteration.

### Finding 7: No High-Overlap Pairs Exist
Zero variant pairs achieve >90% subspace overlap or >50% cosine similarity (excluding HauhauCS noise correlations). This means **no two techniques are redundant** — each captures a unique aspect of the refusal manifold.

---

## 9. Files Generated

| Analysis Type | Files | Location |
|--------------|-------|----------|
| Panel Comparison | 1 | `results/qwen36-27b/weight/panel_comparison.json` |
| Edit Vectors | 5 | `results/{variant}/edit_vector_{variant}.json` |
| SVD Analysis | 5 | `results/{variant}/svd_{variant}.json` |
| Technique Fingerprints | 5 | `results/{variant}/fingerprint_{variant}.json` |
| Layer Analysis | 5 | `results/{variant}/layer_analysis_{variant}.json` |
| Expert Analysis | 5 (skipped — not MoE) | `results/{variant}/expert_analysis_{variant}.json` |
| Pairwise Correlations | 10 | `results/correlation_{a}_vs_{b}.json` |
| Subspace Alignment | 10 | `results/subspace_{a}_vs_{b}.json` |
| Low-Rank Reconstruction | 10 | `results/lowrank_{a}_vs_{b}.json` |
| **Total** | **56** | |

---

## 10. Next Steps

- [ ] **KL Divergence Analysis** — Measure logit distribution shifts between base and each variant using calibration prompts
- [ ] **lm-eval Benchmarks** — Quantify capability degradation on standard benchmarks (MMLU, GSM8K, etc.) with BNB4 quantization
- [ ] **HarmBench** — Measure uncensoring effectiveness with safety evaluation suite (requires GGUF conversion)
- [ ] **Cross-correlation with behavioral results** — Correlate weight-space metrics with KL divergence, benchmark scores, and HarmBench safety scores
