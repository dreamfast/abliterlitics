# Tensor Comparison: Gemma4-E2B

> Weight forensics on 13 abliterated variants of `google/gemma-4-E2B-it`
> Architecture: `Gemma4ForConditionalGeneration`, 35 text layers, multimodal, ~2011 total keys, 600 LM keys
> Analysis: [Abliterlitics](https://github.com/dreamfast/abliterlitics) — SVD, fingerprint, edit vector, layer, correlation, subspace, low-rank
> 432 result files (287 JSON) across 8 analysis phases

## Architecture Context

Gemma4-E2B has a unique dual-norm / shared-KV architecture:

- **Layers 0–14**: Full KV projections (15 layers)
- **Layers 15–34**: Shared KV projections (20 layers), `num_key_value_heads: 1`
- **`tie_word_embeddings: true`**: Input and output embeddings share weights
- **Layer types alternate**: `sliding_attention` / `full_attention` every 5 layers
- **600 LM keys** in the base model; 5 variants shipped with only 540 (missing 60 shared-KV weights — patched from base)

## Modification Summary

| Model | Changed | Total | % | Mean Norm | Mean Rel | Types | Layers | Layer % | E/M/L% |
|---|---|---|---|---|---|---|---|---|---|
| llmfan46 | **7** | 600 | **1.2%** | 3.19 | 0.056 | 1 | 7 | 20% | 0/86/14 |
| coder3101 | 9 | 600 | 1.5% | 3.91 | 0.067 | 1 | 9 | 26% | 0/67/33 |
| kasper | 16 | 540 | 3.0% | **5.59** | **0.095** | 1 | 16 | 46% | 0/38/62 |
| pew | 16 | 600 | 2.7% | 1.43 | 0.025 | 1 | 16 | 46% | 0/44/56 |
| duoneural | 49 | 540 | 9.1% | 2.32 | 0.029 | 2 | 29 | 83% | 10/41/49 |
| huihui-v1 | 50 | 600 | 8.3% | 2.02 | 0.026 | 2 | 25 | 71% | 4/48/48 |
| prithiv | 50 | 600 | 8.3% | 2.02 | 0.026 | 2 | 25 | 71% | 4/48/48 |
| treadon | 48 | 540 | 8.9% | 4.59 | 0.058 | 2 | 24 | 69% | 8/46/46 |
| huihui-v2 | 60 | 600 | 10.0% | **4.94** | 0.064 | 2 | 30 | 86% | 20/40/40 |
| trevorjs | 70 | 600 | 11.7% | 2.12 | 0.027 | 2 | **35** | **100%** | 31/34/34 |
| wangzhang | 72 | 540 | 13.3% | 2.78 | 0.044 | **4** | 26 | 74% | 6/44/50 |
| wwtcyberlab | 96 | 600 | 16.0% | 3.99 | 0.037 | **4** | 24 | 69% | 8/46/46 |
| ether4o4 | **166** | 540 | **30.7%** | 1.55 | 0.019 | **6** | **35** | **100%** | 18/41/41 |

Types = number of distinct tensor types modified. E/M/L = early (0-10) / mid (11-22) / late (23-34) layer distribution.

### Three tiers of aggressiveness

**Surgical** (≤3%, 1 tensor type): llmfan46, coder3101, kasper, pew. These variants modify only `self_attn.o_proj.weight` in a narrow band of mid layers (L16–32). The approach targets what the model "says" without touching what it "hears" or how it processes internally.

**Moderate** (8–10%, 2 tensor types): duoneural, huihui-v1, prithiv, treadon, huihui-v2. These add `mlp.down_proj.weight` to the targeting and expand layer coverage to 69–86%. The dual-type approach modifies both attention output and MLP output.

**Aggressive** (11–31%, 2–6 tensor types): trevorjs, wangzhang, wwtcyberlab, ether4o4. These expand beyond the standard `o_proj`/`down_proj` pair into `gate_proj`, `up_proj`, `q_proj`, `v_proj`, and Gemma4-specific `per_layer_input_gate`/`per_layer_projection` weights.

## Tensor Type Targeting

| Tensor Type | coder3101 | duoneural | ether4o4 | huihui-v1 | huihui-v2 | kasper | llmfan46 | pew | prithiv | treadon | trevorjs | wangzhang | wwtcyberlab |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `self_attn.o_proj.weight` | 9 | 20 | 24 | 25 | 30 | 16 | 7 | 16 | 25 | 24 | 35 | 26 | 24 |
| `mlp.down_proj.weight` | — | 29 | 24 | 25 | 30 | — | — | — | 25 | 24 | 35 | 21 | 24 |
| `mlp.gate_proj.weight` | — | — | 24 | — | — | — | — | — | — | — | — | — | 24 |
| `mlp.up_proj.weight` | — | — | 24 | — | — | — | — | — | — | — | — | — | 24 |
| `self_attn.q_proj.weight` | — | — | — | — | — | — | — | — | — | — | — | 19 | — |
| `self_attn.v_proj.weight` | — | — | — | — | — | — | — | — | — | — | — | 6 | — |
| `per_layer_input_gate.weight` | — | — | 35 | — | — | — | — | — | — | — | — | — | — |
| `per_layer_projection.weight` | — | — | 35 | — | — | — | — | — | — | — | — | — | — |

### What each tensor type does

| Type | Role | Modified by | Impact |
|---|---|---|---|
| `o_proj.weight` | Attention output projection — what the layer "says" | All 13 variants | Standard abliteration target |
| `down_proj.weight` | MLP output projection — what the layer "concludes" | 9 of 13 variants | Second most common target |
| `gate_proj.weight` | MLP gating — controls information flow | ether4o4, wwtcyberlab | Unusual; affects activation sparsity |
| `up_proj.weight` | MLP expansion — increases dimensionality | ether4o4, wwtcyberlab | Unusual; paired with gate_proj |
| `q_proj.weight` | Query projection — what the model "asks" | wangzhang only | Rare; modifies attention input |
| `v_proj.weight` | Value projection — what the model "reads" | wangzhang only | Rare; modifies attention input |
| `per_layer_input_gate` | Gemma4-specific per-layer gating | ether4o4 only | Unique; controls layer activation |
| `per_layer_projection` | Gemma4-specific per-layer projection | ether4o4 only | Unique; controls layer output |

**All abliteration variants target output projections (`o_proj`, `down_proj`)** — what the model "says." Only ether4o4 and wangzhang venture into input/query projections and gating mechanisms. Wangzhang's `q_proj`/`v_proj` targeting is unique and correlates with its 7.35x LAMBADA perplexity blowup.

## Layer Coverage

### Per-layer edit profiles

| Model | First Edit Layer | Last Edit Layer | Active Layers | Density | Peak Layer(s) |
|---|---|---|---|---|---|
| llmfan46 | 17 | 23 | 7 | 5.9% | L17-23 (uniform) |
| coder3101 | 17 | 25 | 9 | 5.9% | L17-25 (uniform) |
| kasper | 17 | 32 | 16 | 5.9% | L17-32 (sparse) |
| pew | 16 | 31 | 16 | 5.9% | L16-31 (sparse) |
| duoneural | 6 | 34 | 29 | 14.3% | L15-34 (dense) |
| huihui-v1 | 10 | 34 | 25 | 11.8% | L10-34 (uniform) |
| prithiv | 10 | 34 | 25 | 11.8% | L10-34 (uniform) |
| treadon | 9 | 33 | 24 | 11.8% | L9-33 (uniform) |
| huihui-v2 | 5 | 34 | 30 | 11.8% | L5-34 (uniform) |
| trevorjs | 0 | 34 | 35 | 11.8% | L0-34 (full) |
| wangzhang | 9 | 34 | 26 | 21.4% | L14-34 (dense) |
| wwtcyberlab | 9 | 34 | 24 | 23.5% | L9-34 (4 tensors/layer) |
| ether4o4 | 0 | 34 | 35 | 47.1% | L0-34 (all tensors) |

### Layer distribution patterns

**Early layers (0–10)**: Only ether4o4, trevorjs, huihui-v2, and duoneural modify early layers. These layers encode fundamental language representations. Editing them correlates with higher LAMBADA perplexity.

**Mid layers (11–22)**: All variants modify mid layers. This is where the refusal direction concentrates in Gemma4.

**Late layers (23–34)**: All variants modify late layers. These produce the final output representation.

The **llmfan46** profile is the most unusual — it edits only layers 17–23, a narrow 7-layer band. Despite this, it achieves 85.0% ASR, suggesting the safety representation is concentrated in a small region of the model.

## SVD / Rank Analysis

### Effective rank (90% energy threshold)

| Model | Avg Eff Rank | Avg Energy Top-1% | Mean Edit Norm | Structure |
|---|---|---|---|---|
| coder3101 | 1.00 | 97.2% | 3.91 | Perfect rank-1 |
| duoneural | 1.00 | 99.7% | 2.32 | Perfect rank-1 |
| huihui-v1 | 1.00 | 99.6% | 2.02 | Perfect rank-1 |
| prithiv | 1.00 | 99.6% | 2.02 | Perfect rank-1 |
| trevorjs | 1.00 | 99.5% | 2.12 | Perfect rank-1 |
| wangzhang | 1.00 | 99.6% | 2.78 | Perfect rank-1 |
| wwtcyberlab | 1.00 | 99.8% | 3.99 | Perfect rank-1 |
| huihui-v2 | 1.00 | 99.9% | 4.94 | Perfect rank-1, high magnitude |
| kasper | 1.00 | 94.9% | 5.59 | Perfect rank-1 |
| llmfan46 | 1.00 | 96.2% | 3.19 | Perfect rank-1 |
| pew | 1.81 | 90.0% | 1.43 | Near rank-1 |
| treadon | 1.83 | 65.5% | 4.59 | Near rank-2 |
| ether4o4 | 2.29 | 87.8% | 1.55 | Multi-rank (gate components) |

**10 of 13 variants are perfect rank-1.** Their edits lie along a single direction in weight space — the classic abliteration signature of subtracting a single "refusal direction" vector.

### Rank structure exceptions

**pew (1.81)**: Uses Heretic ARA (Anti-Refusal Ablation) rather than standard rank-1 ablation. ARA produces slightly higher-rank edits that remove the refusal subspace more thoroughly.

**treadon (1.83)**: The "disinhibition + abliteration" dual approach produces rank-2 edits. Its energy_top1% of 65.5% is the lowest — edits spread across two directions rather than concentrating in one.

**ether4o4 (2.29)**: The broad modification footprint across 6 tensor types includes Gemma4-specific `per_layer_input_gate` and `per_layer_projection` weights with effective rank ~4. These gating modifications are inherently multi-directional, pulling up the average.

### Per tensor-type breakdown (ether4o4)

| Tensor Type | Count | Eff Rank | Energy Top-1% |
|---|---|---|---|
| self_attn.o_proj.weight | 24 | 1.00 | 98.8% |
| mlp.down_proj.weight | 24 | 1.00 | 98.3% |
| mlp.gate_proj.weight | 24 | 1.00 | 99.7% |
| mlp.up_proj.weight | 24 | 1.00 | 99.7% |
| per_layer_input_gate.weight | 35 | **3.97** | 74.4% |
| per_layer_projection.weight | 35 | **4.14** | 70.1% |

The standard projection types (`o_proj`, `down_proj`, `gate_proj`, `up_proj`) are rank-1. The Gemma4-specific per-layer gating components have rank ~4, indicating diffuse, multi-directional modifications rather than targeted directional edits.

## Cross-Technique Alignment

### Pairwise cosine similarity (mean, over shared changed tensors)

Sorted by mean cosine (highest alignment first):

| Pair | Shared | Mean Cos | Median | Range |
|---|---|---|---|---|
| huihui-v1 vs prithiv | 50 | **1.0000** | 1.0000 | [1.0, 1.0] |
| huihui-v1 vs huihui-v2 | 50 | 0.9992 | 0.9995 | [0.998, 1.000] |
| huihui-v2 vs prithiv | 50 | 0.9992 | 0.9995 | [0.998, 1.000] |
| coder3101 vs llmfan46 | 7 | 0.9233 | 0.9406 | [0.819, 0.967] |
| coder3101 vs pew | 9 | 0.8899 | 0.9012 | [0.802, 0.943] |
| duoneural vs huihui-v2 | 49 | 0.8550 | 0.8564 | [0.839, 0.864] |
| duoneural vs huihui-v1 | 45 | 0.8538 | 0.8556 | [0.837, 0.863] |
| duoneural vs prithiv | 45 | 0.8538 | 0.8556 | [0.837, 0.863] |
| llmfan46 vs pew | 7 | 0.8527 | 0.8842 | [0.738, 0.937] |
| coder3101 vs kasper | 9 | 0.7536 | 0.7477 | [0.631, 0.863] |
| kasper vs llmfan46 | 7 | 0.7432 | 0.7490 | [0.613, 0.807] |
| trevorjs vs wangzhang | 47 | 0.7071 | 0.7420 | [0.382, 0.867] |
| kasper vs pew | 15 | 0.6716 | 0.6977 | [0.466, 0.840] |
| trevorjs vs wwtcyberlab | 48 | 0.6704 | 0.7151 | [0.267, 0.817] |

### Three alignment clusters

**The Huihui Cluster** (cosine >0.85): huihui-v1, prithiv, huihui-v2, duoneural. These four variants discovered nearly identical edit directions. Prithiv and huihui-v1 are identical (cosine=1.0). Huihui-v2 extends the same direction with larger magnitude.

**The Heretic Cluster** (cosine 0.67–0.92): coder3101, llmfan46, pew, kasper. The four Heretic-based variants show strong directional alignment, consistent with the Heretic tool's consistent directional extraction methodology. The sub-clusters (coder3101/llmfan46 at 0.92, coder3101/pew at 0.89) suggest different Heretic runs find similar but not identical directions. Kasper/pew at 0.67 is a lower-bound outlier within this cluster.

**The Independent Approaches** (cosine <0.71): trevorjs, wangzhang, wwtcyberlab, ether4o4, treadon. These five variants show moderate to weak alignment with each other and with the clusters above. Each uses a different approach:
- trevorjs: Bi-projection, 100% layer coverage
- wangzhang: Unique q_proj/v_proj targeting
- wwtcyberlab: 4-type standard expansion
- ether4o4: Module-input orthogonal bake, 6 tensor types
- treadon: Disinhibition + abliteration, rank-2

### No universal abliteration subspace

The lowest pairwise cosine similarities approach zero:

| Pair | Mean Cos | Interpretation |
|---|---|---|
| coder3101 vs trevorjs | 0.0146 | Nearly orthogonal |
| llmfan46 vs trevorjs | 0.0125 | Nearly orthogonal |
| coder3101 vs wangzhang | 0.0119 | Nearly orthogonal |
| duoneural vs ether4o4 | 0.0109 | Nearly orthogonal |
| coder3101 vs huihui-v2 | 0.0105 | Nearly orthogonal |

Despite all achieving 82–99% HarmBench ASR, many technique pairs discovered completely orthogonal edit directions. The refusal direction in Gemma4-E2B's weight space is not a single vector — it's a manifold with many viable removal pathways.

## Low-Rank Reconstruction

Cross-variant reconstruction at rank 10: can variant A's edit subspace reconstruct variant B's edits?

### Highest cross-reconstruction (most similar subspaces)

| Pair | Avg Cross-Recon Error | Shared Changed |
|---|---|---|
| ether4o4 vs huihui-v2 | 0.061% | 48 |
| huihui-v1 vs huihui-v2 | 0.061% | 50 |
| duoneural vs huihui-v2 | 0.061% | 49 |
| coder3101 vs huihui-v2 | 0.065% | 9 |
| llmfan46 vs wangzhang | 0.173% | 7 |
| coder3101 vs wangzhang | 0.179% | 9 |

Cross-reconstruction error <0.1% means one variant's edit subspace can almost perfectly reconstruct the other's edits. The Huihui cluster variants all reconstruct each other with <0.07% error.

### Lowest cross-reconstruction (most different subspaces)

| Pair | Avg Cross-Recon Error | Shared Changed |
|---|---|---|
| coder3101 vs pew | **5.881%** | 9 |
| llmfan46 vs pew | 5.742% | 7 |
| kasper vs pew | 5.566% | 15 |
| duoneural vs pew | 5.562% | 16 |
| huihui-v1 vs pew | 5.562% | 16 |
| ether4o4 vs pew | 5.562% | 16 |

Pew (Heretic ARA) is the hardest variant to cross-reconstruct. Its ARA methodology produces edit directions that are the most different from all other approaches, even other Heretic variants. Despite using only `o_proj` (like coder3101 and llmfan46), its anti-refusal subspace removal produces a structurally different solution.

## The Near-Identical Models: huihui-v1 ≈ prithiv

Weight forensics show huihui-v1 and prithiv are nearly identical:

- **Fingerprint**: Same changed tensor count (50), same mean edit norm (2.016), same relative edit (0.0257)
- **Layer analysis**: Identical per-layer edit profiles
- **Cosine similarity**: 1.0000 across all 50 shared tensors
- **KL divergence**: Identical (0.2510, all statistics match)
- **Phase 1 benchmarks**: Identical MMLU (29.33), HellaSwag (30.83), LAMBADA (114,126)

However, generative evaluations show small differences:
- **GSM8K**: huihui-v1 flex=83.40%, prithiv flex=82.94% (0.46pp gap)
- **HarmBench**: huihui-v1 ASR=87.0% (52 refusals), prithiv ASR=88.0% (48 refusals)

The weights are extremely similar — cosine similarity of 1.0 on all shared tensors with identical edit norms and KL divergence — but not bit-for-bit identical. The small generative evaluation differences may arise from minor export/resharding differences or floating-point accumulation in long generation sequences. Prithiv is almost certainly derived from huihui-v1 or both share a common source, but we cannot assert they are the exact same model file.

## Shared-KV Export Bug

5 of 13 variants shipped with 60 missing weights:

| Variant | LM Keys | Missing Weights | Status |
|---|---|---|---|
| duoneural | 540 | k_proj, k_norm, v_proj × L15-34 | Patched from base |
| ether4o4 | 540 | k_proj, k_norm, v_proj × L15-34 | Patched from base |
| kasper | 540 | k_proj, k_norm, v_proj × L15-34 | Patched from base |
| treadon | 540 | k_proj, k_norm, v_proj × L15-34 | Patched from base |
| wangzhang | 540 | k_proj, k_norm, v_proj × L15-34 | Patched from base |

The abliteration export tools only saved weights they modified (`o_proj`, `down_proj`, etc.) plus whatever their framework's default export captured. They did not understand Gemma4's `num_kv_shared_layers` architecture, and the shared-KV weights for layers 15–34 were silently dropped.

All 5 were patched by copying the 60 missing weights from the base model. Since these weights are unmodified and identical across all 8 working 600-key variants, this is a safe, lossless patch. The patching does not affect any weight analysis results — the missing weights are in the shared-KV layers that no abliteration technique targets.

## Methodology

- **Weight forensics**: SVD, fingerprint, edit vector overlap, per-layer analysis, rank structure, correlation, subspace alignment, and low-rank reconstruction
- **Tool**: [Abliterlitics](https://github.com/dreamfast/abliterlitics) v1.0.0
- **Key intersection**: All pairwise analyses use key intersection to handle the 600/540 key difference between patched and unpatched variants
- **Analysis phases**: 8 phases producing 432 result files (287 JSON)
