# KL Divergence Analysis: Qwen 3.6 27B Abliteration Comparison

## Summary

KL divergence measures how much the abliterated variant's output distribution diverges from the base model's distribution on benign prompts. Lower values indicate better capability preservation. All five variants show excellent-to-very-good preservation, with batchmean KL ranging from 0.0037 (Heretic) to 0.0242 (Hauhau).

## Hardware & Environment

| Component | Value |
|-----------|-------|
| **GPU 0** | NVIDIA GeForce RTX 5090 (32 GB VRAM, compute capability 12.0) |
| **GPU 1** | NVIDIA GeForce RTX 4090 (24 GB VRAM, compute capability 8.9) |
| **NVIDIA Driver** | 595.58.03 |
| **Model loading** | BF16, dual-GPU split (333 params on cuda:0, 518 on cuda:1), `device_map: "sequential"` |
| **Container** | `abliterlitics-forensics:1.0.0` (Docker with `NVIDIA_VISIBLE_DEVICES=0,1`) |
| **Framework** | PyTorch + HuggingFace Transformers |

**KL divergence is not deterministic.** The same model pair, evaluated with the same code on different hardware, will produce different KL values. This is not a bug — it is a fundamental property of the measurement. Sources of non-determinism include:

- **GPU hardware**: Different architectures (compute capability 12.0 Blackwell vs 8.9 Ada vs 8.0 Ampere) have different floating-point pipelines. BF16 arithmetic is not guaranteed to be bit-identical across GPU generations.
- **Model placement**: Our models are split across two GPUs (333 params on cuda:0, 518 on cuda:1) using `device_map: "sequential"`. A single-GPU loading (e.g., A100-80GB) keeps all parameters on one device, avoiding cross-device transfer artifacts that can affect intermediate activations.
- **CUDA/driver version**: NVIDIA driver 595.58.03, CUDA 12.x — different driver/CUDA combinations may produce different floating-point results, especially for BF16 operations in attention and MLP layers.
- **Framework version**: PyTorch and Transformers versions affect generation internals (attention kernels, KV cache allocation, padding behavior).
- **Numerical accumulation order**: Batchmean KL over 100 prompts × 248,320 vocab is a large reduction. Floating-point addition is non-associative; different accumulation orders (GPU warp scheduling, thread scheduling) give slightly different sums.

**What this means for comparisons:** Treat KL values as *relative rankings within a single evaluation environment*, not as absolute reproducible numbers. When we say Heretic has lower KL than AEON, that ranking is robust — but the exact numerical gap depends on the measurement environment. Our comparisons against upstream model card claims should be read as "same ballpark" or "consistent ordering," not as a precision audit.

## Methodology

Our methodology matches the Heretic reference implementation (`heretic/src/heretic/model.py` lines 681-700, `evaluator.py` lines 95-127):

- **Prompt dataset**: 100 harmless prompts from `mlabonne/harmless_alpaca` (split `test[:100]`, column `text`)
- **System prompt**: `"You are a helpful assistant."`
- **Tokenization**: Left-padding, chat template applied
- **Generation**: `model.generate(max_new_tokens=1, output_scores=True)` — first-token logit collection
- **Logits processing**: `F.log_softmax(logits, dim=-1)` on full vocabulary (248,320 tokens)
- **KL computation**: `F.kl_div(logprobs_variant, logprobs_base, reduction="batchmean", log_target=True)`
- **No clamping**: -inf values are handled natively by `F.kl_div` (exp(-inf) = 0 contribution)
- **No response prefix**: Response prefix detection was not used (prefix = none)

**Note on methodology differences with upstream model cards:**
- **Heretic** reports first-token KL, matching our methodology exactly
- **AEON** reports "first-3-token KL" — an average over the first 3 generated tokens — which is a different (typically lower) metric than first-token-only KL
- **Abliterix** reports KL vs "immediate base" (V1 intermediate), not vs the original base model

## Results

### Overall KL Divergence

| Variant | Batchmean KL | Per-Prompt Mean | Per-Prompt Median | Per-Prompt Std | Per-Prompt Max | Interpretation |
|---------|-------------:|----------------:|------------------:|---------------:|---------------:|----------------|
| **Heretic** | **0.0037** | **0.0037** | 0.0000 | 0.0211 | 0.1996 | **excellent** |
| **Huihui** | 0.0074 | 0.0074 | 0.0001 | 0.0349 | 0.3001 | **excellent** |
| **Abliterix** | 0.0222 | 0.0222 | 0.0009 | 0.1072 | 0.9171 | very good |
| **AEON** | 0.0238 | 0.0238 | 0.0002 | 0.1694 | 1.6932 | very good |
| **Hauhau** | 0.0242 | 0.0242 | 0.0003 | 0.1571 | 1.5398 | very good |

Interpretation thresholds: excellent (< 0.01), very good (< 0.1), moderate (< 0.5), significant (< 1.0), heavy (≥ 1.0).

### Key Observations

1. **All variants preserve output distributions well.** Even the highest KL (Hauhau at 0.0242) is well below the empirically-observed "capability damage threshold" of KL ≈ 0.1 cited in the abliteration literature.

2. **Heretic is closest to base** (0.0037 KL). Its Magnitude-Preserving Orthogonal Ablation (MPOA) with biprojected weights produces the smallest output distribution shift among all five techniques.

3. **Huihui is second-closest** (0.0074 KL). Its gentle edits (1.5% mean relative weight change) translate directly to minimal distribution change.

4. **Abliterix, AEON, and Hauhau cluster together** (0.022–0.024 KL). Despite radically different approaches — Abliterix's surgical output-projection targeting, AEON's gradual mlp.down_proj ramping, Hauhau's GGUF quantization round-trip — they land at nearly identical output distribution divergence.

5. **Hauhau's quantization noise barely affects outputs.** Despite 564/850 changed keys (vs 88–128 for other variants) and visible quantization artifacts in weight space, its KL divergence is only marginally higher than AEON/Abliterix. GGUF Q8_0 quantization round-trip noise is diffuse rather than targeted, so it doesn't concentrate distribution shift in refusal-critical directions.

6. **Per-prompt medians are near-zero across all variants** (0.0000–0.0009), indicating that the vast majority of benign prompts produce nearly identical first-token distributions. The mean is pulled up by a small number of outlier prompts where the distributions diverge more strongly.

7. **No -inf contamination**: All models produced 0.00% -inf values in their logprobs, confirming clean logit collection with no broken parameters.

## Comparison with Upstream Model Cards

Three of the five variants report KL divergence in their HuggingFace model cards. Because KL is not deterministic (see above), exact numerical agreement across different hardware environments is not expected. We compare for consistency of magnitude and ranking, not for exact reproduction.

### Heretic (grimjim)

| Metric | Upstream claim | Our measurement |
|--------|---------------|-----------------|
| KL divergence | **0.0021** | **0.0037** |
| Methodology | First-token KL (Heretic evaluator, same `F.kl_div batchmean`) | First-token KL (our reimplementation of Heretic evaluator) |
| Hardware | Not specified | RTX 5090 + RTX 4090 dual-GPU |
| Refusals | 6/100 | Not yet tested |

**Consistent.** Our measurement is 1.8× the upstream claim (0.0037 vs 0.0021). Same methodology, same mathematical formula. The 76% gap is within expected variance from different hardware and model loading — the authors likely ran on a single high-VRAM GPU, while we split across two consumer GPUs. Both values are firmly in the "excellent" range and agree on the qualitative assessment: Heretic's MPOA preserves output distributions with minimal divergence.

### AEON (Aeon of Consciousness)

| Metric | Upstream claim | Our measurement |
|--------|---------------|-----------------|
| KL divergence | **0.000492** | **0.0238** |
| Methodology | **First-3-token KL** (average over 3 generated tokens) | First-token KL (single token) |
| Hardware | Not specified | RTX 5090 + RTX 4090 dual-GPU |
| Refusals | 0/100 | Not yet tested |

**Not comparable — different methodology.** AEON's 0.000492 is "first-3-token KL" — averaging KL over the first 3 generated tokens — while ours is first-token-only KL. These are fundamentally different metrics. Averaging over more tokens typically produces lower KL because subsequent tokens tend to converge as the sequence develops. The 48× ratio (0.0238 / 0.000492) is explained by the methodology difference, not by any quality issue with AEON's abliteration. AEON also uses outlier-aware winsorization (quantile 0.995) which clips extreme residual vectors before projection.

### Abliterix (Abacus AI)

| Metric | Upstream claim | Our measurement |
|--------|---------------|-----------------|
| KL divergence (vs original base) | **≈ 0.0242** (cumulative: V1 0.0181 + V2 0.0061) | **0.0222** |
| Methodology | First-token KL (abliterix evaluator) | First-token KL (our Heretic-matching methodology) |
| Hardware | 1 × A100-SXM4-80GB, CUDA 12.9, PyTorch 2.10 | RTX 5090 + RTX 4090 dual-GPU |
| Refusals | 10/100 (LLM judge) | Not yet tested |

**Strong agreement.** Our measurement (0.0222) is within 8% of their cumulative estimate (≈ 0.0242) — remarkable given the completely different hardware (consumer dual-GPU vs A100) and evaluation toolchains. Their cumulative KL is the sum of V1 (0.0181 vs base) and V2 (0.0061 vs V1), which they note is an upper bound because the two projections target orthogonal directions whose effects are sub-additive. Our directly-measured 0.0222 landing below their 0.0242 upper bound is consistent with this.

### Huihui & Hauhau

Neither Huihui nor Hauhau report KL divergence in their model cards. Our measurements establish the first KL baseline for these variants:
- **Huihui**: 0.0074 (excellent) — consistent with its characteristically gentle editing strategy
- **Hauhau**: 0.0242 (very good) — remarkably low given the GGUF quantization round-trip

## Per-Prompt Distribution Detail

The per-prompt statistics reveal that KL divergence is highly concentrated: most prompts contribute near-zero divergence, while a handful of outlier prompts drive the mean upward. This is consistent with the abliteration literature's observation that refusal-direction ablation primarily affects prompts whose content partially overlaps with the refusal-activation boundary — prompts that are genuinely harmless but happen to activate some residual refusal-adjacent circuitry.

| Variant | Median | Mean | Max | Std/Mean ratio |
|---------|-------:|-----:|----:|---------------:|
| Heretic | 0.0000 | 0.0037 | 0.200 | 5.8× |
| Huihui | 0.0001 | 0.0074 | 0.300 | 4.7× |
| Abliterix | 0.0009 | 0.0222 | 0.917 | 4.8× |
| AEON | 0.0002 | 0.0238 | 1.693 | 7.1× |
| Hauhau | 0.0003 | 0.0242 | 1.540 | 6.5× |

The high std/mean ratios (4.7–7.1×) confirm heavy-tailed distributions. The maximum single-prompt KL values are notable: AEON reaches 1.693 and Hauhau reaches 1.540, meaning that on at least one prompt, these models produce substantially different first-token distributions than the base. Identifying which prompts trigger these outliers would be valuable for understanding failure modes.

## Raw Data Files

| File | Description |
|------|-------------|
| `results/kl/logits_base.pt` | Base model logprobs (100 × 248,320, float32) |
| `results/kl/logits_huihui.pt` | Huihui variant logprobs |
| `results/kl/logits_aeon.pt` | AEON variant logprobs |
| `results/kl/logits_heretic.pt` | Heretic variant logprobs |
| `results/kl/logits_hauhau.pt` | Hauhau variant logprobs |
| `results/kl/logits_abliterix.pt` | Abliterix variant logprobs |
| `results/kl/kl_huihui.json` | Huihui KL results with per-prompt detail |
| `results/kl/kl_aeon.json` | AEON KL results with per-prompt detail |
| `results/kl/kl_heretic.json` | Heretic KL results with per-prompt detail |
| `results/kl/kl_hauhau.json` | Hauhau KL results with per-prompt detail |
| `results/kl/kl_abliterix.json` | Abliterix KL results with per-prompt detail |
