# KL Divergence Analysis: Gemma4-E2B

> Full vocab (262,144 tokens) first-token KL divergence on 100 benign prompts from [mlabonne/harmless_alpaca](https://huggingface.co/datasets/mlabonne/harmless_alpaca) `test[:100]`
> Method: `F.kl_div(logprobs_variant, logprobs_base, reduction='batchmean', log_target=True)` via `model.generate(max_new_tokens=1, output_scores=True)`
> System prompt: "You are a helpful assistant." — matching [Heretic evaluator](https://github.com/p-e-w/heretic/blob/master/src/heretic/evaluator.py) methodology v2.0

## Results

Sorted by KL divergence (lowest = closest to base model behavior on benign prompts):

| Rank | Model | KL (batchmean) | Median | Std | Min | Max | Rating |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | **llmfan46** | **0.0677** | 0.00018 | 0.566 | ~0 | 5.681 | very good |
| 2 | pew | 0.1526 | 0.00007 | 1.491 | ~0 | 14.985 | moderate |
| 3 | coder3101 | 0.1673 | 0.00027 | 1.527 | ~0 | 15.351 | moderate |
| 4 | duoneural | 0.1872 | 0.00030 | 1.632 | ~0 | 16.415 | moderate |
| 5 | kasper | 0.1933 | 0.00093 | 1.076 | ~0 | 10.422 | moderate |
| 6 | huihui-v1 | 0.2510 | 0.00037 | 2.086 | ~0 | 20.973 | moderate |
| 7 | prithiv | 0.2510 | 0.00037 | 2.086 | ~0 | 20.973 | moderate |
| 8 | trevorjs | 0.3653 | 0.00105 | 2.610 | ~0 | 26.221 | moderate |
| 9 | huihui-v2 | 0.5302 | 0.00432 | 2.414 | ~0 | 22.938 | significant |
| 10 | ether4o4 | 0.6688 | 0.17530 | 1.181 | ~0 | 9.145 | significant |
| 11 | wangzhang | 0.6984 | 0.01531 | 2.784 | ~0 | 26.123 | significant |
| 12 | wwtcyberlab | 0.9640 | 0.00493 | 4.506 | ~0 | 42.451 | significant |
| 13 | treadon | **3.9713** | 0.46938 | 5.855 | ~0 | 23.482 | heavy |

## Rating Scale

Based on the Heretic evaluator interpretation, adapted for this comparison's wider KL range:

| Rating | KL Range | Meaning |
|---|---|---|
| excellent | < 0.01 | Nearly indistinguishable from base |
| very good | 0.01 – 0.1 | Minimal distributional shift |
| moderate | 0.1 – 0.4 | Noticeable but controlled shift |
| significant | 0.4 – 1.0 | Substantial distributional change |
| heavy | > 1.0 | Major behavioral divergence |

## What the KL tells us

### The only "very good" variant: llmfan46

llmfan46's KL of 0.0677 is 2.3x lower than the next closest variant (pew at 0.1526). It modified only 7 tensors (1.2% of 600) — the fewest of any variant — all `self_attn.o_proj.weight` in layers 17–23. The surgical approach produces the smallest output distribution shift. On benchmarks, llmfan46 achieves 85.0% HarmBench ASR and 83.9% GSM8K flex, suggesting its conservatism trades safety removal for capability preservation.

### The moderate cluster (pew through trevorjs)

Seven variants cluster in the 0.15–0.37 KL range, all rated "moderate." This cluster represents the typical abliteration tradeoff: noticeable distribution shift but within controlled bounds.

Key observations within this cluster:
- **huihui-v1 and prithiv are numerically identical** (KL=0.2510, all statistics match to full precision). Weight forensics show cosine similarity of 1.0 across all shared tensors, but small differences appear in generative evaluations (GSM8K, HarmBench), so they are near-identical rather than bit-for-bit identical.
- **pew and coder3101** (Heretic-based) have low medians (~0.0001–0.0003) meaning most prompts see almost no shift, with spikes on a few outlier prompts.
- **kasper** has the second-highest median (0.00093) in this cluster after trevorjs (0.00105), despite a lower mean than huihui-v1, indicating more pervasive but smaller shifts.
- **trevorjs** at 0.365 is the upper bound of "moderate" — it modified 70 tensors (11.7%), the most of any variant still in this range.

### The significant cluster (huihui-v2 through wwtcyberlab)

Four variants show "significant" distribution shift (KL 0.53–0.96):

- **ether4o4** stands out with the highest median (0.175) among non-heavy variants. This means the distribution shift is pervasive — nearly every prompt sees meaningful output changes, not just outliers. This is consistent with its broad modification footprint (166 tensors, 6 tensor types).
- **huihui-v2** at 0.530 has 2x higher KL than huihui-v1 (0.251) despite targeting the same tensor types. The v2 abliteration applies much larger edit magnitudes (mean norm 4.94 vs 2.02).
- **wangzhang** at 0.698 uniquely modified `q_proj` and `v_proj` weights in addition to the standard targets. Modifying attention input projections (what the model "hears") rather than just output projections (what it "says") causes more distributional disruption.
- **wwtcyberlab** at 0.964 is the highest non-treadon variant, with the largest max KL spike at 42.45 — the widest single-prompt distributional gap in the dataset.

### The outlier: treadon

Treadon at KL=3.97 is **4.1x higher** than the next worst variant (wwtcyberlab). Its median of 0.47 means nearly every prompt sees substantial distributional shift — this is not driven by outliers. The "Disinhibition + abliteration" dual approach produces the most dramatic behavioral change of any variant, consistent with:
- Its highest HarmBench ASR (98.8%)
- Its high GSM8K empty response rate (2.9%, third after ether4o4 at 6.4% and huihui-v2 at 4.1%)
- Its highest avg token count (2,067) and most truncations (21)

Treadon's approach fundamentally alters the model's reasoning patterns, not just its refusal behavior.

## KL vs HarmBench ASR Correlation

| KL Tier | Models | ASR Range |
|---|---|---|
| very good (<0.1) | llmfan46 | 85.0% |
| moderate (0.1–0.4) | pew, coder3101, duoneural, kasper, huihui-v1, prithiv, trevorjs | 81.8%–97.2% |
| significant (0.4–1.0) | huihui-v2, ether4o4, wangzhang, wwtcyberlab | 97.0%–98.8% |
| heavy (>1.0) | treadon | 98.8% |

Higher KL correlates with higher ASR but with diminishing returns. The jump from moderate to significant KL gains only ~10pp more ASR (87% → 97%). Wangzhang in the significant tier already matches treadon's 98.8% ASR at less than one-fifth the KL cost (0.698 vs 3.971), confirming that heavy KL is rarely worth the damage.

**The optimal tradeoff is in the moderate KL range.** Variants like coder3101 (KL=0.167, ASR=95.8%) and trevorjs (KL=0.365, ASR=97.2%) achieve near-maximal safety removal with controlled distribution shift.

## KL vs Benchmark Impact

| Model | KL | MMLU Δ | GSM8K Flex Δ | LAMBADA PPL Ratio |
|---|---|---|---|---|
| llmfan46 | 0.068 | -0.6pp | +0.5pp | 1.03x |
| coder3101 | 0.167 | -0.3pp | +1.4pp | 0.95x |
| trevorjs | 0.365 | -0.1pp | -1.0pp | 1.17x |
| ether4o4 | 0.669 | -0.8pp | -6.9pp | 2.28x |
| treadon | 3.971 | -1.0pp | -2.9pp | 1.36x |

LAMBADA perplexity is the most sensitive metric to KL divergence. Ether4o4 at KL=0.67 shows 2.28x LAMBADA degradation despite only -0.8pp MMLU loss. LAMBADA's sensitivity to next-token distribution quality makes it an early warning indicator for capability damage.

## Comparison with Other Model Families

KL divergence measures how far the abliteration technique has shifted normal prompt outputs from the base model — it is primarily a function of the edit's magnitude and direction, not the model's parameter count. However, model architecture can influence sensitivity: smaller models may have less redundancy to absorb the same relative weight change, and vocabulary size (Gemma4's 262K vs Qwen's 151K) affects KL magnitude. Cross-family comparisons should be interpreted with these caveats.

| Model Family | Best KL | Worst KL | Spread |
|---|---|---|---|
| Qwen3.6-27B | 0.004 (Heretic) | 0.024 (HauhauCS) | 6x |
| GLM-4.7-Flash | 0.008 (Huihui) | 0.053 (Abliterix) | 6.6x |
| Qwen3.5-27B | 0.063 (Heretic) | 0.256 (HauhauCS) | 4.1x |
| **Gemma4-E2B** | **0.068 (llmfan46)** | **3.971 (treadon)** | **58.7x** |

Gemma4-E2B's best KL (0.068) is comparable to Qwen3.5-27B's best (0.063), reflecting that the most surgical technique in each family produces similar distributional shift. However, its worst KL (3.97) is an order of magnitude higher than any other family's worst. The 58.7x spread between best and worst KL is the largest in the project.

The large spread reflects the diversity of abliteration approaches: from surgical single-type edits (llmfan46, 7 tensors) to aggressive multi-method approaches (treadon, disinhibition + abliteration). Other families tested fewer variants (3–5), so their spread is narrower partly because the tails aren't sampled.

## Methodology Notes

1. **Full vocab (262K)**: Gemma4 has one of the largest vocabularies tested. KL divergence over 262K tokens is more sensitive to distribution shifts than smaller vocabularies (e.g., Qwen's 151K).
2. **First-token only**: The methodology measures the divergence at the first generated token position. This captures the model's immediate behavioral tendency but doesn't reflect cumulative distribution shifts over longer sequences.
3. **Benign prompts only**: The harmless_alpaca dataset contains no harmful prompts. The KL divergence measures how much the abliteration changes behavior on safe inputs, which directly corresponds to capability preservation.
4. **Batchmean reduction**: The `batchmean` reduction averages the per-prompt KL divergence across all 100 prompts. Each per-prompt KL is itself the sum of `p_base * (log p_base - log p_variant)` over the full vocabulary.
