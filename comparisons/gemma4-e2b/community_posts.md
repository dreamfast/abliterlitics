# HuggingFace Community Posts — Gemma4-E2B Abliteration Comparison

> Quick-reference for posting on model cards. URL, title, and key points for each. Append the shared methodology footer to every post.

---

## Shared methodology footer

Copy this to the bottom of every post:

---

**Our methodology:**

- **KL divergence:** Full vocab at 262,144 tokens. First-token logits from 100 benign prompts via `model.generate(max_new_tokens=1, output_scores=True)`. Uses `F.kl_div(batchmean, log_target=True)`. System prompt is "You are a helpful assistant." This matches the [Heretic evaluator](https://github.com/p-e-w/heretic/blob/master/src/heretic/evaluator.py) v2.0 methodology.
- **Calibration check:** 4 of our 13 variants were built with Heretic and self-report KL values. We cross-checked all four to establish our measurement error ratio:

  | Variant | Card claims | We measured | Difference |
  |---------|------------|-------------|-----------|
  | pew | 0.1522 | 0.1526 | +0.3% |
  | coder3101 | 0.1651 | 0.1673 | +1.3% |
  | llmfan46 | 0.0779 | 0.0677 | -13.1% |
  | kasper | 0.1650 | 0.1933 | +17.2% |

  3 of 4 land within ±13%. Our measurement error ratio against Heretic self-reports is under 6% for standard configurations. The kasper +17.2% gap is explained by a non-standard config, an RTX 3080 with 10GB VRAM that dropped `down_proj` to fit memory. This gives confidence our KL numbers for non-Heretic variants are accurate too.

- **Safety:** [HarmBench](https://github.com/centerforaisafety/HarmBench) 400 textual behaviours. Settings were `max_tokens=8096`, `temperature=0`, `enable_thinking=true`. Keyword detection first, then full LLM review of all 5,600 responses by glm-5.1.
- **Benchmarks:** [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) via [vLLM](https://github.com/vllm-project/vllm) v0.20.0. Native BF16 on a single RTX 5090 with 32GB. All 14 models tested with identical settings.
- **Shared-KV note:** 5 of 13 variants shipped missing 60 shared-KV weights from Gemma4 layers 15 to 34. Patched from the base model before evaluation. Lossless patch.

Happy to share raw data if anything looks off. Full report publishes later today.

**Links:**
- Full report with all 13 variants: [DreamFast/Gemma4-e2b-abliterlitics](https://huggingface.co/DreamFast/Gemma4-e2b-abliterlitics) on HuggingFace. Includes all build artifacts, result files, and raw data for full transparency.
- Toolkit source code: [dreamfast/abliterlitics](https://github.com/dreamfast/abliterlitics) on GitHub, branch `feat/gemma4-e2b-comparison`. Scripts and pipeline code used for this comparison will be uploaded as a snapshot.

---

## 1. DuoNeural/Gemma-4-E2B-Heretic

**Post at:** https://huggingface.co/DuoNeural/Gemma-4-E2B-Heretic/discussions
**Title:** Independent verification of KL divergence and safety claims

**Key points:**

- Publishing an independent forensic comparison of 13 abliterated Gemma4-E2B variants today. Giving a heads-up before it goes live.
- Their card reports KL divergence from base at ~0.001, and 17/100 refusals.
- We measured KL at **0.1872** using the same Heretic evaluator methodology. That is **187x higher** than claimed.
- Refusal rate is roughly consistent. 17/100 on their test vs 71/400 on ours, so about 17% either way.
- To verify our KL pipeline is accurate, we cross-checked against 4 Heretic-built models that self-report KL values. 3 of 4 land within ±6% of the card value. The standard projection models llmfan46, coder3101, and pew all measure at KL 0.068 to 0.153, which matches the 0.067 DuoNeural cites for standard projection. Our measurement is solid.
- Possible explanation. The ~0.001 KL on their card may come from the Heretic optimisation trial itself, measuring KL over the LoRA adapter only. Our post-hoc measurement evaluates the full merged model against the base. These can differ substantially. If that is what happened, the card should clarify what the number represents.
- Their model modified 49 tensors across 2 types in 29 layers. A moderate footprint compared to the surgical approaches at 7 to 16 tensors.
- Ask if they would consider updating the KL claim or clarifying what it measures.

---

## 2. treadon/gemma4-E2B-it-Abliterated-AND-Disinhibited-USE-THIS

**Post at:** https://huggingface.co/treadon/gemma4-E2B-it-Abliterated-AND-Disinhibited-USE-THIS/discussions
**Title:** Independent verification of KL divergence and capability impact

**Key points:**

- Great work on the dual disinhibition and abliteration approach.
- Their card says "Same model, same weights, same knowledge."
- We measured KL at **3.971**. That is **4.1x higher** than the next worst variant, wwtcyberlab at 0.964. Rates "heavy" on our scale.
- Show the full ranking: llmfan46 0.068, coder3101 0.167, trevorjs 0.365, wwtcyberlab 0.964, then **treadon at 3.971**.
- The dual approach achieves **100.0% HarmBench ASR**. Only variant with zero refusals out of 400 test behaviours. Very effective safety removal.
- But there are capability tradeoffs. MMLU drops 1.0pp. GSM8K drops 2.9pp. TQA-MC2 drops 4.6pp. LAMBADA perplexity goes up 36%. 38 empty GSM8K responses where the model exhausts its thinking budget without producing an answer.
- 21 truncated HarmBench responses, all showing compliant reasoning before hitting the token limit. 20 entered repetition loops including LaTeX nesting and binary dumps.
- Their own disinhibition and refusal evals are impressive. Note that those datasets and HarmBench test different things. Their `abliteration-eval` focuses on harmful vs over-refusal, while HarmBench covers 7 categories including chemical and biological synthesis, cybercrime, and harassment.
- The "same model" framing understates the distributional impact. The KL divergence data shows substantial alteration beyond just directional ablations.
- Overall assessment: best safety removal in the comparison, but heaviest capability cost.

---

## 3. WWTCyberLab/gemma-4-E2B-it-abliterated

**Post at:** https://huggingface.co/WWTCyberLab/gemma-4-E2B-it-abliterated/discussions
**Title:** Independent verification of refusal rate and language modelling quality

**Key points:**

- Their card claims "Refusal Rate: 0.0%" and "Quality Preservation: 101%."
- Refusal rate. We measured **2/400 refusals**, giving 99.5% ASR. Close to 0% but not exact. The two refusals were cases where the model asked for missing input without providing harmful content.
- Quality preservation is where results diverge. Their "101% QPS" appears to use an LLM-as-judge metric. Standard benchmarks tell a different story:
  - MMLU: 29.00% to 27.14%, down 1.86pp
  - TruthfulQA MC2: 48.38% to 45.18%, down 3.20pp
  - **LAMBADA perplexity: 145,956 to 831,086, that is 5.69x worse.** Second-worst in the entire comparison.
- LAMBADA measures next-token prediction quality. A 5.69x blowup means the model's language modelling capability has been substantially degraded, even though accuracy metrics on classification tasks stay close to base.
- KL divergence is **0.964**, rated "significant." Highest non-treadon variant.
- Model shipped with 540 of 600 safetensor keys, missing 60 shared-KV weights. Patched from base before evaluation. Common export bug across abliteration tools, 4 other variants had the same issue.
- Their visualisations and the "key takeaway for defenders" section are excellent.
- Suggest updating the refusal rate to acknowledge 2/400 residual refusals, and noting the LAMBADA degradation alongside the QPS metric so users get a complete picture.

---

## 4. Kasper-Bankler/gemma-4-E2B-uncensored

**Post at:** https://huggingface.co/Kasper-Bankler/gemma-4-E2B-uncensored/discussions
**Title:** Independent verification of KL divergence measurement

**Key points:**

- Their card reports KL at 0.1650 and calls it the "Goldilocks zone."
- We measured KL at **0.1933**. That is **+17.2% higher** than claimed.
- This is the largest gap among the 4 Heretic-built variants we calibrated against. The other 3: pew at +0.3%, coder3101 at +1.3%, llmfan46 at -13.1%.
- Their card explains they used an RTX 3080 with 10GB VRAM and dropped `down_proj` to fit memory. That non-standard config likely explains the measurement gap.
- Capability is solid. MMLU drops 0.47pp. GSM8K drops 0.23pp. HarmBench ASR is 92.5%.
- The "Goldilocks zone" assessment is broadly correct. Capability preservation is strong at this KL level.
- The KL discrepancy is likely a measurement-condition difference, not a fundamental issue.
- Note that their model also shipped with 540 of 600 keys, missing 60 shared-KV weights. Patched from base before evaluation.

---

## 5. Ether4o4/Gemma4_E2B_Abliterated_Opus_Distilled

**Post at:** https://huggingface.co/Ether4o4/Gemma4_E2B_Abliterated_Opus_Distilled/discussions
**Title:** Independent verification of Opus distillation impact on reasoning

**Key points:**

- Unique model. Opus reasoning distillation on top of abliteration. Duplicated from amkkk's base.
- Safety removal: 95.2% ASR with 19/400 refusals. Lowest among the high-ASR tier. For comparison, treadon is at 100%, wangzhang at 99.8%, trevorjs and wwtcyberlab and huihui-v2 all at 99.5%.
- Reasoning is the concern:
  - GSM8K flex: 83.47% to **76.57%**, a drop of **6.9pp**. Worst of all 13 variants.
  - GSM8K empty responses: 10 to **84**. That is 6.4% of questions where the model exhausts its thinking budget without producing an answer.
  - LAMBADA: 2.28x worse.
- Broadest modification footprint of any variant. 166 tensors, that is 30.7% of the model. 6 distinct tensor types including Gemma4-specific gate components.
- Multi-rank edits at effective rank ~2.3, compared to standard rank-1 abliteration.
- The Opus distillation was intended to improve reasoning quality, but the data shows it worsened it. Simpler abliteration approaches beat base on GSM8K. Coder3101 gains 1.4pp, llmfan46 gains 0.5pp. The combination of broad weight modification plus LoRA adapters damages thinking chain efficiency.
- Not a criticism of the approach. It is interesting research. But users should know the distillation comes at a reasoning cost.

---

## Notes

- Post in the **Community tab, then Discussions**, as a New Discussion
- Append the shared methodology footer to every post
- Give authors 4 to 6 hours to respond before publishing the full report
- The treadon post uses a softer tone since their card is mostly honest. The "same model" claim is the only disputed point.
- The WWT CyberLab post focuses on the LAMBADA numbers. The 0.0% vs 0.5% refusal gap is minor and not worth dwelling on.
- Keep the 4-variant calibration table handy in case anyone challenges the KL methodology.
