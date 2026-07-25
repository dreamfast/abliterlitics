---
base_model: {{ORG}}/{{MODEL_NAME}}
language:
- en
{%- if model_supports_zh %}
- zh
{%- endif %}
library_name: transformers
license: {{LICENSE}}
{%- if license_link %}
license_link: {{LICENSE_LINK}}
{%- endif %}
pipeline_tag: text-generation
tags:
{%- for tag in tags %}
- {{tag}}
{%- endfor %}
base_model:
- {{ORG}}/{{MODEL_NAME}}
---

# {{MODEL_DISPLAY_NAME}}: {{VARIANT_DISPLAY_NAME}}

> Forensic analysis by [Abliterlitics](https://github.com/dreamfast/abliterlitics), open-source abliteration forensics toolkit
>
> **Data & artifacts:** [HuggingFace]({{HF_DATASET_URL}}) | **Report:** [abliterlitics.dev]({{ABLITERLITICS_DEV_URL}}) | **Code:** [GitHub]({{GITHUB_BRANCH_URL}})

{{INTRO_PARAGRAPH}}
<!-- ELI5: Explain like the reader is 5. No jargon, no acronyms, no technical terms. What is this model? What was done to it? Why should anyone care? One plain-English paragraph anyone can understand. -->

{{CLAIMS_PARAGRAPH}}
<!-- ELI5: What did the model creator claim? Keep it conversational and accessible. -->

{{CLAIMS_VERIFICATION_SECTION}}
<!-- ELI5: Did those claims hold up? Plain language, no KL or ASR jargon. "They said X, we found Y." -->

{{WHICH_ONE_SECTION}}
<!-- ELI5: If someone just wants to know which model to use, what's the answer? Write for someone who has never heard of abliteration. -->

## Key findings

{{KEY_FINDINGS}}
<!-- ELI5: The 3-5 most important things a non-expert should know. Plain English. No metrics without explaining what they mean in human terms. -->

## TL;DR

{{TLDR}}
<!-- ELI5: One-paragraph summary a complete beginner would understand. The "too long, didn't read" version. -->

<!-- ═══════════════════════════════════════════════════════════════════
     TECHNICAL SECTIONS BEGIN HERE
     Everything above this line is written for a general audience.
     Everything below uses normal technical language.
     ═══════════════════════════════════════════════════════════════════ -->

## Quick Facts

| | |
|---|---|
| **Base model** | [{{ORG}}/{{MODEL_NAME}}]({{HF_BASE_URL}}) |
| **Architecture** | {{ARCHITECTURE}} |
| **Parameters** | ~{{PARAM_COUNT}} |
| **Precision** | {{PRECISION}} |
{%- if source_format %}
| **Source** | {{SOURCE_DESCRIPTION}} |
{%- endif %}
| **Context length** | {{CONTEXT_LENGTH}} tokens |
{%- if is_thinking_model %}
| **Thinking** | {{THINKING_DESCRIPTION}} |
{%- endif %}

{{ARCHITECTURE_NOTES}}

{{SOURCE_RECOVERY_SECTION}}

## Benchmarks

Evaluated with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness){{BENCHMARK_METHOD}}.

{{BENCHMARK_PHASE_HEADER}}

| Task | Base | {{VARIANT_NAMES_TABLE_HEADER}} |
|------|------|{{VARIANT_TABLE_SEPARATOR}}|
| MMLU | {{MMLU_BASE}} | {{MMLU_VARIANTS}} |
| GSM8K | {{GSM8K_BASE}} | {{GSM8K_VARIANTS}} |
| HellaSwag | {{HELLASWAG_BASE}} | {{HELLASWAG_VARIANTS}} |
| ARC Challenge | {{ARC_BASE}} | {{ARC_VARIANTS}} |
| WinoGrande | {{WINOGRANDE_BASE}} | {{WINOGRANDE_VARIANTS}} |
| TruthfulQA MC1 | {{TQA_MC1_BASE}} | {{TQA_MC1_VARIANTS}} |
| TruthfulQA MC2 | {{TQA_MC2_BASE}} | {{TQA_MC2_VARIANTS}} |
| PiQA | {{PIQA_BASE}} | {{PIQA_VARIANTS}} |
| Lambada (ppl ↓) | {{LAMBADA_BASE}} | {{LAMBADA_VARIANTS}} |

{{BENCHMARK_NOTES}}

### Delta vs base

| Task | {{VARIANT_NAMES_TABLE_HEADER}} |
|------|{{VARIANT_TABLE_SEPARATOR}}|
| MMLU | {{MMLU_DELTAS}} |
| GSM8K | {{GSM8K_DELTAS}} |
| HellaSwag | {{HELLASWAG_DELTAS}} |
| ARC Challenge | {{ARC_DELTAS}} |
| WinoGrande | {{WINOGRANDE_DELTAS}} |
| TruthfulQA MC2 | {{TQA_DELTAS}} |
| PiQA | {{PIQA_DELTAS}} |

{{DELTA_NOTES}}

### What the benchmarks tell us

{{BENCHMARK_ANALYSIS}}

{{BENCHMARK_IMAGES}}

{{GSM8K_REASONING_SECTION}}

{{IS_IT_REALLY_LOSSLESS_SECTION}}

## Safety: HarmBench

[HarmBench](https://github.com/centerforaisafety/HarmBench) with 400 textual behaviours, `max_tokens={{HB_MAX_TOKENS}}`, temperature={{HB_TEMPERATURE}}{%- if is_thinking_model %}, `enable_thinking=true`{%- endif %}. Classified with {{HB_CLASSIFIER}}.

| Variant | ASR | Refusals | Truncated | Avg Tokens |
|---------|-----|----------|-----------|------------|
| [Base]({{HF_BASE_URL}}) | {{HB_BASE_ASR}} | {{HB_BASE_REFUSALS}} | {{HB_BASE_TRUNCATED}} | {{HB_BASE_AVG_TOKENS}} |
{{HARMBENCH_VARIANT_ROWS}}

{{HARMBENCH_OVERVIEW}}

{{HARMBENCH_IMAGES}}

{{COT_VERIFICATION_SECTION}}

### ASR by category

| Category | Items | Base | {{VARIANT_NAMES_TABLE_HEADER}} |
|----------|-------|------|{{VARIANT_TABLE_SEPARATOR}}|
| chemical_biological | 56 | {{CB_BASE}} | {{CB_VARIANTS}} |
| copyright | 100 | {{COPY_BASE}} | {{COPY_VARIANTS}} |
| cybercrime_intrusion | 67 | {{CYBER_BASE}} | {{CYBER_VARIANTS}} |
| harassment_bullying | 25 | {{HARASS_BASE}} | {{HARASS_VARIANTS}} |
| harmful | 22 | {{HARMFUL_BASE}} | {{HARMFUL_VARIANTS}} |
| illegal | 65 | {{ILLEGAL_BASE}} | {{ILLEGAL_VARIANTS}} |
| misinformation | 65 | {{MISINFO_BASE}} | {{MISINFO_VARIANTS}} |

{{HARMBENCH_ANALYSIS}}

{{HARMBENCH_CATEGORY_IMAGES}}

{{HARDEST_ITEMS_SECTION}}

{{FALSE_REFUSALS_SECTION}}

{{SOFT_REFUSALS_SECTION}}

{{TRUNCATED_COT_SECTION}}

{{LLM_REVIEW_SECTION}}

{{TRANSITION_MATRIX_SECTION}}

{{THINKING_BUDGET_SECTION}}

{{BASE_MODEL_SAFETY_SECTION}}

## KL Divergence

Methodology: `F.kl_div(logprobs_variant, logprobs_base, reduction="batchmean", log_target=True)` on full vocab first-token logits from [mlabonne/harmless_alpaca](https://huggingface.co/datasets/mlabonne/harmless_alpaca) `test[:100]`, matching the [Heretic evaluator](https://github.com/p-e-w/heretic/blob/master/src/heretic/evaluator.py). System prompt: "You are a helpful assistant."

| Variant | KL Divergence | Rating |
|---------|--------------|--------|
{{KL_DIVERGENCE_ROWS}}

Rating scale: excellent below 0.01, very good 0.01 to 0.1, moderate 0.1 to 0.4, significant 0.4 to 1.0, heavy above 1.0.

{{KL_ANALYSIS}}

{{KL_IMAGES}}

{{KL_CALIBRATION_SECTION}}

{{KL_VS_BENCHMARK_SECTION}}

{{KL_VS_HERETIC_CARD_SECTION}}

{{CROSS_FAMILY_KL_SECTION}}

## Weight Analysis

### Modification summary

| | {{VARIANT_NAMES_TABLE_HEADER}} |
|---|{{VARIANT_TABLE_SEPARATOR}}|
| Tensors changed | {{TENSORS_CHANGED}} |
| Relative edit | {{RELATIVE_EDIT}} |
| Tensor types | {{TENSOR_TYPES}} |
| Layers modified | {{LAYERS_MODIFIED}} |
{%- if has_moe %}
| Params modified | {{PARAMS_MODIFIED}} |
{%- endif %}

{{WEIGHT_ANALYSIS}}

{{ABLITERATION_SIGNAL_VS_NOISE_SECTION}}

{{THREE_TIERS_SECTION}}

### Which tensor types get modified

| Component | Role | {{VARIANT_NAMES_TABLE_HEADER}} |
|-----------|------|{{VARIANT_TABLE_SEPARATOR}}|
{{TENSOR_TYPE_ROWS}}

{{TENSOR_TYPE_ANALYSIS}}

{{WEIGHT_IMAGES}}

{{SVD_RANK_SECTION}}

{{LAYER_COVERAGE_SECTION}}

{{CROSS_TECHNIQUE_SECTION}}

{{COSINE_SIMILARITY_SECTION}}

{{NEAR_IDENTICAL_MODELS_SECTION}}

{{EXPORT_BUG_SECTION}}

{{FORENSIC_SIGNATURES_SECTION}}

{{EXPERT_HEATMAPS_SECTION}}

## The Optimal Tradeoff

{{OPTIMAL_TRADEOFF_TABLE}}

{{OPTIMAL_TRADEOFF_ANALYSIS}}

## Summary

| Metric | {{VARIANT_NAMES_TABLE_HEADER}} |
|--------|{{VARIANT_TABLE_SEPARATOR}}|
| **HarmBench ASR** | {{SUMMARY_ASR}} |
| **MMLU** | {{SUMMARY_MMLU}} |
| **GSM8K** | {{SUMMARY_GSM8K}} |
| **KL divergence** | {{SUMMARY_KL}} |
| Tensors changed | {{SUMMARY_TENSORS}} |
| Strategy | {{SUMMARY_STRATEGY}} |
{%- if has_avg_delta %}
| Avg |delta| (excl GSM8K, Lambada) | {{SUMMARY_AVG_DELTA}} |
{%- endif %}

{{SUMMARY_NOTE}}

{{VARIANT_SUMMARIES}}

## Evaluation Timeline

{{EVALUATION_TIMELINE}}

{{FAILED_RUNS_SECTION}}

{{GPU_HOURS_SECTION}}

{{HOW_THIS_WAS_MADE_SECTION}}

## Methodology

- **Capability:** [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) via [vLLM](https://github.com/vllm-project/vllm){{LM_EVAL_VERSION}}, {{QUANT_METHOD}} on {{HARDWARE}}
{%- if has_phase2 %}
- **GSM8K:** {{GSM8K_METHOD}}
{%- endif %}
- **Safety:** [HarmBench](https://github.com/centerforaisafety/HarmBench) 400 textual behaviours, `max_tokens={{HB_MAX_TOKENS}}, temperature={{HB_TEMPERATURE}}`{%- if is_thinking_model %}, `enable_thinking=true`{%- endif %}, {{HB_CLASSIFIER}}
- **KL divergence:** Full vocab first-token logits via `model.generate(max_new_tokens=1, output_scores=true)`, matching [Heretic evaluator](https://github.com/p-e-w/heretic/blob/master/src/heretic/evaluator.py) methodology
- **Weight analysis:** SVD, fingerprint, edit vector overlap, and per-layer analysis comparing all abliteration variants against the base, using [Abliterlitics](https://github.com/dreamfast/abliterlitics)
- **Hardware:** {{HARDWARE_FULL}}

{{METHODOLOGY_LESSONS_SECTION}}

{{CROSS_MODEL_COMPARISONS_SECTION}}

## Disclaimer

This model has had safety alignment removed. It will comply with harmful requests, including generating content related to violence, illegal activities, and other harmful behaviours. Use responsibly and in accordance with applicable laws and regulations. The authors do not condone or encourage the use of this model for harmful purposes.

---

<small>While I have taken the time to verify all results thoroughly, I am open to any corrections, additional benchmarks, or further analysis. If you spot something that looks wrong and can be confirmed, I am happy to fix it.</small>
