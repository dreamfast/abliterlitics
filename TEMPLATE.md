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

{{INTRO_PARAGRAPH}}

{{CLAIMS_PARAGRAPH}}

## TL;DR

{{TLDR}}

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

{{ARCHITECTURE_NOTES}}

## Benchmarks

Evaluated with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness){{BENCHMARK_METHOD}}.

| Task | Base | {{VARIANT_NAMES_TABLE_HEADER}} |
|------|------|{{VARIANT_TABLE_SEPARATOR}}|
| MMLU | {{MMLU_BASE}} | {{MMLU_VARIANTS}} |
| GSM8K | {{GSM8K_BASE}} | {{GSM8K_VARIANTS}} |
| HellaSwag | {{HELLASWAG_BASE}} | {{HELLASWAG_VARIANTS}} |
| ARC Challenge | {{ARC_BASE}} | {{ARC_VARIANTS}} |
| WinoGrande | {{WINOGRANDE_BASE}} | {{WINOGRANDE_VARIANTS}} |
| TruthfulQA MC2 | {{TQA_BASE}} | {{TQA_VARIANTS}} |
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

## Safety: HarmBench

[HarmBench](https://github.com/centerforaisafety/HarmBench) with 400 textual behaviours, `max_tokens={{HB_MAX_TOKENS}}`, temperature={{HB_TEMPERATURE}}. Classified with {{HB_CLASSIFIER}}.

| Variant | ASR | Refusals |
|---------|-----|----------|
| [Base]({{HF_BASE_URL}}) | {{HB_BASE_ASR}} | {{HB_BASE_REFUSALS}} |
{{HARMBENCH_VARIANT_ROWS}}

### ASR by category

| Category | Base | {{VARIANT_NAMES_TABLE_HEADER}} |
|----------|------|{{VARIANT_TABLE_SEPARATOR}}|
| chemical_biological | {{CB_BASE}} | {{CB_VARIANTS}} |
| copyright | {{COPY_BASE}} | {{COPY_VARIANTS}} |
| cybercrime_intrusion | {{CYBER_BASE}} | {{CYBER_VARIANTS}} |
| harassment_bullying | {{HARASS_BASE}} | {{HARASS_VARIANTS}} |
| harmful | {{HARMFUL_BASE}} | {{HARMFUL_VARIANTS}} |
| illegal | {{ILLEGAL_BASE}} | {{ILLEGAL_VARIANTS}} |
| misinformation | {{MISINFO_BASE}} | {{MISINFO_VARIANTS}} |

{{HARMBENCH_ANALYSIS}}

{{HARMBENCH_IMAGES}}

{{THINKING_BUDGET_SECTION}}

## KL Divergence

Methodology: `F.kl_div(logprobs_variant, logprobs_base, reduction="batchmean", log_target=True)` on full vocab first-token logits from [mlabonne/harmless_alpaca](https://huggingface.co/datasets/mlabonne/harmless_alpaca) `test[:100]`, matching the [Heretic evaluator](https://github.com/p-e-w/heretic/blob/master/src/heretic/evaluator.py). System prompt: "You are a helpful assistant."

| Variant | KL Divergence | Rating |
|---------|--------------|--------|
{{KL_DIVERGENCE_ROWS}}

{{KL_ANALYSIS}}

{{KL_IMAGES}}

## Weight Analysis

### Modification summary

| | {{VARIANT_NAMES_TABLE_HEADER}} |
|---|{{VARIANT_TABLE_SEPARATOR}}|
| Tensors changed | {{TENSORS_CHANGED}} |
| Relative edit | {{RELATIVE_EDIT}} |
| Tensor types | {{TENSOR_TYPES}} |
| Layers modified | {{LAYERS_MODIFIED}} |

{{WEIGHT_ANALYSIS}}

### Which tensor types get modified

| Component | {{VARIANT_NAMES_TABLE_HEADER}} |
|-----------|{{VARIANT_TABLE_SEPARATOR}}|
{{TENSOR_TYPE_ROWS}}

{{TENSOR_TYPE_ANALYSIS}}

{{WEIGHT_IMAGES}}

{{CROSS_TECHNIQUE_SECTION}}

## Summary

| Metric | {{VARIANT_NAMES_TABLE_HEADER}} |
|--------|{{VARIANT_TABLE_SEPARATOR}}|
| **HarmBench ASR** | {{SUMMARY_ASR}} |
| **MMLU** | {{SUMMARY_MMLU}} |
| **GSM8K** | {{SUMMARY_GSM8K}} |
| **KL divergence** | {{SUMMARY_KL}} |
| Tensors changed | {{SUMMARY_TENSORS}} |
| Strategy | {{SUMMARY_STRATEGY}} |

{{SUMMARY_NOTE}}

{{VARIANT_SUMMARIES}}

## Methodology

- **Capability:** [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) via [vLLM](https://github.com/vllm-project/vllm){{LM_EVAL_VERSION}}, {{QUANT_METHOD}} on {{HARDWARE}}
- **Safety:** [HarmBench](https://github.com/centerforaisafety/HarmBench) 400 textual behaviours, `max_tokens={{HB_MAX_TOKENS}}, temperature={{HB_TEMPERATURE}}`, {{HB_CLASSIFIER}}
- **KL divergence:** Full vocab first-token logits via `model.generate(max_new_tokens=1, output_scores=true)`, matching [Heretic evaluator](https://github.com/p-e-w/heretic/blob/master/src/heretic/evaluator.py) methodology
- **Weight analysis:** SVD, fingerprint, edit vector overlap, and per-layer analysis comparing all abliteration variants against the base, using [Abliterlitics](https://github.com/dreamfast/abliterlitics)
- **Hardware:** {{HARDWARE_FULL}}

{{FORENSIC_NOTES_SECTION}}

{{CROSS_MODEL_COMPARISONS_SECTION}}

## Disclaimer

This model has had safety alignment removed. It will comply with harmful requests, including generating content related to violence, illegal activities, and other harmful behaviours. Use responsibly and in accordance with applicable laws and regulations. The authors do not condone or encourage the use of this model for harmful purposes.

---

<small>While I have taken the time to verify all results thoroughly, I am open to any corrections, additional benchmarks, or further analysis. If you spot something that looks wrong and can be confirmed, I am happy to fix it.</small>
