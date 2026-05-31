#!/bin/bash
echo "=== Benchmarks ==="
for model in base heretic hauhau huihui aeon abliterix; do
  echo "Model: $model"
  jq -r '.results | "MMLU: \(.mmlu["acc,none"])\nHellaSwag: \(.hellaswag["acc_norm,none"])\nARC: \(.arc_challenge["acc,none"])\nWinoGrande: \(.winogrande["acc,none"])\nTruthfulQA: \(.truthfulqa_mc2["acc,none"])\nPiQA: \(.piqa["acc,none"])\nGSM8K: \(.gsm8k["exact_match,flexible-extract"])\nLambada: \(.lambada_openai["perplexity,none"])"' results/lm_eval/lm_eval_${model}.json
  echo ""
done

echo "=== HarmBench ==="
for model in base heretic hauhau huihui aeon abliterix; do
  echo "Model: $model"
  jq -r '{total_asr, empty_count, asr_by_category, full_cot_asr} | "Total ASR: \(.total_asr)\nEmpty: \(.empty_count)\nFull CoT ASR: \(.full_cot_asr)\nCategory ASR: \(.asr_by_category)"' results/qwen36-27b/harmbench/harmbench_${model}_scores.json
  echo ""
done

echo "=== KL Divergence ==="
for model in heretic hauhau huihui aeon abliterix; do
  echo "Model: $model"
  jq -r '.kl_divergence_batchmean' results/kl/kl_${model}.json
  echo ""
done
