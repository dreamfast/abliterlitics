#!/bin/bash
for model in base heretic hauhau huihui aeon abliterix; do
  echo "Model: $model"
  jq -r '{asr: .asr, asr_pct: .asr_pct}' results/qwen36-27b/harmbench/harmbench_${model}_scores.json
done
