#!/usr/bin/env bash
# Driver for src/kl/kl_methods.py — KL METHOD comparison (Heretic v1.4.0 / Apostate /
# Abliterix vs production Heretic v1.2.0), all on harmless_alpaca.
#
# Usage:
#   bash runners/run_kl_methods.sh ./comparisons/gemma4-e4b/
#   bash runners/run_kl_methods.sh ./comparisons/gemma4-e4b/ "heretic,nullpo,huihui"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

COMP_DIR="${1:?Error: comparison_dir required.}"
VARIANTS="${2:-heretic,nullpo,trevorjs,huihui,abliterix,treadon-combo,bendernina,obliteratus}"
RESULTS_NAME="${RESULTS_NAME:-kl_methods}"

load_comparison "$COMP_DIR"
GPU="${GPU:-0}"

BASE_DIR="$(cd "${BASE_DIR}" && pwd)"
TOKENIZER_DIR="$(cd "${TOKENIZER_DIR:-${BASE_DIR}}" && pwd)"
COMPARISON_DIR="$(cd "$(dirname "${COMPARISON_DIR}")" && pwd)/$(basename "${COMPARISON_DIR}")"

EXP_RESULTS="${COMPARISON_DIR}/results/${RESULTS_NAME}"
mkdir -p "$EXP_RESULTS"
HF_CACHE_DIR="$(cd "${ABL_ROOT}" && pwd)/.cache/hf"

log "KL methods: comparison=${COMPARISON_NAME} gpu=${GPU} variants=${VARIANTS}"

# Phase 1: collect-base (forward-pass last-1 + last-8 logprobs)
if [[ -f "${EXP_RESULTS}/methods_base.pt" ]]; then
    log "Skipping collect-base (methods_base.pt exists)"
else
    log "Phase 1: collect-base"
    docker_run "$GPU" "$FORENSICS_IMAGE" \
        -v "${BASE_DIR}:/model:ro" -v "${TOKENIZER_DIR}:/tokenizer:ro" \
        -v "${EXP_RESULTS}:/results" -v "${ABL_ROOT}:/app:ro" \
        -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
        -e PYTHONPATH=/app -e HF_HOME=/root/.cache/huggingface \
        -e HF_DATASETS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -- \
        /app/src/kl/kl_methods.py collect-base \
            --model /model --tokenizer /tokenizer --output /results/methods_base.pt
fi

# Phase 2: per-variant collect + compute
IFS=',' read -ra VAR_ARRAY <<< "$VARIANTS"
for variant in "${VAR_ARRAY[@]}"; do
    variant="${variant// /}"; [[ -z "$variant" ]] && continue
    variant_slug="$(echo "$variant" | tr '-' '_')"
    variant_path_var="VARIANT_PATH_${variant_slug}"
    variant_path="${!variant_path_var}"
    [[ -z "${variant_path}" ]] && { error "no path for ${variant}"; continue; }
    variant_path="$(cd "${variant_path}" && pwd)"
    out_json="${EXP_RESULTS}/methods_${variant}.json"
    if [[ -f "$out_json" ]]; then log "Skipping ${variant}"; continue; fi
    log "Phase 2: collect-variant ${variant}"
    docker_run "$GPU" "$FORENSICS_IMAGE" \
        -v "${variant_path}:/model:ro" -v "${TOKENIZER_DIR}:/tokenizer:ro" \
        -v "${EXP_RESULTS}:/results" -v "${ABL_ROOT}:/app:ro" \
        -e PYTHONPATH=/app -- \
        /app/src/kl/kl_methods.py collect-variant \
            --model /model --tokenizer /tokenizer \
            --base-data /results/methods_base.pt \
            --variant-label "${variant}" --output "/results/methods_${variant}.json"
done

# Phase 3: report
log "Phase 3: report"
docker_run "$GPU" "$FORENSICS_IMAGE" \
    -v "${EXP_RESULTS}:/results" -v "${COMPARISON_DIR}:/work:ro" -v "${ABL_ROOT}:/app:ro" \
    -e PYTHONPATH=/app -- \
    /app/src/kl/kl_methods.py report \
        --experiment-dir /results --existing-kl-dir /work/results/kl \
        --output /results/methods_comparison.json

log "Done. See ${EXP_RESULTS}/methods_comparison.json"
