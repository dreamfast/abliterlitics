#!/usr/bin/env bash
# Driver for src/kl/kl_experiment.py — the Dolly + 32-token robustness proof.
# Standalone (does NOT touch production run_kl.sh / kl_divergence.py).
#
# Usage:
#   bash runners/run_kl_experiment.sh ./comparisons/gemma4-e4b/
#   bash runners/run_kl_experiment.sh ./comparisons/gemma4-e4b/ "heretic,nullpo,huihui"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

COMP_DIR="${1:?Error: comparison_dir required.}"
# Representative spread across the KL range + heretic/non-heretic divide.
VARIANTS="${2:-heretic,nullpo,trevorjs,huihui,abliterix,treadon-combo,bendernina,obliteratus}"

# Prompt source + results subdir (override via env). gpqa needs the offline
# datasets cache mounted; dolly works unauthenticated from the hub.
PROMPT_SOURCE="${PROMPT_SOURCE:-dolly}"
RESULTS_NAME="${RESULTS_NAME:-kl_experiment}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
THINKING="${THINKING:-enabled}"   # enabled | disabled
# Deterministic response prefix. Auto-detection is flaky per-run (sometimes '',
# sometimes '<|channel>thought...'), which scores different runs at different token
# positions and confounds cross-dataset absolutes. Default 'none' scores the first
# generated token consistently, matching production (harmless_alpaca detected '').
RESPONSE_PREFIX="${RESPONSE_PREFIX:-none}"

load_comparison "$COMP_DIR"
GPU="${GPU:-0}"   # RTX 5090 (32GB). nvidia-smi is absent on host -> hardcode 0.

# Resolve absolute paths (loader emits relative-to-comp-dir paths).
BASE_DIR="$(cd "${BASE_DIR}" && pwd)"
TOKENIZER_DIR="$(cd "${TOKENIZER_DIR:-${BASE_DIR}}" && pwd)"
COMPARISON_DIR="$(cd "$(dirname "${COMPARISON_DIR}")" && pwd)/$(basename "${COMPARISON_DIR}")"

EXP_RESULTS="${COMPARISON_DIR}/results/${RESULTS_NAME}"
mkdir -p "$EXP_RESULTS"

log "KL experiment: comparison=${COMPARISON_NAME} gpu=${GPU} source=${PROMPT_SOURCE}"
log "  base    = ${BASE_DIR}"
log "  tokenizer = ${TOKENIZER_DIR}"
log "  variants= ${VARIANTS}"
log "  results = ${EXP_RESULTS}"

# Dataset cache mount + offline flags (required for gpqa; harmless for dolly).
HF_CACHE_DIR="$(cd "${ABL_ROOT}" && pwd)/.cache/hf"

# --------------------------------------------------------------------------
# Phase 1: collect base (continuations + base logprobs)
# --------------------------------------------------------------------------
if [[ -f "${EXP_RESULTS}/exp_base.pt" ]]; then
    log "Skipping collect-base (exp_base.pt exists)"
else
    log "Phase 1: collect-base"
    docker_run "$GPU" "$FORENSICS_IMAGE" \
        -v "${BASE_DIR}:/model:ro" \
        -v "${TOKENIZER_DIR}:/tokenizer:ro" \
        -v "${EXP_RESULTS}:/results" \
        -v "${ABL_ROOT}:/app:ro" \
        -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
        -e PYTHONPATH=/app \
        -e HF_HOME=/root/.cache/huggingface \
        -e HF_DATASETS_OFFLINE=1 -e HF_HUB_OFFLINE=1 \
        -- \
        /app/src/kl/kl_experiment.py collect-base \
            --model /model --tokenizer /tokenizer \
            --output /results/exp_base.pt \
            --num-prompts "${NUM_PROMPTS}" --num-tokens 32 \
            --prompt-source "${PROMPT_SOURCE}" \
            --thinking "${THINKING}" \
            --response-prefix "${RESPONSE_PREFIX}"
fi

# --------------------------------------------------------------------------
# Phase 2: per-variant collect + compute
# --------------------------------------------------------------------------
IFS=',' read -ra VAR_ARRAY <<< "$VARIANTS"
for variant in "${VAR_ARRAY[@]}"; do
    variant="${variant// /}"   # trim stray spaces
    [[ -z "$variant" ]] && continue
    variant_slug="$(echo "$variant" | tr '-' '_')"
    variant_path_var="VARIANT_PATH_${variant_slug}"
    variant_path="${!variant_path_var}"
    if [[ -z "${variant_path}" ]]; then
        error "no path for variant ${variant} (VARIANT_PATH_${variant_slug} unset)"; continue
    fi
    variant_path="$(cd "${variant_path}" && pwd)"

    out_json="${EXP_RESULTS}/exp_${variant}.json"
    if [[ -f "$out_json" ]]; then
        log "Skipping ${variant} (exp_${variant}.json exists)"
        continue
    fi
    log "Phase 2: collect-variant ${variant} (${variant_path})"
    docker_run "$GPU" "$FORENSICS_IMAGE" \
        -v "${variant_path}:/model:ro" \
        -v "${TOKENIZER_DIR}:/tokenizer:ro" \
        -v "${EXP_RESULTS}:/results" \
        -v "${ABL_ROOT}:/app:ro" \
        -e PYTHONPATH=/app \
        -- \
        /app/src/kl/kl_experiment.py collect-variant \
            --model /model --tokenizer /tokenizer \
            --base-data /results/exp_base.pt \
            --variant-label "${variant}" \
            --output "/results/exp_${variant}.json"
done

# --------------------------------------------------------------------------
# Phase 3: report (merge + compare to existing production numbers)
# --------------------------------------------------------------------------
log "Phase 3: report"
docker_run "$GPU" "$FORENSICS_IMAGE" \
    -v "${EXP_RESULTS}:/results" \
    -v "${COMPARISON_DIR}:/work:ro" \
    -v "${ABL_ROOT}:/app:ro" \
    -e PYTHONPATH=/app \
    -- \
    /app/src/kl/kl_experiment.py report \
        --experiment-dir /results \
        --existing-kl-dir /work/results/kl \
        --output /results/comparison.json

log "Done. See ${EXP_RESULTS}/comparison.json"
