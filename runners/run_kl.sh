#!/usr/bin/env bash
# KL divergence runner — two-phase: collect base logits, then variant logits + KL
# Uses Docker forensics image for GPU access.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <comparison_dir>

KL divergence analysis (2 phases):
  1. base      — Collect base model first-token logits
  2. variants  — Collect variant logits and compute KL divergence

Options:
   --gpu GPU          GPU index (default: auto-detect)
   --num-prompts N    Number of prompts from dataset (default: 100)
   --skip-existing    Skip existing results (default)
   --force            Overwrite existing results
   --dry-run          Show what would run without executing
   -h, --help         Show this help
EOF
    exit 0
}

GPU=""
NUM_PROMPTS=""
DRY_RUN=0
SKIP_EXISTING=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)           GPU="$2"; shift 2 ;;
        --num-prompts)   NUM_PROMPTS="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING=1; shift ;;
        --force)         SKIP_EXISTING=0; shift ;;
        --dry-run)       DRY_RUN=1; shift ;;
        -h|--help)       usage ;;
        *)               break ;;
    esac
done

COMP_DIR="${1:?Error: comparison_dir required. Use --help for usage.}"

load_comparison "$COMP_DIR"
GPU="${GPU:-$(detect_best_gpu)}"
COMPARISON_DIR="$(cd "$(dirname "${COMPARISON_DIR}")" && pwd)/$(basename "${COMPARISON_DIR}")"
BASE_DIR="$(cd "${BASE_DIR}" && pwd)"
for variant in $VARIANT_NAMES; do
    variant_slug="$(echo "$variant" | tr '-' '_')"
    variant_path_var="VARIANT_PATH_${variant_slug}"
    eval "VARIANT_PATH_${variant_slug}=\"\$(cd \"${!variant_path_var}\" && pwd)\""
done
RESULTS_DIR="${COMPARISON_DIR}/results/kl"
mkdir -p "$RESULTS_DIR"

EXTRA_ARGS=()
[[ -n "$NUM_PROMPTS" ]] && EXTRA_ARGS+=(--num-prompts "$NUM_PROMPTS")

# Phase 1: Collect base model logits
BASE_LOGITS="${RESULTS_DIR}/logits_base.pt"
if result_exists "$BASE_LOGITS"; then
    log "Skipping (exists): base logits"
else
    log "Phase 1: Collecting base logits from ${BASE_DIR}"
    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [dry-run] docker_run ${GPU} ${FORENSICS_IMAGE} kl_divergence.py collect --model /model --output /results/logits_base.pt"
    else
        docker_run "$GPU" "$FORENSICS_IMAGE" \
            -v "${BASE_DIR}:/model:ro" \
            -v "${RESULTS_DIR}:/results" \
            -v "${ABL_ROOT}:/app:ro" \
            -e PYTHONPATH=/app \
            -- \
            /app/src/kl/kl_divergence.py collect \
                --model /model \
                --output /results/logits_base.pt \
                --response-prefix auto \
                --save-prefix /results/response_prefix.txt \
                "${EXTRA_ARGS[@]}"
    fi
fi

# Phase 2: Collect variant logits + compute KL divergence
for variant in $VARIANT_NAMES; do
    variant_path_var="VARIANT_PATH_$(echo "$variant" | tr '-' '_')"
    variant_path="${!variant_path_var}"
    variant_logits="${RESULTS_DIR}/logits_${variant}.pt"

    # Phase 2a: Collect variant logits
    if result_exists "$variant_logits"; then
        log "Skipping (exists): logits for ${variant}"
    else
        log "Phase 2a: Collecting logits for variant ${variant}"
        if [[ $DRY_RUN -eq 1 ]]; then
            log "  [dry-run] docker_run ${GPU} ${FORENSICS_IMAGE} kl_divergence.py collect --model /model --output /results/logits_${variant}.pt"
        else
            docker_run "$GPU" "$FORENSICS_IMAGE" \
                -v "${variant_path}:/model:ro" \
                -v "${BASE_DIR}:/tokenizer:ro" \
                -v "${RESULTS_DIR}:/results" \
                -v "${ABL_ROOT}:/app:ro" \
                -e PYTHONPATH=/app \
                -- \
                /app/src/kl/kl_divergence.py collect \
                    --model /model \
                    --tokenizer /tokenizer \
                    --output "/results/logits_${variant}.pt" \
                    --response-prefix "$(cat "${RESULTS_DIR}/response_prefix.txt" 2>/dev/null || echo none)" \
                    "${EXTRA_ARGS[@]}"
        fi
    fi

    # Phase 2b: Compute KL divergence
    out_file="${RESULTS_DIR}/kl_${variant}.json"
    if result_exists "$out_file"; then
        log "Skipping (exists): KL for ${variant}"
        continue
    fi

    log "Phase 2b: Computing KL divergence for variant ${variant}"
    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [dry-run] docker_run ${GPU} ${FORENSICS_IMAGE} kl_divergence.py compute --base-logits /results/logits_base.pt --variant-logits /results/logits_${variant}.pt"
        continue
    fi

    docker_run "$GPU" "$FORENSICS_IMAGE" \
        -v "${RESULTS_DIR}:/results" \
        -v "${ABL_ROOT}:/app:ro" \
        -e PYTHONPATH=/app \
        -- \
        /app/src/kl/kl_divergence.py compute \
            --base-logits /results/logits_base.pt \
            --variant-logits "/results/logits_${variant}.pt" \
            --variant-label "${variant}" \
            --output "/results/kl_${variant}.json"
done

log "KL analysis complete"
