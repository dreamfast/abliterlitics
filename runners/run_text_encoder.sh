#!/usr/bin/env bash
# Text-encoder conditioning fidelity analysis runner.
# Runs run_analysis.py inside the abliterlitics-te Docker image (CPU by default).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <comparison_dir>

Text-encoder conditioning fidelity analysis (tebench):
  Measures how much an abliterated/quantized text encoder perturbs the
  conditioning tensor that enters a diffusion transformer's cross-attention.

Options:
   --gpu GPU          Use GPU instead of CPU (default: CPU)
   --skip-existing    Skip if results exist (default)
   --force            Overwrite existing results
   --dry-run          Show what would run without executing
   -h, --help         Show this help
EOF
    exit 0
}

GPU=""
DRY_RUN=0
SKIP_EXISTING=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)           GPU="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING=1; shift ;;
        --force)         SKIP_EXISTING=0; shift ;;
        --dry-run)       DRY_RUN=1; shift ;;
        -h|--help)       usage ;;
        *)               break ;;
    esac
done

COMP_DIR="${1:?Error: comparison_dir required. Use --help for usage.}"

load_comparison "$COMP_DIR"
COMPARISON_DIR="$(cd "$(dirname "${COMPARISON_DIR}")" && pwd)/$(basename "${COMPARISON_DIR}")"
RESULTS_DIR="${COMPARISON_DIR}/results"
TE_RESULTS_DIR="${RESULTS_DIR}/${COMPARISON_NAME}/text_encoder"
TE_CACHE_DIR="${COMPARISON_DIR}/.te_cache"
mkdir -p "$RESULTS_DIR" "$TE_RESULTS_DIR" "$TE_CACHE_DIR"

# Initialize DB tracking
if [[ $DRY_RUN -eq 0 ]]; then
    db_init "$COMP_DIR"
    db_stage_start "$COMPARISON_NAME" "all" "text_encoder"
fi

# Determine docker mode: CPU by default, GPU if requested
USE_GPU=0
if [[ -n "$GPU" ]]; then
    USE_GPU=1
fi

PROMPT_SUITE_PATH="${ABL_ROOT}/prompts/te_suite.yaml"
if [[ "$TEBENCH_PROMPT_SUITE" != "default" && -n "$TEBENCH_PROMPT_SUITE" ]]; then
    PROMPT_SUITE_PATH="${COMPARISON_DIR}/${TEBENCH_PROMPT_SUITE}"
fi

# Run for each variant
for variant in $VARIANT_NAMES; do
    var_slug="$(echo "$variant" | tr '-' '_')"
    skip_var="VARIANT_SKIP_TE_${var_slug}"
    if [[ "${!skip_var}" == "1" ]]; then
        log "Skipping text-encoder analysis for ${variant} (skip_te=true)"
        continue
    fi

    variant_path_var="VARIANT_PATH_${var_slug}"
    result_file="${TE_RESULTS_DIR}/te_${variant}.json"

    if result_exists "$result_file"; then
        log "Skipping (exists): te_${variant}.json"
        continue
    fi

    log "Running text-encoder analysis for ${variant}"

    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [dry-run] text-encoder analysis: base -> ${variant}"
        continue
    fi

    docker_args=(
        -v "${COMPARISON_DIR}:/comparison:ro"
        -v "${ABL_ROOT}/models:/models:ro"
        -v "${RESULTS_DIR}:/results"
        -v "${ABL_ROOT}:/app:ro"
        -v "${PROMPT_SUITE_PATH}:/prompts/te_suite.yaml:ro"
        -v "${TE_CACHE_DIR}:/te_cache"
        -e PYTHONPATH=/app
        -e TEBENCH_BACKEND="${TEBENCH_BACKEND}"
        -e TEBENCH_DEVICE="${TEBENCH_DEVICE}"
        -e TEBENCH_BATCH_SIZE="${TEBENCH_BATCH_SIZE}"
        -e TEBENCH_MAX_LENGTH="${TEBENCH_MAX_LENGTH}"
        -e TEBENCH_NUM_BOOTSTRAP="${TEBENCH_NUM_BOOTSTRAP}"
    )

    if [[ -n "$TEBENCH_DIT_MODEL_DIR" ]]; then
        docker_args+=(-e TEBENCH_DIT_MODEL_DIR="${TEBENCH_DIT_MODEL_DIR}")
    fi

    if [[ $USE_GPU -eq 1 ]]; then
        docker_run "$GPU" "$TE_IMAGE" \
            "${docker_args[@]}" \
            -- \
            python3 /app/src/text_encoder/run_analysis.py \
                --base "${BASE_DIR}" \
                --variant "${!variant_path_var}" \
                --backend "${TEBENCH_BACKEND}" \
                --prompts /prompts/te_suite.yaml \
                --results-dir "/results/${COMPARISON_NAME}/text_encoder" \
                --cache-dir /te_cache \
                --device "${TEBENCH_DEVICE}" \
                --batch-size "${TEBENCH_BATCH_SIZE}" \
                --max-length "${TEBENCH_MAX_LENGTH}" \
                --n-boot "${TEBENCH_NUM_BOOTSTRAP}" \
                --variant-name "${variant}"
    else
        docker_run_cpu "$TE_IMAGE" \
            "${docker_args[@]}" \
            -- \
            python3 /app/src/text_encoder/run_analysis.py \
                --base "${BASE_DIR}" \
                --variant "${!variant_path_var}" \
                --backend "${TEBENCH_BACKEND}" \
                --prompts /prompts/te_suite.yaml \
                --results-dir "/results/${COMPARISON_NAME}/text_encoder" \
                --cache-dir /te_cache \
                --device "${TEBENCH_DEVICE}" \
                --batch-size "${TEBENCH_BATCH_SIZE}" \
                --max-length "${TEBENCH_MAX_LENGTH}" \
                --n-boot "${TEBENCH_NUM_BOOTSTRAP}" \
                --variant-name "${variant}"
    fi
done

# Import results to DB
if [[ $DRY_RUN -eq 0 ]]; then
    db_import_results "$COMP_DIR"
    db_stage_complete "$COMPARISON_NAME" "all" "text_encoder"
fi

log "Text-encoder analysis complete"
