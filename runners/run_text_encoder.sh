#!/usr/bin/env bash
# Text-encoder conditioning fidelity analysis runner.
# Runs run_analysis.py inside the abliterlitics-te Docker image (CPU by default).
# Models are ComfyUI-format safetensors; loading uses ComfyUI's own model classes.
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
  Uses ComfyUI's own model classes on ComfyUI-format safetensors.

Requires comparison.json settings.tebench:
  backend, base_te_file, dit_file, (comfyui_path)

Options:
   --gpu GPU          Use GPU instead of CPU (default: CPU)
   --limit N          Limit prompt count (testing)
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
LIMIT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)           GPU="$2"; shift 2 ;;
        --limit)         LIMIT="$2"; shift 2 ;;
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

# Guard: tebench must be configured
if [[ -z "$TEBENCH_BACKEND" ]]; then
    error "tebench not configured — add 'settings.tebench' block to comparison.json"
    exit 1
fi
if [[ -z "$TEBENCH_BASE_TE_FILE" || -z "$TEBENCH_DIT_FILE" ]]; then
    error "tebench requires base_te_file and dit_file in comparison.json"
    exit 1
fi

# Default ComfyUI path (host side)
COMFYUI_HOST_PATH="${TEBENCH_COMFYUI_PATH:-/home/maxx/comfyui-docker/comfyui}"
if [[ ! -d "$COMFYUI_HOST_PATH/comfy" ]]; then
    error "ComfyUI source not found at: $COMFYUI_HOST_PATH (expected comfy/ subdir)"
    exit 1
fi

# Detect the models root so te_file paths outside the comparison dir can be mounted.
# TE/DIT files typically live in the ComfyUI models tree (outside the repo).
MODELS_HOST_ROOT="$(dirname "$(dirname "$TEBENCH_DIT_FILE")")"
if [[ ! -d "$MODELS_HOST_ROOT" ]]; then
    error "Models host root not found: ${MODELS_HOST_ROOT} (derived from dit_file)"
    exit 1
fi
log "Models host root detected: ${MODELS_HOST_ROOT}"

# Validate TE/DIT files exist on the host before launching containers
for f in "$TEBENCH_BASE_TE_FILE" "$TEBENCH_DIT_FILE"; do
    if [[ ! -f "$f" ]]; then
        error "File not found: $f"
        exit 1
    fi
done

# Initialize DB tracking
if [[ $DRY_RUN -eq 0 ]]; then
    db_init "$COMP_DIR"
    db_stage_start "$COMPARISON_NAME" "all" "text_encoder"
fi

# Translate host paths to container paths.
# Mounts: comparison dir -> /comparison, models root -> /models, comfyui -> /comfyui
host_to_container() {
    local host_path="$1"
    if [[ "$host_path" == "$COMPARISON_DIR"/* ]]; then
        echo "/comparison/${host_path#$COMPARISON_DIR/}"
    elif [[ "$host_path" == "$MODELS_HOST_ROOT"/* ]]; then
        echo "/models/${host_path#$MODELS_HOST_ROOT/}"
    elif [[ "$host_path" == "$COMFYUI_HOST_PATH"/* ]]; then
        echo "/comfyui/${host_path#$COMFYUI_HOST_PATH/}"
    else
        echo "$host_path"
    fi
}

PROMPT_SUITE_PATH="${ABL_ROOT}/prompts/te_suite.yaml"

# Run for each variant
for variant in $VARIANT_NAMES; do
    var_slug="$(echo "$variant" | tr '-' '_')"
    skip_var="VARIANT_SKIP_TE_${var_slug}"
    te_file_var="VARIANT_TE_FILE_${var_slug}"
    if [[ "${!skip_var}" == "1" ]]; then
        log "Skipping text-encoder analysis for ${variant} (skip_te=true)"
        continue
    fi
    if [[ -z "${!te_file_var}" ]]; then
        log "Skipping text-encoder analysis for ${variant} (no te_file set)"
        continue
    fi

    result_file="${TE_RESULTS_DIR}/te_${variant}.json"

    if result_exists "$result_file"; then
        log "Skipping (exists): te_${variant}.json"
        continue
    fi

    log "Running text-encoder analysis: base -> ${variant}"

    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [dry-run] base=${TEBENCH_BASE_TE_FILE} variant=${!te_file_var} dit=${TEBENCH_DIT_FILE}"
        continue
    fi

    docker_args=(
        -v "${COMPARISON_DIR}:/comparison:ro"
        -v "${MODELS_HOST_ROOT}:/models:ro"
        -v "${COMFYUI_HOST_PATH}:/comfyui:ro"
        -v "${RESULTS_DIR}:/results"
        -v "${ABL_ROOT}:/app:ro"
        -v "${PROMPT_SUITE_PATH}:/prompts/te_suite.yaml:ro"
        -v "${TE_CACHE_DIR}:/te_cache"
        -e PYTHONPATH=/app:/comfyui
    )

    cmd_args=(
        /app/src/text_encoder/run_analysis.py
        --base "$(host_to_container "$TEBENCH_BASE_TE_FILE")"
        --variant "$(host_to_container "${!te_file_var}")"
        --dit "$(host_to_container "$TEBENCH_DIT_FILE")"
        --backend "${TEBENCH_BACKEND}"
        --comfyui /comfyui
        --prompts /prompts/te_suite.yaml
        --results-dir "/results/${COMPARISON_NAME}/text_encoder"
        --cache-dir /te_cache
        --device "${TEBENCH_DEVICE}"
        --n-boot "${TEBENCH_NUM_BOOTSTRAP}"
        --variant-name "${variant}"
    )
    if [[ "$LIMIT" != "0" ]]; then
        cmd_args+=(--limit "$LIMIT")
    fi

    if [[ -n "$GPU" ]]; then
        docker_run "$GPU" "$TE_IMAGE" "${docker_args[@]}" -- "${cmd_args[@]}"
    else
        docker_run_cpu "$TE_IMAGE" "${docker_args[@]}" -- "${cmd_args[@]}"
    fi
done

# Import results to DB
if [[ $DRY_RUN -eq 0 ]]; then
    db_import_results "$COMP_DIR"
    db_stage_complete "$COMPARISON_NAME" "all" "text_encoder"
fi

log "Text-encoder analysis complete"
