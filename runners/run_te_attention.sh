#!/usr/bin/env bash
# Attention readout runner — what does the DiT READ per TE variant (Phase 4 T2.2).
# Uses ComfyUI's optimized_attention_override (B0 spike GO, journal #123).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <comparison_dir>

Options:
   --variants LIST   Comma-separated variant names (default: heretic,base_int8,base_int4,base_gguf_q3_k_m,base_gguf_q8_0)
   --n-steps N       Euler steps (default 20)
   --layer-sample N  Compute readout summaries every Nth step (default 4)
   --force           Overwrite existing tebench_attention_*.json
   --dry-run         Preview
   -h, --help        Show this help
EOF
    exit 0
}

ATTN_VARIANTS="heretic,base_int8,base_int4,base_gguf_q3_k_m,base_gguf_q8_0"
N_STEPS=20
LAYER_SAMPLE=4
FORCE=0
DRY_RUN=0
COMP_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --variants) ATTN_VARIANTS="$2"; shift 2 ;;
        --n-steps) N_STEPS="$2"; shift 2 ;;
        --layer-sample) LAYER_SAMPLE="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage ;;
        -*) error "Unknown option: $1"; usage ;;
        *) [[ -z "$COMP_DIR" ]] && { COMP_DIR="$1"; } || { error "Multiple comparison dirs"; usage; }
           shift ;;
    esac
done
[[ -z "$COMP_DIR" ]] && { error "comparison_dir required"; usage; }

load_comparison "$COMP_DIR"
COMPARISON_DIR="$(cd "$(dirname "${COMPARISON_DIR}")" && pwd)/$(basename "${COMPARISON_DIR}")"
RESULTS_DIR="${COMPARISON_DIR}/results/${COMPARISON_NAME}/text_encoder"
TE_CACHE_DIR="${COMPARISON_DIR}/.te_cache"
MODELS_HOST_ROOT="$(dirname "$(dirname "$TEBENCH_DIT_FILE")")"
COMFYUI_HOST_PATH="${TEBENCH_COMFYUI_PATH:-/home/maxx/comfyui-docker/comfyui}"

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

map_list_to_container() {
    # Map EACH comma-separated host path individually, rejoin with commas.
    local joined=""
    local item
    IFS=',' read -ra _items <<< "$1"
    for item in "${_items[@]}"; do
        joined="${joined:+${joined},}$(host_to_container "${item}")"
    done
    echo "${joined}"
}

# Resolve variant list -> files + names, skipping existing unless --force
files=""
names=""
for v in ${ATTN_VARIANTS//,/ }; do
    slug="$(echo "$v" | tr '-' '_')"
    te_var="VARIANT_TE_FILE_${slug}"
    f="${!te_var:-}"
    [[ -z "${f}" || ! -f "${f}" ]] && { error "No te_file for variant ${v}"; exit 1; }
    if [[ "${FORCE}" -eq 0 && -f "${RESULTS_DIR}/tebench_attention_${v}.json" ]]; then
        log "Skipping (exists): tebench_attention_${v}.json"
        continue
    fi
    files="${files:+${files},}${f}"
    names="${names:+${names},}${v}"
done
[[ -z "${names}" ]] && { log "Nothing to do (all attention results exist)"; exit 0; }

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "DRY-RUN: attention for ${names}"
    exit 0
fi

GPU="${GPU:-0}"
mkdir -p "$RESULTS_DIR"
log "Running attention readout for ${COMPARISON_NAME}: ${names}"

docker_run "$GPU" "$TE_IMAGE" \
    -v "${COMPARISON_DIR}:/comparison:ro" \
    -v "${MODELS_HOST_ROOT}:/models:ro" \
    -v "${COMFYUI_HOST_PATH}:/comfyui:ro" \
    -v "${RESULTS_DIR}:/results" \
    -v "${ABL_ROOT}:/app:ro" \
    -v "${ABL_ROOT}/prompts/te_suite.yaml:/prompts/te_suite.yaml:ro" \
    -v "${TE_CACHE_DIR}:/te_cache" \
    -e PYTHONPATH=/app \
    -- \
    /app/src/text_encoder/run_attention.py \
        --base "$(host_to_container "$TEBENCH_BASE_TE_FILE")" \
        --variants "$(map_list_to_container "${files}")" \
        --variant-names "${names}" \
        --dit "$(host_to_container "$TEBENCH_DIT_FILE")" \
        --cache-dir /te_cache \
        --results-dir /results \
        --device cuda \
        --n-steps "${N_STEPS}" \
        --layer-sample "${LAYER_SAMPLE}"

log "Attention readout complete: ${RESULTS_DIR}/tebench_attention_*.json"
