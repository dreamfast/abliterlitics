#!/usr/bin/env bash
# Control battery runner — counterfactual conditioning divergence (Phase 4 T3.2).
# Gaussian dose curve, rank-1 control, taboo injection, alpha-scaled heretic
# delta, Spingarn shuffles, null anchor — through the divergence harness.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <comparison_dir>

Options:
   --heretic NAME        Heretic variant name (default: heretic)
   --dose-variants LIST  Comma-separated variant names defining max dose
   --conditions LIST     Condition families (default: noise,rank1,taboo,heretic_alpha,order_shuffle,embed_shuffle,null)
   --n-steps N           Euler steps (default 20)
   --force               Overwrite tebench_controls.json
   --dry-run             Preview
   -h, --help            Show this help
EOF
    exit 0
}

HERETIC="heretic"
DOSE_VARIANTS=""
CONDITIONS="noise,rank1,taboo,heretic_alpha,order_shuffle,embed_shuffle,null"
N_STEPS=20
FORCE=0
DRY_RUN=0
COMP_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --heretic) HERETIC="$2"; shift 2 ;;
        --dose-variants) DOSE_VARIANTS="$2"; shift 2 ;;
        --conditions) CONDITIONS="$2"; shift 2 ;;
        --n-steps) N_STEPS="$2"; shift 2 ;;
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
GPU="${GPU:-0}"

if [[ -f "${RESULTS_DIR}/tebench_controls.json" && "${FORCE}" -eq 0 ]]; then
    log "tebench_controls.json exists — use --force to overwrite"
    exit 0
fi

# Variant TE resolution: _load_comparison exports VARIANT_TE_FILE_<slug>
heretic_slug="$(echo "${HERETIC}" | tr '-' '_')"
heretic_te_var="VARIANT_TE_FILE_${heretic_slug}"
heretic_te_host="${!heretic_te_var:-}"
[[ -z "${heretic_te_host}" || ! -f "${heretic_te_host}" ]] && { error "No te_file for variant ${HERETIC}"; exit 1; }

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

dose_args=""
if [[ -n "${DOSE_VARIANTS}" ]]; then
    files=""
    for v in ${DOSE_VARIANTS//,/ }; do
        slug="$(echo "$v" | tr '-' '_')"
        te_var="VARIANT_TE_FILE_${slug}"
        f="${!te_var:-}"
        [[ -z "${f}" || ! -f "${f}" ]] && { error "No te_file for dose variant ${v}"; exit 1; }
        files="${files:+${files},}${f}"
    done
    dose_args="--dose-variants $(map_list_to_container "${files}")"
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "DRY-RUN: controls heretic=${HERETIC} conditions=${CONDITIONS}"
    exit 0
fi

log "Running control battery for ${COMPARISON_NAME} (heretic=${HERETIC})"

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
    /app/src/text_encoder/run_controls.py \
        --base "$(host_to_container "$TEBENCH_BASE_TE_FILE")" \
        --heretic "$(host_to_container "${heretic_te_host}")" \
        ${dose_args} \
        --dit "$(host_to_container "$TEBENCH_DIT_FILE")" \
        --cache-dir /te_cache \
        --results-dir /results \
        --device cuda \
        --n-steps "${N_STEPS}" \
        --conditions "${CONDITIONS}"

log "Control battery complete: ${RESULTS_DIR}/tebench_controls.json"
