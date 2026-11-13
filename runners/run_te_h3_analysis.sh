#!/usr/bin/env bash
# H3 conditioning analysis runner — base vs variant TE at the H3 interface
# (RAW_HIDDEN 50 layers + consumed_hidden, PROJECTED, CONNECTOR_OUT).
# Phase 7 T2 of the te-h3 battery. CPU-only (32B TE, ~33GB int8 weights).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <comparison_dir>

Runs src.text_encoder.run_analysis with --backend h3 for the BASE and each
variant TE with a te_file, writing te_<name>.json to
results/<name>/text_encoder/.

Options:
   --variant NAME     Analyze only this variant (default: all with te_file)
   --limit N          Cap prompt count (debug)
   --force            Rerun even if results exist (run_analysis skips via cache)
   --dry-run          Preview
   -h, --help         Show this help
EOF
    exit 0
}

EXTRA_ARGS=()
DRY_RUN=0
COMP_DIR=""
VARIANT_FILTER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit) EXTRA_ARGS+=("--limit" "$2"); shift 2 ;;
        --variant) VARIANT_FILTER="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage ;;
        -*) error "Unknown option: $1"; usage ;;
        *)
            [[ -z "${COMP_DIR}" ]] && { COMP_DIR="$1"; } || { error "Multiple comparison dirs given"; usage; }
            shift ;;
    esac
done
[[ -z "${COMP_DIR}" ]] && { error "comparison_dir required"; usage; }
[[ ! -d "${COMP_DIR}" ]] && { error "Comparison dir not found: ${COMP_DIR}"; exit 1; }

load_comparison "$COMP_DIR"
COMPARISON_DIR="$(cd "$(dirname "${COMPARISON_DIR}")" && pwd)/$(basename "${COMPARISON_DIR}")"
RESULTS_DIR="${COMPARISON_DIR}/results/${COMPARISON_NAME}/text_encoder"
CACHE_DIR="${COMPARISON_DIR}/results/.te_cache"
mkdir -p "${RESULTS_DIR}" "${CACHE_DIR}"

BACKEND="${TEBENCH_BACKEND:-h3}"
BASE_TE="${TEBENCH_BASE_TE_FILE:-}"
DIT_FILE="${TEBENCH_DIT_FILE:-}"
COMFYUI_PATH="${TEBENCH_COMFYUI_PATH:-${HOME}/comfyui-docker/comfyui}"
# ComfyUI models dir, mounted 1:1 at its own absolute path (host == container)
# so comfy_te_name() can relativize te_file entries (AGENTS.md #40). Env
# override replaces the machine-specific default (audit 2026-10-14).
COMFYUI_MODELS_DIR="${COMFYUI_MODELS_DIR:-${HOME}/extra/comfyui-models}"
PROMPT_SUITE="${TEBENCH_PROMPT_SUITE:-prompts/te_h3_suite.yaml}"

[[ "${BACKEND}" == "h3" ]] || { error "This runner is h3-only (backend=${BACKEND})"; exit 1; }
[[ -n "${BASE_TE}" ]] || { error "TEBENCH_BASE_TE_FILE not set (settings.tebench.base_te_file)"; exit 1; }
[[ -n "${DIT_FILE}" ]] || { error "TEBENCH_DIT_FILE not set (settings.tebench.dit_file)"; exit 1; }

# Variants: NAME=TE_FILE from VARIANT_TE_FILE_<name> exports
declare -A VARIANTS=()
while read -r envline; do
    name="${envline#VARIANT_TE_FILE_}"
    name="${name%%=*}"
    te_file="${envline#*=}"
    [[ -n "${te_file}" ]] || continue
    VARIANTS["${name}"]="${te_file}"
done < <(env | grep -E '^VARIANT_TE_FILE_[a-z0-9_]+=' || true)

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "DRY-RUN: backend=${BACKEND} base=$(basename "${BASE_TE}") dit=$(basename "${DIT_FILE}")"
    log "DRY-RUN: variants: ${!VARIANTS[@]}"
    exit 0
fi

te_readable() {
    # Host-side probe: packed-quant files (nvfp4_awq) are unreadable by
    # safetensors — analysis cannot run on them (generation-only arms).
    # Missing/corrupt files are a hard error, only header-unreadable skips.
    local f="$1"
    [[ -f "${f}" ]] || { error "TE file not found: ${f}"; return 2; }
    python3 - "${f}" <<'PYEOF'
import sys
from safetensors import safe_open
try:
    with safe_open(sys.argv[1], framework="pt") as f:
        _ = len(list(f.keys()))
    print("ok")
except Exception as e:
    if isinstance(e, FileNotFoundError):
        print("missing")
    else:
        print("unreadable")
PYEOF
}

run_one() {
    local name="$1" te_file="$2"
    if [[ "$(te_readable "${te_file}")" == "unreadable" ]]; then
        log "SKIP ${name}: $(basename "${te_file}") unreadable host-side (packed quant) — generation-only arm"
        return 0
    fi
    log "Analyzing ${name}: $(basename "${te_file}")"
    docker_run_cpu "${TE_IMAGE}" \
        -v "${COMPARISON_DIR}:/comp:rw" \
        -v "${ABL_ROOT}:/app:ro" \
        -v "${COMFYUI_PATH}:/comfyui:ro" \
        -v "${COMFYUI_MODELS_DIR}:${COMFYUI_MODELS_DIR}:ro" \
        -e PYTHONPATH=/app \
        -- \
        -m src.text_encoder.run_analysis \
            --base "${BASE_TE}" \
            --variant "${te_file}" \
            --variant-name "${name}" \
            --dit "${DIT_FILE}" \
            --backend "${BACKEND}" \
            --comfyui /comfyui \
            --prompts "/app/${PROMPT_SUITE}" \
            --results-dir "/comp/results/${COMPARISON_NAME}/text_encoder" \
            --cache-dir "/comp/results/.te_cache" \
            --device cpu \
            ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
}

run_one base "${BASE_TE}"
if [[ -n "${VARIANT_FILTER}" ]]; then
    [[ -n "${VARIANTS[${VARIANT_FILTER}]+x}" ]] || { error "Unknown variant: ${VARIANT_FILTER}"; exit 1; }
    [[ "${VARIANT_FILTER}" == "base" ]] && exit 0
    run_one "${VARIANT_FILTER}" "${VARIANTS[${VARIANT_FILTER}]}"
else
    for name in "${!VARIANTS[@]}"; do
        [[ "${name}" == "base" ]] && continue
        run_one "${name}" "${VARIANTS[${name}]}"
    done
fi

log "H3 analysis complete: ${RESULTS_DIR}"
