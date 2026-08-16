#!/usr/bin/env bash
# H3 attention-readout runner — what do the AUDIO and VIDEO streams read from
# conditioning, base vs variant TE (phase 9). GPU 0 (5090) — the 21GB int8
# fl2va DiT does not fit the 4090 alongside ComfyUI. Results:
# tebench_attention_<variant>.json with jsd_per_segment (audio/video).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <comparison_dir>

Options:
   --variant NAME     Only this variant (default: all with te_file)
   --n-prompts N      Prompts to encode (default 2; 32B TE encode is slow on CPU)
   --n-steps N        Euler steps (default 4)
   --device DEV       cuda | cpu (default cuda; GPU 0)
   --dry-run          Preview
   -h, --help
EOF
    exit 0
}

N_PROMPTS=2
N_STEPS=4
DEVICE="cuda"
VARIANT_FILTER=""
DRY_RUN=0
COMP_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-prompts|--limit) N_PROMPTS="$2"; shift 2 ;;
        --n-steps) N_STEPS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
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
DIT_FILE="${TEBENCH_DIT_FILE:-}"
BASE_TE="${TEBENCH_BASE_TE_FILE:-}"
SUITE="${TEBENCH_PROMPT_SUITE:-prompts/te_h3_suite.yaml}"
COMFYUI_PATH="${TEBENCH_COMFYUI_PATH:-/home/maxx/comfyui-docker/comfyui}"

[[ "${TEBENCH_BACKEND:-}" == "h3" ]] || { error "This runner is h3-only"; exit 1; }
[[ -n "${DIT_FILE}" ]] || { error "TEBENCH_DIT_FILE not set"; exit 1; }
[[ -n "${BASE_TE}" ]] || { error "TEBENCH_BASE_TE_FILE not set"; exit 1; }

declare -A VARIANTS=()
while read -r envline; do
    name="${envline#VARIANT_TE_FILE_}"
    name="${name%%=*}"
    te_file="${envline#*=}"
    [[ -n "${te_file}" && -f "${te_file}" ]] || continue
    VARIANTS["${name}"]="${te_file}"
done < <(env | grep -E '^VARIANT_TE_FILE_[a-z0-9_]+=' || true)

VAR_ARGS=""
VAR_NAMES=""
if [[ -n "${VARIANT_FILTER}" ]]; then
    [[ -n "${VARIANTS[${VARIANT_FILTER}]+x}" ]] || { error "Unknown variant: ${VARIANT_FILTER}"; exit 1; }
    VAR_ARGS="${VARIANTS[${VARIANT_FILTER}]}"
    VAR_NAMES="${VARIANT_FILTER}"
else
    for name in "${!VARIANTS[@]}"; do
        [[ -z "${VAR_ARGS}" ]] && VAR_ARGS="${VARIANTS[${name}]}" || VAR_ARGS="${VAR_ARGS},${VARIANTS[${name}]}"
        [[ -z "${VAR_NAMES}" ]] && VAR_NAMES="${name}" || VAR_NAMES="${VAR_NAMES},${name}"
    done
fi
[[ -n "${VAR_ARGS}" ]] || { error "No variants with te_file"; exit 1; }

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "DRY-RUN: h3 attention base=$(basename "${BASE_TE}") dit=$(basename "${DIT_FILE}") variants=${VAR_NAMES}"
    exit 0
fi

log "H3 attention readout: variants=${VAR_NAMES} n_prompts=${N_PROMPTS} steps=${N_STEPS} device=${DEVICE}"

if [[ "${DEVICE}" == "cuda" ]]; then
    RUN_FN=docker_run
    RUN_EXTRA=("0")
else
    RUN_FN=docker_run_cpu
    RUN_EXTRA=()
fi

"${RUN_FN}" "${RUN_EXTRA[@]}" "${TE_IMAGE}" \
    -v "${COMPARISON_DIR}:/comp:rw" \
    -v "${ABL_ROOT}:/app:ro" \
    -v "${COMFYUI_PATH}:/comfyui:ro" \
    -v "/home/maxx/extra/comfyui-models:/home/maxx/extra/comfyui-models:ro" \
    -e PYTHONPATH=/app \
    -- \
    -m src.text_encoder.run_h3_attention \
        --dit "${DIT_FILE}" \
        --base "${BASE_TE}" \
        --variants "${VAR_ARGS}" \
        --variant-names "${VAR_NAMES}" \
        --comfyui /comfyui \
        --suite "/app/${SUITE}" \
        --n-prompts "${N_PROMPTS}" \
        --n-steps "${N_STEPS}" \
        --device "${DEVICE}" \
        --results-dir "/comp/results/${COMPARISON_NAME}/text_encoder"

log "H3 attention readout complete: ${RESULTS_DIR}/tebench_attention_*.json"
