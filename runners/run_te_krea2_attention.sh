#!/usr/bin/env bash
# Krea2 attention-readout runner — what does the DiT read from conditioning,
# base vs variant TE (phase 9). GPU (default 0) — free ComfyUI models first
# if the card is held. Results: tebench_attention_<variant>.json (te_attention
# importer consumes them unchanged).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <comparison_dir>

Options:
   --variant NAME     Only this variant (default: all with te_file + te_reference)
   --suite PATH       Prompt suite (default prompts/te_krea2_prompts.yaml)
   --n-prompts N      Prompts to encode (default 4; one per category)
   --n-steps N        Euler steps (default 6)
   --device DEV       cuda | cpu (default cuda; GPU 0)
   --limit            alias of --n-prompts for symmetry
   --dry-run          Preview
   -h, --help
EOF
    exit 0
}

SUITE="prompts/te_krea2_prompts.yaml"
N_PROMPTS=4
N_STEPS=6
DEVICE="cuda"
VARIANT_FILTER=""
DRY_RUN=0
COMP_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite) SUITE="$2"; SUITE_CLI=1; shift 2 ;;
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
if [[ -z "${SUITE_CLI}" ]]; then SUITE="${TEBENCH_PROMPT_SUITE:-prompts/te_krea2_prompts.yaml}"; fi
COMFYUI_PATH="${TEBENCH_COMFYUI_PATH:-/home/maxx/comfyui-docker/comfyui}"

[[ "${TEBENCH_BACKEND:-}" == "krea2" ]] || { error "This runner is krea2-only"; exit 1; }
[[ -n "${DIT_FILE}" ]] || { error "TEBENCH_DIT_FILE not set"; exit 1; }
[[ -n "${BASE_TE}" ]] || { error "TEBENCH_BASE_TE_FILE not set"; exit 1; }

# base TE path: canonical models dir, repo models/te fallback
base_te_path() {
    if [[ -f "${BASE_TE}" ]]; then echo "${BASE_TE}"
    elif [[ -f "${COMPARISON_DIR}/models/te/$(basename "${BASE_TE}")" ]]; then
        echo "/comp/models/te/$(basename "${BASE_TE}")"
    else error "base TE not found: ${BASE_TE}"; return 2; fi
}

# variants: te_file arms with a resolvable path
declare -A VARIANTS=()
while read -r envline; do
    name="${envline#VARIANT_TE_FILE_}"
    name="${name%%=*}"
    te_file="${envline#*=}"
    [[ -n "${te_file}" ]] || continue
    if [[ -f "${te_file}" ]] || [[ -f "${COMPARISON_DIR}/models/te/$(basename "${te_file}")" ]]; then
        VARIANTS["${name}"]="${te_file}"
    fi
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
[[ -n "${VAR_ARGS}" ]] || { error "No variants with resolvable te_file"; exit 1; }

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "DRY-RUN: krea2 attention base=$(basename "${BASE_TE}") dit=$(basename "${DIT_FILE}") variants=${VAR_NAMES} n_prompts=${N_PROMPTS}"
    exit 0
fi

# container path resolution for variant files (repo fallback under /comp)
VAR_ARGS_C=""
IFS=',' read -ra _vf <<< "${VAR_ARGS}"
for f in "${_vf[@]}"; do
    if [[ -f "${f}" ]]; then p="${f}"; else p="/comp/models/te/$(basename "${f}")"; fi
    [[ -z "${VAR_ARGS_C}" ]] && VAR_ARGS_C="${p}" || VAR_ARGS_C="${VAR_ARGS_C},${p}"
done

BASE_C="$(base_te_path)"

log "Krea2 attention readout: variants=${VAR_NAMES} n_prompts=${N_PROMPTS} steps=${N_STEPS} device=${DEVICE}"

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
    -m src.text_encoder.run_krea2_attention \
        --dit "${DIT_FILE}" \
        --base "${BASE_C}" \
        --variants "${VAR_ARGS_C}" \
        --variant-names "${VAR_NAMES}" \
        --comfyui /comfyui \
        --suite "/app/${SUITE}" \
        --n-prompts "${N_PROMPTS}" \
        --n-steps "${N_STEPS}" \
        --device "${DEVICE}" \
        --results-dir "/comp/results/${COMPARISON_NAME}/text_encoder"

log "Krea2 attention readout complete: ${RESULTS_DIR}/tebench_attention_*.json"
