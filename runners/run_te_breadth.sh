#!/usr/bin/env bash
# Breadth battery scorer — per-category LPIPS/CLIP over the breadth A/B set.
# Phase 5 (P10). Scores ab_images_breadth/manifest.json against the breadth
# suite; writes tebench_breadth.json. GPU 0 when free; CPU via TE_IMG_DEVICE=cpu.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <comparison_dir>

Options:
   --force              Overwrite tebench_breadth.json (default: skip if present)
   --dry-run            Preview
   -h, --help           Show this help
EOF
    exit 0
}

FORCE=0
DRY_RUN=0
COMP_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
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
MANIFEST="${RESULTS_DIR}/ab_images_breadth/manifest.json"
SUITE="${ABL_ROOT}/prompts/te_breadth_prompts.yaml"

[[ -f "${MANIFEST}" ]] || { error "No breadth manifest at ${MANIFEST} — run run_ab_test over the breadth suite first"; exit 1; }
[[ -f "${SUITE}" ]] || { error "No breadth suite at ${SUITE}"; exit 1; }

if [[ -f "${RESULTS_DIR}/tebench_breadth.json" && "${FORCE}" -eq 0 ]]; then
    log "tebench_breadth.json exists — use --force to overwrite"
    exit 0
fi

IMG_DEVICE="${TE_IMG_DEVICE:-cuda}"
if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "DRY-RUN: would score ${MANIFEST} (device=${IMG_DEVICE})"
    exit 0
fi

mkdir -p "${ABL_ROOT}/.cache/hf" "${ABL_ROOT}/.cache/torch"

RUN_FN=docker_run_cpu
RUN_EXTRA=()
if [[ "${IMG_DEVICE}" == "cuda" ]]; then
    RUN_FN=docker_run
    RUN_EXTRA=("0")
fi

"${RUN_FN}" "${RUN_EXTRA[@]}" "${TE_IMAGE}" \
    -v "${ABL_ROOT}:/app" \
    -w /app \
    -v "${ABL_ROOT}/.cache/hf:/root/.cache/huggingface" \
    -v "${ABL_ROOT}/.cache/torch:/root/.cache/torch" \
    -e PYTHONPATH=/app \
    -- \
    -m src.text_encoder.run_breadth \
        --manifest "$(realpath --relative-to="${ABL_ROOT}" "${MANIFEST}")" \
        --suite prompts/te_breadth_prompts.yaml \
        --out "$(realpath --relative-to="${ABL_ROOT}" "${RESULTS_DIR}/tebench_breadth.json")" \
        --device "${IMG_DEVICE}"

log "Breadth scoring complete: ${RESULTS_DIR}/tebench_breadth.json"
