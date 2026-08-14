#!/usr/bin/env bash
# Image-space metrics runner — LPIPS ladder + CLIP + SSIM/PSNR over the A/B set.
# Phase 4 of the tebench battery (journal #122). CPU or GPU; no model loading
# beyond LPIPS-AlexNet + open_clip ViT-L-14 (weights cached under .cache/).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <comparison_dir>

Scores the A/B image manifest with image-space metrics and writes
tebench_img_<variant>.json + tebench_img_anchors.json to
results/<name>/text_encoder/. Pair types: te_swap (variant vs reference),
cross_prompt (meaning-change anchor), seed_change (needs multi-seed top-up),
floor_band (int8-vs-fp8 quant-floor band).

Options:
   --lpips-only       Skip CLIP metrics (fast pass; no CLIP weights needed)
   --limit N          Debug: cap pairs per kind
   --force            Overwrite existing tebench_img_*.json (default: skip)
   --dry-run          Preview
   -h, --help         Show this help
EOF
    exit 0
}

EXTRA_ARGS=()
DRY_RUN=0
FORCE=0
COMP_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lpips-only) EXTRA_ARGS+=("--lpips-only"); shift ;;
        --limit) EXTRA_ARGS+=("--limit" "$2"); shift 2 ;;
        --force) FORCE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage ;;
        -*) error "Unknown option: $1"; usage ;;
        *)
            [[ -z "$COMP_DIR" ]] && { COMP_DIR="$1"; } || { error "Multiple comparison dirs given"; usage; }
            shift ;;
    esac
done
[[ -z "$COMP_DIR" ]] && { error "comparison_dir required"; usage; }
[[ ! -d "$COMP_DIR" ]] && { error "Comparison dir not found: ${COMP_DIR}"; exit 1; }

load_comparison "$COMP_DIR"
COMPARISON_DIR="$(cd "$(dirname "${COMPARISON_DIR}")" && pwd)/$(basename "${COMPARISON_DIR}")"
RESULTS_DIR="${COMPARISON_DIR}/results/${COMPARISON_NAME}/text_encoder"
MANIFEST="${RESULTS_DIR}/ab_images/manifest.json"

if [[ ! -f "${MANIFEST}" ]]; then
    error "No A/B manifest at ${MANIFEST}"
    error "Run the A/B generation first (run_ab_test.py)"
    exit 1
fi

if [[ "${FORCE}" -eq 0 ]]; then
    existing=$(ls "${RESULTS_DIR}"/tebench_img_*.json 2>/dev/null | wc -l || true)
    if [[ "${existing}" -gt 0 ]]; then
        log "tebench_img_*.json already present (${existing} files) — use --force to overwrite"
        exit 0
    fi
fi

# Defaults from comparison.json (exported by _load_comparison.py)
IMG_REFERENCE="${TEBENCH_IMG_REFERENCE:-base_int8}"
IMG_CLIP_MODEL="${TEBENCH_IMG_CLIP_MODEL:-ViT-L-14}"
IMG_CLIP_PRETRAINED="${TEBENCH_IMG_CLIP_PRETRAINED:-openai}"
IMG_FLOOR_PAIR="${TEBENCH_IMG_FLOOR_PAIR:-base_int8,base_fp8}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "DRY-RUN: would score ${MANIFEST} (ref=${IMG_REFERENCE}, clip=${IMG_CLIP_MODEL}/${IMG_CLIP_PRETRAINED})"
    exit 0
fi

mkdir -p "${ABL_ROOT}/.cache/hf" "${ABL_ROOT}/.cache/torch"

log "Scoring A/B images for ${COMPARISON_NAME} (ref=${IMG_REFERENCE})"

docker_run_cpu "${TE_IMAGE}" \
    -v "${MANIFEST}:/manifest.json:ro" \
    -v "${RESULTS_DIR}:/results" \
    -v "${ABL_ROOT}:/app:ro" \
    -v "${ABL_ROOT}/.cache/hf:/root/.cache/huggingface" \
    -v "${ABL_ROOT}/.cache/torch:/root/.cache/torch" \
    -e PYTHONPATH=/app \
    -- \
    -m src.text_encoder.run_img_metrics \
        --manifest /manifest.json \
        --out-dir /results \
        --reference "${IMG_REFERENCE}" \
        --floor-pair "${IMG_FLOOR_PAIR}" \
        --clip-model "${IMG_CLIP_MODEL}" \
        --clip-pretrained "${IMG_CLIP_PRETRAINED}" \
        "${EXTRA_ARGS[@]}"

log "Image metrics complete: ${RESULTS_DIR}/tebench_img_*.json"
