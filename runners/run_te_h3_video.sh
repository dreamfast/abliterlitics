#!/usr/bin/env bash
# H3 video battery runner — frame-LPIPS + CLIP adherence over ab_videos.
# Phase 7 T5 of the te-h3 battery. LPIPS/CLIP are small models; can run on
# GPU 1 (4090) to leave the 5090/ComfyUI stack alone, or CPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <comparison_dir>

Scores the ab_videos manifest clips with the video-suite metrics and writes
tebench_h3_video.json to results/<name>/text_encoder/.

Options:
   --gpu N        GPU id for docker (default 1 = 4090; pass 0 for 5090)
   --cpu          Run on CPU (no GPU needed)
   --frame-k N    Frames sampled per clip (default 8)
   --arms LIST    Arm filter (default: all in manifest)
   --dry-run      Preview
   -h, --help     Show this help
EOF
    exit 0
}

GPU="1"
RUN_FN_ARG=("1")
FRAME_K="8"
ARMS=""
DRY_RUN=0
COMP_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; RUN_FN_ARG=("$2"); shift 2 ;;
        --cpu) RUN_FN_ARG=(); CPU_MODE=1; shift ;;
        --frame-k) FRAME_K="$2"; shift 2 ;;
        --arms) ARMS="$2"; shift 2 ;;
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
MANIFEST="${COMPARISON_DIR}/results/${COMPARISON_NAME}/ab_videos/manifest.json"
SUITE="prompts/te_h3_video_prompts.yaml"

[[ -f "${MANIFEST}" ]] || { error "No manifest at ${MANIFEST} — run run_te_h3_ab.sh first"; exit 1; }
mkdir -p "${ABL_ROOT}/.cache/hf" "${ABL_ROOT}/.cache/torch"

DEVICE="cuda"
if [[ -n "${CPU_MODE:-}" ]]; then
    RUN_FN=docker_run_cpu
    RUN_ARG=()
    DEVICE="cpu"
else
    RUN_FN=docker_run
    RUN_ARG=("${RUN_FN_ARG[@]}")
fi

ARM_ARGS=()
[[ -n "${ARMS}" ]] && ARM_ARGS=(--arms "${ARMS}")

log "Scoring H3 video battery (device=${DEVICE}, gpu=${GPU:-cpu}, frame-k=${FRAME_K})"

"${RUN_FN}" ${RUN_ARG[@]+"${RUN_ARG[@]}"} "${TE_IMAGE}" \
    -v "${COMPARISON_DIR}:/comp:rw" \
    -v "${ABL_ROOT}:/app:ro" \
    -v "${ABL_ROOT}/.cache/hf:/root/.cache/huggingface" \
    -v "${ABL_ROOT}/.cache/torch:/root/.cache/torch" \
    -e PYTHONPATH=/app \
    -- \
    -m src.text_encoder.run_h3_video \
        --manifest "/comp/results/${COMPARISON_NAME}/ab_videos/manifest.json" \
        --suite "/app/${SUITE}" \
        --out "/comp/results/${COMPARISON_NAME}/text_encoder/tebench_h3_video.json" \
        --frame-k "${FRAME_K}" \
        --device "${DEVICE}" \
        ${ARM_ARGS[@]+"${ARM_ARGS[@]}"} \
        $([[ "${DRY_RUN}" -eq 1 ]] && echo --dry-run)

log "H3 video battery complete: ${RESULTS_DIR}/tebench_h3_video.json"
