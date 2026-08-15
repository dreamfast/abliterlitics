#!/usr/bin/env bash
# H3 audio battery runner — ASR-scored target-word fidelity (the headline test).
# Phase 7 T4 of the te-h3 battery. Whisper on GPU 0; audio decoded via PyAV.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <comparison_dir>

Transcribes the ab_videos manifest clips with the audio-battery suite and
writes tebench_h3_audio.json to results/<name>/text_encoder/.

Options:
   --seeds LIST        Seeds to generate first (default: none — score existing)
   --gen               Also GENERATE the audio clips first (run_h3_ab)
   --gen-seeds LIST    Seeds for generation (default 42,43,44)
   --arms LIST         Arm filter (default: all in manifest)
   --device DEV        Whisper device (default cuda)
   --gpu N             GPU id when device=cuda (default 0; 1 = 4090)
   --smoke             Tag generation smoke=true
   --dry-run           Preview scoring
   -h, --help          Show this help
EOF
    exit 0
}

GEN=0
GEN_SEEDS="42,43,44"
ARMS=""
DEVICE="cuda"
GPU="0"
SMOKE_FLAG=()
DRY_RUN=0
COMP_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gen) GEN=1; shift ;;
        --gen-seeds) GEN_SEEDS="$2"; shift 2 ;;
        --arms) ARMS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --smoke) SMOKE_FLAG=("--smoke"); shift ;;
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
SUITE="prompts/te_h3_audio_prompts.yaml"
WHISPER_MODEL="${TEBENCH_WHISPER_MODEL:-openai/whisper-large-v3}"

ARM_ARGS=()
[[ -n "${ARMS}" ]] && ARM_ARGS=(--arms "${ARMS}")

if [[ "${GEN}" -eq 1 ]]; then
    log "Generating audio-battery clips (seeds=${GEN_SEEDS})"
    # Default arms: base + every non-stand-in variant with a te_file (the
    # "base_<quant>" prefix marks generation-only stand-ins per NOTES.md).
    GEN_VARIANTS="${ARMS}"
    if [[ -z "${GEN_VARIANTS}" ]]; then
        GEN_VARIANTS="base$(python3 -c "
import json, sys
cfg = json.load(open('${COMPARISON_DIR}/comparison.json'))
vs = [n for n, v in cfg.get('variants', {}).items() if v.get('te_file') and not n.startswith('base_')]
print(',' + ','.join(vs) if vs else '', end='')
")"
    fi
    log "Generation arms: ${GEN_VARIANTS}"
    "${SCRIPT_DIR}/run_te_h3_ab.sh" \
        --suite "${SUITE}" \
        --seeds "${GEN_SEEDS}" \
        --variants "${GEN_VARIANTS}" \
        ${SMOKE_FLAG[@]+"${SMOKE_FLAG[@]}"} \
        "${COMP_DIR}"
fi

[[ -f "${MANIFEST}" ]] || { error "No manifest at ${MANIFEST} — run with --gen or run_te_h3_ab first"; exit 1; }
mkdir -p "${ABL_ROOT}/.cache/hf" "${ABL_ROOT}/.cache/torch"

RUN_FN=docker_run_cpu
RUN_EXTRA=()
if [[ "${DEVICE}" == "cuda" ]]; then
    RUN_FN=docker_run
    RUN_EXTRA=("${GPU:-0}")  # GPU 0 = 5090; GPU=1 uses the 4090 (5090 busy)
fi

log "Scoring H3 audio battery (whisper=${WHISPER_MODEL}, device=${DEVICE})"

"${RUN_FN}" "${RUN_EXTRA[@]}" "${TE_IMAGE}" \
    -v "${COMPARISON_DIR}:/comp:rw" \
    -v "${ABL_ROOT}:/app:ro" \
    -v "${ABL_ROOT}/.cache/hf:/root/.cache/huggingface" \
    -v "${ABL_ROOT}/.cache/torch:/root/.cache/torch" \
    -e PYTHONPATH=/app \
    -e HF_HOME=/root/.cache/huggingface \
    -- \
    -m src.text_encoder.run_h3_audio \
        --manifest "/comp/results/${COMPARISON_NAME}/ab_videos/manifest.json" \
        --suite "/app/${SUITE}" \
        --out "/comp/results/${COMPARISON_NAME}/text_encoder/tebench_h3_audio.json" \
        --whisper-model "${WHISPER_MODEL}" \
        --device "${DEVICE}" \
        ${ARM_ARGS[@]+"${ARM_ARGS[@]}"} \
        $([[ "${DRY_RUN}" -eq 1 ]] && echo --dry-run)

log "H3 audio battery complete: ${RESULTS_DIR}/tebench_h3_audio.json"
