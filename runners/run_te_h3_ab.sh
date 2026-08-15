#!/usr/bin/env bash
# H3 A/B video+audio generation runner — arms x prompts x seeds through the
# running ComfyUI (host.docker.internal:8191). Phase 7 T3 of the te-h3 battery.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <comparison_dir> [-- EXTRA ARGS]

Runs src.text_encoder.run_h3_ab against the running ComfyUI for the BASE
and each variant TE with a te_file. Clips land in
results/<name>/ab_videos/<variant>/ + manifest.json.

Options:
   --suite PATH        Prompt suite (default prompts/te_h3_video_prompts.yaml;
                       use prompts/te_h3_audio_prompts.yaml for the audio battery)
   --seeds LIST        Comma-separated seeds (default 42; real battery 42,43,44)
   --variants LIST     Arm filter (default: base + all with te_file)
   --labels LIST       Prompt-label filter
   --duration SEC      Clip duration (default 5)
   --smoke             Tag manifest smoke=true (stand-in runs)
   --dry-run           Preview
   -h, --help          Show this help
EOF
    exit 0
}

SUITE="prompts/te_h3_video_prompts.yaml"
SEEDS="42"
VARIANTS=""
LABELS=""
DURATION="5"
SMOKE_FLAG=()
DRY_RUN=0
COMP_DIR=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite) SUITE="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --variants) VARIANTS="$2"; shift 2 ;;
        --labels) LABELS="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --smoke) SMOKE_FLAG=("--smoke"); shift ;;
        --dry-run) DRY_RUN=1; EXTRA_ARGS+=("--dry-run"); shift ;;
        -h|--help) usage ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
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
OUT_DIR="${COMPARISON_DIR}/results/${COMPARISON_NAME}/ab_videos"
WORKFLOW="${COMPARISON_DIR}/workflows/h3_t2v_api.json"
[[ -f "${WORKFLOW}" ]] || { error "Workflow not found: ${WORKFLOW}"; exit 1; }

[[ -n "${VARIANTS}" ]] && EXTRA_ARGS+=("--variants" "${VARIANTS}")
[[ -n "${LABELS}" ]] && EXTRA_ARGS+=("--labels" "${LABELS}")

log "H3 A/B generation: suite=${SUITE} seeds=${SEEDS} out=${OUT_DIR}"

docker_run_cpu "${TE_IMAGE}" \
    -v "${COMPARISON_DIR}:/comp:rw" \
    -v "${ABL_ROOT}:/app:ro" \
    -v "/home/maxx/extra/comfyui-models:/home/maxx/extra/comfyui-models:ro" \
    --add-host=host.docker.internal:host-gateway \
    -e PYTHONPATH=/app \
    -- \
    -m src.text_encoder.run_h3_ab \
        --workflow /comp/workflows/h3_t2v_api.json \
        --comparison /comp/comparison.json \
        --api-url http://host.docker.internal:8191 \
        --out-dir "/comp/results/${COMPARISON_NAME}/ab_videos" \
        --prompt-suite "/app/${SUITE}" \
        --seeds "${SEEDS}" \
        --duration "${DURATION}" \
        ${SMOKE_FLAG[@]+"${SMOKE_FLAG[@]}"} \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

log "H3 A/B generation complete: ${OUT_DIR}"
