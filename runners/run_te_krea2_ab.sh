#!/usr/bin/env bash
# Krea2 A/B image generation runner — arms x prompts x seeds through the
# running ComfyUI on the 4090 (host.docker.internal:8190). Phase 8 T3 of the
# te-krea2 battery. Two-stage sampler (6-step gen + 2-step refine): both
# KSampler seeds are patched to the same A/B seed by run_krea2_ab.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <comparison_dir> [-- EXTRA ARGS]

Runs src.text_encoder.run_krea2_ab against the running ComfyUI for the BASE
and each variant TE with a te_file. Images land in
results/<name>/ab_images/<variant>/ + manifest.json.

Options:
   --suite PATH        Prompt suite (default prompts/te_krea2_prompts.yaml)
   --seeds LIST        Comma-separated seeds (default 42; real battery 42,43)
   --variants LIST     Arm filter (default: base + all with te_file)
   --labels LIST       Prompt-label filter
   --api-url URL       ComfyUI endpoint (default http://host.docker.internal:8190)
   --smoke             Tag manifest smoke=true (stand-in runs)
   --dry-run           Preview
   -h, --help          Show this help
EOF
    exit 0
}

SUITE="prompts/te_krea2_prompts.yaml"
SEEDS="42"
VARIANTS=""
LABELS=""
API_URL="http://host.docker.internal:8190"
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
        --api-url) API_URL="$2"; shift 2 ;;
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
OUT_DIR="${COMPARISON_DIR}/results/${COMPARISON_NAME}/text_encoder/ab_images"
WORKFLOW="${COMPARISON_DIR}/workflows/krea2_t2i_api.json"
[[ -f "${WORKFLOW}" ]] || { error "Workflow not found: ${WORKFLOW}"; exit 1; }

[[ -n "${VARIANTS}" ]] && EXTRA_ARGS+=("--variants" "${VARIANTS}")
[[ -n "${LABELS}" ]] && EXTRA_ARGS+=("--labels" "${LABELS}")

# ComfyUI models dir, mounted 1:1 at its own absolute path (host == container)
# so te_file paths stay valid inside the container (AGENTS.md #40). Env
# override replaces the machine-specific default (audit 2026-10-14).
COMFYUI_MODELS_DIR="${COMFYUI_MODELS_DIR:-${HOME}/extra/comfyui-models}"

log "Krea2 A/B generation: suite=${SUITE} seeds=${SEEDS} out=${OUT_DIR}"

docker_run_cpu "${TE_IMAGE}" \
    -v "${COMPARISON_DIR}:/comp:rw" \
    -v "${ABL_ROOT}:/app:ro" \
    -v "${COMFYUI_MODELS_DIR}:${COMFYUI_MODELS_DIR}:ro" \
    --add-host=host.docker.internal:host-gateway \
    -e PYTHONPATH=/app \
    -- \
    -m src.text_encoder.run_krea2_ab \
        --workflow /comp/workflows/krea2_t2i_api.json \
        --comparison /comp/comparison.json \
        --api-url "${API_URL}" \
        --out-dir "/comp/results/${COMPARISON_NAME}/text_encoder/ab_images" \
        --prompt-suite "/app/${SUITE}" \
        --seeds "${SEEDS}" \
        ${SMOKE_FLAG[@]+"${SMOKE_FLAG[@]}"} \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

log "Krea2 A/B generation complete: ${OUT_DIR}"
