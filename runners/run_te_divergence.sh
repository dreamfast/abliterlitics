#!/usr/bin/env bash
# DiT-step divergence runner — does the Z-Image DiT respond to TE conditioning changes?
# Runs paired Euler trajectories from identical noise; measures per-step divergence
# with a different-seed noise control. Requires cached artifacts from run_text_encoder.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <comparison_dir>

DiT-step divergence analysis (tebench). For each variant with te_file:
runs paired flow-matching Euler trajectories (identical seed) with base vs
variant conditioning through ComfyUI's own NextDiT, measuring velocity +
latent divergence per step and a different-seed noise control.

Requires: text-encoder analysis already run (activation cache populated).

Options:
   --gpu GPU          GPU index (required for the 6B DiT; default 0)
   --n-steps N        Euler steps per trajectory (default 20)
   --n-prompts N      Prompts to run, one per category (default 6)
   --latent-hw N      Latent spatial size (default 128 = 1024px)
   --skip-existing    Skip variants with existing divergence results (default)
   --force            Overwrite existing results
   --dry-run          Preview
   -h, --help         Show this help
EOF
    exit 0
}

GPU=""
N_STEPS=20
N_PROMPTS=6
LATENT_HW=128
DRY_RUN=0
SKIP_EXISTING=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)           GPU="$2"; shift 2 ;;
        --n-steps)       N_STEPS="$2"; shift 2 ;;
        --n-prompts)     N_PROMPTS="$2"; shift 2 ;;
        --latent-hw)     LATENT_HW="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING=1; shift ;;
        --force)         SKIP_EXISTING=0; shift ;;
        --dry-run)       DRY_RUN=1; shift ;;
        -h|--help)       usage ;;
        *)               break ;;
    esac
done

COMP_DIR="${1:?Error: comparison_dir required. Use --help for usage.}"

load_comparison "$COMP_DIR"
COMPARISON_DIR="$(cd "$(dirname "${COMPARISON_DIR}")" && pwd)/$(basename "${COMPARISON_DIR}")"
RESULTS_DIR="${COMPARISON_DIR}/results/${COMPARISON_NAME}/text_encoder"
TE_CACHE_DIR="${COMPARISON_DIR}/.te_cache"

if [[ -z "$TEBENCH_BACKEND" || "$TEBENCH_BACKEND" != "zimage" ]]; then
    error "DiT divergence currently supports the zimage backend only (got: '${TEBENCH_BACKEND}')"
    exit 1
fi
if [[ ! -d "$TE_CACHE_DIR" ]] || [[ -z "$(ls -A "$TE_CACHE_DIR" 2>/dev/null)" ]]; then
    error "Activation cache empty: ${TE_CACHE_DIR}"
    error "Run './abliterlitics.sh text-encoder ${COMP_DIR}' first"
    exit 1
fi

MODELS_HOST_ROOT="$(dirname "$(dirname "$TEBENCH_DIT_FILE")")"
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

GPU="${GPU:-0}"
mkdir -p "$RESULTS_DIR"

for variant in $VARIANT_NAMES; do
    var_slug="$(echo "$variant" | tr '-' '_')"
    skip_var="VARIANT_SKIP_TE_${var_slug}"
    if [[ "${!skip_var}" == "1" ]]; then
        continue
    fi
    te_file_var="VARIANT_TE_FILE_${var_slug}"
    variant_te_host="${!te_file_var:-}"
    if [[ -z "$variant_te_host" || ! -f "$variant_te_host" ]]; then
        continue
    fi

    result_file="${RESULTS_DIR}/te_dit_divergence_${variant}.json"
    if [[ "${SKIP_EXISTING:-1}" == "1" ]] && result_exists "$result_file"; then
        log "Skipping (exists): te_dit_divergence_${variant}.json"
        continue
    fi

    log "Running DiT-step divergence: base -> ${variant}"

    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [dry-run] divergence for ${variant} (${N_STEPS} steps, ${N_PROMPTS} prompts)"
        continue
    fi

    docker_run "$GPU" "$TE_IMAGE" \
        -v "${COMPARISON_DIR}:/comparison:ro" \
        -v "${MODELS_HOST_ROOT}:/models:ro" \
        -v "${COMFYUI_HOST_PATH}:/comfyui:ro" \
        -v "${RESULTS_DIR}:/results" \
        -v "${ABL_ROOT}:/app:ro" \
        -v "${ABL_ROOT}/prompts/te_suite.yaml:/prompts/te_suite.yaml:ro" \
        -v "${TE_CACHE_DIR}:/te_cache" \
        -e PYTHONPATH=/app:/comfyui \
        -- \
        /app/src/text_encoder/run_dit_divergence.py \
            --base "$(host_to_container "$TEBENCH_BASE_TE_FILE")" \
            --variant "$(host_to_container "${variant_te_host}")" \
            --variant-name "${variant}" \
            --dit "$(host_to_container "$TEBENCH_DIT_FILE")" \
            --cache-dir /te_cache \
            --results-dir /results \
            --device cuda \
            --n-steps "${N_STEPS}" \
            --n-prompts "${N_PROMPTS}" \
            --latent-hw "${LATENT_HW}"
done

log "DiT-step divergence complete"
