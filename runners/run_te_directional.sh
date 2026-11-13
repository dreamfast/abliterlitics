#!/usr/bin/env bash
# Directional analysis runner — is the abliteration delta a direction or noise?
# Reads cached raw_hidden activations (from run_text_encoder.sh), pairs each
# heretic variant with its matched-quant base (te_reference or global base),
# and computes per-layer delta-vs-taboo-direction geometry with permutation nulls.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <comparison_dir>

Directional analysis (tebench). For each variant with te_file, measures
whether the ablation delta aligns with the base model's taboo direction
per layer, whether it moves taboo prompts toward the innocent cluster,
and how much delta energy is orthogonal noise (permutation nulls).

Pairs: variant vs its te_reference when set (matched quant — quant noise
removed from the delta), else vs the global base.

Requires: text-encoder analysis already run (activation cache populated).

Options:
   --n-perm N         Permutation resamples (default 200)
   --skip-existing    Skip variants with existing directional results (default)
   --force            Overwrite existing results
   --dry-run          Preview
   -h, --help         Show this help
EOF
    exit 0
}

N_PERM=200
DRY_RUN=0
SKIP_EXISTING=1
FORCE=0
COMP_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-perm) N_PERM="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --skip-existing) SKIP_EXISTING=1; shift ;;
        --force) FORCE=1; SKIP_EXISTING=0; shift ;;
        -h|--help) usage ;;
        -*) error "Unknown option: $1"; usage ;;
        *) [[ -z "$COMP_DIR" ]] && COMP_DIR="$1" || error "Multiple comparison dirs"; shift ;;
    esac
done
[[ -z "$COMP_DIR" ]] && { error "comparison dir required"; usage; }
if [[ ! -d "$COMP_DIR" ]]; then
    error "Comparison dir not found: $COMP_DIR"
    exit 1
fi

load_comparison "$COMP_DIR"
COMPARISON_DIR="$(cd "$(dirname "${COMPARISON_DIR}")" && pwd)/$(basename "${COMPARISON_DIR}")"
RESULTS_DIR="${COMPARISON_DIR}/results/${COMPARISON_NAME}/text_encoder"
CACHE_DIR="${COMPARISON_DIR}/.te_cache"

if [[ ! -d "$CACHE_DIR" ]]; then
    error "No activation cache at ${CACHE_DIR} — run './abliterlitics.sh te ${COMP_DIR}' first"
    exit 1
fi
mkdir -p "$RESULTS_DIR"

# tebench settings
TE_CKPT_BASE="${TEBENCH_BASE_TE_FILE:-}"
if [[ -z "$TE_CKPT_BASE" ]]; then
    error "tebench.base_te_file not set in comparison.json"
    exit 1
fi

# Translate host paths -> container paths (same mounts as run_text_encoder.sh:
# models root detected from the DIT file, NOT the base TE — cached ckpt_ids
# were written under that convention).
MODELS_HOST_ROOT="$(dirname "$(dirname "${TEBENCH_DIT_FILE:-$TE_CKPT_BASE}")")"
COMFYUI_HOST_PATH="${TEBENCH_COMFYUI_PATH:-${HOME}/comfyui-docker/comfyui}"


# Build pairs: NAME=VAR_CKPT:REF_CKPT (container paths)
PAIR_ARGS=()
for variant in $VARIANT_NAMES; do
    var_slug="$(echo "$variant" | tr '.-' '__')"
    skip_var="VARIANT_SKIP_TE_${var_slug}"
    te_file_var="VARIANT_TE_FILE_${var_slug}"
    ref_var="VARIANT_TE_REF_${var_slug}"
    if [[ "${!skip_var}" == "1" || -z "${!te_file_var}" ]]; then
        continue
    fi
    variant_te_host="${!te_file_var}"
    if [[ ! -f "$variant_te_host" ]]; then
        log "WARNING: TE file missing, skipping ${variant}: ${variant_te_host}"
        continue
    fi
    # Directional analysis is about the ABLATION delta — only heretic-style
    # variants (with a reference) are meaningful; pure base quants have no
    # ablation to analyze.
    if [[ -z "${!ref_var}" ]]; then
        log "Skipping ${variant} (no te_reference — no ablation delta to analyze)"
        continue
    fi
    if [[ $SKIP_EXISTING -eq 1 && -f "${RESULTS_DIR}/te_directional_${variant}.json" && $FORCE -eq 0 ]]; then
        log "Skipping ${variant} (directional result exists)"
        continue
    fi
    ref_slug="$(echo "${!ref_var}" | tr '.-' '__')"
    ref_file_var="VARIANT_TE_FILE_${ref_slug}"
    ref_file_host="${!ref_file_var:-}"
    if [[ -n "$ref_file_host" && -f "$ref_file_host" ]]; then
        ref_host="$ref_file_host"
    else
        ref_host="$TE_CKPT_BASE"
        log "  Reference '${!ref_var}' has no te_file — using global base"
    fi
    var_container="$(host_to_container "$variant_te_host")"
    ref_container="$(host_to_container "$ref_host")"
    PAIR_ARGS+=("${variant}=${var_container}:${ref_container}")
    log "Directional pair: ${variant} vs ${ref_host##*/}"
    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [dry-run] pair ${variant}: var=${var_container} ref=${ref_container} (n_perm=${N_PERM})"
    fi
done

if [[ ${#PAIR_ARGS[@]} -eq 0 ]]; then
    log "No directional pairs to run"
    exit 0
fi
if [[ $DRY_RUN -eq 1 ]]; then
    log "[dry-run] would run ${#PAIR_ARGS[@]} directional pairs"
    exit 0
fi

log "Running directional analysis for ${#PAIR_ARGS[@]} pairs (n_perm=${N_PERM})"

docker_run_cpu "$TE_IMAGE" \
    -v "${CACHE_DIR}:/cache:ro" \
    -v "${RESULTS_DIR}:/results" \
    -v "${ABL_ROOT}:/app:ro" \
    -e PYTHONPATH=/app \
    -- \
    /app/src/text_encoder/run_directional.py \
        --cache-dir /cache \
        --results-dir /results \
        --n-perm "$N_PERM" \
        --pairs "${PAIR_ARGS[@]}"

log "Directional analysis complete"
