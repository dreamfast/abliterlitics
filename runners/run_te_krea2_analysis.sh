#!/usr/bin/env bash
# Krea2 conditioning analysis runner — base vs variant TE at the 12-tap
# txtfusion interface (RAW_HIDDEN taps + tap_stack, PROJECTED, CONNECTOR_OUT).
# Phase 8 T5 of the te-krea2 battery. CPU-only by default (4B TE fits).
#
# Pairing: each variant is compared against its te_reference arm when set
# (heretic -> base bf16, heretic_int8 -> base_int8 matched quant,
#  base_int8 -> base = the literal quant floor). Variants without
# te_reference compare against settings.tebench.base_te_file.
#
# TE path resolution: settings/te_file paths point at the comfy models dir
# (canonical after the user copies the built base files). Until then, a
# missing file falls back to <comparison>/models/te/<basename> (the repo
# copies built by scripts/build_krea2_base_te.py) via the /comp mount.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <comparison_dir>

Runs src.text_encoder.run_analysis with --backend krea2 for each variant TE
with a te_file, against its te_reference base, writing te_<name>.json to
results/<name>/text_encoder/.

Options:
   --variant NAME     Analyze only this variant (default: all with te_file)
   --limit N          Cap prompt count (debug)
   --force            Rerun even if results exist (run_analysis skips via cache)
   --dry-run          Preview
   -h, --help         Show this help
EOF
    exit 0
}

EXTRA_ARGS=()
DRY_RUN=0
COMP_DIR=""
VARIANT_FILTER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit) EXTRA_ARGS+=("--limit" "$2"); shift 2 ;;
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
CACHE_DIR="${COMPARISON_DIR}/results/.te_cache"
mkdir -p "${RESULTS_DIR}" "${CACHE_DIR}"

BACKEND="${TEBENCH_BACKEND:-}"
DIT_FILE="${TEBENCH_DIT_FILE:-}"
BASE_TE="${TEBENCH_BASE_TE_FILE:-}"
COMFYUI_PATH="${TEBENCH_COMFYUI_PATH:-/home/maxx/comfyui-docker/comfyui}"
PROMPT_SUITE="${TEBENCH_PROMPT_SUITE:-prompts/te_krea2_prompts.yaml}"

[[ "${BACKEND}" == "krea2" ]] || { error "This runner is krea2-only (backend=${BACKEND})"; exit 1; }
[[ -n "${BASE_TE}" ]] || { error "TEBENCH_BASE_TE_FILE not set (settings.tebench.base_te_file)"; exit 1; }
[[ -n "${DIT_FILE}" ]] || { error "TEBENCH_DIT_FILE not set (settings.tebench.dit_file)"; exit 1; }

# Container-visible TE path: models dir is mounted at its own path; repo
# fallback lives under the /comp comparison mount.
te_container_path() {
    local f="$1"
    if [[ -f "${f}" ]]; then
        echo "${f}"
    elif [[ -f "${COMPARISON_DIR}/models/te/$(basename "${f}")" ]]; then
        echo "/comp/models/te/$(basename "${f}")"
    else
        error "TE file not found: ${f} (nor models/te/$(basename "${f}"))"
        return 2
    fi
}

te_readable() {
    local f="$1"
    local host_f="${f}"
    [[ -f "${host_f}" ]] || host_f="${COMPARISON_DIR}/models/te/$(basename "${f}")"
    [[ -f "${host_f}" ]] || { error "TE file not found: ${f}"; return 2; }
    # GGUF arms load via the backend's gguf path (text stack + mmproj sidecar)
    # — safetensors probing does not apply; file-exists is the gate.
    if [[ "${host_f}" == *.gguf ]]; then
        echo "ok"
        return 0
    fi
    python3 - "${host_f}" <<'PYEOF'
import sys
from safetensors import safe_open
try:
    with safe_open(sys.argv[1], framework="pt") as f:
        _ = len(list(f.keys()))
    print("ok")
except FileNotFoundError:
    print("missing")
except Exception:
    print("unreadable")
PYEOF
}

# Reference base TE file for a variant: te_reference names an ARM — resolve
# that arm's te_file; empty/missing falls back to the global base_te_file.
reference_te_for() {
    local name="$1"
    python3 - "${COMPARISON_DIR}/comparison.json" "${name}" <<'PYEOF'
import json, sys
cmp = json.load(open(sys.argv[1]))
name = sys.argv[2]
ref = cmp["variants"].get(name, {}).get("te_reference") or ""
if ref and ref in cmp["variants"]:
    path = cmp["variants"][ref].get("te_file", "")
else:
    path = ""
print(path or cmp["settings"]["tebench"]["base_te_file"])
PYEOF
}

run_one() {
    local name="$1" te_file="$2" ref_file="$3"
    if [[ "${te_file}" == "${ref_file}" ]]; then
        log "SKIP ${name}: te_file == reference (self-comparison)"
        return 0
    fi
    if [[ "$(te_readable "${te_file}")" == "unreadable" ]]; then
        log "SKIP ${name}: $(basename "${te_file}") unreadable host-side (packed quant) — generation-only arm"
        return 0
    fi
    if [[ "$(te_readable "${ref_file}")" == "unreadable" ]]; then
        log "SKIP ${name}: reference $(basename "${ref_file}") unreadable — cannot pair"
        return 0
    fi
    local te_c ref_c
    te_c="$(te_container_path "${te_file}")" || exit 1
    ref_c="$(te_container_path "${ref_file}")" || exit 1
    log "Analyzing ${name}: $(basename "${te_file}") vs reference $(basename "${ref_file}")"
    docker_run_cpu "${TE_IMAGE}" \
        -v "${COMPARISON_DIR}:/comp:rw" \
        -v "${ABL_ROOT}:/app:ro" \
        -v "${COMFYUI_PATH}:/comfyui:ro" \
        -v "/home/maxx/extra/comfyui-models:/home/maxx/extra/comfyui-models:ro" \
        -e PYTHONPATH=/app \
        -- \
        -m src.text_encoder.run_analysis \
            --base "${ref_c}" \
            --variant "${te_c}" \
            --variant-name "${name}" \
            --dit "${DIT_FILE}" \
            --backend krea2 \
            --comfyui /comfyui \
            --prompts "/app/${PROMPT_SUITE}" \
            --results-dir "/comp/results/${COMPARISON_NAME}/text_encoder" \
            --cache-dir "/comp/results/.te_cache" \
            --device cpu \
            ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
}

# Variants: NAME=TE_FILE from VARIANT_TE_FILE_<name> exports
declare -A VARIANTS=()
while read -r envline; do
    name="${envline#VARIANT_TE_FILE_}"
    name="${name%%=*}"
    te_file="${envline#*=}"
    [[ -n "${te_file}" ]] || continue
    VARIANTS["${name}"]="${te_file}"
done < <(env | grep -E '^VARIANT_TE_FILE_[a-z0-9_]+=' || true)

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "DRY-RUN: backend=krea2 dit=$(basename "${DIT_FILE}") suite=${PROMPT_SUITE}"
    for name in "${!VARIANTS[@]}"; do
        log "DRY-RUN: ${name} ($(basename "${VARIANTS[${name}]}")) vs $(basename "$(reference_te_for "${name}")")"
    done
    exit 0
fi

if [[ -n "${VARIANT_FILTER}" ]]; then
    [[ -n "${VARIANTS[${VARIANT_FILTER}]+x}" ]] || { error "Unknown variant: ${VARIANT_FILTER}"; exit 1; }
    run_one "${VARIANT_FILTER}" "${VARIANTS[${VARIANT_FILTER}]}" "$(reference_te_for "${VARIANT_FILTER}")"
else
    for name in "${!VARIANTS[@]}"; do
        run_one "${name}" "${VARIANTS[${name}]}" "$(reference_te_for "${name}")"
    done
fi

log "Krea2 analysis complete: ${RESULTS_DIR}"
