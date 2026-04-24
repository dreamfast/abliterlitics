#!/usr/bin/env bash
# Shared functions for Abliterlitics runners
set -uo pipefail

ABL_VERSION="1.0.0"

# Logging
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
error() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2; }

# Docker image names (versioned)
FORENSICS_IMAGE="abliterlitics-forensics:1.0.0"
LMEVAL_IMAGE="abliterlitics-lmeval:1.0.0"
LLAMACPP_IMAGE="abliterlitics-llamacpp:1.0.0"
IK_LLAMACPP_IMAGE="abliterlitics-ik-llamacpp:1.0.0"

# Source directory (for mounting into Docker)
ABL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Docker run helper — uses --runtime=nvidia (never --gpus)
# Usage: docker_run GPU_INDEX IMAGE [additional docker args] -- [command]
docker_run() {
    local gpu="$1"; shift
    local image="$1"; shift
    local docker_args=()
    local command=()
    local found_separator=0
    for arg in "$@"; do
        if [[ "$arg" == "--" && $found_separator -eq 0 ]]; then
            found_separator=1; continue
        fi
        if [[ $found_separator -eq 1 ]]; then
            command+=("$arg")
        else
            docker_args+=("$arg")
        fi
    done
    docker run --rm --runtime=nvidia \
        -e "NVIDIA_VISIBLE_DEVICES=${gpu}" \
        -e "CUDA_VISIBLE_DEVICES=0" \
        "${docker_args[@]}" \
        "$image" \
        "${command[@]}"
}

# Load comparison.json using a Python helper script.
# All values are shlex-quoted to prevent shell injection.
# Passes paths as sys.argv (no string interpolation into Python source).
# Exports: COMPARISON_NAME, COMPARISON_DIR, BASE_DIR, VARIANT_COUNT,
#          VARIANT_NAMES (space-separated), and VARIANT_PATH_<slug> for each variant.
# Also exports: INFERENCE_BACKEND, LM_EVAL_TASKS
load_comparison() {
    local comp_dir="$1"
    local comp_json
    if [[ -f "${comp_dir}/comparison.json" ]]; then
        comp_json="${comp_dir}/comparison.json"
    elif [[ -f "${comp_dir}" && "$(basename "${comp_dir}")" == "comparison.json" ]]; then
        comp_json="${comp_dir}"
        comp_dir="$(dirname "${comp_dir}")"
    else
        error "comparison.json not found: ${comp_dir}"
        return 1
    fi
    eval "$(python3 "${ABL_ROOT}/runners/_load_comparison.py" "${comp_json}" "${comp_dir}")"
    log "Loaded comparison: $COMPARISON_NAME ($VARIANT_COUNT variants)"
}

# Detect best GPU (most VRAM)
detect_best_gpu() {
    nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits 2>/dev/null | \
        sort -t',' -k2 -nr | head -1 | cut -d',' -f1 | tr -d ' '
}

# Check if result file exists (for skip-existing)
result_exists() {
    [[ "${SKIP_EXISTING:-1}" == "1" && -f "$1" ]] && return 0
    return 1
}
