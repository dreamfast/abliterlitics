#!/usr/bin/env bash
# Shared functions for Abliterlitics runners
set -uo pipefail

ABL_VERSION="1.0.0"

# Logging
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
error() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2; }

# Docker image names (versioned).
# These honor env-var overrides so per-comparison selection works: the
# _load_comparison.py loader exports LMEVAL_IMAGE when comparison.json
# sets "settings.lmeval_image" — useful for models that need a newer vLLM
# (e.g. Gemma 4 thinking → abliterlitics-lmeval-nightly:1.0.0 with the
# gemma4_reasoning_parser; see AGENTS.md lesson #11 / NOTES.md).
FORENSICS_IMAGE="${FORENSICS_IMAGE:-abliterlitics-forensics:1.0.0}"
LMEVAL_IMAGE="${LMEVAL_IMAGE:-abliterlitics-lmeval:1.0.0}"
LLAMACPP_IMAGE="${LLAMACPP_IMAGE:-abliterlitics-llamacpp:1.0.0}"
IK_LLAMACPP_IMAGE="${IK_LLAMACPP_IMAGE:-abliterlitics-ik-llamacpp:1.0.0}"
TE_IMAGE="${TE_IMAGE:-abliterlitics-te:1.2.0}"

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

# Docker run helper for CPU-only containers (no --runtime=nvidia).
# Used by text-encoder analysis which runs on CPU by default.
# Usage: docker_run_cpu IMAGE [additional docker args] -- [command]
docker_run_cpu() {
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
    docker run --rm \
        "${docker_args[@]}" \
        "$image" \
        "${command[@]}"
}

# Map a host path to its in-container path for the TE battery mounts.
# Shared implementation (was byte-identical in 6 run_te_*/run_text_encoder
# runners — audit 2026-10-14 duplication set #1). Relies on these globals
# being set by the CALLER before first use:
#   COMPARISON_DIR     → /comparison
#   MODELS_HOST_ROOT   → /models
#   COMFYUI_HOST_PATH  → /comfyui
# Paths under none of these roots pass through unchanged (e.g. the comfyui
# models dir, which is mounted 1:1 at its own absolute path — AGENTS.md #40).
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

# Detect best GPU (most VRAM). Defaults to GPU 0 if nvidia-smi is unavailable
# on the host (e.g. when running inside a container or headless environment).
detect_best_gpu() {
    local gpu
    gpu=$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits 2>/dev/null | \
        sort -t',' -k2 -nr | head -1 | cut -d',' -f1 | tr -d ' ')
    echo "${gpu:-0}"
}

# Check if result file exists (for skip-existing)
result_exists() {
    [[ "${SKIP_EXISTING:-1}" == "1" && -f "$1" ]] && return 0
    return 1
}

# ---------------------------------------------------------------------------
# Database helpers — all call python3 -m src.db
# All are fire-and-forget (|| true) — DB failures never block the pipeline.
# All should be guarded by [[ $DRY_RUN -eq 0 ]] in the calling runner.
# ---------------------------------------------------------------------------

db_init() {
    local comp_dir="$1"
    python3 -m src.db init "$comp_dir" 2>/dev/null || true
}

db_stage_start() {
    local comp="$1" model="$2" stage="$3"
    python3 -m src.db stage-start --comparison "$comp" --model "$model" --stage "$stage" 2>/dev/null || true
}

# Lesson #83 guard: read-only check whether a model stage is 'completed'.
# No-work skip paths must never demote a completed stage — wrap status writes
# as:  [[ "$(stage_completed C M S)" != "yes" ]] && db_stage_skip/start ...
# Best-effort: on ANY error returns "no" (write proceeds exactly as before the
# guard existed). DB is read via file:...?mode=ro; requires the runner's cwd
# to be the repo root (same assumption as the db_stage_* helpers above).
stage_completed() {
    python3 - "$1" "$2" "$3" <<'PYEOF'
import sqlite3, sys
try:
    conn = sqlite3.connect("file:abliterlitics.db?mode=ro", uri=True)
    row = conn.execute(
        "SELECT 1 FROM model_stages WHERE comparison=? AND model_name=? AND stage=? AND status='completed'",
        (sys.argv[1], sys.argv[2], sys.argv[3]),
    ).fetchone()
    print("yes" if row else "no")
except Exception:
    print("no")
PYEOF
}

db_stage_complete() {
    local comp="$1" model="$2" stage="$3"
    shift 3
    python3 -m src.db stage-complete --comparison "$comp" --model "$model" --stage "$stage" "$@" 2>/dev/null || true
}

db_stage_fail() {
    local comp="$1" model="$2" stage="$3" error="$4"
    python3 -m src.db stage-fail --comparison "$comp" --model "$model" --stage "$stage" --error "$error" 2>/dev/null || true
}

db_stage_skip() {
    local comp="$1" model="$2" stage="$3"
    python3 -m src.db stage-skip --comparison "$comp" --model "$model" --stage "$stage" 2>/dev/null || true
}

db_import_results() {
    local comp_dir="$1"
    python3 -m src.db import-results "$comp_dir" 2>/dev/null || true
}
