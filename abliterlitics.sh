#!/usr/bin/env bash
# Abliterlitics — Main CLI entry point
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNERS_DIR="${SCRIPT_DIR}/runners"
# shellcheck source=runners/common.sh
source "${RUNNERS_DIR}/common.sh"

USAGE() {
    cat <<EOF
Abliterlitics v${ABL_VERSION} — Model comparison analysis toolkit

Usage: $(basename "$0") [GLOBAL_OPTIONS] <COMMAND> [COMMAND_ARGS]

Commands:
  auto          Run full pipeline (weights + kl + lm-eval + harmbench)
  weights       Weight analysis (panel, edit, svd, correlation, novel, graphs)
  kl            KL divergence analysis
  lm-eval       Run lm-evaluation-harness
  harmbench     Run HarmBench evaluation
  graphs        Generate all graphs from existing results
  report        Generate provenance report from results
  build         Build Docker images
  status        Show status of comparison results
  clean         Remove generated results
  validate      Validate comparison.json structure

Global Options:
  --gpu GPU          GPU index (default: auto-detect)
  --backend BACKEND  Override inference backend (vllm/llamacpp/ik_llamacpp/auto)
  --skip-existing    Skip steps with existing results (default)
  --force            Overwrite existing results
  --dry-run          Show what would run without executing
  -h, --help         Show this help
  -v, --version      Show version

Examples:
  $(basename "$0") auto ./my_comparison
  $(basename "$0") --gpu 2 weights ./my_comparison
  $(basename "$0") --force lm-eval ./my_comparison
  $(basename "$0") status ./my_comparison
EOF
    exit 0
}

VERSION() { echo "Abliterlitics v${ABL_VERSION}"; exit 0; }

GLOBAL_GPU=""
GLOBAL_BACKEND=""
GLOBAL_SKIP=1
GLOBAL_FORCE=0
GLOBAL_DRY_RUN=0

global_flags() {
    local flags=()
    [[ -n "$GLOBAL_GPU" ]] && flags+=(--gpu "$GLOBAL_GPU")
    [[ -n "$GLOBAL_BACKEND" ]] && flags+=(--backend "$GLOBAL_BACKEND")
    [[ $GLOBAL_SKIP -eq 1 ]] && flags+=(--skip-existing)
    [[ $GLOBAL_FORCE -eq 1 ]] && flags+=(--force)
    [[ $GLOBAL_DRY_RUN -eq 1 ]] && flags+=(--dry-run)
    echo "${flags[*]}"
}

find_comp_dir() {
    local dir="$1"
    if [[ -f "${dir}/comparison.json" || -f "${dir}" ]]; then
        echo "$dir"
    else
        error "comparison.json not found in: ${dir}"
        exit 1
    fi
}

cmd_auto() {
    local comp_dir
    comp_dir="$(find_comp_dir "$1")"
    local flags
    flags="$(global_flags)"
    log "Running full pipeline for ${comp_dir}"
    bash "${RUNNERS_DIR}/run_weights.sh" $flags "$comp_dir"
    bash "${RUNNERS_DIR}/run_kl.sh" $flags "$comp_dir"
    bash "${RUNNERS_DIR}/run_lm_eval.sh" $flags "$comp_dir"
    python3 "${RUNNERS_DIR}/run_harmbench.py" $flags "$comp_dir"
    cmd_graphs "$comp_dir"
}

cmd_weights() {
    shift
    bash "${RUNNERS_DIR}/run_weights.sh" $(global_flags) "$@"
}

cmd_kl() {
    shift
    bash "${RUNNERS_DIR}/run_kl.sh" $(global_flags) "$@"
}

cmd_lm_eval() {
    shift
    bash "${RUNNERS_DIR}/run_lm_eval.sh" $(global_flags) "$@"
}

cmd_harmbench() {
    shift
    python3 "${RUNNERS_DIR}/run_harmbench.py" $(global_flags) "$@"
}

cmd_graphs() {
    local comp_dir
    comp_dir="$(find_comp_dir "${1:-.}")"
    local flags
    local results_dir="${comp_dir}/results"
    local graphs_dir="${comp_dir}/graphs"
    mkdir -p "$graphs_dir"

    log "Generating graphs for ${comp_dir}"
    # Run generate_graphs.py inside Docker
    if [[ $GLOBAL_DRY_RUN -eq 1 ]]; then
        log "  [dry-run] docker_run auto ${FORENSICS_IMAGE} generate_graphs.py"
        return
    fi

    docker_run "${GLOBAL_GPU:-$(detect_best_gpu)}" "$FORENSICS_IMAGE" \
        -v "${results_dir}:/results:ro" \
        -v "${graphs_dir}:/graphs" \
        -v "${ABL_ROOT}/src:/app/src:ro" \
        -e PYTHONPATH=/app/src \
        -- \
        python3 /app/src/report/generate_graphs.py \
            --results-dir /results \
            --output-dir /graphs
}

cmd_report() {
    local comp_dir
    comp_dir="$(find_comp_dir "${1:-.}")"
    local results_dir="${comp_dir}/results"
    local report_dir="${comp_dir}/report"
    mkdir -p "$report_dir"

    log "Generating report for ${comp_dir}"
    if [[ $GLOBAL_DRY_RUN -eq 1 ]]; then
        log "  [dry-run] docker_run auto ${FORENSICS_IMAGE} provenance_report.py"
        return
    fi

    docker_run "${GLOBAL_GPU:-$(detect_best_gpu)}" "$FORENSICS_IMAGE" \
        -v "${results_dir}:/results:ro" \
        -v "${report_dir}:/report" \
        -v "${ABL_ROOT}/src:/app/src:ro" \
        -e PYTHONPATH=/app/src \
        -- \
        python3 /app/src/report/provenance_report.py \
            --results-dir /results \
            --output-dir /report
}

cmd_build() {
    log "Building Docker images..."
    local docker_dir="${SCRIPT_DIR}/docker"

    # Build forensics image (most important)
    log "  Building ${FORENSICS_IMAGE}..."
    [[ $GLOBAL_DRY_RUN -eq 0 ]] && docker build -t "$FORENSICS_IMAGE" -f "${docker_dir}/Dockerfile.forensics" "$SCRIPT_DIR"

    # Build lmeval image
    log "  Building ${LMEVAL_IMAGE}..."
    [[ $GLOBAL_DRY_RUN -eq 0 ]] && docker build -t "$LMEVAL_IMAGE" -f "${docker_dir}/Dockerfile.lmeval" "$docker_dir"

    # Build llamacpp image (optional)
    log "  Building ${LLAMACPP_IMAGE}..."
    [[ $GLOBAL_DRY_RUN -eq 0 ]] && docker build -t "$LLAMACPP_IMAGE" -f "${docker_dir}/Dockerfile.llamacpp" "$docker_dir"

    # Build ik-llamacpp image (optional, requires commit SHA)
    log "  Building ${IK_LLAMACPP_IMAGE} (requires IK_LLAMA_CPP_COMMIT)..."
    [[ $GLOBAL_DRY_RUN -eq 0 ]] && docker build -t "$IK_LLAMACPP_IMAGE" -f "${docker_dir}/Dockerfile.ik-llamacpp" "$docker_dir"

    log "Build complete"
}

cmd_status() {
    local comp_dir
    comp_dir="$(find_comp_dir "${1:-.}")"
    load_comparison "$comp_dir"
    echo "Comparison: ${COMPARISON_NAME}"
    echo "Variants:   ${VARIANT_NAMES}"
    echo ""
    for subdir in weight kl lm_eval harmbench; do
        local count=0
        local dir="${comp_dir}/results/${subdir}"
        if [[ -d "$dir" ]]; then
            count=$(find "$dir" -type f -name '*.json' | wc -l)
        fi
        printf "  %-12s %d files\n" "$subdir" "$count"
    done
}

cmd_clean() {
    local comp_dir
    comp_dir="$(find_comp_dir "${1:-.}")"
    local results_dir="${comp_dir}/results"
    if [[ $GLOBAL_DRY_RUN -eq 1 ]]; then
        log "[dry-run] Would remove: ${results_dir}"
        return
    fi
    rm -rf "$results_dir"
    log "Cleaned: ${results_dir}"
}

cmd_validate() {
    local comp_dir
    comp_dir="$(find_comp_dir "${1:-.}")"
    log "Validating ${comp_dir}..."
    load_comparison "$comp_dir"

    local errors=0
    if [[ ! -d "${BASE_DIR}" ]]; then
        error "Base model not found: ${BASE_DIR}"
        ((errors++))
    fi
    for variant in $VARIANT_NAMES; do
        local variant_path_var="VARIANT_PATH_${variant}"
        if [[ ! -d "${!variant_path_var}" ]]; then
            error "Variant not found: ${!variant_path_var}"
            ((errors++))
        fi
    done

    if [[ $errors -eq 0 ]]; then
        log "Validation passed: ${COMPARISON_NAME} (${VARIANT_COUNT} variants)"
    else
        error "Validation failed: ${errors} error(s)"
        return 1
    fi
}

if [[ $# -lt 1 ]]; then
    USAGE
fi

# Parse global args in the main scope (not a subshell)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)         GLOBAL_GPU="$2"; shift 2 ;;
        --backend)     GLOBAL_BACKEND="$2"; shift 2 ;;
        --skip-existing) GLOBAL_SKIP=1; shift ;;
        --force)       GLOBAL_FORCE=1; GLOBAL_SKIP=0; shift ;;
        --dry-run)     GLOBAL_DRY_RUN=1; shift ;;
        -h|--help)     USAGE ;;
        -v|--version)  VERSION ;;
        *)             break ;;
    esac
done

COMMAND="${1:-}"
shift || true

case "$COMMAND" in
    auto)      cmd_auto "${1:-.}" ;;
    weights)   cmd_weights "$COMMAND" "$@" ;;
    kl)        cmd_kl "$COMMAND" "$@" ;;
    lm-eval)   cmd_lm_eval "$COMMAND" "$@" ;;
    harmbench) cmd_harmbench "$COMMAND" "$@" ;;
    graphs)    cmd_graphs "${1:-.}" ;;
    report)    cmd_report "${1:-.}" ;;
    build)     cmd_build ;;
    status)    cmd_status "${1:-.}" ;;
    clean)     cmd_clean "${1:-.}" ;;
    validate)  cmd_validate "${1:-.}" ;;
    *)         error "Unknown command: ${COMMAND}"; USAGE ;;
esac
