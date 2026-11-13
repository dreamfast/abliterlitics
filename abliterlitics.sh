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
  text-encoder  Text-encoder conditioning fidelity analysis (tebench)
  te            Alias for text-encoder
  te-report     Generate SVG graphs + HTML report from text-encoder results
  te-img-metrics Image-space metrics over the A/B set (LPIPS/CLIP/SSIM)
  te-controls    Counterfactual conditioning battery (noise/taboo/shuffles/null)
  te-attention   Attention readout: what the DiT reads per TE
  te-discriminability Inter-prompt spread per TE (semantic-collapse detector)
  te-divergence DiT-step divergence analysis (paired sampling trajectories)
  te-directional Directional analysis: ablation delta vs taboo direction (tebench)
  graphs        Generate all graphs from existing results
  build         Build Docker images
  status        Show status of comparison results
  db            Database operations (init, import-results, pipeline-status, etc.)
  tables        Generate markdown tables from DB (benchmark, harmbench, kl, summary, all)
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
    # Optional exit-code arg: `USAGE 1` = usage-on-error (unknown command /
    # no args must exit nonzero for scriptable/CI callers); bare USAGE exits 0.
    exit "${1:-0}"
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
    db_init "$comp_dir"
    bash "${RUNNERS_DIR}/run_weights.sh" $flags "$comp_dir"
    bash "${RUNNERS_DIR}/run_kl.sh" $flags "$comp_dir"
    bash "${RUNNERS_DIR}/run_lm_eval.sh" $flags "$comp_dir"
    # load_comparison needed before run_harmbench.py (Python needs exported env vars)
    load_comparison "$comp_dir"
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
    # The comparison directory is the last positional arg that is an existing dir
    local comp_dir="."
    local args_after_shift=("$@")
    for (( i=${#args_after_shift[@]}-1; i>=0; i-- )); do
        arg="${args_after_shift[$i]}"
        if [[ ! "$arg" =~ ^- ]] && [[ -d "$arg" || -f "$arg/comparison.json" || "$(basename "$arg")" == "comparison.json" ]]; then
            comp_dir="$arg"
            break
        fi
    done
    load_comparison "$(find_comp_dir "$comp_dir")"
    python3 "${RUNNERS_DIR}/run_harmbench.py" $(global_flags) "$@"
}

cmd_text_encoder() {
    shift
    bash "${RUNNERS_DIR}/run_text_encoder.sh" $(global_flags) "$@"
}

# te-controls/te-attention/te-discriminability/te-img-metrics/te-report all
# receive the leading COMMAND slot (same dispatch shape as the other case
# arms) and drop it with an unconditional shift. The old `[[ --* ]] && shift`
# guard ran on USER args here (the case arm passed no $COMMAND) and silently
# ate the first user flag (e.g. `te-report --split-images <dir>` lost
# --split-images; audit root-packaging.md B3).
cmd_te_controls() {
    shift
    bash "${RUNNERS_DIR}/run_te_controls.sh" "$@"
}

cmd_te_attention() {
    shift
    bash "${RUNNERS_DIR}/run_te_attention.sh" "$@"
}

cmd_te_discriminability() {
    shift
    bash "${RUNNERS_DIR}/run_te_discriminability.sh" "$@"
}

cmd_te_img_metrics() {
    shift
    bash "${RUNNERS_DIR}/run_te_img_metrics.sh" "$@"
}

cmd_te_report() {
    # Pass ALL user args (flags like --split-images + the comparison dir).
    shift
    bash "${RUNNERS_DIR}/run_te_report.sh" "$@"
}

cmd_te_divergence() {
    shift
    bash "${RUNNERS_DIR}/run_te_divergence.sh" $(global_flags) "$@"
}

cmd_te_directional() {
    shift
    bash "${RUNNERS_DIR}/run_te_directional.sh" $(global_flags) "$@"
}

cmd_graphs() {
    local comp_dir
    comp_dir="$(find_comp_dir "${1:-.}")"
    load_comparison "$comp_dir"
    local results_dir
    results_dir="$(cd "${comp_dir}/results" && pwd)"
    local graphs_dir
    graphs_dir="$(cd "${comp_dir}" && pwd)/graphs"
    mkdir -p "$graphs_dir"

    log "Generating graphs for ${comp_dir}"
    # Run generate_graphs.py inside Docker (image ENTRYPOINT is python3)
    if [[ $GLOBAL_DRY_RUN -eq 1 ]]; then
        log "  [dry-run] docker_run auto ${FORENSICS_IMAGE} generate_graphs.py --label ${COMPARISON_NAME}"
        return
    fi

    docker_run "${GLOBAL_GPU:-$(detect_best_gpu)}" "$FORENSICS_IMAGE" \
        -v "${results_dir}:/results:ro" \
        -v "${graphs_dir}:/graphs" \
        -v "${ABL_ROOT}/src:/app/src:ro" \
        -e PYTHONPATH=/app \
        -- \
        /app/src/report/generate_graphs.py \
            --results-dir /results \
            --output-dir /graphs \
            --label "${COMPARISON_NAME}"
}

cmd_report() {
    # RETIRED: the provenance_report.py generator has been removed (audit
    # root-packaging.md B5 — it globbed for result files no runner produces
    # and printed hardcoded one-off conclusions). Graphs and DB tables remain:
    #   graphs → SVG graphs from result files (cmd_graphs)
    #   tables → markdown tables from the DB (cmd_tables)
    error "The 'report' subcommand was retired (provenance_report.py removed)."
    error "Use 'graphs' for SVG graphs or 'tables' for markdown tables from the DB."
    exit 1
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

    # Build gemma4unified lmeval image (Gemma4UnifiedForConditionalGeneration
    # support — AGENTS.md image table; see docker/Dockerfile.lmeval_gemma4_unified)
    local gemma4unified_image="${LMEVAL_GEMMA4UNIFIED_IMAGE:-abliterlitics-lmeval-gemma4unified:1.0.0}"
    log "  Building ${gemma4unified_image}..."
    [[ $GLOBAL_DRY_RUN -eq 0 ]] && docker build -t "$gemma4unified_image" -f "${docker_dir}/Dockerfile.lmeval_gemma4_unified" "$docker_dir"

    # Build llamacpp image (optional)
    log "  Building ${LLAMACPP_IMAGE}..."
    [[ $GLOBAL_DRY_RUN -eq 0 ]] && docker build -t "$LLAMACPP_IMAGE" -f "${docker_dir}/Dockerfile.llamacpp" "$docker_dir"

    # Build ik-llamacpp image (optional, requires commit SHA)
    log "  Building ${IK_LLAMACPP_IMAGE} (requires IK_LLAMA_CPP_COMMIT)..."
    [[ $GLOBAL_DRY_RUN -eq 0 ]] && docker build -t "$IK_LLAMACPP_IMAGE" -f "${docker_dir}/Dockerfile.ik-llamacpp" "$docker_dir"

    # Build text-encoder image (for tebench analysis)
    log "  Building ${TE_IMAGE}..."
    [[ $GLOBAL_DRY_RUN -eq 0 ]] && docker build -t "$TE_IMAGE" -f "${docker_dir}/Dockerfile.te" "$SCRIPT_DIR"

    # Not buildable from this repo: abliterlitics-lmeval-nightly:1.0.0 has no
    # Dockerfile in docker/ (referenced by AGENTS.md lesson #11 and
    # runners/common.sh) — see docker/README.md.
    log "  Skipping ${LMEVAL_NIGHTLY_IMAGE:-abliterlitics-lmeval-nightly:1.0.0}: no Dockerfile in docker/ (see docker/README.md)"

    log "Build complete"
}

cmd_status() {
    local comp_dir
    comp_dir="$(find_comp_dir "${1:-.}")"
    load_comparison "$comp_dir"
    echo "Comparison: ${COMPARISON_NAME}"
    echo "Variants:   ${VARIANT_NAMES}"
    echo ""

    # Try DB pipeline-status first; fall back to file counting if DB not initialized
    if python3 -m src.db pipeline-status "$comp_dir" 2>/dev/null; then
        :  # DB status printed successfully
    else
        _status_file_count "$comp_dir"
    fi
}

_status_file_count() {
    local comp_dir="$1"
    for subdir in weight kl lm_eval harmbench; do
        local count=0
        local dir="${comp_dir}/results/${subdir}"
        if [[ -d "$dir" ]]; then
            count=$(find "$dir" -type f -name '*.json' | wc -l)
        fi
        printf "  %-12s %d files\n" "$subdir" "$count"
    done
}

cmd_db() {
    # PYTHONPATH makes the module invocation cwd-independent (same pattern as
    # cmd_tables); without it `python3 -m src.db` fails outside the repo root,
    # and db_init's `2>/dev/null || true` would swallow that silently (B4).
    PYTHONPATH="${ABL_ROOT}" python3 -m src.db "$@"
}

cmd_tables() {
    local args=("$@")
    # If the last positional arg is a comparison directory, resolve to its name
    local last="${args[-1]:-}"
    if [[ -n "$last" ]]; then
        local stripped="$last"
        [[ "$stripped" == */ ]] && stripped="${stripped%/}"
        if [[ -f "${stripped}/comparison.json" ]]; then
            args[-1]="$(basename "$stripped")"
        fi
    fi
    PYTHONPATH="${ABL_ROOT}" python3 "${ABL_ROOT}/src/db_tables.py" "${args[@]}"
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
        errors=$((errors+1))
    fi
    for variant in $VARIANT_NAMES; do
        local variant_slug
        variant_slug="$(echo "$variant" | tr '-' '_')"
        local variant_path_var="VARIANT_PATH_${variant_slug}"
        if [[ ! -d "${!variant_path_var}" ]]; then
            error "Variant not found: ${!variant_path_var}"
            errors=$((errors+1))
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
    # Usage-on-error: nonzero exit so scriptable/CI callers detect it.
    USAGE 1
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
    text-encoder) cmd_text_encoder "$COMMAND" "$@" ;;
    te)           cmd_text_encoder "$COMMAND" "$@" ;;
    te-report)    cmd_te_report "$COMMAND" "$@" ;;
    te-img-metrics) cmd_te_img_metrics "$COMMAND" "$@" ;;
    te-controls)    cmd_te_controls "$COMMAND" "$@" ;;
    te-attention)   cmd_te_attention "$COMMAND" "$@" ;;
    te-discriminability) cmd_te_discriminability "$COMMAND" "$@" ;;
    te-divergence) cmd_te_divergence "$COMMAND" "$@" ;;
    te-directional) cmd_te_directional "$COMMAND" "$@" ;;
    graphs)    cmd_graphs "${1:-.}" ;;
    report)    cmd_report ;;
    build)     cmd_build ;;
    status)    cmd_status "${1:-.}" ;;
    db)        cmd_db "$@" ;;
    tables)    cmd_tables "$@" ;;
    clean)     cmd_clean "${1:-.}" ;;
    validate)  cmd_validate "${1:-.}" ;;
    *)         error "Unknown command: ${COMMAND}"; USAGE 1 ;;
esac
