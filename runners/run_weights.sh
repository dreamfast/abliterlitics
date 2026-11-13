#!/usr/bin/env bash
# Weight analysis orchestrator — runs 10 weight scripts via Docker
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <comparison_dir>

Weight analysis pipeline (10 scripts — stacking_analysis.py is orphaned
from the pipeline and not invoked here):
   panel        — Panel comparison (which tensors changed, by how much)
   edit         — Edit vector analysis (direction and magnitude)
   svd          — SVD decomposition of edit vectors
   fingerprint  — Technique fingerprinting
   layer        — Layer-wise analysis
   correlation  — Cross-technique correlation
   subspace     — Subspace alignment
   lowrank      — Low-rank reconstruction
   expert       — Expert analysis (MoE models only)
   cross_arch   — Cross-architecture comparison

Options:
   --gpu GPU          GPU index (default: auto-detect)
   --skip-existing    Skip scripts with existing results (default)
   --force            Overwrite existing results
   --dry-run          Show what would run without executing
   -h, --help         Show this help
EOF
    exit 0
}

GPU=""
DRY_RUN=0
SKIP_EXISTING=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)           GPU="$2"; shift 2 ;;
        --backend)       shift 2 ;;  # consumed for CLI compat, weights always uses Docker
        --skip-existing) SKIP_EXISTING=1; shift ;;
        --force)         SKIP_EXISTING=0; shift ;;
        --dry-run)       DRY_RUN=1; shift ;;
        -h|--help)       usage ;;
        *)               break ;;
    esac
done

COMP_DIR="${1:?Error: comparison_dir required. Use --help for usage.}"

load_comparison "$COMP_DIR"
GPU="${GPU:-$(detect_best_gpu)}"
COMPARISON_DIR="$(cd "$(dirname "${COMPARISON_DIR}")" && pwd)/$(basename "${COMPARISON_DIR}")"
RESULTS_DIR="${COMPARISON_DIR}/results"
mkdir -p "$RESULTS_DIR"

# Initialize DB tracking
if [[ $DRY_RUN -eq 0 ]]; then
    db_init "$COMP_DIR"
fi

# Run a weight analysis script inside Docker.
# Mounts comparison dir at /comparison and results at /results.
# Usage: run_script <script_name> <output_spec>
#   output_spec tells the skip-check which files the script ACTUALLY writes,
#   as paths RELATIVE to RESULTS_DIR (lesson #28c — the old check looked under
#   ${RESULTS_DIR}/${COMPARISON_NAME}/weight/<file>, which nothing writes, so
#   the skip never triggered and every re-run re-executed all 10 phases):
#     file:<name>          single aggregate at the results root
#     per-variant:<prefix> <variant>/<prefix>_<variant>.json for every variant
#     per-pair:<prefix>    <prefix>_<a>_vs_<b>.json for every unordered pair
#   (paths verified against src/weight/*.py output code and the gemma4-12b
#   results layout; _common.py:variant_output_path + pairwise run_analysis.)
run_script() {
    local script_name="$1"
    local spec="$2"

    local -a expected=()
    local kind="${spec%%:*}" arg="${spec#*:}"
    local v
    case "$kind" in
        file)        expected=("$arg") ;;
        per-variant) for v in $VARIANT_NAMES; do expected+=("${v}/${arg}_${v}.json"); done ;;
        per-pair)    local -a pair_names=($VARIANT_NAMES)
                     local pi pj pn=${#pair_names[@]}
                     for ((pi=0; pi<pn; pi++)); do
                         for ((pj=pi+1; pj<pn; pj++)); do
                             expected+=("${arg}_${pair_names[pi]}_vs_${pair_names[pj]}.json")
                         done
                     done ;;
        *) error "run_script: bad output spec '${spec}'"; return 1 ;;
    esac

    local p all_exist=1
    for p in "${expected[@]}"; do
        if [[ ! -f "${RESULTS_DIR}/${p}" ]]; then all_exist=0; break; fi
    done
    if [[ $all_exist -eq 1 ]] && result_exists "${RESULTS_DIR}/${expected[0]}"; then
        log "Skipping (all ${#expected[@]} outputs exist): ${script_name}"
        return 0
    fi

    log "Running ${script_name}"
    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [dry-run] docker_run ${GPU} ${FORENSICS_IMAGE} ${script_name} --comparison /comparison"
        return 0
    fi

    docker_run "$GPU" "$FORENSICS_IMAGE" \
        -v "${COMPARISON_DIR}:/comparison:ro" \
        -v "${ABL_ROOT}/models:/models:ro" \
        -v "${RESULTS_DIR}:/results" \
        -v "${ABL_ROOT}:/app:ro" \
        -e PYTHONPATH=/app \
        -- \
        "/app/src/weight/${script_name}" \
            --comparison /comparison \
            --results-dir /results
}

# Lesson #83 guard: only mark a stage 'running' when it isn't already
# 'completed' — skip-existing re-runs would otherwise demote completed stages
# that this runner never restores (stage-complete is done externally).
mark_weights_stage_start() {
    if [[ $DRY_RUN -eq 0 ]] && [[ "$(stage_completed "$1" "$2" "weights")" != "yes" ]]; then
        db_stage_start "$1" "$2" "weights"
    fi
}

# Phase 1: Panel comparison
mark_weights_stage_start "$COMPARISON_NAME" "all"
run_script "panel_comparison.py" "file:multi_model_panel.json"

# NOTE: Each analysis script, when invoked with --comparison, loops over ALL
# variants (or ALL variant pairs) internally. They are therefore invoked ONCE
# here. Invoking once per variant/pair caused Nx (and N^2) redundant Docker
# runs that multiplied runtime by ~22x (and 231x for pairwise).

# Phase 2: Edit vectors (script loops all variants internally)
for variant in $VARIANT_NAMES; do
    mark_weights_stage_start "$COMPARISON_NAME" "$variant"
done
run_script "edit_vector_analysis.py" "per-variant:edit_vector"

# Phase 3: SVD (script loops all variants internally)
run_script "svd_analysis.py" "per-variant:svd"

# Phase 4: Technique fingerprint (script loops all variants internally)
run_script "technique_fingerprint.py" "per-variant:fingerprint"

# Phase 5: Layer analysis (script loops all variants internally)
run_script "layer_analysis.py" "per-variant:layer_analysis"

# Phase 6: Cross-technique correlation (script loops all pairs internally w/ skip)
run_script "technique_correlation.py" "per-pair:correlation"

# Phase 7: Subspace alignment (script loops all pairs internally w/ skip)
run_script "subspace_alignment.py" "per-pair:subspace"

# Phase 8: Low-rank reconstruction (script loops all pairs internally w/ skip)
run_script "lowrank_reconstruction.py" "per-pair:lowrank"

# Phase 9: Expert analysis (only for MoE models)
run_script "expert_analysis.py" "per-variant:expert_analysis"

# Phase 10: Cross-architecture comparison (only meaningful when multiple
# architectures/comparisons are available; non-fatal — single-architecture
# runs are expected to skip it).
if ! run_script "cross_arch_comparison.py" "file:cross_arch.json"; then
    log "cross_arch_comparison.py did not complete (expected for single-architecture comparisons) — continuing"
fi

# Import results to DB
if [[ $DRY_RUN -eq 0 ]]; then
    db_import_results "$COMP_DIR"
fi

log "Weight analysis complete"
