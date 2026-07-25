#!/usr/bin/env bash
# Weight analysis orchestrator — runs 11 weight scripts via Docker
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <comparison_dir>

Weight analysis pipeline (11 scripts):
   panel        — Panel comparison (which tensors changed, by how much)
   edit         — Edit vector analysis (direction and magnitude)
   svd          — SVD decomposition of edit vectors
   fingerprint  — Technique fingerprinting
   layer        — Layer-wise analysis
   correlation  — Cross-technique correlation
   subspace     — Subspace alignment
   lowrank      — Low-rank reconstruction
   expert       — Expert analysis (MoE models only)
   stacking     — Stacking investigation
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
# Usage: run_script <script_name> <output_file> [extra_args...]
run_script() {
    local script_name="$1"
    local output_file="$2"
    shift 2

    local out_path="${RESULTS_DIR}/${COMPARISON_NAME}/weight/${output_file}"
    if result_exists "$out_path"; then
        log "Skipping (exists): ${output_file}"
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
            --results-dir /results \
            "$@"
}

# Phase 1: Panel comparison
if [[ $DRY_RUN -eq 0 ]]; then
    db_stage_start "$COMPARISON_NAME" "all" "weights"
fi
run_script "panel_comparison.py" "panel_comparison.json"

# NOTE: Each analysis script, when invoked with --comparison, loops over ALL
# variants (or ALL variant pairs) internally. They are therefore invoked ONCE
# here. Invoking once per variant/pair caused Nx (and N^2) redundant Docker
# runs that multiplied runtime by ~22x (and 231x for pairwise).

# Phase 2: Edit vectors (script loops all variants internally)
for variant in $VARIANT_NAMES; do
    if [[ $DRY_RUN -eq 0 ]]; then
        db_stage_start "$COMPARISON_NAME" "$variant" "weights"
    fi
done
run_script "edit_vector_analysis.py" "edit_vectors.json"

# Phase 3: SVD (script loops all variants internally)
run_script "svd_analysis.py" "svd_all.json"

# Phase 4: Technique fingerprint (script loops all variants internally)
run_script "technique_fingerprint.py" "fingerprint_all.json"

# Phase 5: Layer analysis (script loops all variants internally)
run_script "layer_analysis.py" "layer_all.json"

# Phase 6: Cross-technique correlation (script loops all pairs internally w/ skip)
run_script "technique_correlation.py" "correlation_all.json"

# Phase 7: Subspace alignment (script loops all pairs internally w/ skip)
run_script "subspace_alignment.py" "subspace_all.json"

# Phase 8: Low-rank reconstruction (script loops all pairs internally w/ skip)
run_script "lowrank_reconstruction.py" "lowrank_all.json"

# Phase 9: Expert analysis (only for MoE models)
run_script "expert_analysis.py" "expert_analysis.json"

# Phase 10: Cross-architecture comparison (only meaningful when multiple
# architectures/comparisons are available; non-fatal — single-architecture
# runs are expected to skip it).
if ! run_script "cross_arch_comparison.py" "cross_arch.json"; then
    log "cross_arch_comparison.py did not complete (expected for single-architecture comparisons) — continuing"
fi

# Import results to DB
if [[ $DRY_RUN -eq 0 ]]; then
    db_import_results "$COMP_DIR"
fi

log "Weight analysis complete"
