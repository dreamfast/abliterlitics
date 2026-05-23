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
run_script "panel_comparison.py" "panel_comparison.json"

# Phase 2: Edit vectors — one per variant
for variant in $VARIANT_NAMES; do
    run_script "edit_vector_analysis.py" "edit_${variant}.json"
done

# Phase 3: SVD — one per variant
for variant in $VARIANT_NAMES; do
    run_script "svd_analysis.py" "svd_${variant}.json"
done

# Phase 4: Technique fingerprint — one per variant
for variant in $VARIANT_NAMES; do
    run_script "technique_fingerprint.py" "fingerprint_${variant}.json"
done

# Phase 5: Layer analysis — one per variant
for variant in $VARIANT_NAMES; do
    run_script "layer_analysis.py" "layer_${variant}.json"
done

# Phase 6: Cross-technique correlation (pairwise)
names_arr=($VARIANT_NAMES)
for ((i=0; i<${#names_arr[@]}; i++)); do
    for ((j=i+1; j<${#names_arr[@]}; j++)); do
        v1="${names_arr[$i]}"
        v2="${names_arr[$j]}"
        run_script "technique_correlation.py" "correlation_${v1}_vs_${v2}.json"
    done
done

# Phase 7: Subspace alignment (pairwise)
for ((i=0; i<${#names_arr[@]}; i++)); do
    for ((j=i+1; j<${#names_arr[@]}; j++)); do
        v1="${names_arr[$i]}"
        v2="${names_arr[$j]}"
        run_script "subspace_alignment.py" "subspace_${v1}_vs_${v2}.json"
    done
done

# Phase 8: Low-rank reconstruction (pairwise)
for ((i=0; i<${#names_arr[@]}; i++)); do
    for ((j=i+1; j<${#names_arr[@]}; j++)); do
        v1="${names_arr[$i]}"
        v2="${names_arr[$j]}"
        run_script "lowrank_reconstruction.py" "lowrank_${v1}_vs_${v2}.json"
    done
done

# Phase 9: Expert analysis (only for MoE models)
run_script "expert_analysis.py" "expert_analysis.json"

# Phase 10: Cross-architecture comparison (if multiple comparisons available)
run_script "cross_arch_comparison.py" "cross_arch.json"

log "Weight analysis complete"
