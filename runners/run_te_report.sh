#!/usr/bin/env bash
# Text-encoder report generator runner — SVG graphs + HTML report from results.
# CPU-only, no model loading needed. Runs in the abliterlitics-te image.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") <comparison_dir>

Generates SVG graphs + a self-contained HTML report (te_report.html) from
existing text-encoder results (results/<name>/text_encoder/te_*.json).

Output: <comparison_dir>/graphs/te_*.svg + <comparison_dir>/graphs/te_report.html
EOF
    exit 0
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage

COMP_DIR="${1:?Error: comparison_dir required. Use --help for usage.}"

load_comparison "$COMP_DIR"
COMPARISON_DIR="$(cd "$(dirname "${COMPARISON_DIR}")" && pwd)/$(basename "${COMPARISON_DIR}")"
RESULTS_DIR="${COMPARISON_DIR}/results/${COMPARISON_NAME}/text_encoder"
GRAPHS_DIR="${COMPARISON_DIR}/graphs"

if [[ ! -d "$RESULTS_DIR" ]]; then
    error "No text-encoder results found: ${RESULTS_DIR}"
    error "Run './abliterlitics.sh text-encoder ${COMP_DIR}' first"
    exit 1
fi

mkdir -p "$GRAPHS_DIR"

log "Generating TE graphs + HTML report for ${COMPARISON_NAME}"

docker_run_cpu "$TE_IMAGE" \
    -v "${RESULTS_DIR}:/results:ro" \
    -v "${GRAPHS_DIR}:/graphs" \
    -v "${ABL_ROOT}:/app:ro" \
    -e PYTHONPATH=/app \
    -- \
    /app/src/text_encoder/generate_report.py \
        --results-dir /results \
        --output-dir /graphs

log "Report complete: ${GRAPHS_DIR}/te_report.html"
