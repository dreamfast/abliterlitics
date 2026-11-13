#!/usr/bin/env bash
# Text-encoder report generator runner — SVG graphs + HTML report from results.
# CPU-only, no model loading needed. Runs in the abliterlitics-te image.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") [--split-images] <comparison_dir>

Generates SVG graphs + the HTML report (te_report.html) from existing
text-encoder results (results/<name>/text_encoder/te_*.json).

Options:
   --split-images   Write A/B gallery images to graphs/ab_images/ with
                    relative paths instead of base64-embedding them
                    (smaller HTML for large A/B matrices; keep the folder
                    next to te_report.html when hosting).

Output: <comparison_dir>/graphs/te_*.svg + te_report.html [+ ab_images/]
EOF
    exit 0
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage

# Parse args: flags first, then the single positional comparison dir.
EXTRA_ARGS=()
COMP_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --split-images) EXTRA_ARGS+=("--split-images"); shift ;;
        -*) shift ;;  # ignore passthrough flags we don't know (--dry-run etc.)
        *)
            [[ -n "$COMP_DIR" ]] && { error "Multiple comparison dirs given"; usage; }
            COMP_DIR="$1"; shift ;;
    esac
done
[[ -z "$COMP_DIR" ]] && { error "comparison_dir required"; usage; }

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
        --output-dir /graphs \
        "${EXTRA_ARGS[@]}"

log "Report complete: ${GRAPHS_DIR}/te_report.html"
