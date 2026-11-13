#!/usr/bin/env bash
# Sequencer: runs all weight-analysis phases in one container.
# Output goes to a file (not stdout pipe) so SIGPIPE can't kill it.
set -uo pipefail

LOG=/results/sequencer.log
COMP=/comparison
RESULTS=/results
PY="python3"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

log "=== sequencer start (pid $$, host=$(hostname)) ==="
log "comparison=$COMP results=$RESULTS"

# Phases in order. Each loops variants/pairs internally.
PHASES=(
    "panel_comparison.py"
    "edit_vector_analysis.py"
    "svd_analysis.py"
    "technique_fingerprint.py"
    "layer_analysis.py"
    "technique_correlation.py"
    "subspace_alignment.py"
    "lowrank_reconstruction.py"
    "expert_analysis.py"
    "cross_arch_comparison.py"
)

for phase in "${PHASES[@]}"; do
    log "--- phase: $phase ---"
    start=$(date +%s)
    if $PY "/app/src/weight/$phase" --comparison "$COMP" --results-dir "$RESULTS" >> "$LOG" 2>&1; then
        elapsed=$(( $(date +%s) - start ))
        log "--- $phase: OK (${elapsed}s) ---"
    else
        rc=$?
        elapsed=$(( $(date +%s) - start ))
        log "--- $phase: FAIL rc=$rc (${elapsed}s) ---"
    fi
done

log "=== sequencer done ==="
