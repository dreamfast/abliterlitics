#!/usr/bin/env bash
# Monitor the Abliterix HarmBench generation run.
# Checks progress every 20 minutes, logs status.
# Usage: bash runners/monitor_abliterix.sh
set -euo pipefail

PARTIAL="results/qwen36-27b/harmbench/harmbench_abliterix_responses.partial.json"
FINAL="results/qwen36-27b/harmbench/harmbench_abliterix_responses.json"
INTERVAL=1200  # 20 minutes

log() { echo "[$(date '+%H:%M:%S')] $*"; }

if [[ ! -f "$PARTIAL" ]]; then
    echo "ERROR: $PARTIAL not found"
    exit 1
fi

log "Starting monitor — checking every $((INTERVAL / 60)) minutes"

while true; do
    # Check if run completed (final file exists)
    if [[ -f "$FINAL" ]]; then
        TOTAL=$(python3 -c "
import json
with open('$FINAL') as f:
    d = json.load(f)
hb = d.get('harmbench', [])
filled = sum(1 for x in hb if x and x.get('response'))
print(f'{filled}/{len(hb)}')
")
        log "✅ RUN COMPLETE — $TOTAL responses"
        break
    fi

    # Check partial progress
    PROGRESS=$(python3 -c "
import json
try:
    with open('$PARTIAL') as f:
        d = json.load(f)
    hb = d.get('harmbench', [])
    total = len(hb)
    filled = sum(1 for x in hb if x and x.get('response'))
    nulls = sum(1 for x in hb if x is None)
    errors = sum(1 for x in hb if x and x.get('error'))
    pct = filled / total * 100 if total else 0
    print(f'{filled}/{total} ({pct:.1f}%) | {nulls} pending | {errors} errors')
except Exception as e:
    print(f'ERROR: {e}')
")

    # Check if generator process is alive
    GEN_PID=$(pgrep -f "harmbench_generate.*abliterix" || true)
    VLLM_PID=$(pgrep -f "vllm.*api_server" || true)
    if [[ -n "$GEN_PID" ]]; then
        log "⏳ $PROGRESS (generator PID $GEN_PID, vLLM ${VLLM_PID:-dead})"
    else
        log "⚠️  $PROGRESS (generator DEAD, vLLM ${VLLM_PID:-dead})"
        # Check if partial file was recently modified (maybe just finished)
        MOD_AGE=$(( $(date +%s) - $(stat -c %Y "$PARTIAL") ))
        if [[ $MOD_AGE -gt 600 ]]; then
            log "❌ Generator dead and no recent writes. Run may have stalled."
        fi
    fi

    sleep "$INTERVAL"
done
