#!/usr/bin/env bash
# Compact status for the qwen38-27b lm-eval followup run (ektome + davidau).
set -u
cd /home/maxx/projects/abliterlitics
R=comparisons/qwen38-27b/results

echo "=== $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
echo "--- watchdog (last 2 log lines):"
tr '\r' '\n' < "$R/lm_eval_watchdog.log" 2>/dev/null | grep -E '\[watchdog [0-9]+\] ' | tail -2 | cut -c1-190
echo "--- queue processes:"
pgrep -af 'abliterlitics\.sh lm-eval|run_lm_eval\.sh' 2>/dev/null | grep -v grep | awk '{print "  " $1 " " $2 " " $3 " " $4}' | head -3
[[ -z "$(pgrep -f 'run_lm_eval\.sh' 2>/dev/null)" ]] && echo "  (none — queue not running)"
echo "--- newest phase log + last progress token:"
python3 - "$R/lm_eval_logs" <<'PY'
import re, sys, glob, os
d = sys.argv[1]
logs = glob.glob(os.path.join(d, '*_phase*.log'))
if not logs:
    print("  (no phase logs)")
else:
    f = max(logs, key=os.path.getmtime)
    import time
    age = int(time.time() - os.path.getmtime(f))
    data = open(f, 'rb').read()[-8000:].decode('utf-8', 'ignore')
    tok = ""
    for pat in [r'Requesting API:\s+\d+%\|[^|]*\|\s+(\d+)/(\d+)\s+\[([^\]]+)\]',
                r'Checking cached requests:\s+(\d+)/(\d+)',
                r'Running generate_until requests[\s\S]*?(\d+)%\|[^|]*\|\s+(\d+)/(\d+)\s+\[([^\]]+)\]']:
        m = re.findall(pat, data)
        if m:
            tok = m[-1]; break
    print(f"  {os.path.basename(f)} (age {age}s): {tok}")
PY
echo "--- server crash scan (last 200 lines of newest server log):"
python3 - "$R/lm_eval_logs" <<'PY'
import glob, os, sys
d = sys.argv[1]
logs = glob.glob(os.path.join(d, '*_vllm_server.log'))
if logs:
    f = max(logs, key=os.path.getmtime)
    data = open(f, 'rb').read()[-200000:].decode('utf-8', 'ignore')
    hits = [l for l in data.splitlines() if 'EngineDeadError' in l or 'Segfault encountered' in l or 'CUDA out of memory' in l]
    print(f"  {os.path.basename(f)}: {'CLEAN' if not hits else hits[-3:]}")
PY
echo "--- canonical results present (names only):"
ls "$R/lm_eval" 2>/dev/null | grep -E '^lm_eval_.*\.json$' | tr '\n' ' '; echo
echo "--- 3num reports:"
ls "$R/lm_eval" 2>/dev/null | grep -E '^gsm8k_3num' | tr '\n' ' '; echo
echo "--- gsm8k phase2 samples files (newest 3):"
ls -t "$R"/__model/samples_gsm8k_*.jsonl "$R"/*/samples_gsm8k_*.jsonl 2>/dev/null | head -3
echo "--- GPU0:"
docker run --rm --gpus 'device=0' --entrypoint nvidia-smi abliterlitics-lmeval-gemma4unified:1.0.0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi unavailable)"
