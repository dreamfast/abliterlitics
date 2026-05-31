#!/bin/bash
set -euo pipefail

# Patch lm-eval bug: line 545 references `outputs` before assignment
API_FILE="/usr/local/lib/python3.12/dist-packages/lm_eval/models/api_models.py"
python3 -c "
with open('$API_FILE') as f:
    src = f.read()
old = 'eval_logger.error(f\"Exception:{repr(e)}, {outputs}, retrying.\")'
new = 'eval_logger.error(f\"Exception:{repr(e)}, retrying.\")'
if old in src:
    src = src.replace(old, new)
    with open('$API_FILE', 'w') as f:
        f.write(src)
    print('Patched lm-eval bug')
else:
    print('Bug already patched or not found')
"

MODEL_NAME=$1

echo "=== Starting vLLM for $MODEL_NAME ==="
python3 -m vllm.entrypoints.openai.api_server \
  --model /model --dtype bfloat16 --quantization bitsandbytes \
  --load-format bitsandbytes --trust-remote-code --max-model-len 8192 \
  --gpu-memory-utilization 0.90 --enforce-eager \
  --reasoning-parser qwen3 \
  --reasoning-config '{"reasoning_start_str": "<think}", "reasoning_end_str": "</think"}' \
  --port 8080 --host 127.0.0.1 > /logs/${MODEL_NAME}_gsm8k_v4_vllm.log 2>&1 &
SERVER_PID=$!

for i in $(seq 1 180); do
  if curl -s http://127.0.0.1:8080/v1/models | grep -q model; then
    echo "vLLM ready after ${i}s"; break
  fi; sleep 5
done
if ! curl -s http://127.0.0.1:8080/v1/models | grep -q model; then
  echo "VLLM FAILED"; tail -5 /logs/${MODEL_NAME}_gsm8k_v4_vllm.log; exit 1
fi

echo "=== GSM8K 7168 tok for $MODEL_NAME (timeout=900s, concurrent=2) ==="
python3 -m lm_eval --model local-completions \
  --model_args "base_url=http://127.0.0.1:8080/v1/completions,model=/model,tokenizer=/tokenizer,tokenizer_backend=huggingface,num_concurrent=2,max_length=8192,timeout=900" \
  --tasks "gsm8k" --batch_size 4 --gen_kwargs max_gen_toks=7168 \
  --output_path /results/ --log_samples 2>&1 | tee /logs/${MODEL_NAME}_gsm8k_v4.log

echo "=== $MODEL_NAME GSM8K DONE ==="
kill $SERVER_PID 2>/dev/null || true
sleep 5
