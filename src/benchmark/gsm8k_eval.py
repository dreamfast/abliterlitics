#!/usr/bin/env python3
"""GSM8K evaluation using vLLM chat completions with thinking enabled.

Same approach as harmbench_generate.py — direct vLLM OpenAI chat completions
with enable_thinking=True, reasoning token stripping, and retry logic.
"""

import argparse
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GSM8K_FEW_SHOT = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6 trees planted.
#### 6

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.
#### 5

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they have 74 - 35 = 39.
#### 39

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.
#### 8

Question: Shawn has five toys. For Christmas, he got two toys from his mom and two toys from his dad. How many toys does he have now?
Answer: Shawn started with 5 toys. He got 2 from mom and 2 from dad. So he has 5 + 2 + 2 = 9 toys.
#### 9

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Answer: There were originally 9 computers. 5 more were added each day for 4 days. So 5 * 4 = 20 computers were added. 9 + 20 = 29.
#### 29

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer: Michael started with 58 golf balls. He lost 23 on Tuesday, and lost 2 more on Wednesday. So he has 58 - 23 - 2 = 33.
#### 33

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer: Olivia had 23 dollars. 5 bagels cost 5 * 3 = 15 dollars. So she has 23 - 15 = 8 dollars left.
#### 8"""


def extract_answer(text):
    m = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if m:
        ans = m.group(1).replace(',', '')
        try:
            return float(ans)
        except ValueError:
            return None
    nums = re.findall(r'-?\d+\.?\d*', text)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            return None
    return None


def normalize_num(n):
    if n is None:
        return None
    if n == int(n):
        return int(n)
    return round(n, 4)


def generate_one(session, base_url, prompt, max_tokens, request_id):
    for attempt in range(5):
        try:
            r = session.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "thinking_token_budget": max_tokens - 2048,
                    "chat_template_kwargs": {"enable_thinking": True},
                },
                timeout=600,
            )
            if r.status_code == 200:
                choice = r.json()["choices"][0]
                msg = choice["message"]
                content = msg.get("content", "") or ""
                reasoning = msg.get("reasoning", msg.get("reasoning_content", "")) or ""
                finish_reason = choice.get("finish_reason", "")
                usage = r.json().get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)
                return request_id, content, reasoning, None, finish_reason, completion_tokens
            else:
                if attempt < 4:
                    time.sleep(10 * (attempt + 1))
                    continue
                return request_id, "", "", f"HTTP {r.status_code}: {r.text[:200]}", "", 0
        except Exception as e:
            if attempt < 4:
                time.sleep(10 * (attempt + 1))
                continue
            return request_id, "", "", str(e), "", 0
    return request_id, "", "", "Max retries exceeded", "", 0


def main():
    parser = argparse.ArgumentParser(description="GSM8K eval via vLLM chat completions")
    parser.add_argument("--base-url", required=True, default="http://127.0.0.1:8080")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--concurrent", type=int, default=4)
    parser.add_argument("--model-name", default="unknown")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples (0=all)")
    args = parser.parse_args()

    gsm8k_path = Path("/results/gsm8k_test.jsonl")
    if not gsm8k_path.exists():
        log.error("GSM8K dataset not found at %s", gsm8k_path)
        return

    samples = []
    with open(gsm8k_path) as f:
        for line in f:
            samples.append(json.loads(line))

    if args.limit > 0:
        samples = samples[:args.limit]

    log.info(f"Loaded {len(samples)} GSM8K samples")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    partial_file = out_path.with_suffix(".partial.json")
    start_idx = 0
    results = [None] * len(samples)

    if partial_file.exists():
        try:
            with open(partial_file) as f:
                saved = json.load(f)
            for item in saved["samples"]:
                idx = item["index"]
                results[idx] = item
            start_idx = next(i for i, r in enumerate(results) if r is None)
            log.info(f"Resuming from sample {start_idx}/{len(samples)}")
        except Exception:
            log.warning("Could not load partial results, starting fresh")
            start_idx = 0

    session = requests.Session()
    total = len(samples)
    done = sum(1 for r in results if r is not None)
    t0 = time.time()

    pending = [(i, samples[i]) for i in range(len(samples)) if results[i] is None]

    with ThreadPoolExecutor(max_workers=args.concurrent) as pool:
        futures = {}
        batch = pending
        for idx, sample in batch:
            prompt = GSM8K_FEW_SHOT + "\n\nQuestion: " + sample["question"] + "\nAnswer:"
            fut = pool.submit(generate_one, session, args.base_url, prompt, args.max_tokens, idx)
            futures[fut] = idx

        for fut in as_completed(futures):
            idx = futures[fut]
            request_id, content, reasoning, error, finish_reason, completion_tokens = fut.result()
            sample = samples[idx]
            target_answer = extract_answer(sample["answer"])
            pred_answer = extract_answer(content)

            strict_pred = extract_answer(content)
            flex_pred = extract_last_number(content)
            strict_match = (normalize_num(target_answer) == normalize_num(strict_pred))
            flex_match = (normalize_num(target_answer) == normalize_num(flex_pred))

            results[idx] = {
                "index": idx,
                "question": sample["question"],
                "target": sample["answer"],
                "target_answer": normalize_num(target_answer),
                "response": content,
                "reasoning": reasoning,
                "strict_pred": normalize_num(strict_pred),
                "flex_pred": normalize_num(flex_pred),
                "strict_match": strict_match,
                "flexible_match": flex_match,
                "error": error,
                "finish_reason": finish_reason,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": len(reasoning.split()) if reasoning else 0,
            }

            done += 1
            if done % 10 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed * 60 if elapsed > 0 else 0
                strict_c = sum(1 for r in results if r and r["strict_match"])
                flex_c = sum(1 for r in results if r and r["flexible_match"])
                empty = sum(1 for r in results if r and not r["response"].strip())
                log.info(f"  {done}/{total} done, strict={strict_c} ({strict_c/done*100:.1f}%), flex={flex_c} ({flex_c/done*100:.1f}%), {empty} empty, {rate:.1f}/min")

            if done % 50 == 0:
                with open(partial_file, "w") as f:
                    json.dump({"samples": [r for r in results if r is not None], "model": args.model_name}, f)

    correct = sum(1 for r in results if r["exact_match"])
    empty = sum(1 for r in results if not r["response"].strip())
    truncated = sum(1 for r in results if r["finish_reason"] == "length")
    avg_tokens = sum(r["completion_tokens"] for r in results) / len(results) if results else 0
    avg_think = sum(r["reasoning_tokens"] for r in results) / len(results) if results else 0

    summary = {
        "model": args.model_name,
        "task": "gsm8k",
        "total": total,
        "correct": correct,
        "exact_match": correct / total,
        "empty_responses": empty,
        "truncated": truncated,
        "avg_completion_tokens": round(avg_tokens, 1),
        "avg_reasoning_tokens_words": round(avg_think, 1),
        "max_tokens_limit": args.max_tokens,
        "concurrent": args.concurrent,
        "samples": results,
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    if partial_file.exists():
        partial_file.unlink()

    log.info(f"=== DONE === {correct}/{total} = {correct/total*100:.1f}% exact_match")
    log.info(f"  Empty: {empty}, Truncated: {truncated}, Avg tokens: {avg_tokens:.0f}, Avg think: {avg_think:.0f}w")


if __name__ == "__main__":
    main()
