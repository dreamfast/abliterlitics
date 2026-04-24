#!/usr/bin/env python3
"""Flask proxy that translates OpenAI logprobs format for ik_llama.cpp."""

from __future__ import annotations

import argparse
import logging

import requests
from flask import Flask, jsonify, request

app = Flask(__name__)
TARGET = "http://localhost:8080"
TIMEOUT = 300

log = logging.getLogger(__name__)

req_count = 0
err_count = 0


def translate_logprobs(resp):
    for choice in resp.get("choices", []):
        lp = choice.get("logprobs")
        if lp and "content" in lp:
            token_logprobs = [t["logprob"] for t in lp["content"]]
            top_logprobs = [{t["token"]: t["logprob"] for t in entry["top_logprobs"]} for entry in lp["content"]]
            choice["logprobs"] = {
                "token_logprobs": token_logprobs,
                "top_logprobs": top_logprobs,
            }
    return resp


@app.route("/v1/completions", methods=["POST"])
def completions():
    global req_count, err_count
    req_count += 1
    try:
        body = request.json
        body["echo"] = False
        r = requests.post(f"{TARGET}/v1/completions", json=body, timeout=TIMEOUT)
        if r.status_code != 200:
            err_count += 1
            log.warning("req=%d err=%d status=%d: %s", req_count, err_count, r.status_code, r.text[:200])
            return r.text, r.status_code
        if req_count % 500 == 0:
            log.info("req=%d err=%d", req_count, err_count)
        return jsonify(translate_logprobs(r.json()))
    except Exception as e:
        err_count += 1
        log.error("req=%d err=%d exception: %s", req_count, err_count, e)
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    try:
        r = requests.get(f"{TARGET}/health", timeout=5)
        return r.text, r.status_code
    except Exception:
        return "loading", 503


@app.route("/stats", methods=["GET"])
def stats():
    return jsonify({"requests": req_count, "errors": err_count, "target": TARGET})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Logprobs format proxy for ik_llama.cpp")
    parser.add_argument("--target", default="http://localhost:8080", help="Upstream llama-server URL")
    parser.add_argument("--port", type=int, default=8081, help="Proxy listen port")
    parser.add_argument("--timeout", type=int, default=300, help="Upstream request timeout (seconds)")
    args = parser.parse_args()

    TARGET = args.target
    TIMEOUT = args.timeout

    log.info("Proxy starting: target=%s, timeout=%ds, port=%d", TARGET, TIMEOUT, args.port)
    app.run(host="0.0.0.0", port=args.port)
