#!/usr/bin/env python3
"""Safe JSON loader for common.sh load_comparison().

Reads comparison.json, validates slug patterns on name and variant keys,
and prints shell-safe variable assignments using shlex.quote().
Paths are passed as sys.argv (no string interpolation into Python source).

This script is called by common.sh and its output is eval'd.
All values are shlex-quoted to prevent shell injection.

Exported variables:
  Core:       COMPARISON_NAME, COMPARISON_DIR, BASE_DIR, BASE_NAME, VARIANT_COUNT,
              VARIANT_NAMES, VARIANT_PATH_<slug>
  Settings:   INFERENCE_BACKEND, LM_EVAL_TASKS, LM_EVAL_MAX_MODEL_LEN,
              LM_EVAL_MAX_GEN_TOKS, HARMBENCH_MAX_TOKENS, TOKENIZER_DIR,
              GGUF_DIR, KL_NUM_PROMPTS, KL_DATASET
  Reasoning:  REASONING_PARSER, REASONING_CONFIG_JSON
   Skip flags: VARIANT_SKIP_HB_<slug>, VARIANT_SKIP_LM_<slug>,
               VARIANT_SKIP_KL_<slug>, VARIANT_SKIP_WT_<slug>
   Override:   LMEVAL_IMAGE, FORENSICS_IMAGE (per-comparison docker image override)
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys

SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# Reasoning config for architectures that use <think/> tags.
# The reasoning_parser setting in comparison.json overrides auto-detection.
_QWEN3_REASONING_PARSER = "qwen3"
_QWEN3_REASONING_CONFIG = json.dumps(
    {"reasoning_start_str": "<think}", "reasoning_end_str": "</think}"},
    separators=(",", ":"),
)


def _detect_reasoning_parser(base_model_dir: str, override: str | None) -> tuple[str, str]:
    """Auto-detect reasoning parser from base model's config.json.

    Returns (parser, config_json) where both are empty strings if no
    reasoning parser is needed. The override, if non-empty, takes precedence.
    """
    if override:
        if override == "none" or override == "off":
            return "", ""
        # Only Qwen3 needs the explicit start/end string config — Gemma4,
        # deepseek-r1, granite, etc. have parsers that handle their formats
        # internally without reasoning_config.
        if override == _QWEN3_REASONING_PARSER:
            return override, _QWEN3_REASONING_CONFIG
        return override, ""

    # Auto-detect: read config.json from the base model directory.
    config_path = os.path.join(base_model_dir, "config.json")
    if not os.path.isfile(config_path):
        return "", ""

    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return "", ""

    # Check architectures list for known reasoning model families.
    architectures = cfg.get("architectures", [])
    for arch in architectures:
        # Qwen3_5ForConditionalGeneration, Qwen3ForCausalLM, etc.
        if arch.startswith("Qwen3"):
            return _QWEN3_REASONING_PARSER, _QWEN3_REASONING_CONFIG
        # Gemma4ForConditionalGeneration, Gemma4ForCausalLM, etc.
        # Requires vLLM 0.20.1+ (abliterlitics-lmeval-nightly:1.0.0) — the
        # standard abliterlitics-lmeval:1.0.0 lacks gemma4_reasoning_parser.
        # The gemma4 parser handles <|channel>thought...<channel|> internally
        # — no reasoning_config start/end strings needed.
        if arch.startswith("Gemma4"):
            return "gemma4", ""

    return "", ""


def main() -> None:
    comp_json = sys.argv[1]
    comp_dir = sys.argv[2]

    with open(comp_json) as f:
        d = json.load(f)

    name = d["name"]
    if not SLUG.match(name):
        print(f'error "Invalid comparison name: {name}"')
        sys.exit(1)

    base = d["base"]
    base_dir = comp_dir + "/" + base
    base_name = os.path.basename(os.path.normpath(base))
    settings = d.get("settings", {})

    # Core exports
    print(f"export COMPARISON_NAME={shlex.quote(name)}")
    print(f"export COMPARISON_DIR={shlex.quote(comp_dir)}")
    print(f"export BASE_DIR={shlex.quote(base_dir)}")
    print(f"export BASE_NAME={shlex.quote(base_name)}")
    print(f"export VARIANT_COUNT={len(d['variants'])}")

    # Variant exports (paths + skip flags)
    names = []
    for k, v in d["variants"].items():
        if not SLUG.match(k):
            print(f'error "Invalid variant key: {k}"')
            sys.exit(1)
        names.append(k)
        var_name = k.replace("-", "_")
        print(f"export VARIANT_PATH_{var_name}={shlex.quote(comp_dir + '/' + v['path'])}")
        # LoRA adapter path (for vLLM dynamic LoRA loading). When set, the runner
        # uses --enable-lora instead of loading a merged model. The variant's
        # `path` should point to the BASE model (the adapter is applied on top).
        # See AGENTS.md lesson #35 — BF16 merge of small-alpha LoRAs loses signal.
        lora_path = v.get('lora_adapter_path')
        if lora_path:
            print(f"export VARIANT_LORA_ADAPTER_{var_name}={shlex.quote(comp_dir + '/' + lora_path)}")
        else:
            print(f"export VARIANT_LORA_ADAPTER_{var_name}=")
        print(f"export VARIANT_SKIP_HB_{var_name}={'1' if v.get('skip_harmbench', False) else '0'}")
        print(f"export VARIANT_SKIP_LM_{var_name}={'1' if v.get('skip_lm_eval', False) else '0'}")
        print(f"export VARIANT_SKIP_KL_{var_name}={'1' if v.get('skip_kl', False) else '0'}")
        print(f"export VARIANT_SKIP_WT_{var_name}={'1' if v.get('skip_weight', False) else '0'}")

    print(f"export VARIANT_NAMES={shlex.quote(' '.join(names))}")

    # Inference & benchmark settings
    print(f"export INFERENCE_BACKEND={shlex.quote(str(settings.get('inference_backend', 'auto')))}")
    # Default task suite is the modern Open LLM Leaderboard v2 set + GSM8K + TruthfulQA.
    # See AGENTS.md → "Benchmark Tasks" for rationale and per-task details.
    # The legacy v1 tasks (mmlu, hellaswag, arc_challenge, winogrande, piqa, lambada_openai)
    # are deprecated and intentionally excluded — see AGENTS.md lesson #9 for context
    # on why the old loglikelihood scores were unreliable.
    default_tasks = ",".join(
        [
            "leaderboard_mmlu_pro",
            "leaderboard_gpqa_diamond",
            "leaderboard_bbh",
            "leaderboard_musr",
            "leaderboard_ifeval",
            "gsm8k",
            "truthfulqa",
        ]
    )
    print(f"export LM_EVAL_TASKS={shlex.quote(str(settings.get('lm_eval_tasks', default_tasks)))}")
    print(f"export LM_EVAL_MAX_MODEL_LEN={shlex.quote(str(settings.get('lm_eval_max_model_len', 16384)))}")
    print(f"export LM_EVAL_MAX_GEN_TOKS={shlex.quote(str(settings.get('lm_eval_max_gen_toks', 4096)))}")
    # GSM8K Phase 2 generation budget (thinking models need thinking + answer room).
    # Auto-capped to max_model_len - 1024 by the runner.
    print(f"export GSM8K_GEN_TOKS={shlex.quote(str(settings.get('gsm8k_gen_toks', 7168)))}")
    # GPU memory utilization override (empty = use runner's size-based defaults).
    # Large-vocab models (e.g. Gemma, 256k vocab) need lower util so the
    # prompt_logprobs tensor (used by loglikelihood tasks) has headroom.
    print(f"export LM_EVAL_GPU_UTIL={shlex.quote(str(settings.get('lm_eval_gpu_util', '')))}")
    print(f"export HARMBENCH_MAX_TOKENS={shlex.quote(str(settings.get('harmbench_max_tokens', 8096)))}")

    # Tokenizer dir: explicit setting, or fall back to base model
    tokenizer_dir = settings.get("tokenizer_dir")
    tokenizer_dir = comp_dir + "/" + tokenizer_dir if tokenizer_dir else base_dir
    print(f"export TOKENIZER_DIR={shlex.quote(tokenizer_dir)}")

    # GGUF dir: optional, only needed for llamacpp backend
    gguf_dir = settings.get("gguf_dir", "")
    if gguf_dir:
        gguf_dir = comp_dir + "/" + gguf_dir
    print(f"export GGUF_DIR={shlex.quote(gguf_dir)}")

    # KL settings
    print(f"export KL_NUM_PROMPTS={shlex.quote(str(settings.get('kl_num_prompts', 100)))}")
    print(f"export KL_DATASET={shlex.quote(str(settings.get('kl_dataset', 'mlabonne/harmless_alpaca')))}")

    # Reasoning parser: auto-detect from base model config.json, or override
    reasoning_override = settings.get("reasoning_parser")
    parser, config_json = _detect_reasoning_parser(base_dir, reasoning_override)
    print(f"export REASONING_PARSER={shlex.quote(parser)}")
    print(f"export REASONING_CONFIG_JSON={shlex.quote(config_json)}")

    # Per-comparison Docker image overrides. Useful for models that need a
    # newer vLLM than the default abliterlitics-lmeval:1.0.0 (e.g. Gemma 4
    # thinking requires abliterlitics-lmeval-nightly:1.0.0 with the
    # gemma4_reasoning_parser — see AGENTS.md lesson #11).
    for setting_key, env_var in (
        ("lmeval_image", "LMEVAL_IMAGE"),
        ("forensics_image", "FORENSICS_IMAGE"),
    ):
        override = settings.get(setting_key)
        if override:
            print(f"export {env_var}={shlex.quote(override)}")

    # Text-encoder bench settings (optional — only if tebench block exists)
    tebench = settings.get("tebench", {})
    if tebench:
        print(f"export TEBENCH_BACKEND={shlex.quote(str(tebench.get('backend', '')))}")
        base_te = tebench.get("base_te_file", "")
        if base_te and not base_te.startswith("/"):
            base_te = comp_dir + "/" + base_te
        print(f"export TEBENCH_BASE_TE_FILE={shlex.quote(base_te)}")
        dit_file = tebench.get("dit_file", "")
        if dit_file and not dit_file.startswith("/"):
            dit_file = comp_dir + "/" + dit_file
        print(f"export TEBENCH_DIT_FILE={shlex.quote(dit_file)}")
        comfyui_path = tebench.get("comfyui_path", "/home/maxx/comfyui-docker/comfyui")
        print(f"export TEBENCH_COMFYUI_PATH={shlex.quote(comfyui_path)}")
        qf = tebench.get("quant_floor_file", "")
        if qf and not qf.startswith("/"):
            qf = comp_dir + "/" + qf
        print(f"export TEBENCH_QUANT_FLOOR_FILE={shlex.quote(qf)}")
        print(f"export TEBENCH_DEVICE={shlex.quote(str(tebench.get('device', 'cpu')))}")
        print(f"export TEBENCH_NUM_BOOTSTRAP={shlex.quote(str(tebench.get('n_boot', 1000)))}")
        print(f"export TEBENCH_PROMPT_SUITE={shlex.quote(str(tebench.get('prompt_suite', 'default')))}")
        print(f"export TEBENCH_IMG_CLIP_MODEL={shlex.quote(str(tebench.get('img_clip_model', 'ViT-L-14')))}")
        print(f"export TEBENCH_IMG_CLIP_PRETRAINED={shlex.quote(str(tebench.get('img_clip_pretrained', 'openai')))}")
        print(f"export TEBENCH_IMG_REFERENCE={shlex.quote(str(tebench.get('img_reference', 'base_int8')))}")
        floor_pair = tebench.get("img_floor_pair", ["base_int8", "base_fp8"])
        print(f"export TEBENCH_IMG_FLOOR_PAIR={shlex.quote(','.join(str(x) for x in floor_pair))}")
        te_image = tebench.get("te_image", "abliterlitics-te:1.1.0")
        print(f"export TE_IMAGE={shlex.quote(te_image)}")
    else:
        # Export empty defaults so runner scripts don't break on unset vars
        print("export TEBENCH_BACKEND=")
        print("export TE_IMAGE=")

    # Per-variant TE skip flags + te_file paths + per-variant reference override
    for k, v in d["variants"].items():
        var_name = k.replace("-", "_")
        print(f"export VARIANT_SKIP_TE_{var_name}={'1' if v.get('skip_te', False) else '0'}")
        te_f = v.get("te_file", "")
        if te_f and not te_f.startswith("/"):
            te_f = comp_dir + "/" + te_f
        print(f"export VARIANT_TE_FILE_{var_name}={shlex.quote(te_f)}")
        print(f"export VARIANT_TE_REF_{var_name}={shlex.quote(str(v.get('te_reference', '')))}")


if __name__ == "__main__":
    main()
