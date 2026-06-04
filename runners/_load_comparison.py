#!/usr/bin/env python3
"""Safe JSON loader for common.sh load_comparison().

Reads comparison.json, validates slug patterns on name and variant keys,
and prints shell-safe variable assignments using shlex.quote().
Paths are passed as sys.argv (no string interpolation into Python source).

This script is called by common.sh and its output is eval'd.
All values are shlex-quoted to prevent shell injection.

Exported variables:
  Core:       COMPARISON_NAME, COMPARISON_DIR, BASE_DIR, VARIANT_COUNT,
              VARIANT_NAMES, VARIANT_PATH_<slug>
  Settings:   INFERENCE_BACKEND, LM_EVAL_TASKS, LM_EVAL_MAX_MODEL_LEN,
              LM_EVAL_MAX_GEN_TOKS, HARMBENCH_MAX_TOKENS, TOKENIZER_DIR,
              GGUF_DIR, KL_NUM_PROMPTS, KL_DATASET
  Reasoning:  REASONING_PARSER, REASONING_CONFIG_JSON
  Skip flags: VARIANT_SKIP_HB_<slug>, VARIANT_SKIP_LM_<slug>,
              VARIANT_SKIP_KL_<slug>, VARIANT_SKIP_WT_<slug>
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
        return override, _QWEN3_REASONING_CONFIG

    # Auto-detect: read config.json from the base model directory.
    config_path = os.path.join(base_model_dir, "config.json")
    if not os.path.isfile(config_path):
        return "", ""

    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return "", ""

    # Check architectures list for Qwen3-family models.
    architectures = cfg.get("architectures", [])
    for arch in architectures:
        # Qwen3_5ForConditionalGeneration, Qwen3ForCausalLM, etc.
        if arch.startswith("Qwen3"):
            return _QWEN3_REASONING_PARSER, _QWEN3_REASONING_CONFIG

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
    settings = d.get("settings", {})

    # Core exports
    print(f"COMPARISON_NAME={shlex.quote(name)}")
    print(f"COMPARISON_DIR={shlex.quote(comp_dir)}")
    print(f"BASE_DIR={shlex.quote(base_dir)}")
    print(f"VARIANT_COUNT={len(d['variants'])}")

    # Variant exports (paths + skip flags)
    names = []
    for k, v in d["variants"].items():
        if not SLUG.match(k):
            print(f'error "Invalid variant key: {k}"')
            sys.exit(1)
        names.append(k)
        var_name = k.replace("-", "_")
        print(f"VARIANT_PATH_{var_name}={shlex.quote(comp_dir + '/' + v['path'])}")
        print(f"VARIANT_SKIP_HB_{var_name}={'1' if v.get('skip_harmbench', False) else '0'}")
        print(f"VARIANT_SKIP_LM_{var_name}={'1' if v.get('skip_lm_eval', False) else '0'}")
        print(f"VARIANT_SKIP_KL_{var_name}={'1' if v.get('skip_kl', False) else '0'}")
        print(f"VARIANT_SKIP_WT_{var_name}={'1' if v.get('skip_weight', False) else '0'}")

    print(f"VARIANT_NAMES={shlex.quote(' '.join(names))}")

    # Inference & benchmark settings
    print(f"INFERENCE_BACKEND={shlex.quote(str(settings.get('inference_backend', 'auto')))}")
    default_tasks = "mmlu,gsm8k,hellaswag,arc_challenge,winogrande,truthfulqa,piqa,lambada_openai"
    print(f"LM_EVAL_TASKS={shlex.quote(str(settings.get('lm_eval_tasks', default_tasks)))}")
    print(f"LM_EVAL_MAX_MODEL_LEN={shlex.quote(str(settings.get('lm_eval_max_model_len', 8192)))}")
    print(f"LM_EVAL_MAX_GEN_TOKS={shlex.quote(str(settings.get('lm_eval_max_gen_toks', 2048)))}")
    print(f"HARMBENCH_MAX_TOKENS={shlex.quote(str(settings.get('harmbench_max_tokens', 8096)))}")

    # Tokenizer dir: explicit setting, or fall back to base model
    tokenizer_dir = settings.get("tokenizer_dir")
    tokenizer_dir = comp_dir + "/" + tokenizer_dir if tokenizer_dir else base_dir
    print(f"TOKENIZER_DIR={shlex.quote(tokenizer_dir)}")

    # GGUF dir: optional, only needed for llamacpp backend
    gguf_dir = settings.get("gguf_dir", "")
    if gguf_dir:
        gguf_dir = comp_dir + "/" + gguf_dir
    print(f"GGUF_DIR={shlex.quote(gguf_dir)}")

    # KL settings
    print(f"KL_NUM_PROMPTS={shlex.quote(str(settings.get('kl_num_prompts', 100)))}")
    print(f"KL_DATASET={shlex.quote(str(settings.get('kl_dataset', 'mlabonne/harmless_alpaca')))}")

    # Reasoning parser: auto-detect from base model config.json, or override
    reasoning_override = settings.get("reasoning_parser")
    parser, config_json = _detect_reasoning_parser(base_dir, reasoning_override)
    print(f"REASONING_PARSER={shlex.quote(parser)}")
    print(f"REASONING_CONFIG_JSON={shlex.quote(config_json)}")


if __name__ == "__main__":
    main()
