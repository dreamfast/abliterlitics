#!/usr/bin/env python3
"""Safe JSON loader for common.sh load_comparison().

Reads comparison.json, validates slug patterns on name and variant keys,
and prints shell-safe variable assignments using shlex.quote().
Paths are passed as sys.argv (no string interpolation into Python source).

This script is called by common.sh and its output is eval'd.
All values are shlex-quoted to prevent shell injection.
"""

import json
import re
import shlex
import sys

SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


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
    print(f"COMPARISON_NAME={shlex.quote(name)}")
    print(f"COMPARISON_DIR={shlex.quote(comp_dir)}")
    print(f"BASE_DIR={shlex.quote(comp_dir + '/' + base)}")
    print(f"VARIANT_COUNT={len(d['variants'])}")

    names = []
    for k, v in d["variants"].items():
        if not SLUG.match(k):
            print(f'error "Invalid variant key: {k}"')
            sys.exit(1)
        names.append(k)
        print(f"VARIANT_PATH_{k}={shlex.quote(comp_dir + '/' + v['path'])}")

    print(f"VARIANT_NAMES={shlex.quote(' '.join(names))}")

    settings = d.get("settings", {})
    print(f"INFERENCE_BACKEND={shlex.quote(str(settings.get('inference_backend', 'auto')))}")
    print(
        f"LM_EVAL_TASKS={shlex.quote(str(settings.get('lm_eval_tasks', 'mmlu,gsm8k,hellaswag,arc_challenge,winogrande,truthfulqa,piqa,lambada_openai')))}"
    )


if __name__ == "__main__":
    main()
