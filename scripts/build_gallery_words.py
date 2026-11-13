"""Rebuild the website gallery WORDS payload from result artifacts.

Merges the two word channels into the compact gallery format:
- attention readouts (tebench_attention_*.json) → background heat
- conditioning content cosines (tebench_word_content.json) → orange underline

Arms with attention get full rows [word, mb, mv, kind, cos]. Arms with only
content get [word, null, null, kind, cos] — the site shortcode renders them
underline-only. Arms with neither are simply absent, as before.

Usage (te container, or any env with the repo importable):
    python3 scripts/build_gallery_words.py \
        --results comparisons/te-zimage/results/te-zimage/text_encoder \
        --suite prompts/ab_stress_prompts.yaml \
        --out /tmp/words.json
Then splice into ab-data.js (see scripts/Splice-words note in the website).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True, help="te-zimage text_encoder results dir")
    p.add_argument("--suite", required=True, help="suite YAML used by both batteries")
    p.add_argument("--out", required=True, help="output JSON path")
    args = p.parse_args()

    from src.text_encoder.phase4_report import build_gallery_word_data, load_attention, load_word_content
    from src.text_encoder.run_attention import _suite_labels
    from src.text_encoder.suite import load_suite

    results = Path(args.results)
    attn = load_attention(results)
    content = load_word_content(results)

    # label -> prompt text (suite-level, shared by all arms)
    prompts, _cats = load_suite(args.suite)
    labels = _suite_labels(args.suite)
    label_to_prompt = {labels[p]: p for p in prompts if p in labels}

    words = build_gallery_word_data(attn, content)

    n_attn_arms = len(attn)
    added = 0
    for vname, per_label in content.items():
        if vname in attn:
            continue
        for lab_, rows in per_label.items():
            entry = words.setdefault(lab_, {})
            if vname in entry:
                continue
            entry[vname] = {
                "prompt": label_to_prompt.get(lab_, ""),
                "rows": [[str(w), None, None, str(kind), round(float(c), 4)] for w, c, kind in rows],
            }
            added += 1

    doc = {
        "arms_with_attention": sorted(attn.keys()),
        "arms_content_only": sorted(v for v in content if v not in attn),
        "labels": sorted(words.keys()),
        "words": words,
    }
    out = Path(args.out)
    out.write_text(json.dumps(doc, separators=(",", ":")))
    size = out.stat().st_size
    log.info(
        "WORDS payload: %d arms (%d attention, %d content-only), %d labels, %.0f KB",
        len(attn) + len(set(content) - set(attn)),
        n_attn_arms,
        len(set(content) - set(attn)),
        len(words),
        size / 1024,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
