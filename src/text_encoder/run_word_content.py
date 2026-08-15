"""Per-word conditioning-CONTENT readout — the second channel of the gallery.

run_attention measures WHERE the DiT looks (attention mass per word). This
script measures WHAT arrives behind each word: per-token cosine similarity of
the consumed conditioning (the tensor that actually enters cross-attention)
between the base encoder and each variant, grouped into words with the SAME
grouping (token_groups) as the attention readout — so the two channels render
aligned.

Together they explain image-level detail loss: a late detail word can keep
its attention while the CONTENT behind it is scrambled (int4-family
quantization) — the DiT still reads the word but receives noise, and weakly
attended details die first.

Output: tebench_word_content.json — {"variants": {name: {label: [[word, cos, kind], ...]}}}
(tebench_ namespace — importer glob safety). Rows are grouped by token_groups —
the SAME grouping the attention readout renders — so the two channels are
aligned by construction (build_gallery_word_data verifies labels/kinds).

Usage (in the te container, repo at /app):
    python3 -m src.text_encoder.run_word_content \
        --base /models/.../qwen_3_4b.safetensors \
        --variants /models/.../qwen3-4b-heretic.safetensors,... \
        --variant-names heretic,... \
        --dit /models/diffusion_models/z_image_turbo_bf16.safetensors \
        --suite /app/prompts/ab_stress_prompts.yaml \
        --cache-dir /te_cache --results-dir /results --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.text_encoder import TE_RESULTS_VERSION

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-word conditioning content cosine (gallery channel 2)")
    p.add_argument("--base", required=True)
    p.add_argument("--variants", required=True, help="Comma-separated variant TE files")
    p.add_argument("--variant-names", default="", help="Comma-separated display names")
    p.add_argument("--dit", required=True, help="DIT file (cache scope key)")
    p.add_argument("--suite", default="/prompts/te_suite.yaml")
    p.add_argument("--cache-dir", default="/te_cache")
    p.add_argument("--results-dir", default="/results")
    p.add_argument("--device", default="cuda", help="Cache-key device (artifacts load to CPU for compare)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from src.text_encoder.attention_metrics import token_groups
    from src.text_encoder.backends import comfyui_loader
    from src.text_encoder.run_attention import _suite_labels, _token_strings
    from src.text_encoder.run_dit_divergence import _load_consumed_from_cache
    from src.text_encoder.suite import load_suite

    comfyui_loader.import_comfy("/comfyui", cpu=True)
    prompts, categories = load_suite(args.suite)
    labels = _suite_labels(args.suite)
    token_strings = _token_strings(True, prompts)

    cache_dir = Path(args.cache_dir)
    ctx_scope = f"zimage:{Path(args.dit).name}"
    base_ctxs = _load_consumed_from_cache(
        cache_dir, args.base, args.device, ctx_scope, prompts, categories, suite_path=args.suite
    )

    variant_files = [f for f in args.variants.split(",") if f.strip()]
    names = (
        [n.strip() for n in args.variant_names.split(",")]
        if args.variant_names
        else [Path(f).stem for f in variant_files]
    )
    if len(names) != len(variant_files):
        raise SystemExit("--variant-names length mismatch with --variants")

    out_variants: dict[str, dict[str, list[list[Any]]]] = {}
    for vfile, vname in zip(variant_files, names, strict=True):
        var_ctxs = _load_consumed_from_cache(
            cache_dir, vfile, args.device, ctx_scope, prompts, categories, suite_path=args.suite
        )
        per_label: dict[str, list[list[Any]]] = {}
        for pi, (b, v) in enumerate(zip(base_ctxs, var_ctxs, strict=True)):
            toks = token_strings.get(prompts[pi], [])
            lab = labels.get(prompts[pi], prompts[pi][:40])
            if not toks:
                log.warning("%s: no token strings for %s — skipping", vname, lab)
                continue
            n = min(len(toks), b.shape[0], v.shape[0])
            bb = b[:n].float()
            vv = v[:n].float()
            cos = torch.nn.functional.cosine_similarity(bb, vv, dim=-1)  # [n]
            rows: list[list[Any]] = []
            for label, kind, idxs in token_groups(toks[:n]):
                word_cos = float(cos[idxs].mean().item()) if idxs else 1.0
                rows.append([label, round(word_cos, 4), kind])
            per_label[lab] = rows
        out_variants[vname] = per_label
        log.info("%s: content cosine for %d prompts", vname, len(per_label))

    doc = {
        "te_results_version": TE_RESULTS_VERSION,
        "analysis": "word_content",
        "base": args.base,
        "dit": args.dit,
        "suite": args.suite,
        "variants": out_variants,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_path = Path(args.results_dir) / "tebench_word_content.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2))
    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
