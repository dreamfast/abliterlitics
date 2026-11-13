"""Suite-wide word decode probe: do drifted words still decode to themselves?

Extends the earlier single-battery refusal-word probes (12 key words, one
battery; retired 2026-08-17) to EVERY word group in the A/B stress suite,
for every variant with cached conditioning.

Method: per prompt, group consumed-conditioning tokens into words with the
same token_groups used by the gallery. Build a base vocabulary from the mean
base vector of every unique word across the suite. For each variant word
occurrence, decode: nearest base vocabulary word by cosine. Report per
variant the self-decode rate, and per unique word how consistently it decodes
to itself across the prompts it appears in.

Output: tebench_word_parse_suite.json
    {"variants": {name: {"summary": {...}, "per_word": {word: {...}}}}}

Usage (te container):
    python3 -m src.text_encoder.probe_word_parse_suite \
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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.text_encoder import TE_RESULTS_VERSION

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Suite-wide word decode probe (all variants)")
    p.add_argument("--base", required=True)
    p.add_argument("--variants", required=True, help="Comma-separated variant TE files")
    p.add_argument("--variant-names", default="", help="Comma-separated display names")
    p.add_argument("--dit", required=True)
    p.add_argument("--suite", default="/app/prompts/ab_stress_prompts.yaml")
    p.add_argument("--cache-dir", default="/te_cache")
    p.add_argument("--results-dir", default="/results")
    p.add_argument("--device", default="cuda", help="Cache-key device (zimage cache key only)")
    p.add_argument(
        "--backend",
        choices=["zimage", "krea2", "h3"],
        default="zimage",
        help="zimage: cache-driven consumed_hidden. krea2/h3: direct CPU encode of the DiT-input stream.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from src.text_encoder.attention_metrics import decode_bpe_tokens, token_groups
    from src.text_encoder.backends import comfyui_loader
    from src.text_encoder.run_attention import _suite_labels, _token_strings
    from src.text_encoder.run_dit_divergence import _load_consumed_from_cache
    from src.text_encoder.suite import load_suite

    comfyui_loader.import_comfy("/comfyui", cpu=True)
    prompts, categories = load_suite(args.suite)
    labels = _suite_labels(args.suite)

    token_strings: dict[str, list[str]] = {}
    if args.backend == "zimage":
        token_strings = _token_strings(True, prompts)
        cache_dir = Path(args.cache_dir)
        ctx_scope = f"zimage:{Path(args.dit).name}"

        def _ctxs(te_file: str) -> list[torch.Tensor]:
            return _load_consumed_from_cache(
                cache_dir, te_file, args.device, ctx_scope, prompts, categories, suite_path=args.suite
            )

    elif args.backend == "krea2":
        from src.text_encoder.run_krea2_attention import encode_krea2_contexts

        def _ctxs_and_toks(te_file: str) -> tuple[list[torch.Tensor], list[list[str]]]:
            ctxs, toks = encode_krea2_contexts(te_file, prompts, categories, args.dit, "/comfyui", device="cpu")
            return ctxs, toks

        def _ctxs(te_file: str) -> list[torch.Tensor]:
            ctxs, toks = _ctxs_and_toks(te_file)
            for p, t in zip(prompts, toks, strict=True):
                token_strings.setdefault(p, t)
            return ctxs

    else:  # h3 — CPU direct encode of refined_context, raw un-templated tokens
        from src.text_encoder.backends.h3 import H3Backend

        def _ctxs(te_file: str) -> list[torch.Tensor]:
            from src.text_encoder.artifact import Stage

            backend = H3Backend(comfyui_path="/comfyui")
            backend.load(te_file, device="cpu", dit_file=args.dit)
            arts = backend.encode(prompts, categories, stages=[Stage.CONNECTOR_OUT])
            conn = next(a for a in arts if a.stage.value == "connector_out")
            stream = conn.streams["refined_context"]
            ids_all = backend.tokenize(prompts)["input_ids"]
            inner = backend.tokenizer.qwen3vl_32b.tokenizer
            out: list[torch.Tensor] = []
            for i, p in enumerate(prompts):
                acts = stream.activations[i]
                mask = stream.mask[i] if stream.mask is not None else None
                n = int(mask.sum()) if mask is not None else acts.shape[0]
                toks = decode_bpe_tokens(inner.convert_ids_to_tokens(ids_all[i]))
                if n != len(toks):
                    log.warning("%s: prompt %d mask %d vs %d tokens — using min", Path(te_file).name, i, n, len(toks))
                token_strings.setdefault(p, toks[: min(n, len(toks))])
                out.append(acts[: min(n, len(toks))].float())
            backend.close()
            return out

    base_ctxs = _ctxs(args.base)

    # Word groups per prompt on base: word -> list of occurrence mean vectors
    base_word_vecs: dict[str, list[torch.Tensor]] = defaultdict(list)
    per_prompt_groups: list[list[tuple[str, list[int]]]] = []
    for pi, prompt in enumerate(prompts):
        toks = token_strings.get(prompt, [])
        ctx = base_ctxs[pi]
        n = min(len(toks), ctx.shape[0])
        groups: list[tuple[str, list[int]]] = []
        for label, kind, idxs in token_groups(toks[:n]):
            if kind != "word" or not idxs:
                continue
            groups.append((label, idxs))
            base_word_vecs[label].append(ctx[idxs].float().mean(dim=0))
        per_prompt_groups.append(groups)

    # Vocabulary: mean base vector per unique word, stacked for batched cosine
    vocab_words = sorted(base_word_vecs.keys())
    vocab = torch.stack([torch.stack(base_word_vecs[w]).mean(dim=0) for w in vocab_words])
    vocab = vocab / vocab.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    log.info("Vocabulary: %d unique words from %d prompts", len(vocab_words), len(prompts))

    variant_files = [f for f in args.variants.split(",") if f.strip()]
    names = (
        [n.strip() for n in args.variant_names.split(",")]
        if args.variant_names
        else [Path(f).stem for f in variant_files]
    )
    if len(names) != len(variant_files):
        raise SystemExit("--variant-names length mismatch with --variants")

    # Method floor from the unmodified encoder itself: decode every base
    # occurrence against the suite-mean vocabulary. Deterministic encode, so
    # this equals the bit-identical-container floor where one exists (the
    # container's vectors are bitwise the base's). Note both floors share the
    # self-inclusion property: each occurrence contributes to its own word's
    # mean, so the floor is an optimistic self-consistency baseline. Variant
    # deltas read against it are therefore conservative.
    floor_self: list[list[str]] = []
    word_idx = {w: i for i, w in enumerate(vocab_words)}
    for pi, prompt in enumerate(prompts):
        for label, idxs in per_prompt_groups[pi]:
            v = base_ctxs[pi][idxs].float().mean(dim=0)
            v = v / v.norm().clamp_min(1e-8)
            sims = vocab @ v
            nearest = vocab_words[int(torch.argmax(sims))]
            if nearest != label:
                floor_self.append([labels.get(prompt, prompt[:24]), label, nearest])
    out_variants: dict[str, dict[str, Any]] = {}
    for vfile, vname in zip(variant_files, names, strict=True):
        var_ctxs = _ctxs(vfile)
        total = 0
        self_hits = 0
        per_word: dict[str, dict[str, Any]] = {}
        fails: list[list[str]] = []
        for pi, _prompt in enumerate(prompts):
            ctx = var_ctxs[pi]
            plabel = labels.get(prompts[pi], prompts[pi][:24])
            n_ctx = ctx.shape[0]
            for label, idxs in per_prompt_groups[pi]:
                idxs = [i for i in idxs if i < n_ctx]
                if not idxs:
                    continue
                v = ctx[idxs].float().mean(dim=0)
                v = v / v.norm().clamp_min(1e-8)
                sims = vocab @ v
                nearest = vocab_words[int(torch.argmax(sims))]
                total += 1
                if nearest == label:
                    self_hits += 1
                else:
                    fails.append([plabel, label, nearest])
                rec = per_word.setdefault(
                    label,
                    {"occurrences": 0, "self_decodes": 0, "nearest_other": {}, "min_self_cos": None},
                )
                rec["occurrences"] += 1
                if nearest == label:
                    rec["self_decodes"] += 1
                else:
                    rec["nearest_other"][nearest] = rec["nearest_other"].get(nearest, 0) + 1
                self_cos = float(sims[word_idx[label]])
                if rec["min_self_cos"] is None or self_cos < rec["min_self_cos"]:
                    rec["min_self_cos"] = round(self_cos, 4)
        changed = {w: r for w, r in per_word.items() if r["self_decodes"] < r["occurrences"]}
        out_variants[vname] = {
            "summary": {
                "word_occurrences": total,
                "self_decode_rate": round(self_hits / max(total, 1), 5),
                "unique_words": len(per_word),
                "words_with_any_nonsel_decode": len(changed),
            },
            "per_word": per_word,
            "nonself_occurrences": fails,
        }
        log.info(
            "%s: %d occurrences, self-decode %.2f%%, %d words with any non-self decode",
            vname,
            total,
            100 * self_hits / max(total, 1),
            len(changed),
        )

    doc = {
        "te_results_version": TE_RESULTS_VERSION,
        "analysis": "word_parse_suite",
        "base": args.base,
        "dit": args.dit,
        "suite": args.suite,
        "vocabulary_size": len(vocab_words),
        "floor_self": floor_self,
        "backend": args.backend,
        "variants": out_variants,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_path = Path(args.results_dir) / "tebench_word_parse_suite.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2))
    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
