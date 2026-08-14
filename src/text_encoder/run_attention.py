"""Attention-readout runner — paired capture of what the DiT reads per TE variant.

Phase 4 T2.2 execution. For each representative variant: run the SAME Euler
trajectory as the divergence harness twice per prompt (base side + variant
side), capturing the per-layer text-readout distributions via
``AttentionCapture`` (ComfyUI's optimized_attention_override). Then diff the
readouts with attention_metrics (JSD primary, top-k overlap, entropy).

Output: tebench_attention_<variant>.json per variant — per-step-per-layer
summaries aggregated to (a) per-layer mean JSD across steps, (b) per-step
mean JSD across layers, (c) mean top-5 tokens per layer for base/variant.

Usage:
    python3 -m src.text_encoder.run_attention \
        --base /models/qwen_3_4b.safetensors \
        --variants "/models/qwen3-4b-heretic.safetensors,/models/qwen_3_4b_int4.safetensors" \
        --dit /models/z_image_turbo_bf16.safetensors \
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
from src.text_encoder.attention_capture import AttentionCapture
from src.text_encoder.attention_metrics import summarize_readout

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attention readout: what does the DiT read per TE")
    p.add_argument("--base", required=True)
    p.add_argument("--variants", required=True, help="Comma-separated variant TE files")
    p.add_argument("--variant-names", default="", help="Comma-separated display names (default: filenames)")
    p.add_argument("--dit", required=True)
    p.add_argument("--cache-dir", default="/te_cache")
    p.add_argument("--results-dir", default="/results")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-steps", type=int, default=20)
    p.add_argument("--n-prompts", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--latent-hw", type=int, default=128)
    p.add_argument("--layer-sample", type=int, default=4, help="Compute summaries on every Nth layer (all steps)")
    return p.parse_args()


def _token_strings(comfyui_loaded: bool, prompts: list[str]) -> dict[str, list[str]]:
    """Token string per conditioning position for each prompt (defensive).

    Returns {} when the tokenizer is unavailable — callers fall back to
    position indices. Positions beyond the real token count are labeled
    "<pad>" (pad-to-32 padding in the Z-Image conditioning path).
    """
    out: dict[str, list[str]] = {}
    if not comfyui_loaded:
        return out
    try:
        import comfy.text_encoders.z_image as zte

        tok = zte.ZImageTokenizer()
        # Attribute chain (probed empirically): ZImageTokenizer.__dict__['qwen3_4b']
        # = Qwen3Tokenizer(SDTokenizer); its .tokenizer = SLOW transformers
        # Qwen2Tokenizer — no id_to_token, use convert_ids_to_tokens (batch).
        inner = tok.__dict__.get("qwen3_4b")
        hf_tok = getattr(inner, "tokenizer", None) if inner is not None else None
        dec = getattr(hf_tok, "convert_ids_to_tokens", None) if hf_tok is not None else None
        for p in prompts:
            # ZImageTokenizer API (mirrors backends/z_image.py tokenize()):
            # dict keyed by clip field -> per-chunk list of (id, weight) tuples
            res = tok.tokenize_with_weights(p)
            ids = [t[0] for t in res["qwen3_4b"][0]]
            if callable(dec):
                raws: list[str] = [str(r) for r in dec(ids)]
                toks = [r.replace("Ġ", " ") for r in raws]
            else:
                toks = [f"tok#{i}" for i in ids]
            out[p] = toks
    except Exception as exc:  # noqa: BLE001 - display-only, must never kill the run
        log.warning("tokenizer decode unavailable (%s) — word view falls back to indices", exc)
    return out


def main() -> int:
    args = parse_args()
    from src.text_encoder.backends import comfyui_loader
    from src.text_encoder.backends.z_image_dit import _sigma_schedule, load_zimage_dit
    from src.text_encoder.run_dit_divergence import _load_consumed_from_cache
    from src.text_encoder.suite import load_suite

    comfyui_loader.import_comfy("/comfyui", cpu=(args.device == "cpu"))
    prompts, categories = load_suite("/prompts/te_suite.yaml")
    if args.n_prompts < len(prompts):
        seen: set[str] = set()
        idxs: list[int] = []
        for i, c in enumerate(categories):
            if c not in seen:
                idxs.append(i)
                seen.add(c)
        prompts = [prompts[i] for i in idxs[: args.n_prompts]]
        categories = [categories[i] for i in idxs[: args.n_prompts]]

    cache_dir = Path(args.cache_dir)
    ctx_scope = f"zimage:{Path(args.dit).name}"
    base_ctxs = _load_consumed_from_cache(cache_dir, args.base, args.device, ctx_scope, prompts, categories)

    log.info("Loading Z-Image DiT: %s", Path(args.dit).name)
    dit = load_zimage_dit(args.dit, device=args.device)

    variant_files = [f for f in args.variants.split(",") if f.strip()]
    names = (
        [n.strip() for n in args.variant_names.split(",")]
        if args.variant_names
        else [Path(f).stem for f in variant_files]
    )
    img_len = (args.latent_hw // 2) ** 2  # patch 2
    capture = AttentionCapture(img_len=img_len, device=args.device)

    sigmas = _sigma_schedule(args.n_steps).to(args.device)
    g = torch.Generator(device="cpu").manual_seed(args.seed)

    def _forward_capture(x: torch.Tensor, ctx: torch.Tensor, sigma: float, step: int, side: str) -> torch.Tensor:
        capture.reset(step, side)
        timesteps = torch.tensor([sigma], device=args.device, dtype=x.dtype)
        out: torch.Tensor = dit(
            x,
            timesteps,
            ctx.unsqueeze(0),
            torch.tensor([ctx.shape[0]], device=args.device),
            transformer_options=capture.transformer_options,
        )
        recs = capture.take_records()
        _forward_capture.last_records = recs  # type: ignore[attr-defined]
        return out

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for vfile, vname in zip(variant_files, names, strict=True):
        var_ctxs = _load_consumed_from_cache(cache_dir, vfile, args.device, ctx_scope, prompts, categories)
        per_prompt_metrics: list[dict[str, Any]] = []
        token_strings = _token_strings(True, prompts)
        for pi, (ctx_b, ctx_v) in enumerate(zip(base_ctxs, var_ctxs, strict=True)):
            recs_b_all: list[dict[str, Any]] = []
            recs_v_all: list[dict[str, Any]] = []
            x0 = torch.randn(1, 16, args.latent_hw, args.latent_hw, generator=g).to(args.device).to(torch.bfloat16)
            xb, xv = x0.clone(), x0.clone()
            step_layer: dict[tuple[int, int], dict[str, float]] = {}
            for i in range(args.n_steps):
                s = float(sigmas[i])
                s_next = float(sigmas[i + 1])
                cb = ctx_b.to(device=args.device, dtype=torch.bfloat16)
                cv = ctx_v.to(device=args.device, dtype=torch.bfloat16)
                vb = _forward_capture(xb, cb, s, i, "base")
                recs_b = list(_forward_capture.last_records)  # type: ignore[attr-defined]
                vv = _forward_capture(xv, cv, s, i, "variant")
                recs_v = list(_forward_capture.last_records)  # type: ignore[attr-defined]
                dt = s_next - s
                xb = (xb.float() + vb.float() * dt).to(torch.bfloat16)
                xv = (xv.float() + vv.float() * dt).to(torch.bfloat16)
                recs_b_all.extend(recs_b)
                recs_v_all.extend(recs_v)
                if i % args.layer_sample and i != args.n_steps - 1:
                    continue  # sampled layers except always-last step
                for rb, rv in zip(recs_b, recs_v, strict=False):
                    if rb["text_len"] != rv["text_len"]:
                        continue
                    m = summarize_readout(rb, rv)
                    step_layer[(i, rb["call"])] = m
            # Word-readout persistence: mean text-readout per side across
            # captured (step, layer) records — decoded to token strings by
            # the report layer for the "what words does the DiT read" diff.
            import torch as _t

            base_mean = _t.stack([r["text_readout"] for r in recs_b_all]).mean(dim=0) if recs_b_all else None
            var_mean = _t.stack([r["text_readout"] for r in recs_v_all]).mean(dim=0) if recs_v_all else None
            per_prompt_metrics.append(
                {
                    "prompt": prompts[pi],
                    "category": categories[pi],
                    "step_layer": {f"{s}_{ly}": m for (s, ly), m in sorted(step_layer.items())},
                    "base_mean_readout": [round(float(x), 6) for x in base_mean] if base_mean is not None else [],
                    "variant_mean_readout": [round(float(x), 6) for x in var_mean] if var_mean is not None else [],
                    "tokens": token_strings.get(prompts[pi], []),
                }
            )
            log.info("%s prompt %d/%d: %d (step,layer) pairs captured", vname, pi + 1, len(prompts), len(step_layer))

        # Aggregate: per-layer mean over (prompt, step); per-step mean over (prompt, layer)
        layer_acc: dict[int, list[float]] = {}
        step_acc: dict[int, list[float]] = {}
        for pp in per_prompt_metrics:
            for key, m in pp["step_layer"].items():
                s_str, l_str = key.split("_")
                layer_acc.setdefault(int(l_str), []).append(m["jsd"])
                step_acc.setdefault(int(s_str), []).append(m["jsd"])
        doc = {
            "te_results_version": TE_RESULTS_VERSION,
            "analysis": "attention_readout",
            "backend": "zimage",
            "base": args.base,
            "variant": vfile,
            "variant_name": vname,
            "dit": args.dit,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_steps": args.n_steps,
            "seed": args.seed,
            "layer_sample": args.layer_sample,
            "jsd_per_layer": {str(ly): sum(v) / len(v) for ly, v in sorted(layer_acc.items())},
            "jsd_per_step": {str(s): sum(v) / len(v) for s, v in sorted(step_acc.items())},
            "jsd_mean_all": (sum(sum(v) for v in layer_acc.values()) / max(sum(len(v) for v in layer_acc.values()), 1)),
            "per_prompt": [
                {k: v for k, v in pp.items() if k != "step_layer"} | {"n_pairs": len(pp["step_layer"])}
                for pp in per_prompt_metrics
            ],
            "has_word_readout": bool(token_strings),
        }
        out_path = results_dir / f"tebench_attention_{vname}.json"
        out_path.write_text(json.dumps(doc, indent=2))
        log.info("%s: mean JSD %.5f -> %s", vname, doc["jsd_mean_all"], out_path)

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
