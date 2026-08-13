#!/usr/bin/env python3
"""
DiT-step divergence analysis — does the Z-Image diffusion model RESPOND to
the text-encoder conditioning perturbation?

Runs paired flow-matching Euler trajectories from identical noise with base
vs variant conditioning, measuring per-step velocity + latent divergence,
with a different-seed noise control as the "totally different image" anchor.

Uses the activation cache from run_analysis.py (consumed_hidden streams) —
run the main TE analysis first.

Usage:
    python3 run_dit_divergence.py \
        --base-cache-key-file /models/.../qwen_3_4b.safetensors \
        --variant qwen3-4b-heretic_int8.safetensors \
        --dit /models/diffusion_models/z_image_turbo_bf16.safetensors \
        --cache-dir /te_cache --results-dir /results --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Z-Image DiT-step divergence analysis")
    p.add_argument("--base", required=True, help="BASE TE safetensors (cache key)")
    p.add_argument("--variant", required=True, help="VARIANT TE safetensors (cache key)")
    p.add_argument("--variant-name", default="variant", help="Variant display name")
    p.add_argument("--dit", required=True, help="Z-Image DiT checkpoint")
    p.add_argument("--cache-dir", default="/te_cache", help="Activation cache (from run_analysis)")
    p.add_argument("--results-dir", default="/results", help="Output directory")
    p.add_argument("--device", default="cuda", help="cpu or cuda")
    p.add_argument("--n-steps", type=int, default=20, help="Euler steps per trajectory")
    p.add_argument("--n-prompts", type=int, default=6, help="Prompts to run (one per category by default)")
    p.add_argument("--latent-hw", type=int, default=128, help="Latent spatial size (128 = 1024px)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_consumed_from_cache(
    cache_dir: Path, te_file: str, device: str, ctx: str, prompts: list[str], categories: list[str]
) -> list[torch.Tensor]:
    """Per-prompt consumed_hidden [L, 2560] for the sampled prompts.

    The cache is keyed by the FULL suite hash (from run_analysis), so load
    the full artifact and slice to the sampled prompts by prompt text.
    """
    from src.text_encoder.cache import WILDCARD_DTYPE, cache_key, load_artifact
    from src.text_encoder.suite import load_suite, suite_hash

    full_prompts, _full_cats = load_suite("/prompts/te_suite.yaml")
    p_hash = suite_hash(full_prompts, _full_cats)
    key = cache_key(te_file, p_hash, WILDCARD_DTYPE, device, "raw_hidden", "orig", context=ctx)
    art = load_artifact(cache_dir, key)
    if art is None:
        raise FileNotFoundError(f"No cached raw_hidden artifact for {Path(te_file).name}. Run run_analysis.py first.")
    stream = art.streams["consumed_hidden"]
    acts, mask = stream.activations, stream.mask
    full_idx = {p: i for i, p in enumerate(art.prompts)}
    out: list[torch.Tensor] = []
    for prompt_text in prompts:
        row = full_idx.get(prompt_text)
        if row is None:
            raise KeyError(f"Prompt not in cached artifact: {prompt_text[:60]!r}")
        m = mask[row] if mask is not None else None
        if m is not None:
            real = int(m.sum().item())
            out.append(acts[row, :real].clone())
        else:
            out.append(acts[row].clone())
    return out


def main() -> int:
    args = parse_args()
    from src.text_encoder.backends import comfyui_loader
    from src.text_encoder.backends.z_image_dit import load_zimage_dit, run_divergence
    from src.text_encoder.suite import load_suite

    comfyui_path = "/comfyui"
    comfyui_loader.import_comfy(comfyui_path, cpu=(args.device == "cpu"))

    prompts, categories = load_suite("/prompts/te_suite.yaml")
    # One prompt per category (balanced sample) unless n_prompts covers all
    if args.n_prompts < len(prompts):
        seen: set[str] = set()
        idxs: list[int] = []
        for i, c in enumerate(categories):
            if c not in seen:
                idxs.append(i)
                seen.add(c)
        prompts = [prompts[i] for i in idxs[: args.n_prompts]]
        categories = [categories[i] for i in idxs[: args.n_prompts]]
    log.info("Running divergence on %d prompts (one per category)", len(prompts))

    cache_dir = Path(args.cache_dir)
    ctx_scope = f"zimage:{Path(args.dit).name}"

    base_ctxs = _load_consumed_from_cache(cache_dir, args.base, args.device, ctx_scope, prompts, categories)
    var_ctxs = _load_consumed_from_cache(cache_dir, args.variant, args.device, ctx_scope, prompts, categories)

    log.info("Loading Z-Image DiT: %s", Path(args.dit).name)
    dit = load_zimage_dit(args.dit, device=args.device)

    log.info("Running paired trajectories: %d steps, %d prompts, seed %d", args.n_steps, len(prompts), args.seed)
    result = run_divergence(
        dit,
        base_ctxs,
        var_ctxs,
        device=args.device,
        n_steps=args.n_steps,
        seed=args.seed,
        latent_hw=args.latent_hw,
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "te_results_version": 1,
        "analysis": "dit_step_divergence",
        "backend": "zimage",
        "base": args.base,
        "variant": args.variant,
        "variant_name": args.variant_name,
        "dit": args.dit,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompts": prompts,
        "categories": categories,
        **result,
    }
    # Summary numbers
    finals = [pp["final_latent_div"] for pp in result["per_prompt"]]
    vels = [pp["final_velocity_div"] for pp in result["per_prompt"]]
    output["summary"] = {
        "mean_final_latent_div": sum(finals) / len(finals),
        "mean_final_velocity_div": sum(vels) / len(vels),
        "noise_control_final": result["noise_control_final"],
        "div_control_ratio": (sum(finals) / len(finals)) / max(result["noise_control_final"], 1e-12),
    }

    out_path = results_dir / f"te_dit_divergence_{args.variant_name}.json"
    out_path.write_text(json.dumps(output, indent=2))
    log.info("Wrote: %s", out_path)

    s = output["summary"]
    log.info(
        "=== DiT-step divergence: final latent div %.5f (%.1f%% of noise-control %.3f) ===",
        s["mean_final_latent_div"],
        s["div_control_ratio"] * 100,
        s["noise_control_final"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
