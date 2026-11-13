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

from src.text_encoder._cli import add_backend, add_cache_dir, add_device, add_results_dir

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
    add_cache_dir(p, default="/te_cache", help_text="Activation cache (from run_analysis)")
    add_results_dir(p, default="/results", help_text="Output directory")
    add_device(p, default="cuda", help_text="cpu or cuda (DiT + trajectories)")
    p.add_argument(
        "--cache-device",
        default="",
        help="Device key the forensics cache was written with (default: cpu — the "
        "krea2/z-image analysis runners cache on cpu; set to 'cuda' only for "
        "cuda-cached artifacts)",
    )
    p.add_argument(
        "--n-steps", type=int, default=None, help="Euler steps (None = backend default: 20 zimage / 6 krea2)"
    )
    p.add_argument("--n-prompts", type=int, default=6, help="Prompts to run (one per category by default)")
    p.add_argument(
        "--latent-hw",
        type=int,
        default=None,
        help="Latent spatial size (None = backend default: 128 zimage / 64 krea2)",
    )
    p.add_argument("--seed", type=int, default=42)
    add_backend(
        p,
        default="zimage",
        choices=("zimage", "krea2"),
        help_text="DiT + conditioning-interface family (stream name, loader, forward)",
    )
    p.add_argument("--suite", default="/prompts/te_suite.yaml", help="Prompt suite (must match run_analysis cache)")
    return p.parse_args()


def _load_consumed_from_cache(
    cache_dir: Path,
    te_file: str,
    device: str,
    ctx: str,
    prompts: list[str],
    categories: list[str],
    suite_path: str = "/prompts/te_suite.yaml",
    stream: str = "consumed_hidden",
) -> list[torch.Tensor]:
    """Per-prompt conditioning tensors for the sampled prompts.

    z-image: consumed_hidden [L, 2560]. krea2: tap_stack [L, 30720]
    (the flattened 12-tap interface the DiT consumes).

    The cache is keyed by the FULL suite hash (from run_analysis), so load
    the full artifact and slice to the sampled prompts by prompt text.
    ``suite_path`` must match the suite the artifacts were cached with
    (run_analysis --prompts); the A/B stress suite is a valid choice — its
    {prompts: [...]} layout is accepted by load_suite.
    """
    from src.text_encoder.cache import WILDCARD_DTYPE, cache_key, load_artifact
    from src.text_encoder.suite import load_suite, suite_hash

    full_prompts, _full_cats = load_suite(suite_path)
    p_hash = suite_hash(full_prompts, _full_cats)
    key = cache_key(te_file, p_hash, WILDCARD_DTYPE, device, "raw_hidden", "orig", context=ctx)
    art = load_artifact(cache_dir, key)
    if art is None:
        raise FileNotFoundError(f"No cached raw_hidden artifact for {Path(te_file).name}. Run run_analysis.py first.")
    stream_data = art.streams[stream]
    acts, mask = stream_data.activations, stream_data.mask
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
    from src.text_encoder.suite import load_suite

    comfyui_path = "/comfyui"
    comfyui_loader.import_comfy(comfyui_path, cpu=(args.device == "cpu"))

    if args.backend == "krea2":
        from src.text_encoder.backends.krea2_dit import load_krea2_dit
        from src.text_encoder.backends.krea2_dit import run_krea2_divergence as run_divergence

        dit_loader = load_krea2_dit
        cond_stream = "tap_stack"
        default_steps, default_hw = 6, 64
    else:
        from src.text_encoder.backends.z_image_dit import load_zimage_dit, run_divergence

        dit_loader = load_zimage_dit
        cond_stream = "consumed_hidden"
        default_steps, default_hw = 20, 128
    if args.n_steps is None:
        args.n_steps = default_steps
    if args.latent_hw is None:
        args.latent_hw = default_hw
    cache_dev = args.cache_device or "cpu"

    prompts, categories = load_suite(args.suite)
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
    ctx_scope = f"{args.backend}:{Path(args.dit).name}"

    base_ctxs = _load_consumed_from_cache(
        cache_dir, args.base, cache_dev, ctx_scope, prompts, categories, suite_path=args.suite, stream=cond_stream
    )
    var_ctxs = _load_consumed_from_cache(
        cache_dir, args.variant, cache_dev, ctx_scope, prompts, categories, suite_path=args.suite, stream=cond_stream
    )

    log.info("Loading %s DiT: %s", args.backend, Path(args.dit).name)
    dit = dit_loader(args.dit, device=args.device)

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
        "backend": args.backend,
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
