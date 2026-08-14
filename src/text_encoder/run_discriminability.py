"""Inter-prompt discriminability — does the variant conditioning stay informative?

Phase 4 T3.3. For each TE: run one trajectory per prompt from the SAME
initial noise (per prompt index), then measure the pairwise spread of final
latents ACROSS prompts. If a variant's conditioning is degraded, the DiT's
outputs collapse toward each other (semantic loss invisible to same-prompt
metrics). Reported as mean pairwise final-latent divergence + ratio vs base.

Output: tebench_discriminability_<variant>.json per variant.
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
    p = argparse.ArgumentParser(description="Inter-prompt discriminability per TE")
    p.add_argument("--base", required=True)
    p.add_argument("--variants", required=True, help="Comma-separated variant TE files")
    p.add_argument("--variant-names", default="")
    p.add_argument("--dit", required=True)
    p.add_argument("--cache-dir", default="/te_cache")
    p.add_argument("--results-dir", default="/results")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-steps", type=int, default=20)
    p.add_argument("--n-prompts", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--latent-hw", type=int, default=128)
    return p.parse_args()


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
    log.info("Loading Z-Image DiT: %s", Path(args.dit).name)
    dit = load_zimage_dit(args.dit, device=args.device)
    sigmas = _sigma_schedule(args.n_steps).to(args.device)

    def _trajectory(ctx: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        x = x0.clone()
        for i in range(args.n_steps):
            s = float(sigmas[i])
            s_next = float(sigmas[i + 1])
            c = ctx.to(device=args.device, dtype=torch.bfloat16)
            timesteps = torch.tensor([s], device=args.device, dtype=x.dtype)
            v: torch.Tensor = dit(x, timesteps, c.unsqueeze(0), torch.tensor([c.shape[0]], device=args.device))
            x = (x.float() + v.float() * (s_next - s)).to(torch.bfloat16)
        return x.float()

    def _spread(ctxs: list[torch.Tensor]) -> float:
        """Mean pairwise final-latent relative divergence across prompts."""
        finals = []
        for i, ctx in enumerate(ctxs):
            # SAME noise per prompt index across TEs (seeded by index):
            # differences in finals are due to conditioning only
            gi = torch.Generator(device="cpu").manual_seed(args.seed + i)
            x0 = torch.randn(1, 16, args.latent_hw, args.latent_hw, generator=gi).to(args.device).to(torch.bfloat16)
            finals.append(_trajectory(ctx, x0).flatten())
        n = len(finals)
        divs = []
        for i in range(n):
            for j in range(i + 1, n):
                num = (finals[i] - finals[j]).norm().item()
                den = max(finals[i].norm().item(), 1e-12)
                divs.append(num / den)
        spread: float = sum(divs) / len(divs)
        return spread

    base_ctxs = _load_consumed_from_cache(cache_dir, args.base, args.device, ctx_scope, prompts, categories)
    base_spread = _spread(base_ctxs)
    log.info("Base inter-prompt spread: %.5f", base_spread)

    variant_files = [f for f in args.variants.split(",") if f.strip()]
    names = (
        [n.strip() for n in args.variant_names.split(",")]
        if args.variant_names
        else [Path(f).stem for f in variant_files]
    )
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for vfile, vname in zip(variant_files, names, strict=True):
        var_ctxs = _load_consumed_from_cache(cache_dir, vfile, args.device, ctx_scope, prompts, categories)
        spread = _spread(var_ctxs)
        doc: dict[str, Any] = {
            "te_results_version": TE_RESULTS_VERSION,
            "analysis": "discriminability",
            "backend": "zimage",
            "base": args.base,
            "variant": vfile,
            "variant_name": vname,
            "dit": args.dit,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_steps": args.n_steps,
            "seed": args.seed,
            "base_spread": base_spread,
            "variant_spread": spread,
            "spread_ratio": spread / max(base_spread, 1e-12),
        }
        out = results_dir / f"tebench_discriminability_{vname}.json"
        out.write_text(json.dumps(doc, indent=2))
        log.info("%s: spread %.5f (ratio %.3f) -> %s", vname, spread, doc["spread_ratio"], out)

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
