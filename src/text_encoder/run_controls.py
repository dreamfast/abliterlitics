"""Control battery runner — counterfactual conditioning through the DiT harness.

Phase 4 (T3.2). Pushes synthesized conditioning (perturb.py) through the same
paired-trajectory harness as run_dit_divergence, producing the anchor ladder
and dose-response controls:

  noise_<dose>     Gaussian at matched MSE (>=5 doses, 0.25x..2x max-variant)
  rank1_<dose>     fixed random direction (structured-but-meaningless control)
  taboo_<dose>     taboo-direction injection (norm-matched — NOTE: "1x" here
                   matches the heretic delta's mean per-token NORM, not the
                   noise family's MSE; cross-family "1x" labels are NOT the
                   same perturbation magnitude (deepseek-R7))
  heretic_a<alpha> alpha-scaled REAL heretic delta (0.5/1/2/10)
  order_shuffle    Spingarn syntax-destruction
  embed_shuffle    Spingarn meaning-destruction (derangement)
  null             zero-information anchor

Output: tebench_controls.json (single file, all conditions + summaries) in
the tebench_ namespace (Phase-4 plan rule — importer glob safety).

Usage (mirrors run_dit_divergence):
    python3 -m src.text_encoder.run_controls \
        --base /models/qwen_3_4b.safetensors \
        --heretic /models/qwen3-4b-heretic.safetensors \
        --dit /models/z_image_turbo_bf16.safetensors \
        --cache-dir /te_cache --results-dir /results --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.text_encoder import TE_RESULTS_VERSION

log = logging.getLogger(__name__)

DEFAULT_DOSE_STEPS = (0.25, 0.5, 1.0, 1.5, 2.0)
DEFAULT_ALPHAS = (0.5, 1.0, 2.0, 10.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Control battery: counterfactual conditioning divergence")
    p.add_argument("--base", required=True, help="BASE TE safetensors (cache key)")
    p.add_argument("--heretic", required=True, help="Heretic TE safetensors (delta source, cache key)")
    p.add_argument(
        "--dose-variants",
        default="",
        help="Comma-separated variant TE FILES whose cached consumed_hidden defines the max dose "
        "(default: base + heretic only)",
    )
    p.add_argument("--dit", required=True)
    p.add_argument("--cache-dir", default="/te_cache")
    p.add_argument("--results-dir", default="/results")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-steps", type=int, default=20)
    p.add_argument("--n-prompts", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--latent-hw", type=int, default=128)
    p.add_argument("--dose-steps", default=",".join(str(x) for x in DEFAULT_DOSE_STEPS))
    p.add_argument("--alphas", default=",".join(str(x) for x in DEFAULT_ALPHAS))
    p.add_argument(
        "--conditions",
        default="noise,rank1,taboo,heretic_alpha,order_shuffle,embed_shuffle,null",
        help="Comma-separated condition families to run",
    )
    p.add_argument(
        "--merge",
        action="store_true",
        help="Merge into an existing tebench_controls.json instead of replacing it "
        "(required when adding bracket conditions post-hoc; refuses on metadata mismatch)",
    )
    return p.parse_args()


def _summarize(result: dict[str, Any]) -> dict[str, Any]:
    finals = [pp["final_latent_div"] for pp in result["per_prompt"]]
    ctrl = result["noise_control_final"]
    return {
        "mean_final_latent_div": sum(finals) / len(finals),
        "noise_control_final": ctrl,
        "div_control_ratio": (sum(finals) / len(finals)) / max(ctrl, 1e-12),
        "per_prompt_finals": finals,
    }


def main() -> int:
    args = parse_args()
    from src.text_encoder.backends import comfyui_loader
    from src.text_encoder.backends.z_image_dit import load_zimage_dit, run_divergence
    from src.text_encoder.directional import taboo_direction
    from src.text_encoder.perturb import (
        delta_scale,
        direction_injection,
        embedding_assignment_shuffle,
        gaussian_delta,
        null_conditioning,
        per_prompt_delta_norm,
        rank1_random_delta,
        token_order_shuffle,
    )
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
    log.info("Control battery on %d prompts (one per category)", len(prompts))

    cache_dir = Path(args.cache_dir)
    ctx_scope = f"zimage:{Path(args.dit).name}"
    base_ctxs = _load_consumed_from_cache(cache_dir, args.base, args.device, ctx_scope, prompts, categories)
    her_ctxs = _load_consumed_from_cache(cache_dir, args.heretic, args.device, ctx_scope, prompts, categories)

    # Dose reference: max per-prompt MSE across heretic + optional extra variants
    dose_files = [args.heretic] + [f for f in args.dose_variants.split(",") if f.strip()]
    max_dose = torch.zeros(len(prompts))
    for f in dose_files:
        try:
            vctx = _load_consumed_from_cache(cache_dir, f, args.device, ctx_scope, prompts, categories)
        except FileNotFoundError:
            log.warning("dose variant not cached, skipping: %s", Path(f).name)
            continue
        for i, (b, v) in enumerate(zip(base_ctxs, vctx, strict=True)):
            mse = float(((b.float() - v.float()) ** 2).mean())
            max_dose[i] = max(float(max_dose[i]), mse)
    log.info(
        "Max dose (MSE at DiT input): mean=%.4e per-prompt=%s",
        float(max_dose.mean()),
        [f"{m:.2e}" for m in max_dose.tolist()],
    )

    # Taboo direction from the cached layer-34 pooled stream (full 60-prompt suite)
    from src.text_encoder.cache import WILDCARD_DTYPE, cache_key, load_artifact
    from src.text_encoder.suite import suite_hash

    full_prompts, full_cats = load_suite("/prompts/te_suite.yaml")
    p_hash = suite_hash(full_prompts, full_cats)
    key = cache_key(args.base, p_hash, WILDCARD_DTYPE, args.device, "raw_hidden", "orig", context=ctx_scope)
    base_art = load_artifact(cache_dir, key)
    if base_art is None:
        raise FileNotFoundError("No cached base artifact for taboo direction")
    layer34 = base_art.streams["layer_34"].activations  # [60, 1, 2560] pooled
    if layer34.dim() == 3:
        layer34 = layer34[:, 0, :]
    d_taboo = taboo_direction(layer34, list(base_art.categories))
    log.info("Taboo direction computed from 60-prompt layer-34 pooled acts")

    log.info("Loading Z-Image DiT: %s", Path(args.dit).name)
    dit = load_zimage_dit(args.dit, device=args.device)

    dose_steps = [float(x) for x in args.dose_steps.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    families = {f.strip() for f in args.conditions.split(",")}

    def _mk_noise(scale: float) -> Callable[[list[torch.Tensor]], list[torch.Tensor]]:
        def make(ctxs: list[torch.Tensor]) -> list[torch.Tensor]:
            out = []
            for i, b in enumerate(ctxs):
                delta = gaussian_delta(b.unsqueeze(0), float(max_dose[i]) * scale, seed=1000 + i)
                out.append((b.float() + delta).squeeze(0))
            return out

        return make

    def _mk_rank1(scale: float) -> Callable[[list[torch.Tensor]], list[torch.Tensor]]:
        def make(ctxs: list[torch.Tensor]) -> list[torch.Tensor]:
            out = []
            for i, b in enumerate(ctxs):
                delta = rank1_random_delta(
                    b.unsqueeze(0), float(max_dose[i]) * scale, orthogonal_to=d_taboo, seed=2000 + i
                )
                out.append((b.float() + delta).squeeze(0))
            return out

        return make

    heretic_norms = [
        float(per_prompt_delta_norm(h.float() - b.float())) for b, h in zip(base_ctxs, her_ctxs, strict=True)
    ]

    def _mk_taboo(scale: float) -> Callable[[list[torch.Tensor]], list[torch.Tensor]]:
        def make(ctxs: list[torch.Tensor]) -> list[torch.Tensor]:
            out = []
            for i, b in enumerate(ctxs):
                delta = direction_injection(b.unsqueeze(0), d_taboo, heretic_norms[i] * scale)
                out.append((b.float() + delta).squeeze(0))
            return out

        return make

    def _mk_alpha(a: float) -> Callable[[list[torch.Tensor]], list[torch.Tensor]]:
        def make(ctxs: list[torch.Tensor]) -> list[torch.Tensor]:
            out = []
            for b, h in zip(ctxs, her_ctxs, strict=True):
                delta = delta_scale(b.unsqueeze(0), h.unsqueeze(0), a)
                out.append((b.float() + delta).squeeze(0))
            return out

        return make

    def _mk_shuffle(
        fn: Callable[..., torch.Tensor], seed_offset: int
    ) -> Callable[[list[torch.Tensor]], list[torch.Tensor]]:
        def make(ctxs: list[torch.Tensor]) -> list[torch.Tensor]:
            return [fn(c.unsqueeze(0), seed=seed_offset + i).squeeze(0) for i, c in enumerate(ctxs)]

        return make

    def _mk_null() -> Callable[[list[torch.Tensor]], list[torch.Tensor]]:
        def make(ctxs: list[torch.Tensor]) -> list[torch.Tensor]:
            return [null_conditioning(c) for c in ctxs]

        return make

    conditions: dict[str, Any] = {}
    if "noise" in families:
        for s in dose_steps:
            conditions[f"noise_{s:g}x"] = _mk_noise(s)
    if "rank1" in families:
        for s in dose_steps:
            conditions[f"rank1_{s:g}x"] = _mk_rank1(s)
    if "taboo" in families:
        for s in dose_steps:
            conditions[f"taboo_{s:g}x"] = _mk_taboo(s)
    if "heretic_alpha" in families:
        for a in alphas:
            conditions[f"heretic_a{a:g}"] = _mk_alpha(a)
    if "order_shuffle" in families:
        conditions["order_shuffle"] = _mk_shuffle(token_order_shuffle, 3000)
    if "embed_shuffle" in families:
        conditions["embed_shuffle"] = _mk_shuffle(embedding_assignment_shuffle, 4000)
    if "null" in families:
        conditions["null"] = _mk_null()

    results: dict[str, Any] = {
        "te_results_version": TE_RESULTS_VERSION,
        "analysis": "control_battery",
        "backend": "zimage",
        "base": args.base,
        "heretic": args.heretic,
        "dit": args.dit,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompts": prompts,
        "categories": categories,
        "dose_steps": dose_steps,
        "alphas": alphas,
        "max_dose_mse": [float(m) for m in max_dose.tolist()],
        "heretic_delta_norm": heretic_norms,
        "conditions": {},
    }

    # --merge: union with an existing run's conditions (dose bracket top-up).
    out_path = Path(args.results_dir) / "tebench_controls.json"
    if args.merge:
        if not out_path.exists():
            log.warning("--merge passed but %s does not exist; writing fresh", out_path)
        else:
            prior = json.loads(out_path.read_text())
            same_setup = (
                prior.get("base") == results["base"]
                and prior.get("heretic") == results["heretic"]
                and prior.get("dit") == results["dit"]
                and len(prior.get("max_dose_mse", [])) == len(results["max_dose_mse"])
                and all(
                    abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(a)))
                    for a, b in zip(prior["max_dose_mse"], results["max_dose_mse"], strict=True)
                )
            )
            if not same_setup:
                raise SystemExit(
                    "--merge refused: existing tebench_controls.json was produced with a "
                    "different base/heretic/DIT or dose reference — delete it or drop --merge"
                )
            overlap = sorted(set(prior.get("conditions", {})) & set(conditions))
            if overlap:
                log.warning("--merge will overwrite existing conditions: %s", ", ".join(overlap))
            results["conditions"] = {**prior.get("conditions", {})}
            results["dose_steps"] = sorted({*(float(x) for x in prior.get("dose_steps", [])), *dose_steps})
            results["alphas"] = sorted({*(float(x) for x in prior.get("alphas", [])), *alphas})
            log.info(
                "Merging into existing battery (%d prior conditions, %d new)",
                len(prior.get("conditions", {})),
                len(conditions),
            )

    for name, make in conditions.items():
        log.info("=== Condition %s ===", name)
        var_ctxs = make(base_ctxs)
        result = run_divergence(
            dit,
            base_ctxs,
            var_ctxs,
            device=args.device,
            n_steps=args.n_steps,
            seed=args.seed,
            latent_hw=args.latent_hw,
        )
        summary = _summarize(result)
        results["conditions"][name] = {
            "summary": summary,
            "per_prompt": result["per_prompt"],
            "n_steps": args.n_steps,
            "seed": args.seed,
        }
        log.info(
            "%s: final div %.5f (%.2f%% of noise control)",
            name,
            summary["mean_final_latent_div"],
            summary["div_control_ratio"] * 100,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    log.info("Wrote %s (%d conditions)", out_path, len(results["conditions"]))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
