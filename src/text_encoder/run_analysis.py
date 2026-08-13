#!/usr/bin/env python3
"""
Text-encoder conditioning fidelity analysis — main entry point.

Loads base + variant text encoders (ComfyUI-format safetensors), captures
conditioning at multiple pipeline stages via ComfyUI's own model classes,
computes CKA/MSE/damage metrics, and writes the report.

Usage:
    python3 run_analysis.py \
        --base /models/gemma-3-12b-it_int8.safetensors \
        --variant /models/gemma-3-12b-it-heretic-v2_int8.safetensors \
        --dit /models/ltx-2.3-22b-dev-fp8.safetensors \
        --backend ltx2 --comfyui /comfyui \
        --prompts /prompts/te_suite.yaml \
        --results-dir /results --cache-dir /te_cache
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Text-encoder conditioning fidelity analysis")
    p.add_argument("--base", required=True, help="Path to BASE text encoder safetensors (ComfyUI format)")
    p.add_argument("--variant", required=True, help="Path to VARIANT text encoder safetensors")
    p.add_argument("--dit", required=True, help="Path to diffusion model safetensors (projection source)")
    p.add_argument(
        "--reference",
        default="",
        help="Optional: path to a REFERENCE TE safetensors to compare the variant against "
        "instead of --base (e.g. base_int8 for matched-quant abliteration isolation). "
        "Defaults to --base. The global base is still encoded for the noise floor.",
    )
    p.add_argument(
        "--reference-name",
        default="",
        help="Optional: variant NAME of the reference (e.g. base_int8) — stored in the "
        "report env so downstream tooling matches by stable name, not filename.",
    )
    p.add_argument(
        "--quant-floor-file",
        default="",
        help="Optional: path to a REAL quantized base safetensors used as the noise floor "
        "instead of fake-quantizing --base (literal damage-in-quant-units). "
        "Falls back to fake-quant int8 when not provided.",
    )
    p.add_argument("--backend", default="zimage", choices=["zimage", "ltx2"], help="Backend name")
    p.add_argument("--comfyui", default="/comfyui", help="Path to ComfyUI repo (contains comfy/)")
    p.add_argument("--variant-name", default="variant", help="Variant display name")
    p.add_argument("--prompts", default="/prompts/te_suite.yaml", help="Path to prompt suite YAML")
    p.add_argument("--results-dir", default="/results", help="Output directory for results")
    p.add_argument("--cache-dir", default="/te_cache", help="Activation cache directory")
    p.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    p.add_argument("--limit", type=int, default=0, help="Limit prompt count (0 = all)")
    p.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples for CIs")
    return p.parse_args()


def _encode_and_cache(
    backend_name: str,
    comfyui_path: str,
    te_file: str,
    dit_file: str,
    prompts: list[str],
    categories: list[str],
    device: str,
    cache_dir: Path,
    transform: str,
    fake_quant: str | None = None,
    cache_context: str = "",
) -> list[Any]:
    """Encode one model with all stages, using the disk cache.

    Args:
        transform: Cache-key transform label ("orig", "fakequant_int8", ...).
        fake_quant: If set, apply fake-quantization to the loaded backbone
                    before encoding (noise-floor reference builds).
    """
    from src.text_encoder.artifact import ConditioningArtifact
    from src.text_encoder.backends import get_backend
    from src.text_encoder.cache import WILDCARD_DTYPE, cache_key, load_artifact, save_artifact
    from src.text_encoder.suite import suite_hash

    p_hash = suite_hash(prompts, categories)
    # Derive produced stages from the backend's declared streams (reviewer:
    # hardcoding connector_out defeats the cache for ltx2 which doesn't make it)
    stage_list = sorted({s.stage.value for s in get_backend(backend_name).streams})

    # Check cache for all stages first
    cached_artifacts: list[ConditioningArtifact] = []
    for stage in stage_list:
        key = cache_key(str(te_file), p_hash, WILDCARD_DTYPE, device, stage, transform, context=cache_context)
        art = load_artifact(cache_dir, key)
        if art is None:
            cached_artifacts = []
            break
        cached_artifacts.append(art)
    if cached_artifacts:
        log.info("Cache hit for all stages: %s (transform=%s)", Path(te_file).name, transform)
        return cached_artifacts

    log.info("Cache miss — encoding %s (transform=%s)", Path(te_file).name, transform)
    backend = get_backend(backend_name)
    backend.load(te_file, device=device, dit_file=dit_file)
    try:
        if fake_quant:
            backend.apply_fake_quant(fake_quant)  # type: ignore[attr-defined]
        artifacts = backend.encode(prompts, categories)
    finally:
        backend.close()

    # Canonicalize stage order (cache loads alphabetical, encode returns
    # pipeline order — zip alignment depends on a consistent sort).
    artifacts = sorted(artifacts, key=lambda a: a.stage.value)
    for art in artifacts:
        art.meta["transform"] = transform
        art.meta["cache_context"] = cache_context
        save_artifact(art, cache_dir)
    return artifacts


def main() -> int:
    args = parse_args()

    from src.text_encoder.suite import load_suite

    prompts, categories = load_suite(args.prompts)
    if args.limit and args.limit < len(prompts):
        prompts = prompts[: args.limit]
        categories = categories[: args.limit]
    log.info("Loaded %d prompts (%d categories)", len(prompts), len(set(categories)))

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Cache invalidation scope: backend + DiT file (conditioning source)
    cache_ctx = f"{args.backend}:{Path(args.dit).name}"

    # --- Encode base ---
    log.info("=== Encoding BASE: %s ===", Path(args.base).name)
    base_artifacts = _encode_and_cache(
        args.backend,
        args.comfyui,
        args.base,
        args.dit,
        prompts,
        categories,
        args.device,
        cache_dir,
        "orig",
        cache_context=cache_ctx,
    )

    # --- Noise floor: real quantized base if provided, else fake-quant ---
    if args.quant_floor_file:
        log.info("=== Encoding QUANT FLOOR (real): %s ===", Path(args.quant_floor_file).name)
        quant_floor_artifacts = _encode_and_cache(
            args.backend,
            args.comfyui,
            args.quant_floor_file,
            args.dit,
            prompts,
            categories,
            args.device,
            cache_dir,
            "orig",
            cache_context=cache_ctx,
        )
        quant_floor_label = Path(args.quant_floor_file).name
    else:
        log.info("=== Encoding BASE (fake-quant int8 noise floor): %s ===", Path(args.base).name)
        quant_floor_artifacts = _encode_and_cache(
            args.backend,
            args.comfyui,
            args.base,
            args.dit,
            prompts,
            categories,
            args.device,
            cache_dir,
            "fakequant_int8",
            fake_quant="int8",
            cache_context=cache_ctx,
        )
        quant_floor_label = "fakequant_int8"

    # --- Comparison reference: per-variant override (matched-quant isolation)
    #     or the global base ---
    reference_file = args.reference or args.base
    if reference_file == args.base:
        reference_artifacts = base_artifacts
    else:
        log.info("=== Encoding REFERENCE: %s ===", Path(reference_file).name)
        reference_artifacts = _encode_and_cache(
            args.backend,
            args.comfyui,
            reference_file,
            args.dit,
            prompts,
            categories,
            args.device,
            cache_dir,
            "orig",
            cache_context=cache_ctx,
        )

    # --- Encode variant ---
    log.info("=== Encoding VARIANT: %s ===", Path(args.variant).name)
    variant_artifacts = _encode_and_cache(
        args.backend,
        args.comfyui,
        args.variant,
        args.dit,
        prompts,
        categories,
        args.device,
        cache_dir,
        "orig",
        cache_context=cache_ctx,
    )

    # --- Compare per stage ---
    from src.text_encoder.compare import compare
    from src.text_encoder.report import Report

    variant_reports = []
    for ref_art, var_art, qf_art, base_art in zip(
        reference_artifacts, variant_artifacts, quant_floor_artifacts, base_artifacts, strict=True
    ):
        if ref_art.stage != var_art.stage:
            raise RuntimeError(f"Stage mismatch: {ref_art.stage} vs {var_art.stage}")
        log.info("Comparing stage: %s (%d streams)", ref_art.stage.value, len(ref_art.streams))
        # floor_base: damage denominators are always measured against the
        # GLOBAL base, even when the comparison reference is a matched quant.
        vr = compare(ref_art, var_art, quant_floor=qf_art, floor_base=base_art, n_boot=args.n_boot)
        variant_reports.append(vr)

    # --- Report ---
    report = Report(
        base_id=args.base,
        backend=args.backend,
        reference_floors={"quant": quant_floor_label},
        variants=variant_reports,
        prompt_set_hash=base_artifacts[0].prompt_set_hash if base_artifacts else "",
        env={
            "device": args.device,
            "comfyui": args.comfyui,
            "n_prompts": str(len(prompts)),
            "reference": args.reference_name or Path(reference_file).name,
            "quant_floor": quant_floor_label,
        },
    )

    written = report.save(results_dir, variant_name=args.variant_name)
    for w in written:
        log.info("Wrote: %s", w)

    # --- Console summary: damage table ---
    for vr in variant_reports:
        for sr in vr.streams:
            if sr.damage_quant_units is not None:
                log.info(
                    "  %-14s %-11s CKA=%.6f  MSE=%.3e  damage=%.4f",
                    sr.stream,
                    sr.stage,
                    sr.cka_linear.value,
                    sr.mse.value,
                    sr.damage_quant_units.value,
                )

    log.info("=== Text-encoder analysis complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
