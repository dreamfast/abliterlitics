"""A/B image generation via the ComfyUI API — real-stack image evidence.

For each variant with a te_file, patches the exported Z-Image workflow
(``zimage_ab_api.json``): swaps the text-encoder loader node (CLIPLoader for
safetensors, CLIPLoaderGGUF for .gguf containers), keeps seed/steps/sampler
identical, queues via /prompt, polls /history, downloads the image.

Outputs:
    <out-dir>/<variant>/<sanitized-prompt>__seed<seed>.webp   (raw)
    <out-dir>/manifest.json                                    (provenance)

Annotation (metrics banner + gallery) is a separate pass — annotate_ab.py —
so generation (GPU-bound) and presentation stay decoupled.

Usage:
    python3 -m src.text_encoder.run_ab_test \
        --workflow comparisons/te-zimage/workflows/zimage_ab_api.json \
        --comparison comparisons/te-zimage/comparison.json \
        --api-url http://host.docker.internal:8191 \
        --out-dir comparisons/te-zimage/results/te-zimage/ab_images \
        --prompts-per-category 1 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import yaml

from src.text_encoder.comfy_client import (
    comfy_te_name,
    free_models,
    queue_and_fetch,
    sanitize,
)
from src.text_encoder.comfy_client import (
    query_vram as _query_vram,
)

log = logging.getLogger(__name__)


# Node ids in the exported workflows (verified against zimage_ab_api.json
# and zimage_ab_gguf_api.json — same graph, TE loader node differs).
NODE_UNET = "34"
NODE_VAE = "36"
NODE_POS = "6"
NODE_NEG = "7"
NODE_CLIP_SAFETENSORS = "35"  # CLIPLoader
NODE_CLIP_GGUF = "38"  # CLIPLoaderGGUF (GGUF export has no node 35)

TE_LOADER_SAFETENSORS = "CLIPLoader"
TE_LOADER_GGUF = "CLIPLoaderGGUF"


def patch_workflow(
    workflow: dict[str, Any],
    clip_node_id: str,
    te_name: str,
    positive: str,
    negative: str,
    seed: int,
    filename_prefix: str,
) -> dict[str, Any]:
    """Return a copy of the workflow with TE + prompt + output patched.

    ``clip_node_id`` selects the TE loader node of the BASE workflow
    (NODE_CLIP_SAFETENSORS or NODE_CLIP_GGUF — the graphs are otherwise
    identical). The loader class is verified against the te_name extension
    so a stale workflow file fails loud instead of silently mis-loading.
    """
    wf: dict[str, Any] = json.loads(json.dumps(workflow))  # deep copy
    clip_node = wf[clip_node_id]
    is_gguf = te_name.lower().endswith(".gguf")
    loader = TE_LOADER_GGUF if is_gguf else TE_LOADER_SAFETENSORS
    if clip_node["class_type"] != loader:
        # Swap class, preserving the type/device inputs ComfyUI-GGUF shares
        old_inputs = clip_node["inputs"]
        wf[clip_node_id] = {
            "class_type": loader,
            "inputs": {
                "clip_name": te_name,
                "type": old_inputs.get("type", "lumina2"),
                **({"device": old_inputs["device"]} if "device" in old_inputs else {}),
            },
        }
    else:
        clip_node["inputs"]["clip_name"] = te_name
    wf[NODE_POS]["inputs"]["text"] = positive
    wf[NODE_NEG]["inputs"]["text"] = negative
    # Fixed seed for A/B comparability (denoise=1 => seed fully determines noise)
    for node in wf.values():
        if node.get("class_type") == "KSampler":
            node["inputs"]["seed"] = seed
    for node in wf.values():
        if "filename_prefix" in node.get("inputs", {}):
            node["inputs"]["filename_prefix"] = filename_prefix
    return wf


def load_prompt_suite(suite_path: Path, per_category: int) -> list[dict[str, str]]:
    """Load prompts from a suite file.

    Supports both layouts:
    - te_suite.yaml: flat ``[{text, category}]`` list
    - A/B stress file: ``{prompts: [{text, category, label?, checks?}]}``
    """
    raw = yaml.safe_load(suite_path.read_text())
    if isinstance(raw, dict):
        raw = raw.get("prompts", [])
    by_cat: dict[str, list[dict[str, str]]] = {}
    for p in raw:
        by_cat.setdefault(p["category"], []).append(p)
    out: list[dict[str, str]] = []
    if per_category > 0:
        for cat in sorted(by_cat):
            out.extend(by_cat[cat][:per_category])
    else:
        # 0 = take all (stress files carry their own curation + labels)
        for cat in sorted(by_cat):
            out.extend(by_cat[cat])
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A/B image generation via ComfyUI API")
    p.add_argument("--workflow", required=True)
    p.add_argument(
        "--gguf-workflow",
        default="",
        help="GGUF-TE workflow (CLIPLoaderGGUF graph); defaults to <workflow dir>/zimage_ab_gguf_api.json",
    )
    p.add_argument("--comparison", required=True)
    p.add_argument("--api-url", default="http://host.docker.internal:8191")
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--models-root",
        default=os.environ.get("COMFYUI_MODELS", ""),
        help="ComfyUI models root (TE files live under its text_encoders/). "
        "Defaults to $COMFYUI_MODELS; required if that is unset.",
    )
    p.add_argument("--prompt-suite", default="prompts/te_suite.yaml")
    p.add_argument("--prompts-per-category", type=int, default=1, help="0 = take all prompts in the suite")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--seeds",
        default="",
        help="Comma-separated seed LIST for multi-seed runs (e.g. '43,44,45'); overrides --seed. "
        "Used by the Phase-4 seed top-up to build the seed-change LPIPS anchor.",
    )
    p.add_argument(
        "--labels",
        default="",
        help="Comma-separated prompt-label filter (subset of the suite, e.g. stress prompts only)",
    )
    p.add_argument(
        "--passes",
        default="cold,warm",
        help="Comma-separated passes to run (default cold,warm; top-up uses warm only)",
    )
    p.add_argument("--negative", default="", help="Negative prompt (empty for turbo cfg=1)")
    p.add_argument("--variants", default="", help="Comma-separated filter (default: all with te_file)")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--cold-free",
        action="store_true",
        help="Call ComfyUI /free (unload all models) before each variant's cold pass "
        "for deterministic cold-start timing (includes DiT+VAE+TE disk load).",
    )
    p.add_argument(
        "--vram-only",
        action="store_true",
        help="Fast VRAM profiling: one cold image per variant, record peak VRAM. "
        "Merges into the existing manifest timing entries.",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if not args.models_root:
        p.error("--models-root is required (pass --models-root or set $COMFYUI_MODELS)")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    gguf_wf_path = (
        Path(args.gguf_workflow) if args.gguf_workflow else Path(args.workflow).parent / "zimage_ab_gguf_api.json"
    )
    if not gguf_wf_path.exists():
        gguf_wf_path = Path(args.workflow)  # single-graph fallback (CLIPLoaderGGUF already)
    workflows: dict[str, tuple[str, dict[str, Any]]] = {
        # loader kind -> (TE node id, workflow graph)
        "safetensors": (NODE_CLIP_SAFETENSORS, json.loads(Path(args.workflow).read_text())),
        "gguf": (
            NODE_CLIP_GGUF,
            json.loads(gguf_wf_path.read_text()),
        ),
    }
    comparison = json.loads(Path(args.comparison).read_text())
    suite = load_prompt_suite(Path(args.prompt_suite), args.prompts_per_category)
    if args.labels:
        label_filter = {s.strip() for s in args.labels.split(",") if s.strip()}
        suite = [pr for pr in suite if (pr.get("label") or sanitize(pr["text"])) in label_filter]
        if not suite:
            raise SystemExit(f"--labels filter matched no prompts: {sorted(label_filter)}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or [args.seed]
    pass_list = [s.strip() for s in args.passes.split(",") if s.strip()]

    # Variant list from comparison.json (dict form), with the global BASE
    # TE prepended as the "base" A-side anchor (it is not a variant entry).
    variants_cfg = comparison["variants"]
    filter_set = {v.strip() for v in args.variants.split(",") if v.strip()}
    jobs: list[tuple[str, str]] = []  # (variant_name, comfy_te_name)
    base_te = comparison.get("settings", {}).get("tebench", {}).get("base_te_file", "")
    if base_te and (not filter_set or "base" in filter_set):
        jobs.append(("base", comfy_te_name(base_te, args.models_root)))
    for name, cfg in variants_cfg.items():
        te_file = cfg.get("te_file")
        if not te_file:
            continue
        if filter_set and name not in filter_set:
            continue
        te_name = comfy_te_name(te_file, args.models_root)
        jobs.append((name, te_name))

    log.info(
        "A/B plan: %d variants x %d prompts x %d seeds %s",
        len(jobs),
        len(suite),
        len(seeds),
        f"(base {args.seed})" if len(seeds) == 1 else str(seeds),
    )

    out_dir = Path(args.out_dir)
    manifest: dict[str, Any] = {
        "seed": args.seed,
        "cold_free": bool(args.cold_free),
        "workflow": str(Path(args.workflow)),
        "api_url": args.api_url,
        "prompt_suite": str(args.prompt_suite),
        "images": [],
    }
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    # --- VRAM profiling fast path: one image per variant, record peak VRAM ---
    if args.vram_only:
        log.info("VRAM profiling pass: %d variants, 1 image each", len(jobs))
        for name, te_name in jobs:
            loader_kind = "gguf" if te_name.lower().endswith(".gguf") else "safetensors"
            clip_node_id, base_wf = workflows[loader_kind]
            wf = patch_workflow(
                base_wf,
                clip_node_id,
                te_name=te_name,
                positive="a red circle on white background, minimal test",
                negative="",
                seed=args.seed,
                filename_prefix=f"abliterlitics-vram/{name}",
            )
            if args.dry_run:
                log.info("[dry-run] %s vram probe (te=%s)", name, te_name)
                continue
            if args.cold_free:
                free_models(args.api_url)
            before = _query_vram(args.api_url)
            data, perf = queue_and_fetch(args.api_url, wf, te_node_id=clip_node_id)
            after = _query_vram(args.api_url)
            vram = {
                **after,
                "delta_mb": round(after["vram_used_mb"] - before["vram_used_mb"], 1),
                "cold_start_ms": perf.get("server_gen_ms"),
                "cold_free": bool(args.cold_free),
            }
            log.info(
                "VRAM %s: used=%.0fMB torch=%.0fMB gen=%dms (te=%s)",
                name,
                vram["vram_used_mb"],
                vram["torch_vram_mb"],
                perf.get("server_gen_ms", 0),
                te_name,
            )
            manifest.setdefault("timing", {}).setdefault(name, {}).update(
                {
                    "variant": name,
                    "te_name": te_name,
                    "loader": loader_kind,
                    "vram": vram,
                }
            )
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2))
        log.info("=== VRAM profiling complete ===")
        return

    done: set[str] = {e["key"] for e in manifest.get("images", [])}

    # Group jobs by variant: each variant runs its full prompt set in
    # PASS 1 (cold: TE freshly loaded into VRAM) and PASS 2 (warm: TE
    # cached by ComfyUI's model manager). Per-variant timing = first-image
    # latency per pass; per-image timing recorded per pass.
    from collections import OrderedDict

    by_variant: OrderedDict[str, str] = OrderedDict(jobs)

    for name, te_name in by_variant.items():
        loader_kind = "gguf" if te_name.lower().endswith(".gguf") else "safetensors"
        clip_node_id, base_wf = workflows[loader_kind]
        timing: dict[str, Any] = {"variant": name, "te_name": te_name, "loader": loader_kind, "passes": {}}
        for seed in seeds:
            for pass_name in pass_list:
                if pass_name == "cold" and args.cold_free:
                    free_models(args.api_url)
                pass_t0 = time.monotonic()
                per_image: list[float] = []
                for pr in suite:
                    stem = pr.get("label") or sanitize(pr["text"])
                    key = f"{name}::{pr.get('label', pr['category'])}::{pr['text']}"
                    out_path = out_dir / name / f"{stem}__pass-{pass_name}__seed{seed}.webp"
                    if args.skip_existing and key in done and out_path.exists():
                        continue
                    if args.dry_run:
                        log.info(
                            "[dry-run] %s/%s seed%d :: %s -> %s (te=%s)",
                            name,
                            pass_name,
                            seed,
                            pr["category"],
                            out_path,
                            te_name,
                        )
                        continue
                    wf = patch_workflow(
                        base_wf,
                        clip_node_id,
                        te_name=te_name,
                        positive=pr["text"],
                        negative=args.negative,
                        seed=seed,
                        filename_prefix=f"abliterlitics-ab/{name}",
                    )
                    t0 = time.monotonic()
                    data, perf = queue_and_fetch(args.api_url, wf, te_node_id=clip_node_id)
                    dt = time.monotonic() - t0
                    per_image.append(dt)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_bytes(data)
                    log.info(
                        "%s/%s seed%d | %s | %s (%.1fs, %d KB)",
                        name,
                        pass_name,
                        seed,
                        pr["category"],
                        out_path.name,
                        dt,
                        len(data) // 1024,
                    )
                    # Capture VRAM after cold-pass first image (peak from TE load)
                    vram_cap: dict[str, float] | None = None
                    if pass_name == "cold" and len(per_image) == 1:
                        vram_cap = _query_vram(args.api_url)
                    manifest.setdefault("images", []).append(
                        {
                            "key": key,
                            "variant": name,
                            "te_name": te_name,
                            "loader": loader_kind,
                            "pass": pass_name,
                            "gen_seconds": round(dt, 2),
                            "perf": perf,
                            "vram": vram_cap,
                            "category": pr["category"],
                            "label": pr.get("label", ""),
                            "prompt": pr["text"],
                            "checks": pr.get("checks", []),
                            "path": str(out_path),
                            "seed": seed,
                        }
                    )
                    manifest_path.parent.mkdir(parents=True, exist_ok=True)
                    manifest_path.write_text(json.dumps(manifest, indent=2))
                if per_image:
                    timing["passes"][pass_name] = {
                        "first_image_seconds": round(per_image[0], 2),
                        "mean_image_seconds": round(sum(per_image) / len(per_image), 2),
                        "n_images": len(per_image),
                        "pass_wall_seconds": round(time.monotonic() - pass_t0, 2),
                    }
        # Merge into existing timing (top-up runs partial passes — never clobber)
        existing = manifest.setdefault("timing", {}).get(name)
        if existing and existing.get("passes"):
            existing["passes"].update(timing["passes"])
        else:
            manifest["timing"][name] = timing
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        warm_first = timing["passes"].get("warm", {}).get("first_image_seconds")
        if warm_first:
            log.info("Timing %s: warm first-image %ss (this run)", name, warm_first)

    log.info("=== A/B generation complete: %d images ===", len(manifest.get("images", [])))


if __name__ == "__main__":
    main()
