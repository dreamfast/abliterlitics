"""A/B image generation for Krea2 via the ComfyUI API.

For each arm's TE, patches the exported Krea2 t2i workflow
(``krea2_t2i_api.json``, converted from the user's ``krea2_pro_bird.json``):
swaps the CLIPLoader clip_name, sets the CLIPTextEncode prompt, fixes BOTH
KSampler seeds (two-stage graph: 6-step gen + 2-step denoise-0.2 refine —
each stage derives its noise from its seed), and sets the SaveImageWebP
filename_prefix. Shares the ComfyUI client (one copy of the lesson-#39
semantics; WebP outputs fetch exactly like images).

Outputs:
    <out-dir>/<variant>/<label>__seed<seed>.webp
    <out-dir>/manifest.json                      (provenance + perf + targets;
                                                  z-image "images" layout so
                                                  run_img_metrics consumes it)

Node targeting is class-type based (assert exactly one match each, except
KSampler which is ALL-of-them — both stages must share the A/B seed),
NOT hardcoded ids.

Usage:
    python3 -m src.text_encoder.run_krea2_ab \
        --workflow comparisons/te-krea2/workflows/krea2_t2i_api.json \
        --comparison comparisons/te-krea2/comparison.json \
        --api-url http://host.docker.internal:8190 \
        --out-dir comparisons/te-krea2/results/te-krea2/ab_images \
        --prompt-suite prompts/te_krea2_prompts.yaml \
        --seeds 42 --variants heretic_int8 --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import yaml

from src.text_encoder.comfy_client import comfy_te_name, query_vram, queue_and_fetch, sanitize

log = logging.getLogger(__name__)

TE_LOADER = "CLIPLoader"  # krea2 TEs are safetensors (no GGUF arms planned)


def _find_node(wf: dict[str, Any], class_type: str, input_name: str = "") -> tuple[str, dict[str, Any]]:
    """Find the UNIQUE node of a class_type (fail loud on 0 or >1 matches)."""
    matches = [(nid, n) for nid, n in wf.items() if n.get("class_type") == class_type]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {class_type} node in workflow, found {len(matches)}: {[nid for nid, _ in matches]}"
        )
    nid, node = matches[0]
    if input_name and input_name not in node.get("inputs", {}):
        raise ValueError(f"Node {nid} ({class_type}) lacks input {input_name!r}")
    return nid, node


def krea2_patch_workflow(
    workflow: dict[str, Any],
    te_name: str,
    prompt: str,
    seed: int,
    filename_prefix: str,
) -> dict[str, Any]:
    """Copy of the workflow with TE + prompt + seeds + prefix patched.

    Fail-loud: CLIPLoader/CLIPTextEncode asserted unique; every KSampler
    (both stages of the two-stage graph) gets the same fixed seed so the
    A/B pair differs ONLY by the text encoder.
    """
    wf: dict[str, Any] = json.loads(json.dumps(workflow))  # deep copy

    _nid, clip_node = _find_node(wf, TE_LOADER, input_name="clip_name")
    clip_node["inputs"]["clip_name"] = te_name
    if clip_node["inputs"].get("type") != "krea2":
        raise ValueError(
            f"CLIPLoader type is {clip_node['inputs'].get('type')!r}, expected 'krea2' — "
            "wrong workflow file (this is the 12-tap VL interface, not qwen_image)"
        )

    _nid, encode_node = _find_node(wf, "CLIPTextEncode", input_name="text")
    encode_node["inputs"]["text"] = prompt

    sampler_nodes = [(nid, n) for nid, n in wf.items() if n.get("class_type") == "KSampler"]
    if not sampler_nodes:
        raise ValueError("No KSampler node found in workflow")
    for _nid, node in sampler_nodes:
        node["inputs"]["seed"] = seed

    prefix_nodes = 0
    for node in wf.values():
        if "filename_prefix" in node.get("inputs", {}):
            node["inputs"]["filename_prefix"] = filename_prefix
            prefix_nodes += 1
    if prefix_nodes == 0:
        raise ValueError("No node with filename_prefix found in workflow")
    return wf


def load_krea2_suite(suite_path: Path) -> list[dict[str, Any]]:
    """Load prompt entries: {prompts: [{text, category, label?, targets?}]}.

    ``targets`` (K4 word-content checks) lists words expected to appear IN
    the rendered image; passed through to the manifest for the scorer.
    """
    raw = yaml.safe_load(suite_path.read_text())
    if isinstance(raw, dict):
        raw = raw.get("prompts", [])
    if not raw:
        raise ValueError(f"No prompts in {suite_path}")
    entries: list[dict[str, Any]] = list(raw)
    for p in entries:
        if "text" not in p or "category" not in p:
            raise ValueError(f"Prompt entry missing text/category: {p}")
    return entries


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="A/B Krea2 image generation via ComfyUI API")
    p.add_argument("--workflow", required=True)
    p.add_argument("--comparison", required=True)
    p.add_argument("--api-url", default="http://host.docker.internal:8190")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--models-root", default="/home/maxx/extra/comfyui-models")
    p.add_argument("--prompt-suite", required=True)
    p.add_argument("--seeds", default="42", help="Comma-separated seed list (default 42)")
    p.add_argument("--labels", default="", help="Comma-separated prompt-label filter")
    p.add_argument("--variants", default="", help="Comma-separated arm filter (default: base + all with te_file)")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--smoke", action="store_true", help="Tag manifest smoke=true (stand-in runs; report refuses to publish)"
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    workflow = json.loads(Path(args.workflow).read_text())
    comparison = json.loads(Path(args.comparison).read_text())
    suite = load_krea2_suite(Path(args.prompt_suite))
    if args.labels:
        label_filter = {s.strip() for s in args.labels.split(",") if s.strip()}
        suite = [pr for pr in suite if (pr.get("label") or sanitize(pr["text"])) in label_filter]
        if not suite:
            raise SystemExit(f"--labels filter matched no prompts: {sorted(label_filter)}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    # Arm list: global base first, then variants with te_file
    filter_set = {v.strip() for v in args.variants.split(",") if v.strip()}
    jobs: list[tuple[str, str]] = []
    base_te = comparison.get("settings", {}).get("tebench", {}).get("base_te_file", "")
    if base_te and (not filter_set or "base" in filter_set):
        jobs.append(("base", comfy_te_name(base_te, args.models_root)))
    for name, cfg in comparison["variants"].items():
        te_file = cfg.get("te_file")
        if not te_file:
            continue
        if filter_set and name not in filter_set:
            continue
        jobs.append((name, comfy_te_name(te_file, args.models_root)))

    if not jobs:
        raise SystemExit("No arms to generate (no base_te_file and no variants with te_file)")
    log.info("Krea2 A/B plan: %d arms x %d prompts x %d seeds", len(jobs), len(suite), len(seeds))

    out_dir = Path(args.out_dir)
    # Manifest contract matches run_ab_test/run_img_metrics (z-image layout):
    # entries under "images" + the fixed A/B base seed at top level.
    manifest: dict[str, Any] = {
        "workflow": str(Path(args.workflow)),
        "api_url": args.api_url,
        "prompt_suite": str(args.prompt_suite),
        "seed": seeds[0],
        "smoke": bool(args.smoke),
        "images": [],
    }
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if args.smoke:
            manifest["smoke"] = True

    done: set[str] = {e["key"] for e in manifest.get("images", [])}
    by_variant: OrderedDict[str, str] = OrderedDict(jobs)

    for name, te_name in by_variant.items():
        first_clip = True
        for seed in seeds:
            for pr in suite:
                stem = pr.get("label") or sanitize(pr["text"])
                key = f"{name}::{pr.get('label', pr['category'])}::{pr['text']}"
                out_path = out_dir / name / f"{stem}__seed{seed}.webp"
                if args.skip_existing and key in done and out_path.exists():
                    continue
                if args.dry_run:
                    log.info("[dry-run] %s seed%d :: %s -> %s (te=%s)", name, seed, pr["category"], out_path, te_name)
                    continue
                wf = krea2_patch_workflow(
                    workflow,
                    te_name=te_name,
                    prompt=pr["text"],
                    seed=seed,
                    filename_prefix=f"abliterlitics-krea2/{name}",
                )
                data, perf = queue_and_fetch(args.api_url, wf, te_node_id=_find_node(wf, TE_LOADER)[0])
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(data)
                vram_cap = query_vram(args.api_url) if first_clip else None
                first_clip = False
                log.info(
                    "%s seed%d | %s | %s (%d KB, gen %sms)",
                    name,
                    seed,
                    pr["category"],
                    out_path.name,
                    len(data) // 1024,
                    perf.get("server_gen_ms"),
                )
                manifest.setdefault("images", []).append(
                    {
                        "key": key,
                        "pass": "warm",
                        "variant": name,
                        "te_name": te_name,
                        "gen_seconds": round(perf.get("client_wall_ms", 0) / 1000, 2),
                        "perf": perf,
                        "vram": vram_cap,
                        "category": pr["category"],
                        "label": pr.get("label", ""),
                        "prompt": pr["text"],
                        "targets": pr.get("targets", []),
                        "path": str(out_path),
                        "seed": seed,
                    }
                )
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(json.dumps(manifest, indent=2))

    log.info("=== Krea2 A/B generation complete: %d images ===", len(manifest.get("images", [])))


if __name__ == "__main__":
    main()
