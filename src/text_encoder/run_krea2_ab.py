"""A/B image generation for Krea2 via the ComfyUI API.

For each arm's TE, patches the exported Krea2 t2i workflow
(``krea2_t2i_api.json``, converted from the user's ``krea2_pro_bird.json``):
swaps the TE loader (``CLIPLoader`` for safetensors, ``CLIPLoaderGGUF`` for
.gguf arms — class swapped in place, ``type: krea2`` preserved), optionally
patches the ``UNETLoader`` DiT name (config-driven stand-in/turbo swap), sets
the CLIPTextEncode prompt, fixes BOTH KSampler seeds (two-stage graph:
6-step gen + 2-step denoise-0.2 refine — each stage derives its noise from
its seed), and sets the SaveImageWebP filename_prefix. Shares the ComfyUI
client (one copy of the lesson-#39 semantics; WebP outputs fetch exactly
like images).

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
import os
from pathlib import Path
from typing import Any

from src.text_encoder.common_ab import (
    build_arm_jobs,
    filter_suite_labels,
    find_unique_node,
    load_prompt_suite,
    run_ab_loop,
)

log = logging.getLogger(__name__)

TE_LOADER_SAFETENSORS = "CLIPLoader"
TE_LOADER_GGUF = "CLIPLoaderGGUF"


def _find_te_loader(wf: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Find the UNIQUE TE loader node — CLIPLoader or CLIPLoaderGGUF.

    The workflow carries ONE TE loader whose class depends on the arm being
    patched (safetensors vs .gguf), so matching either class keeps both the
    patcher and the post-patch te_node_id lookup correct.
    """
    matches = [(nid, n) for nid, n in wf.items() if n.get("class_type") in (TE_LOADER_SAFETENSORS, TE_LOADER_GGUF)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one TE loader ({TE_LOADER_SAFETENSORS}/{TE_LOADER_GGUF}) node in workflow, "
            f"found {len(matches)}: {[nid for nid, _ in matches]}"
        )
    nid, node = matches[0]
    if "clip_name" not in node.get("inputs", {}):
        raise ValueError(f"TE loader node {nid} ({node.get('class_type')}) lacks input 'clip_name'")
    return nid, node


def krea2_patch_workflow(
    workflow: dict[str, Any],
    te_name: str,
    prompt: str,
    seed: int,
    filename_prefix: str,
    dit_name: str | None = None,
) -> dict[str, Any]:
    """Copy of the workflow with TE + prompt + seeds + prefix (+ DiT) patched.

    Fail-loud: TE loader/CLIPTextEncode asserted unique; every KSampler
    (both stages of the two-stage graph) gets the same fixed seed so the
    A/B pair differs ONLY by the text encoder. ``te_name`` ending in .gguf
    swaps the loader class to CLIPLoaderGGUF in place (type/device inputs
    preserved); ``dit_name`` patches the UNETLoader so the stand-in/turbo
    DiT swap is config-driven (None = leave the workflow's DiT untouched).
    """
    wf: dict[str, Any] = json.loads(json.dumps(workflow))  # deep copy

    clip_nid, clip_node = _find_te_loader(wf)
    te_type = clip_node["inputs"].get("type")
    if te_type != "krea2":
        raise ValueError(
            f"TE loader type is {te_type!r}, expected 'krea2' — "
            "wrong workflow file (this is the 12-tap VL interface, not qwen_image)"
        )
    is_gguf = te_name.lower().endswith(".gguf")
    loader = TE_LOADER_GGUF if is_gguf else TE_LOADER_SAFETENSORS
    if clip_node["class_type"] != loader:
        # Swap class, preserving the type/device inputs ComfyUI-GGUF shares
        old_inputs = clip_node["inputs"]
        wf[clip_nid] = {
            "class_type": loader,
            "inputs": {
                "clip_name": te_name,
                "type": te_type,
                **({"device": old_inputs["device"]} if "device" in old_inputs else {}),
            },
        }
    else:
        clip_node["inputs"]["clip_name"] = te_name

    if dit_name is not None:
        _unid, unet_node = find_unique_node(wf, "UNETLoader", input_name="unet_name")
        unet_node["inputs"]["unet_name"] = dit_name

    _nid, encode_node = find_unique_node(wf, "CLIPTextEncode", input_name="text")
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
    """Load prompt entries (shared loader; contract in common_ab).

    ``targets`` (K4 word-content checks) lists words expected to appear IN
    the rendered image; passed through to the manifest for the scorer.
    """
    return load_prompt_suite(suite_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="A/B Krea2 image generation via ComfyUI API")
    p.add_argument("--workflow", required=True)
    p.add_argument("--comparison", required=True)
    p.add_argument("--api-url", default="http://host.docker.internal:8190")
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--models-root",
        default=os.environ.get("COMFYUI_MODELS_ROOT", ""),
        help="ComfyUI models root for te_name relativization (default: $COMFYUI_MODELS_ROOT)",
    )
    p.add_argument("--prompt-suite", required=True)
    p.add_argument(
        "--dit-name",
        default="",
        help="UNETLoader unet_name to patch in (default: basename of settings.tebench.dit_file; "
        "'none' (any case) leaves the workflow's DiT untouched)",
    )
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

    # DiT resolution: explicit --dit-name > settings.tebench.dit_file basename
    # > None (workflow untouched). Config-driven so the stand-in -> turbo
    # swap never needs a workflow file edit.
    if args.dit_name:
        dit_name = None if args.dit_name.lower() == "none" else args.dit_name
    else:
        cfg_dit = comparison.get("settings", {}).get("tebench", {}).get("dit_file", "")
        dit_name = Path(cfg_dit).name if cfg_dit else None
    if dit_name:
        log.info("DiT: patching UNETLoader -> %s", dit_name)
    else:
        log.info("DiT: leaving workflow DiT untouched")
    suite = filter_suite_labels(suite, args.labels)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    jobs = build_arm_jobs(comparison, args.models_root, args.variants)

    log.info("Krea2 A/B plan: %d arms x %d prompts x %d seeds", len(jobs), len(suite), len(seeds))

    out_dir = Path(args.out_dir)
    # Manifest contract matches run_ab_test/run_img_metrics (z-image layout):
    # entries under "images" + the fixed A/B base seed at top level.
    manifest: dict[str, Any] = {
        "workflow": str(Path(args.workflow)),
        "api_url": args.api_url,
        "prompt_suite": str(args.prompt_suite),
        "seed": seeds[0],
        "dit_name": dit_name,
        "smoke": bool(args.smoke),
        "images": [],
    }
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if args.smoke:
            manifest["smoke"] = True
        # Resume path: keep dit_name provenance honest — update when this run
        # knows it, warn if it contradicts what earlier entries were built with.
        if dit_name is not None:
            prior = manifest.get("dit_name")
            if prior is not None and prior != dit_name:
                log.warning("Manifest dit_name %r != this run's %r — mixed-DiT manifest!", prior, dit_name)
            manifest["dit_name"] = dit_name

    def patch_and_node(
        wf_base: dict[str, Any], name: str, te_name: str, prompt: str, seed: int
    ) -> tuple[dict[str, Any], str]:
        wf = krea2_patch_workflow(
            wf_base,
            te_name=te_name,
            prompt=prompt,
            seed=seed,
            filename_prefix=f"abliterlitics-krea2/{name}",
            dit_name=dit_name,
        )
        return wf, _find_te_loader(wf)[0]

    run_ab_loop(
        manifest=manifest,
        manifest_path=manifest_path,
        out_dir=out_dir,
        jobs=jobs,
        suite=suite,
        seeds=seeds,
        workflow=workflow,
        api_url=args.api_url,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
        list_key="images",
        ext=".webp",
        patch_and_node=patch_and_node,
        extra_entry={"pass": "warm"},
    )

    log.info("=== Krea2 A/B generation complete: %d images ===", len(manifest.get("images", [])))


if __name__ == "__main__":
    main()
