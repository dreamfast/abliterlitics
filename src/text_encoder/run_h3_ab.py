"""A/B video+audio generation for MiniMax H3 via the ComfyUI API.

For each arm's TE, patches the exported H3 t2v workflow
(``h3_t2v_api.json``): swaps the CLIPLoader clip_name, sets the prompt
(the full H3 3-field format incl. ``<d>[Language] ...</d>`` dialogue), the
RandomNoise seed, the SaveVideo filename_prefix, and the PrimitiveInt frame
count. Shares the ComfyUI client with run_ab_test.py (comfy_client.py —
one copy of the lesson-#39 semantics; SaveVideo outputs fetch exactly like
images).

Outputs:
    <out-dir>/<variant>/<label>__seed<seed>.mp4   (video with muxed audio)
    <out-dir>/manifest.json                       (provenance + perf)

Node targeting is class-type based (assert exactly one match each), NOT
hardcoded ids — workflow exports renumber freely.

Usage:
    python3 -m src.text_encoder.run_h3_ab \
        --workflow comparisons/te-h3/workflows/h3_t2v_api.json \
        --comparison comparisons/te-h3/comparison.json \
        --api-url http://host.docker.internal:8191 \
        --out-dir comparisons/te-h3/results/te-h3/ab_videos \
        --prompt-suite prompts/te_h3_video_prompts.yaml \
        --seeds 42 --variants base,base_nvfp4
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

FPS = 24


def frames_for_duration(duration_s: float, fps: int = FPS) -> int:
    """H3 latent length constraint: frames % 17 == 5 (from the example graph)."""
    n = max(5, round(duration_s * fps))
    return n + (5 - (n % 17)) % 17


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


def _find_frames_node(wf: dict[str, Any], prompt_node_id: str) -> tuple[str, dict[str, Any]]:
    """Locate the frame-count source: the PrimitiveInt feeding 'length'.

    Follows the MiniMaxH3ImageToVideo 'length' input link instead of
    assuming PrimitiveInt is unique (utility nodes multiply in re-exports).
    """
    length = wf[prompt_node_id]["inputs"].get("length")
    if not (isinstance(length, list) and len(length) == 2):
        raise ValueError(f"Node {prompt_node_id} 'length' input is not a link: {length!r}")
    fid = str(length[0])
    node = wf.get(fid)
    if node is None:
        raise ValueError(f"'length' links to missing node {fid}")
    if node.get("class_type") != "PrimitiveInt":
        raise ValueError(f"'length' source node {fid} is {node.get('class_type')}, not PrimitiveInt")
    return fid, node


def h3_patch_workflow(
    workflow: dict[str, Any],
    te_name: str,
    prompt: str,
    seed: int,
    filename_prefix: str,
    frames: int,
) -> dict[str, Any]:
    """Copy of the workflow with TE + prompt + seed + prefix + frames patched.

    Fail-loud: every targeted node class is asserted (CLIPLoader for the TE,
    RandomNoise for the seed, MiniMaxH3ImageToVideo for the prompt,
    PrimitiveInt for the frame count). A stale/edited workflow file fails
    here instead of silently generating with the wrong arm.
    """
    wf: dict[str, Any] = json.loads(json.dumps(workflow))  # deep copy

    _nid, clip_node = _find_node(wf, "CLIPLoader")
    if clip_node["inputs"].get("clip_name") is None:
        raise ValueError("CLIPLoader node lacks clip_name input")
    clip_node["inputs"]["clip_name"] = te_name

    _nid, noise_node = _find_node(wf, "RandomNoise")
    noise_node["inputs"]["noise_seed"] = seed

    _nid, prompt_node = _find_node(wf, "MiniMaxH3ImageToVideo", input_name="prompt")
    prompt_node["inputs"]["prompt"] = prompt

    _fid, frames_node = _find_frames_node(wf, _nid)
    frames_node["inputs"]["value"] = frames

    prefix_nodes = 0
    for node in wf.values():
        if "filename_prefix" in node.get("inputs", {}):
            node["inputs"]["filename_prefix"] = filename_prefix
            prefix_nodes += 1
    if prefix_nodes == 0:
        raise ValueError("No node with filename_prefix found in workflow")
    return wf


def load_h3_suite(suite_path: Path) -> list[dict[str, Any]]:
    """Load H3 prompt entries: {prompts: [{text, category, label?, targets?}]}.

    ``text`` carries the FULL H3 3-field prompt (integrated description +
    soundscape + music) including any <d> dialogue wrappers. ``targets``
    (audio battery) lists the target words to score; passed through to the
    manifest for the audio scorer.
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
    p = argparse.ArgumentParser(description="A/B H3 video+audio generation via ComfyUI API")
    p.add_argument("--workflow", required=True)
    p.add_argument("--comparison", required=True)
    p.add_argument("--api-url", default="http://host.docker.internal:8191")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--models-root", default="/home/maxx/extra/comfyui-models")
    p.add_argument("--prompt-suite", required=True)
    p.add_argument("--seeds", default="42", help="Comma-separated seed list (default 42)")
    p.add_argument("--labels", default="", help="Comma-separated prompt-label filter")
    p.add_argument("--variants", default="", help="Comma-separated arm filter (default: base + all with te_file)")
    p.add_argument("--duration", type=float, default=5.0, help="Clip duration seconds (frames rounded to %17==5)")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--smoke", action="store_true", help="Tag manifest smoke=true (stand-in runs; report refuses to publish)"
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    workflow = json.loads(Path(args.workflow).read_text())
    comparison = json.loads(Path(args.comparison).read_text())
    suite = load_h3_suite(Path(args.prompt_suite))
    if args.labels:
        label_filter = {s.strip() for s in args.labels.split(",") if s.strip()}
        suite = [pr for pr in suite if (pr.get("label") or sanitize(pr["text"])) in label_filter]
        if not suite:
            raise SystemExit(f"--labels filter matched no prompts: {sorted(label_filter)}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    frames = frames_for_duration(args.duration)

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
    log.info(
        "H3 A/B plan: %d arms x %d prompts x %d seeds x %ds (%d frames)",
        len(jobs),
        len(suite),
        len(seeds),
        args.duration,
        frames,
    )

    out_dir = Path(args.out_dir)
    manifest: dict[str, Any] = {
        "workflow": str(Path(args.workflow)),
        "api_url": args.api_url,
        "prompt_suite": str(args.prompt_suite),
        "duration_s": args.duration,
        "frames": frames,
        "fps": FPS,
        "smoke": bool(args.smoke),
        "clips": [],
    }
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if args.smoke:
            manifest["smoke"] = True

    done: set[str] = {e["key"] for e in manifest.get("clips", [])}
    by_variant: OrderedDict[str, str] = OrderedDict(jobs)

    for name, te_name in by_variant.items():
        first_clip = True
        for seed in seeds:
            for pr in suite:
                stem = pr.get("label") or sanitize(pr["text"])
                key = f"{name}::{pr.get('label', pr['category'])}::{pr['text']}"
                out_path = out_dir / name / f"{stem}__seed{seed}.mp4"
                if args.skip_existing and key in done and out_path.exists():
                    continue
                if args.dry_run:
                    log.info("[dry-run] %s seed%d :: %s -> %s (te=%s)", name, seed, pr["category"], out_path, te_name)
                    continue
                wf = h3_patch_workflow(
                    workflow,
                    te_name=te_name,
                    prompt=pr["text"],
                    seed=seed,
                    filename_prefix=f"abliterlitics-h3/{name}",
                    frames=frames,
                )
                data, perf = queue_and_fetch(args.api_url, wf, te_node_id=_find_node(wf, "CLIPLoader")[0])
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
                manifest.setdefault("clips", []).append(
                    {
                        "key": key,
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

    log.info("=== H3 A/B generation complete: %d clips ===", len(manifest.get("clips", [])))


if __name__ == "__main__":
    main()
