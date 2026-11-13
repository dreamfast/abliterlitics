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

FPS = 24


def frames_for_duration(duration_s: float, fps: int = FPS) -> int:
    """H3 latent length constraint: frames % 17 == 5 (from the example graph)."""
    n = max(5, round(duration_s * fps))
    return n + (5 - (n % 17)) % 17


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

    _nid, clip_node = find_unique_node(wf, "CLIPLoader")
    if clip_node["inputs"].get("clip_name") is None:
        raise ValueError("CLIPLoader node lacks clip_name input")
    clip_node["inputs"]["clip_name"] = te_name

    _nid, noise_node = find_unique_node(wf, "RandomNoise")
    noise_node["inputs"]["noise_seed"] = seed

    _nid, prompt_node = find_unique_node(wf, "MiniMaxH3ImageToVideo", input_name="prompt")
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
    """Load H3 prompt entries (shared loader; contract in common_ab).

    ``text`` carries the FULL H3 3-field prompt (integrated description +
    soundscape + music) including any <d> dialogue wrappers. ``targets``
    (audio battery) lists the target words to score; passed through to the
    manifest for the audio scorer.
    """
    return load_prompt_suite(suite_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="A/B H3 video+audio generation via ComfyUI API")
    p.add_argument("--workflow", required=True)
    p.add_argument("--comparison", required=True)
    p.add_argument("--api-url", default="http://host.docker.internal:8191")
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--models-root",
        default=os.environ.get("COMFYUI_MODELS_ROOT", ""),
        help="ComfyUI models root for te_name relativization (default: $COMFYUI_MODELS_ROOT)",
    )
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
    suite = filter_suite_labels(suite, args.labels)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    frames = frames_for_duration(args.duration)
    jobs = build_arm_jobs(comparison, args.models_root, args.variants)

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

    def patch_and_node(
        wf_base: dict[str, Any], name: str, te_name: str, prompt: str, seed: int
    ) -> tuple[dict[str, Any], str]:
        wf = h3_patch_workflow(
            wf_base,
            te_name=te_name,
            prompt=prompt,
            seed=seed,
            filename_prefix=f"abliterlitics-h3/{name}",
            frames=frames,
        )
        return wf, find_unique_node(wf, "CLIPLoader")[0]

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
        list_key="clips",
        ext=".mp4",
        patch_and_node=patch_and_node,
    )

    log.info("=== H3 A/B generation complete: %d clips ===", len(manifest.get("clips", [])))


if __name__ == "__main__":
    main()
