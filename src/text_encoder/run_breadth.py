"""Breadth battery scorer — is the encoder shift uniform across prompt categories?

Phase 5 (P10). Scores the breadth A/B manifest (run_ab_test output over
prompts/te_breadth_prompts.yaml) with:

  - per-category LPIPS(variant vs base) at matched (label, seed)
  - per-category CLIP-sim(image, prompt) for variant and base
  - a per-category SEED ANCHOR (base seed42 vs seed43) so the variant shift
    can be read against "how different is a different picture of THIS
    category" rather than a global anchor

Output: tebench_breadth.json (tebench_ namespace — importer glob safety).

Usage (in the te container, repo at /app):
    python3 -m src.text_encoder.run_breadth \
        --manifest comparisons/.../ab_images_breadth/manifest.json \
        --suite prompts/te_breadth_prompts.yaml \
        --out comparisons/.../tebench_breadth.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy
import torch
import yaml
from PIL import Image

from src.text_encoder import TE_RESULTS_VERSION
from src.text_encoder._cli import add_device
from src.text_encoder.image_metrics import CLIPMeter, LPIPSMeter

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Breadth battery scorer (P10)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--suite", required=True, help="te_breadth_prompts.yaml")
    p.add_argument("--base", default="base")
    p.add_argument("--out", required=True)
    add_device(p, default="cuda")
    p.add_argument("--clip-model", default="ViT-L-14")
    p.add_argument("--clip-pretrained", default="openai")
    p.add_argument("--pass-name", default="warm")
    p.add_argument(
        "--anchor-seeds",
        default="",
        help="Seed pair 'A,B' for the per-category seed anchor "
        "(default: derived from the seeds present in the BASE records)",
    )
    return p.parse_args()


def _load_image(path: Path) -> tuple[torch.Tensor, Image.Image]:
    pil = Image.open(path).convert("RGB")
    arr = numpy.asarray(pil, dtype=numpy.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0), pil


def load_records(manifest: Path, pass_name: str) -> dict[tuple[str, str, int], dict[str, Any]]:
    """(variant, label, seed) -> record, warm pass only."""
    data = json.loads(manifest.read_text())
    recs = data["images"] if isinstance(data, dict) and "images" in data else data
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    for r in recs:
        if str(r.get("pass", "")) != pass_name:
            continue
        out[(str(r["variant"]), str(r["label"]), int(r["seed"]))] = r
    return out


def _anchor_seed_pair(
    records: dict[tuple[str, str, int], dict[str, Any]],
    base: str,
    anchor_arg: str,
) -> tuple[int, int] | None:
    """Seed pair for the per-category seed anchor.

    ``--anchor-seeds 'A,B'`` wins; otherwise derived from the seeds present in
    the BASE records (audit bug #7: the old code hardcoded 42/43 regardless of
    the manifest, silently producing n=0 / null anchors for manifests
    generated with other seed lists — the lesson #48 silent-empty class).
    Returns None (with a loud warning) when fewer than 2 base seeds exist.
    """
    if anchor_arg.strip():
        parts = [int(s) for s in anchor_arg.split(",") if s.strip()]
        if len(parts) != 2:
            raise SystemExit(f"--anchor-seeds needs exactly 2 seeds 'A,B', got: {anchor_arg!r}")
        return parts[0], parts[1]
    seeds = sorted({s for (v, _lab, s) in records if v == base})
    if len(seeds) < 2:
        log.warning(
            "Seed anchor skipped: only %d seed(s) in base records %s — "
            "generate a second seed or pass --anchor-seeds 'A,B'",
            len(seeds),
            seeds,
        )
        return None
    return seeds[0], seeds[1]


def main() -> int:
    args = parse_args()
    manifest = Path(args.manifest)
    suite_raw = yaml.safe_load(Path(args.suite).read_text())
    entries = suite_raw.get("prompts", suite_raw) if isinstance(suite_raw, dict) else suite_raw
    by_label = {e["label"]: e for e in entries}

    records = load_records(manifest, args.pass_name)
    variants = sorted({v for (v, _, _) in records} - {args.base})
    categories = sorted({e["category"] for e in by_label.values()})
    log.info(
        "Breadth manifest: %d records, %d non-base variants, %d categories",
        len(records),
        len(variants),
        len(categories),
    )

    lpips_meter = LPIPSMeter(device=args.device)
    clip = CLIPMeter(args.clip_model, args.clip_pretrained, device=args.device)

    def _rec_path(rec: dict[str, Any]) -> Path:
        """Resolve a manifest path (repo-relative from cwd, or manifest-relative)."""
        p = Path(str(rec["path"]))
        if p.is_absolute():
            return p
        if p.exists():
            return p
        return manifest.parent / p

    # Pre-embed prompts (one per label)
    labels = sorted(by_label)
    text_emb = {lab: clip.encode_text([str(by_label[lab]["text"])]).squeeze(0) for lab in labels}

    # image embedding cache keyed by record path
    _emb_cache: dict[str, torch.Tensor] = {}

    def _emb(rec: dict[str, Any]) -> torch.Tensor:
        key = str(rec["path"])
        e = _emb_cache.get(key)
        if e is None:
            _, pil = _load_image(_rec_path(rec))
            e = clip.encode_image(clip.preprocess_pil(pil))
            _emb_cache[key] = e
        return e

    per_pair: list[dict[str, Any]] = []
    per_cat: dict[str, dict[str, dict[str, Any]]] = {}
    for var in variants:
        per_cat[var] = {}
        for cat in categories:
            agg: dict[str, Any] = {"n": 0, "lpips": [], "clip_sim": [], "clip_delta": []}
            for lab in labels:
                if by_label[lab]["category"] != cat:
                    continue
                for seed in sorted({s for (v, lb, s) in records if v == args.base and lb == lab}):
                    rv = records.get((var, lab, seed))
                    rb = records.get((args.base, lab, seed))
                    if rv is None or rb is None:
                        continue
                    t_v, _ = _load_image(_rec_path(rv))
                    t_b, _ = _load_image(_rec_path(rb))
                    lp = lpips_meter.score(t_v * 2 - 1, t_b * 2 - 1)
                    sim_v = float((_emb(rv).squeeze(0) @ text_emb[lab]).item())
                    sim_b = float((_emb(rb).squeeze(0) @ text_emb[lab]).item())
                    per_pair.append(
                        {
                            "variant": var,
                            "category": cat,
                            "label": lab,
                            "seed": seed,
                            "lpips": lp,
                            "clip_sim": sim_v,
                            "clip_sim_base": sim_b,
                            "clip_delta": sim_v - sim_b,
                        }
                    )
                    agg["n"] += 1
                    agg["lpips"].append(lp)
                    agg["clip_sim"].append(sim_v)
                    agg["clip_delta"].append(sim_v - sim_b)
            per_cat[var][cat] = {
                "n": agg["n"],
                "lpips_mean": sum(agg["lpips"]) / len(agg["lpips"]) if agg["lpips"] else None,
                "clip_sim_mean": sum(agg["clip_sim"]) / len(agg["clip_sim"]) if agg["clip_sim"] else None,
                "clip_delta_mean": (sum(agg["clip_delta"]) / len(agg["clip_delta"]) if agg["clip_delta"] else None),
            }

    # per-category seed anchor: base seedA vs seedB (same label) — seeds
    # derived from the base records or --anchor-seeds (see _anchor_seed_pair)
    anchor_pair = _anchor_seed_pair(records, args.base, args.anchor_seeds)
    seed_anchor: dict[str, dict[str, Any]] = {}
    if anchor_pair is not None:
        seed_a, seed_b = anchor_pair
        for cat in categories:
            vals: list[float] = []
            for lab in labels:
                if by_label[lab]["category"] != cat:
                    continue
                r_a = records.get((args.base, lab, seed_a))
                r_b = records.get((args.base, lab, seed_b))
                if r_a is None or r_b is None:
                    continue
                t_a, _ = _load_image(_rec_path(r_a))
                t_b, _ = _load_image(_rec_path(r_b))
                vals.append(lpips_meter.score(t_a * 2 - 1, t_b * 2 - 1))
            seed_anchor[cat] = {
                "n": len(vals),
                "lpips_mean": sum(vals) / len(vals) if vals else None,
            }

    out = {
        "te_results_version": TE_RESULTS_VERSION,
        "analysis": "breadth_battery",
        "base": args.base,
        "manifest": str(manifest),
        "suite": str(args.suite),
        "clip_model": f"{args.clip_model}/{args.clip_pretrained}",
        "categories": categories,
        "per_category": per_cat,
        "anchor_seeds": list(anchor_pair) if anchor_pair else [],
        "seed_anchor_per_category": seed_anchor,
        "per_pair": per_pair,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    log.info("Wrote %s (%d pairs)", args.out, len(per_pair))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
