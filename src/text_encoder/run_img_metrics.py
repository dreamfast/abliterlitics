"""Image-metrics runner — score the A/B image set (LPIPS ladder + CLIP + SSIM/PSNR).

Reads the A/B manifest (run_ab_test.py output), mines the four pair types
(te_swap / cross_prompt / seed_change / floor_band — see image_metrics.py),
scores each pair with LPIPS + CLIP-image-distance + SSIM/PSNR, computes
CLIP-scores for te_swap pairs (prompt adherence proxy), and writes:

    <out-dir>/tebench_img_<variant>.json    per-variant pair values + summaries
    <out-dir>/tebench_img_anchors.json      shared anchors (seed_change, floor_band)

The ``tebench_`` prefix is MANDATORY: the te_results importer globs
``te_*.json`` and would otherwise mis-parse these files (Phase-4 plan rule).

Usage:
    python3 -m src.text_encoder.run_img_metrics \
        --manifest comparisons/te-zimage/results/te-zimage/text_encoder/ab_images/manifest.json \
        --out-dir comparisons/te-zimage/results/te-zimage/text_encoder \
        --reference base_int8 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy
import torch
from PIL import Image

from src.text_encoder import TE_RESULTS_VERSION
from src.text_encoder.image_metrics import (
    CLIPMeter,
    LPIPSMeter,
    PairScores,
    psnr,
    ssim,
)

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Image-space metrics for the TE A/B set")
    p.add_argument("--manifest", required=True, help="Path to ab_images/manifest.json")
    p.add_argument("--out-dir", required=True, help="Output directory (results/<cmp>/text_encoder)")
    p.add_argument("--reference", default="base_int8", help="Reference variant for te_swap pairs")
    p.add_argument("--floor-pair", default="base_int8,base_fp8", help="Two variants defining the quant-floor band")
    p.add_argument("--clip-model", default="ViT-L-14", help="open_clip model name")
    p.add_argument("--clip-pretrained", default="openai", help="open_clip pretrained tag")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--base-seed", type=int, default=42, help="The A/B fixed seed")
    p.add_argument("--cross-prompt-max", type=int, default=30, help="Cross-prompt pairs per variant")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--limit", type=int, default=0, help="Debug: cap pairs per kind (0 = all)")
    p.add_argument("--lpips-only", action="store_true", help="Skip CLIP metrics (fast pass)")
    return p.parse_args()


_REPO_ROOTS = (Path("/app"), Path.cwd())


def _resolve_path(p: str, results_dir: Path) -> Path:
    """Resolve a manifest image path across writer conventions.

    The manifest mixes styles across runs: repo-relative ("comparisons/..."),
    absolute container ("/results/..."), and absolute host paths. Try, in
    order: as-is; under each repo root; the ab_images-relative tail under the
    results dir.
    """
    cand = Path(p)
    if cand.exists():
        return cand
    for root in _REPO_ROOTS:
        joined = root / p.lstrip("/")
        if joined.exists():
            return joined
    if "ab_images/" in p:
        tail = p.split("ab_images/", 1)[1]
        under_results = results_dir / "ab_images" / tail
        if under_results.exists():
            return under_results
    return cand  # let the caller's open() raise the real error


@lru_cache(maxsize=256)
def _load_image(path: str, _results_dir_str: str = "") -> tuple[torch.Tensor, Any]:
    """Load an image once -> ([1,3,H,W] in [0,1], PIL.Image).

    Cached: each image participates in ~2-3 pairs; 256 slots × ~12MB ≈ 3GB.
    """
    resolved = _resolve_path(path, Path(_results_dir_str)) if _results_dir_str else Path(path)
    pil = Image.open(resolved).convert("RGB")
    arr = numpy.asarray(pil, dtype=numpy.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t, pil


class _CachedClipEncoder:
    """Per-path CLIP preprocessing + embedding cache (the reference image would
    otherwise be re-encoded once per variant — 27x waste)."""

    def __init__(self, meter: CLIPMeter) -> None:
        self.meter = meter
        self._cache: dict[str, torch.Tensor] = {}

    def embedding(self, path: str, pil: Any) -> torch.Tensor:
        emb = self._cache.get(path)
        if emb is None:
            emb = self.meter.encode_image(self.meter.preprocess_pil(pil))
            self._cache[path] = emb
        return emb

    def distance(self, path_a: str, pil_a: Any, path_b: str, pil_b: Any) -> float:
        ea = self.embedding(path_a, pil_a)
        eb = self.embedding(path_b, pil_b)
        return float(1.0 - (ea @ eb.T).squeeze().item())


def _score_pairs(
    pairs: list[Any],
    kind: str,
    lpips_meter: LPIPSMeter,
    clip_cache: _CachedClipEncoder | None,
    results_dir: Path,
    limit: int = 0,
) -> PairScores:
    """Load + score a pair list into a PairScores bucket."""
    scores = PairScores(kind=kind)
    selected = pairs if not limit else pairs[:limit]
    rs = str(results_dir)
    for pair in selected:
        try:
            t_a, pil_a = _load_image(pair.path_a, rs)
            t_b, pil_b = _load_image(pair.path_b, rs)
        except (OSError, ValueError) as exc:
            log.warning("failed to load pair %s/%s: %s", pair.path_a, pair.path_b, exc)
            continue
        scores.pairs.append(pair)
        scores.lpips.append(lpips_meter.score(t_a * 2.0 - 1.0, t_b * 2.0 - 1.0))
        scores.psnr_db.append(psnr(t_a, t_b))
        scores.ssim_val.append(ssim(t_a, t_b))
        if clip_cache is not None:
            scores.clip_img_dist.append(clip_cache.distance(pair.path_a, pil_a, pair.path_b, pil_b))
    return scores


def _bucket_to_json(bucket: PairScores, n_boot: int) -> dict[str, Any]:
    """Serialize a PairScores bucket: values + bootstrap summaries."""
    return {
        "kind": bucket.kind,
        "n_pairs": len(bucket.pairs),
        "pairs": [
            {
                "variant_a": p.variant_a,
                "variant_b": p.variant_b,
                "label": p.label,
                "seed_a": p.seed_a,
                "seed_b": p.seed_b,
            }
            for p in bucket.pairs
        ],
        "values": {
            "lpips": bucket.lpips,
            "clip_img_dist": bucket.clip_img_dist,
            "psnr_db": bucket.psnr_db,
            "ssim": bucket.ssim_val,
        },
        "summaries": [
            {"metric": s.metric, "n": s.n, "mean": s.mean, "ci_low": s.ci_low, "ci_high": s.ci_high}
            for s in bucket.summarize(n_boot=n_boot)
        ],
    }


def _summary_mean(bucket_json: dict[str, Any], metric: str) -> float | None:
    for s in bucket_json.get("summaries", []):
        if s["metric"] == metric:
            return float(s["mean"])
    return None


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = json.loads(manifest_path.read_text())
    images: list[dict[str, Any]] = manifest.get("images", [])
    log.info(
        "Manifest: %d images, %d variants, base seed %s",
        len(images),
        len({i.get("variant") for i in images}),
        manifest.get("seed"),
    )

    floor_pair = tuple(s.strip() for s in args.floor_pair.split(","))
    lpips_meter = LPIPSMeter(device=args.device)
    clip_meter: CLIPMeter | None = None
    clip_cache: _CachedClipEncoder | None = None
    if not args.lpips_only:
        clip_meter = CLIPMeter(args.clip_model, args.clip_pretrained, device=args.device)
        clip_cache = _CachedClipEncoder(clip_meter)

    # --- mine pairs -------------------------------------------------------
    from src.text_encoder.image_metrics import (
        mine_cross_prompt_pairs,
        mine_floor_band_pairs,
        mine_seed_change_pairs,
        mine_te_swap_pairs,
    )

    te_swap = mine_te_swap_pairs(images, reference=args.reference, base_seed=args.base_seed)
    cross_prompt = mine_cross_prompt_pairs(images, per_variant=args.cross_prompt_max, base_seed=args.base_seed)
    seed_change = mine_seed_change_pairs(images, base_seed=args.base_seed)
    floor_band = mine_floor_band_pairs(images, floor_pair=floor_pair, base_seed=args.base_seed)
    log.info(
        "Pairs: te_swap=%d cross_prompt=%d seed_change=%d floor_band=%d",
        len(te_swap),
        len(cross_prompt),
        len(seed_change),
        len(floor_band),
    )

    t0 = time.monotonic()

    # --- score anchors (shared) -------------------------------------------
    seed_bucket = _score_pairs(seed_change, "seed_change", lpips_meter, clip_cache, out_dir, args.limit)
    floor_bucket = _score_pairs(floor_band, "floor_band", lpips_meter, clip_cache, out_dir, args.limit)
    anchors: dict[str, Any] = {
        "te_results_version": TE_RESULTS_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "env": {
            "device": args.device,
            "lpips_net": "alex",
            "clip_model": None if clip_meter is None else f"{args.clip_model}/{args.clip_pretrained}",
            "reference": args.reference,
            "floor_pair": list(floor_pair),
            "base_seed": args.base_seed,
        },
        "seed_change": _bucket_to_json(seed_bucket, args.n_boot),
        "floor_band": _bucket_to_json(floor_bucket, args.n_boot),
    }
    (out_dir / "tebench_img_anchors.json").write_text(json.dumps(anchors, indent=2))
    seed_mean_lpips = _summary_mean(anchors["seed_change"], "lpips")
    floor_mean_lpips = _summary_mean(anchors["floor_band"], "lpips")
    log.info(
        "Anchors: seed_change lpips=%s floor_band lpips=%s (%.1fs)",
        seed_mean_lpips,
        floor_mean_lpips,
        time.monotonic() - t0,
    )

    # --- score per-variant buckets ----------------------------------------
    prompt_by_label: dict[str, str] = {str(i["label"]): str(i.get("prompt", "")) for i in images}
    clip_scores_by_variant: dict[str, dict[str, dict[str, float]]] = {}
    if clip_meter is not None:
        for pair in te_swap if not args.limit else te_swap[: args.limit]:
            vs = clip_scores_by_variant.setdefault(pair.variant_a, {})
            if pair.label in vs:
                continue
            try:
                _, pil_a = _load_image(pair.path_a, str(out_dir))
                _, pil_b = _load_image(pair.path_b, str(out_dir))
            except (OSError, ValueError) as exc:
                log.warning("clip-score load failed %s: %s", pair.label, exc)
                continue
            prompt = prompt_by_label.get(pair.label, "")
            v = clip_meter.clip_score(clip_meter.preprocess_pil(pil_a), prompt)
            r = clip_meter.clip_score(clip_meter.preprocess_pil(pil_b), prompt)
            vs[pair.label] = {"variant": v, "reference": r, "delta": v - r}

    by_variant: dict[str, list[Any]] = {}
    for pair in te_swap:
        by_variant.setdefault(pair.variant_a, []).append(pair)
    cp_by_variant: dict[str, list[Any]] = {}
    for pair in cross_prompt:
        cp_by_variant.setdefault(pair.variant_a, []).append(pair)

    n_written = 0
    for variant in sorted(by_variant):
        swap_bucket = _score_pairs(by_variant[variant], "te_swap", lpips_meter, clip_cache, out_dir, args.limit)
        cp_bucket = _score_pairs(
            cp_by_variant.get(variant, []), "cross_prompt", lpips_meter, clip_cache, out_dir, args.limit
        )
        swap_json = _bucket_to_json(swap_bucket, args.n_boot)
        cp_json = _bucket_to_json(cp_bucket, args.n_boot)
        ratios: dict[str, float | None] = {}
        for anchor_name, anchor_mean in (("seed_change", seed_mean_lpips), ("floor_band", floor_mean_lpips)):
            m = _summary_mean(swap_json, "lpips")
            if m is not None and anchor_mean is not None:
                ratios[f"lpips/{anchor_name}"] = m / anchor_mean
        swap_json["anchor_ratios"] = ratios
        cs = clip_scores_by_variant.get(variant, {})
        clip_delta_mean = float(sum(v["delta"] for v in cs.values()) / len(cs)) if cs else None
        doc = {
            "te_results_version": TE_RESULTS_VERSION,
            "variant": variant,
            "reference": args.reference,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "te_swap": swap_json,
            "cross_prompt": cp_json,
            "clip_scores": cs,
            "clip_score_delta_mean": clip_delta_mean,
        }
        (out_dir / f"tebench_img_{variant}.json").write_text(json.dumps(doc, indent=2))
        n_written += 1
        log.info(
            "%s: te_swap n=%d lpips=%s | %d pair files done",
            variant,
            swap_json["n_pairs"],
            _summary_mean(swap_json, "lpips"),
            n_written,
        )

    log.info("Wrote %d variant files + anchors to %s", n_written, out_dir)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
