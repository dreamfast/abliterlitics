#!/usr/bin/env python3
"""
Stacking Analysis -- Forensic provenance test for model derivation.

Tests whether variant_b (Hauhau) was built on top of variant_a (Heretic)
by decomposing edit vectors: D_a = D_h + D_residual.

Key metrics per tensor:
  - ||D_r|| / ||D_a|| ratio  (stacking evidence: <1 means residual is smaller)
  - Regression slope: D_a ≈ alpha * D_h + D_extra
  - cos(D_h, D_r)             (residual orthogonality -- descriptive only)
  - Bitwise/near-exact match counts
  - Comparison against fresh Heretic variants for baseline

NOTE: This script does NOT use comparison.json. It takes explicit
--extra-variants and --extra-labels flags only (investigation-specific tool).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import torch

from src.model_config import build_shard_map, detect_architecture, load_tensor

log = logging.getLogger(__name__)

THRESHOLD = 1e-10


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    dot = (a_f * b_f).sum()
    na = a_f.norm()
    nb = b_f.norm()
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return max(-1.0, min(1.0, (dot / (na * nb)).item()))


def regression_slope(predictor: torch.Tensor, response: torch.Tensor) -> float:
    x = predictor.float().flatten()
    y = response.float().flatten()
    xc = x - x.mean()
    yc = y - y.mean()
    denom = (xc * xc).sum()
    if denom < 1e-20:
        return 0.0
    return ((xc * yc).sum() / denom).item()


def explained_variance(predictor: torch.Tensor, response: torch.Tensor) -> float:
    x = predictor.float().flatten()
    y = response.float().flatten()
    if x.numel() < 2:
        return 0.0
    xc = x - x.mean()
    yc = y - y.mean()
    denom = (xc * xc).sum()
    if denom < 1e-20:
        return 0.0
    slope = (xc * yc).sum() / denom
    intercept = y.mean() - slope * x.mean()
    ss_res = ((y - (slope * x + intercept)) ** 2).sum()
    ss_tot = (yc**2).sum()
    if ss_tot < 1e-20:
        return 1.0
    return (1.0 - ss_res / ss_tot).item()


def safe_mean(lst: list[float]) -> float | None:
    return sum(lst) / len(lst) if lst else None


def safe_median(lst: list[float]) -> float | None:
    if not lst:
        return None
    s = sorted(lst)
    mid = len(s) // 2
    return (s[mid - 1] + s[mid]) / 2 if len(s) % 2 == 0 else s[mid]


def safe_std(lst: list[float]) -> float | None:
    if len(lst) < 2:
        return None
    m = sum(lst) / len(lst)
    return math.sqrt(sum((x - m) ** 2 for x in lst) / len(lst))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stacking provenance analysis")
    parser.add_argument("--base", required=True, help="Base model directory")
    parser.add_argument("--variant-a", required=True, help="Primary variant (e.g. original heretic)")
    parser.add_argument("--variant-b", required=True, help="Suspect variant (e.g. hauhau)")
    parser.add_argument("--label-a", default="variant_a")
    parser.add_argument("--label-b", default="variant_b")
    parser.add_argument("--extra-variants", nargs="*", default=[], help="Extra variant dirs (e.g. fresh heretics)")
    parser.add_argument("--extra-labels", nargs="*", default=[], help="Labels for extra variants")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    while len(args.extra_labels) < len(args.extra_variants):
        args.extra_labels.append(f"extra_{len(args.extra_labels)}")

    log.info("Detecting architecture...")
    cfg = detect_architecture(args.base)
    log.info("  Family: %s, Layers: %d", cfg.family, cfg.layer_count)

    log.info("Building shard maps...")
    base_map = build_shard_map(args.base, cfg)
    mapa = build_shard_map(args.variant_a, cfg)
    mapb = build_shard_map(args.variant_b, cfg)

    extra_maps: dict[str, dict] = {}
    for label, path in zip(args.extra_labels, args.extra_variants, strict=False):
        extra_maps[label] = build_shard_map(path, cfg)

    all_maps = {"base": base_map, args.label_a: mapa, args.label_b: mapb, **extra_maps}
    all_keys = sorted(set.intersection(*(set(m) for m in all_maps.values())))
    log.info("  Common keys: %d (across %d models)", len(all_keys), len(all_maps))

    per_tensor: list[dict] = []
    a_changed_keys: list[str] = []
    b_changed_keys: list[str] = []

    shared_cosines: list[float] = []
    ratio_on_shared: list[float] = []
    slopes_on_shared: list[float] = []
    r2_on_shared: list[float] = []
    ortho_on_shared: list[float] = []

    extra_cosines_vs_a: dict[str, list[float]] = defaultdict(list)
    extra_cosines_vs_b: dict[str, list[float]] = defaultdict(list)
    extra_dists_to_b: dict[str, list[float]] = defaultdict(list)
    extra_dists_to_a: dict[str, list[float]] = defaultdict(list)
    extra_stacking_ratios: dict[str, list[float]] = defaultdict(list)

    exact_match_count = 0
    near_exact_count = 0

    for i, ck in enumerate(all_keys):
        t_base = load_tensor(base_map, ck)
        t_a = load_tensor(mapa, ck)
        t_b = load_tensor(mapb, ck)
        if t_base is None or t_a is None or t_b is None:
            continue

        t_base = t_base.float()
        t_a = t_a.float()
        t_b = t_b.float()

        D_a = t_a - t_base
        D_b = t_b - t_base
        D_r = t_b - t_a

        norm_a = D_a.norm().item()
        norm_b = D_b.norm().item()
        norm_r = D_r.norm().item()

        a_changed = norm_a > THRESHOLD
        b_changed = norm_b > THRESHOLD

        layer = cfg.get_layer_index(ck)
        ttype = cfg.get_tensor_type(ck)

        entry: dict = {
            "key": ck,
            "layer": layer,
            "tensor_type": ttype,
            "numel": t_base.numel(),
            "norm_D_a": norm_a,
            "norm_D_b": norm_b,
            "norm_D_r": norm_r,
            "a_changed": a_changed,
            "b_changed": b_changed,
        }

        if norm_b > THRESHOLD:
            ratio = norm_r / norm_b
            entry["ratio_Dr_Db"] = ratio

        if a_changed:
            a_changed_keys.append(ck)
        if b_changed:
            b_changed_keys.append(ck)

        if a_changed and b_changed:
            cos_ab = cosine_sim(D_a, D_b)
            entry["cos_Da_Db"] = cos_ab
            shared_cosines.append(cos_ab)

            if norm_b > THRESHOLD:
                entry["ratio_Dr_Db"] = norm_r / norm_b
                ratio_on_shared.append(norm_r / norm_b)

            slope = regression_slope(D_a, D_b)
            r2 = explained_variance(D_a, D_b)
            entry["slope_Da_predicts_Db"] = slope
            entry["r2_Da_predicts_Db"] = r2
            slopes_on_shared.append(slope)
            r2_on_shared.append(r2)

            if norm_r > THRESHOLD:
                cos_ar = cosine_sim(D_a, D_r)
                entry["cos_Da_Dr"] = cos_ar
                ortho_on_shared.append(cos_ar)

            if norm_r < 1e-8:
                exact_match_count += 1
            if norm_b > THRESHOLD and norm_r / norm_b < 0.01:
                near_exact_count += 1

        for elabel in args.extra_labels:
            t_e = load_tensor(extra_maps[elabel], ck)
            if t_e is None:
                continue
            t_e = t_e.float()
            D_e = t_e - t_base
            norm_e = D_e.norm().item()

            if norm_e > THRESHOLD:
                if a_changed:
                    cos_ea = cosine_sim(D_e, D_a)
                    extra_cosines_vs_a[elabel].append(cos_ea)
                if b_changed:
                    cos_eb = cosine_sim(D_e, D_b)
                    extra_cosines_vs_b[elabel].append(cos_eb)

            dist_be = (t_b - t_e).norm().item()
            extra_dists_to_b[elabel].append(dist_be)
            dist_ae = (t_a - t_e).norm().item()
            extra_dists_to_a[elabel].append(dist_ae)

            if norm_b > THRESHOLD:
                extra_stacking_ratios[elabel].append(dist_be / norm_b)

            del t_e, D_e

        per_tensor.append(entry)
        del t_base, t_a, t_b, D_a, D_b, D_r
        if (i + 1) % 100 == 0:
            gc.collect()
            log.info("  processed %d/%d keys", i + 1, len(all_keys))

    log.info("Shared changed keys: %d", len(shared_cosines))
    log.info("Exact matches (||D_r|| < 1e-8): %d", exact_match_count)
    log.info("Near-exact (||D_r||/||D_b|| < 0.01): %d", near_exact_count)

    if shared_cosines:
        sc = safe_mean(shared_cosines)
        sm = safe_median(shared_cosines)
        log.info("cos(D_a, D_b): mean=%.4f, median=%.4f", sc, sm)
    if ratio_on_shared:
        log.info("||D_r||/||D_b||: mean=%.4f, median=%.4f", safe_mean(ratio_on_shared), safe_median(ratio_on_shared))
    if slopes_on_shared:
        log.info("Regression slope: mean=%.4f, median=%.4f", safe_mean(slopes_on_shared), safe_median(slopes_on_shared))

    report: dict = {
        "results_version": 1,
        "metadata": {},
        "architecture": {"family": cfg.family, "layers": cfg.layer_count},
        "variant_a": args.label_a,
        "variant_b": args.label_b,
        "extra_variants": args.extra_labels,
        "total_keys": len(all_keys),
        "a_changed_count": len(a_changed_keys),
        "b_changed_count": len(b_changed_keys),
        "shared_changed_count": len(shared_cosines),
        "stacking_metrics": {
            "cos_Da_Db": {
                "mean": safe_mean(shared_cosines),
                "median": safe_median(shared_cosines),
                "std": safe_std(shared_cosines),
                "count": len(shared_cosines),
                "min": min(shared_cosines) if shared_cosines else None,
                "max": max(shared_cosines) if shared_cosines else None,
            },
            "ratio_Dr_Db": {
                "mean": safe_mean(ratio_on_shared),
                "median": safe_median(ratio_on_shared),
                "std": safe_std(ratio_on_shared),
                "count": len(ratio_on_shared),
            },
            "slope": {
                "mean": safe_mean(slopes_on_shared),
                "median": safe_median(slopes_on_shared),
                "count": len(slopes_on_shared),
            },
            "r2": {"mean": safe_mean(r2_on_shared), "median": safe_median(r2_on_shared), "count": len(r2_on_shared)},
            "cos_Da_Dr": {
                "mean": safe_mean(ortho_on_shared),
                "median": safe_median(ortho_on_shared),
                "count": len(ortho_on_shared),
            },
            "exact_match_count": exact_match_count,
            "near_exact_count": near_exact_count,
        },
        "extra_variant_baselines": {},
        "per_tensor": per_tensor,
    }

    for elabel in args.extra_labels:
        ev: dict = {
            "cos_vs_a": {
                "mean": safe_mean(extra_cosines_vs_a.get(elabel, [])),
                "median": safe_median(extra_cosines_vs_a.get(elabel, [])),
                "count": len(extra_cosines_vs_a.get(elabel, [])),
            },
            "cos_vs_b": {
                "mean": safe_mean(extra_cosines_vs_b.get(elabel, [])),
                "median": safe_median(extra_cosines_vs_b.get(elabel, [])),
                "count": len(extra_cosines_vs_b.get(elabel, [])),
            },
            "dist_to_b_mean": safe_mean(extra_dists_to_b.get(elabel, [])),
            "dist_to_a_mean": safe_mean(extra_dists_to_a.get(elabel, [])),
            "stacking_ratio_vs_b": {
                "mean": safe_mean(extra_stacking_ratios.get(elabel, [])),
                "median": safe_median(extra_stacking_ratios.get(elabel, [])),
            },
        }
        report["extra_variant_baselines"][elabel] = ev
        log.info("  %s vs %s: cos=%.4f", elabel, args.label_b, ev["cos_vs_b"]["mean"] or 0)
        log.info("  %s vs %s: cos=%.4f", elabel, args.label_a, ev["cos_vs_a"]["mean"] or 0)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    log.info("Saved: %s", out)


if __name__ == "__main__":
    main()
