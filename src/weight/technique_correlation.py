#!/usr/bin/env python3
"""
Cross-technique correlation analysis -- architecture-agnostic.

Compares edit vectors across multiple techniques to test:
1. Do different abliteration techniques edit in similar directions?
2. Is any technique's edit a superset of another?
3. Can we detect model merging signatures?
4. Establish independent-edit baselines for comparison.

In --comparison mode, computes all pairwise comparisons across N variants
(read from comparison.json). Each pair is written to a separate output file.
Without --comparison, compares exactly two variants specified via CLI.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import torch

from src.config import ComparisonConfig, make_metadata
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


def project_out(vec: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    d = direction.float().flatten()
    v = vec.float().flatten()
    denom = (d * d).sum()
    if denom < 1e-20:
        return vec.float()
    proj = (v * d).sum() / denom
    return (v - proj * d).reshape(vec.shape)


def safe_mean(lst: list[float]) -> float | None:
    return sum(lst) / len(lst) if lst else None


def safe_median(lst: list[float]) -> float | None:
    if not lst:
        return None
    s = sorted(lst)
    mid = len(s) // 2
    return (s[mid - 1] + s[mid]) / 2 if len(s) % 2 == 0 else s[mid]


def run_analysis(
    base_path: str,
    variant_a_path: str,
    variant_b_path: str,
    label_a: str,
    label_b: str,
    output_path: str,
    config: ComparisonConfig | None = None,
    pair_label: str = "",
) -> None:
    log.info("Detecting architecture...")
    cfg = detect_architecture(base_path)
    log.info("  Family: %s, Layers: %d", cfg.family, cfg.layer_count)

    log.info("Building shard maps...")
    base_map = build_shard_map(base_path, cfg)
    mapa = build_shard_map(variant_a_path, cfg)
    mapb = build_shard_map(variant_b_path, cfg)

    all_keys = sorted(set(base_map) & set(mapa) & set(mapb))
    log.info("  Common keys: %d", len(all_keys))

    cosines: list[float] = []
    per_key: list[dict] = []

    for i, ck in enumerate(all_keys):
        t_base = load_tensor(base_map, ck)
        t_a = load_tensor(mapa, ck)
        t_b = load_tensor(mapb, ck)
        if t_base is None or t_a is None or t_b is None:
            continue

        t_base = t_base.float()
        t_a = t_a.float()
        t_b = t_b.float()
        delta_a = t_a - t_base
        delta_b = t_b - t_base

        norm_a = delta_a.norm().item()
        norm_b = delta_b.norm().item()
        a_changed = norm_a > THRESHOLD
        b_changed = norm_b > THRESHOLD

        entry: dict = {
            "key": ck,
            "tensor_type": cfg.get_tensor_type(ck),
            "layer": cfg.get_layer_index(ck),
            "category": cfg.tensor_category(ck),
            f"{label_a}_edit_norm": norm_a,
            f"{label_b}_edit_norm": norm_b,
            f"{label_a}_changed": a_changed,
            f"{label_b}_changed": b_changed,
        }

        if a_changed and b_changed:
            cos = cosine_sim(delta_a, delta_b)
            entry[f"cos_{label_a}_vs_{label_b}"] = cos
            cosines.append(cos)

            # Residual analysis: how much of B remains after removing A's direction?
            res = project_out(delta_b, delta_a)
            energy_b = norm_b**2
            if energy_b > 1e-20:
                entry[f"residual_{label_b}_after_removing_{label_a}_pct"] = (
                    res.norm().item() ** 2 / energy_b * 100
                )

        per_key.append(entry)
        del t_base, t_a, t_b, delta_a, delta_b
        if (i + 1) % 200 == 0:
            gc.collect()
            log.info("  processed %d/%d", i + 1, len(all_keys))

    by_type: dict[str, list[dict]] = {}
    for r in per_key:
        tt = r["tensor_type"]
        by_type.setdefault(tt, []).append(r)

    cos_key = f"cos_{label_a}_vs_{label_b}"
    type_summary: dict[str, dict] = {}
    for tt in sorted(by_type):
        group = by_type[tt]
        ts: dict = {"count": len(group)}
        vals = [r.get(cos_key) for r in group if cos_key in r]
        ts[f"mean_{cos_key}"] = safe_mean(vals)
        type_summary[tt] = ts

    report: dict = {}
    if config is not None:
        report["metadata"] = make_metadata(config, variant=pair_label)
    report["results_version"] = 1
    report.update(
        {
            "architecture": {"family": cfg.family, "layers": cfg.layer_count},
            "variants": [label_a, label_b],
            "total_keys": len(all_keys),
            "pairwise_cosines": {
                f"{label_a}_vs_{label_b}": {
                    "mean": safe_mean(cosines),
                    "median": safe_median(cosines),
                    "count": len(cosines),
                    "min": min(cosines) if cosines else None,
                    "max": max(cosines) if cosines else None,
                }
            },
            "type_summary": type_summary,
            "per_key_details": per_key,
        }
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    log.info("Saved: %s", out)

    log.info("=== TECHNIQUE CORRELATION SUMMARY ===")
    if cosines:
        m = safe_mean(cosines)
        med = safe_median(cosines)
        log.info(
            "  cos(%s vs %s): mean=%.4f, median=%.4f (n=%d)",
            label_a, label_b, m, med, len(cosines),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", help="Path to comparison.json dir (batch mode)")
    parser.add_argument("--base", help="Base model dir")
    parser.add_argument("--variant-a", help="First variant dir")
    parser.add_argument("--variant-b", help="Second variant dir")
    parser.add_argument("--label-a", default="a")
    parser.add_argument("--label-b", default="b")
    parser.add_argument("--output", help="Output JSON path (single-pair mode only; ignored with --comparison)")
    parser.add_argument("--results-dir", help="Override output directory (used with --comparison)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.comparison:
        config = ComparisonConfig.from_dir(Path(args.comparison))
        # Use --output's parent directory when available (runner mounts /results),
        # otherwise fall back to config-derived path.
        results_dir = (
            Path(args.output).parent
            if args.output
            else (Path(args.results_dir) if args.results_dir else config.weight_results_dir(Path("results")))
        )
        if len(config.variants) < 2:
            parser.error("comparison.json must have at least 2 variants for pairwise analysis")
        for i in range(len(config.variants)):
            for j in range(i + 1, len(config.variants)):
                v1, v2 = config.variants[i], config.variants[j]
                out_path = results_dir / f"correlation_{v1.name}_vs_{v2.name}.json"
                if out_path.exists():
                    log.info("Skipping (exists): %s", out_path.name)
                    continue
                log.info("Pair: %s vs %s", v1.display_name, v2.display_name)
                run_analysis(
                    base_path=str(config.base_path),
                    variant_a_path=str(v1.path),
                    variant_b_path=str(v2.path),
                    label_a=v1.display_name,
                    label_b=v2.display_name,
                    output_path=str(out_path),
                    config=config,
                    pair_label=f"{v1.name}_vs_{v2.name}",
                )
    else:
        if not all([args.base, args.variant_a, args.variant_b, args.output]):
            parser.error("--base, --variant-a, --variant-b, and --output are required without --comparison")
        run_analysis(
            base_path=args.base,
            variant_a_path=args.variant_a,
            variant_b_path=args.variant_b,
            label_a=args.label_a,
            label_b=args.label_b,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
