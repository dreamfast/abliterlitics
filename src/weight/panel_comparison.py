#!/usr/bin/env python3
"""
Streaming multi-model panel comparison -- architecture-agnostic.

Compares any number of model variants pairwise using lazy tensor loading.
Auto-detects architecture from safetensors keys. Variants are specified
via comparison.json (any number, any names) or explicit CLI flags.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
from itertools import combinations
from pathlib import Path

from src.config import ComparisonConfig, make_metadata
from src.model_config import (
    ArchitectureConfig,
    build_shard_map,
    detect_architecture,
    load_tensor,
)

log = logging.getLogger(__name__)


def changed_keys_lazy(
    map_a: dict[str, tuple[str, str]],
    map_b: dict[str, tuple[str, str]],
    cfg: ArchitectureConfig,
    label: str,
) -> list[dict]:
    common = sorted(set(map_a) & set(map_b))
    log.info("  %s: %d common canonical keys", label, len(common))
    changed: list[dict] = []
    for i, ck in enumerate(common):
        ta = load_tensor(map_a, ck)
        tb = load_tensor(map_b, ck)
        if ta is None or tb is None:
            continue
        if ta.shape != tb.shape:
            changed.append({"canonical": ck, "mean_abs_diff": float("inf")})
        else:
            diff = (tb.float() - ta.float()).abs().mean().item()
            if diff > 0:
                changed.append({"canonical": ck, "mean_abs_diff": diff})
        del ta, tb
        if (i + 1) % 200 == 0:
            gc.collect()
            log.info("    processed %d/%d", i + 1, len(common))
    changed.sort(key=lambda x: x["mean_abs_diff"], reverse=True)
    return changed


def run_analysis(
    base_path: str,
    variant_paths: dict[str, str],
    output_dir: Path,
    config: ComparisonConfig | None = None,
) -> None:
    """Run pairwise panel comparison.

    Args:
        base_path: Path to the base (unmodified) model.
        variant_paths: Dict mapping variant name -> model path.
                       e.g. {"heretic": "/models/heretic", "abliterix": "/models/abliterix"}
        output_dir: Directory for output JSON and CSV files.
        config: Optional comparison config for metadata.
    """
    log.info("Detecting architecture from base model...")
    cfg = detect_architecture(base_path)
    log.info(
        "  Family: %s, Layers: %d, Keys: %d, Experts: %d, Mamba: %s",
        cfg.family,
        cfg.layer_count,
        cfg.total_keys,
        cfg.expert_count,
        cfg.has_mamba,
    )

    log.info("Building shard maps...")
    base_map = build_shard_map(base_path, cfg)
    log.info("  base: %d keys", len(base_map))

    models: dict[str, dict] = {"base": base_map}
    for vname, vpath in variant_paths.items():
        vmap = build_shard_map(vpath, cfg)
        models[vname] = vmap
        log.info("  %s: %d keys", vname, len(vmap))

    # Generate all pairwise combinations
    model_names = sorted(models.keys())
    pairs = list(combinations(model_names, 2))

    pairwise: dict[str, list[dict]] = {}
    for name_a, name_b in pairs:
        label = f"{name_a}->{name_b}"
        log.info("Computing %s deltas...", label)
        changed = changed_keys_lazy(models[name_a], models[name_b], cfg, label)
        log.info("  Changed: %d", len(changed))
        pairwise[label] = changed

    changed_sets: dict[str, set[str]] = {}
    for label, items in pairwise.items():
        changed_sets[label] = set(x["canonical"] for x in items)

    report: dict = {}

    if config is not None:
        report["metadata"] = make_metadata(config)

    report.update(
        {
            "results_version": 1,
            "architecture": {
                "family": cfg.family,
                "layer_count": cfg.layer_count,
                "total_keys": cfg.total_keys,
                "expert_count": cfg.expert_count,
                "has_mamba": cfg.has_mamba,
                "has_experts": cfg.has_experts,
                "has_shared_experts": cfg.has_shared_experts,
            },
            "model_key_counts": {name: len(m) for name, m in models.items()},
            "pairwise_changed_counts": {label: len(items) for label, items in pairwise.items()},
        }
    )

    for label, keys in changed_sets.items():
        report[f"{label}_keys"] = sorted(keys)

    # Compute overlap for every pair of base->variant comparisons
    base_variant_labels = [l for l in changed_sets if l.startswith("base->")]
    for label_a, label_b in combinations(base_variant_labels, 2):
        va_name = label_a.removeprefix("base->")
        vb_name = label_b.removeprefix("base->")
        va_keys = changed_sets[label_a]
        vb_keys = changed_sets[label_b]
        overlap = va_keys & vb_keys
        overlap_key = f"{va_name}_{vb_name}_overlap"
        report[overlap_key] = {
            f"{va_name}_count": len(va_keys),
            f"{vb_name}_count": len(vb_keys),
            "overlap_count": len(overlap),
            f"{va_name}_only_count": len(va_keys - vb_keys),
            f"{vb_name}_only_count": len(vb_keys - va_keys),
            "overlap_pct_of_smaller": (
                len(overlap) / min(len(va_keys), len(vb_keys)) * 100 if va_keys and vb_keys else 0
            ),
        }

    per_layer: dict[str, dict[int, dict]] = {}
    for label, items in pairwise.items():
        layer_data: dict[int, dict] = {}
        for item in items:
            ck = item["canonical"]
            layer_idx = cfg.get_layer_index(ck)
            if layer_idx is None:
                continue
            ttype = cfg.get_tensor_type(ck)
            if layer_idx not in layer_data:
                layer_data[layer_idx] = {}
            layer_data[layer_idx][ttype] = item["mean_abs_diff"]
        per_layer[label] = dict(sorted(layer_data.items()))

    report["per_layer_changes"] = {}
    for label, layers in per_layer.items():
        report["per_layer_changes"][label] = {
            str(k): {"count": len(v), "types": sorted(v.keys())} for k, v in sorted(layers.items())
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "multi_model_panel.json").write_text(json.dumps(report, indent=2))
    log.info("Saved: %s", output_dir / "multi_model_panel.json")

    for label, items in pairwise.items():
        csv_path = output_dir / f"{label.replace('->', '_to_')}_keys.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tensor", "mean_abs_diff"])
            for item in items:
                w.writerow([item["canonical"], item["mean_abs_diff"]])
        log.info("Saved: %s", csv_path)

    log.info("=== SUMMARY ===")
    for label, items in pairwise.items():
        log.info("  %s: %d changed tensors", label, len(items))
    for key in sorted(k for k in report if k.endswith("_overlap")):
        o = report[key]
        names = key.replace("_overlap", "").split("_")
        log.info(
            "  %s/%s overlap: %d (%.1f%% of smaller)",
            names[0],
            names[1],
            o["overlap_count"],
            o["overlap_pct_of_smaller"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pairwise panel comparison of model variants",
    )
    parser.add_argument("--comparison", help="Path to comparison.json dir (batch mode)")
    parser.add_argument("--base", help="Base model dir (overrides --comparison)")
    parser.add_argument("--variant", action="append", metavar="NAME=PATH",
                        help="Variant as name=path (repeatable, e.g. --variant heretic=/models/heretic)")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--results-dir", help="Base results directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.comparison:
        config = ComparisonConfig.from_dir(Path(args.comparison))
        results_dir = Path(args.results_dir) if args.results_dir else config.weight_results_dir(Path("results"))
        results_dir.mkdir(parents=True, exist_ok=True)

        if len(config.variants) < 2:
            log.error("panel_comparison requires at least 2 variants in comparison.json")
            return

        variant_paths = {v.name: str(v.path) for v in config.variants}

        run_analysis(
            base_path=str(config.base_path),
            variant_paths=variant_paths,
            output_dir=results_dir,
            config=config,
        )
    else:
        if not args.base or not args.variant or not args.output:
            parser.error("--base, at least two --variant, and --output are required without --comparison")

        variant_paths: dict[str, str] = {}
        for vspec in args.variant:
            if "=" not in vspec:
                parser.error(f"--variant must be NAME=PATH, got: {vspec}")
            name, path = vspec.split("=", 1)
            variant_paths[name] = path

        if len(variant_paths) < 2:
            parser.error("At least 2 --variant entries required for pairwise comparison")

        run_analysis(
            base_path=args.base,
            variant_paths=variant_paths,
            output_dir=Path(args.output),
        )


if __name__ == "__main__":
    main()
