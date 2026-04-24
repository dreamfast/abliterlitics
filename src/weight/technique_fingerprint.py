#!/usr/bin/env python3
"""
Technique Fingerprinting -- build a signature for each abliteration method.

Analyzes: rank profile, magnitude distribution, layer targeting pattern,
tensor type preferences, edit density. The fingerprint can identify
techniques on unknown models.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
from collections import Counter
from pathlib import Path

from src.config import ComparisonConfig, make_metadata
from src.model_config import build_shard_map, detect_architecture, load_tensor

log = logging.getLogger(__name__)

THRESHOLD = 1e-10


def _layer_entropy(layer_counts: dict[int | None, int], total_layers: int) -> float:
    if not layer_counts or total_layers == 0:
        return 0
    total = sum(layer_counts.values())
    if total == 0:
        return 0
    entropy = 0.0
    for layer in range(total_layers):
        c = layer_counts.get(layer, 0)
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    max_entropy = math.log2(total_layers) if total_layers > 1 else 1
    return entropy / max_entropy * 100 if max_entropy > 0 else 0


def run_analysis(
    base_path: str,
    variant_path: str,
    label: str,
    output_path: str,
    config: ComparisonConfig | None = None,
    variant_name: str = "",
) -> None:
    log.info("Detecting architecture...")
    cfg = detect_architecture(base_path)
    log.info("  Family: %s, Layers: %d", cfg.family, cfg.layer_count)

    log.info("Building shard maps...")
    base_map = build_shard_map(base_path, cfg)
    var_map = build_shard_map(variant_path, cfg)
    all_keys = sorted(set(base_map) & set(var_map))
    log.info("  Common keys: %d", len(all_keys))

    changed_types: Counter = Counter()
    changed_layers: Counter = Counter()
    changed_categories: Counter = Counter()
    edit_norms: list[float] = []
    relative_edits: list[float] = []
    per_key: list[dict] = []

    for i, ck in enumerate(all_keys):
        t_base = load_tensor(base_map, ck).float()
        t_var = load_tensor(var_map, ck).float()
        delta = t_var - t_base
        norm = delta.norm().item()
        base_norm = t_base.norm().item()
        rel = norm / base_norm if base_norm > 0 else 0
        changed = norm > THRESHOLD

        ttype = cfg.get_tensor_type(ck)
        layer = cfg.get_layer_index(ck)
        category = cfg.tensor_category(ck)

        entry: dict = {
            "key": ck,
            "type": ttype,
            "layer": layer,
            "category": category,
            "edit_norm": norm,
            "base_norm": base_norm,
            "rel_norm": rel,
            "changed": changed,
            "numel": t_base.numel(),
        }

        if changed:
            changed_types[ttype] += 1
            if layer is not None:
                changed_layers[layer] += 1
            changed_categories[category] += 1
            edit_norms.append(norm)
            relative_edits.append(rel)

        per_key.append(entry)
        del t_base, t_var, delta
        if (i + 1) % 200 == 0:
            gc.collect()

    total_changed = len(edit_norms)
    total_params = sum(e["numel"] for e in per_key if e["changed"])
    total_all_params = sum(e["numel"] for e in per_key)

    def _safe_mean(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0

    def _safe_median(lst: list[float]) -> float:
        if not lst:
            return 0
        s = sorted(lst)
        mid = len(s) // 2
        return (s[mid - 1] + s[mid]) / 2 if len(s) % 2 == 0 else s[mid]

    def _percentile(lst: list[float], p: float) -> float:
        if not lst:
            return 0
        s = sorted(lst)
        idx = int(len(s) * p / 100)
        return s[min(idx, len(s) - 1)]

    fingerprint: dict = {}
    if config is not None:
        fingerprint["metadata"] = make_metadata(config, variant=variant_name)
    fingerprint["results_version"] = 1
    fingerprint.update(
        {
            "label": label,
            "architecture": {
                "family": cfg.family,
                "layers": cfg.layer_count,
                "total_keys": cfg.total_keys,
                "experts": cfg.expert_count,
            },
            "scope": {
                "total_tensors": len(all_keys),
                "changed_tensors": total_changed,
                "changed_pct": total_changed / len(all_keys) * 100 if all_keys else 0,
                "total_params_changed": total_params,
                "param_edit_density": total_params / total_all_params * 100 if total_all_params else 0,
            },
            "magnitude": {
                "edit_norm_mean": _safe_mean(edit_norms),
                "edit_norm_median": _safe_median(edit_norms),
                "edit_norm_p25": _percentile(edit_norms, 25),
                "edit_norm_p75": _percentile(edit_norms, 75),
                "edit_norm_p95": _percentile(edit_norms, 95),
                "relative_edit_mean": _safe_mean(relative_edits),
                "relative_edit_median": _safe_median(relative_edits),
            },
            "targeting": {
                "tensor_types": dict(changed_types.most_common()),
                "categories": dict(changed_categories.most_common()),
                "layers_modified": len(changed_layers),
                "layer_coverage_pct": len(changed_layers) / cfg.layer_count * 100 if cfg.layer_count else 0,
            },
            "layer_profile": {
                "mean_edits_per_layer": _safe_mean(list(changed_layers.values())) if changed_layers else 0,
                "max_edits_layer": max(changed_layers.values()) if changed_layers else 0,
                "layer_entropy": _layer_entropy(changed_layers, cfg.layer_count),
                "early_layer_pct": sum(
                    v for k, v in changed_layers.items() if k is not None and k < cfg.layer_count // 3
                )
                / max(total_changed, 1)
                * 100,
                "mid_layer_pct": sum(
                    v
                    for k, v in changed_layers.items()
                    if k is not None and cfg.layer_count // 3 <= k < 2 * cfg.layer_count // 3
                )
                / max(total_changed, 1)
                * 100,
                "late_layer_pct": sum(
                    v for k, v in changed_layers.items() if k is not None and k >= 2 * cfg.layer_count // 3
                )
                / max(total_changed, 1)
                * 100,
            },
            "per_key": per_key,
        }
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fingerprint, indent=2))
    log.info("Saved: %s", out)

    log.info("=== TECHNIQUE FINGERPRINT: %s ===", label)
    log.info("  Scope: %d/%d tensors (%.1f%%)", total_changed, len(all_keys), fingerprint["scope"]["changed_pct"])
    log.info("  Param density: %.2f%%", fingerprint["scope"]["param_edit_density"])
    log.info("  Mean edit norm: %.4f", fingerprint["magnitude"]["edit_norm_mean"])
    log.info("  Mean relative edit: %.6f", fingerprint["magnitude"]["relative_edit_mean"])
    log.info(
        "  Layer coverage: %d/%d (%.1f%%)",
        fingerprint["targeting"]["layers_modified"],
        cfg.layer_count,
        fingerprint["targeting"]["layer_coverage_pct"],
    )
    log.info(
        "  Depth profile: early=%.1f%% mid=%.1f%% late=%.1f%%",
        fingerprint["layer_profile"]["early_layer_pct"],
        fingerprint["layer_profile"]["mid_layer_pct"],
        fingerprint["layer_profile"]["late_layer_pct"],
    )
    log.info("  Top types: %s", changed_types.most_common(5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", help="Path to comparison.json dir (batch mode)")
    parser.add_argument("--base", help="Base model dir")
    parser.add_argument("--variant", help="Variant model dir")
    parser.add_argument("--label", help="Variant label")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--results-dir", help="Base results directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.comparison:
        config = ComparisonConfig.from_dir(Path(args.comparison))
        results_dir = Path(args.results_dir) if args.results_dir else config.weight_results_dir(Path("results"))
        for variant in config.variants:
            out_path = results_dir / variant.name / f"fingerprint_{variant.name}.json"
            run_analysis(
                base_path=str(config.base_path),
                variant_path=str(variant.path),
                label=variant.display_name,
                output_path=str(out_path),
                config=config,
                variant_name=variant.name,
            )
    else:
        if not all([args.base, args.variant, args.label, args.output]):
            parser.error("--base, --variant, --label, and --output are required without --comparison")
        run_analysis(
            base_path=args.base,
            variant_path=args.variant,
            label=args.label,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
