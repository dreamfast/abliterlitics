#!/usr/bin/env python3
"""
Combined: Edit Magnitude Progression + Edit Density + Norm Shift + Inter-Tensor Correlation.

Layer-wise analysis of abliteration patterns per technique:
- Edit magnitude as function of layer depth
- Fraction of parameters meaningfully changed per layer (density)
- Whether edits increase or decrease weight norms (projection vs fine-tuning)
- Correlation between edits to different tensor types within the same layer
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from collections import defaultdict
from pathlib import Path

from src.config import ComparisonConfig, make_metadata
from src.model_config import build_shard_map, detect_architecture, load_tensor

log = logging.getLogger(__name__)

THRESHOLD = 1e-10


def run_analysis(
    base_path: str,
    variant_path: str,
    variant_b_path: str | None,
    label: str,
    label_b: str | None,
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
    varb_map = build_shard_map(variant_b_path, cfg) if variant_b_path else None
    all_keys = sorted(set(base_map) & set(var_map))
    log.info("  Common keys: %d", len(all_keys))

    layer_data: dict[int, dict] = defaultdict(
        lambda: {
            "tensors": [],
            "edit_norms": [],
            "base_norms": [],
            "rel_norms": [],
            "norm_shifts": [],
            "types": defaultdict(list),
        }
    )
    if varb_map:
        layer_data_b: dict[int, dict] = defaultdict(
            lambda: {
                "tensors": [],
                "edit_norms": [],
                "base_norms": [],
                "rel_norms": [],
                "norm_shifts": [],
                "types": defaultdict(list),
            }
        )

    layer_type_deltas: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for i, ck in enumerate(all_keys):
        layer = cfg.get_layer_index(ck)
        if layer is None:
            continue

        t_base = load_tensor(base_map, ck)
        t_var = load_tensor(var_map, ck)
        if t_base is None or t_var is None:
            continue
        t_base = t_base.float()
        t_var = t_var.float()
        delta = t_var - t_base
        edit_norm = delta.norm().item()
        base_norm = t_base.norm().item()
        var_norm = t_var.norm().item()
        ttype = cfg.get_tensor_type(ck)

        ld = layer_data[layer]
        ld["tensors"].append(ck)
        ld["edit_norms"].append(edit_norm)
        ld["base_norms"].append(base_norm)
        ld["rel_norms"].append(edit_norm / base_norm if base_norm > 0 else 0)
        ld["norm_shifts"].append((var_norm - base_norm) / base_norm if base_norm > 0 else 0)
        ld["types"][ttype].append(edit_norm)

        if edit_norm > THRESHOLD:
            layer_type_deltas[layer][ttype].append(edit_norm)

        if varb_map:
            t_varb = load_tensor(varb_map, ck)
            if t_varb is not None:
                t_varb = t_varb.float()
                delta_b = t_varb - t_base
                edit_norm_b = delta_b.norm().item()
                var_norm_b = t_varb.norm().item()
                ldb = layer_data_b[layer]
                ldb["tensors"].append(ck)
                ldb["edit_norms"].append(edit_norm_b)
                ldb["base_norms"].append(base_norm)
                ldb["rel_norms"].append(edit_norm_b / base_norm if base_norm > 0 else 0)
                ldb["norm_shifts"].append((var_norm_b - base_norm) / base_norm if base_norm > 0 else 0)
                ldb["types"][ttype].append(edit_norm_b)
                del t_varb, delta_b

        del t_base, t_var, delta
        if (i + 1) % 200 == 0:
            gc.collect()

    progression: dict[str, dict] = {}
    for layer in sorted(layer_data):
        ld = layer_data[layer]
        total_tensors = len(ld["tensors"])
        changed = sum(1 for n in ld["edit_norms"] if n > THRESHOLD)
        mean_edit = sum(ld["edit_norms"]) / len(ld["edit_norms"]) if ld["edit_norms"] else 0
        mean_rel = sum(ld["rel_norms"]) / len(ld["rel_norms"]) if ld["rel_norms"] else 0
        mean_shift = sum(ld["norm_shifts"]) / len(ld["norm_shifts"]) if ld["norm_shifts"] else 0

        type_edits = {t: sum(v) / len(v) for t, v in ld["types"].items() if v}

        types_list = list(ld["types"].keys())
        inter_corr: dict[str, dict] = {}
        if len(types_list) >= 2:
            type_norms = layer_type_deltas.get(layer, {})
            for ti in range(len(types_list)):
                for tj in range(ti + 1, len(types_list)):
                    ta, tb = types_list[ti], types_list[tj]
                    norms_a = type_norms.get(ta, [])
                    norms_b = type_norms.get(tb, [])
                    inter_corr[f"{ta}_vs_{tb}"] = {
                        "both_edited": bool(norms_a and norms_b),
                        "a_count": len(norms_a),
                        "b_count": len(norms_b),
                        "a_mean_norm": sum(norms_a) / len(norms_a) if norms_a else 0,
                        "b_mean_norm": sum(norms_b) / len(norms_b) if norms_b else 0,
                    }

        progression[str(layer)] = {
            "total_tensors": total_tensors,
            "changed_tensors": changed,
            "edit_density": changed / total_tensors * 100 if total_tensors else 0,
            "mean_edit_norm": mean_edit,
            "mean_relative_edit": mean_rel,
            "mean_norm_shift": mean_shift,
            "norm_shift_direction": "reduction"
            if mean_shift < -0.001
            else "increase"
            if mean_shift > 0.001
            else "neutral",
            "type_edits": type_edits,
            "inter_tensor_correlation": inter_corr if inter_corr else None,
        }

    report: dict = {}
    if config is not None:
        report["metadata"] = make_metadata(config, variant=variant_name)
    report["results_version"] = 1
    report.update(
        {
            "architecture": {"family": cfg.family, "layers": cfg.layer_count},
            "variant": label,
            "layer_progression": progression,
        }
    )

    if varb_map:
        progression_b: dict[str, dict] = {}
        for layer in sorted(layer_data_b):
            ld = layer_data_b[layer]
            changed = sum(1 for n in ld["edit_norms"] if n > THRESHOLD)
            total = len(ld["tensors"])
            progression_b[str(layer)] = {
                "total_tensors": total,
                "changed_tensors": changed,
                "edit_density": changed / total * 100 if total else 0,
                "mean_edit_norm": sum(ld["edit_norms"]) / len(ld["edit_norms"]) if ld["edit_norms"] else 0,
                "mean_norm_shift": sum(ld["norm_shifts"]) / len(ld["norm_shifts"]) if ld["norm_shifts"] else 0,
            }
        report["variant_b"] = label_b
        report["layer_progression_b"] = progression_b

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    log.info("Saved: %s", out)

    log.info("=== LAYER ANALYSIS: %s ===", label)
    for lstr in sorted(progression, key=lambda x: int(x)):
        d = progression[lstr]
        log.info(
            "L%s  changed=%d density=%.1f mean_edit=%.4f shift=%.6f %s",
            lstr,
            d["changed_tensors"],
            d["edit_density"],
            d["mean_edit_norm"],
            d["mean_norm_shift"],
            d["norm_shift_direction"],
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", help="Path to comparison.json dir (batch mode)")
    parser.add_argument("--base", help="Base model dir")
    parser.add_argument("--variant", help="Variant model dir")
    parser.add_argument("--variant-b", default=None, help="Second variant dir")
    parser.add_argument("--label", default="variant")
    parser.add_argument("--label-b", default=None)
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--results-dir", help="Base results directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.comparison:
        config = ComparisonConfig.from_dir(Path(args.comparison))
        results_dir = Path(args.results_dir) if args.results_dir else config.weight_results_dir(Path("results"))
        for variant in config.variants:
            out_path = results_dir / variant.name / f"layer_analysis_{variant.name}.json"
            run_analysis(
                base_path=str(config.base_path),
                variant_path=str(variant.path),
                variant_b_path=args.variant_b,
                label=variant.display_name,
                label_b=args.label_b,
                output_path=str(out_path),
                config=config,
                variant_name=variant.name,
            )
    else:
        if not all([args.base, args.variant, args.output]):
            parser.error("--base, --variant, and --output are required without --comparison")
        run_analysis(
            base_path=args.base,
            variant_path=args.variant,
            variant_b_path=args.variant_b,
            label=args.label,
            label_b=args.label_b,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
