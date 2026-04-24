#!/usr/bin/env python3
"""
GLM MoE Expert-Specific Analysis.

Analyzes which of the 64 MoE experts get modified by each abliteration technique.
Identifies refusal experts, expert clustering patterns, and router modification.
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
    variant_a_path: str,
    variant_b_path: str | None,
    variant_c_path: str | None,
    label_a: str,
    label_b: str,
    label_c: str,
    output_path: str,
    config: ComparisonConfig | None = None,
    variant_name: str = "",
) -> None:
    log.info("Detecting architecture...")
    cfg = detect_architecture(base_path)
    if not cfg.has_experts:
        log.info("Model has no MoE experts. Nothing to analyze.")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps({"error": "No MoE experts found"}))
        return

    log.info("  Experts: %d, Shared: %s", cfg.expert_count, cfg.has_shared_experts)

    log.info("Building shard maps...")
    base_map = build_shard_map(base_path, cfg)
    variants: dict[str, dict] = {label_a: build_shard_map(variant_a_path, cfg)}
    labels = [label_a]
    if variant_b_path:
        variants[label_b] = build_shard_map(variant_b_path, cfg)
        labels.append(label_b)
    if variant_c_path:
        variants[label_c] = build_shard_map(variant_c_path, cfg)
        labels.append(label_c)

    all_keys = sorted(set(base_map) & set.intersection(*(set(v) for v in variants.values())))

    expert_keys = [k for k in all_keys if cfg.is_expert_tensor(k)]
    shared_keys = [k for k in all_keys if cfg.is_shared_expert_tensor(k)]
    router_keys = [k for k in all_keys if cfg.is_router_tensor(k)]
    other_keys = [
        k
        for k in all_keys
        if not cfg.is_expert_tensor(k) and not cfg.is_shared_expert_tensor(k) and not cfg.is_router_tensor(k)
    ]

    log.info(
        "  Expert tensors: %d, Shared: %d, Router: %d, Other: %d",
        len(expert_keys),
        len(shared_keys),
        len(router_keys),
        len(other_keys),
    )

    expert_edits: dict[str, dict[tuple[int, int], list[dict]]] = {label: defaultdict(list) for label in labels}
    expert_norms: dict[str, dict[tuple[int, int], float]] = {label: defaultdict(float) for label in labels}
    per_expert: dict[str, dict[str, dict]] = {label: {} for label in labels}

    for i, ck in enumerate(expert_keys):
        layer = cfg.get_layer_index(ck)
        eid = cfg.get_expert_id(ck)
        ttype = cfg.get_tensor_type(ck)
        t_base = load_tensor(base_map, ck)
        if t_base is None:
            continue
        t_base = t_base.float()

        for label in labels:
            t_var = load_tensor(variants[label], ck)
            if t_var is None:
                continue
            t_var = t_var.float()
            delta = t_var - t_base
            norm = delta.norm().item()
            rel_norm = norm / t_base.norm().item() if t_base.norm().item() > 0 else 0

            expert_edits[label][(layer, eid)].append(
                {
                    "type": ttype,
                    "abs_norm": norm,
                    "rel_norm": rel_norm,
                    "changed": norm > THRESHOLD,
                }
            )
            expert_norms[label][(layer, eid)] += norm

            del t_var, delta
        del t_base

        if (i + 1) % 500 == 0:
            gc.collect()
            log.info("  processed %d/%d expert tensors", i + 1, len(expert_keys))

    router_edits: dict[int, dict] = {}
    for ck in router_keys:
        layer = cfg.get_layer_index(ck)
        t_base = load_tensor(base_map, ck)
        if t_base is None:
            continue
        t_base = t_base.float()
        for label in labels:
            t_var = load_tensor(variants[label], ck)
            if t_var is None:
                continue
            t_var = t_var.float()
            delta = t_var - t_base
            norm = delta.norm().item()
            router_edits.setdefault(layer, {})[label] = {
                "key": ck,
                "type": cfg.get_tensor_type(ck),
                "edit_norm": norm,
                "base_norm": t_base.norm().item(),
                "rel_norm": norm / t_base.norm().item() if t_base.norm().item() > 0 else 0,
                "changed": norm > THRESHOLD,
            }
            del t_var, delta
        del t_base

    shared_edits: dict[int, dict] = {}
    for ck in shared_keys:
        layer = cfg.get_layer_index(ck)
        t_base = load_tensor(base_map, ck)
        if t_base is None:
            continue
        t_base = t_base.float()
        for label in labels:
            t_var = load_tensor(variants[label], ck)
            if t_var is None:
                continue
            t_var = t_var.float()
            delta = t_var - t_base
            norm = delta.norm().item()
            shared_edits.setdefault(layer, {})[label] = {
                "key": ck,
                "type": cfg.get_tensor_type(ck),
                "edit_norm": norm,
                "rel_norm": norm / t_base.norm().item() if t_base.norm().item() > 0 else 0,
                "changed": norm > THRESHOLD,
            }
            del t_var, delta
        del t_base

    for label in labels:
        for (layer, eid), edits in sorted(expert_edits[label].items()):
            total_norm = expert_norms[label][(layer, eid)]
            changed_count = sum(1 for e in edits if e["changed"])
            if changed_count > 0:
                per_expert[label][f"L{layer}_E{eid}"] = {
                    "layer": layer,
                    "expert_id": eid,
                    "total_edit_norm": total_norm,
                    "tensors_changed": changed_count,
                    "tensors_total": len(edits),
                    "details": edits,
                }

    changed_experts_per_layer: dict[str, dict[int, int]] = {}
    for label in labels:
        layer_counts: dict[int, int] = defaultdict(int)
        for _key, info in per_expert[label].items():
            layer_counts[info["layer"]] += 1
        changed_experts_per_layer[label] = dict(sorted(layer_counts.items()))

    report: dict = {}
    if config is not None:
        report["metadata"] = make_metadata(config, variant=variant_name)
    report["results_version"] = 1
    report.update(
        {
            "architecture": {
                "family": cfg.family,
                "experts": cfg.expert_count,
                "shared_experts": cfg.has_shared_experts,
                "layers": cfg.layer_count,
            },
            "variants": labels,
            "total_expert_tensors": len(expert_keys),
            "expert_edits_summary": {
                label: {
                    "changed_expert_layer_pairs": len(per_expert[label]),
                    "total_edit_energy": sum(v["total_edit_norm"] for v in per_expert[label].values()),
                }
                for label in labels
            },
            "changed_experts_per_layer": changed_experts_per_layer,
            "router_edits": router_edits,
            "shared_expert_edits": shared_edits,
            "per_expert_details": {label: dict(sorted(v.items())) for label, v in per_expert.items()},
        }
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    log.info("Saved: %s", out)

    log.info("=== EXPERT ANALYSIS SUMMARY ===")
    for label in labels:
        n = len(per_expert[label])
        energy = sum(v["total_edit_norm"] for v in per_expert[label].values())
        log.info("  %s: %d expert-layer pairs changed, total energy=%.2f", label, n, energy)
        router_changed = sum(1 for lv in router_edits.values() if label in lv and lv[label].get("changed"))
        log.info("    Router modified: %d/%d layers", router_changed, len(router_edits))
        shared_changed = sum(1 for lv in shared_edits.values() if label in lv and lv[label].get("changed"))
        log.info("    Shared expert modified: %d/%d layers", shared_changed, len(shared_edits))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", help="Path to comparison.json dir (batch mode)")
    parser.add_argument("--base", help="Base model dir")
    parser.add_argument("--variant-a", help="First variant dir")
    parser.add_argument("--variant-b", default=None, help="Second variant dir")
    parser.add_argument("--variant-c", default=None, help="Third variant dir")
    parser.add_argument("--label-a", default=None, help="Label for first variant")
    parser.add_argument("--label-b", default=None, help="Label for second variant")
    parser.add_argument("--label-c", default=None, help="Label for third variant")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--results-dir", help="Base results directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.comparison:
        config = ComparisonConfig.from_dir(Path(args.comparison))
        results_dir = Path(args.results_dir) if args.results_dir else config.weight_results_dir(Path("results"))
        for variant in config.variants:
            out_path = results_dir / variant.name / f"expert_analysis_{variant.name}.json"
            run_analysis(
                base_path=str(config.base_path),
                variant_a_path=str(variant.path),
                variant_b_path=args.variant_b,
                variant_c_path=args.variant_c,
                label_a=variant.display_name or variant.name,
                label_b=args.label_b,
                label_c=args.label_c,
                output_path=str(out_path),
                config=config,
                variant_name=variant.name,
            )
    else:
        if not all([args.base, args.variant_a, args.output, args.label_a]):
            parser.error("--base, --variant-a, --label-a, and --output are required without --comparison")
        run_analysis(
            base_path=args.base,
            variant_a_path=args.variant_a,
            variant_b_path=args.variant_b,
            variant_c_path=args.variant_c,
            label_a=args.label_a,
            label_b=args.label_b,
            label_c=args.label_c,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
