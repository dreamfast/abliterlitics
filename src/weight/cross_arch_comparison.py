#!/usr/bin/env python3
"""
Cross-Architecture Comparison -- technique fingerprinting.

Compares how the same abliteration technique targets different architectures.
Requires panel data from at least 2 model families.

NOTE: This script reads existing panel JSON files via --panels and --labels.
It does NOT use comparison.json or model directories directly.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panels", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    assert len(args.panels) == len(args.labels)

    panels: list[dict] = []
    for path, label in zip(args.panels, args.labels, strict=False):
        with open(path) as f:
            panels.append({"label": label, "data": json.load(f)})

    targeting: dict[str, dict] = {}
    for p in panels:
        label = p["label"]
        data = p["data"]
        arch = data.get("architecture", {})

        counts = data.get("pairwise_changed_counts", {})
        entry: dict = {
            "label": label,
            "family": arch.get("family", "unknown"),
            "layers": arch.get("layer_count", 0),
            "total_keys": arch.get("total_keys", 0),
            "pairwise_counts": counts,
        }

        per_layer = data.get("per_layer_changes", {})
        layer_profiles: dict[str, list[dict]] = {}
        for pair_key, layers in per_layer.items():
            profile: list[dict] = []
            for layer_str, info in sorted(layers.items(), key=lambda x: int(x[0])):
                profile.append({"layer": int(layer_str), "count": info["count"], "types": info["types"]})
            layer_profiles[pair_key] = profile
        entry["layer_profiles"] = layer_profiles

        type_counts: dict[str, int] = defaultdict(int)
        for pair_key, layers in per_layer.items():
            for layer_str, info in layers.items():
                for t in info["types"]:
                    type_counts[t] += 1
        entry["type_counts"] = dict(sorted(type_counts.items()))

        targeting[label] = entry

    all_types = sorted(set(t for entry in targeting.values() for t in entry["type_counts"]))

    technique_matrix: dict[str, dict[str, int]] = {}
    for entry in targeting.values():
        label = entry["label"]
        for pair_key, count in entry["pairwise_counts"].items():
            technique_matrix.setdefault(pair_key, {})[label] = count

    type_comparison: dict[str, dict[str, int]] = {}
    for t in all_types:
        type_comparison[t] = {}
        for label, entry in targeting.items():
            type_comparison[t][label] = entry["type_counts"].get(t, 0)

    report: dict = {
        "results_version": 1,
        "metadata": {},
        "models": targeting,
        "technique_edit_counts_matrix": technique_matrix,
        "type_comparison": type_comparison,
        "all_tensor_types": all_types,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    log.info("Saved: %s", out)

    log.info("=== CROSS-ARCHITECTURE COMPARISON ===")
    log.info("Technique edit counts by model:")
    for pair_key in sorted(technique_matrix):
        parts = [f"{technique_matrix[pair_key].get(l, 'N/A')}" for l in args.labels]
        log.info("  %s: %s", pair_key, " | ".join(parts))

    log.info("Tensor type targeting across architectures:")
    for t in all_types:
        parts = [f"{type_comparison[t].get(l, 0)}" for l in args.labels]
        log.info("  %s: %s", t, " | ".join(parts))


if __name__ == "__main__":
    main()
