#!/usr/bin/env python3
"""
Cross-Architecture Comparison -- technique fingerprinting.

Compares how the same abliteration technique targets different architectures.
Requires panel data from at least 2 model families.

Two invocation modes:

1. Panels mode (explicit): ``--panels <json...> --labels <name...> --output <json>``.
   Reads existing panel JSON files directly; does NOT use comparison.json or
   model directories.

2. Comparison mode (pipeline): ``--comparison <dir> [--results-dir <dir>]``.
   As invoked by runners/run_weights.sh and runners/weights_sequencer.sh, which
   pass ``--comparison/--results-dir`` uniformly to every weight script. Globs
   ``multi_model_panel.json`` under the results directory (so a results dir
   spanning multiple comparisons yields multiple panels), labels each panel by
   its metadata ``comparison_name`` (falling back to the comparison directory
   name), and writes ``cross_arch.json`` into the results directory. Exits 1
   with a clear message when no panel exists yet.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from src.config import ComparisonConfig

log = logging.getLogger(__name__)


def _load_panel(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Skipping unreadable panel %s: %s", path, exc)
        return None


def _panel_label(path: Path, data: dict, index: int) -> str:
    """Label a panel by its comparison name, falling back to directory layout.

    Standard layout is ``<comparison>/results/weight/multi_model_panel.json``,
    so ``parents[2]`` is the comparison directory.
    """
    label = data.get("metadata", {}).get("comparison_name")
    if label:
        return str(label)
    if len(path.parents) > 2:
        return path.parents[2].name
    return f"panel_{index}"


def build_report(panels: list[dict], labels: list[str]) -> dict:
    """Build the cross-architecture report from loaded panel JSONs."""
    targeting: dict[str, dict] = {}
    for label, data in zip(labels, panels, strict=False):
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

    return {
        "results_version": 1,
        "metadata": {},
        "models": targeting,
        "technique_edit_counts_matrix": technique_matrix,
        "type_comparison": type_comparison,
        "all_tensor_types": all_types,
    }


def run_comparison_mode(comparison: str, results_dir: str | None, output: str | None) -> int:
    """Pipeline mode: glob panels under the results dir and write cross_arch.json.

    Returns a process exit code (0 on success, 1 when no usable panel exists).
    """
    config = ComparisonConfig.from_dir(Path(comparison))
    results_path = Path(results_dir) if results_dir else config.weight_results_dir(Path("results"))
    panel_paths = sorted(results_path.rglob("multi_model_panel.json"))
    if not panel_paths:
        log.error(
            "No multi_model_panel.json found under %s — run panel_comparison.py first "
            "(cross-architecture comparison requires panel data)",
            results_path,
        )
        return 1

    panels: list[dict] = []
    labels: list[str] = []
    for i, panel_path in enumerate(panel_paths):
        data = _load_panel(panel_path)
        if data is None:
            continue
        panels.append(data)
        labels.append(_panel_label(panel_path, data, i))

    if not panels:
        log.error("Panel files found under %s but none could be parsed", results_path)
        return 1

    report = build_report(panels, labels)

    out_path = Path(output) if output else results_path / "cross_arch.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    log.info("Saved: %s", out_path)

    _log_summary(report, labels)
    return 0


def _log_summary(report: dict, labels: list[str]) -> None:
    log.info("=== CROSS-ARCHITECTURE COMPARISON ===")
    log.info("Technique edit counts by model:")
    for pair_key in sorted(report["technique_edit_counts_matrix"]):
        parts = [f"{report['technique_edit_counts_matrix'][pair_key].get(l, 'N/A')}" for l in labels]
        log.info("  %s: %s", pair_key, " | ".join(parts))

    log.info("Tensor type targeting across architectures:")
    for t in report["all_tensor_types"]:
        parts = [f"{report['type_comparison'][t].get(l, 0)}" for l in labels]
        log.info("  %s: %s", t, " | ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", help="Path to comparison.json dir (pipeline mode)")
    parser.add_argument("--results-dir", help="Results directory to glob for panels (with --comparison)")
    parser.add_argument("--panels", nargs="+", required=False, help="Panel JSON paths (explicit mode)")
    parser.add_argument("--labels", nargs="+", required=False, help="Labels for --panels (explicit mode)")
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.comparison:
        rc = run_comparison_mode(args.comparison, args.results_dir, args.output)
        if rc:
            sys.exit(rc)
        return

    if not args.panels or not args.labels:
        parser.error("either --comparison, or both --panels and --labels are required")
    if not args.output:
        parser.error("--output is required in --panels mode")
    assert len(args.panels) == len(args.labels)

    panels: list[dict] = []
    for path, label in zip(args.panels, args.labels, strict=False):
        data = _load_panel(Path(path))
        if data is None:
            parser.error(f"cannot read panel: {path}")
        panels.append({"label": label, "data": data})

    report = build_report([p["data"] for p in panels], [p["label"] for p in panels])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    log.info("Saved: %s", out)

    _log_summary(report, [p["label"] for p in panels])


if __name__ == "__main__":
    main()
