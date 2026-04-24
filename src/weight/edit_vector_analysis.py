#!/usr/bin/env python3
"""
Edit Vector Comparative Analysis -- architecture-agnostic.

Computes edit vectors for any pair of model variants against a base.
Generates per-key metrics: cosine similarity, R-squared, norm ratios, correlations.
Works with any architecture (Qwen3.5, Qwen3, GLM) via model_config auto-detection.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
from pathlib import Path

import torch

from src.config import ComparisonConfig, make_metadata
from src.model_config import build_shard_map, detect_architecture, load_tensor

log = logging.getLogger(__name__)

TRIVIAL_THRESHOLD = 1e-10


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    dot = (a_f * b_f).sum()
    na = a_f.norm()
    nb = b_f.norm()
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return max(-1.0, min(1.0, (dot / (na * nb)).item()))


def pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    a_c = a_f - a_f.mean()
    b_c = b_f - b_f.mean()
    na = a_c.norm()
    nb = b_c.norm()
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return ((a_c * b_c).sum() / (na * nb)).item()


def linear_r2(predictor: torch.Tensor, response: torch.Tensor) -> float:
    x = predictor.float().flatten()
    y = response.float().flatten()
    if x.numel() < 2:
        return 0.0
    xc = x - x.mean()
    yc = y - y.mean()
    denom = (xc * xc).sum()
    if denom < 1e-12:
        return 0.0
    slope = (xc * yc).sum() / denom
    intercept = y.mean() - slope * x.mean()
    ss_res = ((y - (slope * x + intercept)) ** 2).sum()
    ss_tot = (yc**2).sum()
    if ss_tot < 1e-12:
        return 1.0
    return (1.0 - ss_res / ss_tot).item()


def null_cosine_expected(dim: int) -> dict:
    std = 1.0 / math.sqrt(dim) if dim > 0 else 0.0
    return {"expected_mean": 0.0, "expected_std": std, "dimensionality": dim, "three_sigma": 3.0 * std}


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


def summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": safe_mean(values),
        "median": safe_median(values),
        "std": safe_std(values),
        "min": min(values),
        "max": max(values),
    }


def analyze_single_variant(
    base_map: dict,
    variant_map: dict,
    label: str,
    cfg: object,
) -> list[dict]:
    all_keys = sorted(set(base_map) & set(variant_map))
    log.info("  %s: %d common keys", label, len(all_keys))

    per_key: list[dict] = []
    for i, ck in enumerate(all_keys):
        t_base = load_tensor(base_map, ck)
        t_var = load_tensor(variant_map, ck)
        if t_base is None or t_var is None:
            continue
        t_base = t_base.float()
        t_var = t_var.float()
        v = t_var - t_base
        norm_v = v.norm().item()
        per_key.append(
            {
                "key": ck,
                "tensor_type": cfg.get_tensor_type(ck),
                "layer": cfg.get_layer_index(ck),
                "category": cfg.tensor_category(ck),
                "edit_norm": norm_v,
                "base_norm": t_base.norm().item(),
                "relative_edit": norm_v / t_base.norm().item() if t_base.norm().item() > 0 else 0,
                "numel": t_base.numel(),
                "changed": norm_v > TRIVIAL_THRESHOLD,
            }
        )
        del t_base, t_var, v
        if (i + 1) % 200 == 0:
            gc.collect()
    return per_key


def analyze_two_variants(
    base_map: dict,
    mapa: dict,
    mapb: dict,
    label_a: str,
    label_b: str,
    cfg: object,
) -> list[dict]:
    all_keys = sorted(set(base_map) & set(mapa) & set(mapb))
    log.info("  Comparing %s vs %s: %d common keys", label_a, label_b, len(all_keys))

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

        v_a = t_a - t_base
        v_b = t_b - t_base
        v_delta = t_b - t_a

        norm_a = v_a.norm().item()
        norm_b = v_b.norm().item()
        norm_d = v_delta.norm().item()

        a_changed = norm_a > TRIVIAL_THRESHOLD
        b_changed = norm_b > TRIVIAL_THRESHOLD
        is_trivial = norm_d < TRIVIAL_THRESHOLD

        entry: dict = {
            "key": ck,
            "tensor_type": cfg.get_tensor_type(ck),
            "layer": cfg.get_layer_index(ck),
            "category": cfg.tensor_category(ck),
            "a_edit_norm": norm_a,
            "b_edit_norm": norm_b,
            "delta_norm": norm_d,
            "numel": t_base.numel(),
            "a_changed": a_changed,
            "b_changed": b_changed,
            "is_trivial_copy": is_trivial if (a_changed and b_changed) else None,
        }

        if a_changed and b_changed:
            entry["cosine_edit_directions"] = cosine_sim(v_a, v_b)
            entry["r2_a_predicts_b"] = linear_r2(v_a, v_b)
        if a_changed and b_changed and not is_trivial:
            entry["corr_a_delta"] = pearson_corr(v_a, v_delta)

        per_key.append(entry)
        del t_base, t_a, t_b, v_a, v_b, v_delta
        if (i + 1) % 200 == 0:
            gc.collect()
            log.info("    processed %d/%d", i + 1, len(all_keys))
    return per_key


def run_analysis(
    base_path: str,
    variant_a_path: str,
    variant_b_path: str | None,
    label_a: str,
    label_b: str,
    output_path: str,
    config: ComparisonConfig | None = None,
    variant_name: str = "",
) -> None:
    log.info("Detecting architecture...")
    cfg = detect_architecture(base_path)
    log.info("  Family: %s, Layers: %d, Keys: %d", cfg.family, cfg.layer_count, cfg.total_keys)

    log.info("Building shard maps...")
    base_map = build_shard_map(base_path, cfg)
    mapa = build_shard_map(variant_a_path, cfg)

    report: dict = {}
    if config is not None:
        report["metadata"] = make_metadata(config, variant=variant_name)
    report["results_version"] = 1
    report["architecture"] = {"family": cfg.family, "layers": cfg.layer_count, "total_keys": cfg.total_keys}

    if variant_b_path:
        mapb = build_shard_map(variant_b_path, cfg)
        log.info("  Base: %d, A(%s): %d, B(%s): %d", len(base_map), label_a, len(mapa), label_b, len(mapb))

        per_key = analyze_two_variants(base_map, mapa, mapb, label_a, label_b, cfg)

        overlap = [r for r in per_key if r["a_changed"] and r["b_changed"]]
        trivial = [r for r in overlap if r.get("is_trivial_copy")]
        nontrivial = [r for r in overlap if not r.get("is_trivial_copy")]
        a_only = [r for r in per_key if r["a_changed"] and not r["b_changed"]]
        b_only = [r for r in per_key if not r["a_changed"] and r["b_changed"]]

        cos_vals = [r["cosine_edit_directions"] for r in overlap if "cosine_edit_directions" in r]
        nt_cos = [r["cosine_edit_directions"] for r in nontrivial if "cosine_edit_directions" in r]
        r2_vals = [r["r2_a_predicts_b"] for r in overlap if "r2_a_predicts_b" in r]
        nt_r2 = [r["r2_a_predicts_b"] for r in nontrivial if "r2_a_predicts_b" in r]
        corr_vals = [r["corr_a_delta"] for r in nontrivial if "corr_a_delta" in r]

        nt_dims = [r["numel"] for r in nontrivial]
        median_dim = sorted(nt_dims)[len(nt_dims) // 2] if nt_dims else 0

        report.update(
            {
                "variant_a": label_a,
                "variant_b": label_b,
                "total_keys": len(per_key),
                "overlap_count": len(overlap),
                "trivial_copy_count": len(trivial),
                "nontrivial_overlap_count": len(nontrivial),
                "a_only_count": len(a_only),
                "b_only_count": len(b_only),
                "cosine_all": {**summary(cos_vals), "description": "ALL overlap keys"},
                "cosine_nontrivial": {**summary(nt_cos), "description": "Non-trivial overlap only"},
                "r2_all": {**summary(r2_vals), "description": "ALL overlap keys"},
                "r2_nontrivial": {**summary(nt_r2), "description": "Non-trivial overlap only"},
                "corr_a_delta": {**summary(corr_vals), "description": "Corr(edit_a, delta_b-a) on non-trivial"},
                "null_baseline": null_cosine_expected(median_dim),
                "per_key_details": per_key,
            }
        )

        log.info("Overlap: %d (trivial: %d, non-trivial: %d)", len(overlap), len(trivial), len(nontrivial))
        if nt_cos:
            s = summary(nt_cos)
            nb = null_cosine_expected(median_dim)
            log.info("Non-trivial cosine: mean=%.4f, null 3sigma=%.4f", s["mean"], nb["three_sigma"])
    else:
        log.info("  Base: %d, A(%s): %d", len(base_map), label_a, len(mapa))
        per_key = analyze_single_variant(base_map, mapa, label_a, cfg)

        changed = [r for r in per_key if r["changed"]]
        report.update(
            {
                "variant": label_a,
                "total_keys": len(per_key),
                "changed_count": len(changed),
                "edit_norms": summary([r["edit_norm"] for r in changed]),
                "relative_edits": summary([r["relative_edit"] for r in changed]),
                "per_key_details": per_key,
            }
        )
        log.info("Changed: %d / %d", len(changed), len(per_key))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    log.info("Saved: %s", out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", help="Path to comparison.json dir (batch mode)")
    parser.add_argument("--base", help="Base model dir (overrides --comparison)")
    parser.add_argument("--variant-a", help="First variant (e.g. heretic)")
    parser.add_argument(
        "--variant-b", default=None, help="Second variant (e.g. hauhau). Omit for single-variant analysis."
    )
    parser.add_argument("--label-a", default="variant_a")
    parser.add_argument("--label-b", default="variant_b")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--results-dir", help="Base results directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.comparison:
        config = ComparisonConfig.from_dir(Path(args.comparison))
        results_dir = Path(args.results_dir) if args.results_dir else config.weight_results_dir(Path("results"))
        for variant in config.variants:
            out_path = results_dir / variant.name / f"edit_vector_{variant.name}.json"
            run_analysis(
                base_path=str(config.base_path),
                variant_a_path=str(variant.path),
                variant_b_path=args.variant_b,
                label_a=variant.display_name,
                label_b=args.label_b,
                output_path=str(out_path),
                config=config,
                variant_name=variant.name,
            )
    else:
        if not all([args.base, args.variant_a, args.output]):
            parser.error("--base, --variant-a, and --output are required without --comparison")
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
