#!/usr/bin/env python3
"""
SVD technique analysis -- architecture-agnostic, streaming.

Computes SVD of edit deltas for any variant vs base.
Uses torch.svd_lowrank(k=TOP_K) by default. Use --full-svd for exact.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import torch

from src.config import ComparisonConfig, make_metadata
from src.model_config import build_shard_map, detect_architecture, load_tensor

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_svd_stats(
    delta: torch.Tensor,
    top_k: int = 20,
    full_svd: bool = False,
) -> dict:
    true_frobenius = float(delta.norm())
    entry: dict = {
        "frobenius_norm": true_frobenius,
        "mean_abs": float(delta.abs().mean()),
        "max_abs": float(delta.abs().max()),
    }

    if delta.numel() <= 1 or delta.abs().max() < 1e-10:
        return entry

    if delta.dim() == 1:
        entry.update(
            {"effective_rank_90pct_energy": 1, "energy_top1_pct": 100.0, "top_singular_values": [float(delta.norm())]}
        )
        return entry

    try:
        if full_svd:
            U, S, Vh = torch.linalg.svd(delta.float(), full_matrices=False)
        else:
            k = min(top_k, min(delta.shape) - 1)
            if k < 1:
                k = 1
            U, S, Vh = torch.svd_lowrank(delta.float(), q=k)

        all_sv = S.tolist()
        true_total_energy = true_frobenius**2
        total_energy = (S**2).sum() if full_svd else torch.tensor(true_total_energy, device=S.device)

        if total_energy < 1e-20:
            return entry

        cumulative = (S**2).cumsum(0) / total_energy
        eff_rank_90 = int((cumulative < 0.9).sum()) + 1
        eff_rank_99 = int((cumulative < 0.99).sum()) + 1
        sv_ratio = float(S[0] / S[1]) if len(S) > 1 and S[1] > 1e-10 else float("inf")

        entry.update(
            {
                "effective_rank_90pct_energy": min(eff_rank_90, len(S)),
                "effective_rank_99pct_energy": min(eff_rank_99, len(S)),
                "sv_ratio_top2": sv_ratio,
                "top_singular_values": all_sv[:50],
                "total_rank": len(S),
                "energy_top1_pct": float(S[0] ** 2 / total_energy * 100),
                "energy_top5_pct": float((S[:5] ** 2).sum() / total_energy * 100),
                "top_k_computed": len(S),
                "full_svd": full_svd,
            }
        )

        if not full_svd:
            topk_energy = float((S**2).sum())
            entry["lowrank_captured_energy_pct"] = (
                float(topk_energy / true_total_energy * 100) if true_total_energy > 1e-20 else 0.0
            )

        if full_svd:
            total_sv = len(S)
            entry["energy_at_50pct_rank"] = float((S[: max(1, total_sv // 2)] ** 2).sum() / total_energy * 100)
    except Exception as e:
        entry["svd_error"] = str(e)

    return entry


def run_analysis(
    base_path: str,
    variant_a_path: str,
    variant_b_path: str | None,
    label_a: str,
    label_b: str,
    output_path: str,
    top_k: int = 20,
    full_svd: bool = False,
    panel_path: str | None = None,
    config: ComparisonConfig | None = None,
    variant_name: str = "",
) -> None:
    OUT = Path(output_path)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    TOP_K = top_k

    log.info("\nSVD ANALYSIS (%s) on %s", "FULL" if full_svd else f"lowrank k={TOP_K}", DEVICE)

    log.info("Detecting architecture & building shard maps...")
    cfg = detect_architecture(base_path)
    log.info("  Family: %s, Layers: %d", cfg.family, cfg.layer_count)

    base_map = build_shard_map(base_path, cfg)
    mapa = build_shard_map(variant_a_path, cfg)
    mapb = build_shard_map(variant_b_path, cfg) if variant_b_path else None

    log.info("  Base: %d, A: %d%s", len(base_map), len(mapa), f", B: {len(mapb)}" if mapb else "")

    if panel_path:
        with open(panel_path) as f:
            panel = json.load(f)
        panel_key = f"base->{label_a}"
        a_changed = set(panel.get(f"{panel_key}_keys", []))
        if not a_changed:
            all_pairs = panel.get("pairwise_changed_counts", {})
            for k in all_pairs:
                if label_a in k:
                    a_changed = set(panel.get(f"{k}_keys", []))
                    break
        keys_to_analyze = sorted(a_changed) if a_changed else sorted(set(base_map) & set(mapa))
    else:
        keys_to_analyze = sorted(set(base_map) & set(mapa))

    log.info("  Analyzing %d tensors...", len(keys_to_analyze))

    t0 = time.time()
    results: list[dict] = []
    for i, key in enumerate(keys_to_analyze):
        t_base = load_tensor(base_map, key)
        t_a = load_tensor(mapa, key)
        if t_base is None or t_a is None:
            continue
        if t_base.shape != t_a.shape:
            continue

        delta_a = t_a.float().to(DEVICE) - t_base.float().to(DEVICE)

        result: dict = {
            "key": key,
            "layer": cfg.get_layer_index(key),
            "tensor_type": cfg.get_tensor_type(key),
            "category": cfg.tensor_category(key),
            "shape": list(t_base.shape),
            "numel": t_base.numel(),
            f"{label_a}_vs_base": compute_svd_stats(delta_a, TOP_K, full_svd),
        }

        if mapb:
            t_b = load_tensor(mapb, key)
            if t_b is not None and t_b.shape == t_base.shape:
                delta_b = t_b.float().to(DEVICE) - t_base.float().to(DEVICE)
                delta_ab = t_b.float().to(DEVICE) - t_a.float().to(DEVICE)
                result[f"{label_b}_vs_base"] = compute_svd_stats(delta_b, TOP_K, full_svd)
                result[f"{label_b}_minus_{label_a}"] = compute_svd_stats(delta_ab, TOP_K, full_svd)
                del t_b, delta_b, delta_ab

        results.append(result)

        rank_val = result.get(f"{label_a}_vs_base", {}).get("effective_rank_90pct_energy", "?")
        short = key.split("layers.")[-1][:40] if "layers." in key else key[:40]
        log.info("  [%d/%d] %s rank90=%s (%ds)", i + 1, len(keys_to_analyze), short, rank_val, int(time.time() - t0))

        del t_base, t_a, delta_a
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    report: dict = {}
    if config is not None:
        report["metadata"] = make_metadata(config, variant=variant_name)
    report["results_version"] = 1
    report.update(
        {
            "architecture": {"family": cfg.family, "layers": cfg.layer_count},
            "variant_a": label_a,
            "variant_b": label_b if mapb else None,
            "total_analyzed": len(results),
            "full_svd": full_svd,
            "tensor_results": results,
        }
    )
    OUT.write_text(json.dumps(report, indent=2))
    log.info("Saved: %s (%d tensors, %ds)", OUT, len(results), int(time.time() - t0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", help="Path to comparison.json dir (batch mode)")
    parser.add_argument("--base", help="Base model dir")
    parser.add_argument("--variant-a", help="First variant")
    parser.add_argument("--variant-b", default=None, help="Second variant (optional)")
    parser.add_argument("--label-a", default="variant_a")
    parser.add_argument("--label-b", default="variant_b")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--full-svd", action="store_true")
    parser.add_argument("--panel", default=None, help="Panel JSON to get changed keys (optional)")
    parser.add_argument("--results-dir", help="Base results directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.comparison:
        config = ComparisonConfig.from_dir(Path(args.comparison))
        results_dir = Path(args.results_dir) if args.results_dir else config.weight_results_dir(Path("results"))
        for variant in config.variants:
            out_path = results_dir / variant.name / f"svd_{variant.name}.json"
            run_analysis(
                base_path=str(config.base_path),
                variant_a_path=str(variant.path),
                variant_b_path=args.variant_b,
                label_a=variant.display_name,
                label_b=args.label_b,
                output_path=str(out_path),
                top_k=args.top_k,
                full_svd=args.full_svd,
                panel_path=args.panel,
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
            top_k=args.top_k,
            full_svd=args.full_svd,
            panel_path=args.panel,
        )


if __name__ == "__main__":
    main()
