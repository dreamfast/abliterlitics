#!/usr/bin/env python3
"""
Low-Rank Reconstruction -- how many PCs capture each technique's edits.

Tests whether one technique's edits can be reconstructed from a low-rank
approximation of another's. This quantifies subspace sharing between techniques.
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


def _svd_for_ranks(
    delta: torch.Tensor,
    ranks: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_rank = max(ranks)
    min_dim = min(delta.shape)
    if min_dim <= max_rank * 2 + 10:
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    else:
        q = min(max_rank + 10, min_dim)
        U, S, V = torch.svd_lowrank(delta, q=q)
        Vh = V.T
    return U, S, Vh


def run_analysis(
    base_path: str,
    variant_a_path: str,
    variant_b_path: str,
    label_a: str,
    label_b: str,
    output_path: str,
    ranks: list[int] | None = None,
    config: ComparisonConfig | None = None,
    pair_label: str = "",
) -> None:
    if ranks is None:
        ranks = [1, 2, 5, 10, 20]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("Detecting architecture...")
    cfg = detect_architecture(base_path)
    log.info("  Family: %s", cfg.family)

    log.info("Building shard maps...")
    base_map = build_shard_map(base_path, cfg)
    mapa = build_shard_map(variant_a_path, cfg)
    mapb = build_shard_map(variant_b_path, cfg)

    all_keys = sorted(set(base_map) & set(mapa) & set(mapb))
    log.info("  Common keys: %d", len(all_keys))

    MAX_GPU_ELEMENT_BYTES = 2 * 1024**3

    per_tensor: list[dict] = []
    total_orig_energy_a = 0.0
    total_orig_energy_b = 0.0

    for i, ck in enumerate(all_keys):
        t_base = load_tensor(base_map, ck)
        t_a = load_tensor(mapa, ck)
        t_b = load_tensor(mapb, ck)
        if t_base is None or t_a is None or t_b is None:
            continue

        numel = t_base.numel()
        elem_bytes = numel * 4
        use_dev = device if elem_bytes <= MAX_GPU_ELEMENT_BYTES else torch.device("cpu")
        if use_dev.type == "cpu" and device.type == "cuda":
            log.info("  %s: CPU fallback (%d elements)", ck, numel)

        t_base_f = t_base.float()
        delta_a = (t_a.float() - t_base_f).to(use_dev)
        delta_b = (t_b.float() - t_base_f).to(use_dev)
        del t_base, t_a, t_b, t_base_f

        norm_a = delta_a.norm().item()
        norm_b = delta_b.norm().item()

        if norm_a < THRESHOLD and norm_b < THRESHOLD:
            del delta_a, delta_b
            continue

        entry: dict = {
            "key": ck,
            "type": cfg.get_tensor_type(ck),
            "layer": cfg.get_layer_index(ck),
            "numel": numel,
            "edit_norm_a": norm_a,
            "edit_norm_b": norm_b,
        }

        if norm_a > THRESHOLD:
            total_orig_energy_a += norm_a**2
        if norm_b > THRESHOLD:
            total_orig_energy_b += norm_b**2

        if delta_a.dim() > 1 and norm_a > THRESHOLD:
            try:
                U, S, Vh = _svd_for_ranks(delta_a, ranks)
                for r in ranks:
                    if r > len(S):
                        continue
                    recon = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]
                    error = (delta_a - recon).norm().item() ** 2
                    orig = norm_a**2
                    entry[f"recon_a_rank{r}_error_pct"] = error / orig * 100 if orig > 0 else 0
            except Exception as exc:
                log.warning("SVD failed for %s (variant a): %s", ck, exc)

        if delta_b.dim() > 1 and norm_b > THRESHOLD:
            try:
                U, S, Vh = _svd_for_ranks(delta_b, ranks)
                for r in ranks:
                    if r > len(S):
                        continue
                    recon = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]
                    error = (delta_b - recon).norm().item() ** 2
                    orig = norm_b**2
                    entry[f"recon_b_rank{r}_error_pct"] = error / orig * 100 if orig > 0 else 0
            except Exception as exc:
                log.warning("SVD failed for %s (variant b): %s", ck, exc)

        if norm_a > THRESHOLD and norm_b > THRESHOLD and delta_a.dim() > 1:
            try:
                U_a, S_a, Vh_a = _svd_for_ranks(delta_a, ranks)
                for r in ranks:
                    if r > len(S_a):
                        continue
                    proj = U_a[:, :r] @ (U_a[:, :r].T @ delta_b.flatten().reshape(delta_b.shape))
                    recon_error = (delta_b - proj).norm().item() ** 2
                    orig_b = norm_b**2
                    entry[f"recon_b_from_a_rank{r}_error_pct"] = recon_error / orig_b * 100 if orig_b > 0 else 100

                    proj2 = U_a[:, :r] @ (U_a[:, :r].T @ delta_a.flatten().reshape(delta_a.shape))
                    entry[f"self_recon_a_rank{r}_error_pct"] = (
                        (delta_a - proj2).norm().item() ** 2 / (norm_a**2) * 100 if norm_a > 0 else 0
                    )
            except Exception as exc:
                log.warning("Cross-reconstruction SVD failed for %s: %s", ck, exc)

        per_tensor.append(entry)
        del delta_a, delta_b
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (i + 1) % 200 == 0:
            gc.collect()
            log.info("  processed %d/%d", i + 1, len(all_keys))

    report: dict = {}
    if config is not None:
        report["metadata"] = make_metadata(config, variant=pair_label)
    report["results_version"] = 1
    report.update(
        {
            "architecture": {"family": cfg.family, "layers": cfg.layer_count},
            "variant_a": label_a,
            "variant_b": label_b,
            "ranks_tested": ranks,
            "total_tensors_analyzed": len(per_tensor),
            "per_tensor": per_tensor,
        }
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    log.info("Saved: %s", out)

    log.info("=== LOW-RANK RECONSTRUCTION SUMMARY ===")
    for r in ranks:
        cross_key = f"recon_b_from_a_rank{r}_error_pct"
        vals = [e[cross_key] for e in per_tensor if cross_key in e and e.get("edit_norm_b", 0) > THRESHOLD]
        if vals:
            log.info(
                "  Rank-%d: mean error reconstructing %s from %s subspace = %.1f%%",
                r,
                label_b,
                label_a,
                sum(vals) / len(vals),
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
    parser.add_argument("--ranks", type=str, default="1,2,5,10,20", help="Comma-separated ranks to test")
    parser.add_argument("--results-dir", help="Override output directory (used with --comparison)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ranks = [int(r) for r in args.ranks.split(",")]

    if args.comparison:
        config = ComparisonConfig.from_dir(Path(args.comparison))
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
                out_path = results_dir / f"lowrank_{v1.name}_vs_{v2.name}.json"
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
                    ranks=ranks,
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
            ranks=ranks,
        )


if __name__ == "__main__":
    main()
