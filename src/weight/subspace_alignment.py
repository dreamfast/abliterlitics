#!/usr/bin/env python3
"""
Edit Direction Alignment -- principal angles between edit subspaces.

Computes subspace overlap between different techniques' edit directions
across model sizes/families. Tests whether techniques find the same
"refusal direction" in weight space.

Approach: Group edits by tensor type so that vectors within each group have
the same dimensionality.  Compute per-type principal angles (via SVD of the
cross-Gram matrix) then aggregate.  This avoids the torch.stack crash that
occurs when tensors of different shapes are stacked.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from collections import defaultdict
from pathlib import Path

import torch

from src.config import ComparisonConfig, make_metadata
from src.model_config import build_shard_map, detect_architecture, load_tensor

log = logging.getLogger(__name__)

THRESHOLD = 1e-10
MAX_GPU_BYTES = 4 * 1024**3


def _safe_device_for_stack(
    vecs: list[torch.Tensor],
    device: torch.device,
) -> torch.device:
    dim = vecs[0].numel()
    n = len(vecs)
    needed = dim * n * 4
    if needed > MAX_GPU_BYTES:
        return torch.device("cpu")
    return device


def principal_angles(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    top_k: int = 10,
) -> list[float]:
    k_a = min(top_k, mat_a.shape[1])
    k_b = min(top_k, mat_b.shape[1])
    A = mat_a[:, :k_a].float()
    B = mat_b[:, :k_b].float()

    Q_a, _ = torch.linalg.qr(A)
    Q_b, _ = torch.linalg.qr(B)

    M = Q_a.T @ Q_b
    _, s, _ = torch.linalg.svd(M)
    return torch.clamp(s, 0.0, 1.0).tolist()


def run_analysis(
    base_path: str,
    variant_a_path: str,
    variant_b_path: str,
    label_a: str,
    label_b: str,
    output_path: str,
    top_k: int = 10,
    config: ComparisonConfig | None = None,
    pair_label: str = "",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("Detecting architecture...")
    cfg = detect_architecture(base_path)
    log.info("  Family: %s, Layers: %d", cfg.family, cfg.layer_count)

    log.info("Building shard maps...")
    base_map = build_shard_map(base_path, cfg)
    mapa = build_shard_map(variant_a_path, cfg)
    mapb = build_shard_map(variant_b_path, cfg)

    all_keys = sorted(set(base_map) & set(mapa) & set(mapb))
    log.info("  Common keys: %d", len(all_keys))

    type_vectors_a: dict[str, list[torch.Tensor]] = defaultdict(list)
    type_vectors_b: dict[str, list[torch.Tensor]] = defaultdict(list)
    per_key_cos: list[dict] = []
    total_both_changed = 0

    for i, ck in enumerate(all_keys):
        t_base = load_tensor(base_map, ck)
        t_a = load_tensor(mapa, ck)
        t_b = load_tensor(mapb, ck)
        if t_base is None or t_a is None or t_b is None:
            continue

        t_base = t_base.float()
        t_a = t_a.float()
        t_b = t_b.float()

        v_a = (t_a - t_base).flatten()
        v_b = (t_b - t_base).flatten()

        norm_a = v_a.norm().item()
        norm_b = v_b.norm().item()

        a_changed = norm_a > THRESHOLD
        b_changed = norm_b > THRESHOLD
        ttype = cfg.get_tensor_type(ck)

        if a_changed and b_changed:
            cos = torch.nn.functional.cosine_similarity(v_a.unsqueeze(0), v_b.unsqueeze(0)).item()
            type_vectors_a[ttype].append(v_a / v_a.norm())
            type_vectors_b[ttype].append(v_b / v_b.norm())
            per_key_cos.append({"key": ck, "type": ttype, "layer": cfg.get_layer_index(ck), "cosine": cos})
            total_both_changed += 1

        del t_base, t_a, t_b, v_a, v_b
        if (i + 1) % 200 == 0:
            gc.collect()

    log.info("  Both-variant changed tensors: %d", total_both_changed)
    log.info("  Tensor types with overlap: %d", len(type_vectors_a))

    if total_both_changed < 2:
        log.info("Not enough overlap for subspace analysis.")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps({"error": "insufficient overlap"}))
        return

    per_type_results: dict[str, dict] = {}
    all_angles_ab: list[float] = []

    for ttype in sorted(type_vectors_a.keys()):
        vecs_a = type_vectors_a[ttype]
        vecs_b = type_vectors_b[ttype]

        dim_groups_a: dict[int, list[torch.Tensor]] = defaultdict(list)
        dim_groups_b: dict[int, list[torch.Tensor]] = defaultdict(list)
        for v in vecs_a:
            dim_groups_a[v.numel()].append(v)
        for v in vecs_b:
            dim_groups_b[v.numel()].append(v)

        common_dims = sorted(set(dim_groups_a) & set(dim_groups_b))
        if not common_dims:
            log.info("    %s: no common dimensions, skipping", ttype)
            continue

        type_angles: list[float] = []
        type_n_vectors = 0
        type_dims: list[int] = []

        for dim in common_dims:
            ga = dim_groups_a[dim]
            gb = dim_groups_b[dim]
            if len(ga) < 2 or len(gb) < 2:
                continue

            use_dev = _safe_device_for_stack(ga, device)
            if use_dev.type == "cpu" and device.type == "cuda":
                log.info("    %s dim=%d: CPU fallback (n=%d)", ttype, dim, len(ga))
            mat_a = torch.stack(ga, dim=1).to(use_dev)
            mat_b = torch.stack(gb, dim=1).to(use_dev)

            angles = principal_angles(mat_a, mat_b, top_k=top_k)
            type_angles.extend(angles)
            type_n_vectors += len(ga)
            type_dims.append(dim)

            del mat_a, mat_b
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if not type_angles:
            continue

        per_type_results[ttype] = {
            "n_vectors": type_n_vectors,
            "dims": type_dims,
            "principal_angles": type_angles,
            "mean_cosine": sum(type_angles) / len(type_angles) if type_angles else 0,
            "overlap_fraction_gt_0.9": sum(1 for a in type_angles if a > 0.9) / len(type_angles) if type_angles else 0,
        }
        all_angles_ab.extend(type_angles)

    mean_cosine_global = sum(all_angles_ab) / len(all_angles_ab) if all_angles_ab else 0
    overlap_frac_global = sum(1 for a in all_angles_ab if a > 0.9) / len(all_angles_ab) if all_angles_ab else 0

    results: dict = {}
    if config is not None:
        results["metadata"] = make_metadata(config, variant=pair_label)
    results["results_version"] = 1
    results.update(
        {
            "num_overlap_tensors": total_both_changed,
            "num_types_analyzed": len(per_type_results),
            "top_k": top_k,
            "global_mean_cosine_principal": mean_cosine_global,
            "global_overlap_fraction_gt_0.9": overlap_frac_global,
            "all_principal_angles": all_angles_ab,
            "per_type": per_type_results,
            "per_key_cosines": sorted(per_key_cos, key=lambda x: abs(x["cosine"]), reverse=True),
        }
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log.info("Saved: %s", out)

    log.info("=== SUBSPACE ALIGNMENT ===")
    log.info("  Overlap tensors: %d", total_both_changed)
    log.info("  Types analyzed: %d", len(per_type_results))
    log.info("  Global mean cosine (%s vs %s): %.4f", label_a, label_b, mean_cosine_global)
    log.info("  Global overlap (>0.9): %.2f%%", overlap_frac_global * 100)
    for ttype, info in sorted(per_type_results.items(), key=lambda x: -x[1]["mean_cosine"])[:5]:
        log.info("    %s: mean_cos=%.4f, n=%d", ttype, info["mean_cosine"], info["n_vectors"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", help="Path to comparison.json dir (batch mode)")
    parser.add_argument("--base", help="Base model dir")
    parser.add_argument("--variant-a", help="First variant dir")
    parser.add_argument("--variant-b", help="Second variant dir")
    parser.add_argument("--label-a", default="a")
    parser.add_argument("--label-b", default="b")
    parser.add_argument("--output", help="Output JSON path (single-pair mode only; ignored with --comparison)")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K singular vectors for subspace")
    parser.add_argument("--results-dir", help="Override output directory (used with --comparison)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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
                out_path = results_dir / f"subspace_{v1.name}_vs_{v2.name}.json"
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
                    top_k=args.top_k,
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
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
