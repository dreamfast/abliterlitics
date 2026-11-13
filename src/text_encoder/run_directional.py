"""Directional analysis entry point — ablation delta geometry from cache.

Reads cached raw_hidden artifacts (per-layer pooled activations written by
run_analysis.py), pairs each heretic variant with its matched-quant base
(te_reference when set, else the global base), and computes the directional
metrics per layer (src/text_encoder/directional.py).

Writes te_directional_<variant>.json per variant next to the other results.

Usage (inside the TE container):
    python3 -m src.text_encoder.run_directional \
        --cache-dir /comparison/.te_cache --results-dir /results \
        --pairs heretic=/models/base.safetensors ...
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import torch

from src.text_encoder import TE_RESULTS_VERSION
from src.text_encoder.directional import (
    DEFAULT_REFERENCE_CATEGORIES,
    DEFAULT_TABOO_CATEGORIES,
    DirectionalReport,
    analyze_layer,
)

log = logging.getLogger(__name__)

_LAYER_RE = re.compile(r"^layer_(\d+)$")


def _load_raw_hidden(cache_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Index cached raw_hidden payloads by (ckpt_id, transform, device).

    run_analysis writes raw_hidden for the SAME ckpt_id under several
    transforms ("orig" plus the "fakequant_int8" noise floor it builds by
    default) and potentially multiple devices into one cache dir. The old
    last-write-wins ckpt_id index let a directional pair silently resolve to
    the fake-quantized base payload — index the full key instead and let
    ``_resolve_payload`` pick the entry explicitly.
    """
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for f in sorted(cache_dir.glob("*.pt")):
        try:
            payload = torch.load(f, weights_only=False, map_location="cpu")
        except Exception as exc:  # noqa: BLE001 — cache corruption is skippable
            log.warning("Skipping unreadable cache file %s: %s", f.name, exc)
            continue
        if payload.get("stage") != "raw_hidden":
            continue
        meta = payload.get("meta") or {}
        by_key[(str(payload["ckpt_id"]), str(meta.get("transform", "orig")), str(payload.get("device", "")))] = payload
    return by_key


def _resolve_payload(
    index: dict[tuple[str, str, str], dict[str, Any]],
    ckpt_id: str,
    transform: str = "orig",
) -> dict[str, Any] | None:
    """Pick the cache payload for ``ckpt_id``, preferring the orig transform.

    Non-matching transforms (quant noise floors) and duplicate device entries
    are dropped with a warning; picks are deterministic (sha-sorted filenames
    upstream + sorted keys here).
    """
    matches = sorted((k, v) for k, v in index.items() if k[0] == ckpt_id)
    if not matches:
        return None
    wanted = [(k, v) for k, v in matches if k[1] == transform]
    if wanted:
        if len(wanted) < len(matches):
            dropped = sorted({k[1] for k, _ in matches if k[1] != transform})
            log.warning(
                "ckpt %s: ignoring cached transform(s) %s — using %r",
                Path(ckpt_id).name,
                dropped,
                transform,
            )
        matches = wanted
    else:
        available = sorted({k[1] for k, _ in matches})
        log.warning(
            "ckpt %s: no %r-transform cache entry — falling back to %s",
            Path(ckpt_id).name,
            transform,
            available,
        )
    if len(matches) > 1:
        devices = sorted({k[2] for k, _ in matches})
        log.warning(
            "ckpt %s: transform %r cached on multiple devices %s — using %r",
            Path(ckpt_id).name,
            matches[0][0][1],
            devices,
            matches[0][0][2],
        )
    return matches[0][1]


def run_pair(
    base_payload: dict[str, Any],
    var_payload: dict[str, Any],
    n_perm: int = 200,
) -> DirectionalReport:
    """Directional analysis for one variant vs one reference payload."""
    categories = base_payload["categories"]
    if categories != var_payload["categories"]:
        raise ValueError("prompt suites differ between base and variant caches")
    layers: list[Any] = []
    for name, stream in base_payload["streams"].items():
        m = _LAYER_RE.match(name)
        if not m:
            continue  # consumed_hidden handled separately below (same layer pool)
        var_stream = var_payload["streams"].get(name)
        if var_stream is None:
            log.warning("Variant missing layer stream %s — skipping layer", name)
            continue
        layers.append(
            analyze_layer(
                stream["activations"].float(),
                var_stream["activations"].float(),
                categories,
                layer=int(m.group(1)),
                n_perm=n_perm,
            )
        )
    layers.sort(key=lambda r: r.layer)
    return DirectionalReport(
        variant_id=var_payload["ckpt_id"],
        reference_id=base_payload["ckpt_id"],
        taboo_categories=DEFAULT_TABOO_CATEGORIES,
        reference_categories=DEFAULT_REFERENCE_CATEGORIES,
        layers=layers,
    )


def report_to_json(report: DirectionalReport, variant_name: str, n_perm: int) -> dict[str, Any]:
    return {
        "te_results_version": TE_RESULTS_VERSION,
        "analysis": "directional",
        "variant_name": variant_name,
        "variant_id": report.variant_id,
        "reference_id": report.reference_id,
        "taboo_categories": list(report.taboo_categories),
        "reference_categories": list(report.reference_categories),
        "n_perm": n_perm,
        "summary": report.summary(),
        "layers": [
            {
                "layer": r.layer,
                "n_taboo": r.n_taboo,
                "n_ref": r.n_ref,
                "delta_norm_taboo": r.delta_norm_taboo,
                "delta_norm_ref": r.delta_norm_ref,
                "cos_taboo_mean": r.cos_taboo_mean,
                "cos_taboo_abs_mean": r.cos_taboo_abs_mean,
                "cos_null_std": r.cos_null_std,
                "cos_z": r.cos_z,
                "orth_fraction": r.orth_fraction,
                "toward_innocent_mean": r.toward_innocent_mean,
                "toward_innocent_null_std": r.toward_innocent_null_std,
                "toward_innocent_z": r.toward_innocent_z,
            }
            for r in report.layers
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Directional analysis from cached TE activations")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--results-dir", required=True)
    # pairs: NAME=REFERENCE_CKPT_ID; variant ckpt resolved from cache by name
    # is not possible (cache keys are hashes) — instead pairs are
    # VAR_CKPT_ID:REF_CKPT_ID (colon-separated, paths contain '=' rarely,
    # never ':').
    p.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="variant_display_name=variant_ckpt_id:reference_ckpt_id entries",
    )
    p.add_argument("--n-perm", type=int, default=200)
    return p.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    cache = _load_raw_hidden(Path(args.cache_dir))
    log.info("Indexed %d raw_hidden artifacts in cache", len(cache))

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for pair in args.pairs:
        name, _, spec = pair.partition("=")
        var_id, _, ref_id = spec.partition(":")
        var_payload = _resolve_payload(cache, var_id)
        ref_payload = _resolve_payload(cache, ref_id)
        if var_payload is None or ref_payload is None:
            log.warning("Pair %s: cache miss (var=%s ref=%s) — skipping", name, var_id[-40:], ref_id[-40:])
            continue
        report = run_pair(ref_payload, var_payload, n_perm=args.n_perm)
        out = results_dir / f"te_directional_{name}.json"
        summary: dict[str, Any] = report.summary()
        out.write_text(json.dumps(report_to_json(report, name, args.n_perm), indent=2))
        log.info(
            "Directional %s: %d layers, peak |cos|=%.3f @ L%s (z=%.1f), orth=%.1f%%",
            name,
            summary.get("n_layers", 0),
            float(summary.get("peak_abs_cos", 0.0)),
            summary.get("peak_abs_cos_layer", -1),
            float(summary.get("peak_z", 0.0)),
            100 * float(summary.get("mean_orth_fraction", 0.0)),
        )
    log.info("=== Directional analysis complete ===")


if __name__ == "__main__":
    main()
