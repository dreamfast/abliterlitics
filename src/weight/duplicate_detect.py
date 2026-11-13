#!/usr/bin/env python3
"""
CPU-only SHA256 fingerprinting of model directories.

Models on HuggingFace are frequently reuploaded under new names with identical
weights. A naive file-level SHA256 is useless because safetensors are sharded
and the shard boundaries / file sizes differ between uploads. This module hashes
each *tensor's* raw bytes via ``safe_open`` (lazy mmap) and folds them into a
single per-model SHA256, which is independent of how tensors were split across
shards. Two models with the same SHA256 are byte-identical reuploads.

Runs entirely on CPU (``safe_open(..., device="cpu")``); no GPU is touched.
Throughput is ~1 GB/s per core, so a 16 GB model fingerprints in ~15-20s.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import torch
from safetensors import safe_open

from src import RESULTS_VERSION, __version__

log = logging.getLogger(__name__)


@dataclass
class ModelFingerprint:
    """SHA256 fingerprint of a single model directory.

    ``sha256`` folds the detected architecture, every tensor key, and every
    tensor's raw bytes into one digest. Two byte-identical models always share
    it, even when their files are sharded differently.
    """

    name: str
    path: str
    architecture: str
    tensor_count: int
    total_bytes: int
    sha256: str
    elapsed_seconds: float


def _hash_tensor(t: torch.Tensor) -> str:
    """SHA256 of a tensor's raw bytes, dtype-agnostic.

    Reinterprets storage as uint8 so bf16 / fp16 / fp32 / int8 tensors are all
    hashed by their on-disk byte content. 0-dim (scalar) tensors are reshaped to
    1-D first because ``view`` refuses 0-dim inputs.
    """
    t = t.contiguous()
    if t.dim() == 0:
        t = t.reshape(1)
    return hashlib.sha256(t.view(torch.uint8).numpy()).hexdigest()


def detect_family(keys: set[str]) -> str:
    """Lightweight architecture family detection from a sample of tensor keys."""
    if any("layer_scalar" in k or "per_layer_projection" in k for k in keys):
        return "gemma4"
    if any("language_model" in k for k in keys):
        return "qwen3.5"
    if any(".experts." in k for k in keys):
        return "glm"
    return "qwen3"


def fingerprint_model(model_dir: str | Path, name: str = "") -> ModelFingerprint:
    """Compute the SHA256 fingerprint of a model (CPU-only).

    Each shard is opened exactly once and every tensor in it is hashed in turn,
    so the per-tensor cost is dominated by the mmap read rather than repeated
    file opens. The final digest folds architecture + sorted (key, hash) pairs,
    making it independent of shard boundaries and key iteration order.

    Raises:
        FileNotFoundError: if the directory contains no ``*.safetensors`` files.
    """
    model_dir = Path(model_dir)
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files found in: {model_dir}")

    digest = hashlib.sha256()
    tensor_count = 0
    total_bytes = 0
    family = ""
    t0 = time.time()

    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as sf:  # type: ignore[no-untyped-call]
            shard_keys = list(sf.keys())
            if not family:
                family = detect_family(set(shard_keys))
            for raw_key in shard_keys:
                tensor: torch.Tensor = sf.get_tensor(raw_key)
                tensor_hash = _hash_tensor(tensor)
                digest.update(raw_key.encode())
                digest.update(tensor_hash.encode())
                tensor_count += 1
                total_bytes += tensor.nelement() * tensor.element_size()

    if not family:
        family = detect_family(set())  # empty -> default qwen3

    digest.update(family.encode())

    return ModelFingerprint(
        name=name or model_dir.name,
        path=str(model_dir.resolve()),
        architecture=family,
        tensor_count=tensor_count,
        total_bytes=total_bytes,
        sha256=digest.hexdigest(),
        elapsed_seconds=time.time() - t0,
    )


def group_duplicates(fingerprints: list[ModelFingerprint]) -> dict[str, object]:
    """Group fingerprints by SHA256 and return a JSON-serializable report.

    Any group with more than one model is a set of byte-identical reuploads.
    """
    by_hash: dict[str, list[str]] = {}
    for fp in fingerprints:
        by_hash.setdefault(fp.sha256, []).append(fp.name)

    duplicate_groups = [sorted(names) for names in by_hash.values() if len(names) > 1]
    duplicate_groups.sort()

    return {
        "metadata": {
            "tool": "abliterlitics",
            "version": __version__,
            "results_version": RESULTS_VERSION,
            "module": "weight.duplicate_detect",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "model_count": len(fingerprints),
        "duplicate_group_count": len(duplicate_groups),
        "duplicates": duplicate_groups,
        "models": [
            {
                "name": fp.name,
                "architecture": fp.architecture,
                "tensor_count": fp.tensor_count,
                "total_bytes": fp.total_bytes,
                "sha256": fp.sha256,
                "elapsed_seconds": round(fp.elapsed_seconds, 2),
            }
            for fp in sorted(fingerprints, key=lambda x: x.name)
        ],
    }


def scan_models(
    models_dir: str | Path,
    subset: list[str] | None = None,
) -> list[ModelFingerprint]:
    """Fingerprint every model directory under ``models_dir`` (or a subset).

    Args:
        models_dir: Directory whose immediate children are model dirs and must
            each contain at least one ``*.safetensors`` file.
        subset: Optional list of directory names to restrict the scan to.
    """
    models_dir = Path(models_dir)
    subset_set = set(subset) if subset else None
    candidates = sorted(d for d in models_dir.iterdir() if d.is_dir())
    if subset_set is not None:
        candidates = [d for d in candidates if d.name in subset_set]
        missing = subset_set - {d.name for d in candidates}
        if missing:
            log.warning("Requested models not found in %s: %s", models_dir, sorted(missing))

    fingerprints: list[ModelFingerprint] = []
    for i, d in enumerate(candidates, 1):
        if not list(d.glob("*.safetensors")):
            log.info("[%d/%d] SKIP %s (no safetensors)", i, len(candidates), d.name)
            continue
        log.info("[%d/%d] fingerprinting %s ...", i, len(candidates), d.name)
        fp = fingerprint_model(d, name=d.name)
        log.info(
            "  -> %s: arch=%s tensors=%d bytes=%.2fGB sha256=%.16s in %.1fs",
            d.name,
            fp.architecture,
            fp.tensor_count,
            fp.total_bytes / 1e9,
            fp.sha256,
            fp.elapsed_seconds,
        )
        fingerprints.append(fp)
    return fingerprints


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU-only SHA256 fingerprinting of model directories",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory whose children are model dirs (default: models)",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated subset of model directory names to scan",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write the full JSON report to this path",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    subset = [s.strip() for s in args.models.split(",")] if args.models else None
    fingerprints = scan_models(args.models_dir, subset=subset)
    if not fingerprints:
        log.error("No models with safetensors found under %s", args.models_dir)
        return

    report = group_duplicates(fingerprints)
    dup_groups = cast("list[list[str]]", report["duplicates"])
    dup_count = len(dup_groups)

    log.info("=" * 70)
    log.info("SHA256 SCAN: %d models", report["model_count"])
    if dup_count == 0:
        log.info("No byte-identical duplicates found.")
    else:
        log.info("IDENTICAL (byte-for-byte reuploads) -- %d group(s):", dup_count)
        for group in dup_groups:
            log.info("  %s", ", ".join(group))
    log.info("=" * 70)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        log.info("Saved report: %s", out)


if __name__ == "__main__":
    main()
