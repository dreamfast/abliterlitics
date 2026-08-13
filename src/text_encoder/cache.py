"""Disk cache for ConditioningArtifacts.

Avoids re-encoding models across variant comparisons. The base model + noise
floor artifacts are computed once and cached; each variant comparison reuses
them.

Key includes a `transform` field (reviewer amendment C1) to prevent collisions
between the real base, fp16 reference, and fake-quantized base — all of which
share the same ckpt_id but are different models.

Cache files are torch.save'd dicts containing the artifact's tensors + metadata.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from src.text_encoder.artifact import ConditioningArtifact

log = logging.getLogger(__name__)


def cache_key(
    ckpt_id: str,
    prompt_set_hash: str,
    dtype: str,
    device: str,
    stage: str,
    transform: str = "orig",
) -> str:
    """Deterministic cache key for an artifact.

    Args:
        ckpt_id: Model checkpoint identifier (path or repo ID).
        prompt_set_hash: Hash of the prompt set (from ConditioningArtifact).
        dtype: Source dtype ("bf16", "fp32", "fp8", etc.).
        device: "cpu" or "cuda".
        stage: Pipeline stage ("raw_hidden", "projected", etc.).
        transform: What transform was applied (reviewer C1):
            "orig" — original weights
            "fp16_ref" — fp16 reference for numerical floor
            "fakequant_fp8" — fake fp8 quantization
            "fakequant_int8" — fake int8 quantization

    Returns:
        Hex string cache key.
    """
    payload = f"{ckpt_id}|{prompt_set_hash}|{dtype}|{device}|{stage}|{transform}"
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def save_artifact(artifact: ConditioningArtifact, cache_dir: Path | str) -> Path:
    """Save an artifact to the disk cache.

    Args:
        artifact: The ConditioningArtifact to cache.
        cache_dir: Directory for cache files.

    Returns:
        Path to the written cache file.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Extract transform from meta if present
    transform = artifact.meta.get("transform", "orig")
    key = cache_key(
        artifact.ckpt_id,
        artifact.prompt_set_hash,
        artifact.dtype,
        artifact.device,
        artifact.stage.value,
        transform,
    )
    path = cache_dir / f"{key}.pt"

    # Serialize: streams as dict of {name: (activations, mask, spec_dict)}
    stream_data: dict[str, dict[str, Any]] = {}
    for name, stream in artifact.streams.items():
        stream_data[name] = {
            "activations": stream.activations.cpu(),
            "mask": stream.mask.cpu() if stream.mask is not None else None,
            "spec": {
                "name": stream.spec.name,
                "source_encoder": stream.spec.source_encoder,
                "layers": stream.spec.layers if isinstance(stream.spec.layers, str) else list(stream.spec.layers),
                "pooling": stream.spec.pooling.value,
                "stage": stream.spec.stage.value,
                "dim": stream.spec.dim,
            },
        }

    payload = {
        "model": artifact.model,
        "backend": artifact.backend,
        "ckpt_id": artifact.ckpt_id,
        "stage": artifact.stage.value,
        "streams": stream_data,
        "prompts": artifact.prompts,
        "categories": artifact.categories,
        "dtype": artifact.dtype,
        "device": artifact.device,
        "meta": artifact.meta,
    }

    torch.save(payload, path)
    log.debug("Cached artifact to %s (key=%s)", path, key)
    return path


def load_artifact(cache_dir: Path | str, key: str) -> ConditioningArtifact | None:
    """Load an artifact from the disk cache by key.

    Args:
        cache_dir: Directory for cache files.
        key: Cache key from cache_key().

    Returns:
        ConditioningArtifact or None if not found.
    """
    path = Path(cache_dir) / f"{key}.pt"
    if not path.exists():
        return None

    payload = torch.load(path, weights_only=False)

    # Reconstruct streams
    from src.text_encoder.artifact import Pooling, Stage, Stream, StreamSpec

    streams: dict[str, Stream] = {}
    for name, sd in payload["streams"].items():
        spec_data = sd["spec"]
        layers = spec_data["layers"]
        if isinstance(layers, list):
            layers = tuple(layers)
        spec = StreamSpec(
            name=spec_data["name"],
            source_encoder=spec_data["source_encoder"],
            layers=layers,
            pooling=Pooling(spec_data["pooling"]),
            stage=Stage(spec_data["stage"]),
            dim=spec_data["dim"],
        )
        streams[name] = Stream(
            name=name,
            spec=spec,
            activations=sd["activations"],
            mask=sd["mask"],
        )

    return ConditioningArtifact(
        model=payload["model"],
        backend=payload["backend"],
        ckpt_id=payload["ckpt_id"],
        stage=Stage(payload["stage"]),
        streams=streams,
        prompts=payload["prompts"],
        categories=payload["categories"],
        dtype=payload["dtype"],
        device=payload["device"],
        meta=payload.get("meta", {}),
    )


def get_or_compute(
    cache_dir: Path | str,
    ckpt_id: str,
    prompt_set_hash: str,
    dtype: str,
    device: str,
    stage: str,
    transform: str,
    compute_fn: Callable[[], ConditioningArtifact],
) -> ConditioningArtifact:
    """Get artifact from cache, or compute + cache it.

    Args:
        cache_dir: Cache directory.
        ckpt_id, prompt_set_hash, dtype, device, stage, transform: Cache key components.
        compute_fn: Callable() -> ConditioningArtifact, called on cache miss.

    Returns:
        The artifact (from cache or freshly computed).
    """
    key = cache_key(ckpt_id, prompt_set_hash, dtype, device, stage, transform)
    cached = load_artifact(cache_dir, key)
    if cached is not None:
        log.info("Cache hit: %s (transform=%s, stage=%s)", ckpt_id, transform, stage)
        return cached

    log.info("Cache miss: %s (transform=%s, stage=%s) — computing", ckpt_id, transform, stage)
    artifact: ConditioningArtifact = compute_fn()
    save_artifact(artifact, cache_dir)
    return artifact
