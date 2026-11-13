"""Shared helper: build an architecture-extending variant mock model.

Mirrors the apostate_diode class from AGENTS.md lesson #79: same tensor keys
as the base, but one tensor's shape is extended (e.g. an appended MLP neuron).
Scripts without shape guards crash on these with a torch broadcast RuntimeError;
guarded scripts must skip (and count) them instead.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SHARD_FILENAME = "model-00001-of-00001.safetensors"


def make_arch_extending_variant(src_dir: Path, dest_dir: Path, key: str, growth: int = 2) -> Path:
    """Copy ``src_dir``'s mock model to ``dest_dir`` with one tensor shape-extended.

    All keys are preserved; ``key``'s tensor gets ``growth``x its first dim, so
    base-vs-variant diffs on that key are shape-mismatched while every other
    key loads normally.
    """
    tensors = load_file(str(src_dir / SHARD_FILENAME))
    original = tensors[key]
    grown_shape = (original.shape[0] * growth, *original.shape[1:])
    gen = torch.Generator().manual_seed(7)
    tensors[key] = torch.empty(grown_shape, dtype=original.dtype).normal_(generator=gen).to(original.dtype)
    dest_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(dest_dir / SHARD_FILENAME))
    (dest_dir / "config.json").write_text((src_dir / "config.json").read_text())
    return dest_dir
