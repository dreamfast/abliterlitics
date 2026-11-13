"""Core data types for text-encoder conditioning analysis.

The central abstraction is ``ConditioningArtifact`` — a bag of named
conditioning streams captured at a named pipeline stage. This object crosses
the boundary between model-specific backends (which know how to load a
specific text encoder and hook its internals) and the model-agnostic metrics
core (which operates on pure tensors).

See ``ABL_ITERLITICS_PLAN.md`` (in ``.llm/``) for the full design.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

log = logging.getLogger(__name__)


class Stage(str, Enum):
    """Pipeline stage at which a conditioning tensor is captured.

    The further downstream, the closer to what the DiT actually sees.
    """

    RAW_HIDDEN = "raw_hidden"  # encoder hidden states, pre-projection
    PROJECTED = "projected"  # after the linear projection(s)
    CONNECTOR_OUT = "connector_out"  # after any 1D connector transformer


class Pooling(str, Enum):
    """How multiple layers/tokens collapse into a single feature vector."""

    NONE = "none"  # no pooling — keep full [n, L, d]
    MEAN = "mean"  # mean across the sequence dimension


@dataclass(frozen=True)
class StreamSpec:
    """Declared by a backend. Describes one conditioning stream it produces.

    Frozen so it can be used as a dict key / set member.
    """

    name: str  # "video_proj", "layer48", etc.
    source_encoder: str  # "gemma3", "qwen3vl", "gemma4"
    layers: tuple[int, ...] | str  # (48,) | (0, 8, ..., 48) | "all"
    pooling: Pooling  # how multiple layers/tokens collapse
    stage: Stage  # pipeline stage this stream is captured at
    dim: int  # resulting feature dim d

    def __post_init__(self) -> None:
        if not isinstance(self.layers, tuple | str):
            raise TypeError(f"layers must be tuple[int, ...] or str, got {type(self.layers).__name__}")


@dataclass
class Stream:
    """One conditioning stream: activations + optional attention mask.

    Attributes:
        name: Stream name (must match a ``StreamSpec.name``).
        spec: The declared specification.
        activations: Always ``[n_prompts, seq_len, d]`` in float32.
                     For pooled streams, seq_len is squeezed to 1.
        mask: Attention mask ``[n_prompts, seq_len]`` (1 = real token,
              0 = pad). ``None`` if the stream is already pooled (seq_len <= 1)
              or if there is no padding.
    """

    name: str
    spec: StreamSpec
    activations: torch.Tensor
    mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.activations.dtype != torch.float32:
            self.activations = self.activations.float()
        if self.activations.ndim == 2:
            # Pooled: [n, d] -> expand to [n, 1, d] for uniform handling
            self.activations = self.activations.unsqueeze(1)
        if self.activations.ndim != 3:
            raise ValueError(f"activations must be [n, L, d] or [n, d], got shape {tuple(self.activations.shape)}")
        n_prompts, seq_len, dim = self.activations.shape
        # Cross-check dim against spec
        if dim != self.spec.dim:
            raise ValueError(f"activations dim ({dim}) != spec.dim ({self.spec.dim}) for stream '{self.name}'")
        if seq_len > 1 and self.mask is None:
            raise ValueError(f"mask is required when seq_len > 1 (stream={self.name}, seq_len={seq_len})")
        if self.mask is not None:
            if self.mask.ndim != 2:
                raise ValueError(f"mask must be 2D [n, L], got shape {tuple(self.mask.shape)}")
            if self.mask.shape[0] != n_prompts:
                raise ValueError(f"mask rows ({self.mask.shape[0]}) != activations rows ({n_prompts})")
            if self.mask.shape[1] != seq_len:
                raise ValueError(
                    f"mask cols ({self.mask.shape[1]}) != activations seq_len ({seq_len}) for stream '{self.name}'"
                )


@dataclass
class ConditioningArtifact:
    """A bag of conditioning streams captured at a named pipeline stage.

    This is the shared object that crosses the boundary between backends
    (model-specific) and the metrics core (model-agnostic). Backends produce
    artifacts; ``compare()`` consumes them.
    """

    model: str  # "ltx2", "h3", "ltx25"
    backend: str  # backend class name
    ckpt_id: str  # path or repo id
    stage: Stage
    streams: dict[str, Stream]
    prompts: list[str]
    categories: list[str]  # len == len(prompts), aligned
    dtype: str  # source dtype: "bf16" | "fp32" | "fp8" | ...
    device: str  # "cpu" | "cuda"
    meta: dict[str, Any] = field(default_factory=dict)  # tokenization hash, layer count, etc.

    def __post_init__(self) -> None:
        n = len(self.prompts)
        if len(self.categories) != n:
            raise ValueError(f"len(categories) ({len(self.categories)}) != len(prompts) ({n})")
        for name, stream in self.streams.items():
            if stream.activations.shape[0] != n:
                raise ValueError(f"stream '{name}' has {stream.activations.shape[0]} rows but there are {n} prompts")
            if stream.name != name:
                raise ValueError(f"stream key '{name}' does not match stream.name '{stream.name}'")

    @property
    def n_prompts(self) -> int:
        """Number of prompts in this artifact."""
        return len(self.prompts)

    @property
    def prompt_set_hash(self) -> str:
        """Deterministic hash of the prompt set (for cache keying + alignment checks).

        The hash is over (prompt_text, category) pairs, sorted, so reordering
        prompts with the same content produces the same hash.
        """
        pairs = sorted(zip(self.prompts, self.categories, strict=True))
        payload = "\n".join(f"{p}\t{c}" for p, c in pairs)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def assert_aligned(self, other: ConditioningArtifact) -> None:
        """Assert two artifacts share the same prompts, categories, and prompt order.

        Does NOT check tokenization (that's backend-specific). Use
        ``Backend.tokenize`` for that.
        """
        if self.prompts != other.prompts:
            raise ValueError("Prompt lists do not match between artifacts")
        if self.categories != other.categories:
            raise ValueError("Category lists do not match between artifacts")
        if self.prompt_set_hash != other.prompt_set_hash:
            # Should be unreachable if the above pass, but as a belt-and-suspenders
            raise ValueError("prompt_set_hash mismatch despite matching prompts (internal error)")
