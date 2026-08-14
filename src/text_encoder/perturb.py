"""Counterfactual conditioning perturbations for the control battery (Phase 4).

Synthesizes modified conditioning tensors from cached artifacts so the DiT's
response can be attributed: unstructured noise vs structured edits, syntax vs
meaning destruction, intended direction vs random direction (Spingarn /
deepseek-R1 / hy3-C4 control set — journals #122, #123).

All synthesizers are PURE functions of cached tensors (seeded, stateless).
Doses are matched to a target MSE per prompt so each control is comparable to
a real variant's measured divergence at the DiT input.

Terminology (journals #117/#118): controls measure the DiT's response to
DIVERGENCE, typed by construction (noise / shift / destruction).
"""

from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)


def _per_prompt_mse(base: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Mean squared error per prompt between [n, L, d] tensors -> [n]."""
    return ((base - other) ** 2).mean(dim=(1, 2))


def _match_mse(delta: torch.Tensor, target_mse: torch.Tensor) -> torch.Tensor:
    """Scale per-prompt deltas so each prompt's MSE equals its target.

    delta: [n, L, d]; target_mse: [n]. Returns scaled delta.
    """
    current = (delta**2).mean(dim=tuple(range(1, delta.dim())))  # [n]
    scale = torch.sqrt(target_mse / current.clamp(min=1e-20))
    return delta * scale.view(-1, *([1] * (delta.dim() - 1)))


# ---------------------------------------------------------------------------
# Unstructured / structured additive perturbations
# ---------------------------------------------------------------------------


def gaussian_delta(
    base: torch.Tensor,
    target_mse: torch.Tensor | float,
    seed: int = 1000,
    generator_device: str = "cpu",
) -> torch.Tensor:
    """Zero-mean Gaussian delta, per-prompt MSE matched to target.

    Args:
        base: [n, L, d] reference conditioning (shape donor only).
        target_mse: scalar or [n] per-prompt target MSE.
        seed: reproducibility.
    """
    g = torch.Generator(device=generator_device).manual_seed(seed)
    delta = torch.randn(base.shape, generator=g, dtype=torch.float32)
    target = (
        torch.full((base.shape[0],), float(target_mse)) if isinstance(target_mse, (int, float)) else target_mse.float()
    )
    return _match_mse(delta, target)


def rank1_random_delta(
    base: torch.Tensor,
    target_mse: torch.Tensor | float,
    orthogonal_to: torch.Tensor | None = None,
    seed: int = 2000,
) -> torch.Tensor:
    """Random fixed DIRECTION (same unit vector for all prompts/tokens), magnitude
    scaled per prompt to the target MSE. The structured-but-meaningless control
    (deepseek-R1): separates "DiT ignores low-rank structure" from "DiT ignores
    this specific direction".

    orthogonal_to: optional [d] vector to orthogonalize against (e.g. the
    taboo direction) so the control cannot accidentally align with it.
    """
    d = base.shape[-1]
    g = torch.Generator().manual_seed(seed)
    direction = torch.randn(d, generator=g, dtype=torch.float32)
    if orthogonal_to is not None:
        o = orthogonal_to.float()
        proj = (direction @ o) / o.dot(o).clamp(min=1e-20)
        direction = direction - proj * o
    direction = direction / direction.norm().clamp(min=1e-12)
    delta = direction.view(1, 1, d).expand_as(base).clone()
    target = (
        torch.full((base.shape[0],), float(target_mse)) if isinstance(target_mse, (int, float)) else target_mse.float()
    )
    scaled: torch.Tensor = _match_mse(delta, target)
    return scaled


def _derangement(n: int, generator: torch.Generator) -> torch.Tensor:
    """Random permutation with NO fixed points (rejection sampling, expected O(1) retries)."""
    while True:
        perm = torch.randperm(n, generator=generator)
        if not bool((perm == torch.arange(n)).any()):
            return perm


def direction_injection(
    base: torch.Tensor,
    direction: torch.Tensor,
    target_norm: torch.Tensor | float,
) -> torch.Tensor:
    """Add a fixed unit direction at a per-prompt target L2 norm.

    The taboo-direction counterfactual: base + alpha * d_taboo with alpha
    chosen so ||delta||_2 per prompt matches target_norm (e.g. the heretic
    delta's mean norm).
    """
    if base.shape[-1] != direction.shape[-1]:
        raise ValueError(f"direction dim {direction.shape[-1]} != base dim {base.shape[-1]}")
    unit = direction.float() / direction.float().norm().clamp(min=1e-12)
    target = (
        torch.full((base.shape[0],), float(target_norm))
        if isinstance(target_norm, (int, float))
        else target_norm.float()
    )
    alpha = target.view(-1, 1, 1)
    injected: torch.Tensor = unit.view(1, 1, -1).expand_as(base).clone() * alpha
    return injected


def delta_scale(base: torch.Tensor, variant: torch.Tensor, alpha: float) -> torch.Tensor:
    """alpha-scaled real delta: alpha * (variant - base) — the dose-response
    control on the ACTUAL structured heretic shift (alpha in {0.5, 1, 2, 10})."""
    if base.shape != variant.shape:
        raise ValueError(f"shape mismatch: {tuple(base.shape)} vs {tuple(variant.shape)}")
    return alpha * (variant.float() - base.float())


# ---------------------------------------------------------------------------
# Structure destructions (Spingarn pair)
# ---------------------------------------------------------------------------


def token_order_shuffle(base: torch.Tensor, length_mask: torch.Tensor | None = None, seed: int = 3000) -> torch.Tensor:
    """Shuffle token ORDER within each prompt: destroys syntax, keeps the
    multiset of word representations (Spingarn order-control). Real (non-pad)
    positions only when a mask is given; pad positions stay in place.
    """
    g = torch.Generator().manual_seed(seed)
    out = base.clone()
    for i in range(base.shape[0]):
        n = int(length_mask[i].sum().item()) if length_mask is not None else base.shape[1]
        if n > 1:
            perm = torch.randperm(n, generator=g)
            out[i, :n] = base[i, perm]
    return out


def embedding_assignment_shuffle(
    base: torch.Tensor, length_mask: torch.Tensor | None = None, seed: int = 4000
) -> torch.Tensor:
    """Shuffle which TOKEN each embedding lands on via a derangement: destroys
    word meaning/assignment, keeps positions and the embedding multiset
    (Spingarn meaning-control). Every real position receives a DIFFERENT
    token's embedding (no fixed points).
    """
    g = torch.Generator().manual_seed(seed)
    out = base.clone()
    for i in range(base.shape[0]):
        n = int(length_mask[i].sum().item()) if length_mask is not None else base.shape[1]
        if n > 1:
            perm = _derangement(n, g)
            out[i, :n] = base[i, perm]
    result: torch.Tensor = out
    return result


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------


def null_conditioning(base: torch.Tensor) -> torch.Tensor:
    """Zero-information anchor: same shape, all zeros (empty conditioning)."""
    return torch.zeros_like(base)


def per_prompt_delta_norm(delta: torch.Tensor) -> torch.Tensor:
    """Mean per-token L2 norm of a delta [n, L, d] per prompt -> [n]."""
    norms: torch.Tensor = delta.norm(dim=-1).mean(dim=-1)
    return norms


def per_prompt_delta_mse(base: torch.Tensor, variant: torch.Tensor) -> torch.Tensor:
    """Per-prompt MSE of a real variant delta (dose reference for matching)."""
    return _per_prompt_mse(base.float(), variant.float())
