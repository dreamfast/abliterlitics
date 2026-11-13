"""Directional analysis — is the abliteration delta a direction or noise?

The mechanistic claim behind "abliterated text encoders" is that the edit
maps taboo representations toward benign ones (refusal-direction removal).
This module tests that claim geometrically on cached activations:

For each encoder layer (mean-pooled per prompt):
- ``d_taboo``  — base-model taboo direction: unit(mean(taboo) − mean(innocent))
- ``Δ_i``      — per-prompt ablation delta (variant − reference, matched quant)
- ``cos_i``    — cos(Δ_i, d_taboo): does the delta move along the taboo axis?
- ``toward_innocent_i`` — signed projection of Δ_i onto
  (innocent_centroid − base_i), normalized: does the delta move taboo
  prompts TOWARD the innocent cluster?
- ``orth_fraction`` — share of ||Δ||² orthogonal to d_taboo (noise share)
- permutation null — Δ rows shuffled across prompts give the null cos
  distribution; the z-score separates signal from chance alignment.

Interpretation guide:
- abliteration-as-direction  → large |cos| (z ≫ 2), toward_innocent ≫ 0
- abliteration-as-noise      → |cos| ≈ null, orth_fraction ≈ 1,
                                toward_innocent ≈ 0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

log = logging.getLogger(__name__)

DEFAULT_TABOO_CATEGORIES = ("violent", "sexual", "extreme")
DEFAULT_REFERENCE_CATEGORIES = ("innocent",)


@dataclass(frozen=True)
class LayerDirectional:
    """Directional metrics for one layer."""

    layer: int
    n_taboo: int
    n_ref: int
    delta_norm_taboo: float  # mean ||Δ|| on taboo prompts
    delta_norm_ref: float  # mean ||Δ|| on reference (innocent) prompts
    cos_taboo_mean: float  # mean cos(Δ, d_taboo) on taboo prompts
    cos_taboo_abs_mean: float  # mean |cos|
    cos_null_std: float  # std of cos under prompt-permutation null
    cos_z: float  # (|cos_mean| − null_mean) / null_std
    orth_fraction: float  # mean 1 − cos² (share of energy ⊥ d_taboo)
    toward_innocent_mean: float  # normalized signed projection, taboo prompts
    toward_innocent_null_std: float
    toward_innocent_z: float


@dataclass
class DirectionalReport:
    """Full directional report for one variant vs one reference."""

    variant_id: str
    reference_id: str
    taboo_categories: tuple[str, ...]
    reference_categories: tuple[str, ...]
    layers: list[LayerDirectional] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        if not self.layers:
            return {}
        peak_abs = max(self.layers, key=lambda r: abs(r.cos_taboo_mean))
        peak_z = max(self.layers, key=lambda r: r.cos_z)
        return {
            "n_layers": len(self.layers),
            "peak_abs_cos_layer": peak_abs.layer,
            "peak_abs_cos": peak_abs.cos_taboo_mean,
            "peak_z_layer": peak_z.layer,
            "peak_z": peak_z.cos_z,
            "mean_orth_fraction": sum(r.orth_fraction for r in self.layers) / len(self.layers),
        }


def _mean(acts: torch.Tensor) -> torch.Tensor:
    return acts.mean(dim=0).clone()


def taboo_direction(
    base_acts: torch.Tensor,
    categories: list[str],
    taboo: tuple[str, ...] = DEFAULT_TABOO_CATEGORIES,
    reference: tuple[str, ...] = DEFAULT_REFERENCE_CATEGORIES,
) -> torch.Tensor:
    """Unit taboo direction from BASE activations: mean(taboo) − mean(ref).

    Raises ValueError if either category group is absent.
    """
    taboo_mask = torch.tensor([c in taboo for c in categories], dtype=torch.bool)
    ref_mask = torch.tensor([c in reference for c in categories], dtype=torch.bool)
    if not taboo_mask.any() or not ref_mask.any():
        raise ValueError(f"category groups missing: taboo={taboo} ref={reference} in {sorted(set(categories))}")
    d = _mean(base_acts[taboo_mask]) - _mean(base_acts[ref_mask])
    n = d.norm()
    if n <= 0:
        raise ValueError("degenerate taboo direction (zero norm)")
    result: torch.Tensor = d / n  # torch stubs type Tensor/0-dim div as Any
    return result


def _permutation_null(
    deltas: torch.Tensor,
    direction: torch.Tensor,
    n_perm: int = 200,
    generator: torch.Generator | None = None,
) -> tuple[float, float]:
    """Null distribution of mean cos under prompt relabeling.

    Under H0 (deltas exchangeable across prompts), per-prompt cosines are
    exchangeable — shuffling the taboo subset membership is equivalent to
    shuffling the cos values. Returns (null_mean, null_std) of the mean cos
    over a same-sized random subset.
    """
    if generator is None:
        generator = torch.Generator().manual_seed(42)
    m = deltas.shape[0]
    cos = torch.nn.functional.cosine_similarity(deltas, direction.unsqueeze(0), dim=1)
    subset = max(m // 2, 1)
    stats = [float(cos[torch.randperm(m, generator=generator)[:subset]].mean()) for _ in range(n_perm)]
    t = torch.tensor(stats)
    return float(t.mean()), float(t.std().clamp_min(1e-12))


def analyze_layer(
    base_acts: torch.Tensor,
    var_acts: torch.Tensor,
    categories: list[str],
    layer: int,
    taboo: tuple[str, ...] = DEFAULT_TABOO_CATEGORIES,
    reference: tuple[str, ...] = DEFAULT_REFERENCE_CATEGORIES,
    n_perm: int = 200,
) -> LayerDirectional:
    """Directional metrics for one layer's pooled activations [n, d]."""
    if base_acts.ndim == 3 and base_acts.shape[1] == 1:
        base_acts = base_acts[:, 0, :]  # pooled streams cache as [n, 1, d]
    if var_acts.ndim == 3 and var_acts.shape[1] == 1:
        var_acts = var_acts[:, 0, :]
    if base_acts.ndim != 2 or var_acts.ndim != 2:
        raise ValueError(f"expected pooled [n, d] activations, got {tuple(base_acts.shape)} / {tuple(var_acts.shape)}")
    if base_acts.shape != var_acts.shape:
        raise ValueError(f"shape mismatch: base {tuple(base_acts.shape)} vs var {tuple(var_acts.shape)}")

    d = taboo_direction(base_acts, categories, taboo, reference)
    deltas = var_acts - base_acts  # [n, d]

    taboo_mask = torch.tensor([c in taboo for c in categories], dtype=torch.bool)
    ref_mask = torch.tensor([c in reference for c in categories], dtype=torch.bool)

    d_taboo = deltas[taboo_mask]
    d_ref = deltas[ref_mask]

    delta_norm_taboo = float(d_taboo.norm(dim=1).mean())
    delta_norm_ref = float(d_ref.norm(dim=1).mean())

    cos = torch.nn.functional.cosine_similarity(d_taboo, d.unsqueeze(0), dim=1)
    cos_mean = float(cos.mean())
    cos_abs_mean = float(cos.abs().mean())

    # orthogonal fraction over taboo prompts: 1 − cos²(Δ, d)
    orth = 1.0 - cos**2
    orth_fraction = float(orth.mean().clamp(0, 1))

    # toward-innocent: projection of Δ on (innocent_centroid − base_i)
    innocent_centroid = _mean(base_acts[ref_mask])
    ti = torch.stack(
        [
            torch.dot(deltas[i], innocent_centroid - base_acts[i])
            / (innocent_centroid - base_acts[i]).norm().clamp_min(1e-12)
            for i in torch.where(taboo_mask)[0].tolist()
        ]
    )
    ti_mean = float(ti.mean())

    # permutation nulls (fixed seed for reproducibility)
    null_mean, null_std = _permutation_null(d_taboo, d, n_perm=n_perm)
    cos_z = (abs(cos_mean) - null_mean) / null_std

    # toward-innocent null: permute deltas across prompts, recompute TI
    gen = torch.Generator().manual_seed(43)
    n = deltas.shape[0]
    taboo_idx = torch.where(taboo_mask)[0]
    ti_stats: list[float] = []
    for _ in range(n_perm):
        perm = torch.randperm(n, generator=gen)
        shuffled = deltas[perm]
        ti_p = torch.stack(
            [
                torch.dot(shuffled[i], innocent_centroid - base_acts[i])
                / (innocent_centroid - base_acts[i]).norm().clamp_min(1e-12)
                for i in taboo_idx.tolist()
            ]
        )
        ti_stats.append(float(ti_p.mean()))
    ti_t = torch.tensor(ti_stats)
    ti_null_std = float(ti_t.std().clamp_min(1e-12))
    ti_z = (ti_mean - float(ti_t.mean())) / ti_null_std

    return LayerDirectional(
        layer=layer,
        n_taboo=int(taboo_mask.sum()),
        n_ref=int(ref_mask.sum()),
        delta_norm_taboo=delta_norm_taboo,
        delta_norm_ref=delta_norm_ref,
        cos_taboo_mean=cos_mean,
        cos_taboo_abs_mean=cos_abs_mean,
        cos_null_std=null_std,
        cos_z=float(cos_z),
        orth_fraction=orth_fraction,
        toward_innocent_mean=ti_mean,
        toward_innocent_null_std=ti_null_std,
        toward_innocent_z=float(ti_z),
    )
