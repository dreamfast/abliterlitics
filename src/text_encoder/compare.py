"""Comparison orchestrator + noise-floor-relative damage computation.

This module ties the metrics core together: given two ConditioningArtifacts
(base and variant) plus optional reference-floor artifacts, compute all
metrics per-stream and express the damage relative to two anchors:

1. Quantization floor (primary, stable):
     damage_quant_units = MSE(base, variant) / MSE(base, quant_floor)
   "damage 0.1 = ten times smaller than one quantization step."

2. Semantic ceiling (interpretability anchor):
     damage_semantic_units = pooled_MSE(base, variant) / pairwise_pooled_MSE(base)
   The denominator is the mean conditioning distance between DIFFERENT
   prompts (a meaning change). "damage 0.01 = the DiT input moves 1% of a
   meaning change." Pooling is used on both sides so token-aligned damage
   and cross-prompt distances live on the same scale.

CKA is reported as a secondary similarity metric (bounded [0, 1]).

Bootstrap CIs (reviewer amendment C4): the damage ratio is resampled JOINTLY
(numerator + denominator on the same prompt indices) via
``bootstrap_ratio_ci``; per-prompt metrics (MSE, cosine) get standard
nonparametric CIs. CKA CIs are omitted (pooled statistic — recomputation
per resample is O(n^2 d B) and not needed for the headline claims).

Per reviewer amendment C2: the numerical_floor parameter is demoted to an
optional reported reference (bf16-vs-fp16, not fp32) — it is NOT used in
the damage formula to avoid loading base in fp32 (which would need ~128GB
RAM for H3 32B).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from src.text_encoder.artifact import ConditioningArtifact
from src.text_encoder.bootstrap import DEFAULT_N_BOOT, bootstrap_ci, bootstrap_ratio_ci
from src.text_encoder.metrics import (
    MetricResult,
    cka_linear,
    cka_rbf,
    cosine_all_tokens,
    cosine_last_token,
    pool_mean,
    rank_correlation,
)

log = logging.getLogger(__name__)


@dataclass
class StreamReport:
    """All metrics for one (stream, stage) pair."""

    stream: str
    stage: str
    n_prompts: int
    cka_linear: MetricResult
    cka_rbf: MetricResult
    cos_all: MetricResult
    cos_last: MetricResult
    mse: MetricResult
    l2: MetricResult
    rank_corr: MetricResult
    damage_quant_units: MetricResult | None = None  # None if no quant floor provided
    semantic_ceiling_mse: float | None = None  # pairwise pooled MSE between distinct base prompts
    damage_semantic_units: MetricResult | None = None  # pooled damage / semantic ceiling
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)
    # per_category = {"innocent": {"cka_linear": 0.95, "mse": 0.001, ...}, ...}


@dataclass
class VariantReport:
    """Full report for one variant vs base."""

    variant_id: str
    dtype: str
    streams: list[StreamReport]
    # GGUF container provenance (additive; None for safetensors variants).
    gguf: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Per-prompt + pairwise helpers (bootstrap + semantic ceiling)
# ---------------------------------------------------------------------------


def _per_prompt_sq_err(base_acts: torch.Tensor, var_acts: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Per-prompt mean squared error: [n] float32 tensor.

    Masked positions are excluded. The mean of this vector equals the
    global masked MSE (equal per-prompt weights): numerator sums err over
    (L, d) at real tokens, denominator is the count of real elements
    (real_tokens * d).
    """
    err = (base_acts - var_acts) ** 2  # [n, L, d]
    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)  # [n, L, 1]
        m = mask.float()
        n_elem = m.sum(dim=(1, 2)) * err.shape[-1]  # real tokens * d
        per_tok = (err * m).sum(dim=(1, 2)) / n_elem.clamp(min=1.0)
    else:
        per_tok = err.mean(dim=(1, 2))
    return per_tok


def _semantic_ceiling_pooled(base_pooled: torch.Tensor, max_pairs: int = 4000, seed: int = 42) -> float:
    """Mean squared distance between pooled conditioning of DISTINCT prompts.

    The interpretability anchor: how far the conditioning moves when the
    content (meaning) changes entirely. Pairs are subsampled uniformly when
    n*(n-1)/2 exceeds ``max_pairs`` (deterministic via seed).

    Args:
        base_pooled: [n, d] mean-pooled per-prompt base activations.
        max_pairs: Cap on evaluated pairs (n=60 -> 1770 pairs, under cap).

    Returns:
        Scalar pairwise MSE.
    """
    n = base_pooled.shape[0]
    if n < 2:
        return float("nan")
    iu = torch.triu_indices(n, n, offset=1)
    total = iu.shape[1]
    if total > max_pairs:
        g = torch.Generator().manual_seed(seed)
        keep = torch.randperm(total, generator=g)[:max_pairs]
        iu = iu[:, keep]
    a = base_pooled[iu[0]]
    b = base_pooled[iu[1]]
    return float(((a - b) ** 2).mean().item())


def _pooled_mse(base_acts: torch.Tensor, var_acts: torch.Tensor, base_mask: torch.Tensor | None) -> float:
    """MSE between mean-pooled per-prompt vectors (masked)."""
    bp = pool_mean(base_acts, base_mask)
    vp = pool_mean(var_acts, base_mask)
    return float(((bp - vp) ** 2).mean().item())


def compute_stream_metrics(
    base_activations: torch.Tensor,
    variant_activations: torch.Tensor,
    base_mask: torch.Tensor | None,
    variant_mask: torch.Tensor | None,
) -> dict[str, float]:
    """Compute all pairwise metrics for one stream's activations.

    Args:
        base_activations: [n, L, d] float32 tensor.
        variant_activations: [n, L, d] float32 tensor (same shape).
        base_mask: [n, L] or None.
        variant_mask: [n, L] or None.

    Returns:
        Dict of metric_name -> value.
    """
    # Pool for CKA and rank_correlation (which need [n, d])
    base_pooled = pool_mean(base_activations, base_mask)
    variant_pooled = pool_mean(variant_activations, variant_mask)

    # MSE/L2 use the per-prompt-weighted mean (equal weight per prompt —
    # prompts with more tokens don't dominate). This is the same weighting
    # the bootstrap CI uses, so point estimate and CI are consistent.
    # Masking (pad exclusion) is handled inside _per_prompt_sq_err (W1).
    per_prompt = _per_prompt_sq_err(base_activations, variant_activations, base_mask)
    mse_val = float(per_prompt.mean().item())
    l2_val = float(mse_val**0.5)

    return {
        "cka_linear": cka_linear(base_pooled, variant_pooled),
        "cka_rbf": cka_rbf(base_pooled, variant_pooled),
        "cos_all": cosine_all_tokens(base_activations, variant_activations, base_mask),
        "cos_last": cosine_last_token(base_activations, variant_activations, base_mask),
        "mse": mse_val,
        "l2": l2_val,
        "rank_corr": rank_correlation(base_pooled, variant_pooled),
    }


def compute_damage(
    mse_base_variant: float,
    mse_base_quant: float,
) -> float:
    """Compute damage in quant-floor units.

    damage = MSE(base, variant) / MSE(base, fake_quant)

    A damage of 0.1 = "ten times smaller than the quant floor, almost
    certainly imperceptible." A damage of 3.4 = "three times worse than
    dropping to fp8."

    Returns inf if the quant floor is zero.
    """
    if mse_base_quant < 1e-15:
        log.warning("compute_damage: quant-floor MSE is near-zero, returning inf")
        return float("inf")
    return mse_base_variant / mse_base_quant


def compare(
    base: ConditioningArtifact,
    variant: ConditioningArtifact,
    quant_floor: ConditioningArtifact | None = None,
    numerical_floor: ConditioningArtifact | None = None,
    floor_base: ConditioningArtifact | None = None,
    n_boot: int = DEFAULT_N_BOOT,
) -> VariantReport:
    """Compare a variant artifact against a base/reference.

    Computes all metrics per matching stream, plus:
    - damage in quant-floor units (with JOINT bootstrap CI) when a floor is given
    - semantic ceiling + damage in semantic units (pooled, vs meaning-change)
    - bootstrap CIs on per-prompt metrics (MSE, cosine)
    - per-category breakdown when categories are non-uniform

    Args:
        base: Base (or reference) model artifact — the comparison anchor.
        variant: Variant model artifact (must share prompts/categories).
        quant_floor: Noise-floor artifact for damage computation.
                     If None, damage_quant_units is not computed.
        numerical_floor: Optional bf16-vs-fp16 reference (reported but not
                         used in the damage formula per reviewer C2).
        floor_base: The artifact the quant floor is measured AGAINST when
                    it differs from ``base``. Defaults to ``base``.
        n_boot: Bootstrap resamples for CIs (0 disables).

    Returns:
        VariantReport with per-stream StreamReports.

    Raises:
        ValueError: If base and variant don't share prompts/categories.
    """
    base.assert_aligned(variant)
    floor_ref = floor_base if floor_base is not None else base
    if quant_floor is not None:
        floor_ref.assert_aligned(quant_floor)

    streams_report: list[StreamReport] = []
    common_streams = sorted(set(base.streams) & set(variant.streams))

    for stream_name in common_streams:
        base_stream = base.streams[stream_name]
        var_stream = variant.streams[stream_name]

        # Compute aggregate metrics
        metrics = compute_stream_metrics(
            base_stream.activations,
            var_stream.activations,
            base_stream.mask,
            var_stream.mask,
        )

        # Bootstrap CIs for per-prompt metrics (MSE + cosine)
        mse_ci: tuple[float, float] | None = None
        cos_ci: tuple[float, float] | None = None
        if n_boot > 0:
            per_prompt_mse = _per_prompt_sq_err(base_stream.activations, var_stream.activations, base_stream.mask)
            _, lo, hi = bootstrap_ci(per_prompt_mse, n_boot=n_boot)
            mse_ci = (lo, hi)

            # Per-prompt cosine (masked mean over tokens per prompt)
            x = base_stream.activations
            y = var_stream.activations
            dot = (x * y).sum(dim=-1)
            denom = torch.linalg.norm(x, dim=-1) * torch.linalg.norm(y, dim=-1)
            cos_tok = torch.where(denom > 1e-15, dot / denom, torch.zeros_like(dot))
            if base_stream.mask is not None:
                m = base_stream.mask.float()
                per_prompt_cos = (cos_tok * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            else:
                per_prompt_cos = cos_tok.mean(dim=1)
            _, lo_c, hi_c = bootstrap_ci(per_prompt_cos, n_boot=n_boot)
            cos_ci = (lo_c, hi_c)

        # Semantic ceiling + damage in semantic units (pooled representation)
        base_pooled = pool_mean(base_stream.activations, base_stream.mask)
        semantic_ceiling = _semantic_ceiling_pooled(base_pooled)
        pooled_damage = _pooled_mse(base_stream.activations, var_stream.activations, base_stream.mask)
        damage_semantic: MetricResult | None = None
        semantic_ceiling_out: float | None = None
        if semantic_ceiling == semantic_ceiling and semantic_ceiling > 1e-15:  # not NaN, not ~0
            damage_semantic = MetricResult(value=pooled_damage / semantic_ceiling)
            semantic_ceiling_out = semantic_ceiling
        else:
            if semantic_ceiling != semantic_ceiling:
                log.debug("Semantic ceiling NaN for stream %s (n<2)", stream_name)
            # NaN or ~0 ceiling: leave semantic fields unset (serializes None)

        # Compute damage if quant floor is provided.
        # Denominator is measured against floor_ref (the global base), NOT
        # the comparison base — matched-quant runs (heretic_int8 vs
        # base_int8) would otherwise divide by MSE(base_int8, base_int8)=0.
        damage: MetricResult | None = None
        if quant_floor is not None and stream_name in quant_floor.streams:
            qf_stream = quant_floor.streams[stream_name]
            if stream_name in floor_ref.streams:
                fr_stream = floor_ref.streams[stream_name]
                # Numerator and denominator use the SAME weighting: the
                # per-prompt mean (matches metrics["mse"] and the bootstrap).
                num_pp = _per_prompt_sq_err(base_stream.activations, var_stream.activations, base_stream.mask)
                den_pp = _per_prompt_sq_err(fr_stream.activations, qf_stream.activations, fr_stream.mask)
                damage_val = compute_damage(float(num_pp.mean().item()), float(den_pp.mean().item()))
                # Joint bootstrap CI for the ratio (reviewer C4). The returned
                # point IS the plug-in ratio (or bootstrap median under skew)
                # — adopt it so value and interval are always consistent.
                ci: tuple[float, float] | None = None
                if n_boot > 0 and float(den_pp.mean().item()) > 1e-15:
                    point_d, lo_d, hi_d = bootstrap_ratio_ci(num_pp, den_pp, n_boot=n_boot)
                    damage_val = point_d
                    ci = (lo_d, hi_d)
                damage = MetricResult(value=damage_val, ci_low=ci[0] if ci else None, ci_high=ci[1] if ci else None)

        # Per-category breakdown
        per_category: dict[str, dict[str, float]] = {}
        unique_cats = set(base.categories)
        if len(unique_cats) > 1:
            for cat in sorted(unique_cats):
                cat_idx = [i for i, c in enumerate(base.categories) if c == cat]
                if len(cat_idx) < 2:
                    continue
                cat_base = base_stream.activations[cat_idx]
                cat_var = var_stream.activations[cat_idx]
                cat_base_mask = base_stream.mask[cat_idx] if base_stream.mask is not None else None
                cat_var_mask = var_stream.mask[cat_idx] if var_stream.mask is not None else None
                cat_metrics = compute_stream_metrics(cat_base, cat_var, cat_base_mask, cat_var_mask)
                per_category[cat] = cat_metrics

        streams_report.append(
            StreamReport(
                stream=stream_name,
                stage=base.stage.value,
                n_prompts=base.n_prompts,
                cka_linear=MetricResult(value=metrics["cka_linear"]),
                cka_rbf=MetricResult(value=metrics["cka_rbf"]),
                cos_all=MetricResult(
                    value=metrics["cos_all"],
                    ci_low=cos_ci[0] if cos_ci else None,
                    ci_high=cos_ci[1] if cos_ci else None,
                ),
                cos_last=MetricResult(value=metrics["cos_last"]),
                mse=MetricResult(
                    value=metrics["mse"],
                    ci_low=mse_ci[0] if mse_ci else None,
                    ci_high=mse_ci[1] if mse_ci else None,
                ),
                l2=MetricResult(value=metrics["l2"]),
                rank_corr=MetricResult(value=metrics["rank_corr"]),
                damage_quant_units=damage,
                semantic_ceiling_mse=semantic_ceiling_out,
                damage_semantic_units=damage_semantic,
                per_category=per_category,
            )
        )

    return VariantReport(
        variant_id=variant.ckpt_id,
        dtype=variant.dtype,
        streams=streams_report,
        gguf=variant.meta.get("gguf") if isinstance(variant.meta, dict) else None,
    )
