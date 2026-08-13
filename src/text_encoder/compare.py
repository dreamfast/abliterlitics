"""Comparison orchestrator + noise-floor-relative damage computation.

This module ties the metrics core together: given two ConditioningArtifacts
(base and variant) plus optional reference-floor artifacts, compute all
metrics per-stream and express the damage relative to a quantization floor.

The primary damage metric is MSE-based:
    damage_quant_units = MSE(base, variant) / MSE(base, base_fake_quant)

This is stable (MSE is unbounded, scales linearly with perturbation) and
interpretable: "damage 0.1 = ten times smaller than one quantization step."

CKA is reported as a secondary similarity metric (bounded [0, 1]).

Per reviewer amendment C2: the numerical_floor parameter is demoted to an
optional reported reference (bf16-vs-fp16, not fp32) — it is NOT used in
the damage formula to avoid loading base in fp32 (which would need ~128GB
RAM for H3 32B).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from src.text_encoder.artifact import ConditioningArtifact
from src.text_encoder.metrics import (
    MetricResult,
    cka_linear,
    cka_rbf,
    cosine_all_tokens,
    cosine_last_token,
    l2_distance,
    mse,
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
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)
    # per_category = {"innocent": {"cka_linear": 0.95, "mse": 0.001, ...}, ...}


@dataclass
class VariantReport:
    """Full report for one variant vs base."""

    variant_id: str
    dtype: str
    streams: list[StreamReport]


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

    # For magnitude metrics (MSE/L2), mask out pad positions before computing
    # so padding doesn't bias the damage formula. Only real-token positions
    # contribute (reviewer W1).
    if base_mask is not None:
        mask_3d = base_mask.unsqueeze(-1).float()
        base_masked = base_activations * mask_3d
        var_masked = variant_activations * mask_3d
    else:
        base_masked = base_activations
        var_masked = variant_activations

    return {
        "cka_linear": cka_linear(base_pooled, variant_pooled),
        "cka_rbf": cka_rbf(base_pooled, variant_pooled),
        "cos_all": cosine_all_tokens(base_activations, variant_activations, base_mask),
        "cos_last": cosine_last_token(base_activations, variant_activations, base_mask),
        "mse": mse(base_masked, var_masked),
        "l2": l2_distance(base_masked, var_masked),
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
) -> VariantReport:
    """Compare a variant artifact against the base.

    Computes all metrics per matching stream, plus optional damage in
    quant-floor units. Per-category breakdown is computed when categories
    are present and non-uniform.

    Args:
        base: Base model artifact.
        variant: Variant model artifact (must share prompts/categories).
        quant_floor: Fake-quantized base artifact for damage computation.
                     If None, damage_quant_units is not computed.
        numerical_floor: Optional bf16-vs-fp16 reference (reported but not
                         used in the damage formula per reviewer C2).

    Returns:
        VariantReport with per-stream StreamReports.

    Raises:
        ValueError: If base and variant don't share prompts/categories.
    """
    base.assert_aligned(variant)

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

        # Compute damage if quant floor is provided
        damage: MetricResult | None = None
        if quant_floor is not None and stream_name in quant_floor.streams:
            qf_stream = quant_floor.streams[stream_name]
            mse_base_qf = mse(base_stream.activations, qf_stream.activations)
            damage_val = compute_damage(metrics["mse"], mse_base_qf)
            damage = MetricResult(value=damage_val)

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
                cos_all=MetricResult(value=metrics["cos_all"]),
                cos_last=MetricResult(value=metrics["cos_last"]),
                mse=MetricResult(value=metrics["mse"]),
                l2=MetricResult(value=metrics["l2"]),
                rank_corr=MetricResult(value=metrics["rank_corr"]),
                damage_quant_units=damage,
                per_category=per_category,
            )
        )

    return VariantReport(
        variant_id=variant.ckpt_id,
        dtype=variant.dtype,
        streams=streams_report,
    )
