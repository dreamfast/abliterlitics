"""Bootstrap confidence intervals for text-encoder metrics.

Nonparametric bootstrap over the prompt axis. For per-prompt metrics
(cosine, MSE), this is straightforward resampling. For ratio-based damage,
the numerator and denominator are resampled jointly (same prompt indices)
to produce a valid ratio CI (reviewer amendment C4).

For CKA, the CI requires recomputing CKA on each resampled subset — this is
O(n^2 * d * B) and documented. In practice, CKA CIs are computed only when
explicitly requested; the primary damage metric (MSE) has cheap CIs.
"""

from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)

# Default number of bootstrap resamples
DEFAULT_N_BOOT = 1000


def bootstrap_ci(
    values: torch.Tensor,
    n_boot: int = DEFAULT_N_BOOT,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Nonparametric bootstrap CI for a 1D tensor of per-prompt values.

    Args:
        values: [n] float32 tensor of per-prompt metric values.
        n_boot: Number of bootstrap resamples (default 1000).
        confidence: Confidence level (default 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        (point_estimate, ci_low, ci_high) — the point estimate is the mean
        of the original values, not the mean of bootstrap means.
    """
    n = values.shape[0]
    if n < 2:
        return float(values.mean().item()), float("nan"), float("nan")

    g = torch.Generator().manual_seed(seed)
    alpha = (1.0 - confidence) / 2.0

    # Vectorized bootstrap: generate [n_boot, n] index matrix
    idx = torch.randint(0, n, (n_boot, n), generator=g)
    # Resample and compute mean per bootstrap iteration
    boot_means = values[idx].mean(dim=1)  # [n_boot]

    point = float(values.mean().item())
    ci_low = float(torch.quantile(boot_means, alpha).item())
    ci_high = float(torch.quantile(boot_means, 1.0 - alpha).item())
    return point, ci_low, ci_high


def bootstrap_ratio_ci(
    numerator_values: torch.Tensor,
    denominator_values: torch.Tensor,
    n_boot: int = DEFAULT_N_BOOT,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for a ratio MSE_num / MSE_den with JOINT resampling.

    The numerator and denominator must be resampled with the SAME prompt
    indices to produce a valid ratio CI. Simply dividing independent CIs
    would be mathematically wrong (reviewer amendment C4).

    Args:
        numerator_values: [n] per-prompt squared errors for (base, variant).
        denominator_values: [n] per-prompt squared errors for (base, fake_quant).
        n_boot: Number of bootstrap resamples.
        confidence: Confidence level.
        seed: Random seed.

    Returns:
        (point_ratio, ci_low, ci_high).
    """
    n = numerator_values.shape[0]
    if denominator_values.shape[0] != n:
        raise ValueError(f"numerator ({denominator_values.shape[0]}) and denominator ({n}) must have same length")

    if n < 2:
        den_mean = float(denominator_values.mean().item())
        point = float(numerator_values.mean().item()) / max(den_mean, 1e-15)
        return point, float("nan"), float("nan")

    g = torch.Generator().manual_seed(seed)
    alpha = (1.0 - confidence) / 2.0

    # Joint resampling: same indices for both
    idx = torch.randint(0, n, (n_boot, n), generator=g)
    boot_num = numerator_values[idx].mean(dim=1)  # [n_boot]
    boot_den = denominator_values[idx].mean(dim=1)  # [n_boot]

    # Guard against zero/negative denominators
    safe_den = boot_den.clamp(min=1e-15)
    boot_ratios = boot_num / safe_den.float()

    # Point estimate from full data
    point_den = max(float(denominator_values.mean().item()), 1e-15)
    point = float(numerator_values.mean().item()) / point_den
    ci_low = float(torch.quantile(boot_ratios, alpha).item())
    ci_high = float(torch.quantile(boot_ratios, 1.0 - alpha).item())
    return point, ci_low, ci_high
