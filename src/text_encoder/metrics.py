"""Metrics for text-encoder conditioning fidelity analysis.

All functions are pure-tensor math (fp32, no model imports). The primary
metric is CKA (Centered Kernel Alignment, Kornblith et al. 2019) which measures
representational similarity between two activation matrices. Companion metrics
(cosine, L2, MSE, rank correlation, per-dimension drift) are added in T1.3.

CKA is the established tool for "did two networks produce similar representations?"
It operates on the full [n, d] activation matrix (n = prompts, d = feature dim),
capturing representational geometry across the whole prompt set.

Reference: Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019).
"Similarity of Neural Network Representations Revisited."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

log = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """A metric value with optional bootstrap confidence interval."""

    value: float
    ci_low: float | None = None
    ci_high: float | None = None

    def __repr__(self) -> str:
        if self.ci_low is not None and self.ci_high is not None:
            return f"{self.value:.6f} [{self.ci_low:.6f}, {self.ci_high:.6f}]"
        return f"{self.value:.6f}"


# ---------------------------------------------------------------------------
# Pooling helpers
# ---------------------------------------------------------------------------


def pool_mean(activations: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean-pool activations [n, L, d] -> [n, d] using mask to exclude pad tokens.

    Args:
        activations: [n, L, d] float32 tensor.
        mask: [n, L] int tensor (1 = real token, 0 = pad). None = no padding.

    Returns:
        [n, d] mean-pooled tensor.
    """
    if mask is None:
        return activations.mean(dim=1)
    # Expand mask to [n, L, 1] for broadcasting
    m = mask.unsqueeze(-1).float()
    summed = (activations * m).sum(dim=1)
    counts = m.sum(dim=1).clamp(min=1.0)
    return summed / counts


def pool_last(activations: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Last non-pad token from activations [n, L, d] -> [n, d].

    Args:
        activations: [n, L, d] float32 tensor.
        mask: [n, L] int tensor. None = last position is last token.

    Returns:
        [n, d] tensor of the last real token per prompt. Rows whose mask is
        all-zero (all-pad prompt) have NO real token — they return NaN rather
        than silently yielding the pad-position vector.
    """
    if mask is None:
        return activations[:, -1, :]
    # Find last non-pad position per prompt
    # mask: [n, L], find the index of the last 1 in each row
    lengths = mask.sum(dim=1)  # [n]
    # Index into activations using gathered indices
    idx = (lengths - 1).clamp(min=0).long()  # [n]
    out = activations[torch.arange(activations.shape[0]), idx]  # [n, d] (advanced indexing -> copy)
    empty = lengths == 0
    if bool(empty.any()):
        log.warning("pool_last: %d all-pad row(s) — no real token; NaN returned for those rows", int(empty.sum()))
        out[empty] = float("nan")
    return out


# ---------------------------------------------------------------------------
# CKA — the primary similarity metric
# ---------------------------------------------------------------------------


def cka_linear(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA (Centered Kernel Alignment).

    Measures representational similarity between two activation matrices.
    Bounded [0, 1], symmetric, invariant to orthogonal transforms and
    isotropic scaling.

    Uses the HSIC-based formulation:
        CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    Args:
        x: [n, d1] float32 tensor (one row per prompt, pooled).
        y: [n, d2] float32 tensor (same n, possibly different d).

    Returns:
        CKA similarity in [0, 1]. 1.0 = identical representations.

    Note:
        For n < d (common with high-dim encoders), CKA has positive bias.
        This is documented — CKA is used as a similarity ranking tool, not
        an absolute measure. The damage formula uses MSE (T1.4) which is
        not affected by this bias.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"cka_linear requires same n (rows): x={x.shape[0]}, y={y.shape[0]}")
    if x.shape[0] < 2:
        raise ValueError(f"cka_linear requires n >= 2, got n={x.shape[0]}")

    # Center columns (subtract per-dimension mean across samples)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    # HSIC-based: ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    xtx_norm = torch.linalg.norm(x.T @ x)  # ||X^T X||_F
    yty_norm = torch.linalg.norm(y.T @ y)  # ||Y^T Y||_F
    ytx_norm = torch.linalg.norm(y.T @ x)  # ||Y^T X||_F

    denominator = xtx_norm * yty_norm
    if denominator < 1e-15:
        log.warning("cka_linear: near-zero denominator (degenerate inputs), returning 0.0")
        return 0.0

    result = (ytx_norm * ytx_norm / denominator).clamp(min=0.0, max=1.0)
    return float(result.item())


def _rbf_kernel(x: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    """Compute RBF (Gaussian) kernel matrix K[i,j] = exp(-||x_i - x_j||^2 / (2*sigma^2)).

    Args:
        x: [n, d] float32 tensor.
        sigma: kernel bandwidth. If None, use median heuristic (median of
               pairwise distances). Clamped to minimum 1e-10 to avoid div-by-zero.

    Returns:
        [n, n] kernel matrix.
    """
    # Pairwise squared distances: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i^T x_j
    sq_norms = (x * x).sum(dim=1)  # [n]
    sq_dists = sq_norms.unsqueeze(0) + sq_norms.unsqueeze(1) - 2.0 * (x @ x.T)  # [n, n]
    sq_dists = sq_dists.clamp(min=0.0)  # numerical safety

    if sigma is None:
        # Median heuristic: sigma = median of non-zero pairwise distances
        nonzero_dists = sq_dists[sq_dists > 1e-12]
        sigma = 1e-10 if nonzero_dists.numel() == 0 else torch.sqrt(nonzero_dists.median()).item()
        sigma = max(sigma, 1e-10)  # clamp to avoid div-by-zero

    return torch.exp(-sq_dists / (2.0 * sigma * sigma))


def cka_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float | None = None) -> float:
    """RBF (nonlinear) CKA using Gaussian kernels.

    Same HSIC normalization as linear CKA, but with RBF kernels instead of
    linear. Large linear-vs-RBF divergence is a signal of nonlinear warping.

    Args:
        x: [n, d1] float32 tensor (pooled, one row per prompt).
        y: [n, d2] float32 tensor (same n).
        sigma: RBF bandwidth. If None, median heuristic computed independently
               for each kernel (X and Y get their own sigma).

    Returns:
        CKA similarity in [0, 1].
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"cka_rbf requires same n (rows): x={x.shape[0]}, y={y.shape[0]}")
    if x.shape[0] < 4:
        # Small per-category groups (e.g. stress suites with 2 prompts per
        # category) are legitimate inputs — RBF CKA is simply undefined there.
        # NaN (not a raise) so a single small category can't kill a full run.
        log.warning("cka_rbf undefined for n=%d (< 4); returning NaN", x.shape[0])
        return float("nan")

    # Center the data (for consistency with linear CKA interpretation)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    n = x.shape[0]

    # Compute kernel matrices (with independent median-heuristic sigma)
    k_x = _rbf_kernel(x, sigma)
    k_y = _rbf_kernel(y, sigma)

    # Center the kernels: K_c = H K H  where H = I - (1/n) * ones(n,n)
    h = torch.eye(n, dtype=x.dtype) - 1.0 / n * torch.ones(n, n, dtype=x.dtype)
    k_xc = h @ k_x @ h
    k_yc = h @ k_y @ h

    # HSIC-based CKA via Frobenius inner product of centered kernels:
    #   CKA = <K_xc, K_yc>_F / (||K_xc||_F * ||K_yc||_F)
    hsic_xy = (k_xc * k_yc).sum()
    hsic_xx = (k_xc * k_xc).sum()
    hsic_yy = (k_yc * k_yc).sum()

    denominator = torch.sqrt(hsic_xx * hsic_yy)
    if denominator < 1e-15:
        log.warning("cka_rbf: near-zero denominator (degenerate kernels), returning 0.0")
        return 0.0

    result = (hsic_xy / denominator).clamp(min=0.0, max=1.0)
    return float(result.item())


# ---------------------------------------------------------------------------
# Companion metrics — magnitude-aware + geometry-aware
# ---------------------------------------------------------------------------


def cosine_all_tokens(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None) -> float:
    """Mean cosine similarity across all non-padded tokens.

    For each prompt, computes the mean cosine similarity over all tokens,
    then averages across prompts. This extends the author's last-token-only
    metric to the full sequence because the DiT cross-attends to every token.

    Args:
        x: [n, L, d] float32 tensor.
        y: [n, L, d] float32 tensor (same shape as x).
        mask: [n, L] int tensor (1 = real token, 0 = pad). None = all real.

    Returns:
        Mean cosine similarity in [-1, 1]. 1.0 = identical.
    """
    if x.shape != y.shape:
        raise ValueError(f"cosine_all_tokens requires same shape: x={tuple(x.shape)}, y={tuple(y.shape)}")

    # Per-token cosine: cos(x_i, y_i) = (x . y) / (|x| * |y|)
    dot = (x * y).sum(dim=-1)  # [n, L]
    norm_x = torch.linalg.norm(x, dim=-1)  # [n, L]
    norm_y = torch.linalg.norm(y, dim=-1)  # [n, L]
    denom = norm_x * norm_y
    cos_per_token = torch.where(denom > 1e-15, dot / denom, torch.zeros_like(dot))  # [n, L]

    if mask is not None:
        m = mask.float()
        total = (cos_per_token * m).sum(dim=1)
        counts = m.sum(dim=1).clamp(min=1.0)
        return float((total / counts).mean().item())
    return float(cos_per_token.mean().item())


def cosine_last_token(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None) -> float:
    """Cosine similarity at the last non-pad token (for blog comparability).

    The author's prior measurement used outputs.last_hidden_state[:, -1, :].
    We keep this number for direct comparison but the full-sequence metric
    (cosine_all_tokens) is the trustworthy one.

    Args:
        x: [n, L, d] float32 tensor.
        y: [n, L, d] float32 tensor.
        mask: [n, L] int tensor. None = last position is last token.

    Returns:
        Mean cosine at the last real token in [-1, 1].
    """
    if x.shape != y.shape:
        raise ValueError(f"cosine_last_token requires same shape: x={tuple(x.shape)}, y={tuple(y.shape)}")

    x_last = pool_last(x, mask)  # [n, d]
    y_last = pool_last(y, mask)  # [n, d]
    dot = (x_last * y_last).sum(dim=-1)
    norm_x = torch.linalg.norm(x_last, dim=-1)
    norm_y = torch.linalg.norm(y_last, dim=-1)
    denom = norm_x * norm_y
    cos = torch.where(denom > 1e-15, dot / denom, torch.zeros_like(dot))
    return float(cos.mean().item())


def mse(x: torch.Tensor, y: torch.Tensor) -> float:
    """Mean squared error per element.

    Args:
        x: [n, L, d] or [n, d] float32 tensor.
        y: same shape as x.

    Returns:
        mean(||x - y||^2) — the MSE.
    """
    if x.shape != y.shape:
        raise ValueError(f"mse requires same shape: x={tuple(x.shape)}, y={tuple(y.shape)}")
    return float(((x - y) ** 2).mean().item())


def rank_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """Spearman rank correlation of pairwise prompt distances.

    Tests whether the modified encoder still ranks "concept A closer to B
    than to C" the same way. Computes pairwise Euclidean distances among
    prompts under each encoder (pooled [n, d]), then Spearman correlation
    of the two distance vectors. Catches anisotropic warping CKA can miss.

    Args:
        x: [n, d] float32 tensor (pooled, one row per prompt).
        y: [n, d] float32 tensor (same n, possibly different d).

    Returns:
        Spearman rho in [-1, 1]. 1.0 = identical geometry.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"rank_correlation requires same n: x={x.shape[0]}, y={y.shape[0]}")
    n = x.shape[0]
    if n < 3:
        # Degenerate for small per-category groups (stress suites) — Spearman
        # over 1 pairwise distance is meaningless. NaN, not a raise, so one
        # small category can't kill a full comparison run (phase-6 fix).
        log.warning("rank_correlation undefined for n=%d (< 3); returning NaN", n)
        return float("nan")

    # Pairwise distances (upper triangle only)
    # dist_x[i,j] = ||x_i - x_j||
    dist_x = torch.cdist(x, x)  # [n, n]
    dist_y = torch.cdist(y, y)  # [n, n]

    # Extract upper triangle (excluding diagonal)
    triu_idx = torch.triu_indices(n, n, offset=1)
    dx = dist_x[triu_idx[0], triu_idx[1]]
    dy = dist_y[triu_idx[0], triu_idx[1]]

    # Spearman = Pearson on ranks
    dx_rank = _rank(dx)
    dy_rank = _rank(dy)
    return _pearson(dx_rank, dy_rank)


def _rank(values: torch.Tensor) -> torch.Tensor:
    """Compute ranks of values (1-based, average ties).

    Args:
        values: 1D float32 tensor.

    Returns:
        1D float32 tensor of ranks.
    """
    n = values.shape[0]
    sorted_idx = values.argsort()
    ranks = torch.empty(n, dtype=values.dtype)
    # Assign ranks with average for ties
    sorted_vals = values[sorted_idx]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation coefficient.

    Args:
        x: 1D float32 tensor.
        y: 1D float32 tensor (same length).

    Returns:
        Pearson r in [-1, 1].
    """
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = torch.sqrt((x_centered**2).sum() * (y_centered**2).sum())
    if denom < 1e-15:
        return 0.0
    return float(((x_centered * y_centered).sum() / denom).clamp(-1.0, 1.0).item())
