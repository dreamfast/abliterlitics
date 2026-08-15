"""Attention readout metrics — what does the DiT READ from the conditioning?

Pure-tensor core (Phase 4 T2.3). Operates on attention distributions over
conditioning-token positions captured by attention_capture.py (B0 spike:
image-query rows x text-key columns, head-mean, per step/layer).

Metrics (per step/layer, base vs variant):
- JSD (PRIMARY): symmetric, bounded [0, ln2], epsilon-floored for hard zeros
- KL (secondary): clamped eps=1e-12, symmetrized — direction reported but the
  symmetric form avoids the P>0/Q=0 infinity trap (hy3-C5)
- top-k overlap: Jaccard + rank-biased overlap (RBO) of top-k attended tokens
- entropy: diffusion of the readout distribution
- mass_on: attention mass on a given token-index subset (keyword analysis)

Convention: distributions are length-L nonneg vectors (not necessarily
normalized to 1 by the caller; these helpers normalize defensively).
"""

from __future__ import annotations

import logging
import math

import torch

log = logging.getLogger(__name__)

EPS = 1e-12


def _normalize(p: torch.Tensor) -> torch.Tensor:
    """Normalize rows of a [..., L] distribution defensively (clamp >= 0, sum 1)."""
    p = p.clamp(min=0.0)
    return p / p.sum(dim=-1, keepdim=True).clamp(min=EPS)


def jsd(p: torch.Tensor, q: torch.Tensor, eps: float = EPS) -> float:
    """Jensen-Shannon divergence (nats, bounded [0, ln 2]) between two distributions.

    Epsilon-floored: handles hard zeros on either side (attention rows
    frequently contain exact zeros from padding).
    """
    p = _normalize(p) + eps
    q = _normalize(q) + eps
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    m = 0.5 * (p + q)
    kl_pm = float((p * (p.log() - m.log())).sum(dim=-1))
    kl_qm = float((q * (q.log() - m.log())).sum(dim=-1))
    val = 0.5 * (max(kl_pm, 0.0) + max(kl_qm, 0.0))
    return min(val, math.log(2.0))


def kl_sym(p: torch.Tensor, q: torch.Tensor, eps: float = EPS) -> float:
    """Symmetrized KL (J-divergence) with clamping: KL(p||q) + KL(q||p).

    Clamped at eps on both sides — the clamp makes the value finite and keeps
    the metric a divergence (zero iff equal); the raw one-directional KL is
    intentionally NOT exposed (infinite on support mismatch).
    """
    p = _normalize(p) + eps
    q = _normalize(q) + eps
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    kl_pq = float((p * (p.log() - q.log())).sum(dim=-1))
    kl_qp = float((q * (q.log() - p.log())).sum(dim=-1))
    return max(kl_pq, 0.0) + max(kl_qp, 0.0)


def topk_overlap(
    p: torch.Tensor,
    q: torch.Tensor,
    k: int = 5,
    rbo_p: float = 0.9,
) -> dict[str, float]:
    """Top-k attended-token agreement: Jaccard + rank-biased overlap.

    Jaccard: |top_k(p) & top_k(q)| / |top_k(p) | top_k(q)| in [0, 1].
    RBO: rank-biased overlap (Webber et al. 2010), indefinite version,
    in [0, 1] — rewards agreement at the TOP of the ranking.
    """
    p = _normalize(p)
    q = _normalize(q)
    k = min(k, p.shape[-1])
    tp = set(torch.topk(p, k).indices.tolist())
    tq = set(torch.topk(q, k).indices.tolist())
    jaccard = len(tp & tq) / max(len(tp | tq), 1)

    # RBO over the FULL rankings; when the list is exhausted at ``depth``
    # (depth >= n) use the FINITE normalization X/S; otherwise the standard
    # indefinite extrapolation (assume tail agreement 1.0, +p^depth).
    rp = torch.argsort(p, descending=True).tolist()
    rq = torch.argsort(q, descending=True).tolist()
    n = len(rp)
    depth = min(n, max(k * 4, 20))
    overlap_runs: list[float] = []
    seen_p: set[int] = set()
    seen_q: set[int] = set()
    for d in range(1, depth + 1):
        seen_p.add(rp[d - 1])
        seen_q.add(rq[d - 1])
        overlap_runs.append(len(seen_p & seen_q) / d)
    weights = [(1 - rbo_p) * rbo_p ** (d - 1) for d in range(1, depth + 1)]
    prefix = sum(w * o for w, o in zip(weights, overlap_runs, strict=True))
    weight_sum = sum(weights)
    # finite list: no tail to extrapolate; else standard indefinite extension
    rbo_val = prefix / max(weight_sum, 1e-12) if depth >= n else prefix + rbo_p**depth
    return {"jaccard": float(jaccard), "rbo": float(min(max(rbo_val, 0.0), 1.0))}


def entropy(p: torch.Tensor) -> float:
    """Shannon entropy (nats) of an attention distribution; 0 for a delta peak."""
    p = _normalize(p) + EPS
    p = p / p.sum(dim=-1, keepdim=True)
    return float(-(p * p.log()).sum(dim=-1))


def mass_on(p: torch.Tensor, token_idx: list[int]) -> float:
    """Total attention mass on a subset of token positions (keyword analysis)."""
    if not token_idx:
        return 0.0
    p = _normalize(p)
    return float(p[token_idx].sum())


def summarize_readout(
    base: dict[str, torch.Tensor],
    variant: dict[str, torch.Tensor],
    ks: tuple[int, ...] = (3, 5, 10),
) -> dict[str, float]:
    """Aggregate metrics for one (step, layer) readout pair.

    base/variant: {"text_readout": [L], } — image-query-rows mean attention
    over text-key columns (head-mean already applied by the capture layer).
    Returns a flat metric dict: jsd, kl_sym, entropy_base, entropy_variant,
    topk_{k}_jaccard, topk_{k}_rbo.
    """
    p = base["text_readout"]
    q = variant["text_readout"]
    out: dict[str, float] = {
        "jsd": jsd(p, q),
        "kl_sym": kl_sym(p, q),
        "entropy_base": entropy(p),
        "entropy_variant": entropy(q),
    }
    for k in ks:
        ov = topk_overlap(p, q, k=k)
        out[f"top{k}_jaccard"] = ov["jaccard"]
        out[f"top{k}_rbo"] = ov["rbo"]
    return out


# ---------------------------------------------------------------------------
# Word-level readout diff (pure)
# ---------------------------------------------------------------------------


def group_words(tokens: list[str], mass: list[float]) -> list[tuple[str, float]]:
    """Group BPE token masses into word masses (order-preserving).

    A token starting with whitespace (post-Ġ decode) starts a new word;
    otherwise it continues the previous word. Special/pad tokens (angle
    brackets) and empty token lists are passed through as their own rows.
    Masses are normalized to sum 1 over the given length first (defensive).
    """
    if not tokens:
        return []
    n = min(len(tokens), len(mass))
    if n == 0:
        return []
    total = sum(mass[:n]) or 1.0
    norm = [m / total for m in mass[:n]]
    words: list[tuple[str, float]] = []
    for tok, m in zip(tokens[:n], norm, strict=False):
        t = tok.strip()
        is_special = tok.startswith("<") and tok.endswith(">") or not t
        if is_special:
            label = t if t else "‹pad›"
            words.append((label, m))
        elif tok.startswith(" ") and words:
            words.append((t, m))
        elif words:
            w, prev = words[-1]
            words[-1] = (w + t, prev + m)
        else:
            words.append((t, m))
    # merge duplicate labels (repeated words / consecutive pads) by summing
    merged: dict[str, float] = {}
    order: list[str] = []
    for w, m in words:
        if w not in merged:
            order.append(w)
        merged[w] = merged.get(w, 0.0) + m
    return [(w, merged[w]) for w in order]


def word_diff(
    tokens: list[str],
    base_mass: list[float],
    variant_mass: list[float],
    top_k: int = 15,
) -> list[dict[str, float | str]]:
    """Rank words by |variant mass − base mass| — what the DiT reads differently.

    Returns rows sorted by absolute delta, each: {word, base, variant, delta}.
    """
    b = group_words(tokens, base_mass)
    v = group_words(tokens, variant_mass)
    v_map = dict(v)
    rows: list[dict[str, float | str]] = []
    for w, bm in b:
        vm = v_map.get(w, 0.0)
        delta: float = vm - bm
        rows.append({"word": w, "base": bm, "variant": vm, "delta": delta})
    rows.sort(key=lambda r: abs(float(r["delta"])), reverse=True)
    return rows[:top_k]


# ---------------------------------------------------------------------------
# Ordered word rows (prompt-order presentation — Phase 4 report redesign)
# ---------------------------------------------------------------------------

# Chat-template scaffolding tokens (Qwen/Z-Image llama_template) — rendered as
# one dimmed row, never mixed into content words.
CHAT_TOKENS = {"<|im_start|>", "<|im_end|>", "<|endoftext|>", "user", "assistant", "system", "Ċ"}


def token_groups(tokens: list[str]) -> list[tuple[str, str, list[int]]]:
    """Group BPE token strings into word groups IN PROMPT ORDER.

    Returns (label, kind, token_indices) triples with kind in {"word",
    "chat", "pad"} using the same rules as word_rows_ordered:
    - consecutive non-space BPE pieces glue into one word group
    - consecutive chat-template tokens merge into ONE "[chat template]" group
    - consecutive pad positions merge into ONE "[pad]" group

    Index-tracking companion so per-token measurements (e.g. conditioning
    content cosine) can be aggregated to the SAME rows the attention readout
    renders — guaranteeing the two channels stay aligned.
    """
    groups: list[tuple[str, str, list[int]]] = []
    cur_label: str | None = None
    cur_kind = ""
    cur_idx: list[int] = []

    def _flush() -> None:
        nonlocal cur_label, cur_kind, cur_idx
        if cur_label is not None:
            groups.append((cur_label, cur_kind, cur_idx))
        cur_label, cur_kind, cur_idx = None, "", []

    for i, tok in enumerate(tokens):
        t = tok.strip()
        is_pad = (not t) or tok in ("<pad>", "<|pad|>")
        is_chat = (tok.startswith("<") and tok.endswith(">")) or tok in CHAT_TOKENS
        if is_pad or is_chat:
            _flush()
            kind = "pad" if is_pad else "chat"
            label = "[pad]" if is_pad else "[chat template]"
            if groups and groups[-1][1] == kind:
                groups[-1][2].append(i)
            else:
                groups.append((label, kind, [i]))
        elif tok.startswith(" ") or cur_label is None:
            _flush()
            cur_label, cur_kind, cur_idx = t, "word", [i]
        else:
            assert cur_label is not None
            cur_label = cur_label + t
            cur_idx.append(i)
    _flush()
    return groups


def word_rows_ordered(tokens: list[str], mass: list[float]) -> list[tuple[str, float, str]]:
    """Group BPE tokens into word rows IN PROMPT ORDER (positional, no re-sorting).

    Each row: (label, mass_fraction, kind) with kind in {"word", "chat", "pad"}.
    - Consecutive non-space BPE pieces glue into one word row.
    - Consecutive chat-template tokens merge into ONE "[chat ...]" row.
    - Pad positions merge into one "[pad]" row. Masses normalized over the
      token span first (shares of the PROMPT attention budget).
    """
    n = min(len(tokens), len(mass))
    if n == 0:
        return []
    total = sum(mass[:n]) or 1.0
    rows: list[tuple[str, float, str]] = []
    for label, kind, idxs in token_groups(tokens[:n]):
        m = sum(mass[i] for i in idxs) / total
        if rows and rows[-1][2] == kind and kind in ("chat", "pad"):
            rows[-1] = (rows[-1][0], rows[-1][1] + m, kind)
        else:
            rows.append((label, m, kind))
    return rows
