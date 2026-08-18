"""Literature citations for the TE benchmark report.

Single source of truth for the papers/community sources the hypotheses and
metrics tie into (see TR_COMPARISON.md §4/§10 for the full landscape). Both
the HTML report and the Hugo markdown export render these; keep keys stable —
they are referenced from section builders.
"""

from __future__ import annotations

import html
import logging

log = logging.getLogger(__name__)

# key -> (short label, URL, one-line relationship to our findings)
REFS: dict[str, tuple[str, str, str]] = {
    "arditi2024": (
        "Arditi et al. 2024 — Refusal in LLMs is mediated by a single direction",
        "https://arxiv.org/abs/2406.11717",
        "Origin of the abliteration method; our H3 tests their direction concept at the "
        "DiT conditioning interface (supported, z 50-134).",
    ),
    "spingarn2026": (
        "Spingarn 2026 — Diffusion models need less text than assumed",
        "https://arxiv.org/abs/2606.03715",
        "Motivated our H4 (meaning vs order); our interface shows no such privilege - an "
        "honest divergence from their SD-era finding.",
    ),
    "esd2023": (
        "Schramowski et al. 2023 — Safe Latent Diffusion (ESD)",
        "https://arxiv.org/abs/2303.07345",
        "DiT-side concept erasure: where engineered T2I safety lives. Frames our H6 null "
        "result - this stack has no render-stage refusal to begin with.",
    ),
    "mace2025": (
        "Lu et al. 2025 — When Are Concepts Erased (MACE)",
        "https://arxiv.org/abs/2505.17013",
        "Same erasure line as ESD; supports reading H6's 100% base compliance as 'no DiT-side censor present'.",
    ),
    "ablitsafety2025": (
        "arXiv 2025 — Safety behaviors persist under abliteration",
        "https://arxiv.org/abs/2510.02768",
        "LLM-side parallel to our H6: abliteration does not flip every safety behavior - "
        "here it flips none at the image stage.",
    ),
    "tescale2025": (
        "arXiv 2025 — Scaling down text encoders",
        "https://arxiv.org/abs/2503.19897",
        "Claims TEs tolerate heavy compression; our quant ladder bounds that for "
        "conditioning quality (clean <= Q6/int8, broken at 4-bit).",
    ),
    "nunchaku": (
        "Nunchaku — text-encoder quantization (project docs)",
        "https://deepwiki.com/mit-han-lab/nunchaku/2.4-text-encoder-quantization",
        "The marketing 'lossless TE quant' claim our ladder actually measures, dose by dose.",
    ),
    "qdiffusion2023": (
        "Li et al. 2023 — Q-Diffusion (ICCV)",
        "https://arxiv.org/abs/2302.04304",
        "Representative of DiT-side quantization work; the TE side was unmeasured until this battery.",
    ),
    "kornblith2019": (
        "Kornblith et al. 2019 — CKA similarity",
        "https://arxiv.org/abs/1905.00414",
        "Our conditioning-similarity metric (linear + RBF CKA).",
    ),
    "lpips2018": (
        "Zhang et al. 2018 — LPIPS",
        "https://arxiv.org/abs/1801.03924",
        "Perceptual distance used for the image ladder, anchors, and dose curves.",
    ),
    "clipscore2021": (
        "Hessel et al. 2021 — CLIPScore",
        "https://arxiv.org/abs/2104.08718",
        "Prompt-adherence metric (P8) and the taboo compliance-gap building block.",
    ),
}


def ref(key: str) -> str:
    """Inline HTML anchor for a reference key ('' when unknown — never raises)."""
    entry = REFS.get(key)
    if entry is None:
        log.warning("unknown citation key: %s", key)
        return ""
    label, url, _rel = entry
    return f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener">{html.escape(label)}</a>'


def refs_section_html() -> str:
    """References appendix for the HTML report."""
    items = "".join(
        f'<li><a href="{html.escape(url, quote=True)}" target="_blank" '
        f'rel="noopener"><b>{html.escape(label)}</b></a> — {html.escape(rel)}</li>'
        for label, url, rel in REFS.values()
    )
    return (
        '<h2 id="sec-refs">References — how this study ties into prior work</h2>'
        "<p>Hypotheses H3/H4/H6 and the quant-ladder framing are imported from the "
        "literature and community sources below; the verdict table links back here. "
        "The noise-floor, noise-equivalent dose, attention-thresholded detail loss, "
        "and the abliteration×quantization factorial have no prior measurement we "
        "could find.</p><ul class='findings'>" + items + "</ul>"
    )
