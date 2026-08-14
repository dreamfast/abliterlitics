"""Image-space metrics for the A/B image set — perceptual + semantic divergence.

Phase 4 of the tebench battery (journal #122): measures what the DiT RENDERS
when the text encoder changes, closing the chain
conditioning -> trajectory -> pixels -> semantics (approximate).

Pair types mined from the A/B manifest (all seed-controlled):
- ``te_swap``       variant vs reference at the SAME (label, seed) — the signal:
                    how much the rendered image moves when only the TE changes.
- ``cross_prompt``  different labels within one variant at the SAME seed — the
                    meaning-change anchor in image space ("a different picture
                    for a different prompt").
- ``seed_change``   same variant + label, different seed — the normal-variation
                    anchor ("a different picture from the same prompt"). Needs
                    the multi-seed top-up (run_ab_test --seeds).
- ``floor_band``    two near-lossless quants of the base (default int8 vs fp8)
                    at the same (label, seed) — the quant-floor band: image-level
                    distance between two known-good encoders.

Metrics:
- LPIPS (AlexNet) — primary perceptual divergence (bounded ~[0, 1+] in practice)
- CLIP-score (prompt<->image, 100*cos convention) and its delta vs reference —
  semantic adherence proxy
- CLIP image-embedding distance (1-cos) — cross-image semantic distance
- PSNR / SSIM — classical distortion references

Terminology (journals #117/#118): these are DIVERGENCE measurements. "Damage"
is reserved for collateral unintended divergence with functional consequence
and never appears as a metric name.

Model-backed meters (LPIPSMeter, CLIPMeter) are lazy and injectable; the pure
core (pair mining, PSNR/SSIM, aggregation) is model-free and unit-tested.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import torch

from src.text_encoder.bootstrap import bootstrap_ci

log = logging.getLogger(__name__)

# Reference pair defining the quant-floor band: two near-lossless quants of the
# base encoder. Both ship in the comparison; int8 is the recommended variant.
DEFAULT_FLOOR_PAIR: tuple[str, str] = ("base_int8", "base_fp8")


# ---------------------------------------------------------------------------
# Pair mining (pure — operates on manifest image dicts)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImagePair:
    """One scored image pair.

    kind: te_swap | cross_prompt | seed_change | floor_band
    variant_a/variant_b: TE variant names (a = measured, b = reference/anchor)
    label: prompt label (same for te_swap/seed_change/floor_band; the TWO
           labels joined by '+' for cross_prompt)
    seed_a/seed_b: generation seeds (same for all but seed_change)
    path_a/path_b: image file paths (filled by the caller/loader)
    """

    kind: str
    variant_a: str
    variant_b: str
    label: str
    seed_a: int
    seed_b: int
    path_a: str = ""
    path_b: str = ""


def mine_te_swap_pairs(
    images: list[dict[str, Any]],
    reference: str,
    base_seed: int = 42,
    pass_name: str = "warm",
) -> list[ImagePair]:
    """Mine variant-vs-reference pairs at identical (label, seed).

    Args:
        images: manifest ``images`` entries (dicts with variant/label/seed/
            path/pass keys — see run_ab_test manifest schema).
        reference: reference variant name (e.g. ``base_int8``).
        base_seed: the A/B fixed seed (pairs are seed-matched by construction).
        pass_name: which generation pass to use (``warm`` — cold==warm are
            byte-identical for this stack; warm is always present).

    Returns:
        One ImagePair per (variant, label) where both sides exist.
    """
    by_key: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for img in images:
        if str(img.get("pass", "")) != pass_name:
            continue
        key = (str(img["variant"]), str(img["label"]), int(img["seed"]), pass_name)
        by_key[key] = img
    variants = sorted({k[0] for k in by_key})
    pairs: list[ImagePair] = []
    for var in variants:
        if var == reference:
            continue
        for (v, label, seed, _p), img in sorted(by_key.items()):
            if v != var or seed != base_seed:
                continue
            ref_img = by_key.get((reference, label, seed, pass_name))
            if ref_img is None:
                log.warning("te_swap: no reference image for %s @ %s seed %d", var, label, seed)
                continue
            pairs.append(
                ImagePair(
                    kind="te_swap",
                    variant_a=var,
                    variant_b=reference,
                    label=label,
                    seed_a=seed,
                    seed_b=seed,
                    path_a=str(img.get("path", "")),
                    path_b=str(ref_img.get("path", "")),
                )
            )
    return pairs


def mine_cross_prompt_pairs(
    images: list[dict[str, Any]],
    per_variant: int = 30,
    base_seed: int = 42,
    pass_name: str = "warm",
    seed: int = 7,
) -> list[ImagePair]:
    """Mine different-prompt pairs within each variant (same seed).

    The meaning-change anchor in image space. Full pair sets are huge
    (C(n_labels, 2) per variant), so a seeded uniform subsample is taken.

    Returns:
        ImagePairs with label "a+b" and variant_b == variant_a.
    """
    by_var: dict[str, dict[str, dict[str, Any]]] = {}
    for img in images:
        if str(img.get("pass", "")) != pass_name or int(img["seed"]) != base_seed:
            continue
        by_var.setdefault(str(img["variant"]), {})[str(img["label"])] = img
    g = torch.Generator().manual_seed(seed)
    pairs: list[ImagePair] = []
    for var in sorted(by_var):
        labels = sorted(by_var[var])
        combos = list(itertools.combinations(labels, 2))
        if len(combos) > per_variant:
            idx = torch.randperm(len(combos), generator=g)[:per_variant].tolist()
            combos = [combos[i] for i in sorted(idx)]
        for la, lb in combos:
            pairs.append(
                ImagePair(
                    kind="cross_prompt",
                    variant_a=var,
                    variant_b=var,
                    label=f"{la}+{lb}",
                    seed_a=base_seed,
                    seed_b=base_seed,
                    path_a=str(by_var[var][la].get("path", "")),
                    path_b=str(by_var[var][lb].get("path", "")),
                )
            )
    return pairs


def mine_seed_change_pairs(
    images: list[dict[str, Any]],
    base_seed: int = 42,
    pass_name: str = "warm",
) -> list[ImagePair]:
    """Mine same-variant same-label pairs across seeds (normal-variation anchor).

    Every image at seed != base_seed is paired against its (variant, label)
    counterpart at base_seed. Requires the multi-seed top-up to be useful
    (the base manifest is single-seed and yields zero pairs).
    """
    by_key: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for img in images:
        if str(img.get("pass", "")) != pass_name:
            continue
        by_key[(str(img["variant"]), str(img["label"]), int(img["seed"]), pass_name)] = img
    pairs: list[ImagePair] = []
    for (var, label, seed, _p), img in sorted(by_key.items()):
        if seed == base_seed:
            continue
        base_img = by_key.get((var, label, base_seed, pass_name))
        if base_img is None:
            log.warning("seed_change: no base-seed image for %s @ %s", var, label)
            continue
        pairs.append(
            ImagePair(
                kind="seed_change",
                variant_a=var,
                variant_b=var,
                label=label,
                seed_a=seed,
                seed_b=base_seed,
                path_a=str(img.get("path", "")),
                path_b=str(base_img.get("path", "")),
            )
        )
    return pairs


def mine_floor_band_pairs(
    images: list[dict[str, Any]],
    floor_pair: tuple[str, str] = DEFAULT_FLOOR_PAIR,
    base_seed: int = 42,
    pass_name: str = "warm",
) -> list[ImagePair]:
    """Mine the quant-floor band: two near-lossless base quants, same (label, seed)."""
    a, b = floor_pair
    by_key: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for img in images:
        if str(img.get("pass", "")) != pass_name:
            continue
        by_key[(str(img["variant"]), str(img["label"]), int(img["seed"]), pass_name)] = img
    pairs: list[ImagePair] = []
    for (var, label, seed, _p), img in sorted(by_key.items()):
        if var != a or seed != base_seed:
            continue
        other = by_key.get((b, label, seed, pass_name))
        if other is None:
            log.warning("floor_band: missing %s @ %s", b, label)
            continue
        pairs.append(
            ImagePair(
                kind="floor_band",
                variant_a=a,
                variant_b=b,
                label=label,
                seed_a=seed,
                seed_b=seed,
                path_a=str(img.get("path", "")),
                path_b=str(other.get("path", "")),
            )
        )
    return pairs


# ---------------------------------------------------------------------------
# Pure classical metrics (no models)
# ---------------------------------------------------------------------------


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB between two [1,3,H,W] (or [3,H,W]) tensors.

    Identical inputs return inf (convention). Values in [0, max_val].
    """
    if a.shape != b.shape:
        raise ValueError(f"psnr requires same shape: {tuple(a.shape)} vs {tuple(b.shape)}")
    mse_val = float(((a - b) ** 2).mean().item())
    if mse_val <= 0.0:
        return float("inf")
    return 10.0 * math.log10(max_val * max_val / mse_val)


def _gaussian_window(size: int = 11, sigma: float = 1.5, channels: int = 3) -> torch.Tensor:
    """SSIM gaussian window [channels, 1, size, size] (Wang et al. 2004)."""
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    window = torch.outer(g, g)
    window = window / window.sum()
    return window.unsqueeze(0).unsqueeze(0).expand(channels, 1, size, size).contiguous()


def ssim(
    a: torch.Tensor,
    b: torch.Tensor,
    max_val: float = 1.0,
    window: torch.Tensor | None = None,
) -> float:
    """Mean SSIM (Wang et al. 2004) between two [1,3,H,W] tensors in [0, max_val].

    Uses the standard 11x11 sigma=1.5 gaussian window, per-channel maps
    averaged together. Returns a float in [-1, 1] (practically [0, 1]).
    """
    if a.shape != b.shape:
        raise ValueError(f"ssim requires same shape: {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.dim() != 4 or a.shape[0] != 1:
        raise ValueError(f"ssim expects [1,3,H,W], got {tuple(a.shape)}")
    channels = a.shape[1]
    if window is None:
        window = _gaussian_window(channels=channels)
    pad = window.shape[-1] // 2

    mu_a = torch.nn.functional.conv2d(a, window, padding=pad, groups=channels)
    mu_b = torch.nn.functional.conv2d(b, window, padding=pad, groups=channels)
    mu_aa, mu_bb, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sigma_aa = torch.nn.functional.conv2d(a * a, window, padding=pad, groups=channels) - mu_aa
    sigma_bb = torch.nn.functional.conv2d(b * b, window, padding=pad, groups=channels) - mu_bb
    sigma_ab = torch.nn.functional.conv2d(a * b, window, padding=pad, groups=channels) - mu_ab

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    num = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    den = (mu_aa + mu_bb + c1) * (sigma_aa + sigma_bb + c2)
    return float((num / den).mean().item())


# ---------------------------------------------------------------------------
# Model-backed meters (lazy; injectable for tests)
# ---------------------------------------------------------------------------


class LPIPSMeter:
    """Lazy LPIPS (AlexNet) meter. Inputs: [1,3,H,W] in [-1, 1]."""

    def __init__(self, net: str = "alex", device: str = "cpu") -> None:
        self.net = net
        self.device = device
        self._fn: torch.nn.Module | None = None

    def _ensure(self) -> torch.nn.Module:
        if self._fn is None:
            import lpips  # deferred: downloads weights on first use

            self._fn = lpips.LPIPS(net=self.net, verbose=False).to(self.device)
            for p in self._fn.parameters():
                p.requires_grad_(False)
        return self._fn

    def score(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Perceptual divergence between a and b (0 = identical)."""
        if a.shape != b.shape:
            raise ValueError(f"lpips requires same shape: {tuple(a.shape)} vs {tuple(b.shape)}")
        fn = self._ensure()
        with torch.no_grad():
            val = fn(a.to(self.device), b.to(self.device))
        return float(val.squeeze().item())


class CLIPMeter:
    """Lazy open_clip meter for CLIP-score + image-embedding distance.

    clip_score uses the 100 * cos(image, text) convention (Hessel et al. 2021).
    img_distance is 1 - cos(img_a, img_b).
    """

    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai", device: str = "cpu") -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self._model: Any = None
        self._tokenizer: Any = None
        self._preprocess: Any = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained)
        self._model = model.to(self.device).eval()
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        self._preprocess = preprocess
        for p in self._model.parameters():
            p.requires_grad_(False)

    def preprocess_pil(self, pil_img: Any) -> torch.Tensor:
        """Apply the model's preprocessing to a PIL image -> [1,3,H,W] tensor."""
        self._ensure()
        t = torch.as_tensor(self._preprocess(pil_img))
        return t.unsqueeze(0)

    def encode_image(self, img: torch.Tensor) -> torch.Tensor:
        """Embed a preprocessed image tensor [1,3,H,W] in [0,1] (L2-normalized)."""
        self._ensure()
        with torch.no_grad():
            feats = self._model.encode_image(img.to(self.device))
        return torch.nn.functional.normalize(feats, dim=-1)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Embed a batch of prompt strings (L2-normalized rows)."""
        self._ensure()
        tokens = self._tokenizer(texts).to(self.device)
        with torch.no_grad():
            feats = self._model.encode_text(tokens)
        return torch.nn.functional.normalize(feats, dim=-1)

    def clip_score(self, img: torch.Tensor, prompt: str) -> float:
        """100 * cos(image, text) — higher = better prompt adherence."""
        img_emb = self.encode_image(img)
        txt_emb = self.encode_text([prompt])
        return float(100.0 * (img_emb @ txt_emb.T).squeeze().item())

    def img_distance(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """1 - cos(img_a, img_b) — semantic distance between two images."""
        ea, eb = self.encode_image(a), self.encode_image(b)
        return float(1.0 - (ea @ eb.T).squeeze().item())


# ---------------------------------------------------------------------------
# Aggregation (pure)
# ---------------------------------------------------------------------------


@dataclass
class PairMetricSummary:
    """Aggregated metric values for one (kind, metric) bucket."""

    kind: str
    metric: str
    n: int
    mean: float
    ci_low: float
    ci_high: float


@dataclass
class PairScores:
    """All scored pairs for one kind, ready for aggregation."""

    kind: str
    pairs: list[ImagePair] = field(default_factory=list)
    lpips: list[float] = field(default_factory=list)
    clip_img_dist: list[float] = field(default_factory=list)
    psnr_db: list[float] = field(default_factory=list)
    ssim_val: list[float] = field(default_factory=list)

    def summarize(self, n_boot: int = 1000) -> list[PairMetricSummary]:
        """Bootstrap-CI summaries per metric (non-empty buckets only)."""
        out: list[PairMetricSummary] = []
        for name, vals in (
            ("lpips", self.lpips),
            ("clip_img_dist", self.clip_img_dist),
            ("psnr_db", self.psnr_db),
            ("ssim", self.ssim_val),
        ):
            finite = [v for v in vals if v == v and abs(v) != float("inf")]  # NaN/inf guard
            if len(finite) < 2:
                if vals:
                    log.info("%s/%s: only %d finite values — no CI", self.kind, name, len(finite))
                continue
            t = torch.tensor(finite, dtype=torch.float32)
            mean, lo, hi = bootstrap_ci(t, n_boot=n_boot)
            out.append(
                PairMetricSummary(
                    kind=self.kind,
                    metric=name,
                    n=len(finite),
                    mean=mean,
                    ci_low=lo,
                    ci_high=hi,
                )
            )
        return out


def divergence_ratio(te_swap_mean: float, anchor_mean: float) -> float | None:
    """Express a te_swap mean in units of an anchor mean (P7 dual-anchor reporting).

    Returns None when the anchor is ~0 (undefined ratio).
    """
    if anchor_mean is None or abs(anchor_mean) < 1e-12:
        return None
    return te_swap_mean / anchor_mean
