"""Build the H3 website figures (fig01-04) from canonical tebench JSONs.

The website (abliterlitics_website) is the final report; these SVGs are its
H3 page assets, generated from comparisons/te-h3/results/te-h3/text_encoder/.

  fig01  conditioning interface: (a) per-layer MSE profile vs base (log),
         divergence band shaded; (b) word-content cosine histogram, function
         vs content words (from te_heretic.json + tebench_word_content.json)
  fig02  audio-VAE ladder: (a) frame-cosine by channel and pair type,
         (b) delta-direction consistency vs seed nulls
         (from tebench_h3_audio_damage.json, turbo config)
  fig03  video battery: per-pair frame-LPIPS scatter per arm vs the
         seed-change anchor (from tebench_h3_video.json)
  fig04  attention readout: per-segment JSD bars + per-layer JSD line
         (from tebench_attention_heretic.json)

Usage:
    python3 scripts/build_h3_figures.py \
        --results comparisons/te-h3/results/te-h3/text_encoder \
        --out comparisons/te-h3/graphs/figures

Then copy fig*.svg into abliterlitics_website/static/graphs/h3-te/.
Paths in the page must be absolute: /graphs/h3-te/figNN.svg (lesson 49).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

# Palette matched to the series figures: heretic orange, quant/control grey-blue,
# seed-null green, floor grey.
C_HERETIC = "#d95f02"
C_QUANT = "#7570b3"
C_SEED = "#1b9e77"
C_FLOOR = "#7f7f7f"

FUNCTION_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "we",
    "it",
    "do",
    "does",
    "her",
    "his",
    "they",
    "them",
    "you",
    "i",
    "is",
    "are",
    "was",
    "were",
    "in",
    "on",
    "at",
    "to",
    "of",
    "for",
    "with",
    "that",
    "this",
    "there",
    "here",
    "he",
    "she",
    "my",
    "your",
    "our",
    "as",
    "so",
    "then",
    "than",
    "from",
}


def _load(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def _layer_profile(variants: list[dict[str, Any]]) -> tuple[list[int], list[float]]:
    """Per-layer MSE rows live in the raw_hidden stream group (variants is a
    list of per-stage stream groups for the SAME variant, not a variant list)."""
    layers: list[int] = []
    mses: list[float] = []
    for group in variants:
        for stream in group["streams"]:
            if stream["stream"].startswith("layer_"):
                idx = int(stream["stream"].split("_")[1])
                layers.append(idx)
                mses.append(stream["metrics"]["mse"])
    order = sorted(range(len(layers)), key=lambda i: layers[i])
    return [layers[i] for i in order], [mses[i] for i in order]


def fig01(results: Path, out: Path) -> None:
    te = _load(results / "te_heretic.json")
    wc = _load(results / "tebench_word_content.json")
    layers, mses = _layer_profile(te["variants"])

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    axa.plot(layers, mses, color=C_HERETIC, lw=1.8)
    axa.set_yscale("log")
    axa.axvspan(31, 49, color=C_HERETIC, alpha=0.10)
    axa.annotate(
        "edit band\n(layers 31-49)",
        xy=(40, mses[40]),
        xytext=(12, 3e-2),
        fontsize=9,
        color=C_HERETIC,
        arrowprops=dict(arrowstyle="->", color=C_HERETIC, lw=1),
    )
    axa.set_xlabel("encoder layer")
    axa.set_ylabel("MSE vs base (log)")
    axa.set_title("(a) per-layer divergence, heretic vs base")
    axa.grid(alpha=0.25)

    cos_func: list[float] = []
    cos_content: list[float] = []
    for entries in wc["variants"]["heretic"].values():
        for word, cos, kind in entries:
            if kind != "word":  # excludes 23 'pad' rows; 876 real word occurrences
                continue
            target = cos_func if word.lower().strip(".,!?:;") in FUNCTION_WORDS else cos_content
            target.append(cos)
    axb.hist(
        [cos_content, cos_func],
        bins=40,
        range=(0.75, 1.0005),
        color=[C_QUANT, C_HERETIC],
        label=[f"content words (n={len(cos_content)})", f"function words (n={len(cos_func)})"],
        stacked=True,
    )
    axb.set_xlabel("word-content cosine vs base")
    axb.set_ylabel("occurrences")
    axb.set_title("(b) 876 word occurrences, cosine drift")
    axb.legend(fontsize=8)
    axb.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out / "fig01.svg")
    plt.close(fig)


def fig02(results: Path, out: Path) -> None:
    ad = _load(results / "tebench_h3_audio_damage.json")
    rows = {(r["config"], r["metric_name"]): r["metric_value"] for r in ad["rows"]}

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    # (a) frame-cosine ladder, turbo config, by channel
    labels = [
        "speech\nte swap",
        "speech\nquant floor",
        "speech\nseed reroll",
        "ambient\nte swap",
        "ambient\nquant floor",
    ]
    vals = [
        rows[("turbo", "te_swap_speech_frame")],
        rows[("turbo", "floor_speech_frame")],
        rows[("turbo", "seed_base_speech_frame")],
        rows[("turbo", "te_swap_ambient_frame")],
        rows[("turbo", "floor_ambient_frame")],
    ]
    colors = [C_HERETIC, C_QUANT, C_SEED, C_HERETIC, C_QUANT]
    axa.bar(range(len(vals)), vals, color=colors, alpha=0.85)
    axa.set_xticks(range(len(vals)), labels, fontsize=8)
    axa.set_ylabel("frame cosine (higher = closer to base)")
    axa.set_ylim(0, 1.05)
    axa.set_title("(a) audio-VAE ladder, turbo clips")
    axa.grid(alpha=0.25, axis="y")
    axa.annotate(
        "te swap ~= pure-quant floor\n(both far above a seed reroll)",
        xy=(0.5, 0.80),
        xytext=(1.6, 0.45),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=1),
    )

    # (b) delta-direction consistency
    dl = [
        ("te swap\n(heretic)", rows[("turbo", "te_swap_mean_pairwise_cos")], C_HERETIC),
        ("seed null\n(base)", rows[("turbo", "seed_base_mean_pairwise_cos")], C_SEED),
        ("seed null\n(heretic)", rows[("turbo", "seed_heretic_mean_pairwise_cos")], C_SEED),
        ("quant\nfloor", rows[("turbo", "floor_mean_pairwise_cos")], C_QUANT),
    ]
    axb.bar([d[0] for d in dl], [d[1] for d in dl], color=[d[2] for d in dl], alpha=0.85)
    axb.axhline(0, color="black", lw=0.8)
    axb.set_ylabel("mean pairwise delta cosine")
    axb.set_title("(b) delta-direction consistency")
    axb.tick_params(axis="x", labelsize=8)
    axb.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out / "fig02.svg")
    plt.close(fig)


def fig03(results: Path, out: Path) -> None:
    vid = _load(results / "tebench_h3_video.json")
    anchor = vid["seed_change_anchor_lpips"]

    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    for pos, (name, color) in enumerate([("base_nvfp4", C_QUANT), ("heretic", C_HERETIC)]):
        v = vid["variants"][name]
        xs = [pos + (i % 8) * 0.09 - 0.36 for i in range(len(v["lpips_values"]))]
        ax.scatter(
            xs, v["lpips_values"], s=26, color=color, alpha=0.7, label=f"{name} (mean {v['mean_frame_lpips']:.3f})"
        )
        ax.hlines(v["mean_frame_lpips"], pos - 0.42, pos + 0.42, color=color, lw=2)
    ax.axhline(anchor, color=C_SEED, ls="--", lw=1.4, label=f"seed-change anchor ({anchor:.3f})")
    ax.set_xticks([0, 1], ["base_nvfp4 (quant control)", "heretic"])
    ax.set_ylabel("same-seed frame LPIPS vs base")
    ax.set_title("video battery, 16 prompts x seeds")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out / "fig03.svg")
    plt.close(fig)


def fig04(results: Path, out: Path) -> None:
    at = _load(results / "tebench_attention_heretic.json")

    def find(x: dict[str, Any], key: str) -> dict[str, Any]:
        out_d: dict[str, Any] = {}
        if isinstance(x, dict):
            for k, v in x.items():
                if key in k and isinstance(v, dict):
                    out_d.update(v)
                else:
                    out_d.update(find(v, key))
        return out_d

    seg = find(at, "jsd_per_segment")
    per_layer = find(at, "jsd_per_layer")

    def find_scalar(x: object, key: str) -> float:
        if isinstance(x, dict):
            for k, v in x.items():
                if key in k and isinstance(v, (int, float)):
                    return float(v)
                hit = find_scalar(v, key)
                if hit is not None:
                    return hit
        return None  # type: ignore[return-value]

    mean_all = find_scalar(at, "jsd_mean_all") or 0.0

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    axa.bar(list(seg.keys()), list(seg.values()), color=[C_HERETIC, C_QUANT], alpha=0.85)
    axa.set_ylabel("attention JSD vs base")
    axa.set_title("(a) DiT readout shift, per output stream")
    axa.grid(alpha=0.25, axis="y")

    idx = sorted(int(k) for k in per_layer)
    axb.plot(idx, [per_layer[str(i)] for i in idx], color=C_QUANT, lw=1.8, marker="o", ms=3)
    axb.axhline(mean_all, color=C_HERETIC, ls="--", lw=1.2, label=f"mean over all reads ({mean_all:.4f})")
    axb.set_xlabel("DiT block")
    axb.set_ylabel("attention JSD vs base")
    axb.set_title("(b) readout shift by DiT block")
    axb.legend(fontsize=8)
    axb.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out / "fig04.svg")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for n, fn in [(1, fig01), (2, fig02), (3, fig03), (4, fig04)]:
        fn(args.results, args.out)
        log.info("fig%02d.svg written", n)


if __name__ == "__main__":
    main()
