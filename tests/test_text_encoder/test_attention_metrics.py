"""Tests for attention readout metrics — JSD/KL known answers, top-k, entropy."""

from __future__ import annotations

import math

import pytest
import torch

from src.text_encoder.attention_metrics import (
    entropy,
    jsd,
    kl_sym,
    mass_on,
    summarize_readout,
    topk_overlap,
)


def _dist(n: int = 8, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(n, generator=g).clamp(min=1e-6)
    return p / p.sum()


class TestJSD:
    def test_identical_zero(self) -> None:
        p = _dist()
        assert jsd(p, p.clone()) == pytest.approx(0.0, abs=1e-8)

    def test_disjoint_max_ln2(self) -> None:
        p = torch.zeros(8)
        p[0] = 1.0
        q = torch.zeros(8)
        q[1] = 1.0
        assert jsd(p, q) == pytest.approx(math.log(2.0), rel=1e-3)

    def test_hard_zeros_finite(self) -> None:
        p = torch.tensor([0.5, 0.5, 0.0, 0.0])
        q = torch.tensor([0.0, 0.0, 0.5, 0.5])
        val = jsd(p, q)
        assert math.isfinite(val) and val > 0.5

    def test_symmetric(self) -> None:
        p, q = _dist(seed=1), _dist(seed=2)
        assert jsd(p, q) == pytest.approx(jsd(q, p), abs=1e-9)

    def test_bounded(self) -> None:
        p, q = _dist(seed=3), _dist(seed=4)
        assert 0.0 <= jsd(p, q) <= math.log(2.0) + 1e-9


class TestKLSym:
    def test_identical_near_zero(self) -> None:
        p = _dist()
        assert kl_sym(p, p.clone()) == pytest.approx(0.0, abs=1e-4)

    def test_hard_zeros_finite(self) -> None:
        p = torch.tensor([1.0, 0.0])
        q = torch.tensor([0.0, 1.0])
        assert math.isfinite(kl_sym(p, q))

    def test_symmetric(self) -> None:
        p, q = _dist(seed=1), _dist(seed=2)
        assert kl_sym(p, q) == pytest.approx(kl_sym(q, p), abs=1e-6)


class TestTopkOverlap:
    def test_identical_perfect(self) -> None:
        p = _dist()
        ov = topk_overlap(p, p.clone(), k=3)
        assert ov["jaccard"] == 1.0
        assert ov["rbo"] == pytest.approx(1.0, abs=1e-4)

    def test_disjoint_jaccard_zero(self) -> None:
        p = torch.zeros(10)
        p[:5] = 1.0
        q = torch.zeros(10)
        q[5:] = 1.0
        ov = topk_overlap(p, q, k=3)
        assert ov["jaccard"] == 0.0
        # disjoint halves still converge to 50% set overlap at full depth;
        # RBO floor for that is ~0.25 — well below identical (1.0)
        assert ov["rbo"] < 0.4

    def test_rank_order_rewarded(self) -> None:
        # same set, different order -> jaccard 1.0, rbo < 1.0
        p = torch.tensor([0.4, 0.3, 0.2, 0.1, 0.0])
        q = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.0])
        ov = topk_overlap(p, q, k=4)
        assert ov["jaccard"] == 1.0
        assert ov["rbo"] < 0.9


class TestEntropy:
    def test_delta_peak_zero(self) -> None:
        p = torch.zeros(8)
        p[0] = 1.0
        assert entropy(p) == pytest.approx(0.0, abs=1e-6)

    def test_uniform_max(self) -> None:
        p = torch.full((8,), 1.0 / 8)
        assert entropy(p) == pytest.approx(math.log(8), rel=1e-3)


class TestMassOn:
    def test_subset_mass(self) -> None:
        p = torch.tensor([0.1, 0.2, 0.3, 0.4])
        assert mass_on(p, [0, 2]) == pytest.approx(0.4)
        assert mass_on(p, []) == 0.0


class TestSummarize:
    def test_flat_dict(self) -> None:
        base = {"text_readout": _dist(seed=1)}
        var = {"text_readout": _dist(seed=2)}
        m = summarize_readout(base, var)
        for key in ("jsd", "kl_sym", "entropy_base", "entropy_variant", "top3_jaccard", "top5_rbo", "top10_rbo"):
            assert key in m
        assert m["jsd"] > 0.0


class TestWordReadout:
    def test_group_words_bpe_and_pads(self) -> None:
        from src.text_encoder.attention_metrics import group_words

        toks = ["a", " photo", " of", " a", " red", " square", "<pad>", "<pad>"]
        mass = [0.30, 0.20, 0.10, 0.05, 0.25, 0.05, 0.03, 0.02]
        words = dict(group_words(toks, mass))
        assert words["a"] == pytest.approx(0.35)  # duplicates merged
        assert words["photo"] == pytest.approx(0.20)
        assert words["‹pad›"] == pytest.approx(0.05)  # pad label merged
        assert sum(words.values()) == pytest.approx(1.0)

    def test_word_diff_ranks_by_delta(self) -> None:
        from src.text_encoder.attention_metrics import word_diff

        toks = ["a", " red", " square"]
        base = [0.5, 0.3, 0.2]
        var = [0.5, 0.1, 0.4]
        rows = word_diff(toks, base, var, top_k=3)
        assert rows[0]["word"] == "square" or rows[0]["word"] == "red"
        deltas = [abs(float(r["delta"])) for r in rows]
        assert deltas == sorted(deltas, reverse=True)

    def test_empty(self) -> None:
        from src.text_encoder.attention_metrics import group_words, word_diff

        assert group_words([], []) == []
        assert word_diff(["a"], [], []) == []
