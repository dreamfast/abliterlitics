"""Tests for src/kl/kl_divergence.py — KL computation correctness."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


class TestKLDivergenceMath:
    """Test the core KL divergence formula used in the compute phase.

    These test the exact formula: F.kl_div(logprobs_variant, logprobs_base, reduction="batchmean", log_target=True)
    """

    def test_kl_uniform_uniform_near_zero(self):
        """KL(uniform, uniform) ≈ 0."""
        vocab = 1000
        uniform = torch.full((1, vocab), 1.0 / vocab).log()
        kl = F.kl_div(uniform, uniform, reduction="batchmean", log_target=True).item()
        assert kl == pytest.approx(0.0, abs=1e-6)

    def test_kl_peaked_vs_uniform_positive(self):
        """KL(peaked, uniform) > 0 (known direction)."""
        vocab = 100
        # Peaked distribution: one token has most of the probability
        peaked = torch.zeros(1, vocab)
        peaked[0, 0] = 0.99
        peaked[0, 1:] = 0.01 / (vocab - 1)
        log_peaked = peaked.log()

        uniform = torch.full((1, vocab), 1.0 / vocab)
        log_uniform = uniform.log()

        # KL(peaked || uniform) = sum(peaked * (log(peaked) - log(uniform)))
        kl = F.kl_div(log_uniform, log_peaked, reduction="batchmean", log_target=True).item()
        assert kl > 0

    def test_kl_asymmetry(self):
        """KL(A, B) ≠ KL(B, A) in general."""
        vocab = 100
        a = torch.zeros(1, vocab)
        a[0, 0] = 0.9
        a[0, 1:] = 0.1 / (vocab - 1)

        b = torch.zeros(1, vocab)
        b[0, 50] = 0.8
        b[0, :50] = 0.1 / 50
        b[0, 51:] = 0.1 / 49

        log_a = a.log()
        log_b = b.log()

        kl_ab = F.kl_div(log_a, log_b, reduction="batchmean", log_target=True).item()
        kl_ba = F.kl_div(log_b, log_a, reduction="batchmean", log_target=True).item()
        assert kl_ab != pytest.approx(kl_ba, abs=1e-4)

    def test_kl_identical_distributions_zero(self):
        """KL(P, P) = 0 for any distribution."""
        vocab = 50
        p = torch.randn(1, vocab).softmax(dim=-1)
        log_p = p.log()
        kl = F.kl_div(log_p, log_p, reduction="batchmean", log_target=True).item()
        assert kl == pytest.approx(0.0, abs=1e-6)

    def test_kl_non_negative(self):
        """KL divergence should always be >= 0."""
        torch.manual_seed(42)
        vocab = 100
        for _ in range(10):
            p = torch.randn(1, vocab).softmax(dim=-1)
            q = torch.randn(1, vocab).softmax(dim=-1)
            kl = F.kl_div(q.log(), p.log(), reduction="batchmean", log_target=True).item()
            assert kl >= -1e-6  # tiny numerical tolerance

    def test_kl_batchmean_scales_with_batch(self):
        """batchmean should give consistent results regardless of batch size."""
        vocab = 10
        p = torch.softmax(torch.randn(1, vocab), dim=-1)
        q = torch.softmax(torch.randn(1, vocab), dim=-1)
        log_p = p.log()
        log_q = q.log()

        # Single prompt
        kl1 = F.kl_div(log_q, log_p, reduction="batchmean", log_target=True).item()

        # Two identical prompts — batchmean divides by batch_size so result is the same
        log_q2 = torch.cat([log_q, log_q])
        log_p2 = torch.cat([log_p, log_p])
        kl2 = F.kl_div(log_q2, log_p2, reduction="batchmean", log_target=True).item()

        assert kl1 == pytest.approx(kl2, abs=1e-6)

    def test_kl_batchmean_manual_calculation(self):
        """Verify KL against manual calculation with clean probabilities."""
        # Use distributions that avoid -inf (no zero probs)
        p = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
        q = torch.tensor([[0.25, 0.25, 0.25, 0.25]])

        log_p = p.log()
        log_q = q.log()

        # F.kl_div(input, target, reduction="batchmean", log_target=True)
        # = sum(exp(target) * (target - input)) / batch_size
        # = sum(p * (log_p - log_q)) / 1
        kl = F.kl_div(log_q, log_p, reduction="batchmean", log_target=True).item()

        # Manual: sum(p * (log(p) - log(q)))
        expected = (p * (log_p - log_q)).sum().item()
        assert kl == pytest.approx(expected, abs=1e-4)
        assert kl > 0  # uniform Q < peaked P
