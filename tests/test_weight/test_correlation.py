"""Tests for src/weight/technique_correlation.py — Cosine similarity and projection math."""
from __future__ import annotations

import pytest
import torch

from src.weight.technique_correlation import cosine_sim, project_out


class TestCosineSim:
    def test_identical(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        assert cosine_sim(a, a) == pytest.approx(1.0, abs=1e-6)

    def test_opposite(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        assert cosine_sim(a, -a) == pytest.approx(-1.0, abs=1e-6)

    def test_orthogonal(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        assert cosine_sim(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_zero_vec(self):
        assert cosine_sim(torch.zeros(5), torch.ones(5)) == 0.0


class TestProjectOut:
    def test_removes_component(self):
        """After projecting out a direction, the residual should be orthogonal."""
        vec = torch.tensor([3.0, 4.0, 0.0])
        direction = torch.tensor([1.0, 0.0, 0.0])
        residual = project_out(vec, direction)
        # Residual should be orthogonal to direction
        dot = (residual.flatten() * direction.float().flatten()).sum()
        assert dot.abs().item() < 1e-6

    def test_preserves_orthogonal_component(self):
        """Projecting out a direction should preserve orthogonal parts."""
        vec = torch.tensor([3.0, 4.0])
        direction = torch.tensor([1.0, 0.0])
        residual = project_out(vec, direction)
        assert residual.flatten()[1].item() == pytest.approx(4.0, abs=1e-6)

    def test_zero_direction(self):
        """Zero direction should return original vector."""
        vec = torch.tensor([3.0, 4.0])
        direction = torch.zeros(2)
        residual = project_out(vec, direction)
        assert (residual - vec.float()).abs().max().item() < 1e-6
