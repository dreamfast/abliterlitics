"""Shared fixtures for text-encoder tests.

Provides synthetic tensor generators for the pure-tensor metrics core
(no GPU, no model dependency).
"""

from __future__ import annotations

import pytest
import torch

# ---------------------------------------------------------------------------
# Tensor fixtures — small synthetic activations for unit tests
# ---------------------------------------------------------------------------


@pytest.fixture
def n_prompts() -> int:
    """Default number of synthetic prompts."""
    return 20


@pytest.fixture
def d_dim() -> int:
    """Default feature dimension."""
    return 32


@pytest.fixture
def seq_len() -> int:
    """Default sequence length."""
    return 10


@pytest.fixture
def random_activations(n_prompts: int, seq_len: int, d_dim: int) -> torch.Tensor:
    """Random [n, L, d] float32 tensor (seeded for reproducibility)."""
    g = torch.Generator().manual_seed(42)
    return torch.randn(n_prompts, seq_len, d_dim, generator=g)


@pytest.fixture
def random_mask(n_prompts: int, seq_len: int) -> torch.Tensor:
    """Random attention mask [n, L] with ~80% real tokens."""
    g = torch.Generator().manual_seed(99)
    mask = (torch.rand(n_prompts, seq_len, generator=g) > 0.2).int()
    # Ensure at least one real token per prompt
    mask[:, 0] = 1
    return mask


@pytest.fixture
def identical_activations(random_activations: torch.Tensor) -> torch.Tensor:
    """A copy of random_activations (for identity tests)."""
    return random_activations.clone()


@pytest.fixture
def orthogonal_rotated_activations(random_activations: torch.Tensor, d_dim: int) -> torch.Tensor:
    """Random activations rotated by a random orthogonal matrix (for CKA invariance tests).

    Uses QR decomposition to generate a random orthogonal Q, then applies
    X @ Q to the last dimension.
    """
    g = torch.Generator().manual_seed(7)
    a_mat = torch.randn(d_dim, d_dim, generator=g)
    q_mat, _ = torch.linalg.qr(a_mat)  # q_mat is orthogonal
    # Apply rotation: [n, L, d] @ [d, d] -> [n, L, d]
    return torch.matmul(random_activations, q_mat)


@pytest.fixture
def scaled_activations(random_activations: torch.Tensor) -> torch.Tensor:
    """Activations scaled by a constant factor (for isotropic-scale invariance tests)."""
    return random_activations * 3.14


@pytest.fixture
def independent_activations(n_prompts: int, seq_len: int, d_dim: int) -> torch.Tensor:
    """Independent random tensor (for CKA independence test)."""
    g = torch.Generator().manual_seed(123)
    return torch.randn(n_prompts, seq_len, d_dim, generator=g)
