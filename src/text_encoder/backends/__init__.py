"""Backend registry for text-encoder analysis.

Adding a new model backend = writing one file in backends/ + one entry here.
"""

from __future__ import annotations

import logging

from src.text_encoder.backends.base import BaseBackend, HookManager, fake_quantize_weights

log = logging.getLogger(__name__)


# Registry: backend name -> backend class.
# Only implemented backends are registered (reviewer W8).
BACKENDS: dict[str, type[BaseBackend]] = {}


def register_backend(name: str, cls: type[BaseBackend]) -> None:
    """Register a backend class under a name."""
    BACKENDS[name] = cls
    log.debug("Registered backend: %s -> %s", name, cls.__name__)


def _register_default_backends() -> None:
    """Register implemented backends (lazy imports — comfyui import is heavy)."""
    from src.text_encoder.backends.h3 import H3Backend
    from src.text_encoder.backends.ltx2 import LTX2Backend
    from src.text_encoder.backends.z_image import ZImageBackend

    register_backend("h3", H3Backend)
    register_backend("ltx2", LTX2Backend)
    register_backend("zimage", ZImageBackend)


_register_default_backends()


def get_backend(name: str) -> BaseBackend:
    """Get a backend instance by name.

    Args:
        name: Backend identifier ("ltx2", "h3", etc.)

    Returns:
        A new instance of the requested backend.

    Raises:
        ValueError: If the backend is not registered.
    """
    cls = BACKENDS.get(name)
    if cls is None:
        available = ", ".join(sorted(BACKENDS.keys())) or "(none)"
        raise ValueError(f"Unknown backend: {name!r}. Available: {available}")
    return cls()


__all__ = [
    "BaseBackend",
    "HookManager",
    "fake_quantize_weights",
    "BACKENDS",
    "get_backend",
    "register_backend",
]
