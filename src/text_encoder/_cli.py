"""Shared argparse flag helpers for the TE runner scripts.

Minimal consolidation per the te-runners-zimage audit (Duplication #1): only
the most-duplicated declarations live here (--device ×7, --cache-dir /
--results-dir ×6, --backend ×3+, --comparison). Defaults and help text are
passed EXPLICITLY by each runner so the transplant stays mechanical — the
runners keep ownership of their semantics (they currently disagree on
defaults, e.g. run_analysis --device cpu vs the trajectory runners' cuda).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

__all__ = ["add_backend", "add_cache_dir", "add_comparison", "add_device", "add_results_dir"]


def add_device(p: argparse.ArgumentParser, *, default: str, help_text: str = "Device: cpu or cuda") -> None:
    """--device: compute device for model load + trajectories/metrics."""
    p.add_argument("--device", default=default, help=help_text)


def add_cache_dir(p: argparse.ArgumentParser, *, default: str, help_text: str = "Activation cache directory") -> None:
    """--cache-dir: run_analysis activation-cache directory."""
    p.add_argument("--cache-dir", default=default, help=help_text)


def add_results_dir(p: argparse.ArgumentParser, *, default: str, help_text: str = "Output directory") -> None:
    """--results-dir: output directory for result JSONs."""
    p.add_argument("--results-dir", default=default, help=help_text)


def add_backend(
    p: argparse.ArgumentParser,
    *,
    default: str = "zimage",
    choices: Sequence[str] = ("zimage", "krea2"),
    help_text: str = "DiT + conditioning family",
) -> None:
    """--backend: DiT + conditioning-interface family."""
    p.add_argument("--backend", choices=list(choices), default=default, help=help_text)


def add_comparison(p: argparse.ArgumentParser) -> None:
    """--comparison: comparison.json path (required)."""
    p.add_argument("--comparison", required=True)
