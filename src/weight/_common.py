"""Shared helpers for src/weight analysis scripts.

Extracted from byte-identical per-script copies (see .cache/audit/weight-core.md
and .cache/audit/weight-pairwise.md):

- ``cosine_sim`` / ``safe_mean`` / ``safe_median`` / ``safe_std`` statistics
  (previously duplicated in edit_vector_analysis, stacking_analysis,
  technique_correlation).
- Batch-mode CLI boilerplate (``--comparison/--base/--output/--results-dir``
  flags, results-dir resolution, per-variant output paths) shared by the four
  core scripts (panel, edit_vector, svd, fingerprint).
- ``iter_common_tensors``: the streaming load -> guard -> gc loop skeleton
  (shape-mismatch skip + accounting, gc cadence, optional progress log) that
  previously existed as copy-paste in the pairwise/per-variant scripts and was
  the root cause of the missing-shape-guard bug class (AGENTS.md lesson #79).

Helpers here must stay numerically identical to the code they replaced.
"""

from __future__ import annotations

import argparse
import gc
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from src.config import ComparisonConfig
from src.model_config import load_tensor


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    dot = (a_f * b_f).sum()
    na = a_f.norm()
    nb = b_f.norm()
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return max(-1.0, min(1.0, (dot / (na * nb)).item()))


def safe_mean(lst: list[float]) -> float | None:
    return sum(lst) / len(lst) if lst else None


def safe_median(lst: list[float]) -> float | None:
    if not lst:
        return None
    s = sorted(lst)
    mid = len(s) // 2
    return (s[mid - 1] + s[mid]) / 2 if len(s) % 2 == 0 else s[mid]


def safe_std(lst: list[float]) -> float | None:
    if len(lst) < 2:
        return None
    m = sum(lst) / len(lst)
    return math.sqrt(sum((x - m) ** 2 for x in lst) / len(lst))


# ---------------------------------------------------------------------------
# Batch-mode CLI boilerplate (panel / edit_vector / svd / fingerprint mains)
# ---------------------------------------------------------------------------


def add_batch_args(parser: argparse.ArgumentParser, output_help: str = "Output JSON path") -> None:
    """Add the standard batch-mode CLI flags shared by the core scripts."""
    parser.add_argument("--comparison", help="Path to comparison.json dir (batch mode)")
    parser.add_argument("--base", help="Base model dir")
    parser.add_argument("--output", help=output_help)
    parser.add_argument("--results-dir", help="Base results directory")


def resolve_results_dir(args: argparse.Namespace, config: ComparisonConfig) -> Path:
    """Resolve the weight results directory from --results-dir or the config."""
    return Path(args.results_dir) if getattr(args, "results_dir", None) else config.weight_results_dir(Path("results"))


def variant_output_path(results_dir: Path, variant_name: str, prefix: str) -> Path:
    """Per-variant output file used by the per-variant batch loops."""
    return results_dir / variant_name / f"{prefix}_{variant_name}.json"


# ---------------------------------------------------------------------------
# Streaming loop skeleton
# ---------------------------------------------------------------------------


@dataclass
class StreamStats:
    """Skip accounting for :func:`iter_common_tensors` (lesson #79 observability).

    ``shape_skipped`` counts keys whose loaded tensors disagree on shape
    (architecture-extending edits, e.g. an appended MLP neuron). Emit it into
    the report JSON so silent scope shrinkage is forensically visible.
    """

    shape_skipped: int = 0


def iter_common_tensors(
    maps: Sequence[dict],
    keys: Sequence[str] | None = None,
    *,
    gc_interval: int = 200,
    progress: Callable[[int, int], None] | None = None,
    stats: StreamStats | None = None,
) -> Iterator[tuple[int, str, list[torch.Tensor]]]:
    """Stream ``(index, key, tensors)`` for the common keys of ``maps``.

    Replaces the copy-pasted streaming loop: loads each key's tensor from every
    shard map, silently skips keys where any load returns ``None``, skips (and
    via ``stats`` accounts for) keys whose shapes disagree, and runs
    ``gc.collect()`` every ``gc_interval`` keys followed by the optional
    ``progress(index, total)`` callback — matching the cadence of the loops it
    replaces.

    The yielded tensor list is cleared when the generator resumes, so tensors
    are released as soon as the consumer's loop body finishes (consumers must
    not keep references to the list across iterations; extract floats instead).

    Args:
        maps: Shard maps (``canonical_key -> (raw_key, shard_path)``) to load
            from; every map must contain the key.
        keys: Optional key sequence; defaults to the sorted intersection of all
            maps' keys.
        gc_interval: Collect garbage every N processed keys (0 disables).
        progress: Optional callback invoked after each ``gc.collect()`` with
            the current 0-based index and total key count.
        stats: Optional :class:`StreamStats` to accumulate skip counts into.

    Yields:
        ``(index, key, tensors)`` where ``tensors[i]`` corresponds to
        ``maps[i]`` (all with equal shapes, none ``None``).
    """
    if keys is None:
        keys = sorted(set(maps[0]).intersection(*(set(m) for m in maps[1:])))
    total = len(keys)
    for i, key in enumerate(keys):
        tensors = [load_tensor(m, key) for m in maps]
        if any(t is None for t in tensors):
            continue
        if len({t.shape for t in tensors}) > 1:
            if stats is not None:
                stats.shape_skipped += 1
            continue
        yield i, key, cast("list[torch.Tensor]", tensors)
        tensors.clear()  # release originals; consumer's float() copies survive
        if gc_interval and (i + 1) % gc_interval == 0:
            gc.collect()
            if progress is not None:
                progress(i, total)
