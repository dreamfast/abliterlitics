"""Shared DiT-harness utilities for the backend DiT loaders.

Pure helpers previously duplicated across z_image_dit/krea2_dit/h3_dit.
One definition guards against block-counting drift between the three
harnesses (lesson-#44c class: block regexes stay ANCHORED at the call
site — the caller's contract; only the counting logic is shared).
"""

from __future__ import annotations

import re


def count_blocks(keys: list[str], pattern: str) -> int:
    """Count distinct block indices (regex group 1) matched in ``keys``."""
    pat = re.compile(pattern)
    ids = {int(m.group(1)) for k in keys if (m := pat.search(k))}
    return len(ids)
