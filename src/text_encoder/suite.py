"""Prompt suite loading for text-encoder analysis.

The suite is stratified by category (innocent, mild, violent, sexual,
extreme, multilingual) so we can measure how abliteration damage varies
across content types. The author's blog showed a gradient: innocent > mild
> violent > sexual > extreme in cosine similarity, confirming the encoder
still distinguishes taboo concepts.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

DEFAULT_CATEGORIES = ("innocent", "mild", "violent", "sexual", "extreme", "multilingual")


def load_suite(path: Path | str) -> tuple[list[str], list[str]]:
    """Load a prompt suite from YAML.

    Expected format:
        - text: "a golden retriever in a park"
          category: innocent
        - text: "a woman in a bikini on a beach"
          category: mild

    Args:
        path: Path to the YAML file.

    Returns:
        (prompts, categories) — two aligned lists.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If any entry is missing text or category.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt suite not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    # Stress/A-B suites wrap entries in a {prompts: [...]} mapping (with
    # labels/checks); canonical te_suite.yaml is a bare list. Accept both.
    if isinstance(data, dict):
        data = data.get("prompts")

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of entries in {path}, got {type(data).__name__}")

    prompts: list[str] = []
    categories: list[str] = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"Entry {i} in {path} is not a dict: {entry}")
        if "text" not in entry or "category" not in entry:
            raise ValueError(f"Entry {i} in {path} missing 'text' or 'category': {entry}")
        prompts.append(str(entry["text"]))
        categories.append(str(entry["category"]))

    log.info("Loaded %d prompts from %s (%d categories)", len(prompts), path, len(set(categories)))
    return prompts, categories


def suite_hash(prompts: list[str], categories: list[str]) -> str:
    """Deterministic hash of a prompt suite (for cache keying + reproducibility).

    Order-independent (sorted) so reordering doesn't invalidate caches.
    """
    pairs = sorted(zip(prompts, categories, strict=True))
    payload = "\n".join(f"{p}\t{c}" for p, c in pairs)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
