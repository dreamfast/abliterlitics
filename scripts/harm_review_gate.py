"""Overseer accuracy gate for HarmBench LLM-review batches.

Validates a reviewer results file against its batch (or a 0-indexed range of
it): full coverage on the composite (model_name, behavior_id) key, verdicts
within the 4-way enum, no duplicates, verbatim key copies, and a minimum
specificity floor on review_reasoning.

Usage:
    PYTHONPATH=. python3 scripts/harm_review_gate.py <batch.json> <results.json> [start] [end]

Exit 0 = pass, 1 = violation (re-dispatch the reviewer with the violation quoted).
"""

import json
import sys

ENUM = {"no_refusal", "refusal", "soft_refusal", "degenerate"}


def key(r: dict[str, str]) -> tuple[str, str]:
    return (r["model_name"], r["behavior_id"])


def main() -> None:
    with open(sys.argv[1]) as fh:
        batch = json.load(fh)
    with open(sys.argv[2]) as fh:
        results = json.load(fh)
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    end = int(sys.argv[4]) if len(sys.argv) > 4 else len(batch)
    batch = batch[start:end]

    bset = {key(i) for i in batch}
    ids = [key(r) for r in results]
    bad = [key(r) for r in results if r.get("verdict") not in ENUM]
    missing, extra = bset - set(ids), set(ids) - bset
    dupes = len(ids) != len(set(ids))
    verbat = (
        all(
            i["model_name"] == r.get("model_name") and i["behavior_id"] == r.get("behavior_id")
            for i, r in zip(batch, results, strict=False)
        )
        if len(results) == len(batch)
        else False
    )
    thin = [key(r) for r in results if len(r.get("review_reasoning", "")) < 40]
    print(
        f"covered={len(set(ids) & bset)}/{len(bset)} rows={len(results)} dupes={dupes} "
        f"bad={bad[:3]} missing={len(missing)} extra={len(extra)} "
        f"keys_verbatim={verbat} thin_reasoning={thin}"
    )
    sys.exit(1 if (bad or missing or extra or dupes or not verbat or thin) else 0)


if __name__ == "__main__":
    main()
