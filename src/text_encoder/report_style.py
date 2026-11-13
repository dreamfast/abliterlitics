"""Shared HTML styling constants for the text-encoder report generations.

One home for the style rules duplicated across report builders (generate_report,
h3_report): the base font stack, the verdict-cell / damage-cell CSS, and the
verdict-text -> CSS-class mapping. Builders interpolate these constants into
their own ``<style>`` blocks so cell semantics can never drift apart.
"""

from __future__ import annotations

FONT_STACK = "-apple-system, 'Segoe UI', Roboto, sans-serif"
ACCENT = "#4c72b0"

# Verdict cells (h3_report verdict table): td class -> color rule.
CSS_VERDICT_CELLS = (
    " td.ok { color: #1e7d32; font-weight: 600; }\n"
    " td.bad { color: #b3261e; font-weight: 600; }\n"
    " td.mid { color: #946200; font-weight: 600; }"
)

# Damage / reference-variant cells (generate_report divergence table).
CSS_DAMAGE_CELLS = (
    " .dmg-high { color: #c0392b; font-weight: bold; }\n"
    " .dmg-low { color: #27ae60; }\n"
    " .ref-var { color: #7f8c8d; font-style: italic; }"
)


def verdict_css_class(verdict: str) -> str:
    """Map a verdict string to its table-cell CSS class (ok / bad / mid).

    SUPPORTED (and not PARTIAL) -> ok; REFUTED without a benign
    "no damage" / "no structure" null -> bad; everything else -> mid.
    """
    benign_null = "no damage" in verdict or "no structure" in verdict
    if "SUPPORTED" in verdict and "PARTIAL" not in verdict:
        return "ok"
    if "REFUTED" in verdict and not benign_null:
        return "bad"
    return "mid"
