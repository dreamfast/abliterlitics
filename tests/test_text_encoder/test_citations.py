"""Tests for citations.py — REFS registry, ref() anchors, refs appendix."""

from __future__ import annotations

from src.text_encoder.citations import REFS, ref, refs_section_html


def test_refs_entries_wellformed() -> None:
    assert len(REFS) >= 12
    for key, (label, url, rel) in REFS.items():
        assert label and rel, f"{key}: label/relationship must be non-empty"
        assert url.startswith("https://"), f"{key}: url must be https, got {url}"
        # labels carry author-year for papers; relationship ties to our findings
        assert len(rel) > 20, f"{key}: relationship too terse"


def test_ref_known_key_renders_anchor() -> None:
    html = ref("lpips2018")
    assert html.startswith('<a href="https://')
    assert "Zhang" in html
    assert 'target="_blank"' in html
    assert 'rel="noopener"' in html


def test_ref_unknown_key_returns_empty_and_never_raises() -> None:
    assert ref("does-not-exist") == ""


def test_refs_section_contains_all_entries() -> None:
    section = refs_section_html()
    assert 'id="sec-refs"' in section
    for label, url, _rel in REFS.values():
        assert url in section, f"missing url {url}"
        assert label.split(" — ")[0] in section
    # relationship prose is included (it is the value of the appendix)
    assert section.count("<li>") == len(REFS)
