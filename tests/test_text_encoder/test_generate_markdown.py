"""Tests for generate_markdown.py — HTML report -> Hugo markdown export."""

from __future__ import annotations

import base64

from src.text_encoder.generate_markdown import convert

SVG = b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>'


def _b64_svg() -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(SVG).decode()


def _wrap(body: str) -> str:
    return f"<html><head><style>x</style></head><body><h1>Report title</h1>{body}</body></html>"


def test_front_matter_and_headings(tmp_path) -> None:  # type: ignore[no-untyped-def]
    html = _wrap('<h2 id="sec-x">Section X</h2><p>hello</p>')
    md = convert(html, tmp_path / "figures", "My Report", "te-zimage")
    assert md.startswith("---\n")
    assert 'title: "My Report"' in md
    assert 'comparison: "te-zimage"' in md
    assert "draft: true" in md
    assert "## Section X {#sec-x}" in md
    # first h1 dropped (front matter carries the title)
    assert "Report title" not in md


def test_tables_and_inline_formatting(tmp_path) -> None:  # type: ignore[no-untyped-def]
    html = _wrap(
        "<table><tr><th>Variant</th><th>LPIPS</th></tr>"
        "<tr><td><b>heretic</b></td><td>0.29</td></tr></table>"
        '<p>see <a href="https://arxiv.org/abs/1801.03924">LPIPS</a> and <code>x</code></p>'
    )
    md = convert(html, tmp_path / "figures", "T", "te-zimage")
    assert "| Variant | LPIPS |" in md
    assert "| --- | --- |" in md
    assert "**heretic**" in md
    assert "[LPIPS](https://arxiv.org/abs/1801.03924)" in md
    assert "`x`" in md


def test_script_style_dropped(tmp_path) -> None:  # type: ignore[no-untyped-def]
    html = _wrap("<p>keep</p><script>var x=1;</script><style>a{}</style>")
    md = convert(html, tmp_path / "figures", "T", "te-zimage")
    assert "keep" in md
    assert "var x" not in md
    assert "a{}" not in md


def test_svg_figure_extraction(tmp_path) -> None:  # type: ignore[no-untyped-def]
    html = _wrap(f'<p><img src="{_b64_svg()}"></p>')
    md = convert(html, tmp_path / "figures", "T", "te-zimage")
    assert "![figure](figures/fig01.svg)" in md
    fig = tmp_path / "figures" / "fig01.svg"
    assert fig.exists()
    assert fig.read_bytes() == SVG


def test_gallery_replaced_by_shortcode_placeholder(tmp_path) -> None:  # type: ignore[no-untyped-def]
    html = _wrap('<h2 id="sec-ab">A/B gallery</h2><div>slider stuff</div><h2 id="sec-next">Next</h2><p>after</p>')
    md = convert(html, tmp_path / "figures", "T", "te-zimage")
    assert "slider stuff" not in md
    assert "Hugo shortcode placeholder" in md
    assert "## Next" in md  # section AFTER the gallery survives


def test_nav_and_comments(tmp_path) -> None:  # type: ignore[no-untyped-def]
    html = _wrap("<nav><a href='#x'>toc</a></nav><p>body</p><!-- {{< myshortcode >}} -->")
    md = convert(html, tmp_path / "figures", "T", "te-zimage")
    assert "toc" not in md
    assert "<!-- {{< myshortcode >}} -->" in md


def test_publish_flag_clears_draft(tmp_path) -> None:  # type: ignore[no-untyped-def]
    md = convert(_wrap("<p>x</p>"), tmp_path / "figures", "T", "te-zimage", draft=False)
    assert "draft: false" in md


def test_glossary_dl_dt_dd_not_glued(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # the real report emits <details class="glossary"><dl>… — regression test
    # for the dt/dd gluing defect (review round 1)
    html = _wrap(
        "<details><summary>Glossary</summary><dl>"
        "<dt>divergence</dt><dd>the raw measurement.</dd>"
        "<dt>damage</dt><dd>collateral divergence.</dd>"
        "</dl></details>"
    )
    md = convert(html, tmp_path / "figures", "T", "te-zimage")
    assert "**divergence**" in md
    assert "the raw measurement." in md
    assert "**damage**" in md
    # the defect glued term+definition with no separator
    assert "divergencethe" not in md
    assert "damagecollateral" not in md
    assert md.index("divergence") < md.index("the raw measurement.")


def test_glossary_unclosed_dt_dd_generator_style(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # generate_report.py emits UNCLOSED <dt>term<dd>def (HTML implicit close)
    html = _wrap(
        "<details class='glossary'><summary>Glossary</summary><dl>"
        "<dt>divergence<dd>the raw measurement."
        "<dt>damage<dd>collateral divergence."
        "</dl></details>"
    )
    md = convert(html, tmp_path / "figures", "T", "te-zimage")
    assert "**divergence**" in md
    assert "**damage**" in md
    assert "divergencethe" not in md
    # every ** opened must be closed (no stray bold runs)
    assert md.count("**") % 2 == 0


def test_span_ci_inline_confidence(tmp_path) -> None:  # type: ignore[no-untyped-def]
    html = _wrap("<p>score 0.29 <span class='ci'>[0.27, 0.31]</span></p>")
    md = convert(html, tmp_path / "figures", "T", "te-zimage")
    assert "[0.27, 0.31]" in md


def test_multiple_figures_numbered_sequentially(tmp_path) -> None:  # type: ignore[no-untyped-def]
    html = _wrap(f'<p><img src="{_b64_svg()}"><img src="{_b64_svg()}"></p>')
    md = convert(html, tmp_path / "figures", "T", "te-zimage")
    assert "![figure](figures/fig01.svg)" in md
    assert "![figure](figures/fig02.svg)" in md
    assert (tmp_path / "figures" / "fig02.svg").exists()


def test_pipe_in_table_cell_escaped(tmp_path) -> None:  # type: ignore[no-untyped-def]
    html = _wrap("<table><tr><th>a|b</th></tr></table>")
    md = convert(html, tmp_path / "figures", "T", "te-zimage")
    assert "a\\|b" in md
