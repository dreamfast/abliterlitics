"""Hugo markdown export — converts te_report.html into a Hugo-ready .md.

The HTML report is the single source of truth (verdict logic, tables, links);
this module converts its structure to markdown instead of duplicating
renderers. What it does:

  - front matter (title/date/draft) + section heading anchors ({{#sec-x}})
  - h2/h3/p/ul/ol/tables → markdown equivalents
  - <b>/<i>/<code>/<a> inline formatting preserved (links → [text](url))
  - inline SVG figures extracted to files (figures/figNN.svg) and referenced
  - the interactive A/B gallery (#sec-ab) is REPLACED by a placeholder
    comment for a Hugo shortcode (the slider is a site-side concern)
  - <script>/<style>/<noscript> dropped

Usage:
    python3 -m src.text_encoder.generate_markdown \
        --html graphs/te_report.html --out graphs/te_report.md \
        --figures-dir graphs/figures --title "..." --comparison te-zimage
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import re
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

log = logging.getLogger(__name__)

DROP_TAGS = {"script", "style", "noscript", "head", "svg", "hr"}


class _Md(HTMLParser):
    def __init__(self, figures_dir: Path, fig_prefix: str = "fig") -> None:
        super().__init__(convert_charrefs=True)
        self.figures_dir = figures_dir
        self.fig_prefix = fig_prefix
        self.out: list[str] = []
        self.fig_n = 0
        # stack of inline formatting marks
        self.marks: list[str] = []
        # table state
        self.in_table = False
        self.row: list[str] = []
        self.cell: list[str] | None = None
        self.table_rows: list[list[str]] = []
        # list state
        self.list_stack: list[str] = []
        # skip state
        self.drop_depth = 0
        self.cur_heading_id = ""
        # dl/dt/dd state — the report generator emits UNCLOSED <dt>term<dd>def
        # (HTML-legal implicit close), so we track the open item ourselves
        self._dl_item = ""

    # ---------- helpers ----------
    def _close_dl_item(self) -> None:
        """Close an implicitly-open dl item (unclosed <dt> before <dd>/<dt>/</dl>)."""
        if self._dl_item == "dt":
            self._emit_inline("**")
        self._dl_item = ""

    def _emit_inline(self, text: str) -> None:
        if self.cell is not None:
            self.cell.append(text)
        else:
            self.out.append(text)

    def _nl(self, count: int = 1) -> None:
        if self.cell is not None:
            return
        self.out.append("\n" * count)

    def _cell_text(self) -> str:
        t = "".join(self.cell or [])
        t = re.sub(r"\s+", " ", t).strip()
        return t.replace("|", "\\|")

    # ---------- tags ----------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:  # noqa: C901
        a = dict(attrs)
        if tag in DROP_TAGS:
            self.drop_depth += 1
            return
        if self.drop_depth:
            return
        if tag == "table":
            self.in_table = True
            self.table_rows = []
        elif tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell = []
        elif tag in ("ul", "ol"):
            self.list_stack.append(tag)
            self._nl()
        elif tag == "li":
            depth = max(0, len(self.list_stack) - 1)
            marker = "- " if not self.list_stack or self.list_stack[-1] == "ul" else "1. "
            self.out.append("\n" + "  " * depth + marker)
        elif tag in ("h1", "h2", "h3", "h4"):
            self._nl(2)
            self.cur_heading_id = a.get("id") or ""
            # h1 is dropped (front matter carries the title); h2 -> ##
            self.out.append("#" * max(2, int(tag[1])) + " ")
        elif tag == "p":
            self._nl(2)
        elif tag == "br":
            self._emit_inline("\n")
        elif tag == "b" or tag == "strong":
            self.marks.append("**")
            self._emit_inline("**")
        elif tag in ("i", "em"):
            self.marks.append("*")
            self._emit_inline("*")
        elif tag == "code":
            self.marks.append("`")
            self._emit_inline("`")
        elif tag == "a":
            href = a.get("href") or ""
            self.marks.append(f"]({href})" if href else "]()")
            self._emit_inline("[")
        elif tag == "span":
            cls = a.get("class") or ""
            if "ci" in cls.split():
                self._emit_inline(" [")
                self.marks.append("]")
        elif tag == "summary":
            self._nl(2)
            self.marks.append("**")
            self._emit_inline("**")
        elif tag == "dl":
            self._close_dl_item()
            self._nl(2)
        elif tag == "dt":
            self._close_dl_item()
            self._nl(2)
            self._dl_item = "dt"
            self._emit_inline("**")
        elif tag == "dd":
            if self._dl_item == "dt":  # implicit close (unclosed <dt>…<dd>)
                self._emit_inline("**")
            self._dl_item = "dd"
            self._nl(1)
            self._emit_inline("  ")  # indented definition under its term
        elif tag == "details" or tag in ("div", "blockquote", "figure"):
            self._nl(2)
        elif tag == "img":
            src = a.get("src") or ""
            if src.startswith("data:image/svg+xml;base64,"):
                self._extract_figure(src)

    def handle_endtag(self, tag: str) -> None:  # noqa: C901
        if tag in DROP_TAGS:
            self.drop_depth = max(0, self.drop_depth - 1)
            return
        if self.drop_depth:
            return
        if tag == "table":
            self._flush_table()
            self.in_table = False
        elif tag == "tr":
            if self.row:
                self.table_rows.append(self.row)
            self.row = []
        elif tag in ("td", "th"):
            self.row.append(self._cell_text())
            self.cell = None
        elif tag in ("ul", "ol"):
            if self.list_stack:
                self.list_stack.pop()
            self._nl(2)
        elif tag in ("h1", "h2", "h3", "h4"):
            if self.cur_heading_id:
                self.out.append(" {" + f"#{self.cur_heading_id}" + "}")
            self.cur_heading_id = ""
            self._nl(2)
        elif tag == "p":
            self._nl(2)
        elif tag in ("b", "strong", "i", "em", "code"):
            mark = (
                self.marks.pop()
                if self.marks
                else ("**" if tag in ("b", "strong") else ("`" if tag == "code" else "*"))
            )
            self._emit_inline(mark)
        elif tag == "a":
            mark = self.marks.pop() if self.marks else "]()"
            self._emit_inline(mark)
        elif tag == "span":
            if self.marks and self.marks[-1] == "]":
                self.marks.pop()
                self._emit_inline("]")
        elif tag == "summary":
            if self.marks and self.marks[-1] == "**":
                self.marks.pop()
                self._emit_inline("**")
            self._nl(2)
        elif tag == "dl" or tag in ("dt", "dd"):
            self._close_dl_item()
            self._nl(2)
        elif tag in ("details", "div", "blockquote", "figure"):
            self._nl(2)

    def handle_comment(self, data: str) -> None:
        if self.drop_depth or self.cell is not None:
            return
        # HTML comments carry Hugo shortcode placeholders — keep them
        self._nl(2)
        self.out.append(f"<!--{data}-->")
        self._nl(2)

    def handle_data(self, data: str) -> None:
        if self.drop_depth:
            return
        if not data.strip() and self.cell is None:
            # preserve single spaces between inline elements
            if self.out and not self.out[-1].endswith((" ", "\n", "[")):
                self.out.append(" ")
            return
        self._emit_inline(data)

    # ---------- figures & tables ----------
    def _extract_figure(self, src: str) -> None:
        try:
            data = base64.b64decode(src.split(",", 1)[1])
            self.fig_n += 1
            self.figures_dir.mkdir(parents=True, exist_ok=True)
            name = f"{self.fig_prefix}{self.fig_n:02d}.svg"
            (self.figures_dir / name).write_bytes(data)
            # Markdown reference assumes figures_dir is a subdir next to the
            # exported .md (the default out.parent/"figures"); any other
            # --figures-dir location is a caller responsibility.
            link = f"{self.figures_dir.name}/{name}"
            self._emit_inline(f"\n\n![figure]({link})\n\n")
            log.info("extracted figure %s (%d KB)", name, len(data) // 1024)
        except Exception as exc:  # noqa: BLE001 - figure extraction must not kill export
            log.warning("figure extraction failed: %s", exc)

    def _flush_table(self) -> None:
        rows = [r for r in self.table_rows if r]
        if not rows:
            return
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        self._nl(2)
        self.out.append("| " + " | ".join(rows[0]) + " |\n")
        self.out.append("|" + "|".join([" --- "] * width) + "|\n")
        for r in rows[1:]:
            self.out.append("| " + " | ".join(r) + " |\n")
        self._nl(2)

    def markdown(self) -> str:
        raw = "".join(self.out)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip() + "\n"


GALLERY_PLACEHOLDER = """<!-- A/B interactive gallery lives here in the HTML report.
     Hugo shortcode placeholder — the image comparator (same-seed pairing,
     seed slider, per-word attention/content readout) is site-side.
     Images: graphs/ab_images/<label>/<variant>_s<seed>.webp
     Data: WORDS payload + manifest per te_report.html source. -->
"""


def convert(
    html_text: str,
    figures_dir: Path,
    title: str,
    comparison: str,
    draft: bool = True,
) -> str:
    """Full HTML report -> Hugo markdown document."""
    # 1. drop the interactive gallery section (h2#sec-ab .. next h2/</body>).
    #    NOTE: m is matched on the full html but replaced in the body slice —
    #    safe because the gallery always sits after <body> (own-generator
    #    contract), but keep that invariant in mind if the generator changes.
    m = re.search(r'<h2 id="sec-ab".*?(?=<h2|</body>)', html_text, re.S)
    body = html_text[html_text.index("<body") :] if "<body" in html_text else html_text
    body = re.sub(r"<body[^>]*>", "", body, count=1).replace("</body>", "").replace("</html>", "")
    if m:
        body = body.replace(m.group(0), f"<p>{GALLERY_PLACEHOLDER}</p>")
    # 2. drop the HTML TOC (Hugo builds its own); it sits between the h1 block
    #    and the first h2 — identified by <nav ...>...</nav>
    body = re.sub(r"<nav.*?</nav>", "", body, flags=re.S)
    # 3. drop the first h1 (front matter carries the title)
    body = re.sub(r"<h1[^>]*>.*?</h1>", "", body, count=1, flags=re.S)

    parser = _Md(figures_dir)
    parser.feed(body)
    md_body = parser.markdown()

    front = {
        "title": title,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "draft": draft,
        "comparison": comparison,
        "description": "Measurement report: does swapping the text encoder change image generation?",
    }
    fm = "---\n" + "\n".join(f"{k}: {json.dumps(v)}" for k, v in front.items()) + "\n---\n"
    return fm + "\n" + md_body


def main() -> int:
    ap = argparse.ArgumentParser(description="Hugo markdown export of the TE report")
    ap.add_argument("--html", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figures-dir", default="")
    ap.add_argument("--title", default="Text-Encoder Conditioning Fidelity Report")
    ap.add_argument("--comparison", default="te-zimage")
    ap.add_argument("--publish", action="store_true", help="draft: false")
    args = ap.parse_args()

    html_path = Path(args.html)
    out_path = Path(args.out)
    figures_dir = Path(args.figures_dir) if args.figures_dir else out_path.parent / "figures"
    md = convert(html_path.read_text(), figures_dir, args.title, args.comparison, draft=not args.publish)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    log.info("Wrote %s (%d KB, %d figures)", out_path, len(md) // 1024, len(list(figures_dir.glob("*.svg"))))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
