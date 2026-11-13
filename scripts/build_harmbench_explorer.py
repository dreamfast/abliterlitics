#!/usr/bin/env python3
"""Build a browsable, static HarmBench response explorer for the abliterlitics website.

Generates, per comparison:
  <output-dir>/index.html          — summary table (arm -> judge ASR, complied/refused,
                                      classifier-only badge, links) + explainer
  <output-dir>/<model_name>.html   — one page per arm: sticky filter toolbar (verdict /
                                      category / free-text search + counts readout), stats
                                      strip, category breakdown, and one card per response
                                      (behavior text, judge + classifier badges, inline
                                      judge reasoning, collapsible FULL response and FULL
                                      reasoning trace — never truncated, lesson #38).

Design constraints:
  - Plain python3 stdlib ONLY (string templates, no jinja2 dependency).
  - Server-rendered static HTML: every card is fully present with JS disabled;
    filters/search are progressive enhancement (vanilla JS, no libraries,
    no external requests, no images).
  - Scoped CSS (.hb- prefix), professional light theme with the site's navy/teal accent
    family. All model output escaped with html.escape.
  - Comparison-agnostic: arms are read from the DB; think-mode arms absent from the
    models table get derived display names.

Usage:
  python3 scripts/build_harmbench_explorer.py --comparison gemma4-12b \
      --db abliterlitics.db \
      --output-dir abliterlitics_website/static/harmbench/gemma4-12b

Hugo mode (--hugo-content-dir) emits Hugo content pages instead of standalone
HTML: <dir>/_index.md + _index.body.frag (layout: harmbench-index) and leaf
bundles <dir>/<arm>/index.md + <dir>/<arm>/body.frag (layout: harmbench-arm,
explicit url: so mixed-case arm names survive Hugo's URL lowercasing). The .md
files carry front matter only; the body lives in the sibling body.frag file
which the layout templates read verbatim via readFile. Leaf-bundle resources
are not published, and _index.md sets _build.publishResources=false, so the
frag files stay out of public/. This bypasses goldmark entirely — raw HTML
bodies inside markdown would be split at blank lines (CommonMark HTML block
rule) and response text would be re-parsed as markdown, mangling
emphasis/bullets inside <pre> content. Cross-links use the Hugo pretty-URL
form /<site-prefix>/<comparison>/<arm>/.
"""

from __future__ import annotations

import argparse
import html
import re
import sqlite3
import sys
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
from pathlib import Path

CSS = """
:root {
  --hb-bg: #f6f8fa; --hb-surface: #ffffff; --hb-ink: #111827; --hb-ink2: #4b5563;
  --hb-ink3: #6b7280; --hb-line: #e2e8f0; --hb-line2: #cbd5e1;
  --hb-navy: #0a0e17; --hb-teal: #0d9488; --hb-teal-ink: #0f766e;
  --hb-red-bg: #fee2e2; --hb-red-ink: #b91c1c;
  --hb-green-bg: #dcfce7; --hb-green-ink: #15803d;
  --hb-grey-bg: #e5e7eb; --hb-grey-ink: #4b5563;
  --hb-mono: "JetBrains Mono", "Fira Code", "SF Mono", Consolas, monospace;
}
.hb-root { margin: 0; padding: 0 0 4rem; background: var(--hb-bg); color: var(--hb-ink);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
.hb-root * { box-sizing: border-box; }
.hb-wrap { max-width: 68rem; margin: 0 auto; padding: 0 1.25rem; }
.hb-topnav { background: var(--hb-navy); color: #e8edf5; font-size: .85rem; }
.hb-topnav .hb-wrap { display: flex; gap: 1.25rem; align-items: center; padding: .55rem 1.25rem; flex-wrap: wrap; }
.hb-topnav a { color: #7dd3c8; text-decoration: none; }
.hb-topnav a:hover { text-decoration: underline; }
.hb-topnav .hb-crumb { color: #94a3b8; }
.hb-topnav .hb-prevnext { margin-left: auto; display: flex; gap: 1rem; }
.hb-header { border-bottom: 1px solid var(--hb-line2); background: var(--hb-surface); }
.hb-header .hb-wrap { padding: 1.6rem 1.25rem 1.2rem; }
.hb-header h1 { margin: 0 0 .2rem; font-size: 1.55rem; letter-spacing: -.01em; }
.hb-header h1 code { font-family: var(--hb-mono); font-size: .95rem; color: var(--hb-ink2);
  background: #f1f5f9; padding: .1em .45em; border-radius: 4px; vertical-align: middle; }
.hb-header .hb-sub { color: var(--hb-ink3); font-size: .9rem; margin: 0; }
.hb-totals { display: flex; gap: 2rem; flex-wrap: wrap; margin-top: .8rem; color: var(--hb-ink2); font-size: .9rem; }
.hb-totals b { display: block; font-size: 1.3rem; color: var(--hb-ink); }
.hb-toolbar { position: sticky; top: 0; z-index: 10; background: var(--hb-surface);
  border-bottom: 1px solid var(--hb-line2); box-shadow: 0 1px 3px rgba(10,14,23,.06); }
.hb-toolbar .hb-wrap { display: flex; gap: .6rem; align-items: center; flex-wrap: wrap; padding: .6rem 1.25rem; }
.hb-btn { border: 1px solid var(--hb-line2); background: var(--hb-surface); color: var(--hb-ink2);
  border-radius: 6px; padding: .3rem .8rem; font: inherit; font-size: .85rem; cursor: pointer; }
.hb-btn:hover { border-color: var(--hb-teal); color: var(--hb-teal-ink); }
.hb-btn.hb-on { background: var(--hb-teal); border-color: var(--hb-teal); color: #fff; font-weight: 600; }
.hb-select, .hb-search { border: 1px solid var(--hb-line2); border-radius: 6px; padding: .32rem .6rem;
  font: inherit; font-size: .85rem; background: var(--hb-surface); color: var(--hb-ink); }
.hb-search { flex: 1 1 14rem; min-width: 12rem; }
.hb-count { font-size: .82rem; color: var(--hb-ink3); font-variant-numeric: tabular-nums; white-space: nowrap; }
.hb-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(8.5rem, 1fr)); gap: .75rem;
  margin: 1.25rem 0; }
.hb-stat { background: var(--hb-surface); border: 1px solid var(--hb-line); border-radius: 8px; padding: .6rem .9rem; }
.hb-stat b { display: block; font-size: 1.25rem; line-height: 1.3; font-variant-numeric: tabular-nums; }
.hb-stat span { font-size: .78rem; color: var(--hb-ink3); text-transform: uppercase; letter-spacing: .03em; }
.hb-stat.hb-hero { border-left: 4px solid var(--hb-teal); }
.hb-stat.hb-hero b { font-size: 1.9rem; color: var(--hb-teal-ink); }
.hb-section { background: var(--hb-surface); border: 1px solid var(--hb-line); border-radius: 8px;
  padding: 1rem 1.25rem; margin: 1.25rem 0; }
.hb-section h2 { margin: 0 0 .5rem; font-size: 1.02rem; }
.hb-explain { color: var(--hb-ink2); font-size: .9rem; margin: 0; }
.hb-table { border-collapse: collapse; width: 100%; font-size: .88rem; }
.hb-table th, .hb-table td { border-bottom: 1px solid var(--hb-line); padding: .4rem .6rem; text-align: left; }
.hb-table th { color: var(--hb-ink3); font-size: .78rem; text-transform: uppercase; letter-spacing: .03em; }
.hb-table td.hb-num, .hb-table th.hb-num { text-align: right; font-variant-numeric: tabular-nums; }
.hb-table tr:hover td { background: #f8fafc; }
.hb-table a { color: var(--hb-teal-ink); text-decoration: none; font-weight: 600; }
.hb-table a:hover { text-decoration: underline; }
.hb-grouplabel { margin: 1.6rem 0 .4rem; font-size: .8rem; text-transform: uppercase;
  letter-spacing: .05em; color: var(--hb-ink3); }
.hb-badge { display: inline-block; font-size: .72rem; font-weight: 700; padding: .12rem .55rem;
  border-radius: 999px; text-transform: uppercase; letter-spacing: .02em; white-space: nowrap; }
.hb-badge.hb-j-complied { background: var(--hb-red-bg); color: var(--hb-red-ink); }
.hb-badge.hb-j-refused { background: var(--hb-green-bg); color: var(--hb-green-ink); }
.hb-badge.hb-j-soft { background: #fef3c7; color: #92400e; }
.hb-badge.hb-j-degen { background: #ede9fe; color: #5b21b6; }
.hb-badge.hb-j-none { background: var(--hb-grey-bg); color: var(--hb-grey-ink); }
.hb-badge.hb-clf { background: #f1f5f9; color: var(--hb-grey-ink); font-weight: 600; }
.hb-cards { display: flex; flex-direction: column; gap: .65rem; margin-top: 1.25rem; }
.hb-card { background: var(--hb-surface); border: 1px solid var(--hb-line); border-radius: 8px;
  padding: .7rem 1rem .8rem; }
.hb-card.hb-vc { border-left: 4px solid var(--hb-red-ink); }
.hb-card.hb-vr { border-left: 4px solid var(--hb-green-ink); }
.hb-card.hb-vs { border-left: 4px solid #d97706; }
.hb-card.hb-vd { border-left: 4px solid #7c3aed; }
.hb-card.hb-vn { border-left: 4px solid var(--hb-line2); }
.hb-cardhead { display: flex; gap: .65rem; align-items: center; flex-wrap: wrap; }
.hb-bid { font-family: var(--hb-mono); font-weight: 700; font-size: .82rem; }
.hb-chip { font-size: .72rem; color: var(--hb-teal-ink); background: #f0fdfa;
  border: 1px solid #ccfbf1; padding: .08rem .5rem; border-radius: 999px; }
.hb-meta { margin-left: auto; color: var(--hb-ink3); font-size: .75rem;
  font-variant-numeric: tabular-nums; font-family: var(--hb-mono); }
.hb-behavior { margin: .45rem 0 .3rem; color: var(--hb-ink); }
.hb-judge { font-size: .84rem; color: var(--hb-ink2); margin: .3rem 0; }
.hb-judge b { color: var(--hb-ink); }
.hb-block { margin: .3rem 0; }
.hb-block summary { cursor: pointer; color: var(--hb-teal-ink); font-size: .8rem; user-select: none; }
.hb-block summary:hover { text-decoration: underline; }
.hb-block pre { white-space: pre-wrap; word-break: break-word; background: #f8fafc;
  border: 1px solid var(--hb-line); border-radius: 6px; padding: .6rem .75rem; margin: .35rem 0 0;
  font: .78rem/1.5 var(--hb-mono); max-height: 26rem; overflow-y: auto; }
.hb-block pre.hb-empty { color: var(--hb-ink3); font-style: italic; }
.hb-error { color: var(--hb-red-ink); font-size: .8rem; margin: .3rem 0; }
.hb-footer { margin-top: 2rem; color: var(--hb-ink3); font-size: .78rem; text-align: center; }
@media (max-width: 40rem) { .hb-meta { margin-left: 0; width: 100%; } }
"""

JS = """
(function () {
  'use strict';
  var cards = Array.prototype.slice.call(document.querySelectorAll('.hb-card'));
  var counter = document.getElementById('hb-count');
  var buttons = Array.prototype.slice.call(
    document.querySelectorAll('.hb-toolbar .hb-btn'));
  var catSel = document.getElementById('hb-cat');
  var search = document.getElementById('hb-search');
  var verdict = 'all', cat = 'all', query = '', timer = null;
  function apply() {
    var q = query.trim().toLowerCase(), shown = 0;
    cards.forEach(function (c) {
      var ok = (verdict === 'all' || c.getAttribute('data-verdict') === verdict) &&
               (cat === 'all' || c.getAttribute('data-cat') === cat);
      if (ok && q) ok = c.textContent.toLowerCase().indexOf(q) !== -1;
      c.style.display = ok ? '' : 'none';
      if (ok) shown++;
    });
    if (counter) counter.textContent = 'showing ' + shown + ' of ' + cards.length;
  }
  buttons.forEach(function (b) {
    b.addEventListener('click', function () {
      buttons.forEach(function (x) { x.classList.remove('hb-on'); });
      b.classList.add('hb-on');
      verdict = b.getAttribute('data-verdict');
      apply();
    });
  });
  if (catSel) catSel.addEventListener('change', function () {
    cat = catSel.value; apply();
  });
  if (search) search.addEventListener('input', function () {
    clearTimeout(timer);
    timer = setTimeout(function () { query = search.value; apply(); }, 150);
  });
  apply();
})();
"""


def _pct(x):
    """Fraction -> percentage 0-100, half-up (house convention)."""
    if x is None:
        return None
    return float((Decimal(str(x)) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def esc(s) -> str:
    return html.escape(str(s) if s is not None else "", quote=True)


def fmt_int(n) -> str:
    return f"{int(n):,}"


def judge_badge(verdict: str | None) -> str:
    if verdict == "complied":
        return '<span class="hb-badge hb-j-complied">complied</span>'
    if verdict == "refused":
        return '<span class="hb-badge hb-j-refused">refused</span>'
    if verdict == "no_refusal":
        return '<span class="hb-badge hb-j-complied">no_refusal</span>'
    if verdict == "soft_refusal":
        return '<span class="hb-badge hb-j-soft">soft_refusal</span>'
    if verdict == "refusal":
        return '<span class="hb-badge hb-j-refused">refusal</span>'
    if verdict == "degenerate":
        return '<span class="hb-badge hb-j-degen">degenerate</span>'
    return '<span class="hb-badge hb-j-none">no judge</span>'


def clf_badge(label, confidence) -> str:
    if not label:
        return '<span class="hb-badge hb-j-none">no classifier</span>'
    extra = f" &middot; {esc(confidence)}" if confidence else ""
    return f'<span class="hb-badge hb-clf">clf: {esc(label)}{extra}</span>'


def is_thinkmode(model_name: str) -> bool:
    return "thinkmode" in model_name.lower()


def fallback_display_name(model_name: str, models_map: dict[str, str],
                          base_display: str = "Base") -> str:
    """Derive a display name for arms absent (or blank) in the models table."""
    if is_thinkmode(model_name):
        stem = model_name
        for suffix in ("_thinkmode_v2", "_thinkmode"):
            stem = stem[: -len(suffix)] if stem.endswith(suffix) else stem
        if stem in ("", "base"):
            base_dn = base_display
        elif stem in models_map:
            base_dn = models_map[stem]
        else:
            # substring-match a real arm (e.g. "smoothed" -> sdft_smoothed_r2_new)
            cands = [dn for m, dn in models_map.items() if stem in m]
            base_dn = cands[0] if cands else stem
        v2 = " (v2)" if model_name.endswith("_v2") else ""
        return f"{base_dn}{v2} — think render (enable_thinking=true)"
    return model_name


def load_data(db: sqlite3.Connection, comparison: str,
              exclude: set[str] = frozenset()):
    models_map = {
        r[0]: r[1]
        for r in db.execute(
            "SELECT model_name, display_name FROM models WHERE comparison = ?",
            (comparison,),
        )
        if r[1]
    }
    arms = [
        r[0]
        for r in db.execute(
            "SELECT model_name, COUNT(*) FROM responses "
            "WHERE comparison = ? GROUP BY model_name ORDER BY model_name",
            (comparison,),
        )
        if r[0] not in exclude
    ]
    behaviors = {
        r[0]: {"text": r[1] or "", "cat": r[2] or "uncategorized"}
        for r in db.execute(
            "SELECT behavior_id, behavior_text, semantic_category FROM behaviors"
        )
    }
    # Latest review per (model, behavior) wins if several reviewers exist.
    # Display name strips internal rubric-era suffixes (glm-5.3-flash-v2 -> glm-5.3-flash);
    # the DB reviewer id is untouched.
    def reviewer_display(r: str) -> str:
        return re.sub(r"-v\d+$", "", r or "")
    reviews: dict[tuple[str, str], dict] = {}
    for mid, bid, reviewer, verdict, reasoning, at in db.execute(
        "SELECT model_name, behavior_id, reviewer_model, verdict, review_reasoning, reviewed_at "
        "FROM llm_reviews WHERE comparison = ? ORDER BY reviewed_at",
        (comparison,),
    ):
        reviews[(mid, bid)] = {
            "reviewer": reviewer_display(reviewer), "verdict": verdict, "reasoning": reasoning,
        }
    clas: dict[tuple[str, str], dict] = {}
    for mid, bid, label, is_ref, layer, reason, conf, clf in db.execute(
        "SELECT model_name, behavior_id, label, is_refusal, layer, reason, confidence, classifier "
        "FROM v_latest_classification WHERE comparison = ?",
        (comparison,),
    ):
        clas[(mid, bid)] = {
            "label": label, "is_refusal": is_ref, "layer": layer,
            "reason": reason, "confidence": conf, "classifier": clf,
        }
    arms_data = {}
    for arm in arms:
        rows = db.execute(
            "SELECT behavior_id, response_text, reasoning_text, completion_tokens, "
            "finish_reason, error, reasoning_chars "
            "FROM responses WHERE comparison = ? AND model_name = ? ORDER BY behavior_id",
            (comparison, arm),
        ).fetchall()
        items = []
        for bid, resp, reasoning, toks, finish, error, rchars in rows:
            b = behaviors.get(bid, {"text": "", "cat": "uncategorized"})
            rv = reviews.get((arm, bid))
            cl = clas.get((arm, bid))
            items.append({
                "behavior_id": bid, "behavior_text": b["text"], "cat": b["cat"],
                "response": resp or "", "reasoning": reasoning or "",
                "tokens": toks or 0, "finish": finish or "?", "error": error,
                "reasoning_chars": rchars or len(reasoning or ""),
                "judge_verdict": rv["verdict"] if rv else None,
                "judge_reviewer": rv["reviewer"] if rv else None,
                "judge_reasoning": rv["reasoning"] if rv else "",
                "clf": cl,
            })
        arms_data[arm] = items
    base_row = db.execute(
        "SELECT display_name FROM models WHERE comparison=? AND variant_type='base'",
        (comparison,),
    ).fetchone()
    base_display = (base_row[0] if base_row and base_row[0] else "Base")
    return models_map, arms_data, base_display


def arm_stats(items: list[dict]) -> dict:
    """Per-arm judge tallies. Verdict families are rubric-agnostic:
    no-refusal family = no_refusal + legacy complied; refusal family =
    refusal + legacy refused. Four-way arms (2026-09 rubric) additionally
    count soft_refusal and degenerate, and judge ASR uses the determinate
    denominator judged - degenerate."""
    n = len(items)
    fourway = any(
        i["judge_verdict"] in ("no_refusal", "soft_refusal", "refusal", "degenerate")
        for i in items
    )
    j_nr = sum(1 for i in items if i["judge_verdict"] in ("no_refusal", "complied"))
    j_rf = sum(1 for i in items if i["judge_verdict"] in ("refusal", "refused"))
    j_sf = sum(1 for i in items if i["judge_verdict"] == "soft_refusal")
    j_dg = sum(1 for i in items if i["judge_verdict"] == "degenerate")
    j_n = n - j_nr - j_rf - j_sf - j_dg
    empty = sum(1 for i in items if not i["response"].strip())
    trunc = sum(1 for i in items if i["finish"] == "length")
    avg_tok = sum(i["tokens"] for i in items) / n if n else 0
    avg_rc = sum(i["reasoning_chars"] for i in items) / n if n else 0
    judged_n = j_nr + j_rf + j_sf + j_dg
    if fourway:
        det = judged_n - j_dg
        # Float path matches the published report convention: 329/400 renders
        # 82.2, not the Decimal half-up 82.3 (same fix as the graph script's
        # asr_label). Legacy arms keep the original Decimal half-up rendering.
        asr = (float(j_nr) / det * 100.0) if det else None
    else:
        asr = _pct(Decimal(j_nr) / Decimal(n)) if n and judged_n else None
    return {
        "n": n, "j_complied": j_nr, "j_refused": j_rf, "j_soft": j_sf,
        "j_degen": j_dg, "j_none": j_n, "fourway": fourway,
        "judged": judged_n > 0,
        "asr": asr, "empty": empty, "trunc": trunc, "avg_tok": avg_tok, "avg_rc": avg_rc,
    }


def order_arms(arms_data: dict[str, list[dict]]):
    """Default arms by judge ASR desc (unjudged last, by name), then think-mode arms."""
    def key(arm):
        s = arm_stats(arms_data[arm])
        return (-(s["asr"] if s["asr"] is not None else -1.0), arm)

    default = sorted((a for a in arms_data if not is_thinkmode(a)), key=key)
    think = sorted((a for a in arms_data if is_thinkmode(a)), key=key)
    return default, think


def page_skeleton(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title}</title>\n<style>{CSS}</style>\n</head>\n"
        f'<body class="hb-root">\n{body}\n<script>{JS}</script>\n</body>\n</html>\n'
    )


def topnav(comparison: str, site_prefix: str, prev_link: str, next_link: str) -> str:
    pn = []
    if prev_link:
        pn.append(f'<a href="{esc(prev_link)}">&larr; prev arm</a>')
    if next_link:
        pn.append(f'<a href="{esc(next_link)}">next arm &rarr;</a>')
    pn_html = f'<span class="hb-prevnext">{"".join(pn)}</span>' if pn else ""
    return (
        f'<nav class="hb-topnav"><div class="hb-wrap">'
        f'<a href="{esc(site_prefix)}/{esc(comparison)}/">HarmBench explorer</a>'
        f'<span class="hb-crumb">{esc(comparison)}</span>{pn_html}</div></nav>'
    )


def render_card(i: dict) -> str:
    resp, reasoning = i["response"], i["reasoning"]
    empty = not resp.strip()
    jv = i["judge_verdict"]
    vcls = {"complied": "hb-vc", "no_refusal": "hb-vc", "refused": "hb-vr",
            "refusal": "hb-vr", "soft_refusal": "hb-vs",
            "degenerate": "hb-vd"}.get(jv, "hb-vn")
    clf = i["clf"]
    judge_html = ""
    if jv:
        judge_html = (
            f'<div class="hb-judge"><b>LLM-judge ({esc(i["judge_reviewer"])}):</b> '
            f'{esc(i["judge_reasoning"])}</div>'
        )
    error_html = f'<div class="hb-error">runner error: {esc(i["error"])}</div>' if i["error"] else ""
    return f"""<article class="hb-card {vcls}" data-verdict="{esc(jv or 'none')}" data-cat="{esc(i['cat'])}">
  <div class="hb-cardhead">
    <span class="hb-bid">{esc(i['behavior_id'])}</span>
    <span class="hb-chip">{esc(i['cat'])}</span>
    {judge_badge(jv)}
    {clf_badge(clf['label'] if clf else None, clf['confidence'] if clf else None)}
    <span class="hb-meta">{fmt_int(i['tokens'])} tok &middot; finish {esc(i['finish'])} &middot; {fmt_int(i['reasoning_chars'])} reasoning chars</span>
  </div>
  <p class="hb-behavior">{esc(i['behavior_text'])}</p>
  {error_html}
  {judge_html}
  <details class="hb-block">
    <summary>response ({fmt_int(len(resp))} chars{", EMPTY" if empty else ""})</summary>
    <pre class="{'hb-empty' if empty else ''}">{esc(resp) if resp else "(empty response)"}</pre>
  </details>
  <details class="hb-block">
    <summary>reasoning trace ({fmt_int(len(reasoning))} chars, full)</summary>
    <pre class="{'hb-empty' if not reasoning.strip() else ''}">{esc(reasoning) if reasoning else "(no reasoning captured)"}</pre>
  </details>
</article>"""


def static_link(arm: str) -> str:
    return f"{arm}.html"


def hugo_link(site_prefix: str, comparison: str):
    def link(arm: str) -> str:
        return f"{site_prefix}/{comparison}/{arm}/"
    return link


def build_arm_body(comparison: str, arm: str, display: str, items: list[dict],
                   categories: list[str], site_prefix: str, report_url: str,
                   prev_arm: str | None, next_arm: str | None,
                   generated_at: str, link=static_link) -> str:
    s = arm_stats(items)
    asr_big = f"{s['asr']:.1f}%" if s["asr"] is not None else "—"
    if s["fourway"]:
        per_cat = defaultdict(lambda: {"n": 0, "nr": 0, "sf": 0, "rf": 0, "dg": 0})
        for i in items:
            d = per_cat[i["cat"]]
            d["n"] += 1
            jv = i["judge_verdict"]
            if jv in ("no_refusal", "complied"):
                d["nr"] += 1
            elif jv == "soft_refusal":
                d["sf"] += 1
            elif jv in ("refusal", "refused"):
                d["rf"] += 1
            elif jv == "degenerate":
                d["dg"] += 1
        cat_rows = [
            f'<tr><td>{esc(cat)}</td><td class="hb-num">{d["n"]}</td>'
            f'<td class="hb-num">{d["nr"]}</td><td class="hb-num">{d["sf"]}</td>'
            f'<td class="hb-num">{d["rf"]}</td><td class="hb-num">{d["dg"]}</td></tr>'
            for cat in categories if (d := per_cat.get(cat))
        ]
        cat_header = (
            '<table class="hb-table"><thead><tr><th>category</th><th class="hb-num">n</th>\n'
            '      <th class="hb-num">no_refusal</th><th class="hb-num">soft_refusal</th>'
            '<th class="hb-num">refusal</th><th class="hb-num">degenerate</th></tr></thead>'
        )
        verdict_stats = (
            f'<div class="hb-stat"><b>{fmt_int(s["j_complied"])}</b><span>no_refusal (judge)</span></div>\n'
            f'    <div class="hb-stat"><b>{fmt_int(s["j_soft"])}</b><span>soft_refusal (judge)</span></div>\n'
            f'    <div class="hb-stat"><b>{fmt_int(s["j_refused"])}</b><span>refusal (judge)</span></div>\n'
            f'    <div class="hb-stat"><b>{fmt_int(s["j_degen"])}</b><span>degenerate (judge)</span></div>\n'
            f'    <div class="hb-stat"><b>{fmt_int(s["j_none"])}</b><span>no judge</span></div>'
        )
    else:
        per_cat = defaultdict(lambda: {"n": 0, "c": 0, "r": 0, "x": 0})
        for i in items:
            d = per_cat[i["cat"]]
            d["n"] += 1
            if i["judge_verdict"] == "complied":
                d["c"] += 1
            elif i["judge_verdict"] == "refused":
                d["r"] += 1
            else:
                d["x"] += 1
        cat_rows = [
            f'<tr><td>{esc(cat)}</td><td class="hb-num">{d["n"]}</td>'
            f'<td class="hb-num">{d["c"]}</td><td class="hb-num">{d["r"]}</td>'
            f'<td class="hb-num">{d["x"]}</td></tr>'
            for cat in categories if (d := per_cat.get(cat))
        ]
        cat_header = (
            '<table class="hb-table"><thead><tr><th>category</th><th class="hb-num">n</th>\n'
            '      <th class="hb-num">complied</th><th class="hb-num">refused</th><th class="hb-num">no judge</th></tr></thead>'
        )
        verdict_stats = (
            f'<div class="hb-stat"><b>{fmt_int(s["j_complied"])}</b><span>complied (judge)</span></div>\n'
            f'    <div class="hb-stat"><b>{fmt_int(s["j_refused"])}</b><span>refused (judge)</span></div>\n'
            f'    <div class="hb-stat"><b>{fmt_int(s["j_none"])}</b><span>no judge</span></div>'
        )
    verdict_btns = ('<button class="hb-btn hb-on" data-verdict="all" type="button">all</button>\n  '
                    + '\n  '.join(
                        f'<button class="hb-btn" data-verdict="{fam}" type="button">{fam}</button>'
                        for fam, cnt in (
                            (("no_refusal", s["j_complied"]), ("soft_refusal", s["j_soft"]),
                             ("refusal", s["j_refused"]), ("degenerate", s["j_degen"]))
                            if s["fourway"] else
                            (("complied", s["j_complied"]), ("refused", s["j_refused"]))
                        ) if cnt
                    ))
    if s["j_none"]:
        verdict_btns += '\n  <button class="hb-btn" data-verdict="none" type="button">no-judge</button>'
    cat_opts = ['<option value="all">all categories</option>'] + [
        f'<option value="{esc(c)}">{esc(c)}</option>' for c in categories
    ]
    cards = "\n".join(render_card(i) for i in items)
    prev_link = link(prev_arm) if prev_arm else ""
    next_link = link(next_arm) if next_arm else ""
    body = f"""{topnav(comparison, site_prefix, prev_link, next_link)}
<header class="hb-header"><div class="hb-wrap">
  <h1>{esc(display)} <code>{esc(arm)}</code></h1>
  <p class="hb-sub">HarmBench responses for comparison <b>{esc(comparison)}</b> &middot;
    <a href="{esc(site_prefix)}/{esc(comparison)}/" style="color:var(--hb-teal-ink)">explorer index</a> &middot;
    <a href="{esc(report_url)}" style="color:var(--hb-teal-ink)">full report</a></p>
</div></header>
<div class="hb-toolbar"><div class="hb-wrap">
  {verdict_btns}
  <select class="hb-select" id="hb-cat">{''.join(cat_opts)}</select>
  <input class="hb-search" id="hb-search" type="search" placeholder="search behavior id, text, response, reasoning&hellip;" aria-label="search">
  <span class="hb-count" id="hb-count">showing {fmt_int(s['n'])} of {fmt_int(s['n'])}</span>
</div></div>
<div class="hb-wrap">
  <div class="hb-stats">
    <div class="hb-stat hb-hero"><b>{asr_big}</b><span>judge ASR</span></div>
    {verdict_stats}
    <div class="hb-stat"><b>{fmt_int(s['empty'])}</b><span>empty responses</span></div>
    <div class="hb-stat"><b>{fmt_int(s['trunc'])}</b><span>truncated (length)</span></div>
    <div class="hb-stat"><b>{fmt_int(s['avg_tok'])}</b><span>avg completion tokens</span></div>
    <div class="hb-stat"><b>{fmt_int(s['avg_rc'])}</b><span>avg reasoning chars</span></div>
  </div>
  <section class="hb-section"><h2>Category breakdown (judge verdict)</h2>
    {cat_header}
    <tbody>{''.join(cat_rows)}</tbody></table>
  </section>
  <main class="hb-cards">
{cards}
  </main>
  <p class="hb-footer">Generated {esc(generated_at)} from the abliterlitics evaluation database &middot;
    reasoning traces are complete, never truncated &middot; cards render without JavaScript</p>
</div>"""
    return body


def render_arm_page(comparison: str, arm: str, display: str, items: list[dict],
                    categories: list[str], site_prefix: str, report_url: str,
                    prev_arm: str | None, next_arm: str | None,
                    generated_at: str, link=static_link) -> str:
    body = build_arm_body(comparison, arm, display, items, categories, site_prefix,
                          report_url, prev_arm, next_arm, generated_at, link)
    return page_skeleton(f"{display} — HarmBench explorer — {comparison}", body)


def render_arm_fragment(comparison: str, arm: str, display: str, items: list[dict],
                        categories: list[str], site_prefix: str, report_url: str,
                        prev_arm: str | None, next_arm: str | None,
                        generated_at: str, link=static_link) -> str:
    body = build_arm_body(comparison, arm, display, items, categories, site_prefix,
                          report_url, prev_arm, next_arm, generated_at, link)
    return f'<div class="hb-root">\n{body}\n</div>\n'


def build_index_body(comparison: str, models_map: dict[str, str],
                     default: list[str], think: list[str], arms_data: dict[str, list[dict]],
                     reviewers: list[str], site_prefix: str, generated_at: str,
                     base_display: str = "Base", link=static_link,
                     hf_url: str | None = None, report_url: str = "") -> str:
    total_resp = sum(len(v) for v in arms_data.values())
    total_reviews = sum(
        1 for items in arms_data.values() for i in items if i["judge_verdict"]
    )
    fourway_any = any(arm_stats(items)["fourway"] for items in arms_data.values())

    def verdict_cells(s: dict) -> str:
        if fourway_any and s["fourway"]:
            return (
                f'<td class="hb-num">{fmt_int(s["j_complied"])}</td>'
                f'<td class="hb-num">{fmt_int(s["j_soft"])}</td>'
                f'<td class="hb-num">{fmt_int(s["j_refused"])}</td>'
                f'<td class="hb-num">{fmt_int(s["j_degen"])}</td>'
            )
        return (
            f'<td class="hb-num">{fmt_int(s["j_complied"])}</td>'
            f'<td class="hb-num">{fmt_int(s["j_refused"])}</td>'
        )

    verdict_ths = (
        ('<th class="hb-num">no_refusal</th><th class="hb-num">soft_refusal</th>\n'
         '      <th class="hb-num">refusal</th><th class="hb-num">degenerate</th>')
        if fourway_any else
        ('<th class="hb-num">complied</th><th class="hb-num">refused</th>')
    )

    def index_rows(arms: list[str]) -> str:
        rows = []
        for arm in arms:
            items = arms_data[arm]
            s = arm_stats(items)
            display = models_map.get(arm) or fallback_display_name(arm, models_map, base_display)
            asr = f"{s['asr']:.1f}%" if s["asr"] is not None else "—"
            clf_only = (
                ""
                if s["judged"]
                else '<span class="hb-badge hb-j-none">classifier-only</span>'
            )
            rows.append(
                f'<tr><td><a href="{esc(link(arm))}">{esc(display)}</a> '
                f'<code style="font-family:var(--hb-mono);font-size:.75rem;color:var(--hb-ink3)">{esc(arm)}</code></td>'
                f'<td class="hb-num">{asr}</td>{verdict_cells(s)}'
                f'<td class="hb-num">{fmt_int(s["n"])}</td><td>{clf_only}</td></tr>'
            )
        return "".join(rows)

    reviewer_txt = ", ".join(reviewers) if reviewers else "none"
    rubric_txt = (
        "before ruling <i>no_refusal</i>, <i>soft_refusal</i>, <i>refusal</i> or "
        "<i>degenerate</i>, with ASR over determinate items only"
        if fourway_any else
        "before ruling <i>complied</i> or <i>refused</i>"
    )
    explain = (
        f"This explorer exposes every HarmBench response collected for comparison "
        f"<b>{esc(comparison)}</b>: the full model output, the complete reasoning trace "
        f"(never truncated), the refusal-classifier label, and the LLM-judge verdict as "
        f"filterable, searchable cards. The judge ({esc(reviewer_txt)}) reads the full "
        f"reasoning trace plus the final answer {rubric_txt}; "
        f"arms that were never judge-reviewed are marked "
        f"<i>classifier-only</i>. All cards are server-rendered and readable with "
        f"JavaScript disabled — the verdict, category and free-text filters on each arm "
        f"page are progressive enhancement."
    )
    think_note = ""
    if think:
        think_note = (
            '<p class="hb-explain" style="margin:.4rem 0 0"><b>Serving-mode arms</b> rerun the same '
            "400 behaviors with the serving stack rendering thinking-enabled prompts "
            "(<code>enable_thinking=true</code>); see the site write-up for why the rendered "
            "prompt, not the weights, is the stronger behavioral lever.</p>"
        )
    hf_link = (f'<a href="{esc(hf_url)}">Full dataset on HuggingFace</a> · ' if hf_url else "")
    if think:
        think_section = (
            '  <div class="hb-grouplabel">Serving-mode arms, enable_thinking=true</div>\n'
            '  <section class="hb-section">\n'
            f"    {think_note}\n"
            '    <table class="hb-table"><thead><tr><th>arm</th><th class="hb-num">judge ASR</th>\n'
            f'      {verdict_ths}\n'
            '      <th class="hb-num">responses</th><th></th></tr></thead>\n'
            f"    <tbody>{index_rows(think)}</tbody></table>\n"
            '  </section>\n'
        )
    else:
        think_section = ""
    body = f"""{topnav(comparison, site_prefix, "", "")}
<header class="hb-header"><div class="hb-wrap">
  <h1>HarmBench response explorer <code>{esc(comparison)}</code></h1>
  <div class="hb-totals">
    <div><b>{len(arms_data)}</b>arms</div>
    <div><b>{fmt_int(total_resp)}</b>responses</div>
    <div><b>{fmt_int(total_reviews)}</b>judge-reviewed</div>
  </div>
  <p class="hb-links">{hf_link}<a href="{report_url}">Comparison report</a></p>
</div></header>
<div class="hb-wrap">
  <section class="hb-section"><h2>About this explorer</h2><p class="hb-explain">{explain}</p></section>
  <section class="hb-section"><h2>Arms by judge ASR</h2>
    <table class="hb-table"><thead><tr><th>arm</th><th class="hb-num">judge ASR</th>
      {verdict_ths}
      <th class="hb-num">responses</th><th></th></tr></thead>
    <tbody>{index_rows(default)}</tbody></table>
  </section>
{think_section}  <p class="hb-footer">Generated {esc(generated_at)} from the abliterlitics evaluation database &middot;
    judge = {esc(reviewer_txt)}</p>
</div>"""
    return body


def render_index(comparison: str, models_map: dict[str, str],
                 default: list[str], think: list[str], arms_data: dict[str, list[dict]],
                 reviewers: list[str], site_prefix: str, generated_at: str,
                 base_display: str = "Base", link=static_link,
                 hf_url: str | None = None, report_url: str = "") -> str:
    body = build_index_body(comparison, models_map, default, think, arms_data,
                            reviewers, site_prefix, generated_at, base_display, link,
                            hf_url, report_url)
    return page_skeleton(f"HarmBench explorer — {comparison}", body)


def render_index_fragment(comparison: str, models_map: dict[str, str],
                          default: list[str], think: list[str], arms_data: dict[str, list[dict]],
                          reviewers: list[str], site_prefix: str, generated_at: str,
                          base_display: str = "Base", link=static_link,
                          hf_url: str | None = None, report_url: str = "") -> str:
    body = build_index_body(comparison, models_map, default, think, arms_data,
                            reviewers, site_prefix, generated_at, base_display, link,
                            hf_url, report_url)
    return f'<div class="hb-root">\n{body}\n</div>\n'


def yq(s) -> str:
    """Double-quote a string for YAML front matter."""
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def hugo_front_matter(title: str, description: str, layout: str,
                      url: str | None = None, extra: str = "") -> str:
    lines = [
        "---",
        f"title: {yq(title)}",
        f"description: {yq(description)}",
        f"layout: {layout}",
    ]
    if url:
        lines.append(f"url: {yq(url)}")
    if extra:
        lines.append(extra)
    lines.append("---\n")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comparison", required=True)
    ap.add_argument("--db", type=Path, default=Path("abliterlitics.db"))
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="static mode: output dir "
                         "(default: abliterlitics_website/static/harmbench/<comparison>)")
    ap.add_argument("--hugo-content-dir", type=Path, default=None,
                    help="hugo mode: emit content pages (.md + .body.frag) here instead "
                         "of standalone static HTML; cross-links use the pretty-URL form "
                         "/<site-prefix>/<comparison>/<arm>/")
    ap.add_argument("--index-title", default=None,
                    help="hugo mode: title for the _index.md page "
                         "(default: 'HarmBench Response Explorer: <comparison>')")
    ap.add_argument("--site-prefix", default="/harmbench",
                    help="URL path prefix under which the site serves the explorer")
    ap.add_argument("--report-url", default=None,
                    help="cross-link to the comparison report page "
                         "(default: /models/<comparison>/)")
    ap.add_argument("--exclude", default="",
                    help="comma-separated arm names to omit from the explorer "
                         "(e.g. leftover control arms)")
    ap.add_argument("--hf-url", default=None,
                    help="optional HuggingFace dataset URL shown as a link row "
                         "on the index page")
    args = ap.parse_args()

    hugo = args.hugo_content_dir is not None
    out_dir = args.hugo_content_dir if hugo else (
        args.output_dir or Path("abliterlitics_website/static/harmbench") / args.comparison)
    report_url = args.report_url or f"/models/{args.comparison}/"
    site_prefix = args.site_prefix.rstrip("/")
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    link = hugo_link(site_prefix, args.comparison) if hugo else static_link

    db = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    exclude = {a.strip() for a in args.exclude.split(',') if a.strip()}
    models_map, arms_data, base_display = load_data(db, args.comparison, exclude)
    reviewers = [
        re.sub(r"-v\d+$", "", r[0]) for r in db.execute(
            "SELECT DISTINCT reviewer_model FROM llm_reviews WHERE comparison = ?",
            (args.comparison,),
        )
    ]
    db.close()

    if not arms_data:
        print(f"ERROR: no responses found for comparison '{args.comparison}'", file=sys.stderr)
        sys.exit(1)

    categories = sorted({
        i["cat"] for items in arms_data.values() for i in items
    })
    default, think = order_arms(arms_data)
    ordered = default + think

    out_dir.mkdir(parents=True, exist_ok=True)

    total_resp = sum(len(v) for v in arms_data.values())

    if hugo:
        index_title = args.index_title or f"HarmBench Response Explorer: {args.comparison}"
        index_desc = (f"Browse all {fmt_int(total_resp)} HarmBench responses across "
                      f"{len(arms_data)} arms with judge verdicts, classifier labels "
                      "and complete reasoning traces.")
        frag = render_index_fragment(args.comparison, models_map, default, think,
                                     arms_data, reviewers, site_prefix, generated_at,
                                     base_display, link, args.hf_url, report_url)
        frag_path = out_dir / "_index.body.frag"
        frag_path.write_text(frag, encoding="utf-8")
        md_path = out_dir / "_index.md"
        md_path.write_text(
            hugo_front_matter(index_title, index_desc, "harmbench-index",
                              extra="build: { publishResources: false }"),
            encoding="utf-8")
        print(f"wrote {md_path} + {frag_path} ({frag_path.stat().st_size:,} bytes)")
    else:
        index_html = render_index(args.comparison, models_map, default, think,
                                  arms_data, reviewers, site_prefix, generated_at,
                                  base_display, link=static_link, hf_url=args.hf_url,
                                  report_url=report_url)
        (out_dir / "index.html").write_text(index_html, encoding="utf-8")
        print(f"wrote {out_dir/'index.html'} ({(out_dir/'index.html').stat().st_size:,} bytes)")

    for idx, arm in enumerate(ordered):
        items = arms_data[arm]
        display = models_map.get(arm) or fallback_display_name(arm, models_map, base_display)
        prev_arm = ordered[idx - 1] if idx > 0 else None
        next_arm = ordered[idx + 1] if idx < len(ordered) - 1 else None
        s = arm_stats(items)
        asr = f"{s['asr']:.1f}%" if s["asr"] is not None else "clf-only"
        if hugo:
            desc = (f"All {fmt_int(s['n'])} HarmBench responses for {display}, "
                    "with judge verdicts, classifier labels and complete reasoning traces.")
            frag = render_arm_fragment(args.comparison, arm, display, items, categories,
                                       site_prefix, report_url, prev_arm, next_arm,
                                       generated_at, link)
            arm_dir = out_dir / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            frag_path = arm_dir / "body.frag"
            frag_path.write_text(frag, encoding="utf-8")
            md_path = arm_dir / "index.md"
            md_path.write_text(hugo_front_matter(display, desc, "harmbench-arm",
                                                 url=link(arm),
                                                 extra="build: { publishResources: false }"),
                               encoding="utf-8")
            print(f"wrote {md_path} + {frag_path} ({frag_path.stat().st_size:,} bytes, "
                  f"{len(items)} cards, ASR {asr})")
        else:
            page = render_arm_page(args.comparison, arm, display, items, categories,
                                   site_prefix, report_url, prev_arm, next_arm,
                                   generated_at)
            path = out_dir / f"{arm}.html"
            path.write_text(page, encoding="utf-8")
            print(f"wrote {path} ({path.stat().st_size:,} bytes, {len(items)} cards, ASR {asr})")

    mode = "hugo content pages (.md + .body.frag)" if hugo else "standalone HTML pages"
    print(f"done: {len(ordered)} arm pages + index in {out_dir} ({mode})")


if __name__ == "__main__":
    main()
