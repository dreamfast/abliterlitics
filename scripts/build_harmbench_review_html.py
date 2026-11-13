#!/usr/bin/env python3
"""Build a self-contained HarmBench data-appendix HTML from the abliterlitics DB.

For each requested model: every response, full reasoning trace, classifier label,
and LLM-judge verdict + reasoning, as collapsible per-item cards. Optionally
injects the appendix into an existing report page before </body> (idempotent).

Usage:
  python3 scripts/build_harmbench_review_html.py \
      --comparison gemma4-12b \
      --models "gemma-4-12B-it,base_thinkmode,sdft_cybersec_r2,cybersec_thinkmode,
               sdft_smoothed_r2_new,smoothed_thinkmode_v2" \
      --db abliterlitics.db \
      --output /tmp/appendix_fragment.html \
      --inject-into comparisons/gemma4-12b/report/sdft_lora_report.html

Design notes:
- Self-contained: no external requests, no JS libs. Works with JS disabled
  (vanilla JS filter is progressive enhancement only).
- Scoped CSS (.hbd- prefix) so injection cannot fight the host page's styles.
- Full reasoning traces are included (NOT truncated) per AGENTS.md lesson #38.
- Idempotent injection via <!-- HBD-APPENDIX --> ... <!-- /HBD-APPENDIX --> markers.
"""

from __future__ import annotations

import argparse
import html
import sqlite3
import sys
from pathlib import Path
from typing import Any

APPENDIX_OPEN = "<!-- HBD-APPENDIX -->"
APPENDIX_CLOSE = "<!-- /HBD-APPENDIX -->"

# Judge verdict vocabulary: 4-way 2026-09 rubric, with legacy complied/refused
# folded into the matching families (mirrors build_harmbench_explorer.py).
NO_REFUSAL_FAMILY = ("no_refusal", "complied")
REFUSAL_FAMILY = ("refusal", "refused")

VERDICT_BADGE_CLASS = {
    "complied": "hbd-complied",
    "no_refusal": "hbd-complied",
    "refused": "hbd-refused",
    "refusal": "hbd-refused",
    "soft_refusal": "hbd-soft",
    "degenerate": "hbd-degen",
}
VERDICT_CARD_CLASS = {
    "complied": "hbd-card-complied",
    "no_refusal": "hbd-card-complied",
    "refused": "hbd-card-refused",
    "refusal": "hbd-card-refused",
    "soft_refusal": "hbd-card-soft",
    "degenerate": "hbd-card-degen",
}

MODEL_LABELS = {
    "gemma-4-12B-it": "base — default render",
    "base_thinkmode": "base — think render (enable_thinking=true)",
    "sdft_cybersec_r2": "sdft_cybersec_r2 — default render",
    "cybersec_thinkmode": "sdft_cybersec_r2 — think render",
    "sdft_smoothed_r2_new": "sdft_smoothed_r2_new — default render",
    "smoothed_thinkmode_v2": "sdft_smoothed_r2_new — think render",
}


def esc(s: str | None) -> str:
    return html.escape(s or "")


def fetch_items(db: sqlite3.Connection, comparison: str, model: str) -> list[dict[str, Any]]:
    rows = db.execute(
        """
        SELECT r.behavior_id,
               b.behavior_text,
               b.semantic_category,
               r.response_text,
               r.reasoning_text,
               r.completion_tokens,
               r.finish_reason,
               c.label, c.reason, c.confidence, c.classifier
        FROM responses r
        JOIN behaviors b ON r.behavior_id = b.behavior_id
        LEFT JOIN v_latest_classification c
            ON r.comparison = c.comparison
            AND r.model_name = c.model_name
            AND r.behavior_id = c.behavior_id
        WHERE r.comparison = ? AND r.model_name = ?
        ORDER BY r.behavior_id
        """,
        (comparison, model),
    ).fetchall()
    # Latest review per behavior wins (lesson #76: the quarantine + re-review
    # flow can leave >1 reviewer row per key; a raw LEFT JOIN would
    # row-multiply the cards and inflate the judge stats).
    reviews: dict[str, tuple[str | None, str | None, str | None]] = {}
    for bid, verdict, reviewer, reasoning in db.execute(
        """
        SELECT behavior_id, verdict, reviewer_model, review_reasoning
        FROM llm_reviews
        WHERE comparison = ? AND model_name = ?
        ORDER BY reviewed_at
        """,
        (comparison, model),
    ):
        reviews[bid] = (verdict, reviewer, reasoning)
    cols = [
        "behavior_id",
        "behavior_text",
        "semantic_category",
        "response_text",
        "reasoning_text",
        "completion_tokens",
        "finish_reason",
        "classifier_label",
        "classifier_reason",
        "classifier_confidence",
        "classifier_name",
    ]
    items = []
    for r in rows:
        item: dict[str, Any] = dict(zip(cols, r, strict=True))
        verdict, reviewer, reasoning = reviews.get(r[0], (None, None, None))
        item["judge_verdict"] = verdict
        item["judge_reviewer"] = reviewer
        item["judge_reasoning"] = reasoning
        items.append(item)
    return items


def badge(verdict: str | None, kind: str) -> str:
    if not verdict:
        return f'<span class="hbd-badge hbd-na">no {kind}</span>'
    return f'<span class="hbd-badge {VERDICT_BADGE_CLASS.get(verdict, "hbd-na")}">{esc(verdict)}</span>'


def model_section(model: str, items: list[dict[str, Any]]) -> str:
    n = len(items)
    j_nr = sum(1 for i in items if i["judge_verdict"] in NO_REFUSAL_FAMILY)
    j_sf = sum(1 for i in items if i["judge_verdict"] == "soft_refusal")
    j_rf = sum(1 for i in items if i["judge_verdict"] in REFUSAL_FAMILY)
    j_dg = sum(1 for i in items if i["judge_verdict"] == "degenerate")
    j_na = n - j_nr - j_sf - j_rf - j_dg
    det = j_nr + j_sf + j_rf  # determinate denominator: judged minus degenerate
    asr = f"{100 * j_nr / det:.1f}% ASR" if det else "no determinate verdicts"
    c_ref = sum(1 for i in items if (i["classifier_label"] or "").startswith("refus"))
    label = MODEL_LABELS.get(model, model)

    # category stats (4-way)
    cats: dict[str, dict[str, int]] = {}
    for i in items:
        c = cats.setdefault(
            i["semantic_category"] or "uncategorized",
            {"n": 0, "nr": 0, "sf": 0, "rf": 0, "dg": 0},
        )
        c["n"] += 1
        v = i["judge_verdict"]
        if v in NO_REFUSAL_FAMILY:
            c["nr"] += 1
        elif v == "soft_refusal":
            c["sf"] += 1
        elif v in REFUSAL_FAMILY:
            c["rf"] += 1
        elif v == "degenerate":
            c["dg"] += 1

    cat_rows = "".join(
        f'<tr><td>{esc(cat)}</td><td class="hbd-num">{d["n"]}</td>'
        f'<td class="hbd-num">{d["nr"]}</td><td class="hbd-num">{d["sf"]}</td>'
        f'<td class="hbd-num">{d["rf"]}</td><td class="hbd-num">{d["dg"]}</td>'
        f'<td class="hbd-num">{d["n"] - d["nr"] - d["sf"] - d["rf"] - d["dg"]}</td></tr>'
        for cat, d in sorted(cats.items())
    )

    cards = []
    for i in items:
        resp = i["response_text"] or ""
        reasoning = i["reasoning_text"] or ""
        empty = not resp.strip()
        jv = i["judge_verdict"]
        cls = VERDICT_CARD_CLASS.get(jv, "")
        clf_v = "refused" if (i["classifier_label"] or "").startswith("refus") else "complied"
        clf_badge_html = badge(clf_v, "classifier") if i["classifier_label"] else badge(None, "classifier")
        judge_div = (
            f'<div class="hbd-judge"><b>LLM-judge ({esc(i["judge_reviewer"])}):</b> {esc(i["judge_reasoning"])}</div>'
            if jv
            else ""
        )
        clf_div = (
            f'<div class="hbd-clf"><b>Classifier ({esc(i["classifier_name"] or "?")}, '
            f"conf={esc(i['classifier_confidence'] or '?')}):</b> "
            f"{esc(i['classifier_reason'] or '')}</div>"
            if i["classifier_label"]
            else ""
        )
        cards.append(f"""
<div class="hbd-card {cls}" data-verdict="{esc(jv or "none")}" data-cat="{esc(i["semantic_category"] or "")}">
  <div class="hbd-card-head">
    <span class="hbd-bid">{esc(i["behavior_id"])}</span>
    <span class="hbd-cat">{esc(i["semantic_category"] or "")}</span>
    {clf_badge_html}
    {badge(jv, "judge")}
  </div>
  <div class="hbd-behavior">{esc(i["behavior_text"])}</div>
  <details class="hbd-block">
    <summary>response ({len(resp):,} chars{", EMPTY" if empty else ""},
    finish={esc(i["finish_reason"] or "?")}, {i["completion_tokens"] or 0:,} tok)</summary>
    <pre>{esc(resp) if resp else "(empty response)"}</pre>
  </details>
  <details class="hbd-block">
    <summary>reasoning trace ({len(reasoning):,} chars, full)</summary>
    <pre>{esc(reasoning) if reasoning else "(no reasoning captured)"}</pre>
  </details>
  {judge_div}
  {clf_div}
</div>""")
    cards_html = "\n".join(cards)

    return f"""
<section class="hbd-model" id="hbd-{esc(model).replace("/", "_").replace(" ", "_")}">
  <h3>{esc(label)} <code class="hbd-mcode">{esc(model)}</code></h3>
  <div class="hbd-stats">
    <div><b>{n}</b> behaviors</div>
    <div><b>{j_nr}</b> no_refusal ({asr})</div>
    <div><b>{j_sf}</b> soft_refusal</div>
    <div><b>{j_rf}</b> refusal</div>
    <div><b>{j_dg}</b> degenerate</div>
    <div><b>{j_na}</b> not judge-reviewed</div>
    <div><b>{c_ref}</b> classifier-refusals</div>
  </div>
  <details class="hbd-cats"><summary>category breakdown (judge verdict)</summary>
    <table class="hbd-table"><thead><tr><th>category</th><th>n</th><th>no_refusal</th>
    <th>soft_refusal</th><th>refusal</th><th>degenerate</th><th>no judge</th></tr></thead>
    <tbody>{cat_rows}</tbody></table>
  </details>
  <div class="hbd-cards">{cards_html}</div>
</section>"""


def build_fragment(comparison: str, models: list[str], db_path: Path) -> str:
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    sections = []
    for m in models:
        items = fetch_items(db, comparison, m)
        if not items:
            print(f"WARNING: no responses for model '{m}' in '{comparison}'", file=sys.stderr)
            continue
        print(f"  {m}: {len(items)} items", file=sys.stderr)
        sections.append(model_section(m, items))
    db.close()

    total = " + ".join(MODEL_LABELS.get(m, m) for m in models)
    nav_html = "".join(
        f'<a href="#hbd-{esc(m).replace(chr(47), chr(95)).replace(chr(32), chr(95))}">{esc(MODEL_LABELS.get(m, m))}</a>'
        for m in models
    )
    return f"""
{APPENDIX_OPEN}
<section id="harmbench-full-data" class="hbd-root">
  <h2>HarmBench full data appendix</h2>
  <p>Every response, full reasoning trace, classifier label and LLM-judge verdict, read
  directly from the evaluation database (comparison <code>{esc(comparison)}</code>).
  Reasoning traces are complete, not truncated. {esc(total)}.</p>
  <p class="hbd-note">Filter (optional, JavaScript):
    <button onclick="hbdFilter('all')">all</button>
    <button onclick="hbdFilter('no_refusal')">judge-no_refusal</button>
    <button onclick="hbdFilter('soft_refusal')">judge-soft_refusal</button>
    <button onclick="hbdFilter('refusal')">judge-refusal</button>
    <button onclick="hbdFilter('degenerate')">judge-degenerate</button>
  </p>
  <nav class="hbd-nav">{nav_html}</nav>
  {"".join(sections)}
</section>
<style>
.hbd-root {{ font-family: system-ui, sans-serif; font-size: 14px; color: #1a1a2e; }}
.hbd-root h2 {{ border-bottom: 2px solid #ccc; padding-bottom: .3em; }}
.hbd-model {{ margin: 2.5em 0; }}
.hbd-mcode {{ font-size: .8em; background: #f0f0f5; padding: .1em .4em; border-radius: 3px; }}
.hbd-stats {{ display: flex; gap: 1.5em; flex-wrap: wrap; background: #f7f7fa;
  border: 1px solid #e2e2ea; border-radius: 6px; padding: .6em 1em; margin: .8em 0; }}
.hbd-stats b {{ font-size: 1.15em; display: block; }}
.hbd-table {{ border-collapse: collapse; margin: .5em 0; }}
.hbd-table th, .hbd-table td {{ border: 1px solid #ddd; padding: .25em .6em; text-align: left; }}
.hbd-num {{ text-align: right; }}
.hbd-cards {{ margin-top: 1em; }}
.hbd-card {{ border: 1px solid #ddd; border-radius: 6px; margin: .6em 0; padding: .6em .9em; background: #fff; }}
.hbd-card-complied {{ border-left: 4px solid #b33; }}
.hbd-card-refused {{ border-left: 4px solid #2a2; }}
.hbd-card-soft {{ border-left: 4px solid #d97706; }}
.hbd-card-degen {{ border-left: 4px solid #64748b; }}
.hbd-card-head {{ display: flex; gap: .8em; align-items: center; flex-wrap: wrap; }}
.hbd-bid {{ font-family: ui-monospace, monospace; font-weight: 600; }}
.hbd-cat {{ font-size: .8em; color: #666; background: #f0f0f5; padding: .1em .5em; border-radius: 10px; }}
.hbd-badge {{ font-size: .75em; font-weight: 700; padding: .15em .6em;
  border-radius: 10px; text-transform: uppercase; }}
.hbd-complied {{ background: #fde8e8; color: #a00; }}
.hbd-refused {{ background: #e6f6e6; color: #070; }}
.hbd-soft {{ background: #fff3d6; color: #8a6d00; }}
.hbd-degen {{ background: #ece9f7; color: #5b4b9a; }}
.hbd-na {{ background: #eee; color: #888; }}
.hbd-behavior {{ color: #444; margin: .3em 0; }}
.hbd-block {{ margin: .35em 0; }}
.hbd-block summary {{ cursor: pointer; color: #0055aa; font-size: .85em; }}
.hbd-block pre {{ white-space: pre-wrap; word-break: break-word; background: #fafafc;
  border: 1px solid #eee; border-radius: 4px; padding: .6em; max-height: 24em;
  overflow-y: auto; font-size: .8em; }}
.hbd-judge, .hbd-clf {{ font-size: .8em; margin: .3em 0; color: #333; }}
.hbd-nav a {{ margin-right: 1em; }}
.hbd-note {{ font-size: .8em; color: #777; }}
.hbd-note button {{ cursor: pointer; }}
</style>
<script>
function hbdFilter(v) {{
  document.querySelectorAll('.hbd-card').forEach(function (c) {{
    c.style.display = (v === 'all' || c.dataset.verdict === v) ? '' : 'none';
  }});
}}
</script>
{APPENDIX_CLOSE}"""


def inject(host_path: Path, fragment: str) -> None:
    content = host_path.read_text(encoding="utf-8")
    if APPENDIX_OPEN in content and APPENDIX_CLOSE in content:
        start = content.index(APPENDIX_OPEN)
        end = content.index(APPENDIX_CLOSE) + len(APPENDIX_CLOSE)
        content = content[:start] + fragment.strip() + content[end:]
        note = "replaced existing appendix"
    elif APPENDIX_OPEN in content:
        # OPEN marker without CLOSE = truncated host file; index(APPENDIX_CLOSE)
        # would raise ValueError. Replace from OPEN to EOF instead.
        start = content.index(APPENDIX_OPEN)
        content = content[:start] + fragment.strip() + "\n"
        note = "replaced truncated appendix (missing close marker)"
    elif "</body>" in content:
        content = content.replace("</body>", fragment + "\n</body>", 1)
        note = "injected before </body>"
    else:
        content += "\n" + fragment + "\n"
        note = "appended at end (no </body> found)"
    host_path.write_text(content, encoding="utf-8")
    print(f"Injected into {host_path} ({note})", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", required=True)
    p.add_argument("--models", required=True, help="comma-separated model_name list")
    p.add_argument("--db", type=Path, default=Path("abliterlitics.db"))
    p.add_argument("--output", type=Path, help="write fragment to this path (also)")
    p.add_argument("--inject-into", type=Path, help="inject appendix into this HTML file")
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    fragment = build_fragment(args.comparison, models, args.db)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(fragment, encoding="utf-8")
        print(
            f"Fragment written: {args.output} ({args.output.stat().st_size:,} bytes)",
            file=sys.stderr,
        )
    if args.inject_into:
        inject(args.inject_into, fragment)
    if not args.output and not args.inject_into:
        sys.stdout.write(fragment)


if __name__ == "__main__":
    main()
