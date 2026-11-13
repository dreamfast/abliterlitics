#!/usr/bin/env python3
"""GSM8K three-number report (AGENTS.md lessons #20/#32).

Post-processes an lm-eval Phase 2 GSM8K samples JSONL into the 3-number report:
  1. Headline exact_match (flexible-extract, denominator = ALL problems;
     empties count as wrong) — leaderboard-comparable.
  2. Empty count + pct — reasoning-failure rate (thinking-loop proxy).
  3. Answered-only exact_match (denominator = non-empty) — capability ceiling
     when reasoning converges.
The headline-to-answered-only delta is the "thinking-tax".

Also bridges the samples file into __model/ under the explicit
``<model>__samples_gsm8k_<ts>.jsonl`` name when it is not already there
(lesson #51: the importer's priority-1 lookup; creation inside __model/ must
use a RELATIVE path — absolute-path creation in that dir hits ENOENT).

Usage:
  python3 scripts/gsm8k_three_number_report.py \
      --comparison-dir comparisons/qwen38-27b \
      --model huihui \
      --samples comparisons/qwen38-27b/results/lm_eval/__model/samples_gsm8k_<ts>.jsonl \
      [--baseline Qwen3.8-27B]

Baseline adds delta_vs_base fields (headline + answered-only, in pp) using the
existing gsm8k_3num_<baseline>.json report.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def compute(samples_path: Path) -> dict:
    """Compute the 3-number report from one samples JSONL (flexible-extract rows)."""
    per_doc: dict[str, dict] = {}
    total_rows = 0
    with samples_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("filter") != "flexible-extract":
                continue
            total_rows += 1
            doc_id = row.get("doc_id")
            resp = ""
            resps = row.get("resps") or [[]]
            if resps and resps[0]:
                resp = resps[0][0] or ""
            per_doc[doc_id] = {
                "empty": (not resp.strip()),
                "correct": float(row.get("exact_match", 0.0)),
                "resp": resp,
            }

    total = len(per_doc)
    if total == 0:
        raise SystemExit(f"ERROR: no flexible-extract rows found in {samples_path} (rows scanned: {total_rows})")
    empties = [d for d in per_doc.values() if d["empty"]]
    empty_count = len(empties)
    n_answered = total - empty_count
    correct_all = sum(1 for d in per_doc.values() if d["correct"] > 0)
    correct_answered = sum(1 for d in per_doc.values() if not d["empty"] and d["correct"] > 0)

    headline = correct_all / total
    answered_only = (correct_answered / n_answered) if n_answered else 0.0
    return {
        "total_problems": total,
        "empty_count": empty_count,
        "empty_pct": round(100.0 * empty_count / total, 2),
        "headline_exact_match": round(headline, 4),
        "answered_only_exact_match": round(answered_only, 4),
        "thinking_tax_pp": round(100.0 * (headline - answered_only), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comparison-dir", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--samples", required=True, type=Path)
    ap.add_argument("--baseline", default=None, help="Model name of an existing gsm8k_3num_<baseline>.json for delta columns")
    args = ap.parse_args()

    results_dir = args.comparison_dir / "results" / "lm_eval"
    model_dir = results_dir / "__model"
    report = compute(args.samples)

    # Bridge into __model/ under the explicit <model>__ name (idempotent).
    # Lesson #51: creation inside __model/ MUST use a relative path (absolute
    # path creation in that dir returns ENOENT — unrepaired fs anomaly). The
    # dir is root-owned (files written by the lm-eval container), so host-side
    # shutil falls back to a docker alpine copy (root inside container).
    bridged = None
    needs_bridge = args.samples.parent != model_dir or not args.samples.name.startswith(f"{args.model}__")
    if needs_bridge:
        dest_name = f"{args.model}__{args.samples.name}"
        dest = model_dir / dest_name
        if not dest.exists():
            import os
            import subprocess

            repo_root = Path(__file__).resolve().parent.parent
            rel_model_dir = model_dir.resolve().relative_to(repo_root)
            rel_src = args.samples.resolve().relative_to(repo_root)
            docker_cmd = [
                "docker", "run", "--rm", "-v", f"{repo_root}:/w", "alpine",
                "sh", "-c", f"cd /w/{rel_model_dir} && cp /w/{rel_src} ./{dest_name}",
            ]
            try:
                model_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(os.path.relpath(args.samples.resolve(), model_dir), dest_name)
            except (PermissionError, OSError):
                subprocess.run(docker_cmd, check=True)
        bridged = dest
        print(f"[bridge] {args.samples} -> {dest}")

    # Deltas vs baseline report.
    if args.baseline:
        base_file = results_dir / f"gsm8k_3num_{args.baseline}.json"
        if base_file.exists():
            base = json.loads(base_file.read_text())
            report["delta_vs_base"] = {
                "headline_pp": round(100.0 * (report["headline_exact_match"] - base["headline_exact_match"]), 2),
                "answered_only_pp": round(100.0 * (report["answered_only_exact_match"] - base["answered_only_exact_match"]), 2),
                "empty_pct_pp": round(report["empty_pct"] - base["empty_pct"], 2),
            }
        else:
            print(f"[warn] baseline report missing: {base_file}", file=sys.stderr)

    out = {
        "model": args.model,
        "samples_file": str(args.samples.resolve()),
        **report,
    }
    out_path = results_dir / f"gsm8k_3num_{args.model}.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"[written] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
