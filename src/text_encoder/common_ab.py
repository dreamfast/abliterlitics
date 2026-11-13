"""Shared A/B generation scaffolding for the h3/krea2 ComfyUI runners.

run_h3_ab.py and run_krea2_ab.py carried ~150 near-identical lines (node
discovery, suite loading, --labels filtering, arm-list building, and the
generate->fetch->manifest loop). This module is the single home for that
scaffolding; the per-family code (workflow patchers, TE-loader specifics)
stays in each runner behind the ``patch_and_node`` seam.

The manifest contract lives here too: entries are UPSERTED by key, so a
re-run without --skip-existing replaces its prior entry instead of
appending a duplicate (dict-keyed consumers last-win either way, but clip
counts stay honest).
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from src.text_encoder.comfy_client import comfy_te_name, query_vram, queue_and_fetch, sanitize

log = logging.getLogger(__name__)


def find_unique_node(wf: dict[str, Any], class_type: str, input_name: str = "") -> tuple[str, dict[str, Any]]:
    """Find the UNIQUE node of a class_type (fail loud on 0 or >1 matches)."""
    matches = [(nid, n) for nid, n in wf.items() if n.get("class_type") == class_type]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {class_type} node in workflow, found {len(matches)}: {[nid for nid, _ in matches]}"
        )
    nid, node = matches[0]
    if input_name and input_name not in node.get("inputs", {}):
        raise ValueError(f"Node {nid} ({class_type}) lacks input {input_name!r}")
    return nid, node


def load_prompt_suite(suite_path: Path) -> list[dict[str, Any]]:
    """Load prompt entries: {prompts: [{text, category, label?, targets?}]}.

    ``targets`` passes through to the manifest for the downstream scorer
    (h3 audio battery / krea2 word-content checks). Fail loud on missing
    text/category — a typo'd suite must never generate a mistargeted battery.
    """
    raw = yaml.safe_load(suite_path.read_text())
    if isinstance(raw, dict):
        raw = raw.get("prompts", [])
    if not raw:
        raise ValueError(f"No prompts in {suite_path}")
    entries: list[dict[str, Any]] = list(raw)
    for p in entries:
        if "text" not in p or "category" not in p:
            raise ValueError(f"Prompt entry missing text/category: {p}")
    return entries


def filter_suite_labels(suite: list[dict[str, Any]], labels_arg: str) -> list[dict[str, Any]]:
    """Apply the --labels filter: match a prompt's label, or its sanitized
    text when unlabeled. Empty filter = full suite; zero matches = fail loud.
    """
    if not labels_arg:
        return suite
    label_filter = {s.strip() for s in labels_arg.split(",") if s.strip()}
    filtered = [pr for pr in suite if (pr.get("label") or sanitize(pr["text"])) in label_filter]
    if not filtered:
        raise SystemExit(f"--labels filter matched no prompts: {sorted(label_filter)}")
    return filtered


def build_arm_jobs(comparison: dict[str, Any], models_root: str, variants_filter: str) -> list[tuple[str, str]]:
    """A/B arm list: global base first, then variants with a te_file.

    ``variants_filter`` is the raw --variants string ("" = base + all
    variants). ``models_root`` relativizes te_file paths to ComfyUI model
    names; when unset (no --models-root and no $COMFYUI_MODELS_ROOT) an
    absolute te_file is passed through unrelativized with a warning so the
    mismatch surfaces here instead of at ComfyUI load time.
    """
    filter_set = {v.strip() for v in variants_filter.split(",") if v.strip()}
    jobs: list[tuple[str, str]] = []
    base_te = comparison.get("settings", {}).get("tebench", {}).get("base_te_file", "")
    if base_te and (not filter_set or "base" in filter_set):
        jobs.append(("base", comfy_te_name(base_te, models_root)))
    for name, cfg in comparison["variants"].items():
        te_file = cfg.get("te_file")
        if not te_file:
            continue
        if filter_set and name not in filter_set:
            continue
        if not models_root and str(te_file).startswith("/"):
            log.warning("models root unset — passing %r through unrelativized (set COMFYUI_MODELS_ROOT)", te_file)
        jobs.append((name, comfy_te_name(te_file, models_root)))
    if not jobs:
        raise SystemExit("No arms to generate (no base_te_file and no variants with te_file)")
    return jobs


def upsert_manifest_entry(manifest: dict[str, Any], list_key: str, entry: dict[str, Any]) -> bool:
    """Insert entry into manifest[list_key], REPLACING any prior entry with
    the same key. Returns True when an existing entry was replaced.

    A re-run without --skip-existing regenerates clips; appending would grow
    the manifest unboundedly across interrupted reruns and inflate clip
    counts (dict-keyed consumers last-win so metrics stay right — the counts
    were the misleading part).
    """
    entries = manifest.setdefault(list_key, [])
    for i, old in enumerate(entries):
        if old.get("key") == entry["key"]:
            entries[i] = entry
            return True
    entries.append(entry)
    return False


def run_ab_loop(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    out_dir: Path,
    jobs: list[tuple[str, str]],
    suite: list[dict[str, Any]],
    seeds: list[int],
    workflow: dict[str, Any],
    api_url: str,
    skip_existing: bool,
    dry_run: bool,
    list_key: str,
    ext: str,
    patch_and_node: Callable[[dict[str, Any], str, str, str, int], tuple[dict[str, Any], str]],
    extra_entry: dict[str, Any] | None = None,
) -> None:
    """Shared generation loop for the h3/krea2 A/B runners.

    ``patch_and_node(workflow, name, te_name, prompt, seed) -> (patched_wf,
    te_node_id)`` is the only family-specific seam (the runner's patcher +
    its TE-loader lookup). ``list_key`` is the manifest list ("clips" h3 /
    "images" krea2); ``ext`` the output extension. Entries are upserted by
    key (see upsert_manifest_entry) and the manifest is rewritten after
    every clip for crash-resume granularity.
    """
    done: set[str] = {e["key"] for e in manifest.get(list_key, [])}
    by_variant: OrderedDict[str, str] = OrderedDict(jobs)

    for name, te_name in by_variant.items():
        first_clip = True
        for seed in seeds:
            for pr in suite:
                stem = pr.get("label") or sanitize(pr["text"])
                key = f"{name}::{pr.get('label', pr['category'])}::{pr['text']}"
                out_path = out_dir / name / f"{stem}__seed{seed}{ext}"
                if skip_existing and key in done and out_path.exists():
                    continue
                if dry_run:
                    log.info("[dry-run] %s seed%d :: %s -> %s (te=%s)", name, seed, pr["category"], out_path, te_name)
                    continue
                wf, te_node_id = patch_and_node(workflow, name, te_name, pr["text"], seed)
                data, perf = queue_and_fetch(api_url, wf, te_node_id=te_node_id)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(data)
                vram_cap = query_vram(api_url) if first_clip else None
                first_clip = False
                log.info(
                    "%s seed%d | %s | %s (%d KB, gen %sms)",
                    name,
                    seed,
                    pr["category"],
                    out_path.name,
                    len(data) // 1024,
                    perf.get("server_gen_ms"),
                )
                entry: dict[str, Any] = {
                    "key": key,
                    "variant": name,
                    "te_name": te_name,
                    "gen_seconds": round(perf.get("client_wall_ms", 0) / 1000, 2),
                    "perf": perf,
                    "vram": vram_cap,
                    "category": pr["category"],
                    "label": pr.get("label", ""),
                    "prompt": pr["text"],
                    "targets": pr.get("targets", []),
                    "path": str(out_path),
                    "seed": seed,
                }
                if extra_entry:
                    entry.update(extra_entry)
                if upsert_manifest_entry(manifest, list_key, entry):
                    log.info("replaced prior manifest entry for %s (rerun)", key)
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(json.dumps(manifest, indent=2))
