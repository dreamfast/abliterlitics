"""Orchestration tests for run_h3_audio.run_battery — the Transcriber seam.

A fake transcriber exercises: scoring loop, calibration exclusion, per-arm
aggregation, floors A/B, H9 jaccard fields, smoke passthrough — no Whisper,
no real audio.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.text_encoder.run_h3_audio import run_battery, tier_map


class _FakeTranscriber:
    """Returns scripted text per (variant, label): heretic swallows a syllable."""

    def __init__(self, scripts: dict[str, str]) -> None:
        self.scripts = scripts
        self.calls: list[Path] = []

    def transcribe(self, path: Path) -> str:
        self.calls.append(path)
        key = f"{path.parent.name}/{path.stem}"
        return self.scripts[key]


def _manifest_clips(tmp_path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    entries: dict[str, dict[str, Any]] = {
        "line_a": {
            "label": "line_a",
            "category": "speech",
            "text": "prompt text <d>[English] utterly inconceivable truly inconceivable</d>",
            "script": "utterly inconceivable truly inconceivable",
            "targets": ["inconceivable"],
        },
        "line_b": {
            "label": "line_b",
            "category": "speech",
            "text": "prompt text <d>[English] the red cat</d>",
            "script": "the red cat",
            "targets": ["red", "cat"],
        },
    }
    clips: list[dict[str, Any]] = []
    paths: dict[str, str] = {}
    for variant in ("base", "heretic"):
        for seed in (42, 43):
            for label in ("line_a", "line_b"):
                f = tmp_path / variant / f"{label}__seed{seed}.mp4"
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(b"fake")
                clips.append({"variant": variant, "label": label, "seed": seed, "path": str(f)})
                # fake key: variant/label -> transcript
                paths[f"{variant}/{label}__seed{seed}"] = f"{variant}/{label}__seed{seed}"
    # transcripts: base always correct; heretic swallows in line_a only
    scripts: dict[str, str] = {}
    for variant in ("base", "heretic"):
        for seed in (42, 43):
            line_a = (
                "utterly inconceivable truly inconceivable"
                if variant == "base"
                else "utterly inconceivable truly inconcable"
            )
            scripts[f"{variant}/line_a__seed{seed}"] = line_a
            scripts[f"{variant}/line_b__seed{seed}"] = "the red cat"
    return clips, entries, scripts


def test_run_battery_end_to_end_fake(tmp_path: Path) -> None:
    clips, entries, scripts = _manifest_clips(tmp_path)
    fake = _FakeTranscriber(scripts)
    manifest = {"smoke": True}
    result = run_battery(clips, entries, tier_map(), fake, manifest)

    assert result["smoke"] is True
    assert result["n_clips"] == 8
    assert result["calibration_excluded"] == []  # base renders everything

    base = result["arms"]["base"]
    heretic = result["arms"]["heretic"]
    rare_base = base["tiers"]["rare_polysyllabic"]
    rare_heretic = heretic["tiers"]["rare_polysyllabic"]
    assert rare_base["correct"] == 4  # 2 occurrences x 2 seeds
    assert rare_heretic["correct"] == 2  # first occurrence survives
    assert rare_heretic["mispronounced"] == 2  # second swallowed both seeds
    assert rare_heretic["error_rate"] == 0.5

    # H9: heretic fails the SAME word across both seeds -> 1.0; base clean -> None
    assert heretic["cross_seed_jaccard_h9"] == 1.0
    assert base["cross_seed_jaccard_h9"] is None
    # descriptive jaccard keeps the both-clean convention
    assert base["cross_seed_jaccard"] == 1.0

    # Floor A present; Floor B absent (no base_* stand-in arm)
    assert result["floors"]["a_whisper_intrinsic"] is not None
    assert result["floors"]["b_quant_band"] is None

    # json-serializable
    json.dumps(result)


def test_run_battery_calibration_excludes_never_correct(tmp_path: Path) -> None:
    clips, entries, scripts = _manifest_clips(tmp_path)
    # base can never render 'cat' correctly
    for k in list(scripts):
        if k.startswith("base/line_b"):
            scripts[k] = "the red hat"
    fake = _FakeTranscriber(scripts)
    result = run_battery(clips, entries, tier_map(), fake, {})
    assert "cat" in result["calibration_excluded"]
    # 'cat' never counted against heretic: the H9 failure set stays clean of it
    assert result["arms"]["heretic"]["cross_seed_jaccard_h9"] == 1.0


def test_run_battery_no_base_arm_warns_and_disables_calibration(tmp_path: Path, caplog: Any) -> None:
    import logging

    clips, entries, scripts = _manifest_clips(tmp_path)
    clips = [c for c in clips if c["variant"] != "base"]
    fake = _FakeTranscriber(scripts)
    with caplog.at_level(logging.WARNING):
        result = run_battery(clips, entries, tier_map(), fake, {})
    assert "NO BASE ARM" in caplog.text
    assert result["floors"]["a_whisper_intrinsic"] is None


class _FlakyTranscriber:
    """Transcribes per scripts but raises on ONE clip (corrupt mp4 sim)."""

    def __init__(self, scripts: dict[str, str], fail_key: str) -> None:
        self.inner = _FakeTranscriber(scripts)
        self.fail_key = fail_key

    def transcribe(self, path: Path) -> str:
        key = f"{path.parent.name}/{path.stem}"
        if key == self.fail_key:
            raise ValueError("corrupt mp4: no audio decoded")
        return self.inner.transcribe(path)


def test_run_battery_isolates_one_bad_clip(tmp_path: Path, caplog: Any) -> None:
    """One corrupt/undecodable clip skips with a warning; the battery completes.

    Regression for the audit crash fix (run_h3_audio run_battery): a raising
    transcriber used to abort the whole multi-hour battery mid-run.
    """
    import logging

    clips, entries, scripts = _manifest_clips(tmp_path)
    fake = _FlakyTranscriber(scripts, "heretic/line_a__seed42")
    with caplog.at_level(logging.WARNING):
        result = run_battery(clips, entries, tier_map(), fake, {})

    assert "transcribe failed" in caplog.text
    assert result["n_clips"] == 7  # 8 minus the skipped clip
    assert len(result["clips"]) == 7
    # the bad clip is absent, everything else scored
    scored = {(c["variant"], c["label"], c["seed"]) for c in result["clips"]}
    assert ("heretic", "line_a", 42) not in scored
    assert ("heretic", "line_a", 43) in scored
    # battery still completed for both arms with base intact
    assert set(result["arms"]) == {"base", "heretic"}
    assert result["arms"]["base"]["tiers"]["rare_polysyllabic"]["correct"] == 4
    heretic_rare = result["arms"]["heretic"]["tiers"]["rare_polysyllabic"]
    assert heretic_rare["correct"] == 1  # surviving seed only
    assert heretic_rare["mispronounced"] == 1
    json.dumps(result)
