"""H3 audio battery runner — the headline test (p-e-w's "inconceivable").

Reads the A/B video manifest (run_h3_ab.py output — clips carry muxed
stereo audio), transcribes each clip with Whisper (transformers pipeline;
audio decoded via PyAV — no soundfile/librosa in the image), scores every
target word against the scripted line (correct / mispronounced / dropped +
syllable fidelity), aggregates per tier, computes the two floors, and the
cross-seed failure overlap (H9 structured-edit signal).

Calibration gate (runs FIRST, fails the battery):
  every target word must be rendered correctly (>= 0.9 match) in at least
  ONE base-arm clip — otherwise the battery is blind to that word (either
  the model never says it or whisper never hears it) and the word is
  excluded + disclosed rather than counted against any arm.

Floors:
  Floor A (whisper intrinsic): base-arm target error rate per tier.
  Floor B (quant band): |error_rate(base) - error_rate(stand-in arm)| per
  tier — only meaningful when a second base arm exists (smoke stand-in).

Output: tebench_h3_audio.json (+ "smoke": true passthrough from manifest).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from src.text_encoder import TE_RESULTS_VERSION
from src.text_encoder.h3_audio import (
    aggregate_arm,
    calibration_excluded,
    cross_seed_jaccard,
    cross_seed_jaccard_h9,
    score_line,
)

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# tier membership: rare_polysyllabic (the p-e-w tier), common_polysyllabic,
# monosyllabic control. Unknown targets -> "unknown" (disclosed).
RARE = {"inconceivable", "entrepreneurial", "juxtaposition", "extraordinarily", "idiosyncratic", "susceptibility"}
COMMON = {"beautiful", "tomorrow", "remember", "comfortable", "important", "together"}
MONO = {"red", "cat", "bridge", "moon", "box", "steel"}


def tier_map() -> dict[str, str]:
    m: dict[str, str] = {}
    for w in RARE:
        m[w] = "rare_polysyllabic"
    for w in COMMON:
        m[w] = "common_polysyllabic"
    for w in MONO:
        m[w] = "monosyllabic"
    return m


def decode_audio_mono(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode the first audio stream to mono float32 at sample_rate via PyAV.

    Uses av.audio.resampler.AudioResampler for BOTH the channel downmix and
    the rate conversion — band-limited (no aliasing at 32k/44.1k/48k->16k,
    which naive linear interpolation is not) and layout-aware (no
    planar-vs-packed shape guessing).
    """
    import av
    from av.audio.resampler import AudioResampler

    resampler = AudioResampler(format="flt", layout="mono", rate=sample_rate)
    chunks: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            for rf in resampler.resample(frame):
                arr = rf.to_ndarray()  # mono flt: [1, n]
                chunks.append(arr.reshape(-1).astype(np.float32))
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"No audio decoded from {path}")
    peak = float(np.abs(audio).max()) or 1.0
    return (audio / peak).astype(np.float32)


class Transcriber(Protocol):
    """Test seam: anything that turns an audio path into text."""

    def transcribe(self, path: Path) -> str: ...


class WhisperTranscriber:
    """Whisper via transformers pipeline (model id from comparison settings)."""

    def __init__(self, model_id: str, device: str = "cuda") -> None:
        from transformers import pipeline as hf_pipeline

        self.model_id = model_id
        self._pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            torch_dtype="float16" if device.startswith("cuda") else None,
        )

    def transcribe(self, path: Path) -> str:
        audio = decode_audio_mono(path)
        out = self._pipe({"raw": audio, "sampling_rate": SAMPLE_RATE}, generate_kwargs={"language": "en"})
        text: str = out["text"] if isinstance(out, dict) else str(out)
        return text.strip()


def extract_script_targets(entry: dict[str, Any]) -> tuple[str, list[str]]:
    """Script + targets for a suite entry; validate the <d> script matches."""
    script = entry.get("script", "")
    targets = [str(t) for t in entry.get("targets", [])]
    if not script or not targets:
        raise ValueError(f"Audio entry lacks script/targets: label={entry.get('label', '?')}")
    m = re.search(r"<d>\[English\]\s*(.*?)</d>", entry.get("text", ""), re.S)
    if m:
        spoken = " ".join(m.group(1).split())
        # The <d> wrapper must contain the script (whitespace-normalized) so
        # the manifest prompt and the scoring ground truth never diverge.
        norm_script = " ".join(script.split())
        if norm_script not in spoken and spoken not in norm_script:
            log.warning("script does not match <d> content for %s — trusting 'script'", entry.get("label"))
    return script, targets


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="H3 audio battery: ASR-scored target-word fidelity")
    p.add_argument("--manifest", required=True, help="ab_videos/manifest.json from run_h3_ab")
    p.add_argument("--suite", required=True, help="te_h3_audio_prompts.yaml (script/targets ground truth)")
    p.add_argument("--out", required=True, help="Output tebench_h3_audio.json path")
    p.add_argument("--whisper-model", default="openai/whisper-large-v3")
    p.add_argument("--device", default="cuda", help="Whisper device (cuda or cpu)")
    p.add_argument("--arms", default="", help="Comma-separated arm filter (default: all in manifest)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    suite_raw = yaml.safe_load(Path(args.suite).read_text())
    entries: dict[str, dict[str, Any]] = {}
    for e in suite_raw.get("prompts", []):
        entries[e.get("label") or ""] = e
    tmap = tier_map()

    clips = manifest.get("clips", [])
    if args.arms:
        arm_filter = {a.strip() for a in args.arms.split(",")}
        clips = [c for c in clips if c["variant"] in arm_filter]
    audio_clips = [c for c in clips if c.get("label") in entries and c.get("label")]
    log.info("Audio battery: %d clips across arms %s", len(audio_clips), sorted({c["variant"] for c in audio_clips}))
    if not audio_clips:
        log.error("No audio-battery clips in manifest (labels must match the audio suite)")
        return 1

    if args.dry_run:
        for c in audio_clips:
            log.info("[dry-run] %s seed%d :: %s", c["variant"], c["seed"], c["label"])
        return 0

    transcriber: Transcriber = WhisperTranscriber(args.whisper_model, device=args.device)
    result = run_battery(audio_clips, entries, tmap, transcriber, manifest)
    result["whisper_model"] = args.whisper_model
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    log.info("Wrote %s (smoke=%s, %d clips)", out_path, result["smoke"], result["n_clips"])
    return 0


def run_battery(
    audio_clips: list[dict[str, Any]],
    entries: dict[str, dict[str, Any]],
    tmap: dict[str, str],
    transcriber: Transcriber,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Transcribe + score + aggregate the battery.

    Transcriber seam: tests inject a fake here without touching Whisper.
    """
    # --- transcribe + score every clip ---
    per_clip: list[dict[str, Any]] = []
    for c in audio_clips:
        path = Path(c["path"])
        if not path.exists():
            log.warning("clip missing: %s", path)
            continue
        entry = entries[c["label"]]
        script, targets = extract_script_targets(entry)
        text = transcriber.transcribe(path)
        ls = score_line(script, text, targets)
        per_clip.append(
            {
                "variant": c["variant"],
                "label": c["label"],
                "seed": c["seed"],
                **ls,
            }
        )
        log.info(
            "%-12s seed%d %-24s wer=%.3f failed=%s",
            c["variant"],
            c["seed"],
            c["label"],
            ls["wer"],
            ls["failed_targets"],
        )

    # --- calibration gate: targets the base arm NEVER renders correctly ---
    all_targets = sorted({t for e in entries.values() for t in e.get("targets", [])})
    base_scores = [pc for pc in per_clip if pc["variant"] == "base"]
    if not base_scores:
        log.warning(
            "NO BASE ARM in clips — calibration gate disabled, Floor A undefined; "
            "verdicts from this run are uncalibrated"
        )
    excluded = calibration_excluded(base_scores, all_targets) if base_scores else []
    if excluded:
        log.warning("CALIBRATION: base arm never renders %s correctly — excluded from verdicts", excluded)

    # --- per-arm aggregation ---
    arms_out: dict[str, Any] = {}
    for variant in sorted({pc["variant"] for pc in per_clip}):
        v_scores = [pc for pc in per_clip if pc["variant"] == variant]
        agg = aggregate_arm(v_scores, tmap)
        # per-seed failed sets -> cross-seed overlap. H9 uses the failure-only
        # variant (a clean arm returns None, not a vacuous 1.0); the plain
        # jaccard is kept for descriptive comparison.
        by_seed: dict[int, set[str]] = {}
        for pc in v_scores:
            by_seed.setdefault(pc["seed"], set()).update(t for t in pc["failed_targets"] if t not in excluded)
        agg["cross_seed_jaccard"] = cross_seed_jaccard(by_seed)
        agg["cross_seed_jaccard_h9"] = cross_seed_jaccard_h9(by_seed)
        arms_out[variant] = agg

    # --- floors ---
    floors: dict[str, Any] = {"a_whisper_intrinsic": None, "b_quant_band": None}
    if "base" in arms_out:
        floors["a_whisper_intrinsic"] = {tier: agg["error_rate"] for tier, agg in arms_out["base"]["tiers"].items()}
        # Floor B: quant band from a second BASE-family arm. Convention: the
        # stand-in arm is named "base_<quant>" (documented in NOTES.md).
        other_base = next((v for v in arms_out if v != "base" and v.startswith("base_")), None)
        if other_base:
            floors["b_quant_band"] = {
                tier: round(
                    abs(
                        (arms_out["base"]["tiers"].get(tier) or {}).get("error_rate", 0.0)
                        - (arms_out[other_base]["tiers"].get(tier) or {}).get("error_rate", 0.0)
                    ),
                    4,
                )
                for tier in set(arms_out["base"]["tiers"]) | set(arms_out[other_base]["tiers"])
            }

    return {
        "te_results_version": TE_RESULTS_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_clips": len(per_clip),
        "smoke": bool(manifest.get("smoke", False)),
        "calibration_excluded": excluded,
        "arms": arms_out,
        "floors": floors,
        "clips": per_clip,
    }


if __name__ == "__main__":
    sys.exit(main())
