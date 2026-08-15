"""Pure audio-fidelity metrics for the H3 audio battery.

No model code here — every function is deterministic and unit-tested so the
headline verdict logic (H8/H9/P19) runs on trustworthy primitives:

- ``syllable_count``: vowel-group heuristic (documented limitation: silent-e
  and diphthong edge cases; good enough for fidelity RATIOS between target
  and rendered word).
- ``align_target``: fuzzy alignment of a target word against transcript
  tokens (difflib ratio): correct >= 0.9, mispronounced 0.5–0.9 (returns the
  rendered word + ratio + syllable fidelity), dropped < 0.5. A ratio-"correct"
  match whose rendered token has FEWER syllables than the target is
  downgraded to mispronounced — that is exactly the "syllables swallowed"
  artifact (comfortable -> comfortble scores 0.95 by ratio but 3/4 syllables,
  so it counts as mispronounced). Handles the p-e-w case: "inconceivable"
  vs "inconcable" -> mispronounced, syllable fidelity 4/5 = 0.8.
- ``word_error_rate``: token-level Levenshtein WER vs the scripted line.
- ``cross_seed_jaccard``: overlap of failed-target sets across seeds — the
  H9 "same words fail across seeds" signal (structured edit vs noise).
- ``floor_*``: Floor A (whisper intrinsic on the base arm) and Floor B
  (quant band between two base arms) helpers.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

VOWELS = re.compile(r"[aeiouy]+")
WORD_SPLIT = re.compile(r"[^a-z0-9']+")

CORRECT_THRESHOLD = 0.9
MISPRONOUNCED_THRESHOLD = 0.5


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens (apostrophes kept inside words)."""
    return [t for t in WORD_SPLIT.split(text.lower()) if t]


def syllable_count(word: str) -> int:
    """Vowel-group syllable estimate (>= 1)."""
    groups = VOWELS.findall(word.lower())
    return max(1, len(groups))


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def align_target_occurrences(target: str, script: str, transcript: str) -> list[dict[str, Any]]:
    """Align EVERY scripted occurrence of the target to distinct transcript tokens.

    Per-utterance scoring: a target repeated in the line ("truly
    inconceivable") produces one alignment per occurrence. Tokens keep their
    multiplicity and each is consumed at most once — a word rendered twice
    correctly scores TWO correct alignments (an earlier set()-dedup version
    mis-scored the second identical token as dropped, caught on real data).
    Scripted occurrences with no remaining candidate >=
    MISPRONOUNCED_THRESHOLD are "dropped".
    """
    n_occ = tokenize(script).count(target.lower())
    if n_occ == 0:
        n_occ = 1  # defensive: score at least once even on script mismatch
    pool = tokenize(transcript)
    out: list[dict[str, Any]] = []
    for _ in range(n_occ):
        best_i, best_r = -1, 0.0
        for i, t in enumerate(pool):
            r = _ratio(target.lower(), t)
            if r > best_r:
                best_i, best_r = i, r
        if best_i >= 0 and best_r >= MISPRONOUNCED_THRESHOLD:
            out.append(_classify(target, pool[best_i], best_r))
            pool.pop(best_i)
        else:
            out.append(_classify(target, "", best_r if best_i >= 0 else 0.0))
    return out


def _classify(target: str, matched: str, r: float) -> dict[str, Any]:
    """Build the alignment record for a (target, matched-token, ratio) triple.

    A ratio-\"correct\" match with FEWER syllables in the rendered token than
    the target is downgraded to mispronounced (the swallowed-syllable guard:
    difflib ratios stay >= 0.9 for single-syllable drops in long words).
    """
    syl_t = syllable_count(target)
    if r >= CORRECT_THRESHOLD:
        syl_m = syllable_count(matched)
        if matched and syl_m < syl_t:
            return {
                "target": target,
                "verdict": "mispronounced",
                "matched": matched,
                "ratio": round(r, 4),
                "syllables_target": syl_t,
                "syllables_rendered": syl_m,
                "syllable_fidelity": round(syl_m / syl_t, 4),
                "downgraded_by_syllables": True,
            }
        return {
            "target": target,
            "verdict": "correct",
            "matched": matched,
            "ratio": round(r, 4),
            "syllables_target": syl_t,
            "syllables_rendered": syl_t,
            "syllable_fidelity": 1.0,
        }
    if r >= MISPRONOUNCED_THRESHOLD:
        syl_r = syllable_count(matched)
        return {
            "target": target,
            "verdict": "mispronounced",
            "matched": matched,
            "ratio": round(r, 4),
            "syllables_target": syl_t,
            "syllables_rendered": syl_r,
            "syllable_fidelity": round(syl_r / syl_t, 4),
        }
    return {
        "target": target,
        "verdict": "dropped",
        "matched": "",
        "ratio": round(r, 4),
        "syllables_target": syl_t,
        "syllables_rendered": 0,
        "syllable_fidelity": 0.0,
    }


def align_target(target: str, transcript: str) -> dict[str, Any]:
    """Classify one target word against a transcript (best single match).

    Returns dict with:
      target, verdict ("correct" | "mispronounced" | "dropped"),
      matched (best transcript token or ""), ratio,
      syllables_target, syllables_rendered, syllable_fidelity
    """
    toks = tokenize(transcript)
    best, best_r = "", 0.0
    for t in toks:
        r = _ratio(target.lower(), t)
        if r > best_r:
            best, best_r = t, r
    return _classify(target, best, best_r)


def word_error_rate(script: str, transcript: str) -> float:
    """Token-level Levenshtein WER (0 = identical, 1 = nothing right)."""
    ref, hyp = tokenize(script), tokenize(transcript)
    if not ref:
        return 0.0
    # standard DP over tokens
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


def cross_seed_jaccard(failed_sets: dict[Any, set[str]]) -> float | None:
    """Mean pairwise Jaccard overlap of failed-target sets across seeds.

    None when fewer than 2 seeds (cannot estimate overlap). H9 signal:
    heretic expected >= 2x the base arm's overlap.
    """
    keys = sorted(failed_sets, key=str)
    if len(keys) < 2:
        return None
    sims: list[float] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = failed_sets[keys[i]], failed_sets[keys[j]]
            if not a and not b:
                sims.append(1.0)  # both clean: perfectly consistent
                continue
            union = a | b
            sims.append(len(a & b) / len(union) if union else 1.0)
    return sum(sims) / len(sims)


def cross_seed_jaccard_h9(failed_sets: dict[Any, set[str]]) -> float | None:
    """H9-safe overlap: FAILURE seeds only; vacuous-clean arms return None.

    ``cross_seed_jaccard`` treats both-clean seed pairs as 1.0 — right for
    describing consistency, WRONG for the H9 ratio: a clean base (1.0) vs a
    reproducibly-failing heretic (1.0) would mask the structured-edit
    signal. This variant computes overlap only over seeds with >= 1 failure
    and returns None when fewer than 2 such seeds exist (an error-free arm
    has no failure structure to compare — that IS the null result).
    """
    failing = {k: v for k, v in failed_sets.items() if v}
    keys = sorted(failing, key=str)
    if len(keys) < 2:
        return None
    sims: list[float] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = failing[keys[i]], failing[keys[j]]
            union = a | b
            sims.append(len(a & b) / len(union) if union else 1.0)
    return sum(sims) / len(sims)


def score_line(script: str, transcript: str, targets: list[str]) -> dict[str, Any]:
    """Score one clip: per-occurrence target alignments + line WER + failure set.

    ``failed_targets`` is the UNIQUE set of failed targets (occurrence
    multiplicity lives in ``alignments``) — set semantics on purpose, for
    jaccard/calibration consumers.
    """
    alignments: list[dict[str, Any]] = []
    for t in targets:
        alignments.extend(align_target_occurrences(t, script, transcript))
    failed = {a["target"] for a in alignments if a["verdict"] != "correct"}
    return {
        "wer": round(word_error_rate(script, transcript), 4),
        "transcript": transcript,
        "alignments": alignments,
        "failed_targets": sorted(failed),
    }


def tier_of(word: str, tier_map: dict[str, str]) -> str:
    """Look up a target word's tier ("unknown" when not in the map)."""
    return tier_map.get(word.lower(), "unknown")


def aggregate_arm(
    line_scores: list[dict[str, Any]],
    tier_map: dict[str, str],
) -> dict[str, Any]:
    """Aggregate per-clip scores into per-tier counts for one arm.

    Counts are per UTTERANCE (each in-line repetition of a target word is one
    alignment), so "inconceivable, truly inconceivable" scores twice.
    """
    tiers: dict[str, dict[str, Any]] = {}
    for ls in line_scores:
        for a in ls["alignments"]:
            tier = tier_of(a["target"], tier_map)
            agg = tiers.setdefault(tier, {"correct": 0, "mispronounced": 0, "dropped": 0, "total": 0, "_fid": []})
            agg[a["verdict"]] += 1
            agg["total"] += 1
            if a["verdict"] != "correct":
                agg["_fid"].append(a["syllable_fidelity"])
    for _tier, agg in tiers.items():
        n = agg.pop("total")
        fid = agg.pop("_fid")
        agg["total"] = n
        agg["error_rate"] = round((n - agg["correct"]) / n, 4) if n else None
        agg["mispronounced_rate"] = round(agg["mispronounced"] / n, 4) if n else None
        agg["mean_syllable_fidelity"] = round(sum(fid) / len(fid), 4) if fid else None
    wers = [ls["wer"] for ls in line_scores] or [0.0]
    return {
        "n_clips": len(line_scores),
        "mean_wer": round(sum(wers) / len(wers), 4),
        "tiers": tiers,
    }


def calibration_excluded(
    arm_line_scores: list[dict[str, Any]],
    targets: list[str],
) -> list[str]:
    """Targets the BASE arm never renders correctly in ANY clip.

    If whisper/the model can never produce a >=0.9 match for a word even on
    the presumed-clean base arm, the battery is blind to that word — such
    targets are excluded from H8/H9 verdicts and disclosed.
    """
    never = set(targets)
    for ls in arm_line_scores:
        for a in ls["alignments"]:
            if a["verdict"] == "correct":
                never.discard(a["target"])
    return sorted(never)
