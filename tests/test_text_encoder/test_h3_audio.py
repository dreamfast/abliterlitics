"""Tests for h3_audio.py — pure audio-fidelity primitives.

The p-e-w example ("inconceivable" -> "inconcable") is the regression anchor
for the whole battery.
"""

from __future__ import annotations

from src.text_encoder.h3_audio import (
    aggregate_arm,
    align_target,
    align_target_occurrences,
    calibration_excluded,
    cross_seed_jaccard,
    cross_seed_jaccard_h9,
    score_line,
    syllable_count,
    tier_of,
    tokenize,
    word_error_rate,
)

TIER_MAP = {
    "inconceivable": "rare_polysyllabic",
    "entrepreneurial": "rare_polysyllabic",
    "beautiful": "common_polysyllabic",
    "remember": "common_polysyllabic",
    "cat": "monosyllabic",
    "red": "monosyllabic",
}


class TestSyllables:
    def test_inconceivable_is_5(self) -> None:
        assert syllable_count("inconceivable") == 5

    def test_inconcable_is_4(self) -> None:
        # p-e-w's observed rendering: syllables swallowed
        assert syllable_count("inconcable") == 4

    def test_monosyllables(self) -> None:
        for w in ["cat", "red", "box", "strength"]:
            assert syllable_count(w) >= 1

    def test_empty_floor(self) -> None:
        assert syllable_count("x") == 1


class TestAlignTarget:
    def test_correct(self) -> None:
        a = align_target("cat", "the cat sat")
        assert a["verdict"] == "correct"
        assert a["matched"] == "cat"
        assert a["syllable_fidelity"] == 1.0

    def test_pew_case_mispronounced(self) -> None:
        """inconceivable -> inconcable: the exact reported artifact."""
        a = align_target("inconceivable", "it is utterly inconcable colleagues")
        assert a["verdict"] == "mispronounced"
        assert a["matched"] == "inconcable"
        assert 0.5 <= a["ratio"] < 0.9
        assert a["syllables_target"] == 5
        assert a["syllables_rendered"] == 4
        assert a["syllable_fidelity"] == 0.8

    def test_dropped(self) -> None:
        a = align_target("juxtaposition", "the thing of steel and silk here")
        assert a["verdict"] == "dropped"
        assert a["matched"] == ""
        assert a["syllable_fidelity"] == 0.0

    def test_case_insensitive(self) -> None:
        assert align_target("Beautiful", "What a BEAUTIFUL day")["verdict"] == "correct"


class TestAlignOccurrences:
    def test_repeated_target_scores_twice(self) -> None:
        als = align_target_occurrences(
            "inconceivable",
            "utterly inconceivable, truly inconceivable",
            "utterly inconceivable, truly inconcable",
        )
        assert len(als) == 2
        verdicts = sorted(a["verdict"] for a in als)
        assert verdicts == ["correct", "mispronounced"]

    def test_both_swallowed(self) -> None:
        als = align_target_occurrences("inconceivable", "inconceivable inconceivable", "utterly incomprehensible")
        assert all(a["verdict"] != "correct" for a in als)

    def test_single_occurrence_matches_align_target(self) -> None:
        a1 = align_target("cat", "the cat sat")
        a2 = align_target_occurrences("cat", "the cat", "the cat sat")
        assert a2[0]["verdict"] == a1["verdict"] == "correct"

    def test_identical_duplicate_tokens_score_twice_correct(self) -> None:
        """Real-data regression (prof_inconceivable smoke): the same word
        rendered twice must score TWO correct alignments, not one correct +
        one phantom 'dropped' from set()-dedup."""
        als = align_target_occurrences(
            "inconceivable",
            "utterly inconceivable, truly inconceivable, that the committee",
            "it is utterly inconceivable, colleagues, truly inconceivable, that the committee",
        )
        assert len(als) == 2
        assert all(a["verdict"] == "correct" for a in als)
        assert all(a["matched"] == "inconceivable" for a in als)

    def test_second_occurrence_actually_swallowed(self) -> None:
        als = align_target_occurrences(
            "inconceivable",
            "utterly inconceivable, truly inconceivable",
            "utterly inconceivable, truly inconcable",
        )
        verdicts = sorted(a["verdict"] for a in als)
        assert verdicts == ["correct", "mispronounced"]


class TestWER:
    def test_identical(self) -> None:
        assert word_error_rate("the cat sat", "the cat sat") == 0.0

    def test_one_substitution(self) -> None:
        # 3 tokens, 1 wrong -> 1/3
        assert abs(word_error_rate("the cat sat", "the bat sat") - 1 / 3) < 1e-9

    def test_empty_transcript(self) -> None:
        assert word_error_rate("the cat sat", "") == 1.0

    def test_punctuation_ignored(self) -> None:
        assert word_error_rate("it is, truly!", "it is truly") == 0.0


class TestCrossSeedJaccard:
    def test_needs_two_seeds(self) -> None:
        assert cross_seed_jaccard({42: {"cat"}}) is None

    def test_identical_failures_high_overlap(self) -> None:
        j = cross_seed_jaccard({42: {"inconceivable", "moon"}, 43: {"inconceivable", "moon"}})
        assert j == 1.0

    def test_disjoint_failures_low_overlap(self) -> None:
        j = cross_seed_jaccard({42: {"cat"}, 43: {"moon"}})
        assert j == 0.0

    def test_both_clean_is_consistent(self) -> None:
        assert cross_seed_jaccard({42: set(), 43: set()}) == 1.0

    def test_partial(self) -> None:
        # {a,b} vs {b,c}: intersection 1, union 3 -> 1/3
        j = cross_seed_jaccard({42: {"a", "b"}, 43: {"b", "c"}})
        assert abs(j - 1 / 3) < 1e-9


class TestScoreLine:
    def test_mixed_line(self) -> None:
        ls = score_line("the inconceivable cat", "the inconcable cat", ["inconceivable", "cat"])
        assert ls["failed_targets"] == ["inconceivable"]
        verdicts = {a["target"]: a["verdict"] for a in ls["alignments"]}
        assert verdicts["cat"] == "correct"
        assert verdicts["inconceivable"] == "mispronounced"


class TestAggregateArm:
    def test_counts_per_utterance(self) -> None:
        """In-line repetition scores twice (two alignments for the word)."""
        ls1 = score_line(
            "utterly inconceivable, truly inconceivable",
            "utterly inconceivable, truly inconcable",
            ["inconceivable"],
        )
        ls2 = score_line("the cat", "the cat", ["cat"])
        agg = aggregate_arm([ls1, ls2], TIER_MAP)
        rare = agg["tiers"]["rare_polysyllabic"]
        assert rare["total"] == 2
        assert rare["correct"] == 1
        assert rare["mispronounced"] == 1
        assert rare["error_rate"] == 0.5
        mono = agg["tiers"]["monosyllabic"]
        assert mono["total"] == 1 and mono["correct"] == 1
        assert agg["n_clips"] == 2

    def test_syllable_fidelity_mean(self) -> None:
        ls = score_line("inconceivable", "inconcable", ["inconceivable"])
        agg = aggregate_arm([ls], TIER_MAP)
        assert agg["tiers"]["rare_polysyllabic"]["mean_syllable_fidelity"] == 0.8

    def test_empty_arm(self) -> None:
        agg = aggregate_arm([], TIER_MAP)
        assert agg["n_clips"] == 0
        assert agg["tiers"] == {}


class TestCalibration:
    def test_never_correct_excluded(self) -> None:
        scores = [score_line("juxtaposition", "the thing", ["juxtaposition"])]
        assert calibration_excluded(scores, ["juxtaposition"]) == ["juxtaposition"]

    def test_sometime_correct_kept(self) -> None:
        scores = [
            score_line("cat", "the cat", ["cat"]),
            score_line("cat", "the bat", ["cat"]),
        ]
        assert calibration_excluded(scores, ["cat"]) == []


class TestTierOf:
    def test_lookup(self) -> None:
        assert tier_of("INCONCEIVABLE", TIER_MAP) == "rare_polysyllabic"
        assert tier_of("unknownword", TIER_MAP) == "unknown"


class TestSyllableDowngrade:
    """Review round 1: ratio-0.9 'correct' matches with swallowed syllables
    (comfortable -> comfortble, difflib 0.95) must classify as mispronounced."""

    def test_comfortable_comfortble_downgraded(self) -> None:
        a = align_target("comfortable", "this seat is very comfortble")
        assert a["verdict"] == "mispronounced"
        assert a["matched"] == "comfortble"
        assert a["syllable_fidelity"] < 1.0
        assert a.get("downgraded_by_syllables") is True

    def test_exact_match_stays_correct(self) -> None:
        a = align_target("comfortable", "this seat is very comfortable")
        assert a["verdict"] == "correct"
        assert "downgraded_by_syllables" not in a

    def test_pew_case_still_mispronounced(self) -> None:
        a = align_target("inconceivable", "utterly inconcable colleagues")
        assert a["verdict"] == "mispronounced"
        assert a["syllable_fidelity"] == 0.8


class TestJaccardH9:
    def test_vacuous_clean_arm_returns_none(self) -> None:
        """H9-safe: a clean arm has no failure structure — None, not 1.0."""
        assert cross_seed_jaccard_h9({42: set(), 43: set()}) is None

    def test_one_failing_seed_insufficient(self) -> None:
        assert cross_seed_jaccard_h9({42: {"cat"}, 43: set()}) is None

    def test_two_failing_seeds_same_words(self) -> None:
        j = cross_seed_jaccard_h9({42: {"inconceivable"}, 43: {"inconceivable"}})
        assert j == 1.0

    def test_clean_seed_ignored_in_mixed_arm(self) -> None:
        # seeds 42/44 fail identically; 43 clean -> excluded from the mean
        j = cross_seed_jaccard_h9({42: {"cat"}, 43: set(), 44: {"cat"}})
        assert j == 1.0

    def test_descriptive_jaccard_still_both_clean_one(self) -> None:
        assert cross_seed_jaccard({42: set(), 43: set()}) == 1.0


class TestTokenize:
    def test_basic(self) -> None:
        assert tokenize("It's a cat, isn't it?") == ["it's", "a", "cat", "isn't", "it"]
