"""Tests for src/benchmark/gsm8k_eval.py extraction helpers."""

from __future__ import annotations

from src.benchmark.gsm8k_eval import extract_answer, extract_last_number, normalize_num


class TestExtractLastNumber:
    """Flexible GSM8K extraction: last numeric token in the text."""

    def test_returns_last_number(self):
        assert extract_last_number("The answer is 5 + 3 = 8") == 8.0

    def test_returns_last_when_no_marker(self):
        # Model emits a final answer without the canonical #### marker
        assert extract_last_number("So the result is 42") == 42.0

    def test_handles_decimals(self):
        assert extract_last_number("total: 3.14") == 3.14

    def test_handles_negative(self):
        assert extract_last_number("temperature dropped to -7 degrees") == -7.0

    def test_strips_thousands_separators(self):
        assert extract_last_number("revenue was 1,234,567 dollars") == 1234567.0

    def test_returns_none_when_no_number(self):
        assert extract_last_number("no numbers here at all") is None

    def test_takes_very_last_number_not_first(self):
        # Multiple numbers: flexible match uses the LAST one (the final answer)
        assert extract_last_number("step 1: 10, step 2: 20, final 30") == 30.0

    def test_distinct_from_extract_answer_marker_branch(self):
        # extract_answer prefers #### marker; extract_last_number ignores markers
        text = "I got 99 problems but #### 6 is the answer"
        assert extract_answer(text) == 6.0  # marker branch
        assert extract_last_number(text) == 6.0  # last number is also 6

    def test_distinct_from_extract_answer_no_marker(self):
        # Without a marker both fall back to the last number
        text = "after working it out, the answer is 17"
        assert extract_answer(text) == 17.0
        assert extract_last_number(text) == 17.0


class TestNormalizeNum:
    def test_none_passthrough(self):
        assert normalize_num(None) is None

    def test_integer_valued_float_becomes_int(self):
        assert normalize_num(42.0) == 42

    def test_float_stays_float(self):
        assert normalize_num(3.5) == 3.5
