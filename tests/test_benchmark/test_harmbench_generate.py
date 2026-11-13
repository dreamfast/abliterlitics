"""Tests for src/benchmark/harmbench_generate.py — constants and _split_reasoning."""

from __future__ import annotations

import pytest

import src.benchmark.harmbench_generate as hg


class TestHarmbenchGenerateConstants:
    def test_csv_url_importable(self):
        from src.benchmark.harmbench_generate import HARMBENCH_BEHAVIORS_CSV

        assert isinstance(HARMBENCH_BEHAVIORS_CSV, str)
        assert "githubusercontent.com" in HARMBENCH_BEHAVIORS_CSV
        assert "HarmBench" in HARMBENCH_BEHAVIORS_CSV

    def test_csv_url_is_pinned_to_commit_sha(self):
        """The 'Pinned to specific commit SHA' comment must be true (audit: it pointed at main)."""
        from src.benchmark.benchmark_eval import HARMBENCH_BEHAVIORS_CSV as PINNED_EVAL_URL
        from src.benchmark.harmbench_generate import HARMBENCH_BEHAVIORS_CSV

        assert "main/data" not in HARMBENCH_BEHAVIORS_CSV
        # Same pinned commit as benchmark_eval.py (task item 3: aligned pins).
        assert HARMBENCH_BEHAVIORS_CSV == PINNED_EVAL_URL

    def test_refusal_substrings_importable(self):
        from src.benchmark.harmbench_generate import REFUSAL_SUBSTRINGS

        assert isinstance(REFUSAL_SUBSTRINGS, list)
        assert len(REFUSAL_SUBSTRINGS) > 5
        assert "I cannot" in REFUSAL_SUBSTRINGS


class TestSplitReasoning:
    """_split_reasoning is the most forensically load-bearing function in the package.

    A regression here silently corrupts reasoning_chars and classifier inputs for
    entire comparisons (lessons #55c, #61, #72).
    """

    @pytest.fixture(autouse=True)
    def _thinking_on(self, monkeypatch):
        # Default module state; individual tests override explicitly.
        monkeypatch.setattr(hg, "_ENABLE_THINKING", True)

    # --- Lesson #55c: Qwen3.x templates PREFILL <think>, output has ONLY the closer ---

    def test_prefilled_opener_split_at_first_closer(self):
        raw = "Deliberation about the request.\n</think>\nFinal answer here."
        reasoning, content = hg._split_reasoning(raw, api_reasoning="")
        assert reasoning == "Deliberation about the request."
        assert content == "Final answer here."

    def test_prefilled_opener_first_closer_wins(self):
        """Split at the FIRST close tag even if more than one appears."""
        raw = "thinking one</think>answer</think>trailing"
        reasoning, content = hg._split_reasoning(raw, api_reasoning="")
        assert reasoning == "thinking one"
        assert content == "answer</think>trailing"

    # --- Lesson #72: no closer => ALL raw output is deliberation (reasoning-first) ---

    def test_no_closer_reasoning_first_convention(self):
        raw = "I need to analyze this request carefully and consider the implications."
        reasoning, content = hg._split_reasoning(raw, api_reasoning="")
        assert reasoning == raw
        assert content == ""

    def test_no_closer_strips_surrounding_whitespace(self):
        raw = "  \n deliberation only \n "
        reasoning, content = hg._split_reasoning(raw, api_reasoning="")
        assert reasoning == "deliberation only"
        assert content == ""

    # --- --no-thinking runs: untagged output is genuine content, not deliberation ---

    def test_non_thinking_untagged_passthrough(self, monkeypatch):
        monkeypatch.setattr(hg, "_ENABLE_THINKING", False)
        raw = "Here is the information you requested about chemistry."
        reasoning, content = hg._split_reasoning(raw, api_reasoning="")
        assert reasoning == ""
        assert content == raw

    def test_non_thinking_explicit_tags_still_split(self, monkeypatch):
        """Explicit tags split regardless of the flag."""
        monkeypatch.setattr(hg, "_ENABLE_THINKING", False)
        reasoning, content = hg._split_reasoning("<think>plan</think>answer", api_reasoning="")
        assert reasoning == "plan"
        assert content == "answer"

    # --- API-provided reasoning field (vLLM reasoning parser) ---

    def test_api_reasoning_trusted_when_content_present(self):
        reasoning, content = hg._split_reasoning("visible answer", api_reasoning="hidden thinking")
        assert reasoning == "hidden thinking"
        assert content == "visible answer"

    def test_api_reasoning_empty_content_plain_refusal_misclassified(self):
        """Reasoning parser grabbed a refusal with no think tags -> it IS the response."""
        api_reasoning = "I'm sorry, but I can't assist with that."
        reasoning, content = hg._split_reasoning("", api_reasoning=api_reasoning)
        assert reasoning == ""
        assert content == api_reasoning

    def test_api_reasoning_empty_content_genuine_thinking(self):
        reasoning, content = hg._split_reasoning("", api_reasoning="<think>loop without closing</think>")
        assert reasoning == "<think>loop without closing</think>"
        assert content == ""

    # --- Qwen explicit tags (opener present in output) ---

    def test_explicit_think_tags_loop_strip(self):
        """Loop-strip removes tagged blocks; text BETWEEN tags is retained."""
        raw = "<think>alpha</think>visible<think>beta</think>final"
        reasoning, content = hg._split_reasoning(raw, api_reasoning="")
        assert reasoning == "alpha\nbeta"
        assert content == "visiblefinal"

    def test_unclosed_explicit_opener_consumes_all(self):
        raw = "<think>reasoning that never closes"
        reasoning, content = hg._split_reasoning(raw, api_reasoning="")
        assert reasoning == "reasoning that never closes"
        assert content == ""

    # --- Gemma 4 channel format ---

    def test_gemma_channel_format(self):
        raw = "<|channel>thought\nStep 1: analyze.\n<channel|>Answer here.<turn|>"
        reasoning, content = hg._split_reasoning(raw, api_reasoning="")
        assert reasoning == "Step 1: analyze."
        assert content == "Answer here."

    def test_gemma_channel_format_no_close(self):
        raw = "<|channel>thought\nall thinking no answer"
        reasoning, content = hg._split_reasoning(raw, api_reasoning="")
        assert reasoning == "all thinking no answer"
        assert content == ""
