"""Tests for src/benchmark/logprobs_proxy.py — Format translation logic."""
from __future__ import annotations

import pytest

# flask may not be available in test environment; skip if missing
flask = pytest.importorskip("flask", reason="flask not installed")

from src.benchmark.logprobs_proxy import translate_logprobs


class TestTranslateLogprobs:
    def test_translates_content_array_to_flat_format(self):
        """OpenAI content array format should become flat token_logprobs."""
        resp = {
            "choices": [{
                "logprobs": {
                    "content": [
                        {"token": "Hello", "logprob": -1.5, "top_logprobs": [
                            {"token": "Hello", "logprob": -1.5},
                            {"token": "Hi", "logprob": -3.0},
                        ]},
                        {"token": " world", "logprob": -0.5, "top_logprobs": [
                            {"token": " world", "logprob": -0.5},
                        ]},
                    ]
                }
            }]
        }

        result = translate_logprobs(resp)
        choice = result["choices"][0]
        lp = choice["logprobs"]

        assert "token_logprobs" in lp
        assert "top_logprobs" in lp
        assert lp["token_logprobs"] == [-1.5, -0.5]
        assert len(lp["top_logprobs"]) == 2
        assert lp["top_logprobs"][0] == {"Hello": -1.5, "Hi": -3.0}

    def test_no_content_key_passthrough(self):
        """Responses without content key should pass through unchanged."""
        resp = {"choices": [{"logprobs": {"token_logprobs": [-1.0]}}]}
        result = translate_logprobs(resp)
        assert result == resp

    def test_empty_choices(self):
        resp = {"choices": []}
        result = translate_logprobs(resp)
        assert result == {"choices": []}

    def test_no_logprobs_key(self):
        resp = {"choices": [{"text": "Hello"}]}
        result = translate_logprobs(resp)
        assert result == {"choices": [{"text": "Hello"}]}
