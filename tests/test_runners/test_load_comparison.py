"""Tests for runners/_load_comparison.py — shell variable export logic."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "runners" / "_load_comparison.py"


def _run_load(comp_json_path: str, comp_dir: str) -> dict[str, str]:
    """Run _load_comparison.py and parse the shell variable output."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), comp_json_path, comp_dir],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"_load_comparison.py failed: {result.stderr}")
    env: dict[str, str] = {}
    for line in result.stdout.strip().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            # Strip shlex quoting (single quotes)
            if value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            env[key] = value
    return env


@pytest.fixture()
def comparison_dir(tmp_path: Path) -> Path:
    """Create a minimal comparison directory with config.json for base model."""
    # Create base model dir with config.json (for reasoning detection)
    base_dir = tmp_path / "models" / "base-model"
    base_dir.mkdir(parents=True)
    (base_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["SomeModelForCausalLM"],
                "model_type": "generic",
            }
        )
    )
    return tmp_path


def _write_comparison(
    comparison_dir: Path,
    name: str = "test-comp",
    base: str = "models/base-model",
    variants: dict | None = None,
    settings: dict | None = None,
) -> Path:
    """Write a comparison.json and return its path."""
    comp: dict = {
        "name": name,
        "base": base,
        "variants": variants or {"variant1": {"path": "models/variant1"}},
    }
    if settings is not None:
        comp["settings"] = settings
    comp_path = comparison_dir / "comparison.json"
    comp_path.write_text(json.dumps(comp))
    return comp_path


class TestCoreExports:
    """Test basic exports: name, dir, base, variants."""

    def test_basic_exports(self, comparison_dir: Path) -> None:
        comp_path = _write_comparison(comparison_dir)
        env = _run_load(str(comp_path), str(comparison_dir))

        assert env["COMPARISON_NAME"] == "test-comp"
        assert env["BASE_DIR"] == str(comparison_dir / "models" / "base-model")
        assert env["VARIANT_COUNT"] == "1"
        assert env["VARIANT_NAMES"] == "variant1"
        assert "VARIANT_PATH_variant1" in env

    def test_multiple_variants(self, comparison_dir: Path) -> None:
        variants = {
            "heretic": {"path": "models/heretic"},
            "hauhau": {"path": "models/hauhau"},
        }
        comp_path = _write_comparison(comparison_dir, variants=variants)
        env = _run_load(str(comp_path), str(comparison_dir))

        assert env["VARIANT_COUNT"] == "2"
        names = env["VARIANT_NAMES"].split()
        assert "heretic" in names
        assert "hauhau" in names
        assert "VARIANT_PATH_heretic" in env
        assert "VARIANT_PATH_hauhau" in env

    def test_hyphenated_variant_names(self, comparison_dir: Path) -> None:
        variants = {"my-variant": {"path": "models/my-variant"}}
        comp_path = _write_comparison(comparison_dir, variants=variants)
        env = _run_load(str(comp_path), str(comparison_dir))

        # Shell-safe: hyphens replaced with underscores
        assert "VARIANT_PATH_my_variant" in env


class TestSettingsDefaults:
    """Test that all settings have sensible defaults when not specified."""

    def test_defaults_without_settings_block(self, comparison_dir: Path) -> None:
        comp_path = _write_comparison(comparison_dir)
        env = _run_load(str(comp_path), str(comparison_dir))

        assert env["INFERENCE_BACKEND"] == "auto"
        assert "mmlu" in env["LM_EVAL_TASKS"]
        assert env["LM_EVAL_MAX_MODEL_LEN"] == "8192"
        assert env["LM_EVAL_MAX_GEN_TOKS"] == "2048"
        assert env["HARMBENCH_MAX_TOKENS"] == "8096"
        assert env["TOKENIZER_DIR"] == env["BASE_DIR"]
        assert env["GGUF_DIR"] == ""
        assert env["KL_NUM_PROMPTS"] == "100"
        assert env["KL_DATASET"] == "mlabonne/harmless_alpaca"
        assert env["REASONING_PARSER"] == ""
        assert env["REASONING_CONFIG_JSON"] == ""

    def test_explicit_settings_override_defaults(self, comparison_dir: Path) -> None:
        settings = {
            "inference_backend": "vllm",
            "lm_eval_max_model_len": 4096,
            "lm_eval_max_gen_toks": 1024,
            "harmbench_max_tokens": 4096,
            "kl_num_prompts": 50,
        }
        comp_path = _write_comparison(comparison_dir, settings=settings)
        env = _run_load(str(comp_path), str(comparison_dir))

        assert env["INFERENCE_BACKEND"] == "vllm"
        assert env["LM_EVAL_MAX_MODEL_LEN"] == "4096"
        assert env["LM_EVAL_MAX_GEN_TOKS"] == "1024"
        assert env["HARMBENCH_MAX_TOKENS"] == "4096"
        assert env["KL_NUM_PROMPTS"] == "50"


class TestTokenizerDir:
    """Test TOKENIZER_DIR resolution."""

    def test_tokenizer_dir_defaults_to_base(self, comparison_dir: Path) -> None:
        comp_path = _write_comparison(comparison_dir)
        env = _run_load(str(comp_path), str(comparison_dir))
        assert env["TOKENIZER_DIR"] == env["BASE_DIR"]

    def test_tokenizer_dir_explicit(self, comparison_dir: Path) -> None:
        settings = {"tokenizer_dir": "models/shared-tokenizer"}
        comp_path = _write_comparison(comparison_dir, settings=settings)
        env = _run_load(str(comp_path), str(comparison_dir))
        expected = str(comparison_dir / "models" / "shared-tokenizer")
        assert env["TOKENIZER_DIR"] == expected


class TestGgufDir:
    """Test GGUF_DIR resolution."""

    def test_gguf_dir_empty_by_default(self, comparison_dir: Path) -> None:
        comp_path = _write_comparison(comparison_dir)
        env = _run_load(str(comp_path), str(comparison_dir))
        assert env["GGUF_DIR"] == ""

    def test_gguf_dir_explicit(self, comparison_dir: Path) -> None:
        settings = {"gguf_dir": "models/gguf"}
        comp_path = _write_comparison(comparison_dir, settings=settings)
        env = _run_load(str(comp_path), str(comparison_dir))
        expected = str(comparison_dir / "models" / "gguf")
        assert env["GGUF_DIR"] == expected


class TestReasoningParser:
    """Test auto-detection and override of reasoning parser."""

    def test_no_parser_for_generic_model(self, comparison_dir: Path) -> None:
        comp_path = _write_comparison(comparison_dir)
        env = _run_load(str(comp_path), str(comparison_dir))
        assert env["REASONING_PARSER"] == ""
        assert env["REASONING_CONFIG_JSON"] == ""

    def test_auto_detect_qwen3_architecture(self, comparison_dir: Path) -> None:
        # Rewrite base config.json to have Qwen3 architecture
        base_dir = comparison_dir / "models" / "base-model"
        (base_dir / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["Qwen3ForCausalLM"],
                    "model_type": "qwen3",
                }
            )
        )
        comp_path = _write_comparison(comparison_dir)
        env = _run_load(str(comp_path), str(comparison_dir))

        assert env["REASONING_PARSER"] == "qwen3"
        config = json.loads(env["REASONING_CONFIG_JSON"])
        assert config["reasoning_start_str"] == "<think}"
        assert config["reasoning_end_str"] == "</think}"

    def test_auto_detect_qwen35_conditional(self, comparison_dir: Path) -> None:
        base_dir = comparison_dir / "models" / "base-model"
        (base_dir / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["Qwen3_5ForConditionalGeneration"],
                }
            )
        )
        comp_path = _write_comparison(comparison_dir)
        env = _run_load(str(comp_path), str(comparison_dir))

        assert env["REASONING_PARSER"] == "qwen3"

    def test_override_disables_parser(self, comparison_dir: Path) -> None:
        # Qwen3 model but override says "none"
        base_dir = comparison_dir / "models" / "base-model"
        (base_dir / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["Qwen3ForCausalLM"],
                }
            )
        )
        settings = {"reasoning_parser": "none"}
        comp_path = _write_comparison(comparison_dir, settings=settings)
        env = _run_load(str(comp_path), str(comparison_dir))

        assert env["REASONING_PARSER"] == ""
        assert env["REASONING_CONFIG_JSON"] == ""

    def test_override_forces_parser(self, comparison_dir: Path) -> None:
        # Non-Qwen model but override forces it
        settings = {"reasoning_parser": "qwen3"}
        comp_path = _write_comparison(comparison_dir, settings=settings)
        env = _run_load(str(comp_path), str(comparison_dir))

        assert env["REASONING_PARSER"] == "qwen3"

    def test_no_config_json_no_crash(self, comparison_dir: Path) -> None:
        # Remove config.json — should not crash, just no parser
        base_dir = comparison_dir / "models" / "base-model"
        (base_dir / "config.json").unlink()
        comp_path = _write_comparison(comparison_dir)
        env = _run_load(str(comp_path), str(comparison_dir))

        assert env["REASONING_PARSER"] == ""
        assert env["REASONING_CONFIG_JSON"] == ""


class TestSkipFlags:
    """Test per-variant skip flag exports."""

    def test_skip_flags_default_to_zero(self, comparison_dir: Path) -> None:
        comp_path = _write_comparison(comparison_dir)
        env = _run_load(str(comp_path), str(comparison_dir))

        assert env["VARIANT_SKIP_HB_variant1"] == "0"
        assert env["VARIANT_SKIP_LM_variant1"] == "0"
        assert env["VARIANT_SKIP_KL_variant1"] == "0"
        assert env["VARIANT_SKIP_WT_variant1"] == "0"

    def test_skip_flags_set(self, comparison_dir: Path) -> None:
        variants = {
            "heretic": {"path": "models/heretic", "skip_harmbench": True, "skip_lm_eval": True},
            "hauhau": {"path": "models/hauhau", "skip_kl": True, "skip_weight": True},
        }
        comp_path = _write_comparison(comparison_dir, variants=variants)
        env = _run_load(str(comp_path), str(comparison_dir))

        assert env["VARIANT_SKIP_HB_heretic"] == "1"
        assert env["VARIANT_SKIP_LM_heretic"] == "1"
        assert env["VARIANT_SKIP_KL_heretic"] == "0"
        assert env["VARIANT_SKIP_WT_heretic"] == "0"

        assert env["VARIANT_SKIP_HB_hauhau"] == "0"
        assert env["VARIANT_SKIP_LM_hauhau"] == "0"
        assert env["VARIANT_SKIP_KL_hauhau"] == "1"
        assert env["VARIANT_SKIP_WT_hauhau"] == "1"


class TestValidation:
    """Test slug validation and error handling."""

    def test_invalid_comparison_name(self, comparison_dir: Path) -> None:
        comp_path = _write_comparison(comparison_dir, name="INVALID")
        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(comp_path), str(comparison_dir)],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "Invalid comparison name" in result.stdout

    def test_invalid_variant_key(self, comparison_dir: Path) -> None:
        variants = {"UPPER": {"path": "models/upper"}}
        comp_path = _write_comparison(comparison_dir, variants=variants)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(comp_path), str(comparison_dir)],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "Invalid variant key" in result.stdout

    def test_shlex_quoting_special_chars(self, comparison_dir: Path) -> None:
        """Verify paths with spaces are properly quoted."""
        # Create a dir with spaces in the path
        spaced_dir = comparison_dir / "path with spaces"
        spaced_dir.mkdir()
        base_dir = spaced_dir / "base"
        base_dir.mkdir()
        (base_dir / "config.json").write_text("{}")

        comp_path = _write_comparison(comparison_dir, base="path with spaces/base")
        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(comp_path), str(comparison_dir)],
            capture_output=True,
            text=True,
        )
        # Should succeed — the output is shell-safe
        assert result.returncode == 0
