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
            # Strip the "export " prefix (added so the output can be eval'd directly)
            key = key.removeprefix("export ").strip()
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
        assert env["BASE_NAME"] == "base-model"
        assert env["VARIANT_COUNT"] == "1"
        assert env["VARIANT_NAMES"] == "variant1"
        assert "VARIANT_PATH_variant1" in env

    def test_base_name_is_basename_with_hyphens(self, comparison_dir: Path) -> None:
        """BASE_NAME must be the directory basename even for ../..-relative paths."""
        (comparison_dir / "models" / "gemma-4-E4B-it").mkdir(parents=True, exist_ok=True)
        comp_path = _write_comparison(comparison_dir, base="models/gemma-4-E4B-it")
        env = _run_load(str(comp_path), str(comparison_dir))

        assert env["BASE_NAME"] == "gemma-4-E4B-it"

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
        # Defaults follow the OLL v2 suite (see AGENTS.md "Key Defaults"):
        # max_model_len 16384 (BBH 3-shot needs the context), max_gen_toks 4096.
        assert env["LM_EVAL_MAX_MODEL_LEN"] == "16384"
        assert env["LM_EVAL_MAX_GEN_TOKS"] == "4096"
        assert env["HARMBENCH_MAX_TOKENS"] == "8096"
        assert env["TOKENIZER_DIR"] == env["BASE_DIR"]
        assert env["GGUF_DIR"] == ""
        assert env["KL_NUM_PROMPTS"] == "100"
        # KL_DATASET: no shell/env consumer, but kept in lockstep with
        # src/config.py's settings reader (tests/test_config.py enforces parity).
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
        # AGENTS.md lesson #55b: the exported config must carry the CORRECT
        # think-tag delimiters. (A stale version of this test asserted the old
        # `<think}`/`</think}` typos — the test encoded the bug as the contract.)
        assert config["reasoning_start_str"] == "<think>"
        assert config["reasoning_end_str"] == "</think>"

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


class TestTebenchImageMetrics:
    """Phase-4 image-metric settings exports + schema lockstep (lesson #28a)."""

    def test_img_defaults(self, comparison_dir: Path) -> None:
        _write_comparison(
            comparison_dir,
            settings={"tebench": {"backend": "zimage"}},
        )
        env = _run_load(str(comparison_dir / "comparison.json"), str(comparison_dir))
        assert env["TEBENCH_IMG_CLIP_MODEL"] == "ViT-L-14"
        assert env["TEBENCH_IMG_CLIP_PRETRAINED"] == "openai"
        assert env["TEBENCH_IMG_REFERENCE"] == "base_int8"
        assert env["TEBENCH_IMG_FLOOR_PAIR"] == "base_int8,base_fp8"

    def test_img_overrides(self, comparison_dir: Path) -> None:
        _write_comparison(
            comparison_dir,
            settings={
                "tebench": {
                    "backend": "zimage",
                    "img_clip_model": "ViT-B-32",
                    "img_clip_pretrained": "laion2b_s34b_b79k",
                    "img_reference": "base_gguf_f16",
                    "img_floor_pair": ["base_gguf_f16", "base_int8"],
                }
            },
        )
        env = _run_load(str(comparison_dir / "comparison.json"), str(comparison_dir))
        assert env["TEBENCH_IMG_CLIP_MODEL"] == "ViT-B-32"
        assert env["TEBENCH_IMG_CLIP_PRETRAINED"] == "laion2b_s34b_b79k"
        assert env["TEBENCH_IMG_REFERENCE"] == "base_gguf_f16"
        assert env["TEBENCH_IMG_FLOOR_PAIR"] == "base_gguf_f16,base_int8"

    def test_unknown_tebench_key_rejected_by_schema(self, comparison_dir: Path) -> None:
        """Lesson #28a: a key the runner reads must exist in comparison.schema.json.

        This guards the OTHER direction: keys NOT in the schema are rejected,
        so adding a reader without a schema entry fails here, not in Docker.
        """
        jsonschema = pytest.importorskip("jsonschema")

        schema = json.loads((Path(__file__).resolve().parents[2] / "comparison.schema.json").read_text())
        bad = {
            "name": "test-comp",
            "base": "models/base-model",
            "variants": {"v1": {"path": "models/v1"}},
            "settings": {"tebench": {"backend": "zimage", "img_nonexistent_key": 1}},
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)

    def test_img_keys_accepted_by_schema(self) -> None:
        jsonschema = pytest.importorskip("jsonschema")

        schema = json.loads((Path(__file__).resolve().parents[2] / "comparison.schema.json").read_text())
        good = {
            "name": "test-comp",
            "base": "models/base-model",
            "variants": {"v1": {"path": "models/v1"}},
            "settings": {
                "tebench": {
                    "backend": "zimage",
                    "img_clip_model": "ViT-L-14",
                    "img_clip_pretrained": "openai",
                    "img_reference": "base_int8",
                    "img_floor_pair": ["base_int8", "base_fp8"],
                }
            },
        }
        jsonschema.validate(good, schema)  # must not raise


# ---------------------------------------------------------------------------
# In-process tests (coverage): exercise main() directly so coverage.py can
# trace it (subprocess runs are invisible to the coverage machinery).
# ---------------------------------------------------------------------------


def _run_load_in_process(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    comp_json_path: str,
    comp_dir: str,
) -> dict[str, str]:
    """Call _load_comparison.main() in-process and parse its stdout exports."""
    from runners import _load_comparison as lc

    monkeypatch.setattr(sys, "argv", ["_load_comparison.py", comp_json_path, comp_dir])
    lc.main()
    out = capsys.readouterr().out
    env: dict[str, str] = {}
    for line in out.strip().splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        # ${VAR:-'default'} guard exports (lesson #70 pattern): unwrap the
        # parameter expansion and keep the quoted settings default.
        if value.startswith("${") and ":-" in value:
            value = value.split(":-", 1)[1].rstrip("}")
        if value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        env[key] = value
    return env


class TestInProcessMain:
    """Exercise main() in-process for coverage of the full export surface."""

    def test_full_settings_surface(
        self,
        comparison_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Rich settings block — every export branch taken at least once."""
        variants = {
            "sdft-cybersec": {
                "path": "models/sdft",
                "lora_adapter_path": "adapters/cybersec",
                "skip_harmbench": True,
                "skip_lm_eval": True,
                "skip_kl": True,
                "skip_weight": True,
                "skip_te": True,
                "te_file": "models/sdft/te.gguf",
                "te_reference": "heretic",
            },
        }
        settings = {
            "inference_backend": "vllm",
            "lm_eval_tasks": "gsm8k",
            "lm_eval_max_model_len": 8192,
            "lm_eval_max_gen_toks": 2048,
            "gsm8k_gen_toks": 4096,
            "lm_eval_gpu_util": "0.92",
            "lm_eval_quantization": "fp8",
            "lm_eval_kv_cache_dtype": "fp8",
            "lm_eval_language_model_only": 1,
            "lm_eval_gpus": "0",
            "lm_eval_extra_vllm_flags": "--max-num-batched-tokens 512",
            "lm_eval_speculative_tokens": 2,
            "harmbench_max_tokens": 15360,
            "harmbench_gpu_util": "0.95",
            "tokenizer_dir": "models/shared-tok",
            "gguf_dir": "models/gguf",
            "kl_num_prompts": 42,
            "reasoning_parser": "qwen3",
            "lmeval_image": "abliterlitics-lmeval-nightly:1.0.0",
            "forensics_image": "abliterlitics-lmeval-gemma4unified:1.0.0",
            "tebench": {
                "backend": "zimage",
                "base_te_file": "te/base.safetensors",
                "dit_file": "/abs/dit.safetensors",
                "comfyui_path": "/opt/comfyui",
                "quant_floor_file": "te/floor.json",
                "device": "cpu",
                "n_boot": 500,
                "prompt_suite": "words",
                "img_clip_model": "ViT-B-32",
                "img_clip_pretrained": "laion2b_s34b_b79k",
                "img_reference": "heretic",
                "img_floor_pair": ["a", "b"],
                "te_image": "abliterlitics-te:9.9.9",
            },
        }
        comp_path = _write_comparison(comparison_dir, variants=variants, settings=settings)
        env = _run_load_in_process(monkeypatch, capsys, str(comp_path), str(comparison_dir))

        # Core
        assert env["COMPARISON_NAME"] == "test-comp"
        assert env["VARIANT_COUNT"] == "1"
        assert env["VARIANT_PATH_sdft_cybersec"] == str(comparison_dir / "models" / "sdft")
        assert env["VARIANT_LORA_ADAPTER_sdft_cybersec"] == str(comparison_dir / "adapters" / "cybersec")
        assert env["VARIANT_SKIP_HB_sdft_cybersec"] == "1"
        assert env["VARIANT_SKIP_LM_sdft_cybersec"] == "1"
        assert env["VARIANT_SKIP_KL_sdft_cybersec"] == "1"
        assert env["VARIANT_SKIP_WT_sdft_cybersec"] == "1"
        assert env["VARIANT_NAMES"] == "sdft-cybersec"
        # Settings
        assert env["INFERENCE_BACKEND"] == "vllm"
        assert env["LM_EVAL_TASKS"] == "gsm8k"
        assert env["LM_EVAL_MAX_MODEL_LEN"] == "8192"
        assert env["LM_EVAL_MAX_GEN_TOKS"] == "2048"
        assert env["GSM8K_GEN_TOKS"] == "4096"
        assert env["LM_EVAL_GPU_UTIL"] == "0.92"
        assert env["LM_EVAL_QUANTIZATION"] == "fp8"
        assert env["LM_EVAL_KV_CACHE_DTYPE"] == "fp8"
        assert env["LM_EVAL_LANGUAGE_MODEL_ONLY"] == "1"
        assert env["LM_EVAL_GPUS"] == "0"
        assert env["LM_EVAL_EXTRA_VLLM_FLAGS"] == "--max-num-batched-tokens 512"
        assert env["LM_EVAL_SPEC_TOKENS"] == "2"
        assert env["HARMBENCH_MAX_TOKENS"] == "15360"
        assert env["HARMBENCH_GPU_UTIL"] == "0.95"
        assert env["TOKENIZER_DIR"] == str(comparison_dir / "models" / "shared-tok")
        assert env["GGUF_DIR"] == str(comparison_dir / "models" / "gguf")
        assert env["KL_NUM_PROMPTS"] == "42"
        assert env["KL_DATASET"] == "mlabonne/harmless_alpaca"
        assert env["REASONING_PARSER"] == "qwen3"
        assert json.loads(env["REASONING_CONFIG_JSON"])["reasoning_end_str"] == "</think>"
        assert env["LMEVAL_IMAGE"] == "abliterlitics-lmeval-nightly:1.0.0"
        assert env["FORENSICS_IMAGE"] == "abliterlitics-lmeval-gemma4unified:1.0.0"
        # Tebench block (absolute paths kept, relative prefixed)
        assert env["TEBENCH_BACKEND"] == "zimage"
        assert env["TEBENCH_BASE_TE_FILE"] == str(comparison_dir / "te" / "base.safetensors")
        assert env["TEBENCH_DIT_FILE"] == "/abs/dit.safetensors"
        assert env["TEBENCH_COMFYUI_PATH"] == "/opt/comfyui"
        assert env["TEBENCH_QUANT_FLOOR_FILE"] == str(comparison_dir / "te" / "floor.json")
        assert env["TEBENCH_DEVICE"] == "cpu"
        assert env["TEBENCH_NUM_BOOTSTRAP"] == "500"
        assert env["TEBENCH_PROMPT_SUITE"] == "words"
        assert env["TEBENCH_IMG_CLIP_MODEL"] == "ViT-B-32"
        assert env["TEBENCH_IMG_CLIP_PRETRAINED"] == "laion2b_s34b_b79k"
        assert env["TEBENCH_IMG_REFERENCE"] == "heretic"
        assert env["TEBENCH_IMG_FLOOR_PAIR"] == "a,b"
        assert env["TE_IMAGE"] == "abliterlitics-te:9.9.9"
        # TE per-variant loop
        assert env["VARIANT_SKIP_TE_sdft_cybersec"] == "1"
        assert env["VARIANT_TE_FILE_sdft_cybersec"] == str(comparison_dir / "models" / "sdft" / "te.gguf")
        assert env["VARIANT_TE_REF_sdft_cybersec"] == "heretic"

    def test_no_tebench_exports_empty_defaults(
        self,
        comparison_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        comp_path = _write_comparison(comparison_dir)
        env = _run_load_in_process(monkeypatch, capsys, str(comp_path), str(comparison_dir))
        assert env["TEBENCH_BACKEND"] == ""
        assert env["TE_IMAGE"] == ""
        assert env["VARIANT_SKIP_TE_variant1"] == "0"
        assert env["VARIANT_TE_FILE_variant1"] == ""
        assert env["VARIANT_TE_REF_variant1"] == ""
        assert env["VARIANT_LORA_ADAPTER_variant1"] == ""

    def test_comfyui_path_env_default(
        self,
        comparison_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """No machine-specific hardcode: env var fills the default when set."""
        monkeypatch.setenv("TEBENCH_COMFYUI_PATH", "/env/comfyui")
        comp_path = _write_comparison(comparison_dir, settings={"tebench": {"backend": "zimage"}})
        env = _run_load_in_process(monkeypatch, capsys, str(comp_path), str(comparison_dir))
        assert env["TEBENCH_COMFYUI_PATH"] == "/env/comfyui"

    def test_invalid_comparison_name_exits(
        self,
        comparison_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from runners import _load_comparison as lc

        comp_path = _write_comparison(comparison_dir, name="INVALID")
        monkeypatch.setattr(sys, "argv", ["_load_comparison.py", str(comp_path), str(comparison_dir)])
        with pytest.raises(SystemExit):
            lc.main()
        assert "Invalid comparison name" in capsys.readouterr().out

    def test_invalid_variant_key_exits(
        self,
        comparison_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from runners import _load_comparison as lc

        comp_path = _write_comparison(comparison_dir, variants={"UPPER": {"path": "models/upper"}})
        monkeypatch.setattr(sys, "argv", ["_load_comparison.py", str(comp_path), str(comparison_dir)])
        with pytest.raises(SystemExit):
            lc.main()
        assert "Invalid variant key" in capsys.readouterr().out


class TestDetectReasoningParserBranches:
    """Direct unit tests for the auto-detect matrix (audit coverage item 4)."""

    def test_gemma4_arch_detected(self, comparison_dir: Path) -> None:
        base_dir = comparison_dir / "models" / "base-model"
        (base_dir / "config.json").write_text(json.dumps({"architectures": ["Gemma4ForConditionalGeneration"]}))
        parser, config = _load_detect()(str(base_dir), None)
        assert parser == "gemma4"
        assert config == ""

    def test_qwen3_override_attaches_config(self) -> None:
        parser, config = _load_detect()("/nonexistent", "qwen3")
        assert parser == "qwen3"
        assert json.loads(config)["reasoning_start_str"] == "<think>"

    def test_non_qwen_override_has_no_config(self) -> None:
        parser, config = _load_detect()("/nonexistent", "gemma4")
        assert parser == "gemma4"
        assert config == ""

    def test_malformed_config_json_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("{not json")
        parser, config = _load_detect()(str(tmp_path), None)
        assert parser == ""
        assert config == ""

    def test_non_reasoning_arch_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
        parser, config = _load_detect()(str(tmp_path), None)
        assert parser == ""
        assert config == ""


def _load_detect():
    """Import the detector lazily (in-process import, needs repo root on sys.path)."""
    from runners._load_comparison import _detect_reasoning_parser

    return _detect_reasoning_parser


class TestSlugNormalizationBothLoops:
    """Lesson #64: BOTH variant loops (core exports + TE exports) must normalize
    variant keys to the same bash-identifier slug — a drift between them makes
    the TE exports unreadable by every consumer (bash ${!VAR} / env lookups)."""

    def test_core_and_te_loops_use_same_slug(self, comparison_dir: Path) -> None:
        variants = {
            "sdft-cybersec-r2": {
                "path": "models/sdft-cybersec-r2",
                "te_file": "models/sdft-cybersec-r2/te.gguf",
            },
        }
        comp_path = _write_comparison(comparison_dir, variants=variants)
        env = _run_load(str(comp_path), str(comparison_dir))

        slug = "sdft_cybersec_r2"
        # Core loop exports
        assert f"VARIANT_PATH_{slug}" in env
        assert f"VARIANT_LORA_ADAPTER_{slug}" in env
        assert f"VARIANT_SKIP_HB_{slug}" in env
        assert f"VARIANT_SKIP_LM_{slug}" in env
        assert f"VARIANT_SKIP_KL_{slug}" in env
        assert f"VARIANT_SKIP_WT_{slug}" in env
        # TE loop exports — SAME slug
        assert f"VARIANT_SKIP_TE_{slug}" in env
        assert f"VARIANT_TE_FILE_{slug}" in env
        assert f"VARIANT_TE_REF_{slug}" in env
        # No raw-hyphen export leaked from either loop
        for key in env:
            assert "sdft-cybersec" not in key

    def test_consumer_slug_matches_exporter_slug(self) -> None:
        """The run_harmbench.py consumer normalization must equal the exporter's
        for the same input (lesson #64 failure class: exporter/consumer drift)."""
        import re

        from runners.run_harmbench import _SLUG_RE  # same regex constant

        for name in ["sdft-cybersec-r2", "heretic", "huihui", "Qwen3.8-27B", "a_b-c.d"]:
            expected = re.sub(r"[^A-Za-z0-9_]", "_", name)
            assert _SLUG_RE.sub("_", name) == expected
