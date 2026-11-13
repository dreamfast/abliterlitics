"""Tests for src/config.py — ComparisonConfig loading, validation, and path resolution."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from jsonschema import ValidationError as SchemaValidationError

from src.config import (
    SLUG_PATTERN,
    ComparisonConfig,
    _compute_model_size,
    _detect_architecture,
    _resolve_model_path,
    _validate_model_dir,
    load_schema,
    make_metadata,
)

# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------


class TestSchema:
    def test_load_schema_returns_dict(self):
        schema = load_schema()
        assert isinstance(schema, dict)
        assert schema["title"] == "Abliterlitics Comparison Configuration"

    def test_schema_has_required_fields(self):
        schema = load_schema()
        assert "name" in schema["properties"]
        assert "base" in schema["properties"]
        assert "variants" in schema["properties"]


# ---------------------------------------------------------------------------
# Slug validation
# ---------------------------------------------------------------------------


class TestSlugPattern:
    @pytest.mark.parametrize(
        "valid",
        [
            "abc",
            "test-comp",
            "my_model",
            "a1b2",
            "qwen35-2b",
            "abc123",
            "a-b_c",
            "x",
        ],
    )
    def test_valid_slugs(self, valid):
        assert SLUG_PATTERN.match(valid), f"Expected {valid!r} to match"

    @pytest.mark.parametrize(
        "invalid",
        [
            "ABC",
            "Test Comp",
            "a.b",
            " leading",
            "trailing ",
            "",
            " space",
            "UPPER",
            "-dash-first",
            "_underscore_first",
            "has.dot",
            "has space",
        ],
    )
    def test_invalid_slugs(self, invalid):
        assert not SLUG_PATTERN.match(invalid), f"Expected {invalid!r} to NOT match"


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


class TestResolveModelPath:
    def test_relative_path_resolved_against_comparison_dir(self, tmp_path):
        comp_dir = tmp_path / "my-comparison"
        comp_dir.mkdir()
        result = _resolve_model_path("subdir/model", comp_dir)
        assert result == (comp_dir / "subdir" / "model").resolve()

    def test_absolute_path_used_as_is(self, tmp_path):
        abs_path = "/some/absolute/path"
        result = _resolve_model_path(abs_path, tmp_path)
        assert result == Path(abs_path).resolve()


# ---------------------------------------------------------------------------
# Model directory validation
# ---------------------------------------------------------------------------


class TestValidateModelDir:
    def test_valid_dir_with_config_json(self, tmp_model_dir):
        _validate_model_dir(tmp_model_dir)  # should not raise

    def test_missing_directory(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            _validate_model_dir(tmp_path / "nonexistent")

    def test_dir_without_config_json(self, tmp_path):
        empty_dir = tmp_path / "empty_model"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="missing config.json"):
            _validate_model_dir(empty_dir)


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------


class TestDetectArchitecture:
    def test_qwen3_detected(self, tmp_model_dir):
        arch = _detect_architecture(tmp_model_dir)
        assert arch == "qwen3"

    def test_qwen35_detected(self, tmp_model_dir_qwen35):
        arch = _detect_architecture(tmp_model_dir_qwen35)
        assert arch == "qwen3.5"

    def test_glm_detected(self, tmp_model_dir_glm):
        arch = _detect_architecture(tmp_model_dir_glm)
        assert arch == "glm"

    def test_gemma4_detected(self, tmp_model_dir_gemma4):
        arch = _detect_architecture(tmp_model_dir_gemma4)
        assert arch == "gemma4"

    def test_gemma4_config_json_fallback(self, tmp_path):
        """Verify config.json model_type fallback works for Gemma4."""
        from tests.conftest import create_mock_model

        model_dir = create_mock_model(tmp_path / "model", arch="gemma4", seed=42)
        arch = _detect_architecture(model_dir)
        assert arch == "gemma4"


# ---------------------------------------------------------------------------
# Model size computation
# ---------------------------------------------------------------------------


class TestComputeModelSize:
    def test_returns_positive_float(self, tmp_model_dir):
        size = _compute_model_size(tmp_model_dir)
        assert isinstance(size, float)
        assert size > 0

    def test_empty_dir_returns_zero(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert _compute_model_size(empty) == 0.0


# ---------------------------------------------------------------------------
# ComparisonConfig.from_dir — happy path
# ---------------------------------------------------------------------------


class TestComparisonConfigLoad:
    def test_load_valid_comparison(self, tmp_comparison_dir):
        cfg = ComparisonConfig.from_dir(tmp_comparison_dir)
        assert cfg.name == "test-comp"
        assert cfg.base_path.is_dir()
        assert len(cfg.variants) == 2
        assert cfg.variants[0].name == "variant_a"
        assert cfg.variants[1].name == "variant_b"
        assert cfg.architecture != ""
        assert cfg.model_size_gb > 0

    def test_load_from_json_file_directly(self, tmp_comparison_dir):
        cfg = ComparisonConfig.from_dir(tmp_comparison_dir / "comparison.json")
        assert cfg.name == "test-comp"

    def test_results_dir_paths(self, tmp_comparison_dir):
        cfg = ComparisonConfig.from_dir(tmp_comparison_dir)
        base = Path("/tmp/results")
        assert cfg.results_dir(base) == Path("/tmp/results/test-comp")
        assert cfg.weight_results_dir(base) == Path("/tmp/results/test-comp/weight")

    def test_settings_defaults(self, tmp_comparison_dir):
        cfg = ComparisonConfig.from_dir(tmp_comparison_dir)
        # v2 default suite (AGENTS.md lesson #9) — mirrors the runner default
        assert "leaderboard_mmlu_pro" in cfg.lm_eval_tasks
        assert "gsm8k" in cfg.lm_eval_tasks
        assert cfg.lm_eval_max_gen_toks == 4096
        assert cfg.lm_eval_max_model_len == 16384
        assert cfg.kl_num_prompts == 100
        assert cfg.inference_backend == "auto"


# ---------------------------------------------------------------------------
# ComparisonConfig.from_dir — error cases
# ---------------------------------------------------------------------------


class TestComparisonConfigErrors:
    def test_missing_comparison_json(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="comparison.json not found"):
            ComparisonConfig.from_dir(tmp_path / "nonexistent")

    def test_invalid_name_uppercase(self, tmp_comparison_dir):
        comp_file = tmp_comparison_dir / "comparison.json"
        data = json.loads(comp_file.read_text())
        data["name"] = "InvalidName"
        comp_file.write_text(json.dumps(data))
        # JSON Schema catches pattern violation (name doesn't match slug pattern)
        with pytest.raises(SchemaValidationError):
            ComparisonConfig.from_dir(tmp_comparison_dir)

    def test_invalid_name_with_spaces(self, tmp_comparison_dir):
        comp_file = tmp_comparison_dir / "comparison.json"
        data = json.loads(comp_file.read_text())
        data["name"] = "has spaces"
        comp_file.write_text(json.dumps(data))
        with pytest.raises(SchemaValidationError):
            ComparisonConfig.from_dir(tmp_comparison_dir)

    def test_variant_key_base_reserved(self, tmp_comparison_dir):
        comp_file = tmp_comparison_dir / "comparison.json"
        data = json.loads(comp_file.read_text())
        data["variants"]["base"] = {"path": "base"}
        comp_file.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="reserved"):
            ComparisonConfig.from_dir(tmp_comparison_dir)

    def test_variant_key_invalid_slug(self, tmp_comparison_dir):
        comp_file = tmp_comparison_dir / "comparison.json"
        data = json.loads(comp_file.read_text())
        data["variants"]["UPPER"] = {"path": "base"}
        comp_file.write_text(json.dumps(data))
        # JSON Schema patternProperties requires slug pattern for keys
        with pytest.raises(SchemaValidationError):
            ComparisonConfig.from_dir(tmp_comparison_dir)

    def test_missing_base_directory(self, tmp_comparison_dir):
        comp_file = tmp_comparison_dir / "comparison.json"
        data = json.loads(comp_file.read_text())
        data["base"] = "nonexistent_dir"
        comp_file.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="does not exist"):
            ComparisonConfig.from_dir(tmp_comparison_dir)

    def test_missing_variant_directory(self, tmp_comparison_dir):
        comp_file = tmp_comparison_dir / "comparison.json"
        data = json.loads(comp_file.read_text())
        data["variants"]["variant_a"]["path"] = "nonexistent_dir"
        comp_file.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="does not exist"):
            ComparisonConfig.from_dir(tmp_comparison_dir)

    def test_empty_variants_rejected_by_schema(self, tmp_comparison_dir):
        comp_file = tmp_comparison_dir / "comparison.json"
        data = json.loads(comp_file.read_text())
        data["variants"] = {}
        comp_file.write_text(json.dumps(data))
        with pytest.raises(SchemaValidationError):
            ComparisonConfig.from_dir(tmp_comparison_dir)

    def test_relative_paths_resolved(self, tmp_comparison_dir):
        """Verify that relative model paths resolve against comparison dir."""
        cfg = ComparisonConfig.from_dir(tmp_comparison_dir)
        assert cfg.base_path.is_absolute()
        for v in cfg.variants:
            assert v.path.is_absolute()


# ---------------------------------------------------------------------------
# make_metadata
# ---------------------------------------------------------------------------


class TestMakeMetadata:
    def test_metadata_structure(self, tmp_comparison_dir):
        cfg = ComparisonConfig.from_dir(tmp_comparison_dir)
        meta = make_metadata(cfg, variant="variant_a")
        assert meta["tool"] == "abliterlitics"
        assert meta["version"] == "1.0.0"
        assert meta["results_version"] == 1
        assert meta["comparison_name"] == "test-comp"
        assert meta["variant"] == "variant_a"
        assert "timestamp" in meta
        assert "architecture" in meta


# ---------------------------------------------------------------------------
# Lockstep guards: config.py vs runners/_load_comparison.py vs schema
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


def _get_call_keys(source: str, receiver: str) -> dict[str, object]:
    """Collect {key: literal_default} for ``<receiver>.get('key', default)`` calls.

    Read-only AST parse — no import or execution of the parsed module.
    Keys whose default argument is not a plain literal (variable, absent,
    or complex expression) map to None.
    """
    tree = ast.parse(source)
    defaults: dict[str, object] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "get":
            continue
        recv = node.func.value
        if not (isinstance(recv, ast.Name) and recv.id == receiver):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        default = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else None
        defaults[node.args[0].value] = default
    return defaults


def _loader_default_tasks(loader_source: str) -> str:
    """Extract the default v2 task list built into ``default_tasks`` in the loader."""
    tree = ast.parse(loader_source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "default_tasks" for t in node.targets
        ):
            joined = node.value
            assert isinstance(joined, ast.Call), "loader default_tasks is no longer ','.join([...])"
            parts = [elt.value for elt in joined.args[0].elts]  # type: ignore[attr-defined]
            assert all(isinstance(p, str) for p in parts)
            return ",".join(parts)  # type: ignore[arg-type]
    raise AssertionError("default_tasks assignment not found in _load_comparison.py")


class TestLoaderSchemaLockstep:
    """AGENTS.md lesson #28a regression guards (audit .cache/audit/src-top.md).

    runners/_load_comparison.py reads comparison.json settings/variant keys
    that comparison.schema.json must declare: the schema has
    additionalProperties: false, so an undeclared key makes
    ComparisonConfig.from_dir raise ValidationError and every src/weight/*
    script crash — the lesson-#28a failure signature. config.py duplicates a
    subset of the loader's settings parsing; its keys and defaults must not
    drift from the runner, which is the source of truth.
    """

    @staticmethod
    def _loader_source() -> str:
        return (REPO_ROOT / "runners" / "_load_comparison.py").read_text()

    @staticmethod
    def _config_source() -> str:
        return (REPO_ROOT / "src" / "config.py").read_text()

    @staticmethod
    def _schema_settings_properties() -> set[str]:
        return set(load_schema()["properties"]["settings"]["properties"])

    @staticmethod
    def _schema_variant_properties() -> set[str]:
        variants = load_schema()["properties"]["variants"]
        props: set[str] = set()
        for pattern_node in variants.get("patternProperties", {}).values():
            props |= set(pattern_node.get("properties", {}))
        return props

    def test_loader_settings_keys_declared_in_schema(self):
        loader_keys = set(_get_call_keys(self._loader_source(), "settings"))
        assert loader_keys, "parsing regression: no settings.get() calls found in loader"
        missing = loader_keys - self._schema_settings_properties()
        assert not missing, (
            "lesson #28a: settings keys read by _load_comparison.py but missing "
            f"from comparison.schema.json: {sorted(missing)}"
        )

    def test_loader_tebench_keys_declared_in_schema(self):
        loader_keys = set(_get_call_keys(self._loader_source(), "tebench"))
        assert loader_keys, "parsing regression: no tebench.get() calls found in loader"
        schema_te = set(load_schema()["properties"]["settings"]["properties"]["tebench"]["properties"])
        missing = loader_keys - schema_te
        assert not missing, f"tebench keys read by _load_comparison.py but missing from schema: {sorted(missing)}"

    def test_loader_variant_keys_declared_in_schema(self):
        """Includes lora_adapter_path — the schema fix is landed upstream;
        if this key regresses out of the schema the suite goes red here."""
        loader_keys = set(_get_call_keys(self._loader_source(), "v"))
        assert loader_keys, "parsing regression: no v.get() calls found in loader"
        missing = loader_keys - self._schema_variant_properties()
        assert not missing, (
            "lesson #28a: variant keys read by _load_comparison.py but missing "
            f"from comparison.schema.json: {sorted(missing)}"
        )

    def test_config_settings_keys_are_schema_declared_and_loader_read(self):
        config_keys = set(_get_call_keys(self._config_source(), "settings"))
        assert config_keys, "parsing regression: no settings.get() calls found in config.py"
        # config.py validates parsed data against the schema: every key it
        # reads must be schema-declared ...
        missing_schema = config_keys - self._schema_settings_properties()
        assert not missing_schema, f"config.py reads undeclared settings keys: {sorted(missing_schema)}"
        # ... and must stay a subset of what the runner consumes, or the two
        # loaders' defaults can silently diverge (dual-loader divergence).
        loader_keys = set(_get_call_keys(self._loader_source(), "settings"))
        extra = config_keys - loader_keys
        assert not extra, f"config.py reads settings keys the loader ignores: {sorted(extra)}"

    def test_config_defaults_match_loader_defaults(self):
        loader_defs = _get_call_keys(self._loader_source(), "settings")
        config_defs = _get_call_keys(self._config_source(), "settings")
        shared = set(loader_defs) & set(config_defs)
        # Keys whose defaults are not plain literals in both files (e.g. the
        # loader's lm_eval_tasks default is built from a list literal) are
        # compared in dedicated tests below.
        literal_shared = {k for k in shared if loader_defs[k] is not None and config_defs[k] is not None}
        mismatched = {
            k: {"config.py": config_defs[k], "loader": loader_defs[k]}
            for k in literal_shared
            if config_defs[k] != loader_defs[k]
        }
        assert not mismatched, f"config.py defaults diverge from runner truth: {mismatched}"

    def test_config_lm_eval_tasks_default_matches_loader_v2_suite(self):
        config_defs = _get_call_keys(self._config_source(), "settings")
        expected = _loader_default_tasks(self._loader_source())
        assert config_defs["lm_eval_tasks"] == expected, (
            "config.py lm_eval_tasks default diverges from the runner's v2 suite: "
            f"config.py={config_defs['lm_eval_tasks']!r} loader={expected!r}"
        )
