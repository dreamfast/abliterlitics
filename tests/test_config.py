"""Tests for src/config.py — ComparisonConfig loading, validation, and path resolution."""
from __future__ import annotations

import json
import os
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
    @pytest.mark.parametrize("valid", [
        "abc", "test-comp", "my_model", "a1b2", "qwen35-2b",
        "abc123", "a-b_c", "x",
    ])
    def test_valid_slugs(self, valid):
        assert SLUG_PATTERN.match(valid), f"Expected {valid!r} to match"

    @pytest.mark.parametrize("invalid", [
        "ABC", "Test Comp", "a.b", " leading", "trailing ",
        "", " space", "UPPER", "-dash-first", "_underscore_first",
        "has.dot", "has space",
    ])
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
        assert cfg.kl_results_dir(base) == Path("/tmp/results/test-comp/kl")

    def test_get_variant(self, tmp_comparison_dir):
        cfg = ComparisonConfig.from_dir(tmp_comparison_dir)
        v = cfg.get_variant("variant_a")
        assert v.name == "variant_a"

    def test_get_variant_missing_raises(self, tmp_comparison_dir):
        cfg = ComparisonConfig.from_dir(tmp_comparison_dir)
        with pytest.raises(KeyError, match="not_found"):
            cfg.get_variant("not_found")

    def test_settings_defaults(self, tmp_comparison_dir):
        cfg = ComparisonConfig.from_dir(tmp_comparison_dir)
        assert "mmlu" in cfg.lm_eval_tasks
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
        with pytest.raises(ValueError, match='reserved'):
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
