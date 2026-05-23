"""
Core configuration: loads comparison.json, validates schema and paths,
auto-detects architecture, resolves result paths.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonschema

from src import RESULTS_VERSION, __version__

log = logging.getLogger(__name__)

# Slug pattern for comparison names and variant keys
SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass
class VariantConfig:
    """Configuration for a single variant model."""

    name: str  # Slug key (e.g. "heretic")
    path: Path  # Resolved absolute path to model dir
    display_name: str  # Human-readable name
    skip_kl: bool = False
    skip_lm_eval: bool = False
    skip_harmbench: bool = False
    skip_weight: bool = False


@dataclass
class ComparisonConfig:
    """Full configuration loaded from comparison.json."""

    name: str  # Slug name
    comparison_dir: Path  # Directory containing comparison.json
    base_path: Path  # Absolute path to base model dir
    variants: list[VariantConfig]  # List of variant configs
    gguf_dir: Path | None
    tokenizer_dir: Path | None
    lm_eval_tasks: str
    inference_backend: str = "auto"
    lm_eval_max_gen_toks: int = 2048
    lm_eval_max_model_len: int = 4096
    harmbench_max_tokens: int = 2048
    kl_num_prompts: int = 100
    kl_dataset: str = "mlabonne/harmless_alpaca"
    architecture: str = ""  # Auto-detected
    model_size_gb: float = 0.0  # Auto-computed

    @classmethod
    def from_dir(cls, comparison_dir: Path) -> ComparisonConfig:
        """Load comparison.json from a directory, validate, resolve paths."""
        comparison_dir = Path(comparison_dir).resolve()

        # Try loading comparison.json directly if it's a file
        if comparison_dir.is_file() and comparison_dir.name == "comparison.json":
            comp_file = comparison_dir
            comparison_dir = comparison_dir.parent
        else:
            comp_file = comparison_dir / "comparison.json"

        if not comp_file.exists():
            raise FileNotFoundError(f"comparison.json not found: {comp_file}")

        with open(comp_file) as f:
            data = json.load(f)

        # Validate against JSON Schema
        schema = load_schema()
        jsonschema.validate(data, schema)

        # Validate name slug
        name = data["name"]
        if not SLUG_PATTERN.match(name):
            raise ValueError(f"Invalid comparison name: {name!r}. Must match {SLUG_PATTERN.pattern}")

        # Resolve base path
        base_str = data["base"]
        base_path = _resolve_model_path(base_str, comparison_dir)
        _validate_model_dir(base_path)

        # Build variant configs
        variants: list[VariantConfig] = []
        for key, vdata in data["variants"].items():
            if key == "base":
                raise ValueError('Variant key "base" is reserved')
            if not SLUG_PATTERN.match(key):
                raise ValueError(f"Invalid variant key: {key!r}. Must match {SLUG_PATTERN.pattern}")

            vpath = _resolve_model_path(vdata["path"], comparison_dir)
            _validate_model_dir(vpath)

            variants.append(
                VariantConfig(
                    name=key,
                    path=vpath,
                    display_name=vdata.get("display_name", key),
                    skip_kl=vdata.get("skip_kl", False),
                    skip_lm_eval=vdata.get("skip_lm_eval", False),
                    skip_harmbench=vdata.get("skip_harmbench", False),
                    skip_weight=vdata.get("skip_weight", False),
                )
            )

        # Settings with defaults
        settings = data.get("settings", {})

        gguf_dir = None
        if settings.get("gguf_dir"):
            raw_gguf = settings["gguf_dir"]
            gguf_path = Path(raw_gguf)
            if not gguf_path.is_absolute():
                gguf_path = comparison_dir / gguf_path
            gguf_dir = gguf_path.resolve()

        tokenizer_dir = None
        if settings.get("tokenizer_dir"):
            raw_tok = settings["tokenizer_dir"]
            tok_path = Path(raw_tok)
            if not tok_path.is_absolute():
                tok_path = comparison_dir / tok_path
            tokenizer_dir = tok_path.resolve()

        obj = cls(
            name=name,
            comparison_dir=comparison_dir,
            base_path=base_path,
            variants=variants,
            gguf_dir=gguf_dir,
            tokenizer_dir=tokenizer_dir,
            lm_eval_tasks=settings.get(
                "lm_eval_tasks",
                "mmlu,gsm8k,hellaswag,arc_challenge,winogrande,truthfulqa,piqa,lambada_openai",
            ),
            inference_backend=settings.get("inference_backend", "auto"),
            lm_eval_max_gen_toks=settings.get("lm_eval_max_gen_toks", 2048),
            lm_eval_max_model_len=settings.get("lm_eval_max_model_len", 4096),
            harmbench_max_tokens=settings.get("harmbench_max_tokens", 2048),
            kl_num_prompts=settings.get("kl_num_prompts", 100),
            kl_dataset=settings.get("kl_dataset", "mlabonne/harmless_alpaca"),
        )

        # Auto-detect architecture
        obj.architecture = _detect_architecture(base_path)
        log.info("Detected architecture: %s", obj.architecture)

        # Compute model size
        obj.model_size_gb = _compute_model_size(base_path)
        log.info("Model size: %.1f GB", obj.model_size_gb)

        return obj

    def results_dir(self, base_results_dir: Path) -> Path:
        """Return per-comparison results directory."""
        return base_results_dir / self.name

    def weight_results_dir(self, base_results_dir: Path) -> Path:
        return self.results_dir(base_results_dir) / "weight"

    def kl_results_dir(self, base_results_dir: Path) -> Path:
        return self.results_dir(base_results_dir) / "kl"

    def lm_eval_results_dir(self, base_results_dir: Path) -> Path:
        return self.results_dir(base_results_dir) / "lm_eval"

    def harmbench_results_dir(self, base_results_dir: Path) -> Path:
        return self.results_dir(base_results_dir) / "harmbench"

    def get_variant(self, name: str) -> VariantConfig:
        """Get a variant config by name."""
        for v in self.variants:
            if v.name == name:
                return v
        raise KeyError(f"Variant not found: {name!r}")


def _resolve_model_path(path_str: str, comparison_dir: Path) -> Path:
    """Resolve a model path (relative or absolute) against comparison directory.

    Relative paths are resolved against comparison_dir.
    Absolute paths are used as-is.
    All paths are resolved via Path.resolve() to eliminate symlinks.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    return (comparison_dir / p).resolve()


def _validate_model_dir(model_path: Path) -> None:
    """Verify a model directory exists and contains config.json."""
    if not model_path.is_dir():
        raise ValueError(f"Model directory does not exist: {model_path}")
    config_file = model_path / "config.json"
    if not config_file.exists():
        raise ValueError(f"Model directory missing config.json: {model_path}")


def _detect_architecture(model_path: Path) -> str:
    """Detect architecture from safetensors key patterns.

    This is a lightweight detection for config.py. The full detection
    (with ArchitectureConfig dataclass) is in model_config.py.

    Priority: safetensors key scan > model_type from config.json
    """
    # First try: scan safetensors keys (most reliable)
    try:
        from safetensors import safe_open

        has_language_model = False
        has_experts = False
        has_mamba = False
        has_gemma4 = False

        for f in sorted(model_path.glob("*.safetensors")):
            with safe_open(str(f), framework="pt", device="cpu") as sf:  # type: ignore[no-untyped-call]
                keys = sf.keys()
            for k in keys:
                if "language_model" in k:
                    has_language_model = True
                if ".experts." in k:
                    has_experts = True
                if "linear_attn" in k or "conv1d" in k:
                    has_mamba = True
                if "layer_scalar" in k or "per_layer_projection" in k:
                    has_gemma4 = True
            # Check first shard only for fast detection
            break

        if has_gemma4:
            return "gemma4"
        if has_language_model:
            return "qwen3.5"
        if has_experts:
            return "glm"
        if has_mamba:
            return "qwen3.5"
    except Exception:
        pass  # Fall through to config.json detection

    # Second try: config.json model_type
    config_file = model_path / "config.json"
    with open(config_file) as cf:
        config = json.load(cf)

    model_type = config.get("model_type", "")
    if "gemma4" in model_type.lower():
        return "gemma4"
    if "glm" in model_type.lower():
        return "glm"

    return model_type or "unknown"


def _compute_model_size(model_path: Path) -> float:
    """Sum safetensors file sizes in directory, return GB."""
    total_bytes = 0
    for f in model_path.glob("*.safetensors"):
        total_bytes += f.stat().st_size
    return total_bytes / (1024**3)


def load_schema() -> dict[str, Any]:
    """Load comparison.schema.json from package directory."""
    schema_path = Path(__file__).parent.parent / "comparison.schema.json"
    with open(schema_path) as f:
        result: dict[str, Any] = json.load(f)
        return result


def make_metadata(config: ComparisonConfig, variant: str = "") -> dict[str, Any]:
    """Create standard metadata dict for result JSONs."""
    return {
        "tool": "abliterlitics",
        "version": __version__,
        "results_version": RESULTS_VERSION,
        "comparison_name": config.name,
        "variant": variant,
        "architecture": config.architecture,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
