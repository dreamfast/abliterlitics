"""Shared backend infrastructure for text-encoder analysis.

Provides the abstract base class all model-specific backends inherit from,
plus the hook manager and fake-quantizer utilities they need.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from src.text_encoder.artifact import ConditioningArtifact, Stage, StreamSpec

log = logging.getLogger(__name__)


class HookManager:
    """Manages forward hooks to capture intermediate activations.

    Registers hooks on specified modules, collects their outputs into a
    dict keyed by stage name, and cleans up after use.
    """

    def __init__(self) -> None:
        self._hooks: list[Any] = []
        self._captures: dict[str, dict[str, torch.Tensor]] = {}

    def register(self, stage: str, stream: str, module: nn.Module) -> None:
        """Register a forward hook on a module.

        Args:
            stage: Pipeline stage name (e.g. "projected", "connector_out").
            stream: Stream name within the stage.
            module: The PyTorch module to hook.
        """

        def hook_fn(_module: nn.Module, _input: Any, output: Any) -> None:
            # Detach, clone, and move to CPU immediately to free GPU memory
            if isinstance(output, tuple):
                tensor = output[0]
            elif isinstance(output, torch.Tensor):
                tensor = output
            else:
                log.warning("Hook captured non-tensor output from %s: %s", stream, type(output))
                return
            self._captures.setdefault(stage, {})[stream] = tensor.detach().clone().float().cpu()

        handle = module.register_forward_hook(hook_fn)
        self._hooks.append(handle)

    def get_captures(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return all captured activations."""
        return self._captures

    def clear(self) -> None:
        """Remove all hooks and clear captures."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._captures.clear()


def fake_quantize_weights(model: nn.Module, quant_type: str = "fp8") -> nn.Module:
    """Apply fake quantization to model weights in-place.

    Rounds each weight tensor to the target precision, then casts back to
    the original dtype. This simulates the information loss of real
    quantization without requiring fp8/int8 kernels. Used for computing
    the quantization noise floor.

    Args:
        model: The PyTorch model to quantize (modified in-place).
        quant_type: One of "fp8" (e4m3), "int8", "fp16".

    Returns:
        The same model (modified in-place).
    """
    with torch.no_grad():
        for _name, param in model.named_parameters():
            if param.dtype not in (torch.float32, torch.bfloat16, torch.float16):
                continue

            orig_dtype = param.dtype
            fp32_data = param.data.float()

            if quant_type == "fp8":
                # FP8 E4M3: range [-448, 448], 3 mantissa bits
                # Simulate by quantizing to fp8 representable values
                fp8_max = 448.0
                fp8_scale = fp8_max / fp32_data.abs().max().clamp(min=1e-10)
                # Round to fp8 precision (8 levels per binade for e4m3)
                scaled = fp32_data * fp8_scale
                # Clamp to fp8 range
                scaled = scaled.clamp(-fp8_max, fp8_max)
                # Round to nearest representable fp8 value
                # E4M3 has 2^3 = 8 mantissa levels per exponent
                # Simple approximation: round to log-scale
                rounded = _round_to_fp8(scaled)
                quantized = rounded / fp8_scale

            elif quant_type == "int8":
                # Per-tensor symmetric int8
                scale = fp32_data.abs().max() / 127.0
                scale = scale.clamp(min=1e-10)
                quantized = torch.round(fp32_data / scale) * scale

            elif quant_type == "fp16":
                # Just cast to fp16 and back
                quantized = fp32_data.half().float()

            else:
                raise ValueError(f"Unknown quant_type: {quant_type}")

            param.data = quantized.to(orig_dtype)

    log.info("Fake-quantized model weights to %s", quant_type)
    return model


def _round_to_fp8(x: torch.Tensor) -> torch.Tensor:
    """Approximate rounding to FP8 E4M3 representable values.

    This is a simplified simulation — real fp8 hardware uses exact
    representable values. For noise-floor measurement, this approximation
    is sufficient as it captures the magnitude of information loss.
    """
    # Use the sign + magnitude approach
    sign = torch.sign(x)
    magnitude = x.abs()

    # Avoid log(0)
    safe_mag = magnitude.clamp(min=1e-38)

    # FP8 E4M3: exponent bias 7, 3 mantissa bits
    # log2 of magnitude, round exponent, then quantize mantissa
    log2_mag = torch.log2(safe_mag)
    exponent = torch.floor(log2_mag)
    # 3 mantissa bits = 8 levels
    mantissa_frac = torch.pow(2.0, log2_mag - exponent) - 1.0
    quantized_mantissa = torch.round(mantissa_frac * 8.0) / 8.0
    result = sign * torch.pow(2.0, exponent) * (1.0 + quantized_mantissa)

    # Zero stays zero
    result = torch.where(magnitude < 1e-38, torch.zeros_like(result), result)
    return result


class BaseBackend(ABC):
    """Abstract base class for text-encoder backends.

    Each backend knows how to load a specific text encoder (Gemma 3, Qwen3-VL,
    etc.), run a forward pass with hooks to capture intermediate activations
    at multiple pipeline stages, and produce ConditioningArtifacts.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g. 'ltx2', 'h3')."""
        ...

    @property
    @abstractmethod
    def streams(self) -> list[StreamSpec]:
        """Declared stream specifications this backend produces."""
        ...

    @abstractmethod
    def load(self, ckpt: str, dtype: str = "bf16", device: str = "cpu") -> None:
        """Load the text encoder model.

        Args:
            ckpt: Path or repo ID of the checkpoint.
            dtype: Data type ("bf16", "fp32", "fp16").
            device: "cpu" or "cuda".
        """
        ...

    @abstractmethod
    def encode(
        self,
        prompts: list[str],
        categories: list[str],
        stages: list[Stage] | None = None,
    ) -> list[ConditioningArtifact]:
        """Encode prompts and capture conditioning at each stage.

        One forward pass captures all stages via hooks.

        Args:
            prompts: List of prompt strings.
            categories: Category per prompt (aligned).
            stages: Stages to capture (None = all declared).

        Returns:
            One ConditioningArtifact per stage.
        """
        ...

    @abstractmethod
    def tokenize(self, prompts: list[str]) -> dict[str, Any]:
        """Tokenize prompts (for alignment assertion between base/variant).

        Returns:
            Dict with at least 'input_ids' key.
        """
        ...

    def close(self) -> None:  # noqa: B027
        """Release model resources. Override if cleanup is needed."""
        pass
