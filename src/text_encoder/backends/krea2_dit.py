"""Krea2 DiT harness for attention capture — what does the DiT READ from the
12-tap conditioning? (phase 9, mirrors z_image_dit.py)

Loads SingleStreamDiT with ComfyUI's own classes and config derivation
(model_detection.py:949-963 formulas: features/channels/patch/layers/heads/
kvheads/txtlayers/txtdim from the checkpoint; remaining params are the class
defaults comfy itself uses). Handles BOTH checkpoint packaging styles:

- int8_tensorwise (krea2raw_INT8_fixed: I8 + weight_scale + comfy_quant
  keys) via comfy.ops.mixed_precision_ops — the same ops comfy's
  pick_operations selects when detect_layer_quantization finds comfy_quant
  keys;
- plain bf16/f16 (redcraft stand-in) via comfy.ops.disable_weight_init.

Sampling: flow-matching Euler replicated from supported_models.Krea2
sampling_settings (multiplier 1.0, shift 1.15); timestep = sigma*1000
(ModelSamplingDiscreteFlow convention). The conditioning input is the RAW
tap_stack (B, seq, 30720): the DiT unpacks + txtfuses internally — the true
end-to-end path.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from src.text_encoder.backends import comfyui_loader

log = logging.getLogger(__name__)

SAMPLING_SHIFT = 1.15
SAMPLING_MULTIPLIER = 1.0
TIME_SCALE = 1000.0
_DIT_PREFIX = "model.diffusion_model."


def _count_blocks(keys: list[str], pattern: str) -> int:
    pat = re.compile(pattern)
    ids = {int(m.group(1)) for k in keys if (m := pat.search(k))}
    return len(ids)


def load_krea2_dit(dit_file: str | Path, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> Any:
    """Build SingleStreamDiT from a Krea2 checkpoint with ComfyUI's own classes."""
    comfyui_loader.import_comfy("/comfyui", cpu=(device == "cpu"))
    import comfy.ops
    import comfy.utils
    from comfy.ldm.krea2.model import SingleStreamDiT

    dit_file = Path(dit_file)
    if not dit_file.exists():
        raise FileNotFoundError(f"Krea2 DiT checkpoint not found: {dit_file}")

    sd: dict[str, torch.Tensor] = {}
    with safe_open(str(dit_file), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for k in keys:
            sd[k] = f.get_tensor(k)

    def g(name: str) -> torch.Tensor:
        return sd[_DIT_PREFIX + name]

    head_dim = 128
    first_w = g("first.weight")  # (features, channels*patch^2)
    cfg: dict[str, Any] = {
        "features": int(first_w.shape[0]),
        "channels": int(first_w.shape[1]) // 4,  # patch=2 -> patch^2=4
        "patch": 2,
        "layers": _count_blocks(keys, _DIT_PREFIX + r"blocks\.(\d+)\."),
        "heads": int(g("blocks.0.attn.wq.weight").shape[0]) // head_dim,
        "kvheads": int(g("blocks.0.attn.wk.weight").shape[0]) // head_dim,
        "txtlayers": int(g("txtfusion.projector.weight").shape[1]),
        "txtdim": int(g("txtfusion.layerwise_blocks.0.prenorm.scale").shape[0]),
    }
    log.info(
        "Krea2 DiT: features=%d layers=%d heads=%d/%d txt=%dx%d",
        cfg["features"],
        cfg["layers"],
        cfg["heads"],
        cfg["kvheads"],
        cfg["txtlayers"],
        cfg["txtdim"],
    )

    # Two packaging styles on disk: prefixed (redcraft stand-in) and bare
    # (official turbo export) — detect which, use for quant detection + strip.
    dit_prefix = _DIT_PREFIX if any(k.startswith(_DIT_PREFIX) for k in sd) else ""
    quant = comfy.utils.detect_layer_quantization(sd, dit_prefix)
    if quant:
        ops: Any = comfy.ops.mixed_precision_ops(quant, compute_dtype=dtype)
        log.info("checkpoint is layer-quantized (int8_tensorwise) -> mixed_precision_ops")
    else:
        ops = comfy.ops.disable_weight_init

    model = SingleStreamDiT(operations=ops, device=device, dtype=dtype, image_model="krea2", **cfg)
    stripped = {(k[len(dit_prefix) :] if dit_prefix else k): v for k, v in sd.items()}
    unexpected = [k for k in stripped if not k.endswith((".comfy_quant", ".weight_scale"))]
    missing, unexpected_keys = model.load_state_dict(stripped, strict=False)
    if missing:
        raise RuntimeError(f"Krea2 DiT load failed: {len(missing)} missing (first 5: {missing[:5]})")
    bad_unexpected = [k for k in unexpected_keys if not k.endswith((".comfy_quant", ".weight_scale"))]
    if bad_unexpected:
        raise RuntimeError(f"Krea2 DiT load failed: unexpected keys {bad_unexpected[:5]}")
    del unexpected
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log.info("Krea2 DiT loaded: %s", dit_file.name)
    return model


def sigma_schedule_flow(
    n_steps: int,
    shift: float = SAMPLING_SHIFT,
    multiplier: float = SAMPLING_MULTIPLIER,
    descending: bool = True,
) -> torch.Tensor:
    """ModelSamplingDiscreteFlow sigma grid, terminal 0 appended.

    ``descending=True`` (default): sampler order — [shift(1)=sigma_max, ...,
    shift(1/n), 0] — so x0 = noise matches the sigma_max state exactly and
    Euler dt < 0 walks to the clean latent (what the ComfyUI sampler does).
    ``descending=False`` reproduces the raw stored (ascending) grid as the
    z-image phase-5 probe used.
    """
    ts = torch.arange(1, n_steps + 1, 1, dtype=torch.float64) / n_steps * multiplier
    sig = torch.tensor([time_snr_shift(shift, float(t)) for t in ts], dtype=torch.float64)
    if descending:
        sig = sig.flip(0)
    out: torch.Tensor = torch.cat([sig, torch.zeros(1, dtype=torch.float64)]).float()
    return out


def time_snr_shift(alpha: float, t: float) -> float:
    """comfy.model_sampling.time_snr_shift (verbatim)."""
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


@torch.no_grad()
def forward_krea2(
    dit: Any,
    x: torch.Tensor,
    context: torch.Tensor,  # [1, L, 30720] raw tap stack
    sigma: float,
    device: str,
    transformer_options: dict[str, Any] | None = None,
) -> torch.Tensor:
    """One DiT forward at flow-sigma ``sigma`` (timestep = sigma*1000)."""
    timesteps = torch.tensor([sigma * TIME_SCALE], device=device, dtype=x.dtype)
    out: torch.Tensor = dit(
        x,
        timesteps,
        context.to(device=device, dtype=x.dtype),
        attention_mask=None,
        transformer_options=transformer_options or {},
    )
    return out
