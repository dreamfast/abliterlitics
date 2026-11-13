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
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from src.text_encoder.backends import comfyui_loader
from src.text_encoder.backends.dit_common import count_blocks as _count_blocks

log = logging.getLogger(__name__)

SAMPLING_SHIFT = 1.15
SAMPLING_MULTIPLIER = 1.0
TIME_SCALE = 1000.0
_DIT_PREFIX = "model.diffusion_model."


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

    # Two packaging styles: prefixed (redcraft stand-in) and bare (official
    # turbo export) — same dual handling as load_dit_txtfusion_sd.
    _pref = _DIT_PREFIX if keys and keys[0].startswith(_DIT_PREFIX) else ""

    def g(name: str) -> torch.Tensor:
        return sd[_pref + name]

    head_dim = 128
    first_w = g("first.weight")  # (features, channels*patch^2)
    cfg: dict[str, Any] = {
        "features": int(first_w.shape[0]),
        "channels": int(first_w.shape[1]) // 4,  # patch=2 -> patch^2=4
        "patch": 2,
        "layers": _count_blocks(keys, "^" + _pref + r"blocks\.(\d+)\."),
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
    missing, unexpected_keys = model.load_state_dict(stripped, strict=False)
    if missing:
        raise RuntimeError(f"Krea2 DiT load failed: {len(missing)} missing (first 5: {missing[:5]})")
    bad_unexpected = [k for k in unexpected_keys if not k.endswith((".comfy_quant", ".weight_scale"))]
    if bad_unexpected:
        raise RuntimeError(f"Krea2 DiT load failed: unexpected keys {bad_unexpected[:5]}")
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


KREA2_LATENT_CHANNELS = 16  # flow-matching latent channels (matches DiT in_features)


def run_krea2_divergence(
    dit: Any,
    base_contexts: list[torch.Tensor],  # each [L_i, 30720] raw tap stack
    var_contexts: list[torch.Tensor],
    device: str = "cuda",
    n_steps: int = 6,
    seed: int = 42,
    latent_hw: int = 64,
    batch: int = 1,
) -> dict[str, Any]:
    """Paired Euler trajectories + step-wise divergence (krea2 twin of
    z_image_dit.run_divergence).

    Descending flow sigmas (sampler order); shared x0 per prompt so the
    trajectory pair differs ONLY by conditioning. Controls: same base
    conditioning + different seed (the "new image" noise anchor) and an
    identical-seed control (must be ~0).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    sigmas = sigma_schedule_flow(n_steps).to(device)

    def _forward(x: torch.Tensor, ctx: torch.Tensor, sigma: float) -> torch.Tensor:
        return forward_krea2(dit, x, ctx.unsqueeze(0), sigma, device).float()

    per_prompt: list[dict[str, Any]] = []
    noise_control_final: list[float] = []

    for idx, (ctx_b, ctx_v) in enumerate(zip(base_contexts, var_contexts, strict=True)):
        x0 = torch.randn(batch, KREA2_LATENT_CHANNELS, latent_hw, latent_hw, generator=g).to(device).to(torch.bfloat16)
        x0_ctrl = (
            torch.randn(batch, KREA2_LATENT_CHANNELS, latent_hw, latent_hw, generator=g).to(device).to(torch.bfloat16)
        )
        xb, xv, xc = x0.clone(), x0.clone(), x0_ctrl.clone()
        vel_divs: list[float] = []
        lat_divs: list[float] = []
        for i in range(n_steps):
            s = float(sigmas[i])
            s_next = float(sigmas[i + 1])
            cb = ctx_b.to(device=device, dtype=torch.bfloat16)
            cv = ctx_v.to(device=device, dtype=torch.bfloat16)
            vb = _forward(xb, cb, s)
            vv = _forward(xv, cv, s)
            vc = _forward(xc, cb, s)

            vel_div = (vb - vv).norm().item() / max(vb.norm().item(), 1e-12)
            lat_div = (xb.float() - xv.float()).norm().item() / max(xb.float().norm().item(), 1e-12)
            vel_divs.append(vel_div)
            lat_divs.append(lat_div)

            dt = s_next - s  # negative on the descending grid
            xb = (xb.float() + vb * dt).to(torch.bfloat16)
            xv = (xv.float() + vv * dt).to(torch.bfloat16)
            xc = (xc.float() + vc * dt).to(torch.bfloat16)

        per_prompt.append(
            {
                "velocity_div": vel_divs,
                "latent_div": lat_divs,
                "final_latent_div": lat_divs[-1],
                "final_velocity_div": vel_divs[-1],
            }
        )
        noise_control_final.append((xb.float() - xc.float()).norm().item() / max(xb.float().norm().item(), 1e-12))
        log.info(
            "Prompt %d/%d: final latent div %.5f (noise control %.3f)",
            idx + 1,
            len(base_contexts),
            lat_divs[-1],
            noise_control_final[-1],
        )

    return {
        "per_prompt": per_prompt,
        "noise_control_final": float(torch.tensor(noise_control_final).mean().item()),
        "identical_control_final": 0.0,
        "n_steps": n_steps,
        "seed": seed,
        "latent_hw": latent_hw,
    }
