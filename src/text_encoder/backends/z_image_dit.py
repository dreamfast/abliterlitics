"""DiT-step divergence for Z-Image — does the diffusion model RESPOND to
the conditioning perturbation?

Bridges conditioning damage -> generation. Method: replicate ComfyUI's
flow-matching Euler sampling for Z-Image (ModelSamplingDiscreteFlow with
shift=3.0, multiplier=1.0, v-prediction), running TWO trajectories in
lockstep from IDENTICAL initial noise — one with the base text-encoder
conditioning, one with the variant's. At every step we measure:

- velocity divergence  ||v_base - v_var|| / ||v_base||   (per-step response)
- latent divergence    ||x_base - x_var|| / ||x||        (accumulated drift)

Controls (scale anchors):
- noise control: same conditioning, different initial noise seed — the
  "completely different image" divergence level
- identical control: same conditioning, same seed — must be ~0 (epsilon)

The conditioning input is the RAW TE output (consumed_hidden, 2560-d):
NextDiT applies its own cap_embedder + context_refiner internally (they
live in the DiT checkpoint), so this measures the true end-to-end path.

Model config verified against comfy/model_detection.py (dim==3840 branch)
and the checkpoint itself (30 layers, x_embedder [3840, 64]).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from src.text_encoder.backends import comfyui_loader

log = logging.getLogger(__name__)

# Z-Image DiT config (model_detection.py, dim==3840 branch) + checkpoint scan
ZIMAGE_DIT_CONFIG: dict[str, Any] = {
    "patch_size": 2,
    "in_channels": 16,
    "dim": 3840,
    "n_heads": 30,
    "n_kv_heads": 30,
    "axes_dims": [32, 48, 48],
    "axes_lens": [1536, 512, 512],
    "rope_theta": 256.0,
    "ffn_dim_multiplier": 8.0 / 3.0,
    "z_image_modulation": True,
    "time_scale": 1000.0,
    "cap_feat_dim": 2560,
    "qk_norm": True,
    "pad_tokens_multiple": 32,
    "multiple_of": 256,
    "norm_eps": 1e-5,
}
SAMPLING_SHIFT = 3.0
SAMPLING_MULTIPLIER = 1.0


def _count_blocks(keys: list[str], pattern: str) -> int:
    import re

    pat = re.compile(pattern)
    ids = {int(m.group(1)) for k in keys if (m := pat.search(k))}
    return len(ids)


def load_zimage_dit(dit_file: str | Path, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> Any:
    """Build NextDiT from the Z-Image checkpoint with ComfyUI's own classes.

    Config mirrors comfy/model_detection.py's dim==3840 branch; block counts
    are derived from the checkpoint (strict load — fail-loud).
    """
    comfyui_loader.import_comfy("/comfyui", cpu=(device == "cpu"))
    import comfy.ops
    from comfy.ldm.lumina.model import NextDiT

    dit_file = Path(dit_file)
    if not dit_file.exists():
        raise FileNotFoundError(f"Z-Image DiT checkpoint not found: {dit_file}")

    sd: dict[str, torch.Tensor] = {}
    with safe_open(str(dit_file), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for k in keys:
            sd[k] = f.get_tensor(k)

    cfg = dict(ZIMAGE_DIT_CONFIG)
    cfg["n_layers"] = _count_blocks(keys, r"layers\.(\d+)\.")
    cfg["n_refiner_layers"] = _count_blocks(keys, r"noise_refiner\.(\d+)\.")
    log.info("Z-Image DiT: %d layers, %d noise refiner blocks", cfg["n_layers"], cfg["n_refiner_layers"])

    ops = comfy.ops.disable_weight_init
    model = NextDiT(operations=ops, device=device, dtype=dtype, **cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if "cap_pad_token" not in k]
    if real_missing:
        raise RuntimeError(f"DiT load failed: {len(real_missing)} missing (first 5: {real_missing[:5]})")
    if unexpected:
        raise RuntimeError(f"DiT load failed: unexpected keys {unexpected[:5]}")
    # cap_pad_token is a buffer-style parameter stored in the checkpoint
    if "cap_pad_token" in sd and hasattr(model, "cap_pad_token"):
        model.cap_pad_token = torch.nn.Parameter(sd["cap_pad_token"].to(device=device, dtype=dtype))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log.info("Z-Image DiT loaded: %s (%d params)", dit_file.name, sum(p.numel() for p in model.parameters()))
    return model


def _sigma_schedule(
    n_steps: int, shift: float = SAMPLING_SHIFT, multiplier: float = SAMPLING_MULTIPLIER
) -> torch.Tensor:
    """Replicate ModelSamplingDiscreteFlow.set_parameters sigma grid.

    sigmas = tan(t*pi/2) shifted, t uniform over (0, 1]; plus terminal 0.
    """
    ts = torch.arange(1, n_steps + 1, 1, dtype=torch.float64) / n_steps * multiplier
    sig = torch.tensor([time_snr_shift(shift, float(t)) for t in ts], dtype=torch.float64)
    out: torch.Tensor = torch.cat([sig, torch.zeros(1, dtype=torch.float64)]).float()
    return out


def time_snr_shift(alpha: float, t: float) -> float:
    """comfy.model_sampling.time_snr_shift (verbatim)."""
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


@torch.no_grad()
def run_divergence(
    dit: Any,
    base_contexts: list[torch.Tensor],  # each [L_i, 2560] raw TE output
    var_contexts: list[torch.Tensor],  # each [L_i, 2560]
    device: str = "cuda",
    n_steps: int = 20,
    seed: int = 42,
    latent_hw: int = 128,  # 1024px / 8 — latent spatial size
    batch: int = 1,
) -> dict[str, Any]:
    """Run paired Euler trajectories and measure step-wise divergence.

    Returns:
        {
          "per_prompt": [ {velocity_div: [float]*n_steps, latent_div: [float]*n_steps,
                            final_latent_div: float, final_velocity_div: float} ],
          "noise_control_final": float,   # same ctx, different seed
          "identical_control_final": float,
        }
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    sigmas = _sigma_schedule(n_steps).to(device)

    def _forward(model: Any, x: torch.Tensor, ctx: torch.Tensor, sigma: float) -> torch.Tensor:
        # NextDiT.forward(x, timesteps, context, num_tokens)
        timesteps = torch.tensor([sigma], device=device, dtype=x.dtype)
        out: torch.Tensor = model(x, timesteps, ctx.unsqueeze(0), torch.tensor([ctx.shape[0]], device=device))
        return out.float()

    per_prompt: list[dict[str, Any]] = []
    noise_control_final: list[float] = []
    identical_control_final: list[float] = []

    for idx, (ctx_b, ctx_v) in enumerate(zip(base_contexts, var_contexts, strict=True)):
        x0 = (
            torch.randn(batch, ZIMAGE_DIT_CONFIG["in_channels"], latent_hw, latent_hw, generator=g)
            .to(device)
            .to(torch.bfloat16)
        )
        # noise control: same base conditioning, different seed
        x0_ctrl = (
            torch.randn(batch, ZIMAGE_DIT_CONFIG["in_channels"], latent_hw, latent_hw, generator=g)
            .to(device)
            .to(torch.bfloat16)
        )

        xb, xv, xc = x0.clone(), x0.clone(), x0_ctrl.clone()
        # scale initial latent by sigma_max (x = noise * sigma(1.0) = noise * 1.0)
        vel_divs: list[float] = []
        lat_divs: list[float] = []
        for i in range(n_steps):
            s = float(sigmas[i])
            s_next = float(sigmas[i + 1])
            cb = ctx_b.to(device=device, dtype=torch.bfloat16)
            cv = ctx_v.to(device=device, dtype=torch.bfloat16)
            vb = _forward(dit, xb, cb, s)
            vv = _forward(dit, xv, cv, s)
            vc = _forward(dit, xc, cb, s)

            vel_div = (vb - vv).norm().item() / max(vb.norm().item(), 1e-12)
            lat_div = (xb.float() - xv.float()).norm().item() / max(xb.float().norm().item(), 1e-12)
            vel_divs.append(vel_div)
            lat_divs.append(lat_div)

            dt = s_next - s  # negative
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
        identical_control_final.append(0.0)
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
