"""H3 (fl2va) DiT harness for attention capture — what do the AUDIO and VIDEO
streams READ from the text conditioning? (phase 9)

Loads MiniMaxH3Model with ComfyUI's own classes and the model_detection.py:369
config formulas (layers/refiner/hidden/head-dim/heads/ffn/text-dim/time-embed
sizes/rope, plus the checkpoint's safetensors-metadata ``config.transformer``
overrides — sigma shifts etc. — exactly as comfy applies them). The fl2va
checkpoint is int8_tensorwise-quantized (blocks) with bf16 conditioning keys:
mixed_precision_ops handles the I8 rows; condition_proj/token_refiner load as
their stored bf16.

Packed layout for the plain t2va case (no refs/keyframes):
``[text | audio (audio_t*2 rows) | video (latent_t*frame_rows)]`` — built via
comfy's own PackedLayout so segment spans are exact.

Per-modality capture: the runner configures AttentionCapture with query
segments {"audio": ..., "video": ...} from the layout spans — the FIRST
conditioning-level instrument for "does the audio stream read the prompt
differently than the video stream".

Sampling for capture: flow Euler on both streams. The DiT receives the VIDEO
sigma (timestep = sigma_v*1000) and derives the audio sigma internally via
time_shift_sigma(sigma_v, shift_v=12, shift_a=3); we integrate each stream
with its own ladder (video sigma grid + audio sigma derived per step), which
mirrors the ComfyUI H3 sampler's joint update.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from src.text_encoder.backends import comfyui_loader
from src.text_encoder.backends.dit_common import count_blocks as _count_blocks

log = logging.getLogger(__name__)

TIME_SCALE = 1000.0
_DIT_PREFIX = ""  # fl2va conditioning keys are top-level (condition_proj.*, ...)


def load_h3_dit(dit_file: str | Path, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> Any:
    """Build MiniMaxH3Model from the fl2va checkpoint with ComfyUI's own classes."""
    comfyui_loader.import_comfy("/comfyui", cpu=(device == "cpu"))
    import comfy.ops
    import comfy.utils
    from comfy.ldm.minimax.model import MiniMaxH3Model

    dit_file = Path(dit_file)
    if not dit_file.exists():
        raise FileNotFoundError(f"H3 DiT checkpoint not found: {dit_file}")

    sd: dict[str, torch.Tensor] = {}
    metadata: dict[str, str] | None = None
    with safe_open(str(dit_file), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        metadata = f.metadata()
        for k in keys:
            sd[k] = f.get_tensor(k)

    def g(name: str) -> torch.Tensor:
        return sd[name]

    cfg: dict[str, Any] = {
        "num_layers": _count_blocks(keys, r"blocks\.(\d+)\."),
        "token_refiner_num_layers": _count_blocks(keys, r"token_refiner\.blocks\.(\d+)\."),
        "hidden_size": int(g("video_patch_proj.weight").shape[0]),
        "latents_dim": int(g("final_layer.video_out.weight").shape[0]) // 4,  # patch 1x2x2
        "audio_latents_dim": int(g("final_layer.audio_out.weight").shape[0]),
        "attention_head_dim": int(g("blocks.0.attn.q_norm.weight").shape[0]),
        "ffn_hidden_size": int(g("blocks.0.mlp.fc1.weight").shape[0]) // 2,
        "text_dim": int(g("condition_proj.weight").shape[1]),
    }
    cfg["num_attention_heads"] = int(g("blocks.0.attn.qkv_proj.weight").shape[0]) // (3 * cfg["attention_head_dim"])
    if "adaln_t_table" in sd:
        cfg["adaln_curve_grid"] = int(g("adaln_t_table").shape[0])
        cfg["time_embed_dim"] = int(g("adaln_t_table").shape[1])
    else:
        cfg["timestep_input_dim"] = int(g("time_embedder.proj_in.weight").shape[1])
        cfg["time_embed_hidden_size"] = int(g("time_embedder.proj_in.weight").shape[0])
        cfg["time_embed_dim"] = int(g("time_embedder.proj_out.weight").shape[0])
    cfg["rope_inv_freq_len"] = int(g("rope.inv_freq").shape[0])
    # checkpoint metadata overrides (sigma shifts etc.) — same merge comfy does
    if metadata and "config" in metadata:
        cfg.update(json.loads(metadata["config"]).get("transformer", {}))
    log.info(
        "H3 DiT: hidden=%d layers=%d heads=%d text_dim=%d sigma_shifts v%s/a%s",
        cfg["hidden_size"],
        cfg["num_layers"],
        cfg["num_attention_heads"],
        cfg["text_dim"],
        cfg.get("sigma_shift_video"),
        cfg.get("sigma_shift_audio"),
    )

    quant = comfy.utils.detect_layer_quantization(sd, _DIT_PREFIX)
    if quant:
        ops: Any = comfy.ops.mixed_precision_ops(quant, compute_dtype=dtype)
        log.info("checkpoint is layer-quantized (int8_tensorwise) -> mixed_precision_ops")
    else:
        ops = comfy.ops.disable_weight_init

    model = MiniMaxH3Model(operations=ops, device=device, dtype=dtype, image_model="minimax_h3", **cfg)
    missing, unexpected_keys = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"H3 DiT load failed: {len(missing)} missing (first 5: {missing[:5]})")
    bad_unexpected = [k for k in unexpected_keys if not k.endswith((".comfy_quant", ".weight_scale"))]
    if bad_unexpected:
        raise RuntimeError(f"H3 DiT load failed: unexpected keys {bad_unexpected[:5]}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log.info("H3 DiT loaded: %s", dit_file.name)
    return model


def build_layout(
    dit: Any,
    text_len: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
) -> Any:
    """PackedLayout via comfy's own class; returns spans for capture configure."""
    from comfy.ldm.minimax.model import PackedLayout

    layout = PackedLayout(text_len, latent_t, latent_h, latent_w, audio_t)
    spans: dict[str, tuple[int, int]] = {}
    for a, b, kind in layout.segments:
        if kind in ("audio", "video"):
            spans[kind] = (a, b)
    if "audio" not in spans or "video" not in spans:
        raise RuntimeError(f"PackedLayout missing audio/video segments: {layout.segments}")
    return layout, spans


def time_shift_sigma(sigma: float, from_shift: float, to_shift: float) -> float:
    """comfy.ldm.minimax.model.time_shift_sigma (verbatim)."""
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


@torch.no_grad()
def forward_h3(
    dit: Any,
    video_x: torch.Tensor,  # [1, 24, T, H, W]
    audio_x: torch.Tensor,  # [1, 32, 2, T_a]
    context: torch.Tensor,  # [1, L, 5120] raw consumed_hidden
    sigma_v: float,
    device: str,
    transformer_options: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One DiT forward at video sigma ``sigma_v`` (audio sigma derived inside).

    Returns (video_velocity, audio_velocity).
    """
    timestep = torch.tensor([sigma_v * TIME_SCALE], device=device, dtype=torch.float32)
    out = dit(
        [video_x.to(device), audio_x.to(device)],
        timestep,
        context.to(device=device, dtype=torch.bfloat16),
        transformer_options=transformer_options or {},
        minimax_payload=payload or {},
    )
    v_video, v_audio = out[0], out[1]
    return v_video, v_audio
