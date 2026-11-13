"""ComfyUI-native model loading for text-encoder analysis.

Loads ComfyUI-format single-file safetensors (INT8 ConvRot, NVFP4, BF16)
using ComfyUI's own model classes — measuring the EXACT conditioning
ComfyUI produces, with its native quantization handling.

Import safety: importing ``comfy.*`` triggers ``comfy.cli_args`` which parses
``sys.argv`` (crashing on our argparse flags) and ``comfy.model_management``
which probes CUDA. ``import_comfy()`` neutralizes both: it enables ComfyUI's
own args parsing, injects a minimal ``--cpu`` argv, and is idempotent.

Verified against the real models (2026-08-14):
- gemma-3-12b-it-heretic-v2_int8.safetensors: 2062 keys, loads with
  missing_keys=[] (only 'spiece_model' unexpected)
- ltx-2.3-22b-dev-fp8.safetensors: dual_linear projection detected via
  sd_detect on the 4 projection keys (selective load, not the full 28GB)
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

log = logging.getLogger(__name__)

_comfy_imported = False
_commit_cache: dict[str, str] = {}

# Keys loaded selectively from the diffusion model file (DeepSeek C4 —
# avoids materializing the full 28GB DIT state dict for ~1.2GB of weights).
DIT_PROJECTION_PREFIX = "text_embedding_projection."


def import_comfy(comfyui_path: str | Path, cpu: bool = True) -> None:
    """Import ComfyUI's comfy package safely. Idempotent.

    Args:
        comfyui_path: Path to the ComfyUI repo root (contains ``comfy/``).
        cpu: Force ComfyUI's model_management into CPU mode (safe default —
            model_management probes CUDA at import and crashes on CPU-only
            torch). Pass False when running under a CUDA-enabled runtime
            (Docker + nvidia) to allow GPU execution.

    Raises:
        ImportError: If the comfy package cannot be imported.
    """
    global _comfy_imported
    if _comfy_imported:
        return

    comfyui_path = str(Path(comfyui_path).resolve())
    if comfyui_path not in sys.path:
        sys.path.insert(0, comfyui_path)

    # comfy.cli_args parses sys.argv at import when args_parsing is enabled;
    # with it disabled (default) our own argparse flags would leak into
    # ComfyUI's parser OR be ignored (losing --cpu). Enable parsing and
    # inject a minimal argv.
    import comfy.options

    comfy.options.enable_args_parsing(True)
    old_argv = sys.argv
    sys.argv = ["comfy", "--cpu"] if cpu else ["comfy"]
    try:
        import comfy.text_encoders.hunyuan_video  # noqa: F401
        import comfy.text_encoders.lt  # noqa: F401
        import comfy.utils  # noqa: F401

        _comfy_imported = True
        log.info("Imported comfy from %s (cpu=%s)", comfyui_path, cpu)
    finally:
        sys.argv = old_argv


def get_comfyui_commit(comfyui_path: str | Path) -> str:
    """Get the git commit hash of the ComfyUI checkout (cached, for reproducibility)."""
    key = str(comfyui_path)
    if key in _commit_cache:
        return _commit_cache[key]
    try:
        result = subprocess.run(
            ["git", "-C", key, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        commit = result.stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        log.warning("Could not get ComfyUI commit hash: %s", exc)
        commit = "unknown"
    _commit_cache[key] = commit
    return commit


def load_te_state_dict(te_file: str | Path) -> dict[str, torch.Tensor]:
    """Load a text-encoder state dict from a ComfyUI-format safetensors file.

    Args:
        te_file: Path to the TE safetensors (e.g. gemma-3-12b-it-heretic-v2_int8.safetensors).

    Returns:
        Full state dict (backbone + vision + embedded tokenizer).
    """
    te_file = Path(te_file)
    if not te_file.exists():
        raise FileNotFoundError(f"TE safetensors not found: {te_file}")

    sd: dict[str, torch.Tensor] = {}
    with safe_open(str(te_file), framework="pt", device="cpu") as f:
        for k in list(f.keys()):
            sd[k] = f.get_tensor(k)
    log.info("Loaded TE state dict: %s (%d keys)", te_file.name, len(sd))
    return sd


def load_dit_projection_sd(dit_file: str | Path) -> dict[str, torch.Tensor]:
    """Load ONLY the text-embedding projection keys from a diffusion model file.

    Selective load per reviewer C4: the DIT checkpoint is ~28GB but the
    projection is ~1.2GB. For dual_linear models the connectors are DiT-side
    and NOT part of the TE measurement path.

    Args:
        dit_file: Path to the diffusion model safetensors (e.g. ltx-2.3-22b-dev-fp8.safetensors).

    Returns:
        State dict with only text_embedding_projection.* keys.
    """
    dit_file = Path(dit_file)
    if not dit_file.exists():
        raise FileNotFoundError(f"DIT safetensors not found: {dit_file}")

    sd: dict[str, torch.Tensor] = {}
    with safe_open(str(dit_file), framework="pt", device="cpu") as f:
        for k in list(f.keys()):
            if k.startswith(DIT_PROJECTION_PREFIX):
                sd[k] = f.get_tensor(k)
    if not sd:
        raise ValueError(
            f"No {DIT_PROJECTION_PREFIX}* keys found in {dit_file} — is this an LTX-2 diffusion model checkpoint?"
        )
    log.info("Loaded DIT projection: %s (%d keys)", dit_file.name, len(sd))
    return sd


def create_ltx2_model(
    te_sd: dict[str, torch.Tensor],
    dit_proj_sd: dict[str, torch.Tensor],
    device: str = "cpu",
) -> Any:
    """Create an LTXAVTEModel with backbone from te_sd and projection from dit_proj_sd.

    Detection rules (verified):
    - llama_detect runs on the TE state dict (dtype + quantization metadata)
    - sd_detect runs on the DIT projection state dict (single vs dual linear)

    Fail-loud verification (reviewer W4): load_sd(strict=False) never raises,
    so we assert the backbone actually matched.

    Args:
        te_sd: Text encoder state dict (with model.* backbone keys).
        dit_proj_sd: Projection state dict from the DIT file.
        device: "cpu" or "cuda".

    Returns:
        Instantiated LTXAVTEModel with weights loaded.

    Raises:
        RuntimeError: If the backbone fails to load (>5% missing keys).
    """
    import comfy.text_encoders.hunyuan_video as hv
    import comfy.text_encoders.lt as lt

    llama_meta = hv.llama_detect(te_sd)
    sd_meta = lt.sd_detect(dit_proj_sd)
    log.info(
        "Detection: dtype_llama=%s quant=%s projection=%s",
        llama_meta.get("dtype_llama"),
        bool(llama_meta.get("llama_quantization_metadata")),
        sd_meta.get("text_projection_type", "single_linear"),
    )

    model_cls = lt.ltxav_te(
        dtype_llama=llama_meta.get("dtype_llama"),
        llama_quantization_metadata=llama_meta.get("llama_quantization_metadata"),
        text_projection_type=sd_meta.get("text_projection_type", "single_linear"),
    )
    model = model_cls(device=device, dtype=torch.float32)

    # Backbone load. Verified: ComfyUI-format TE keys (model.*) match
    # Gemma3_12B's parameter tree directly — no prefix stripping needed.
    result = model.load_sd(te_sd)
    if result is not None and hasattr(result, "missing_keys"):
        missing = [k for k in result.missing_keys if k != "spiece_model"]
        if missing:
            raise RuntimeError(
                f"Backbone load failed: {len(missing)} missing keys "
                f"(first 5: {missing[:5]}). TE file format not recognized."
            )
    # Sentinel check (reviewer W4): embed_tokens must have loaded real weights
    emb = model.gemma3_12b.transformer.model.embed_tokens.weight
    if emb.numel() > 0 and emb.float().abs().max().item() < 1e-8:
        raise RuntimeError("embed_tokens appears unloaded (all ~zero) — backbone load failed silently")

    # Projection load from DIT
    result2 = model.load_sd(dit_proj_sd)
    if result2:
        missing2, unexpected2 = result2
        real_missing2 = [k for k in missing2 if "connector" not in k]
        if real_missing2:
            raise RuntimeError(f"Projection load failed: missing {real_missing2[:5]}")
        log.debug("Projection load: %d missing, %d unexpected (connectors excluded)", len(missing2), len(unexpected2))

    # execution_device is not initialized in LTXAVTEModel.__init__ (ComfyUI
    # initializes it via the CLIP wrapper) — required by encode_token_weights.
    model.execution_device = device
    model.set_clip_options({"execution_device": device})

    log.info("LTX-2 model created: projection=%s", model.text_projection_type)
    return model


def create_ltx2_tokenizer(te_sd: dict[str, torch.Tensor]) -> Any:
    """Create the LTX Gemma tokenizer using the embedded spiece_model.

    The TE safetensors embeds the tokenizer as a uint8 tensor under the
    'spiece_model' key — ComfyUI's SPieceTokenizer consumes it directly
    (no extraction to disk needed).

    Args:
        te_sd: Text encoder state dict containing 'spiece_model'.

    Returns:
        LTXAVGemmaTokenizer instance.

    Raises:
        ValueError: If 'spiece_model' is missing from the state dict.
    """
    import comfy.text_encoders.lt as lt

    spiece = te_sd.get("spiece_model")
    if spiece is None:
        raise ValueError("TE state dict has no 'spiece_model' — cannot build tokenizer")
    return lt.LTXAVGemmaTokenizer(tokenizer_data={"spiece_model": spiece})
