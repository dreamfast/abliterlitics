"""Build the BASE Krea2 text-encoder files from the HF Qwen3-VL-4B-Instruct checkpoint.

Krea2 conditions on a 12-layer tap stack of the FULL Qwen3-VL-4B (LLM + visual
tower, comfy/text_encoders/krea2.py). The abliterated arms on disk use the
comfy single-file TE layout (model.embed_tokens / model.layers.0..35 /
model.norm / model.visual.*) — the HF checkpoint stores the LLM under
``model.language_model.*`` (post-refactor transformers layout) with the visual
tower already in comfy-compatible naming. This script:

1. bf16 base: remap ``model.language_model.`` -> ``model.``, keep visual as-is,
   drop nothing else (HF ships no lm_head for this checkpoint — tied).
2. int8 base (matched packaging for qwen3-vl-4b-heretic_int8): quantize exactly
   the tensor set that the heretic int8 file quantizes (mirrored via its
   ``.comfy_quant`` key set), using the verified comfy int8_tensorwise recipe:
   scale = max|W|/127 (scalar F32), q = clamp(round(W/scale), -127, 127) (I8),
   plus the constant 29-byte comfy_quant blob. Everything else stays BF16.

Format verification (probe 2026-08-15): manual RTN reproduction of the heretic
int8 file matches stored codes exactly (mean|d| = 0.000 on LLM + visual
tensors); scale satisfies 127*scale == max|W| to bf16 precision.

Usage:
    python3 scripts/build_krea2_base_te.py \
        --hf-dir .cache/hf-te/Qwen3-VL-4B-Instruct \
        --heretic-int8 /home/maxx/extra/comfyui-models/text_encoders/qwen3-vl-4b-heretic_int8.safetensors \
        --out-dir comparisons/te-krea2/models/te
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

log = logging.getLogger(__name__)

LLM_PREFIX_HF = "model.language_model."
BLOB = {"format": "int8_tensorwise"}


def _blob_tensor() -> torch.Tensor:
    return torch.tensor(list(json.dumps(BLOB).encode()), dtype=torch.uint8)


def load_hf_state(hf_dir: Path) -> dict[str, torch.Tensor]:
    """Load HF shards, remapping the LLM prefix to the comfy layout."""
    idx_path = hf_dir / "model.safetensors.index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"missing {idx_path}")
    idx = json.loads(idx_path.read_text())
    shards = sorted(set(idx["weight_map"].values()))
    sd: dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(str(hf_dir / shard), framework="pt", device="cpu") as f:
            for k in list(f.keys()):
                out_k = k
                if k.startswith(LLM_PREFIX_HF):
                    out_k = "model." + k[len(LLM_PREFIX_HF) :]
                sd[out_k] = f.get_tensor(k)
        log.info("loaded %s (%d keys total)", shard, len(sd))
    return sd


def quantize_tensorwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Verified comfy int8_tensorwise recipe (scalar scale, symmetric RTN)."""
    wf = w.to(torch.float32)
    scale = (wf.abs().max() / 127.0).clamp_min(1e-12)
    q = torch.clamp(torch.round(wf / scale), -127, 127).to(torch.int8)
    return q, scale.reshape(()).to(torch.float32)


def build_int8(
    base_sd: dict[str, torch.Tensor],
    heretic_int8: Path,
    out_path: Path,
) -> dict[str, int]:
    """Quantize exactly the heretic-int8 quant set; mirror everything else."""
    with safe_open(str(heretic_int8), framework="pt", device="cpu") as f:
        quant_keys = {k[: -len(".comfy_quant")] for k in f if k.endswith(".comfy_quant")}
        heretic_dtypes = {k: str(f.get_slice(k).get_dtype()) for k in f}

    # quant artifact keys are ``<tensor-base>.comfy_quant`` where the tensor
    # itself is ``<tensor-base>.weight`` (the .weight segment is not part of
    # the quant key name — probe-verified against the heretic file layout).
    missing = {k for k in quant_keys if k + ".weight" not in base_sd}
    if missing:
        raise RuntimeError(f"heretic quant set not in base after remap: {sorted(missing)[:5]} ({len(missing)})")

    out: dict[str, torch.Tensor] = {}
    n_quant = n_copy = 0
    for k, v in base_sd.items():
        tensor_base = k[: -len(".weight")] if k.endswith(".weight") else k
        if tensor_base in quant_keys:
            q, s = quantize_tensorwise(v)
            out[k] = q
            out[tensor_base + ".weight_scale"] = s
            out[tensor_base + ".comfy_quant"] = _blob_tensor()
            n_quant += 1
        else:
            if heretic_dtypes.get(k) == "BF16" and v.dtype != torch.bfloat16:
                v = v.to(torch.bfloat16)
            out[k] = v
            n_copy += 1
    save_file(out, str(out_path))
    return {"quantized": n_quant, "copied": n_copy}


def sha256_prefix(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--hf-dir", required=True, type=Path)
    p.add_argument("--heretic-int8", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sd = load_hf_state(args.hf_dir)
    log.info("base bf16 state: %d tensors (36 LLM layers + visual)", len(sd))

    bf16_path = args.out_dir / "qwen3vl_4b_base.safetensors"
    save_file(sd, str(bf16_path))
    log.info("wrote %s (%.2f GB)", bf16_path, bf16_path.stat().st_size / 2**30)

    int8_path = args.out_dir / "qwen3vl_4b_base_int8_convrot.safetensors"
    stats = build_int8(sd, args.heretic_int8, int8_path)
    log.info("wrote %s (%.2f GB) — %s", int8_path, int8_path.stat().st_size / 2**30, stats)

    for path in (bf16_path, int8_path):
        log.info("sha256[%s] = %s…", path.name, sha256_prefix(path))


if __name__ == "__main__":
    main()
