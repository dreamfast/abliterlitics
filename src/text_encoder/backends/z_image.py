"""Z-Image text-encoder backend (ComfyUI-native).

Measures conditioning for Z-Image (Qwen3-4B text encoder). Verified stream
layout (2026-08-14, empirical):

- RAW_HIDDEN: 36 per-layer streams (layer_00..layer_35, mean-pooled [n, 2560])
  plus ``consumed_hidden`` — the full-sequence layer -2 output (index 34)
  that Z-Image actually consumes (Qwen3_4BModel layer="hidden", layer_idx=-2).
- PROJECTED: ``cap_embedded`` — cap_embedder output [n, L, 3840]
  (RMSNorm(2560) + Linear from the Z-Image DiT checkpoint).
- CONNECTOR_OUT: ``refined_context`` — context_refiner output [n, L, 3840],
  the honest "what the DiT sees". Replicates ComfyUI's embed_cap padding to
  a multiple of 32 with cap_pad_token before the refiner (faithful); pad
  positions are excluded from metrics via mask.

Model files:
- TE: qwen_3_4b.safetensors (base, BF16) / qwen3-4b-heretic_{int8,fp8,nvfp4}
- DiT: z_image_turbo_bf16.safetensors (conditioning modules: cap_embedder,
  context_refiner, cap_pad_token — 26 keys, selective load)
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors import safe_open

from src.text_encoder.artifact import ConditioningArtifact, Pooling, Stage, Stream, StreamSpec
from src.text_encoder.backends import comfyui_loader, gguf_loader
from src.text_encoder.backends.base import BaseBackend, HookManager, _pack_sequences, dtype_label
from src.text_encoder.backends.gguf_loader import GGUFMeta
from src.text_encoder.metrics import pool_mean

log = logging.getLogger(__name__)

N_LAYERS = 36  # Qwen3_4BConfig.num_hidden_layers
HIDDEN_DIM = 2560  # Qwen3_4BConfig.hidden_size
CONSUMED_LAYER_IDX = 34  # layer_idx=-2 of 36
CONTEXT_DIM = 3840  # Z-Image DiT context dim
N_HEADS = 30  # Z-Image (dim 3840, model_detection.py)
CAP_FEAT_DIM = 2560
N_REFINER_LAYERS = 2
PAD_TOKENS_MULTIPLE = 32

_DIT_COND_PREFIXES = ("cap_embedder.", "context_refiner.", "cap_pad_token")


def _build_spec(name: str) -> StreamSpec:
    """Single source of truth for stream specs (streams() AND encode).

    Mirrors the h3/krea2 ``_build_spec`` contract guard: constructing the
    spec in exactly one place keeps the artifact's per-stream ``spec``
    equal to the backend's declared ``streams`` even if one path is
    edited without the other (audit te-backends.md, Bugs #1).
    """
    if name == "consumed_hidden":
        return StreamSpec(
            name=name,
            source_encoder="qwen3",
            layers=(CONSUMED_LAYER_IDX,),
            pooling=Pooling.NONE,
            stage=Stage.RAW_HIDDEN,
            dim=HIDDEN_DIM,
        )
    if name == "cap_embedded":
        return StreamSpec(
            name=name,
            source_encoder="qwen3",
            layers=(CONSUMED_LAYER_IDX,),
            pooling=Pooling.NONE,
            stage=Stage.PROJECTED,
            dim=CONTEXT_DIM,
        )
    if name == "refined_context":
        return StreamSpec(
            name=name,
            source_encoder="qwen3",
            layers=(CONSUMED_LAYER_IDX,),
            pooling=Pooling.NONE,
            stage=Stage.CONNECTOR_OUT,
            dim=CONTEXT_DIM,
        )
    # layer_XX: per-layer mean-pooled raw hidden
    return StreamSpec(
        name=name,
        source_encoder="qwen3",
        layers=(int(name.split("_")[1]),),
        pooling=Pooling.MEAN,
        stage=Stage.RAW_HIDDEN,
        dim=HIDDEN_DIM,
    )


def _load_dit_conditioning_sd(dit_file: str | Path) -> dict[str, torch.Tensor]:
    """Selectively load conditioning keys from the Z-Image DiT safetensors."""
    dit_file = Path(dit_file)
    if not dit_file.exists():
        raise FileNotFoundError(f"Z-Image DiT safetensors not found: {dit_file}")
    sd: dict[str, torch.Tensor] = {}
    with safe_open(str(dit_file), framework="pt", device="cpu") as f:
        for k in list(f.keys()):
            if any(k.startswith(p) for p in _DIT_COND_PREFIXES):
                sd[k] = f.get_tensor(k)
    needed = {"cap_embedder.0.weight", "context_refiner.0.attention.qkv.weight", "cap_pad_token"}
    if not needed.issubset(sd.keys()):
        raise ValueError(
            f"{dit_file.name} is missing Z-Image conditioning keys (found {len(sd)}, need at least {sorted(needed)})"
        )
    log.info("Loaded Z-Image conditioning: %s (%d keys)", dit_file.name, len(sd))
    return sd


def _build_conditioning_modules(
    cond_sd: dict[str, torch.Tensor],
    device: str = "cpu",
) -> tuple[nn.Module, nn.Module, nn.Module, torch.Tensor]:
    """Build cap_embedder + rope + context_refiner using ComfyUI's own classes.

    Config from comfy/model_detection.py (dim==3840 -> Z image branch).
    """
    import comfy.ops
    from comfy.ldm.lumina.model import EmbedND, JointTransformerBlock

    ops = comfy.ops.disable_weight_init
    settings: dict[str, Any] = {"operations": ops, "device": device, "dtype": torch.float32}

    cap_embedder = nn.Sequential(
        ops.RMSNorm(CAP_FEAT_DIM, eps=1e-5, elementwise_affine=True, device=device, dtype=torch.float32),
        ops.Linear(CAP_FEAT_DIM, CONTEXT_DIM, bias=True, device=device, dtype=torch.float32),
    )
    missing, unexpected = cap_embedder.load_state_dict(
        {k.replace("cap_embedder.", ""): v for k, v in cond_sd.items() if k.startswith("cap_embedder.")},
        strict=True,
    )
    if missing or unexpected:
        raise RuntimeError(f"cap_embedder mismatch: missing={missing} unexpected={unexpected}")

    rope_embedder = EmbedND(dim=CONTEXT_DIM // N_HEADS, theta=256.0, axes_dim=[32, 48, 48])

    context_refiner = nn.ModuleList(
        [
            JointTransformerBlock(
                i,
                CONTEXT_DIM,
                N_HEADS,
                N_HEADS,
                256,
                8.0 / 3.0,
                1e-5,
                True,
                modulation=False,
                operation_settings=settings,
            )
            for i in range(N_REFINER_LAYERS)
        ]
    ).to(device)
    missing2, unexpected2 = context_refiner.load_state_dict(
        {k.replace("context_refiner.", ""): v for k, v in cond_sd.items() if k.startswith("context_refiner.")},
        strict=True,
    )
    if missing2 or unexpected2:
        raise RuntimeError(f"context_refiner mismatch: missing={missing2} unexpected={unexpected2}")

    cap_embedder.eval()
    context_refiner.eval()
    # cap_pad_token is stored [1, dim] — flatten to [dim] for repeat math
    cap_pad_token = cond_sd["cap_pad_token"].float().reshape(-1).to(device)
    return cap_embedder, rope_embedder, context_refiner, cap_pad_token


class ZImageBackend(BaseBackend):
    """Z-Image text encoder: Qwen3-4B backbone + DiT-side conditioning chain."""

    def __init__(self, comfyui_path: str | Path = "/comfyui") -> None:
        self.comfyui_path = comfyui_path
        self.model: Any = None
        self.tokenizer: Any = None
        self.cap_embedder: nn.Module | None = None
        self.rope_embedder: nn.Module | None = None
        self.context_refiner: nn.Module | None = None
        self.cap_pad_token: torch.Tensor | None = None
        self._te_sd: dict[str, torch.Tensor] | None = None
        self._te_dtype_probe: torch.dtype | None = None
        self._quantized = False
        self._te_fake_quant: str | None = None
        self._gguf_meta: GGUFMeta | None = None
        self._device = "cpu"
        self.ckpt_id = ""

    @property
    def name(self) -> str:
        return "zimage"

    @property
    def streams(self) -> list[StreamSpec]:
        specs = [_build_spec(f"layer_{i:02d}") for i in range(N_LAYERS)]
        specs.append(_build_spec("consumed_hidden"))
        specs.append(_build_spec("cap_embedded"))
        specs.append(_build_spec("refined_context"))
        return specs

    def load(self, ckpt: str, dtype: str = "bf16", device: str = "cpu", dit_file: str = "") -> None:
        """Load TE backbone + Z-Image conditioning modules.

        dtype is derived from the file via llama_detect (reviewer W7).
        """
        comfyui_loader.import_comfy(self.comfyui_path, cpu=(device == "cpu"))

        if not dit_file:
            raise ValueError("ZImageBackend requires dit_file (Z-Image diffusion model safetensors)")

        self.ckpt_id = str(ckpt)
        self._device = device
        if str(ckpt).endswith(".gguf"):
            # llama.cpp container: eager dequant -> HF-named state dict
            # (ComfyUI-GGUF qwen3 path, see gguf_loader.py).
            self._te_sd, self._gguf_meta = gguf_loader.load_gguf_te_state_dict(ckpt)
            # Fail fast on a wrong/truncated GGUF BEFORE handing ~4GB to
            # load_sd — same posture as krea2.load / h3._probe_layer_count.
            n_layers = gguf_loader.infer_layer_count(self._te_sd)
            if n_layers != N_LAYERS:
                raise ValueError(
                    f"GGUF text stack has {n_layers} layers, expected {N_LAYERS} — wrong model or truncated container"
                )
        else:
            self._te_sd = comfyui_loader.load_te_state_dict(ckpt)

        import comfy.text_encoders.hunyuan_video as hv
        import comfy.text_encoders.z_image as z_image

        llama_meta = hv.llama_detect(self._te_sd)
        self._quantized = bool(llama_meta.get("llama_quantization_metadata"))
        log.info(
            "Detection: dtype_llama=%s quant=%s",
            llama_meta.get("dtype_llama"),
            self._quantized,
        )

        model_cls = z_image.te(
            dtype_llama=llama_meta.get("dtype_llama"),
            llama_quantization_metadata=llama_meta.get("llama_quantization_metadata"),
        )
        self.model = model_cls(device=device, dtype=torch.float32)
        result = self.model.load_sd(self._te_sd)
        if result is not None and hasattr(result, "missing_keys"):
            missing = [k for k in result.missing_keys if k != "spiece_model"]
            if missing:
                raise RuntimeError(f"Backbone load failed: {len(missing)} missing keys (first 5: {missing[:5]})")
        # Sentinel (reviewer W4)
        emb = self.model.qwen3_4b.transformer.model.embed_tokens.weight
        if emb.numel() > 0 and emb.float().abs().max().item() < 1e-8:
            raise RuntimeError("embed_tokens appears unloaded — backbone load failed silently")
        # Free the full TE state dict (~24GB for 12B) — weights are consumed
        # by load_sd; keep only the dtype probe for artifact labeling.
        norm_w = self._te_sd.get("model.norm.weight")
        self._te_dtype_probe = norm_w.dtype if norm_w is not None else None
        self._te_sd = None

        self.tokenizer = z_image.ZImageTokenizer()

        cond_sd = _load_dit_conditioning_sd(dit_file)
        (
            self.cap_embedder,
            self.rope_embedder,
            self.context_refiner,
            self.cap_pad_token,
        ) = _build_conditioning_modules(cond_sd, device=device)

        log.info(
            "ZImageBackend loaded: %s (quantized=%s gguf=%s)",
            Path(ckpt).name,
            self._quantized,
            self._gguf_meta.quant_label if self._gguf_meta else "-",
        )

    def apply_fake_quant(self, quant_type: str = "int8") -> None:
        """Fake-quantize the loaded backbone weights (noise-floor reference).

        Rounds weights to the target precision in-place (fp32 math, original
        dtype preserved). Used on the BASE model to build the quantization
        floor for damage-in-quant-units reporting.
        """
        from src.text_encoder.backends.base import fake_quantize_weights

        if self.model is None:
            raise RuntimeError("Backend not loaded — call load() first")
        transformer = self.model.qwen3_4b.transformer
        fake_quantize_weights(transformer, quant_type)
        self._te_fake_quant = quant_type

    def tokenize(self, prompts: list[str]) -> dict[str, Any]:
        """Tokenize prompts (for base/variant alignment assertion)."""
        if self.tokenizer is None:
            raise RuntimeError("Backend not loaded — call load() first")
        all_ids: list[list[int]] = []
        for p in prompts:
            tokens = self.tokenizer.tokenize_with_weights(p)
            ids = [t[0] for t in tokens["qwen3_4b"][0]]
            all_ids.append(ids)
        return {"input_ids": all_ids, "n_tokens": [len(i) for i in all_ids]}

    def encode(
        self,
        prompts: list[str],
        categories: list[str],
        stages: list[Stage] | None = None,
    ) -> list[ConditioningArtifact]:
        """Encode prompts per-prompt with layer hooks + conditioning chain."""
        if self.model is None:
            raise RuntimeError("Backend not loaded — call load() first")

        wanted = set(stages) if stages else {Stage.RAW_HIDDEN, Stage.PROJECTED, Stage.CONNECTOR_OUT}

        hm = HookManager()
        transformer = self.model.qwen3_4b.transformer
        layer_modules = transformer.model.layers

        if Stage.RAW_HIDDEN in wanted:
            for i in range(N_LAYERS):
                hm.register("raw_hidden", f"layer_{i:02d}", layer_modules[i])

        layer_pooled: list[list[torch.Tensor]] = [[] for _ in range(N_LAYERS)]
        consumed_seqs: list[torch.Tensor] = []  # [L_real, 2560] each
        projected_seqs: list[torch.Tensor] = []  # [L_real, 3840] each
        refined_seqs: list[torch.Tensor] = []  # [L_pad, 3840] each (incl. cap-pad)
        refined_masks: list[torch.Tensor] = []  # [L_pad] each (1 = real token)

        try:
            for idx, prompt in enumerate(prompts):
                tokens = self.tokenizer.tokenize_with_weights(prompt)
                out, _pooled, _extra = self.model.encode_token_weights(tokens)
                # out: [1, L_real, 2560] — ZImageTokenizer pads to min_length=1 only,
                # so sequence length == real token count (verified empirically).
                # Align with the conditioning modules' device (ComfyUI may
                # return the TE output on CPU even for GPU models).
                cap_embedder = self.cap_embedder
                if cap_embedder is None:
                    raise RuntimeError("cap_embedder not built — load() incomplete")
                cond_device = next(cap_embedder.parameters()).device
                consumed = out[0].float().to(cond_device)

                captures = hm.get_captures()
                if Stage.RAW_HIDDEN in wanted:
                    raw = captures.get("raw_hidden", {})
                    for i in range(N_LAYERS):
                        t = raw.get(f"layer_{i:02d}")
                        if t is None:
                            raise RuntimeError(f"Layer {i} hook did not fire")
                        # [1, L, 2560] -> pooled [2560]
                        layer_pooled[i].append(pool_mean(t, None)[0])

                consumed_seqs.append(consumed.cpu())

                if Stage.PROJECTED in wanted or Stage.CONNECTOR_OUT in wanted:
                    cap_embedder = self.cap_embedder
                    if cap_embedder is None:
                        raise RuntimeError("cap_embedder not built — load() incomplete")
                    with torch.no_grad():
                        projected = cap_embedder(consumed.unsqueeze(0))[0]  # [L, 3840]
                        projected_seqs.append(projected.cpu())

                        if Stage.CONNECTOR_OUT in wanted:
                            rope_embedder = self.rope_embedder
                            context_refiner = self.context_refiner
                            cap_pad_token = self.cap_pad_token
                            if rope_embedder is None or context_refiner is None or cap_pad_token is None:
                                raise RuntimeError("conditioning modules not built — load() incomplete")
                            # Faithful replication of ComfyUI embed_cap: pad to
                            # multiple of 32 with cap_pad_token, then refine.
                            # Refiner expects 3D [bsz, L, dim].
                            pad_extra = (-projected.shape[0]) % PAD_TOKENS_MULTIPLE
                            if pad_extra:
                                padded = torch.cat([projected, cap_pad_token.unsqueeze(0).repeat(pad_extra, 1)], dim=0)
                            else:
                                padded = projected
                            seq_l = padded.shape[0]
                            pos_ids = torch.zeros(1, seq_l, 3, device=padded.device)
                            pos_ids[:, :, 0] = torch.arange(seq_l).float() + 1.0
                            freqs_cis = rope_embedder(pos_ids).movedim(1, 2)
                            refined = padded.unsqueeze(0)  # [1, L, 3840]
                            for layer in context_refiner.children():
                                refined = layer(refined, None, freqs_cis, transformer_options={})
                            refined = refined[0].float().cpu()  # back to [L, 3840]
                            mask = torch.zeros(seq_l, dtype=torch.int)
                            mask[: projected.shape[0]] = 1
                            refined_seqs.append(refined)
                            refined_masks.append(mask)

                captures.clear()
                if (idx + 1) % 10 == 0 or idx + 1 == len(prompts):
                    log.info("Encoded %d/%d prompts", idx + 1, len(prompts))
        finally:
            hm.clear()

        artifacts: list[ConditioningArtifact] = []

        if Stage.RAW_HIDDEN in wanted:
            streams: dict[str, Stream] = {}
            for i in range(N_LAYERS):
                name = f"layer_{i:02d}"
                acts = torch.stack(layer_pooled[i])  # [n, 2560]
                streams[name] = Stream(name=name, spec=_build_spec(name), activations=acts)
            # consumed_hidden: full-seq, right-padded to max length
            consumed_padded, consumed_mask = _pack_sequences(consumed_seqs, HIDDEN_DIM)
            streams["consumed_hidden"] = Stream(
                name="consumed_hidden",
                spec=_build_spec("consumed_hidden"),
                activations=consumed_padded,
                mask=consumed_mask,
            )
            artifacts.append(self._make_artifact(Stage.RAW_HIDDEN, streams, prompts, categories))

        if Stage.PROJECTED in wanted:
            acts, mask = _pack_sequences(projected_seqs, CONTEXT_DIM)
            artifacts.append(
                self._make_artifact(
                    Stage.PROJECTED,
                    {
                        "cap_embedded": Stream(
                            name="cap_embedded",
                            spec=_build_spec("cap_embedded"),
                            activations=acts,
                            mask=mask,
                        )
                    },
                    prompts,
                    categories,
                )
            )

        if Stage.CONNECTOR_OUT in wanted:
            # refined_seqs already include cap-pad positions; refined_masks
            # carry 1 = real token / 0 = cap-pad — pack preserves them.
            acts, mask = _pack_sequences(refined_seqs, CONTEXT_DIM, masks=refined_masks)
            artifacts.append(
                self._make_artifact(
                    Stage.CONNECTOR_OUT,
                    {
                        "refined_context": Stream(
                            name="refined_context",
                            spec=_build_spec("refined_context"),
                            activations=acts,
                            mask=mask,
                        )
                    },
                    prompts,
                    categories,
                )
            )

        return artifacts

    def _make_artifact(
        self,
        stage: Stage,
        streams: dict[str, Stream],
        prompts: list[str],
        categories: list[str],
    ) -> ConditioningArtifact:
        dtype_str = dtype_label(
            fake_quant=self._te_fake_quant,
            gguf_quant_label=self._gguf_meta.quant_label if self._gguf_meta is not None else None,
            comfy_quantized=self._quantized,
            comfy_quant_label="comfy_int8_convrot",
        )
        meta: dict[str, Any] = {
            "comfyui_commit": comfyui_loader.get_comfyui_commit(self.comfyui_path),
            "quantized": self._quantized,
            "pad_tokens_multiple": PAD_TOKENS_MULTIPLE,
        }
        if self._gguf_meta is not None:
            # Additive provenance (bpw, qtype mix) — persisted into
            # te_<name>.json via VariantReport and consumed by the report's
            # GGUF ladder section.
            meta["gguf"] = self._gguf_meta.to_dict()
        return ConditioningArtifact(
            model="zimage",
            backend="ZImageBackend",
            ckpt_id=self.ckpt_id,
            stage=stage,
            streams=streams,
            prompts=prompts,
            categories=categories,
            dtype=dtype_str,
            device=self._device,
            meta=meta,
        )

    def close(self) -> None:
        """Free model + modules."""
        self.model = None
        self.tokenizer = None
        self.cap_embedder = None
        self.rope_embedder = None
        self.context_refiner = None
        self.cap_pad_token = None
        self._te_sd = None
        self._gguf_meta = None
        gc.collect()
        log.info("ZImageBackend closed")
