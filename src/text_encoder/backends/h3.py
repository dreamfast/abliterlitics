"""MiniMax H3 text-encoder backend (ComfyUI-native).

Measures conditioning for MiniMax H3 (Qwen3-VL-32B text encoder, truncated
to 50 layers). Verified stream layout (probe 2026-08-15, journal #135):

- RAW_HIDDEN: 50 per-layer streams (layer_00..layer_49, mean-pooled [n, 5120])
  plus ``consumed_hidden`` — the full-sequence UNNORMALIZED layer-49 output
  that H3 actually consumes (comfy/text_encoders/minimax.py: layer="last",
  layer_norm_hidden_state=False, checkpoint truncated before final norm).
  NOT chat-templated: raw prompt ids (add_special_tokens=False), asserted.
- PROJECTED: ``condition_projected`` — H3 DiT ``condition_proj`` output
  [n, L, 5376] (Linear 5120->5376, from the fl2va DiT checkpoint).
- CONNECTOR_OUT: ``refined_context`` — DiT ``token_refiner`` output [n, L, 5376]
  (2 RefinerBlocks + final RMSNorm, no rope), the honest "what the DiT reads".
  Replicates comfy/ldm/minimax/model.py preprocess_text_embeds:
  ``token_refiner(condition_proj(text_states[0]))``.

``minimax_token_tags`` (text=1 / vision-pad=0 per position) are carried in
artifact ``meta`` (NOT a stream — they are discrete tags, not activations)
and also exposed as the text-position mask on ``consumed_hidden``.

Model files:
- TE: qwen3vl_32b_minimax_h3_int8_convrot.safetensors (comfy int8-convrot
  quantized, 50 layers, 1602 keys). The nvfp4_awq TE is packed-quant and
  unreadable host-side — analysis refuses it with a clear error.
- DiT: minimax_h3_fl2va_pruned_int8_convrot.safetensors (conditioning keys
  condition_proj.* + token_refiner.* stored unquantized BF16).
"""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors import safe_open

from src.text_encoder.artifact import ConditioningArtifact, Pooling, Stage, Stream, StreamSpec
from src.text_encoder.backends import comfyui_loader
from src.text_encoder.backends.base import BaseBackend, HookManager
from src.text_encoder.metrics import pool_mean

log = logging.getLogger(__name__)

N_LAYERS = 50  # Qwen3-VL-32B truncated to 50 layers (probe-verified)
HIDDEN_DIM = 5120  # Qwen3-VL-32B LLM hidden size (tokenizer embedding_size)
CONSUMED_LAYER_IDX = 49  # layer="last" of the truncated checkpoint (0-indexed)
DIH_HIDDEN = 5376  # MiniMaxH3Model hidden_size
DIT_COND_PREFIXES = ("condition_proj.", "token_refiner.")
REQUIRED_COND_KEYS = {"condition_proj.weight", "token_refiner.final_norm.weight"}

# Qwen3 chat markers (add_special_tokens=False must keep these out)
_IM_START = 151644
_IM_END = 151645

_TE_LAYER_NAME = "qwen3vl_32b"  # SD1ClipModel layer name for the H3 TE


def _load_dit_conditioning_sd(dit_file: str | Path) -> dict[str, torch.Tensor]:
    """Selectively load H3 conditioning keys from the DiT safetensors."""
    dit_file = Path(dit_file)
    if not dit_file.exists():
        raise FileNotFoundError(f"H3 DiT safetensors not found: {dit_file}")
    sd: dict[str, torch.Tensor] = {}
    with safe_open(str(dit_file), framework="pt", device="cpu") as f:
        for k in list(f.keys()):
            if any(k.startswith(p) for p in DIT_COND_PREFIXES):
                sd[k] = f.get_tensor(k)
    if not REQUIRED_COND_KEYS.issubset(sd.keys()):
        raise ValueError(
            f"{dit_file.name} is missing H3 conditioning keys (found {len(sd)}, "
            f"need at least {sorted(REQUIRED_COND_KEYS)})"
        )
    log.info("Loaded H3 conditioning: %s (%d keys)", dit_file.name, len(sd))
    return sd


def _build_conditioning_modules(
    cond_sd: dict[str, torch.Tensor],
    device: str = "cpu",
) -> tuple[nn.Module, nn.Module]:
    """Build condition_proj + token_refiner using ComfyUI's own classes.

    Mirrors MiniMaxH3Model defaults (hidden 5376, 56 heads x 128, ffn 14336,
    eps 1e-5, 2 refiner blocks). Weights in the checkpoint are unquantized
    BF16; we compute in float32 like the z-image backend.
    """
    import comfy.ops
    from comfy.ldm.minimax.model import TokenRefiner

    ops = comfy.ops.disable_weight_init

    condition_proj = ops.Linear(HIDDEN_DIM, DIH_HIDDEN, bias=True, device=device, dtype=torch.float32)
    missing, unexpected = condition_proj.load_state_dict(
        {k.replace("condition_proj.", ""): v.float() for k, v in cond_sd.items() if k.startswith("condition_proj.")},
        strict=True,
    )
    if missing or unexpected:
        raise RuntimeError(f"condition_proj mismatch: missing={missing} unexpected={unexpected}")

    token_refiner = TokenRefiner(
        num_layers=2,
        hidden=DIH_HIDDEN,
        heads=56,
        head_dim=128,
        ffn=14336,
        eps=1e-5,
        qk_eps=1e-5,
        final_eps=1e-5,
        dtype=torch.float32,
        device=device,
        operations=ops,
    )
    missing2, unexpected2 = token_refiner.load_state_dict(
        {k.replace("token_refiner.", ""): v.float() for k, v in cond_sd.items() if k.startswith("token_refiner.")},
        strict=True,
    )
    if missing2 or unexpected2:
        raise RuntimeError(f"token_refiner mismatch: missing={missing2} unexpected={unexpected2}")

    condition_proj.eval()
    token_refiner.eval()
    return condition_proj, token_refiner


def _build_spec(name: str) -> StreamSpec:
    """Single source of truth for stream specs (used by streams() AND encode).

    Keeping construction in one place guards the cache/compare contract:
    produced artifacts must carry specs identical to the declared ones.
    """
    if name == "consumed_hidden":
        return StreamSpec(
            name=name,
            source_encoder="qwen3vl",
            layers=(CONSUMED_LAYER_IDX,),
            pooling=Pooling.NONE,
            stage=Stage.RAW_HIDDEN,
            dim=HIDDEN_DIM,
        )
    if name == "condition_projected":
        return StreamSpec(
            name=name,
            source_encoder="qwen3vl",
            layers=(CONSUMED_LAYER_IDX,),
            pooling=Pooling.NONE,
            stage=Stage.PROJECTED,
            dim=DIH_HIDDEN,
        )
    if name == "refined_context":
        return StreamSpec(
            name=name,
            source_encoder="qwen3vl",
            layers=(CONSUMED_LAYER_IDX,),
            pooling=Pooling.NONE,
            stage=Stage.CONNECTOR_OUT,
            dim=DIH_HIDDEN,
        )
    return StreamSpec(
        name=name,
        source_encoder="qwen3vl",
        layers=(int(name.split("_")[1]),),
        pooling=Pooling.MEAN,
        stage=Stage.RAW_HIDDEN,
        dim=HIDDEN_DIM,
    )


def _probe_layer_count(ckpt: str | Path) -> int:
    """Layer count from the safetensors header alone (no tensor loads)."""
    with safe_open(str(ckpt), framework="pt") as f:
        return sum(1 for k in list(f.keys()) if k.startswith("model.layers.") and k.endswith(".mlp.down_proj.weight"))


def _assert_no_chat_template(ids: list[int]) -> None:
    """Hard guard: H3 conditioning is raw prompt ids, never chat-wrapped.

    If ComfyUI ever starts injecting <|im_start|>/<|im_end|> wrappers, the
    consumed_hidden alignment vs the prompt text silently breaks.
    """
    if ids and (ids[0] == _IM_START or ids[-1] == _IM_END):
        raise RuntimeError(
            "Chat-template tokens detected in H3 tokenization — H3 consumes raw "
            f"prompt ids (first={ids[0]}, last={ids[-1]}); refusing to encode"
        )


class H3Backend(BaseBackend):
    """MiniMax H3 text encoder: Qwen3-VL-32B (50-layer truncation) + DiT conditioning chain."""

    def __init__(self, comfyui_path: str | Path = "/comfyui") -> None:
        self.comfyui_path = comfyui_path
        self.model: Any = None
        self.tokenizer: Any = None
        self.condition_proj: nn.Module | None = None
        self.token_refiner: nn.Module | None = None
        self._te_sd: dict[str, torch.Tensor] | None = None
        self._te_dtype_probe: torch.dtype | None = None
        self._quantized = False
        self._te_fake_quant: str | None = None
        self._device = "cpu"
        self.ckpt_id = ""

    @property
    def name(self) -> str:
        return "h3"

    @property
    def streams(self) -> list[StreamSpec]:
        return [_build_spec(f"layer_{i:02d}") for i in range(N_LAYERS)] + [
            _build_spec("consumed_hidden"),
            _build_spec("condition_projected"),
            _build_spec("refined_context"),
        ]

    def load(self, ckpt: str, dtype: str = "bf16", device: str = "cpu", dit_file: str = "") -> None:
        """Load the 32B TE backbone (comfy quant path) + H3 conditioning modules.

        dtype is derived from the file via llama_detect; the int8-convrot
        checkpoint keeps quantized weights and dequantizes on the fly.
        """
        if device != "cpu" and os.environ.get("H3_ALLOW_CUDA") != "1":
            # 32B weights (~33GB int8) exceed the 32GB card; cuda attempts
            # die in OOM mid-forward after minutes of loading. Fail fast.
            # H3_ALLOW_CUDA=1 is the explicit escape hatch for bigger hosts.
            raise RuntimeError(
                f"H3Backend runs on CPU (32B TE, ~33GB int8 weights > 32GB VRAM). "
                f"Refusing device={device!r}; set H3_ALLOW_CUDA=1 to override."
            )
        if not dit_file:
            raise ValueError("H3Backend requires dit_file (H3 DiT safetensors with condition_proj/token_refiner)")

        # Fail fast on wrong-architecture TEs BEFORE materializing ~33GB:
        # the layer count is readable from the safetensors header alone.
        n_layers = _probe_layer_count(ckpt)
        if n_layers != N_LAYERS:
            raise ValueError(
                f"Unexpected TE layer count: {n_layers} (expected {N_LAYERS} — H3 uses a "
                "50-layer-truncated Qwen3-VL-32B; a non-truncated or different-size TE is "
                "not comparable at the consumed_hidden interface)"
            )

        comfyui_loader.import_comfy(self.comfyui_path, cpu=True)

        self.ckpt_id = str(ckpt)
        self._device = device
        self._te_sd = comfyui_loader.load_te_state_dict(ckpt)

        import comfy.text_encoders.hunyuan_video as hv
        import comfy.text_encoders.minimax as minimax

        llama_meta = hv.llama_detect(self._te_sd)
        self._quantized = bool(llama_meta.get("llama_quantization_metadata"))
        log.info(
            "Detection: dtype_llama=%s quant=%s layers=%d",
            llama_meta.get("dtype_llama"),
            self._quantized,
            n_layers,
        )

        model_cls = minimax.te(
            dtype_llama=llama_meta.get("dtype_llama"),
            llama_quantization_metadata=llama_meta.get("llama_quantization_metadata"),
        )
        self.model = model_cls(device=device, dtype=torch.float32)
        result = self.model.load_sd(self._te_sd)
        if result is not None and hasattr(result, "missing_keys"):
            missing = [k for k in result.missing_keys if k != "spiece_model"]
            if missing:
                raise RuntimeError(f"Backbone load failed: {len(missing)} missing keys (first 5: {missing[:5]})")
        # Sentinel: embed_tokens must not be all-zero (silent load failure)
        emb = self.model.qwen3vl_32b.transformer.model.embed_tokens.weight
        if emb.numel() > 0 and emb.float().abs().max().item() < 1e-8:
            raise RuntimeError("embed_tokens appears unloaded — backbone load failed silently")
        # Free the full TE state dict — weights are consumed by load_sd;
        # keep only the dtype probe for artifact labeling.
        norm_w = self._te_sd.get("model.norm.weight")
        self._te_dtype_probe = norm_w.dtype if norm_w is not None else None
        self._te_sd = None

        self.tokenizer = minimax.MiniMaxH3Tokenizer()

        cond_sd = _load_dit_conditioning_sd(dit_file)
        self.condition_proj, self.token_refiner = _build_conditioning_modules(cond_sd, device=device)

        log.info("H3Backend loaded: %s (quantized=%s)", Path(ckpt).name, self._quantized)

    def apply_fake_quant(self, quant_type: str = "int8") -> None:
        """Fake-quantize the loaded backbone weights (noise-floor reference)."""
        from src.text_encoder.backends.base import fake_quantize_weights

        if self.model is None:
            raise RuntimeError("Backend not loaded — call load() first")
        transformer = self.model.qwen3vl_32b.transformer
        fake_quantize_weights(transformer, quant_type)
        self._te_fake_quant = quant_type

    def tokenize(self, prompts: list[str]) -> dict[str, Any]:
        """Tokenize prompts (for base/variant alignment assertion)."""
        if self.tokenizer is None:
            raise RuntimeError("Backend not loaded — call load() first")
        all_ids: list[list[int]] = []
        for p in prompts:
            tokens = self.tokenizer.tokenize_with_weights(p)
            ids = [t[0] for t in tokens[_TE_LAYER_NAME][0] if isinstance(t[0], int)]
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
        transformer = self.model.qwen3vl_32b.transformer
        layer_modules = transformer.model.layers

        if Stage.RAW_HIDDEN in wanted:
            for i in range(N_LAYERS):
                hm.register("raw_hidden", f"layer_{i:02d}", layer_modules[i])

        layer_pooled: list[list[torch.Tensor]] = [[] for _ in range(N_LAYERS)]
        consumed_seqs: list[torch.Tensor] = []  # [L_real, 5120] each
        text_masks: list[torch.Tensor] = []  # [L_real] each (1 = text position)
        projected_seqs: list[torch.Tensor] = []  # [L_real, 5376] each
        refined_seqs: list[torch.Tensor] = []  # [L_real, 5376] each
        tags_all: list[list[int]] = []

        try:
            for idx, prompt in enumerate(prompts):
                tokens = self.tokenizer.tokenize_with_weights(prompt)
                ids = [t[0] for t in tokens[_TE_LAYER_NAME][0] if isinstance(t[0], int)]
                _assert_no_chat_template(ids)
                out, _pooled, extra = self.model.encode_token_weights(tokens)
                # out: [1, L_real, 5120] — unnormalized layer-49 output.
                condition_proj = self.condition_proj
                if condition_proj is None:
                    raise RuntimeError("condition_proj not built — load() incomplete")
                cond_device = next(condition_proj.parameters()).device
                consumed = out[0].float().to(cond_device)

                tags = None
                if isinstance(extra, dict):
                    t = extra.get("minimax_token_tags")
                    if t is not None:
                        tags = t.reshape(-1).to(torch.int).cpu()
                if tags is None:
                    # text-only prompt: every position is a text position
                    tags = torch.ones(consumed.shape[0], dtype=torch.int)
                elif tags.shape[0] != consumed.shape[0]:
                    log.warning(
                        "minimax_token_tags length %d != sequence length %d — "
                        "falling back to all-text mask (extraction bug?)",
                        tags.shape[0],
                        consumed.shape[0],
                    )
                    tags = torch.ones(consumed.shape[0], dtype=torch.int)
                tags_all.append([int(v) for v in tags.tolist()])
                text_masks.append((tags == 1).to(torch.int))

                captures = hm.get_captures()
                if Stage.RAW_HIDDEN in wanted:
                    raw = captures.get("raw_hidden", {})
                    for i in range(N_LAYERS):
                        t = raw.get(f"layer_{i:02d}")
                        if t is None:
                            raise RuntimeError(f"Layer {i} hook did not fire")
                        # [1, L, 5120] -> pooled [5120]
                        layer_pooled[i].append(pool_mean(t, None)[0])

                consumed_seqs.append(consumed.cpu())

                if Stage.PROJECTED in wanted or Stage.CONNECTOR_OUT in wanted:
                    token_refiner = self.token_refiner
                    if token_refiner is None:
                        raise RuntimeError("token_refiner not built — load() incomplete")
                    with torch.no_grad():
                        # Faithful replication of MiniMaxH3Model.preprocess_text_embeds:
                        # token_refiner(condition_proj(text_states)) on 2D [L, dim]
                        projected = condition_proj(consumed)  # [L, 5376]
                        if Stage.PROJECTED in wanted:
                            projected_seqs.append(projected.cpu())
                        if Stage.CONNECTOR_OUT in wanted:
                            refined = token_refiner(projected, transformer_options={})  # [L, 5376]
                            refined_seqs.append(refined.cpu())

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
                spec = StreamSpec(
                    name=name,
                    source_encoder="qwen3vl",
                    layers=(i,),
                    pooling=Pooling.MEAN,
                    stage=Stage.RAW_HIDDEN,
                    dim=HIDDEN_DIM,
                )
                acts = torch.stack(layer_pooled[i])  # [n, 5120]
                streams[name] = Stream(name=name, spec=spec, activations=acts)
            # consumed_hidden: full-seq, right-padded to max length; the mask
            # carries text-vs-vision positions (all 1 for text-only prompts)
            max_len = max(s.shape[0] for s in consumed_seqs)
            consumed_padded = torch.zeros(len(prompts), max_len, HIDDEN_DIM)
            consumed_mask = torch.zeros(len(prompts), max_len, dtype=torch.int)
            for j, s in enumerate(consumed_seqs):
                consumed_padded[j, : s.shape[0]] = s
                consumed_mask[j, : s.shape[0]] = text_masks[j]
            spec = _build_spec("consumed_hidden")
            streams["consumed_hidden"] = Stream(
                name="consumed_hidden", spec=spec, activations=consumed_padded, mask=consumed_mask
            )
            artifacts.append(self._make_artifact(Stage.RAW_HIDDEN, streams, prompts, categories, tags_all))

        if Stage.PROJECTED in wanted:
            max_len = max(s.shape[0] for s in projected_seqs)
            acts = torch.zeros(len(prompts), max_len, DIH_HIDDEN)
            mask = torch.zeros(len(prompts), max_len, dtype=torch.int)
            for j, s in enumerate(projected_seqs):
                acts[j, : s.shape[0]] = s
                mask[j, : s.shape[0]] = text_masks[j]
            spec = _build_spec("condition_projected")
            artifacts.append(
                self._make_artifact(
                    Stage.PROJECTED,
                    {"condition_projected": Stream(name="condition_projected", spec=spec, activations=acts, mask=mask)},
                    prompts,
                    categories,
                    tags_all,
                )
            )

        if Stage.CONNECTOR_OUT in wanted:
            max_len = max(s.shape[0] for s in refined_seqs)
            acts = torch.zeros(len(prompts), max_len, DIH_HIDDEN)
            mask = torch.zeros(len(prompts), max_len, dtype=torch.int)
            for j, s in enumerate(refined_seqs):
                acts[j, : s.shape[0]] = s
                mask[j, : s.shape[0]] = text_masks[j]
            spec = _build_spec("refined_context")
            artifacts.append(
                self._make_artifact(
                    Stage.CONNECTOR_OUT,
                    {"refined_context": Stream(name="refined_context", spec=spec, activations=acts, mask=mask)},
                    prompts,
                    categories,
                    tags_all,
                )
            )

        return artifacts

    def _make_artifact(
        self,
        stage: Stage,
        streams: dict[str, Stream],
        prompts: list[str],
        categories: list[str],
        tags_all: list[list[int]] | None = None,
    ) -> ConditioningArtifact:
        if self._te_fake_quant:
            dtype_str = f"fakequant_{self._te_fake_quant}"
        elif self._quantized:
            dtype_str = "comfy_int8_convrot"
        else:
            dtype_str = "comfy_bf16"
        meta: dict[str, Any] = {
            "comfyui_commit": comfyui_loader.get_comfyui_commit(self.comfyui_path),
            "quantized": self._quantized,
            "n_layers": N_LAYERS,
            "consumed_layer_idx": CONSUMED_LAYER_IDX,
        }
        if tags_all is not None:
            meta["minimax_token_tags"] = tags_all
        return ConditioningArtifact(
            model="h3",
            backend="H3Backend",
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
        self.condition_proj = None
        self.token_refiner = None
        self._te_sd = None
        gc.collect()
        log.info("H3Backend closed")
