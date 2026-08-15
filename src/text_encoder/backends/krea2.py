"""Krea 2 (K2) text-encoder backend (ComfyUI-native).

Measures conditioning for Krea 2, whose TE is the FULL Qwen3-VL-4B
(36 LLM layers + visual tower). Verified stream layout (probe 2026-08-15,
comfy master ``comfy/text_encoders/krea2.py`` + ``comfy/ldm/krea2/model.py``):

- Taps: ``hidden_states[2, 5, 8, …, 35]`` — 12 layers stacked
  ``(B, 12, seq, 2560)``, flattened to ``(B, seq, 12·2560=30720)`` by
  ``Krea2TEModel.encode_token_weights`` after stripping the KREA2_TEMPLATE
  (Qwen-Image-style describe-the-image system prompt) system+user prefix.
  The template IS applied (chat-wrapped ids, no-think) — mirror-image of the
  H3 raw-ids rule.
- ``tap_stack``: that full-sequence ``(B, seq, 30720)`` tensor — the literal
  ``context`` input the DiT receives (the consumed interface).
- RAW_HIDDEN per-layer streams: mean-pooled views of the 12 taps post-strip
  (the constant system+user template PREFIX is removed; a short constant
  assistant-turn suffix remains — identical on every arm, so pooled
  comparisons are unaffected).
- PROJECTED ``txtfusion_out``: the DiT ``txtfusion`` adapter output
  ``(B, seq, 2560)``. Semantics (from TextFusionTransformer.forward with
  ``(b, l=seq, n=12, d=2560)``): 2 layerwise blocks attend ACROSS THE 12 TAPS
  per token → learned ``projector`` Linear(12→1) mixes taps → 2 refiner
  blocks attend across tokens. No rope, mask=None.
- CONNECTOR_OUT ``connector_out``: DiT ``txtmlp`` output ``(B, seq, 6144)``
  (RMSNorm → Linear 2560→6144 → GELU → Linear), the stream concatenated with
  image tokens in the single-stream DiT.

Model files:
- TE: any comfy-layout qwen3vl_4b TE — bf16 or comfy int8_tensorwise
  (qwen3-vl-4b-heretic_int8, built qwen3vl_4b_base_int8_convrot, …).
  Must contain the visual tower (a plain Qwen3-4B is the z-image TE — wrong
  interface; fail-fast probe).
- DiT: txtfusion + txtmlp weights read from the Krea2 diffusion model
  safetensors (int8_tensorwise dequantized — krea2raw — or plain BF16 —
  redcraft stand-in; both handled, meta records which file was used).
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
from src.text_encoder.backends import comfyui_loader
from src.text_encoder.backends.base import BaseBackend, fake_quantize_weights
from src.text_encoder.metrics import pool_mean

log = logging.getLogger(__name__)

N_LAYERS = 36  # Qwen3-VL-4B LLM depth
HIDDEN_DIM = 2560  # qwen3vl_4b hidden size (tokenizer embedding_size)
# Mirrors comfy/text_encoders/krea2.py KREA2_TAP_LAYERS (keep in sync — a
# comfy change would alter the tap-stack layout silently).
TAP_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
N_TAPS = len(TAP_LAYERS)
TAP_STACK_DIM = N_TAPS * HIDDEN_DIM  # 30720 — the DiT `context` feature dim
TXT_DIM = 2560  # txtfusion output (per-token, post tap-mixing)
DIT_DIM = 6144  # txtmlp output / single-stream DiT width

DIT_COND_PREFIXES = ("txtfusion.", "txtmlp.")
_DIT_KEY_PREFIX = "model.diffusion_model."
REQUIRED_COND_KEYS = {
    "txtfusion.projector.weight",
    "txtfusion.refiner_blocks.0.attn.wq.weight",
    "txtmlp.1.weight",
}

# Qwen chat markers — Krea2 conditioning IS chat-templated (opposite of H3)
_IM_START = 151644

_TE_LAYER_NAME = "qwen3vl_4b"  # SD1ClipModel layer name for the krea2 TE


def load_dit_txtfusion_sd(dit_file: str | Path) -> tuple[dict[str, torch.Tensor], int]:
    """Selectively load + dequantize txtfusion/txtmlp keys from a Krea2 DiT.

    Handles both packaging styles found on disk: int8_tensorwise
    (I8 weight + scalar F32 weight_scale + comfy_quant blob — krea2raw) and
    plain BF16/F32 (redcraft stand-in). Dequantization is exact
    (probe-verified recipe: q·scale).

    Returns ``(state_dict, n_dequantized)`` — the count of int8 tensors that
    were dequantized (0 for a plain bf16 DiT; reported in artifact meta).
    """
    dit_file = Path(dit_file)
    if not dit_file.exists():
        raise FileNotFoundError(f"Krea2 DiT safetensors not found: {dit_file}")
    raw: dict[str, torch.Tensor] = {}
    with safe_open(str(dit_file), framework="pt", device="cpu") as f:
        for k in list(f.keys()):
            if not k.startswith(_DIT_KEY_PREFIX):
                continue
            short = k[len(_DIT_KEY_PREFIX) :]
            if any(short.startswith(p) for p in DIT_COND_PREFIXES):
                raw[short] = f.get_tensor(k)
    if not REQUIRED_COND_KEYS.issubset(raw.keys()):
        raise ValueError(
            f"{dit_file.name} is missing Krea2 conditioning keys (found {len(raw)}, "
            f"need at least {sorted(REQUIRED_COND_KEYS)})"
        )

    # int8_tensorwise artifacts: quant key <base>.comfy_quant, tensor
    # <base>.weight (I8), scale <base>.weight_scale (scalar F32).
    quant_bases = {k[: -len(".comfy_quant")] for k in raw if k.endswith(".comfy_quant")}
    out: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        base = k[: -len(".weight")] if k.endswith(".weight") else k
        if base in quant_bases and k.endswith(".weight"):
            scale = raw.get(base + ".weight_scale")
            if scale is None:
                raise ValueError(f"int8 tensor {k} lacks weight_scale")
            out[k] = v.to(torch.float32) * scale.to(torch.float32)
        elif k.endswith((".comfy_quant", ".weight_scale")):
            continue  # consumed / metadata only
        else:
            out[k] = v.to(torch.float32)
    log.info(
        "Loaded Krea2 DiT conditioning: %s (%d tensors, %d dequantized)", dit_file.name, len(out), len(quant_bases)
    )
    return out, len(quant_bases)


def build_txtfusion_modules(
    cond_sd: dict[str, torch.Tensor],
    device: str = "cpu",
) -> tuple[nn.Module, nn.Module]:
    """Build txtfusion + txtmlp using ComfyUI's own krea2 classes.

    Shapes verified against the checkpoints (attn.wq [2560,2560] with 20
    heads → head_dim 128; mlp 6912 = mult-4/128-rounded of 2/3·2560;
    txtmlp 2560→6144→6144 with bias). Weights dequantized to float32;
    compute in float32 like the other backends.
    """
    import comfy.ops
    from comfy.ldm.krea2.model import RMSNorm as Krea2RMSNorm
    from comfy.ldm.krea2.model import TextFusionTransformer

    ops = comfy.ops.disable_weight_init

    txtfusion = TextFusionTransformer(
        num_txt_layers=N_TAPS,
        txt_dim=TXT_DIM,
        heads=20,
        multiplier=4,
        bias=False,
        kvheads=20,
        device=device,
        dtype=torch.float32,
        operations=ops,
    )
    tf_sd = {k[len("txtfusion.") :]: v for k, v in cond_sd.items() if k.startswith("txtfusion.")}
    missing, unexpected = txtfusion.load_state_dict(tf_sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"txtfusion mismatch: missing={missing} unexpected={unexpected}")

    txtmlp = nn.Sequential(
        Krea2RMSNorm(TXT_DIM, device=device, dtype=torch.float32, operations=ops),
        ops.Linear(TXT_DIM, DIT_DIM, bias=True, device=device, dtype=torch.float32),
        nn.GELU(approximate="tanh"),
        ops.Linear(DIT_DIM, DIT_DIM, bias=True, device=device, dtype=torch.float32),
    )
    mlp_sd = {k[len("txtmlp.") :]: v for k, v in cond_sd.items() if k.startswith("txtmlp.")}
    missing2, unexpected2 = txtmlp.load_state_dict(mlp_sd, strict=True)
    if missing2 or unexpected2:
        raise RuntimeError(f"txtmlp mismatch: missing={missing2} unexpected={unexpected2}")

    txtfusion.eval()
    txtmlp.eval()
    return txtfusion, txtmlp


def _build_spec(name: str) -> StreamSpec:
    """Single source of truth for stream specs (streams() AND encode)."""
    if name == "tap_stack":
        return StreamSpec(
            name=name,
            source_encoder="qwen3vl",
            layers=TAP_LAYERS,
            pooling=Pooling.NONE,
            stage=Stage.RAW_HIDDEN,
            dim=TAP_STACK_DIM,
        )
    if name == "txtfusion_out":
        return StreamSpec(
            name=name,
            source_encoder="qwen3vl",
            layers=TAP_LAYERS,
            pooling=Pooling.NONE,
            stage=Stage.PROJECTED,
            dim=TXT_DIM,
        )
    if name == "connector_out":
        return StreamSpec(
            name=name,
            source_encoder="qwen3vl",
            layers=TAP_LAYERS,
            pooling=Pooling.NONE,
            stage=Stage.CONNECTOR_OUT,
            dim=DIT_DIM,
        )
    return StreamSpec(
        name=name,
        source_encoder="qwen3vl",
        layers=(int(name.split("_")[1]),),
        pooling=Pooling.MEAN,
        stage=Stage.RAW_HIDDEN,
        dim=HIDDEN_DIM,
    )


def _probe_te(ckpt: str | Path) -> tuple[int, bool]:
    """(layer count, has visual tower) from the safetensors header alone.

    The visual-tower check is what distinguishes the krea2 TE from the
    z-image TE: a plain Qwen3-4B file has the same 36 layers but no
    ``model.visual.*`` keys — wrong interface, fail loud before loading GBs.
    """
    with safe_open(str(ckpt), framework="pt") as f:
        keys = list(f.keys())
    n = sum(1 for k in keys if k.startswith("model.layers.") and k.endswith(".mlp.down_proj.weight"))
    has_visual = any(k.startswith("model.visual.") for k in keys)
    return n, has_visual


def _assert_chat_template(ids: list[int]) -> None:
    """Hard guard: krea2 conditioning IS chat-wrapped (KREA2_TEMPLATE).

    If ComfyUI ever stops applying the template, the tap alignment vs the
    prompt text silently changes (constant prefix gone) and cross-run
    comparisons break.
    """
    if not ids or ids[0] != _IM_START:
        raise RuntimeError(
            "Krea2 tokenization lacks the chat template (expected <|im_start|> "
            f"first, got {ids[:3] if ids else 'empty'}) — KREA2_TEMPLATE must wrap "
            "the prompt; refusing to encode"
        )


class Krea2Backend(BaseBackend):
    """Krea2 text encoder: Qwen3-VL-4B 12-tap interface + DiT txtfusion chain."""

    def __init__(self, comfyui_path: str | Path = "/comfyui") -> None:
        self.comfyui_path = comfyui_path
        self.model: Any = None
        self.tokenizer: Any = None
        self.txtfusion: nn.Module | None = None
        self.txtmlp: nn.Module | None = None
        self._te_sd: dict[str, torch.Tensor] | None = None
        self._quantized = False
        self._te_fake_quant: str | None = None
        self._dit_name = ""
        self._dit_dequantized = 0
        self._device = "cpu"
        self.ckpt_id = ""

    @property
    def name(self) -> str:
        return "krea2"

    @property
    def streams(self) -> list[StreamSpec]:
        return [_build_spec(f"layer_{i:02d}") for i in TAP_LAYERS] + [
            _build_spec("tap_stack"),
            _build_spec("txtfusion_out"),
            _build_spec("connector_out"),
        ]

    def load(self, ckpt: str, dtype: str = "bf16", device: str = "cpu", dit_file: str = "") -> None:
        """Load the qwen3vl_4b backbone (comfy quant path) + txtfusion modules."""
        if not dit_file:
            raise ValueError("Krea2Backend requires dit_file (Krea2 DiT with txtfusion/txtmlp)")

        n_layers, has_visual = _probe_te(ckpt)
        if n_layers != N_LAYERS or not has_visual:
            kind = "plain Qwen3-4B (z-image TE — no visual tower)" if n_layers == N_LAYERS else f"{n_layers} layers"
            raise ValueError(
                f"Unexpected TE for krea2: {kind}, has_visual={has_visual}. Krea2 needs the FULL "
                "Qwen3-VL-4B (36 LLM layers + visual tower); a truncated or text-only TE is not "
                "comparable at the 12-tap interface"
            )

        comfyui_loader.import_comfy(self.comfyui_path, cpu=(device == "cpu"))

        self.ckpt_id = str(ckpt)
        self._device = device
        self._te_sd = comfyui_loader.load_te_state_dict(ckpt)

        import comfy.text_encoders.hunyuan_video as hv
        import comfy.text_encoders.krea2 as krea2_te

        llama_meta = hv.llama_detect(self._te_sd)
        self._quantized = bool(llama_meta.get("llama_quantization_metadata"))
        log.info(
            "Detection: dtype_llama=%s quant=%s layers=%d visual=%s",
            llama_meta.get("dtype_llama"),
            self._quantized,
            n_layers,
            has_visual,
        )

        model_cls = krea2_te.te(
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
        emb = self.model.qwen3vl_4b.transformer.model.embed_tokens.weight
        if emb.numel() > 0 and emb.float().abs().max().item() < 1e-8:
            raise RuntimeError("embed_tokens appears unloaded — backbone load failed silently")
        self._te_sd = None  # weights consumed by load_sd

        self.tokenizer = krea2_te.Krea2Tokenizer()

        cond_sd, n_dequant = load_dit_txtfusion_sd(dit_file)
        self._dit_name = Path(dit_file).name
        self._dit_dequantized = n_dequant
        self.txtfusion, self.txtmlp = build_txtfusion_modules(cond_sd, device=device)

        log.info(
            "Krea2Backend loaded: %s (quantized=%s) + DiT %s",
            Path(ckpt).name,
            self._quantized,
            self._dit_name,
        )

    def apply_fake_quant(self, quant_type: str = "int8") -> None:
        """Fake-quantize the loaded backbone weights (noise-floor reference)."""
        if self.model is None:
            raise RuntimeError("Backend not loaded — call load() first")
        transformer = self.model.qwen3vl_4b.transformer
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
        """Encode prompts; one forward per prompt yields all stages (the model's
        own 12-tap collection — RAW_HIDDEN needs no hooks)."""
        if self.model is None:
            raise RuntimeError("Backend not loaded — call load() first")

        wanted = set(stages) if stages else {Stage.RAW_HIDDEN, Stage.PROJECTED, Stage.CONNECTOR_OUT}

        tap_pooled: list[list[torch.Tensor]] = [[] for _ in range(N_TAPS)]  # per-tap [n, 2560]
        stack_seqs: list[torch.Tensor] = []  # [L, 30720] each
        projected_seqs: list[torch.Tensor] = []  # [L, 2560] each
        connector_seqs: list[torch.Tensor] = []  # [L, 6144] each

        for idx, prompt in enumerate(prompts):
            tokens = self.tokenizer.tokenize_with_weights(prompt)
            ids = [t[0] for t in tokens[_TE_LAYER_NAME][0] if isinstance(t[0], int)]
            _assert_chat_template(ids)
            out, _pooled, _extra = self.model.encode_token_weights(tokens)
            # out: [1, L, 30720] — post-template-strip flattened 12-tap stack,
            # i.e. the literal DiT `context` input.
            if out.shape[-1] != TAP_STACK_DIM:
                raise RuntimeError(f"TE output dim {out.shape[-1]} != {TAP_STACK_DIM} (12×2560)")
            stack = out[0].float()

            if Stage.RAW_HIDDEN in wanted:
                stack_4d = stack.view(stack.shape[0], N_TAPS, HIDDEN_DIM)  # [L, 12, 2560]
                for t in range(N_TAPS):
                    pooled = pool_mean(stack_4d[:, t, :].unsqueeze(0), None)[0]
                    tap_pooled[t].append(pooled.cpu())  # h3 contract: artifacts are CPU
            stack_seqs.append(stack.cpu())

            if Stage.PROJECTED in wanted or Stage.CONNECTOR_OUT in wanted:
                txtfusion = self.txtfusion
                if txtfusion is None or self.txtmlp is None:
                    raise RuntimeError("txtfusion/txtmlp not built — load() incomplete")
                dit_device = next(txtfusion.parameters()).device
                with torch.no_grad():
                    # Faithful replication of SingleStreamDiT._forward:
                    # txtfusion(unpack(context)) then txtmlp — on (1, L, 12, 2560)
                    ctx = stack.view(1, stack.shape[0], N_TAPS, HIDDEN_DIM).to(dit_device)
                    projected = txtfusion(ctx, mask=None, transformer_options={})  # [1, L, 2560]
                    if Stage.PROJECTED in wanted:
                        projected_seqs.append(projected[0].float().cpu())
                    if Stage.CONNECTOR_OUT in wanted:
                        connector = self.txtmlp(projected)  # [1, L, 6144]
                        connector_seqs.append(connector[0].float().cpu())

            if (idx + 1) % 10 == 0 or idx + 1 == len(prompts):
                log.info("Encoded %d/%d prompts", idx + 1, len(prompts))

        artifacts: list[ConditioningArtifact] = []

        if Stage.RAW_HIDDEN in wanted:
            streams: dict[str, Stream] = {}
            for t, layer_idx in enumerate(TAP_LAYERS):
                name = f"layer_{layer_idx:02d}"
                acts = torch.stack(tap_pooled[t])  # [n, 2560]
                streams[name] = Stream(name=name, spec=_build_spec(name), activations=acts)
            # tap_stack: full-seq, right-padded; mask carries real positions
            max_len = max(s.shape[0] for s in stack_seqs)
            acts = torch.zeros(len(prompts), max_len, TAP_STACK_DIM)
            mask = torch.zeros(len(prompts), max_len, dtype=torch.int)
            for j, s in enumerate(stack_seqs):
                acts[j, : s.shape[0]] = s
                mask[j, : s.shape[0]] = 1
            streams["tap_stack"] = Stream(name="tap_stack", spec=_build_spec("tap_stack"), activations=acts, mask=mask)
            artifacts.append(self._make_artifact(Stage.RAW_HIDDEN, streams, prompts, categories))

        if Stage.PROJECTED in wanted:
            max_len = max(s.shape[0] for s in projected_seqs)
            acts = torch.zeros(len(prompts), max_len, TXT_DIM)
            mask = torch.zeros(len(prompts), max_len, dtype=torch.int)
            for j, s in enumerate(projected_seqs):
                acts[j, : s.shape[0]] = s
                mask[j, : s.shape[0]] = 1
            spec = _build_spec("txtfusion_out")
            artifacts.append(
                self._make_artifact(
                    Stage.PROJECTED,
                    {"txtfusion_out": Stream(name="txtfusion_out", spec=spec, activations=acts, mask=mask)},
                    prompts,
                    categories,
                )
            )

        if Stage.CONNECTOR_OUT in wanted:
            max_len = max(s.shape[0] for s in connector_seqs)
            acts = torch.zeros(len(prompts), max_len, DIT_DIM)
            mask = torch.zeros(len(prompts), max_len, dtype=torch.int)
            for j, s in enumerate(connector_seqs):
                acts[j, : s.shape[0]] = s
                mask[j, : s.shape[0]] = 1
            spec = _build_spec("connector_out")
            artifacts.append(
                self._make_artifact(
                    Stage.CONNECTOR_OUT,
                    {"connector_out": Stream(name="connector_out", spec=spec, activations=acts, mask=mask)},
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
        if self._te_fake_quant:
            dtype_str = f"fakequant_{self._te_fake_quant}"
        elif self._quantized:
            dtype_str = "comfy_int8_tensorwise"
        else:
            dtype_str = "comfy_bf16"
        meta: dict[str, Any] = {
            "comfyui_commit": comfyui_loader.get_comfyui_commit(self.comfyui_path),
            "quantized": self._quantized,
            "n_layers": N_LAYERS,
            "tap_layers": list(TAP_LAYERS),
            "dit_file": self._dit_name,
            "dit_cond_dequantized": self._dit_dequantized,  # int8 tensors dequantized (0 = plain bf16 DiT)
        }
        return ConditioningArtifact(
            model="krea2",
            backend="Krea2Backend",
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
        self.txtfusion = None
        self.txtmlp = None
        self._te_sd = None
        gc.collect()
        log.info("Krea2Backend closed")
