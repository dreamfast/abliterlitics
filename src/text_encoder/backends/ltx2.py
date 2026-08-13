"""LTX-2 text-encoder backend (ComfyUI-native).

Measures conditioning from ComfyUI-format safetensors using ComfyUI's own
model classes. Verified stream layout for the dual_linear LTX-2 22B dev
model (ltx-2.3-22b-dev-fp8.safetensors):

- RAW_HIDDEN: 48 per-layer streams (layer_00 .. layer_47), each mean-pooled
  over real tokens to [n, 3840]. Abliteration damage concentrates in the
  final layers — the per-layer profile is the key forensic signal.
- PROJECTED: video_proj [n, L, 4096] + audio_proj [n, L, 2048] — the dual
  linear projections the DiT consumes (hooked as separate submodule streams
  so per-modality CKA is computable).
- CONNECTOR_OUT: NOT AVAILABLE for dual_linear models — the 1D connectors
  are applied DiT-side (ComfyUI returns unprocessed_ltxav_embeds=True).
  Declared absent rather than silently empty (reviewer finding).

Encoding is per-prompt (batch=1), matching ComfyUI's native conditioning
path exactly — LTXAVTEModel trims left-padding via a batch-total token
count, which misaligns for batches of differing prompt lengths.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from src.text_encoder.artifact import ConditioningArtifact, Pooling, Stage, Stream, StreamSpec
from src.text_encoder.backends import comfyui_loader
from src.text_encoder.backends.base import BaseBackend, HookManager
from src.text_encoder.metrics import pool_mean

log = logging.getLogger(__name__)

# Gemma 3 12B has 48 decoder layers (Gemma3_12B_Config.num_hidden_layers = 48).
# The 49th "layer" in the 3840*49 projection input is the token embedding —
# covered by the projection itself, not a decoder layer output.
N_LAYERS = 48
HIDDEN_DIM = 3840
VIDEO_DIM = 4096
AUDIO_DIM = 2048
PAD_TOKEN_ID = 0  # Gemma pad token (left-padding)


class LTX2Backend(BaseBackend):
    """LTX-2 text encoder: Gemma 3 12B backbone + dual linear projection."""

    def __init__(self, comfyui_path: str | Path = "/comfyui") -> None:
        self.comfyui_path = comfyui_path
        self.model: Any = None
        self.tokenizer: Any = None
        self._te_sd: dict[str, torch.Tensor] | None = None
        self._device = "cpu"
        self.ckpt_id: str = ""

    @property
    def name(self) -> str:
        return "ltx2"

    @property
    def streams(self) -> list[StreamSpec]:
        specs = [
            StreamSpec(
                name=f"layer_{i:02d}",
                source_encoder="gemma3",
                layers=(i,),
                pooling=Pooling.MEAN,
                stage=Stage.RAW_HIDDEN,
                dim=HIDDEN_DIM,
            )
            for i in range(N_LAYERS)
        ]
        specs.append(
            StreamSpec(
                name="video_proj",
                source_encoder="gemma3",
                layers="all",
                pooling=Pooling.NONE,
                stage=Stage.PROJECTED,
                dim=VIDEO_DIM,
            )
        )
        specs.append(
            StreamSpec(
                name="audio_proj",
                source_encoder="gemma3",
                layers="all",
                pooling=Pooling.NONE,
                stage=Stage.PROJECTED,
                dim=AUDIO_DIM,
            )
        )
        return specs

    def load(self, ckpt: str, dtype: str = "bf16", device: str = "cpu", dit_file: str = "") -> None:
        """Load the TE backbone + DIT projection.

        Args:
            ckpt: Path to the TE safetensors file (ComfyUI format).
            dtype: Unused — dtype is derived from the file via llama_detect
                   (reviewer W7: quantized models must use their native path).
            device: "cpu" or "cuda".
            dit_file: Path to the LTX-2 diffusion model safetensors
                      (projection + connector source).
        """
        comfyui_loader.import_comfy(self.comfyui_path, cpu=(device == "cpu"))

        if not dit_file:
            raise ValueError("LTX2Backend requires dit_file (LTX-2 diffusion model safetensors)")

        self.ckpt_id = str(ckpt)
        self._device = device
        self._te_sd = comfyui_loader.load_te_state_dict(ckpt)
        dit_proj_sd = comfyui_loader.load_dit_projection_sd(dit_file)

        self.model = comfyui_loader.create_ltx2_model(self._te_sd, dit_proj_sd, device=device)
        self.tokenizer = comfyui_loader.create_ltx2_tokenizer(self._te_sd)
        # Keep the TE sd for the tokenizer reference but free the bulk:
        # the spiece_model tensor is small; backbone tensors are consumed by load_sd.
        log.info("LTX2Backend loaded: %s (device=%s)", Path(ckpt).name, device)

    def tokenize(self, prompts: list[str]) -> dict[str, Any]:
        """Tokenize prompts (for base/variant alignment assertion).

        Returns dict with 'input_ids' (list of per-prompt id lists, unpadded)
        and 'n_tokens'.
        """
        if self.tokenizer is None:
            raise RuntimeError("Backend not loaded — call load() first")
        all_ids: list[list[int]] = []
        for p in prompts:
            tokens = self.tokenizer.tokenize_with_weights(p)
            ids = [t[0] for t in tokens["gemma3_12b"][0]]
            all_ids.append(ids)
        return {"input_ids": all_ids, "n_tokens": [len(i) for i in all_ids]}

    def encode(
        self,
        prompts: list[str],
        categories: list[str],
        stages: list[Stage] | None = None,
    ) -> list[ConditioningArtifact]:
        """Encode prompts per-prompt (batch=1, ComfyUI-native) with stage hooks.

        One forward per prompt captures all declared stages. Per-layer
        activations are mean-pooled immediately (W2 memory: never accumulate
        unpooled [1, 1024, 3840] tensors).
        """
        if self.model is None:
            raise RuntimeError("Backend not loaded — call load() first")

        wanted = set(stages) if stages else {Stage.RAW_HIDDEN, Stage.PROJECTED}

        hm = HookManager()
        transformer = self.model.gemma3_12b.transformer
        layer_modules = transformer.model.layers

        # RAW_HIDDEN: hook each decoder layer (returns (x, kv) — HookManager
        # captures output[0] = layer output [1, L, 3840]).
        if Stage.RAW_HIDDEN in wanted:
            for i in range(N_LAYERS):
                hm.register("raw_hidden", f"layer_{i:02d}", layer_modules[i])

        # PROJECTED: hook the two Linear submodules of DualLinearProjection —
        # separate per-modality streams (more informative than the 6144 concat).
        if Stage.PROJECTED in wanted:
            hm.register("projected", "video_proj", self.model.text_embedding_projection.video_aggregate_embed)
            hm.register("projected", "audio_proj", self.model.text_embedding_projection.audio_aggregate_embed)

        # Accumulators (CPU fp32)
        layer_pooled: list[list[torch.Tensor]] = [[] for _ in range(N_LAYERS)]  # per-layer [n, 3840]
        video_seqs: list[torch.Tensor] = []  # per-prompt [n_real, 4096]
        audio_seqs: list[torch.Tensor] = []  # per-prompt [n_real, 2048]

        try:
            for idx, prompt in enumerate(prompts):
                tokens = self.tokenizer.tokenize_with_weights(prompt)
                gemma_tokens = tokens["gemma3_12b"]
                # Left-padded id list -> attention mask (1 = real token)
                ids = gemma_tokens[0]
                mask_row = torch.tensor([1 if t[0] != PAD_TOKEN_ID else 0 for t in ids], dtype=torch.int)

                _out, _pooled, _extra = self.model.encode_token_weights(tokens)

                captures = hm.get_captures()

                if Stage.RAW_HIDDEN in wanted:
                    raw = captures.get("raw_hidden", {})
                    for i in range(N_LAYERS):
                        t = raw.get(f"layer_{i:02d}")
                        if t is None:
                            raise RuntimeError(f"Layer {i} hook did not fire")
                        # [1, L, 3840] -> pooled [3840] with mask
                        pooled = pool_mean(t, mask_row.unsqueeze(0))
                        layer_pooled[i].append(pooled[0])

                if Stage.PROJECTED in wanted:
                    proj = captures.get("projected", {})
                    v = proj.get("video_proj")
                    a = proj.get("audio_proj")
                    if v is None or a is None:
                        raise RuntimeError("Projection hooks did not fire")
                    # Projection input is trimmed to real tokens by
                    # LTXAVTEModel — no padding in these sequences.
                    video_seqs.append(v[0])
                    audio_seqs.append(a[0])

                # Free per-prompt captures to bound memory
                captures.clear()
                log.debug("Encoded prompt %d/%d", idx + 1, len(prompts))
        finally:
            hm.clear()

        artifacts: list[ConditioningArtifact] = []

        if Stage.RAW_HIDDEN in wanted:
            streams: dict[str, Stream] = {}
            for i in range(N_LAYERS):
                name = f"layer_{i:02d}"
                spec = StreamSpec(
                    name=name,
                    source_encoder="gemma3",
                    layers=(i,),
                    pooling=Pooling.MEAN,
                    stage=Stage.RAW_HIDDEN,
                    dim=HIDDEN_DIM,
                )
                acts = torch.stack([layer_pooled[i][j] for j in range(len(prompts))])  # [n, 3840]
                streams[name] = Stream(name=name, spec=spec, activations=acts)  # [n, 1, 3840]
            artifacts.append(self._make_artifact(Stage.RAW_HIDDEN, streams, prompts, categories))

        if Stage.PROJECTED in wanted:
            # Right-pad variable-length projected sequences to L_max
            streams = {}
            video_max = max(v.shape[0] for v in video_seqs)
            audio_max = max(a.shape[0] for a in audio_seqs)
            video_padded = torch.zeros(len(prompts), video_max, VIDEO_DIM)
            video_mask = torch.zeros(len(prompts), video_max, dtype=torch.int)
            audio_padded = torch.zeros(len(prompts), audio_max, AUDIO_DIM)
            audio_mask = torch.zeros(len(prompts), audio_max, dtype=torch.int)
            for j, (v, a) in enumerate(zip(video_seqs, audio_seqs, strict=True)):
                video_padded[j, : v.shape[0]] = v
                video_mask[j, : v.shape[0]] = 1
                audio_padded[j, : a.shape[0]] = a
                audio_mask[j, : a.shape[0]] = 1

            for name, acts, mask, dim in (
                ("video_proj", video_padded, video_mask, VIDEO_DIM),
                ("audio_proj", audio_padded, audio_mask, AUDIO_DIM),
            ):
                spec = StreamSpec(
                    name=name,
                    source_encoder="gemma3",
                    layers="all",
                    pooling=Pooling.NONE,
                    stage=Stage.PROJECTED,
                    dim=dim,
                )
                streams[name] = Stream(name=name, spec=spec, activations=acts, mask=mask)
            artifacts.append(self._make_artifact(Stage.PROJECTED, streams, prompts, categories))

        return artifacts

    def _make_artifact(
        self,
        stage: Stage,
        streams: dict[str, Stream],
        prompts: list[str],
        categories: list[str],
    ) -> ConditioningArtifact:
        dtype_str = "comfy_int8_convrot"
        if self._te_sd is not None:
            # Best-effort dtype label from the file
            sample = self._te_sd.get("model.norm.weight")
            if sample is not None and sample.dtype == torch.bfloat16:
                dtype_str = "comfy_bf16"
        return ConditioningArtifact(
            model="ltx2",
            backend="LTX2Backend",
            ckpt_id=self.ckpt_id,
            stage=stage,
            streams=streams,
            prompts=prompts,
            categories=categories,
            dtype=dtype_str,
            device=self._device,
            meta={
                "comfyui_commit": comfyui_loader.get_comfyui_commit(self.comfyui_path),
                "projection_type": getattr(self.model, "text_projection_type", "unknown")
                if self.model is not None
                else "unknown",
            },
        )

    def close(self) -> None:
        """Free model + state dict."""
        self.model = None
        self.tokenizer = None
        self._te_sd = None
        import gc

        gc.collect()
        log.info("LTX2Backend closed")
