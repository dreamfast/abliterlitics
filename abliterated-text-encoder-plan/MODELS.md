# Per-Model Backend Specs

Implementation-level detail per backend. Each section: backbone, extraction point, streams, weight mapping, validation gate, and **⚠️ known unknowns** to resolve before trusting a number.

Common notes:
- All backends load the LM backbone with `torch_dtype` per the `dtype` arg, run forward with `output_hidden_states=True`, and capture stages via hooks (`ARCHITECTURE.md` §3).
- One forward pass per (ckpt, prompt-batch); all stages captured together.
- All metric math cast to fp32 on artifact construction; source dtype recorded.

---

## H3 — `backends/h3.py`  (headline model)

### Backbone
- **Qwen3-VL-32B**, language-model part only (`language_model.model`), vision tower unused for text-only encoding.
- `hidden_size = 5120`, `num_hidden_layers = 64` (per FastVideo config), `text_len = 1024`.
- Skip the lm_head (H3's encoder is a conditioning extractor, not a generator; the published ckpt "lacks output capability").

### Extraction
- **Single layer**, README-stated index 50. Capture that layer's hidden states `[n, L, 5120]` at `Stage.RAW_HIDDEN`.
- Then the H3 conditioning projection → `Stage.PROJECTED` / `Stage.DIT_INPUT`.

### Streams
- One stream: `layer50_hidden` (and `layer50_projected` at the projected stage).

### Checkpoints to compare
- **Base:** MiniMax-H3 published text encoder (`MiniMaxAI/MiniMax-H3` → `text_encoder/`).
- **Variant (abliterated):** `ethanfel/Qwen3-VL-32B-Ultra-Heretic-MiniMax-H3-ComfyUI-INT8-ConvRot` (INT8 ConvRot), or `Momoking/Qwen3-VL-32B-Heretic-MiniMax-H3-NVFP4` (NVFP4, 15.7GB).

### ⚠️ Known unknowns (resolve first — `PLAN.md` T4.1)
1. **Is the published ckpt a 50-layer truncation, or layer-50-of-64?** Load it, count `model.language_model.layers`, confirm the extraction index from the real loader (FastVideo `MiniMaxH3Qwen3VLConditioner` or ComfyUI's loader). *This decides whether damage is "near-100% transmitted" (layer 50 of full) vs "transmitted from a pre-refusal truncation" (50-layer ckpt).*
2. **Where exactly is the conditioning projection?** Find the proj module that maps `[L, 5120]` into the H3-Omni-Transformer's cross-attention dim. Hook its output for `Stage.PROJECTED`.
3. **Special tokens:** H3 adds tokens (e.g. `<d>`). Confirm the chat template / tokenization the DiT was trained against, so base & variant tokenize identically (assertion in `suite.py`).

### Validation gate
- Capture one prompt's conditioning through ComfyUI's H3 T2V pipeline (disable prompt-expander; raw prompt only) and assert our standalone forward matches within bf16 tolerance.

### Why it's the headline
- Contested (the PSA thread is ~all about H3).
- Single-layer consumption → damage transmitted at full weight, the sharpest contrast to LTX-2's "~2%". (`RESEARCH.md` R2)

---

## LTX-2 — `backends/ltx2.py`  (reference + acceptance-test model)

### Backbone
- **Gemma 3 12B** via `Gemma3ForConditionalGeneration`; use `self.model.model` to skip lm_head (~−500MB VRAM). Padded to `max_length = 1024`.
- Source: `ltx-core` `GemmaTextEncoder` / `EmbeddingsProcessor`.

### Extraction (multi-stage)
| Stage | Module hook | Tensor |
|---|---|---|
| `RAW_HIDDEN` | backbone, all layers | hidden states `[n, L, 4096]` × n_layers |
| `PROJECTED` | `feature_extractor` output (post `aggregate_embed`) | `[n, L, 3840]` |
| `CONNECTOR_OUT` | `Embeddings1DConnector` output | `[n, L', d_dit]` (the honest DiT input) |

### Feature extractor: **V1** (19B models)
- `_norm_and_concat_padded_batch`: masked mean + range norm, per segment/batch/layer.
- Single linear: `aggregate_embed` `[3840, 188160]` (= `3840 × 49 layers × 3840`). The author reshaped this to inspect per-layer weights → found ~equal ~2.04% per layer, layer 48 rank 26/49.

### Streams
- One stream at each stage: `gemma_hidden` / `aggregate_proj` / `connector_out`.

### Weight mapping (SDOps)
- `language_model.model.*` → `model.model.language_model.*`
- `text_embedding_projection.aggregate_embed.*` → `feature_extractor.aggregate_embed.*`

### Acceptance test (the built-in oracle — `PLAN.md` T3.2)
- At `RAW_HIDDEN`, last-token cosine, the author's published table:
  - innocent 0.869, mild 0.850, violent 0.836, sexual 0.827, extreme 0.825 (±tolerance).
- **If we don't reproduce this, the core is wrong — fix before anything else.**

### "2% revisited" result (T3.5)
- Re-measure at `CONNECTOR_OUT` with CKA + noise floor. Hypothesis: the connector (a full transformer with self-attn + RoPE + RMSNorm + learnable registers) reshapes the damage non-linearly, so the true transmitted divergence differs from the 2% projection-weight proxy. (`RESEARCH.md` R3)

### ⚠️ Known unknowns
- Confirm V1 vs V2 for the exact 19B ckpt used in ComfyUI (the author used LTX-2 19B).
- `Embeddings1DConnector` learnable-register injection: confirm register tokens are excluded from the conditioning comparison (they're pad-replacements, not prompt tokens).

### Checkpoints
- Base: stock Gemma 3 12B + LTX-2 19B feature_extractor/connector weights.
- Variant: `DreamFast/gemma-3-12b-it-heretic` (the author's own upload) + same LTX projector/connector.

---

## LTX-2.5 — `backends/ltx25.py`  (novelty: video/audio split)

### Backbone
- **Gemma 4 (unified) 12B**, fine-tuned for LTX, shipped with the model. Loaded via `ltx-core` Gemma path (ComfyUI: `comfy/text_encoders/gemma4.py`).

### Extraction (multi-stage, dual projection)
| Stage | Module hook | Streams |
|---|---|---|
| `RAW_HIDDEN` | backbone, all layers | `gemma4_hidden` |
| `PROJECTED` | `feature_extractor` V2 outputs | `video_proj`, `audio_proj` |
| `CONNECTOR_OUT` | connector outputs (video, audio) | `video_connector`, `audio_connector` |

### Feature extractor: **V2** (22B models) ⚠️ confirm
- `norm_and_concat_per_token_rms` (per-token RMS norm) + `_rescale_norm`.
- **Dual** linear projections: `video_aggregate_embed` and `audio_aggregate_embed`.

### Streams (the novelty)
- **Two independent conditioning streams** from one LLM. We measure each separately → "does abliteration damage the speech/phonetic conditioning more/less than the visual conditioning?" (`RESEARCH.md` R4) This is unreported.

### Weight mapping
- `text_embedding_projection.video_aggregate_embed.*` → `feature_extractor.video_aggregate_embed.*`
- `text_embedding_projection.audio_aggregate_embed.*` → `feature_extractor.audio_aggregate_embed.*`

### ⚠️ Known unknowns
1. Confirm Gemma 4 layer count and which layers LTX-2.5 aggregates (all? a subset?).
2. Confirm V2 extractor + exact norm/rescale for the shipped LTX-2.5 ckpt.
3. Confirm whether audio stream length differs from video (phoneme/token alignment).

### Checkpoints
- Base: shipped LTX-2.5 Gemma 4 encoder.
- Variant: `DeepNeuralNerd/Gemma-4-12B-it-uncensored-heretic-DeepNeuralNerd-LTX_2.5_ComfyUI`.

### Validation gate
- Same as LTX-2: one-prompt ComfyUI capture, assert standalone forward reproduces conditioning.

---

## Cross-cutting backend concerns (`backends/base.py`)

- **Hook manager:** register/unregister forward hooks; collect into a `dict[Stage, dict[stream, Tensor]]`; detach + move to the artifact dtype.
- **Dtype/device policy:** parameterized; base and variant in a comparison must match.
- **Memory:** for 32B (H3) / 12B models, encode in batches (e.g. 8–16 prompts), accumulate activations on CPU/GPU-pinned, del backbone between base and variant loads.
- **Tokenization alignment helper:** `assert tokenize(base) == tokenize(variant)` per prompt; on mismatch (GGUF drift), record in artifact `meta` and surface in the report rather than silently aligning.
