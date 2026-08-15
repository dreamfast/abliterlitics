# te-krea2 — Krea 2 (K2) text-encoder forensics

Phase 8 of the TE-bench arc (z-image → H3 → **krea2**). Same question as H3:
does the abliterated (heretic) Qwen3-VL-4B damage the conditioning interface
or the generation it drives?

## Non-reportable rule (inherited)

`manifest.json` entries with `smoke: true` are stand-in/verification runs —
reports MUST refuse to publish them. Only smoke=false batteries are citable.
Tonight's runs (RedCraft DiT stand-in) are ALL smoke=true.

## The Krea2 TE interface (probe 2026-08-15, comfy master)

- TE = **full Qwen3-VL-4B** (36 LLM layers + ViT tower; NOT plain Qwen3-4B —
  that was z-image). Comfy file layout: `model.embed_tokens`,
  `model.layers.0..35`, `model.norm`, `model.visual.*` (713 tensors bf16).
- Taps: `hidden_states[2,5,8,…,35]` — 12 layers, `(B, 12, seq, 2560)`,
  flattened to `(B, seq, 30720)`. System+user template prefix STRIPPED
  (comfy/text_encoders/krea2.py `encode_token_weights`), no-think template.
- DiT side (`comfy/ldm/krea2/model.py`): `_unpack_context` → `txtfusion`
  (2 layerwise TextFusionBlocks applied per-tap on `(B·12, seq, 2560)` →
  learned `projector` Linear(12→1) mix → 2 refiner blocks) → `txtmlp`
  (RMSNorm → Linear 2560→6144 → GELU → Linear) → concat with image tokens.
- txtfusion + txtmlp weights are int8-tensorwise in ALL on-disk Krea2 DiTs;
  `projector.weight` F32 [1,12]; single-stream DiT is 6144-wide, Wan2.1 VAE.

## Arms

| arm | file | note |
|---|---|---|
| base | `qwen3vl_4b_base.safetensors` (8.27GB, sha256 `ac78887dbbeb3c3c…`) | built from HF `Qwen/Qwen3-VL-4B-Instruct` via `scripts/build_krea2_base_te.py` (`model.language_model.` → `model.` remap; visual naming already compatible) |
| base_int8 | `qwen3vl_4b_base_int8_convrot.safetensors` (4.50GB, sha256 `98c0fbb5ff99887a…`) | verified comfy int8_tensorwise recipe: scalar scale `max|W|/127`, RTN clamp ±127, blob `{"format": "int8_tensorwise"}`; quantizes EXACTLY the heretic-int8 quant set (356 tensors), embed_tokens kept BF16 (mirrors heretic packaging) |
| heretic | `qwen3-vl-4b-heretic.safetensors` (on disk, publisher) | bf16 abliteration |
| heretic_int8 | `qwen3-vl-4b-heretic_int8.safetensors` (on disk, publisher) | the arm the user's real workflow (`krea2_pro_bird.json`) uses |

Round-trip verification of the quant recipe against the publisher's file:
manual RTN reproduces stored int8 codes exactly (mean|d|=0.000, LLM + visual).
"convrot" appears to be converter branding; the format is plain tensorwise.

## Generation config (disclosed)

- **Tonight (smoke)**: DiT = `redcraft22INT8INT4_2Krea2Edition.safetensors`
  (stand-in until the official Krea2 weights arrive), Wan2.1 VAE, the user's
  `krea2_pro_bird.json` graph converted to API format: 6 steps `res_2s/beta`
  cfg 1 + 2-step `deis_3m/bong_tangent` denoise 0.2 refine pass,
  ConditioningZeroOut negative, SaveImageWebP. No LoRA in this graph.
- **Tomorrow**: official Krea2 DiT swaps into the same graph (txtfusion
  interface identical across all three on-disk DiTs).
- ComfyUI 4090: **port 8190** (compose maps 8190:8188; the 8090 the user
  mentioned is wrong). A/B runs there; the 5090 stays user-owned.

## Tomorrow checklist (for user, ~2 minutes)

The comfy models dir is a read-only mount for the agent. The two built base
files live in `comparisons/te-krea2/models/te/`. Copy them in:

```bash
cp comparisons/te-krea2/models/te/qwen3vl_4b_base.safetensors \
   comparisons/te-krea2/models/te/qwen3vl_4b_base_int8_convrot.safetensors \
   /home/maxx/extra/comfyui-models/text_encoders/
```

That unblocks: (a) full base-vs-heretic A/B via ComfyUI, (b) analysis runs
using the canonical comparison.json paths. Until then, analysis reads the
repo copies (runner remaps paths), and A/B smoke uses heretic_int8 only.

Also planned tomorrow: drop the official Krea2 DiT into
`diffusion_models/` and update `settings.tebench.dit_file`.

## Quirks / gotchas

- HF→comfy remap is ONLY the LLM prefix; visual tower keys are already
  identical (verified set-equal, 315 keys).
- `CLIPLoader` type for this TE is **`krea2`** (not qwen_image — that one is
  plain-Qwen3 z-image style; krea2 wires the 12-tap stack + template strip).
- The bird workflow has TWO KSamplers (6-step gen + 2-step denoise-0.2 refine)
  — A/B patches BOTH seeds, same value.
- Analysis is CPU-only by default (4B TE fits easily; mirrors h3 policy).
- Image metrics reuse `run_img_metrics.py` with `--reference base_int8
  --floor-pair base,base_int8` (seed anchors come from the A/B seed ladder).

## Runbook

```bash
# analysis (interface): base vs each variant
runners/run_te_krea2_analysis.sh comparisons/te-krea2/            # all arms
runners/run_te_krea2_analysis.sh comparisons/te-krea2/ --variant heretic_int8

# A/B generation (ComfyUI 8190)
runners/run_te_krea2_ab.sh comparisons/te-krea2/ --seeds 42,43 --smoke

# image metrics on the A/B set
runners/run_te_krea2_metrics.sh comparisons/te-krea2/
```
