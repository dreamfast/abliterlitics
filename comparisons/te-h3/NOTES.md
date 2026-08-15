# te-h3 — MiniMax H3 text-encoder battery (Phase 7)

Origin: p-e-w's reproducible audio-artifact report on H3 with the heretic TE
("inconceivable" → "inconcable", journal #134). Measures what swapping H3's
Qwen3-VL-32B text encoder does at the conditioning interface AND to generated
video+audio. Method mirrors the z-image study (te-zimage).

## Arms

- `base` — qwen3vl_32b_minimax_h3_int8_convrot (the primary base; host-readable)
- `base_nvfp4` — qwen3vl_32b_minimax_h3_nvfp4_awq (**stand-in smoke arm only**;
  host-side analysis IMPOSSIBLE — packed-quant header unreadable by
  safe_open; ComfyUI generates with it fine)
- `heretic` — PLACEHOLDER, user supplies files later. Swap-in runbook at the
  bottom of this file.

## Non-reportable rule (STAND-IN RESULTS)

Everything generated while `heretic` is pending is PLUMBING SMOKE, produced
with the base vs base_nvfp4 stand-in pair. Any tebench_h3_*.json produced by
that pair carries `"smoke": true` and `h3_report.py` MUST refuse to publish
verdicts from smoke runs. Real verdicts (H8-H12) require the real
base-vs-heretic battery.

## Model quirks (probe-verified 2026-08-15, journal #135)

- H3 TE = Qwen3-VL-32B truncated to 50 layers; conditioning = UNNORMALIZED
  layer-49 (0-indexed) output; NOT chat-templated; token tags text=1/vision=0.
- fl2va DiT conditioning modules (host-readable): `condition_proj`,
  `token_refiner`, `adaln_t_table` — analog of z-image cap_embedder/
  context_refiner.
- API workflow gotchas: SaveVideo outputs under "images"+"animated";
  MiniMaxH3TurboLoRA needs explicit low_vram; frames % 17 == 5; seed on
  RandomNoise; filename_prefix carries a subfolder.
- Audio is muxed into the mp4 (stereo AAC 32kHz); decode via PyAV.
- ~175s/clip cold, seconds warm; GPU offload heavy (33GB TE + 21GB DiT on 32GB).

## Generation configuration disclosure

All clips use the community-standard turbo t2v workflow (from the
ComfyUI-MiniMax-H3-Turbo example pack): MiniMaxH3TurboLoRA v4 step600
@ strength 1.0, MiniMaxH3TurboSampler at 6 steps (vs 50 stock), 0.2MP 16:9,
5s (124 frames). Both arms traverse the identical graph, so base-vs-heretic
differences are purely the TE. Scope note: a turbo adapter could amplify
conditioning sensitivity (fewer steps); if the audio battery returns a null
result against p-e-w's observation, rerun at 50 steps stock as the first
follow-up control (phase 8 candidate).

## Quant-floor semantics (disclosed)

The h3 BASE TE is already int8-convrot, so the automatic "fake-quant int8"
floor is a REQUANT floor (per-tensor requant of per-channel int8 weights),
not a pristine-precision floor like z-image's BF16-base. Empirically
non-degenerate (semantic ceiling MSE ≈ 1.96). Damage-in-quant-units numbers
for h3 must be read as "divergence in requant-floor units" — the report
labels them accordingly. The primary h3 anchors are the empirical audio
floors (A: whisper intrinsic on base arm; B: int8-vs-nvfp4 generation band
from the stand-in pair) and the seed-change video anchor.

## Heretic swap-in runbook (when the user supplies the TE files)

1. Copy the heretic TE safetensors into
   /home/maxx/extra/comfyui-models/text_encoders/ (ask user for preferred name;
   suggest `qwen3vl_32b_minimax_h3_heretic_<quant>.safetensors`).
2. Set `heretic.te_file` in this comparison.json to the absolute path; set
   `te_reference` to the MATCHED-QUANT base ("base" if int8/convrot,
   "base_nvfp4" if nvfp4) — H8/H9 verdicts assert this match.
3. Verify host-side readability (`safe_open` header) — if unreadable like
   nvfp4_awq, analysis-side must skip it and that MUST be disclosed.
4. Confirm architecture: 50-layer truncated Qwen3-VL-32B (layer count from
   safetensors keys). If it is NOT (e.g. 8B fallback), every H8-H12 verdict
   must disclose the architecture mismatch.
5. Re-run: T2 analysis (base vs heretic), T3 generation (base + heretic arms,
   seeds 42/43/44), T4 audio battery (real, no smoke flag), T5 video, T6
   directional, T7 report (verdicts unlock).
