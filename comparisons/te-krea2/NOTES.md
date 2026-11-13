# NOTES — te-krea2 (Krea 2 Turbo × Qwen3-VL-4B TE)

### Status: FULL BATTERY COMPLETE — GGUF arms included in ALL instruments (2026-08-17)

28 arms (12 safetensors + 16 gguf), 16-prompt text-rendering suite, seeds 42/43 + anchor top-up 44-47.
GGUF arms now run forensics + attention too (commit 1e7a451: text-stack load; visual tower absent-by-design,
inert for text conditioning). te_gguf_ladder.svg has data. DB: 2940 te_results + 27 attention.

### Headline results
- **Heretic reroll fraction: 33.5%** (0.1858 / seed anchor 0.5552, n=512) — the whole abliteration moves a krea2 render by a third of a seed change. z-image was 56%.
- Image ladder: Q8_0 26.9% < bf16-floor 28.2% < **heretic 33.5%** < Q4_K_M 40.3% < nvfp4 ~50% < int4 64%
- Forensics: heretic connector_out 0.28 quant-units (sub-quantum-step; cos 0.998 at tap_stack)
- Attention (turbo, 16 prompts): heretic JSD 0.00236 — between fp8 and int8 quants; int4 = 4× the edit
- Report: `graphs/te_report.html` + `te_report.md` + `results/.../ladder_summary.{json,md}`

### How to re-run pieces
```bash
bash runners/run_te_krea2_ab.sh              # generation (or python -m src.text_encoder.run_krea2_ab ...)
bash runners/run_te_krea2_analysis.sh ./comparisons/te-krea2/
bash runners/run_te_img_metrics.sh --force ./comparisons/te-krea2/
bash runners/run_te_krea2_attention.sh --n-prompts 16 --device cuda ./comparisons/te-krea2/
python3 -m src.text_encoder.ladder_summary --results-dir ... --out-dir ...
# report (container):
docker run --rm -v $(pwd):/app -w /app -e PYTHONPATH=/app --entrypoint python3 \
  abliterlitics-te:1.2.0 -m src.text_encoder.generate_report \
  --results-dir /app/comparisons/te-krea2/results/te-krea2/text_encoder \
  --output-dir /app/comparisons/te-krea2/graphs --split-images
```

### Gotchas (see AGENTS lesson 45 + journals #155-#158)
- GGUF arms need mmproj sidecars (3 files incl. `-f16` and `-heretic` twins) + the ComfyUI-GGUF
  patch (`scripts/patch_comfyui_gguf_qwen3vl.py`, applied in-container 2026-08-16).
- Bad-CLIP node-cache: if a GGUF arm fails with the 2560 error and the server log shows NO
  mmproj attempt → restart comfyui-5090 (retry loop POSTs /free but only a restart clears it).
- Turbo DiT = BARE keys; loaders fixed (c9dae2c/a15463e/cfd629a).
- Attention OOM if comfy holds VRAM → POST /free first (overnight.sh does this now).
- z-image-only instruments not ported: per-step divergence, dit-divergence response curve,
  category_gradient (needs taboo-suite categories). (GGUF analysis-space IS done — 1e7a451.)
- Container-written files under results/ + graphs/ need chown (done).

### Models (comfy models dir, text_encoders/qwen3-vl-4b/)
26-file ladder: base + heretic × {bf16,int4,int8,fp8,mxfp8,nvfp4} safetensors (all 356-key
matched sets, embed BF16 — fp8 embed-quantized bug FIXED in newer files) × 8 gguf quants +
imatrix pair + 3 mmproj sidecars (towers byte-identical base/heretic, 315/315).
Parent-dir `qwen3-vl-4b-heretic_fp8_e4m3fn.safetensors` is the STALE embed-quantized one —
delete when convenient.
