# Research Findings & Lessons Learned

Everything discovered during scoping that shapes the design. Each item states the finding, the source, and the design implication. **Known unknowns** (things to confirm against a real loader before trusting a number) are flagged inline with ⚠️.

Sources cited by short tag:
- **[BLOG]** Nathan Sapwell, "Abliterating Gemma 3 12B for LTX-2: Does It Actually Help?" (Jan 2026)
- **[PSA]** p-e-w (Heretic creator), Reddit PSA on r/StableDiffusion (the 235-comment thread)
- **[H3-README]** github.com/MiniMax-AI/MiniMax-H3
- **[H3-CFG]** FastVideo MiniMaxH3Qwen3VL arch config
- **[LTX-WIKI]** deepwiki.com/Lightricks/LTX-2 /3.4-gemma-text-encoder
- **[LTX-README]** LTX-2 packages/ltx-core/README.md

---

## R1. Abliteration is concentrated in the final layer(s) — this is the whole game

**Finding [BLOG]:** Per-layer cosine similarity between base and abliterated Gemma 3 12B: layers 0–47 ≈ 1.0 (essentially identical), **layer 48 = 0.859 (innocent) / 0.820 (taboo).** Abliteration's intervention lands almost entirely in the last layer.

**Why:** Directional ablation (Heretic's core technique; ARA/SOMA in newer versions) modifies residual-stream representations of "harmful" vs "harmless" inputs to confuse the refusal classifier. That classifier reads off the *final* layer, so the edit is localized there.

**Design implication:**
- The metric is most sensitive at the final layer → **always report a per-layer drift heatmap**, not just a single aggregate.
- **How the DiT consumes the encoder determines how much of this localized damage reaches generation.** This is the single most important architectural fact per model (see R2–R4).

---

## R2. H3 consumes a SINGLE layer → damage is transmitted ~100%

**Finding [H3-README]:** *"The H3-Encoder uses the full pretrained weights of Qwen3-VL-32B and provides the hidden states from its **50th layer** to the H3-Omni-Transformer."*

**Finding [H3-CFG]:** `num_hidden_layers: 64`, `hidden_size: 5120`, `text_len: 1024`, `output_hidden_states: True`, `hidden_state_skip_layer: 0`.

**Contrast with LTX-2:** This is the opposite of LTX-2's multi-layer aggregation. Where the author found layer 48 is only ~2% of LTX-2's signal, **H3 takes one layer directly** → whatever divergence abliteration introduces there is transmitted at full weight. **The author's "only 2%" LTX-2 conclusion does NOT transfer to H3.**

⚠️ **Known unknown (load-bearing):** Qwen3-VL-32B has 64 layers but the README says "50th." Two hypotheses:
1. H3 takes **layer 50 of a full 64-layer** Qwen3-VL (a deliberate choice of an intermediate, possibly pre-refusal, layer).
2. The **published checkpoint is a 50-layer truncation** — supported by the existence of `ethanfel/ComfyUI-H3-Qwen3VL-TextGen`, which "*reconnects the published generation tail*" to make the encoder generate text again, and the Reddit note that "the version of Qwen3-VL used for Minimax is ~8GB smaller... it lacks output capability."

   → **First task of the H3 backend: determine which.** Load the published ckpt, count transformer blocks, confirm the extraction index from the real loader. This decides whether the headline is "near-100% transmitted" (hypothesis 1) or "transmitted from a pre-refusal truncation point" (hypothesis 2) — and likely explains a lot of conflicting anecdotal reports.

**Design implication:** H3 is the **sharpest test case** and the contested one (the PSA thread is overwhelmingly about H3). It is the headline model. Single-stream backend (one conditioning stream), simplest backend, validates the core fast.

---

## R3. LTX-2's conditioning pipeline is multi-stage, NOT just "aggregate 49 layers"

**Finding [LTX-WIKI]:** The Gemma text encoder path is:

```
GemmaTextEncoder.encode()                 # runs self.model.model, skips lm_head (-~500MB VRAM)
   → raw hidden states (all layers), padded to max_length=1024
EmbeddingsProcessor.process_hidden_states()
   → FeatureExtractor   (norm + linear projection)
   → convert_to_additive_mask
   → Embeddings1DConnector   (a small TRANSFORMER: self-attn + RoPE + RMSNorm + learnable_registers)
   → conditioning into DiT
```

**The author's "layer 48 = 2%" reasoning inspected only the `aggregate_embed` projection weight** (`[3840, 49, 3840]` reshaped → 49 layers, ~equal ~2.04% each, layer 48 rank 26/49). That is a **lower-bound proxy**, because the trailing **`Embeddings1DConnector` is a full transformer block** — non-linear, capable of amplifying or nulling specific directions regardless of their projection weight.

**Design implication:**
- Measure at **multiple stages**: `raw_hidden` → `projected` (after `aggregate_embed`) → `connector_out` (after the 1D connector). The connector-out stage is the honest "what the DiT sees."
- The "2% revisited" result (re-measure at connector stage with CKA + noise floor) is a real publishable correction to the author's own prior post.

**Feature extractor variants [LTX-WIKI]:**

| Variant | Models | Norm | Projections |
|---|---|---|---|
| **V1** | 19B | per-segment: `_norm_and_concat_padded_batch` (masked mean + range norm) | single `aggregate_embed` |
| **V2** | 22B | per-token RMS (`norm_and_concat_per_token_rms`) + `_rescale_norm` | **dual: `video_aggregate_embed` + `audio_aggregate_embed`** |

⚠️ **Known unknown:** Which extractor variant does LTX-2.5 (Gemma 4) use? Likely V2 (dual video/audio) given its joint audio-video nature, but confirm from `encoder_configurator.py` / the shipped config.

---

## R4. LTX-2.5 has separate VIDEO and AUDIO conditioning streams — a novel measurement axis

**Finding [LTX-README]:** *"LTX-2.3 uses stock Gemma 3 (12B), downloaded separately; **LTX-2.5 uses a Gemma 4 (unified) fine-tuned for LTX** and shipped with the model. Advanced text understanding is critical not only for global language support but for the **phonetic and semantic accuracy of generated speech**."*

**Finding [LTX-WIKI] (V2 extractor):** dual projections `video_aggregate_embed` and `audio_aggregate_embed` feed separate conditioning streams.

**Implication:** Because LTX-2.5 is joint audio-video, abliteration damage to the **speech/phonetic** conditioning can be measured *independently* of the visual conditioning. Nobody has reported this split. A plausible, citable result: *"heretic perturbs LTX-2.5's audio conditioning stream more/less than its video stream"* — directly relevant to anyone using LTX-2.5 for dialogue/voice. This is the project's most novel finding opportunity.

**Design implication:** The artifact abstraction must support **multiple named streams per model** (even though we dropped Flux's T5+CLIP multi-encoder case, multi-stream from one LLM still matters here).

---

## R5. Abliteration removes refusal; it does NOT add knowledge (the confound)

**Finding [PSA, BLOG]:** p-e-w: *"H3 doesn't 'refuse', it simply lacks the knowledge to generate certain imagery. The only fix is training."* The author's own post-conclusion: *"abliteration removes refusal behavior, but it doesn't add knowledge... when we measured embedding differences we might have been comparing two models that both have weak/vague representations of taboo content."*

**Design implication — the deepest methodological risk:** Low divergence between base and abliterated does **not** mean "safe/faithful" — it might mean "both are bad." Two defenses baked into the method (`METRICS.md` §4):
1. **Semantic ceiling reference** (`div(prompt_A, prompt_B)`) calibrates what a real representational difference looks like.
2. **Category-discrimination score** — if divergence tracks the innocent→extreme gradient (the author's cosine did: 0.869 → 0.825), the encoder still *distinguishes* the concepts. A *flat* profile is the red flag.

We explicitly do **not** claim "this proves uncensoring works." We claim "this quantifies conditioning perturbation." Keeping that line clean is what makes the work credible to both camps.

---

## R6. What the PSA actually establishes (so we don't relitigate it)

The 235-comment thread ([PSA]) is the social context. Settled points our work **assumes, doesn't re-prove**:

1. Abliteration is an LLM technique; it operates on the decoder's residual stream. → **scope is LLM-TE models only** (R-scope).
2. Using an abliterated TE as the *conditioner* cannot add knowledge the DiT lacks. → We never claim it can.
3. p-e-w *guesses* it "if anything makes outputs worse" but **explicitly hedges ("if anything") because he hasn't tested it.** → That hedge is the gap we fill.
4. The one valid LLM-TE use is **prompt expansion** (text→text, before the TE). Out of scope for *conditioning* measurement, but we note it.
5. Two Redditors (`afinalsin`, `bassbin`) ran informal swaps and got conflicting vibes; both admit "purely vibes based, no A/B." → Our tool is the A/B they lacked.

**Stance:** The project is *measurement*, not advocacy. Whoever's priors it confirms, the numbers are the contribution.

---

## R7. The cross-encoder "different model, different output" effect is real but orthogonal

**Finding [PSA, afinalsin]:** Swapping to a *different finetune* (not just abliteration) of a TE produces "consistently different" output — a vague style shift, not uncensoring. This is real (any weight change perturbs conditioning) but **not what we measure**. We measure base-vs-modified-of-the-*same*-lineage. A different-lineage TE swap is a separate experiment we can optionally add as a "style perturbation" reference baseline.

---

## R8. Quantization damage is measurable with the same machinery (the second goal)

**Finding [NanoSputnik, PSA]:** Comfy (comfyanonymous) has repeatedly stated GGUF is a poor format for diffusion text encoders — "both low quality and slow," `int8` convrot is better. Yet GGUF TEs are widely used for VRAM.

**Design implication:** The exact same metrics + noise floor quantify this with no new theory. The **quant sweep** (fp8, nxfp4, GGUF Q2–Q8) becomes a "damage ranking table" on the same axis as abliteration — directly answering "is heretic damage ≪ or ≫ one fp8 step?" This is the second headline and shares 100% of the core code.

**Note on GGUF TE support:** ComfyUI-GGUF support for some Gemma TEs is experimental/incomplete (city96/ComfyUI-GGUF#402, referenced in [BLOG]). The quant sweep may need the `gguf` library directly + a custom dequant-to-safetensors step rather than loading through ComfyUI. Flagged in `PLAN.md` Phase 6.

---

## R9. Multilingual drift is an under-explored, likely-real finding

**Reasoning:** Abliteration refusal directions are extracted from (predominantly English) probe sets. A taboo concept that survives only as a euphemism in language X, while its English synonyms are "nuked," may drift asymmetrically across languages. Nobody has measured this.

**Design implication:** Prompt suite includes a multilingual stratum. Cross-lingual divergence profile is a free, novel sub-result.

---

## R10. The "last token" measurement was throwing away most of the signal

**Finding [BLOG]:** The author's cosine measurement used `outputs.last_hidden_state[:, -1, :]` — the **last token only**. But the DiT cross-attends to the **full token sequence**.

**Design implication:** `cos_all_tok` averages over all non-padded tokens (`METRICS.md` §3.1). We keep the last-token number too, for direct comparability with the author's published table (acceptance test), but the full-sequence number is the trustworthy one.

---

## Summary table: what each model needs

| Need | H3 | LTX-2 | LTX-2.5 |
|---|---|---|---|
| Backbone | Qwen3-VL-32B (LM part) | Gemma3ForConditionalGeneration (`.model`, skip lm_head) | Gemma4 (unified) |
| Extraction | single layer (50?) ⚠️ | all layers | all layers |
| Extractor | (H3's own projection) ⚠️ | V1 (aggregate_embed) | V2 (video + audio) ⚠️ |
| Streams | 1 | 1 | 2 (video, audio) |
| Novelty | headline (contested, single-layer) | acceptance test (reproduce blog) + "2% revisited" | video/audio split |
