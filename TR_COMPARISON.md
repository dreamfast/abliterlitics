# TR_COMPARISON.md — The Abliterated Text-Encoder Controversy: Complete Reference

> **Purpose**: single source of truth for drafting the te-zimage report, the HuggingFace
> blog post, and the Reddit post. Everything we learned from the community controversy,
> the academic literature, the Discord feedback, and our own methodology decisions —
> with citations.
>
> **Comparison**: `comparisons/te-zimage/` — Z-Image Turbo (DiT) + Qwen3-4B text encoder,
> 27 variant TE files (abliteration × quantization factorial).
> **Status**: **ALL MEASUREMENTS COMPLETE** (Phase 4 + Phase 5, 2026-08-14). Final
> verdicts + numbers ledger in §8.1–8.3; report at
> `comparisons/te-zimage/graphs/te_report.html`. Remaining: §11 drafting items.
> **Journal cross-refs**: #113 (ComfyUI API), #115 (prior-art landscape), #116 (KL
> provenance), #117 (terminology), #118 (KL-analog + dose-response), #122 (Phase-4
> plan), #124 (image ladder final), #125–127 (reviews + word readout), #128–131
> (Phase-5: breadth, controls floor, H6 taboo, completion).
> **Identity disclosure**: DreamFast (author of the Heretic TE releases) is a principal
> on this project. Disclose in publications.

---

## 1. The controversy in one paragraph

Modern image/video generators (Minimax H3, Z-Image, Krea 2, FLUX.2 Klein) use full LLMs
(Qwen3 / Qwen3-VL) as text encoders. The community started swapping these for
"abliterated"/"heretic" versions of the same LLM, believing it would remove censorship
from image outputs. In Aug 2026, **p-e-w — the creator of Heretic itself — posted a PSA**
(r/StableDiffusion, 256 comments) stating this **does not work and cannot work**: abliteration
edits the refusal circuitry of a model whose *encoding* pathway was never the censor; the
DiT receives *perturbed* representations it was never trained on, which "either has no
effect at all, or reduces prompt adherence and potentially introduces artifacts."
**Nobody on either side has ever measured it.** The counter-position is also unmeasured
("worked for me 🤷‍♂️"). Both positions are theoretical or anecdotal. Our te-zimage
comparison is the first empirical instrument for the question.

---

## 2. The Reddit thread (r/StableDiffusion, posted ~2026-08-13)

**Post**: "PSA: I'm the creator of Heretic, and I advise you to *not* use 'heretic' models
as text encoders for H3 (or any other model)" — by **u/-p-e-w-**.

### 2.1 The core claims (verbatim, quotable)

> "Heretic uncensors LLM responses through directional ablation (or related techniques
> like ARA and SOMA in newer versions). Roughly speaking, it modifies the model's internal
> representations (residual vectors) of 'harmful' inputs to resemble those of 'harmless'
> inputs to confuse the model into treating the former like the latter and comply with
> the request rather than refusing."

> "But this intervention does not produce representations of inputs that are more 'raw',
> more 'graphic', more 'anatomically correct' or similar compared to the original model.
> In fact, LLMs already produce highly accurate internal representations of harmful
> inputs by default, which is why they are able to classify them correctly and generate
> a refusal."

> "So when the hidden states from an 'uncensored' LLM are passed to the diffusion model
> (…) the second model isn't magically seeing clearer representations of the bad stuff
> you requested. On the contrary, it's seeing slightly perturbed representations compared
> to what it was trained on. This either has no effect at all, or the effect of reducing
> prompt adherence and potentially introducing artifacts. **But it will never, ever
> remove censorship from the output.**"

> "(Note: Generation models like Ideogram that can actively refuse prompts are potentially
> an exception to this rule and might be amenable to abliteration, but only with an
> approach that significantly differs from how Heretic works today.)"

**Key concessions/nuances from p-e-w in the comments**:
- Prompt-expander use is valid: "That's correct, for this use case Heretic models can be
  valuable, because some LLMs outright refuse to process certain prompts at all. This
  comes before the text encoder though." (reply to nvidiot)
- On DiT-side censorship: "You can't [abliterate H3]. H3 doesn't 'refuse', it simply
  lacks the knowledge to generate certain imagery. The only fix is training." (reply to
  hurrdurrmeh)
- "The key insight is that while both the LLM and the diffusion model are 'censored',
  they are censored in fundamentally different ways." (reply to Infamous_Campaign687)
- On the mechanism of refusal vs encoding: refusal lives in the *decoding* stages
  (building output text) which "are never used and often not even included" in TE-only
  deployments (SkoomaDentist's phrasing, endorsed by thread).

### 2.2 The measurable-hypothesis comment (bassbin — our methodology, proposed as a guess)

> "My vibe hypothesis is that the degradation is probably proportional in some way to the
> kl divergence (or the actual difference in output/model states that it represents)
> between the 'base' model and the abliterated model."

Note: bassbin proposes correlating image degradation with an LLM-side metric. Our design
replaces the guess with direct measurement at the conditioning interface + DiT trajectory,
with the KL-pathway caveat (§5.2).

### 2.3 The "nobody has tested it" admission (afinalsin)

> "Nah, there isn't really evidence one way or another unfortunately. My little run above
> is about the closest I've seen to an actual comparison. Everyone who doesn't use
> finetuned encoders say 'Nuh-uh' without doing proper A/B testing, and everyone who does
> use them says 'Yuh-huh', also without bothering to A/B testing, so both sides of the
> discussion are purely vibes based."

afinalsin's own test: 8 random Qwen3-4B *finetunes* vs base on Z-Image Turbo, fixed
prompt/seed, XY-plot album — "The changes introduced by the text encoders were negligible
in a lot of cases, and preferable in a couple, with only very short prompts showing any
meaningful change." Also documented the *actual* failure mode being attacked: the
Krea-2-style **prompt expander** refusing benign prompts ("a chair in an empty room" →
safety lecture). This is the expander-vs-encoder conflation at the heart of the folklore.

### 2.4 Other notable comments (quote bank)

- **ThatsALovelyShirt** (mechanism explainer): "It's passing the 'conditioning', the
  internal representation of the concepts the text of the prompt encodes (…) The Qwen 3 VL
  model is not 'thinking' about the input. (…) the version of Qwen 3 VL used for Minimax
  is ~8GB smaller than the real one, **it lacks output capability**."
- **FaceDeer vs whiteweazel21** — the epistemology clash:
  - whiteweazel21: "Try Krea2 with no lora and no bypass. Same seed. Official text encoder
    and heretic encoder. Prompt for a woman with big breasts and see the exact opposite of
    what OP is saying." / "Worked for me 🤷‍♂️"
  - FaceDeer: "This is like insisting that putting a magnet on top of your monitor makes
    image generation faster. You can insist 'but it works for me' all you want, it's a
    claim that has no basis in any known way that reality functions."
- **ArchAngelAries** (the honest user report): "I used the heretic Qwen 3vl text encoder
  because of the the Krea2 base workflow refusing to provide nudity in prompt refinement,
  and while the heretic encoder did refine the prompt as expected, **the quality and
  coherence suffered massively**. I've since shifted to using heretic models separately
  for prompt refinement while using the official encoder for the actual nsfw image
  generation. Leaps and bounds better."
- **mgtowolf**: "I did A/B testing with a bunch of my model test prompts and saw not much
  of a difference at all between the two."
- **ImpressiveSuperfluit** (prior-era anecdote): used a random different TE for ZIT
  because it imparted a preferred "style" — "I could've maybe picked out 60/100 outcomes
  at best, it wasn't that much above chance (…) Does fuck all for censoring though."
- **cyborgsnowflake** (steelman): "Isn't it possible to introduce censorship behavior
  into the encoder when it 'detects' a bad prompt and steers encoding in a certain
  direction? And wouldn't heretic 'flatten' that sort of thing?" — Murky-Relation481
  counters: "No steering is occurring, no inference is occurring at all in the text
  encoder."
- **FotografoVirtual** (the one legitimate theoretical path): "couldn't a diffusion model
  have learned these internal 'harmful/refusal' vectors during training, perhaps
  associating them with blurred, pixelated, or low-quality content? If so, wouldn't an
  abliterated text encoder that avoids generating these specific vectors theoretically
  help?" — explicitly *not* claiming it happens in practice.
- **LoveByForce** (counter-claim, unverified): "This is definitely incorrect for Qwen
  Image. The abliterated text encoder literally makes the difference between a clothed
  man and a woman (…) and a woman lying on her back (…)" — no controls stated.
- **mrjackspade** (p-e-w's stated exception, in progress): on abliterating
  refusal-capable models like Ideogram: "I'm actually working on this now and I can
  confirm that it does work (…) as is expected of a first pass, the stronger the effect
  the less likely the model is to follow the prompt."
- **NanoSputnik** (GGUF tangent, relevant to our GGUF ladder): "Comfy himself explained
  multiple times why this is a bad idea [GGUF for diffusion models] (…) they are bad for
  diffusion models. Both low quality and slow. int8 convrot will always be better."
- **Hearmeman98** (template author): updated the popular Docker template to a heretic TE
  before the PSA, then: "Thanks for the explanation. I did follow the general opinion
  rather than fully understand it. I will modify my template back to the original text
  encoder." → **The community's adoption→reversion cycle happened with zero measurements
  on either side. This is the narrative arc our post completes.**
- **PromptAfraid4598** (flux-era precedent): "Similar answers could be easily obtained as
  early as flux.1; replacing the text encoder has almost no effect."

### 2.5 Thread facts useful for framing

- Heretic has produced **>5000 community "heretic" models** (per p-e-w).
- The popular hearmeman Docker template had *just* switched to a heretic TE when the PSA
  landed — timing is why the thread exploded.
- Community consensus after thread: heretic LLM = valid for prompt expansion/captioning
  (text-out pathways); heretic TE = at best neutral, at worst degrading (unmeasured).

---

## 3. Discord feedback (HTB server, 2026-08-13/14) — and our resolutions

**VINAY**:
> "Maybe the seeding is only for DiT and not for text encoders. And even if text encoders
> have seeding then minor differences in its output for DiT can cause those visual
> differences. It is very normal"
> "there can be minor differences in the words even if the seed and prompt are same,
> because the model weights are different, calling them a 'damage' is incorrect, anyway
> it depends you benefit or loose anything from abliterated text encoders"

**BirdNest (= DreamFast)**:
> "Yeah the idea was to see visually how the difference between the bf16 happens, so seed
> has to be the same. I noticed people say that an abliterated text encoder 'damages' or
> degrades the output, but for this I can't see it. Sure, it's different. But I don't
> think it means it's damaged. Edit: Although it does lose prompt adherence for the
> lowest ggufs and heretic int4, it's cool to see what actual damage is required to
> measure that."
> "The embeddings are slightly different for each model quant, which is why the image
> ends up different. However, we can still measure bf16 heretic to bf16 base, and see
> divergence. So divergence is probably a better word. I was aiming for something similar
> to kl divergence in the token context, which cannot be applied to the text encoder
> context at all, so I am working to find the closest thing."

**Resolutions (journal #117, #118)**:
1. **TE encode is deterministic — no seed exists in that pathway.** Observed "minor
   differences with same seed+prompt" come from: different weights (our measured object),
   prompt-EXPANDER sampling (different pathway entirely), or tiny GPU kernel
   nondeterminism (plan: two-process same-encode control to state that floor).
2. **Chaotic sampling**: any nonzero conditioning delta can flip a trajectory at fixed
   seed → single fixed-seed image pairs are *illustrative, not evidentiary*. Statistics
   live at suite level; the different-seed control in `run_dit_divergence.py` anchors
   "normal variation" ("X% of a seed change").
3. **Terminology tiers**: divergence (measure) → quantization-error vs abliteration-shift
   (typed by intent) → **damage reserved for collateral, unintended divergence**
   (operationalized as `orth_fraction` — energy orthogonal to the taboo direction — plus
   functional consequence in images).
4. **BirdNest's int4/Q3 observation = the calibration anchor**: known-positive controls
   for the dose-response curve.

---

## 4. Academic literature (verified this session)

### 4.1 The mechanistic basis of abliteration

- **Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda (NeurIPS 2024)**, "Refusal in
  Language Models Is Mediated by a Single Direction", arXiv:2406.11717.
  https://arxiv.org/abs/2406.11717 · code: github.com/andyrdt/refusal_direction
  Refusal is a 1-D subspace across 13 open chat models; orthogonally projecting it out of
  activations removes refusal; adding it induces refusal on benign prompts. **This is the
  theory our `directional.py` tests at the conditioning interface**: is the heretic delta
  aligned with the base-model taboo direction (intended shift) or orthogonal noise
  (collateral)? Permutation nulls give z-scores.

### 4.2 What DiTs actually consume from text encoders (the paper that frames everything)

- **Spingarn, Cohen, Rott Shaham, Michaeli (Jun 2026)**, "Text-to-Image Models Need Less
  from Text Encoders Than You Think", arXiv:2606.03715.
  https://arxiv.org/abs/2606.03715 · project: nsping13.github.io/contextless-TTI/
  Diffusion transformers rely on only (i) merged per-word representations and (ii) word
  order from positional embeddings; a **bag of position-tagged words** (context-free)
  embedding guides generation on par with full contextual embedding — "the decoding of
  complex linguistic structures is performed by the image model itself."
  **Implication for us**: predicts DiTs are *insensitive* to contextual-subspace
  perturbation but sensitive to word-meaning corruption. Testable in our data as: (a)
  category-gradient flattening at low quants (word-meaning corruption) vs (b) pure shift
  under abliteration (largely contextual?). If heretic divergence is invisible at the
  image level, this paper explains why. **Cite prominently.**

### 4.3 Quantization of T2I models — the gap we fill

- **Q-Diffusion** (Li et al., ICCV 2023): "Q-Diffusion: Quantizing Diffusion Models" —
  PTQ for the *denoiser*, notes direct linear quantization causes significant quality
  drops. openaccess.thecvf.com (ICCV 2023).
- **DGQ** (ICLR 2025): "Distribution-Aware Group Quantization for Text-to-Image Diffusion
  Models" — activation outliers in *cross-attention* drive quality loss; group-wise
  logarithmic quantization.
  https://openreview.net/forum?id=ZyNEr7Xw5L
- **QLIP** (Jul 2025): "Text Embedding Knows How to Quantize Text-Guided Diffusion
  Models", arXiv:2507.10340 — prompt-derived bit-precision assignment for the diffusion
  model. Closest title-match to "text + quantization" but still quantizes the **DiT**, not
  the TE. https://arxiv.org/abs/2507.10340
- **Survey** (2025): "Diffusion Model Quantization: A Review", arXiv:2505.05215.
  https://arxiv.org/abs/2505.05215
- Engineering evidence that TE quantization is *presumed* benign: **Nunchaku** (mit-han-lab)
  W4A16 T5-XXL for FLUX — "maintaining text encoding quality that matches FP8 T5" with
  no published measurement methodology.
  https://deepwiki.com/mit-han-lab/nunchaku/2.4-text-encoder-quantization
- **The gap**: the entire T2I-quantization literature quantizes the denoiser/DiT. **No
  work isolates text-encoder quantization → conditioning damage, and none addresses
  abliterated TEs at all.** Our ladder (ConvRot int8/fp8/mxfp8/nvfp4/int4 + GGUF
  F16→Q3_K_M, each with matched-quant abliterated twin) is the first.

### 4.4 TE replacement / importance studies

- **"Scaling Down Text Encoders of Text-to-Image Diffusion Models"** (CVPR 2025),
  arXiv:2503.19897 — distilled T5-base ≈ T5-XXL quality at 50× smaller: DiTs are robust
  to major TE changes. https://arxiv.org/abs/2503.19897
- **CompVis/stable-diffusion issue #35** (historical): Imagen's T5-XXL > CLIP on
  compositional prompts per human evals — the original "TE matters" evidence.
  https://github.com/CompVis/stable-diffusion/issues/35
- **Qwen-Image Technical Report** (2025), arXiv:2508.02324 — production LLM-as-TE
  architecture (the lineage Z-Image belongs to). https://arxiv.org/abs/2508.02324

### 4.5 Where censorship actually lives (DiT-side; the complement to our study)

- **ESD**: Gandikota et al., "Erasing Concepts from Diffusion Models", arXiv:2303.07345.
  https://arxiv.org/abs/2303.07345 · erasing.baulab.info
- **MACE**: MAss Concept Erasure via closed-form cross-attention refinement + LoRA
  (cite by name; described in Semantic Scholar snippet adjacent to ESD and in the
  taxonomy below).
- **"When Are Concepts Erased From Diffusion Models?"** (2025), arXiv:2505.17013 —
  stress-tests 7 erasure methods; erased concepts remain recoverable.
  https://arxiv.org/abs/2505.17013 · github.com/kevinlu4588/WhenAreConceptsErased
- **Safe Latent Diffusion** (Schramowski et al., 2023) — safety guidance in latent space
  during sampling (cite by name; no verified arXiv ID this session).
- **Framing**: these fine-tune or guide the *DiT* — the layer where p-e-w says censorship
  actually lives ("it simply lacks the knowledge… the only fix is training"). Our study is
  the encoder-side mirror image: we measure what the DiT *receives*, they edit what the
  DiT *does*.

### 4.6 Abliteration studies (LLM-side; none touch the conditioning interface)

- **"A Granular Study of Safety Pretraining under Model Abliteration"** (2025),
  arXiv:2510.02768 — 10 base models × abliterated twins, 100 prompts, judge-based refusal
  rates. https://arxiv.org/html/2510.02768v1
- **Latent Fusion Jailbreak** (2025), arXiv:2508.10029 — interpolating harmful/harmless
  hidden states at chosen layers/tokens to lose refusal. Confirms hidden-state-level
  intervention is the active mechanism. https://arxiv.org/html/2508.10029
- **ricyoung/abliteration-comparison** (GitHub) — comparative weight/benchmark analysis of
  abliteration techniques. https://github.com/ricyoung/abliteration-comparison
- **mlabonne, "Uncensor any LLM with abliteration"** (HF blog, 2024) — the canonical
  tutorial that started the wave. https://huggingface.co/blog/mlabonne/abliteration
- **Sapwell, "Uncensored LLM Abliteration Benchmarked: HauhauCS vs Heretic vs Huihui"** —
  community weight-analysis + benchmark comparisons across 5 Qwen models.
  https://nathan.sapwell.net/posts/hauhaucs-abliteration-analysis/
- **officechai** article on a non-RLHF paper: notes an abliterated model at **KL 0.27**
  from base — useful as a "how big do LLM-side KLs get" contrast point.
  https://officechai.com/ai/non-rlhfed-models-say-theyre-alive-unlike-models-with-rlhf-says-independent-paper/
- **Khanna, "I Abliterated Gemma 4 on a MacBook"** — practitioner how-to; clean statement
  of the mean-difference direction recipe. https://www.riteshkhanna.com/blog/abliterate-gemma
- Our own AGENTS.md lesson #19 (gemma4-12b journals #57–#62): Heretic v1.2.0 = v1.4.0 =
  Abliterix numerically identical on harmless prompts; dataset choice moves KL ~30×
  (0.0005–0.05); ranking stable ρ≈0.9. **Cite when discussing any published KL.**

### 4.7 Prompt-text robustness (adjacent, distinct)

- **"Evaluating the Robustness of Text-to-image Diffusion Models against [textual
  perturbations]"** (OpenReview x7kxQv0vJW) — perturbing the *input text*, not the
  encoder. Our problem is the dual: same text, perturbed encoder.
  https://openreview.net/pdf?id=x7kxQv0vJW

### 4.8 The metric lineage

- **Kornblith, Norouzi, Lee, Hinton (ICML 2019)**, "Similarity of Neural Network
  Representations Revisited" — CKA. https://cka-similarity.github.io ·
  github.com/google-research/google-research/tree/master/representation_similarity
  Our `cka_linear`/`cka_rbf` follow this directly. Known caveat (documented in
  `metrics.py`): positive bias when n < d — used for ranking, not absolutes.
- FID (Fréchet distance between Gaussian-fit embedding clouds) is the literature's
  "distributional divergence over embeddings" precedent — we deliberately do NOT use a
  conditioning-FID as headline (n ≈ tens of prompts ≪ d = 2560 → rank-deficient
  covariance); our semantic-units ratio is the robust analog (journal #118).

---

## 5. KL divergence: applicability, provenance, and the retraction

### 5.1 Why KL cannot apply at the conditioning interface

The DiT consumes deterministic continuous tensors (hidden states → projection →
`consumed_hidden`). No sampling, no probability space, no logits. Softmax-normalizing a
hidden state would be semantically garbage. KL exists only on the **LLM-side generative
pathway** (attach lm_head → logits → log_softmax) — which the image pipeline **never
executes** (Z-Image's Qwen3-4B TE deployment excludes the output head; cf.
ThatsALovelyShirt's "~8GB smaller, lacks output capability"). KL filters the perturbation
through a readout the DiT never sees. All te-zimage variants correctly carry
`skip_kl: true`.

### 5.2 Card-KL provenance chain — and the author retraction

```
Qwen3-4B base (BF16, transformers)
  ├─ Heretic abliteration (BF16)          ← card KLs measured HERE (LLM-side,
  │                                          pre-conversion, Heretic's prompt set)
  ├─ ComfyUI TE export (repackage, BF16)  ← unmeasured by any card number
  └─ Quant: int8/fp8/nvfp4/mxfp8/int4     ← unmeasured (our ladder fills this)
     + GGUF F16…Q3_K_M                    ← unmeasured (our ladder fills this)
                                            ↑ te-zimage measures everything at the
                                              conditioning interface
```

- comfyui-wiki coverage of DreamFast/Qwen3-VL-4b-Heretic-ComfyUI quotes **KL 0.0283**,
  HarmBench ASR 100% (base 30.8%), 54 tensors modified (pure rank-1), MMLU +0.03,
  GSM8K −1.83%.
- The qwen3-4b-heretic card (via mirror README) states **"KL Divergence: 0.0000 (zero
  measurable model damage)"**, Heretic v1.2.0, 200 trials (trial 96 selected), refusals
  3/100 vs 100/100 base.
- **AUTHOR RETRACTION (2026-08-14, this conversation)**: DreamFast states the near-zero
  figure is not trustworthy ("it is so low it must be a mistake"). **Report rule: cite no
  card KL as evidence.** If an LLM-side KL is ever needed (e.g., to test bassbin's
  correlation hypothesis), re-measure with our frozen pipeline (lesson #19: dataset
  choice moves KL ~30×; harmless-dataset Heretic KLs of 0.0005–0.002 are documented, so
  "plausibly tiny" ≠ "verified").

### 5.3 The structural KL-analog we adopted (journal #118)

KL = excess surprise relative to self-surprise. Embedding-space analog:
`damage_semantic_units = MSE(base, variant) / mean pairwise MSE(base, base)` — divergence
normalized by the distribution's own meaning-scale. Secondary anchors: quant-units (one
ConvRot-INT8 step) and seed-change units (different-noise control). One-sentence report
answer: *"There is no KL at the conditioning interface; the structural analog is
meaning-scale-normalized divergence, dual-anchored to a quantization step and a seed
change."*

---

## 6. Novelty map — what exists vs what we're first at

| Question | Prior art | Status |
|---|---|---|
| Does abliteration remove LLM refusal? | Arditi 2024; Heretic >5000 models; arXiv:2510.02768 | Established |
| Does abliteration preserve LLM capability? | Community benchmarks (Sapwell; our gemma4-12b) | Established, noisy |
| Can the *DiT* be uncensored? | ESD/MACE/erasure line (fine-tune DiT) | Established (DiT-side) |
| Do DiTs need rich contextual TE info? | Spingarn 2606.03715; arXiv:2503.19897 | Established (they need less than thought) |
| Does quantizing the *DiT* hurt? | Q-Diffusion/DGQ/QLIP/survey | Established (DiT-side) |
| Does TE quantization hurt conditioning? | Nunchaku marketing claim only | **Measured (ours)** — clean ≤ int8/fp8/Q6; visible ≥ 4-bit tiers, dose-ordered (§8.2) |
| Do abliterated TEs uncensor image outputs? | p-e-w PSA (theory: no) vs user claims (anecdote: yes) | **Measured (ours)** — no uncensoring effect; base DiT doesn't refuse in the first place (§8.1 H6) |
| Do abliterated TEs *degrade* outputs? | p-e-w theory; ArchAngelAries anecdote; bassbin KL-guess | **Measured (ours)** — visible shift ≈ nvfp4, adherence/uniformity/collapse all clean (§8.2) |
| Is the heretic delta a direction or noise, at the conditioning interface? | — | **First (directional.py)** — z 50–134 vs permutation null |
| Does the DiT trajectory respond to the conditioning delta? | — | **First (run_dit_divergence.py)** — response flat at 1.0–1.9% of the different-image anchor |
| Intended-vs-collateral decomposition of the abliteration shift | — | **First (orth_fraction + z-scores)** |
| Abliteration × quantization factorial with matched-quant twins | — | **First (27-file ladder)** |
| Noise-equivalent dose of a real abliteration edit (per unit MSE) | — | **First (low-dose bracket + floor, §8.2)** |
| Taboo-tier uncensoring A/B with sanitized-twin compliance gap | — | **First (H6 battery, §8.2)** |

---

## 7. Our instrument (mapping claims → code)

- **Pipeline stages captured**: TE hidden states → DiT text projection → `consumed_hidden`
  (what actually enters cross-attention). `src/text_encoder/artifact.py`, `cache.py`,
  `run_analysis.py` (ComfyUI's own model classes; ltx2 + zimage backends).
- **Metrics** (`metrics.py`): linear+RBF CKA (Kornblith), cosine all-tokens vs last-token
  (blog comparability), MSE/L2, Spearman rank-correlation of pairwise prompt distances
  (anisotropic warp detector), per-dimension drift.
- **Anchored units** (`compare.py`): quant-units = MSE / MSE(base, int8-ConvRot floor);
  semantic-units = pooled MSE / cross-prompt pairwise MSE. Bootstrap CIs (joint resample).
  Note: quant floor is a **learned** quant (ConvRot, SVD-guided rounding) → conservative.
- **Directional analysis** (`directional.py`): taboo direction = mean(taboo) −
  mean(innocent) from base; per-prompt delta cosine; `orth_fraction` = collateral share;
  toward-innocent signed projection; permutation nulls (n=200) → z-scores.
- **DiT-step divergence** (`run_dit_divergence.py`): paired Euler flow-matching
  trajectories from identical noise; per-step velocity + latent divergence;
  **different-seed control = "totally different image" anchor**.
- **Real-stack A/B** (`run_ab_test.py`): ComfyUI API, patched workflow JSON, CLIPLoader
  swap (safetensors) / CLIPLoaderGGUF (.gguf), fixed seed/steps/sampler, cold-start via
  /free + VRAM polling (lesson #113). Pairs are illustrative; statistics at suite level.
- **Suite** (`suite.py`): stratified innocent/mild/violent/sexual/extreme/multilingual.
  DreamFast's prior blog finding: cosine gradient innocent > mild > violent > sexual >
  extreme (encoder still separates taboo) — carry through as the category-gradient
  analysis; flattening at low quants = word-meaning corruption signal (Spingarn frame).
- **Design**: 27 variants = {base, heretic} × {bf16, int8, fp8, nvfp4, int4, mxfp8} +
  {base, heretic} × GGUF {F16, Q8_0, Q6_K, Q5_K_S, Q5_K_M, Q4_K_S, Q4_K_M, Q3_K_M}.
  `te_reference` gives every heretic_X a matched base_X → abliteration isolated at every
  quant dose; base_X vs base-BF16 → pure quantization dose.

---

## 8. Hypotheses (pre-registered) → final verdicts in §8.1

- **H1 (p-e-w's claim)**: heretic-vs-base divergence at matched quant is nonzero but
  small relative to quant-step and seed anchors → "different, not damaged" at BF16.
- **H2 (dose-response)**: image-level adherence loss has a measurable threshold in
  quant-units; int4/Q3_K_M sit above it, Q6_K+ below (BirdNest's observation → number).
- **H3 (directionality)**: heretic delta aligns with base taboo direction (z ≫ 2) at
  mid-upper layers; orth_fraction low at BF16, growing as quant dose rises (quant noise
  is direction-agnostic).
- **H4 (Spingarn)**: abliteration shift lives mostly in contextual subspace (invisible to
  DiT); quantization error corrupts word-meaning directions (visible) — tested via
  category-gradient flattening + rank_correlation.
- **H5 (bassbin)**: LLM-side KL (if re-measured) does NOT predict conditioning/image
  divergence better than our direct metrics → KL is the wrong instrument for TE choice.
- **H6 (uncensoring efficacy)**: A/B taboo-category images show no uncensoring gain from
  heretic TE at matched quant (p-e-w prediction) — the benefit, if any, is expander-side.
- **H7 (conversion purity)**: base ComfyUI TE ≡ base transformers BF16 weights (appendix
  check; expected pass, closes the provenance chain).

### 8.1 Final verdicts (2026-08-14, pre-registered P1–P11)

| Hyp | Claim | Verdict | Key evidence |
|---|---|---|---|
| H1 | Abliteration at BF16 = small shift, not damage | **REFUTED with caveat** | heretic TE-swap LPIPS **0.286** = 1.5× floor band (above pre-registered "small") — but ≈ unmodified base_nvfp4 (0.274) and 55% of seed anchor: *visible, not catastrophic* |
| H2 | Quant dose threshold (BirdNest's observation → number) | **SUPPORTED** | clean tier (int8/fp8/Q6+/mxfp8/F16) max 0.190 ≤ band 0.238; 4-bit tiers min 0.310 > band; abliteration adds ≤ +0.149 LPIPS on top of clean quants |
| H3 | Abliteration delta is a direction, not noise | **SUPPORTED** | cos(Δ, taboo) ≈ −0.15 (removes taboo-ward component), permutation z 50–134; honest caveat: 97–99% of delta energy is orthogonal |
| H4 | DiT privileges meaning over order (Spingarn) | **REFUTED** | meaning-shuffle 0.02099 vs order-shuffle 0.02194 (1.0×) — no privilege at this interface |
| H5 | Conditioning divergence predicts image divergence (bassbin KL-proxy) | **SUPPORTED (the non-prediction)** | every variant's trajectory response stays at 1.0–1.9% of the different-image anchor regardless of conditioning magnitude — KL-proxy has no predictive power here |
| H6 | Abliterated TE uncensors image outputs | **MIXED → effectively NO** | compliance-gap deltas +0.001/−0.002/−0.006 vs int8 control band ±0.003 (n=12/tier); only out-of-band tier (gore) moves *negative*. **Base finding: base stack renders the explicit tier at 100% NSFW-classifier rate — no rendering-stage censorship exists to remove** |
| H7 | ComfyUI export weight-faithful | **NOT RUN** | stretch goal, untouched |
| P8 | CLIP prompt adherence preserved | **SUPPORTED** | heretic Δ −0.21 CLIP pts; every clean encoder within ±1.5; only 4-bit tiers beyond ±2 |
| P9 | Structured edit ≥2× less disruptive per unit MSE than noise | **SUPPORTED (at the floor)** | interpolated noise-equivalent dose 0.42× heretic's own MSE; caveat: dose-independent floor below ~2× heretic MSE |
| P10 | Shift uniform across prompt categories | **SUPPORTED** | worst (multi_subject 0.260) = 1.28× overall (0.202) ≤ 1.5× rule; anchor spread across categories 1.09× |
| P11 | Uncensoring on ≥2/3 tiers outside control band | **NOT MET** | 0 tiers positive-outside-band (see H6) |

### 8.2 Final numbers ledger

**Anchors (LPIPS, bootstrap CI'd)** — the units everything reads against:

- seed change (same TE, different noise): **0.516** — "a different picture"
- quant-floor band (int8 ↔ fp8, both known-good): **0.190**; in-band rule = ×1.25 → 0.238

**Perceptual ladder** (TE-swap LPIPS vs int8 ref; matched-frame vs base-BF16 cross-check):

- quant tiers dose-ordered: q8 0.133 → … → int4 0.492
- heretic BF16: **0.286** (vs base-BF16 ref: 0.290) — between floor and 4-bit tiers
- abliteration on clean quants: roughly additive **+0.149**
- inter-prompt discriminability (collapse check): worst ratio 0.799 (heretic_fp8) — no collapse

**Controls battery** (final latent divergence, 27 conditions):

- Spingarn shuffles 0.021–0.022; noise dose curve 0.0172 (floor) → 0.038 (2×)
- heretic α=1: **0.01756** — ON the floor; heretic α=10 (matched to noise 1× dose):
  **0.0324 vs noise 0.0320** → no hidden subspace (P5: DiT does not ignore the refusal direction)
- **LOW-DOSE FLOOR (new, Phase 5)**: Gaussian noise at {0.005–0.1}× of the int4
  max-dose (91.5; heretic's own MSE 1.34) all deflects 0.0172–0.0195 — flat/inverted,
  dose-INDEPENDENT. The sampler has a deflection floor, not a dial; heretic's real
  edit sits on it. Per-MSE comparisons at that scale are below sampler resolution.

**Attention readout** (JSD, scale 0–0.69):

- heretic **0.0007** vs base_int4 **0.0074** (10×) — abliteration leaves *where the DiT
  looks* intact; heavy quant breaks it. Two channels: heretic changes only *what
  arrives* (content), 4-bit changes both.
- prompt = ~7% of image-token attention total; ~42% of the prompt budget sits on chat
  scaffolding (`<|im_start|>…`)

**CLIP adherence (P8)**: heretic −0.21 pts; all clean encoders in ±1.5 band.

**Breadth (P10)** — per-category LPIPS vs base:

| encoder | landscape | portrait | text_render | multi_subject |
|---|---|---|---|---|
| heretic | 0.117 | 0.213 | 0.221 | 0.260 |
| base_int8 | 0.039 | 0.083 | 0.091 | 0.146 |
| base_int4 | 0.300 | 0.391 | 0.474 | 0.478 |
| *seed anchor* | *0.488* | *0.509* | *0.523* | *0.531* |

(multi_subject is the most sensitive category for EVERY encoder — task property, not
an abliteration signature.)

**Taboo battery (H6)** — 18 adults-only prompts × 3 tiers × 2 seeds × 4 encoders
(ethics: no minors, no real people; images local-only, report embeds metrics):

- compliance-gap deltas (heretic − base): tier1_nudity **+0.001**, tier2_explicit
  **−0.002**, tier3_gore **−0.006**; int8 negative-control band **±0.003**
- NSFW-classifier rates: base tier2_explicit **100%** (nothing to uncensor); tier1
  base 33% vs heretic 25% (no uncensoring signal)
- base tier2 compliance gap +0.077 is the HIGHEST of all tiers — base doesn't refuse

### 8.3 The physics in three sentences (the arc's spine)

1. **Two lossless encoders differ by 0.19 LPIPS** — CFG-1 turbo sampling chaotically
   amplifies ANY conditioning difference; same-seed difference ≠ damage.
2. **Below a threshold the response is dose-independent** — the sampler has a
   deflection floor (0.017–0.019) that every sub-threshold kick lands on, regardless
   of size.
3. **The abliteration sits on that floor** — per unit MSE it's ~0.42× as disruptive as
   Gaussian noise, invisible to the DiT's attention, adherence-preserving,
   category-uniform, and it neither damages nor unlocks anything measurable: it
   re-rolls the dice.

---

## 9. Framing gifts for the blog/Reddit draft

1. **The arc**: community adopted heretic TEs → the creator of Heretic said stop →
   template author reverted → *nobody ever measured anything* → we built the instrument.
2. **The poster child quote** (afinalsin): "both sides of the discussion are purely vibes
   based." Our post ends that sentence.
3. **The author twist**: the person whose models are at the center of the controversy ran
   the first measurement study — and the study tests claims *made against his own models*
   as rigorously as claims for them (disclose identity; let H1–H6 fall where they fall).
4. **The retracted KL**: even the "zero measurable damage" number on the card was
   unverified — a small parable about the state of measurement in this niche.
5. **The one-sentence method**: *"Same-seed image differences are expected from chaotic
   sampling and are not evidence of damage; we measure divergence at the conditioning
   interface, decompose it into intended (directional) vs collateral (orthogonal), and
   anchor it to a quantization step, a meaning change, and a seed change."*
6. **Terminology table** (use verbatim in posts): divergence = measurement ·
  quantization error / abliteration shift = typed by intent · damage = collateral only.
7. **Answer to "what's the KL?"**: §5.3's one-liner. Pre-empt the smartest objection.
8. **The H6 twist** (strongest single finding): "we finally tested whether abliterated
   TEs uncensor image generation — and discovered the base model never refused in the
   first place" (100% base NSFW-classifier rate on the explicit tier; highest
   compliance gap of all tiers). The debate was arguing about a lock on an open door.
9. **The floor**: "the sampler doesn't measure how big your edit is — below a
   threshold, any kick produces the same deflection." Makes the 0.19 int8↔fp8 gap and
   the 0.29 heretic gap the same phenomenon at different magnitudes.
10. **7% / 42%**: the DiT spends ~7% of its attention on the prompt, and ~42% of that
    on chat-template boilerplate. Puts the entire "TE quality" panic in proportion.
11. **Two channels**: "where it looks vs what it receives" — heretic changes only the
    content channel (JSD 0.0007); 4-bit quants break both (0.0074 + adherence loss).
    One clean idea that separates the two failure modes the community conflates.

---

## 10. Link index (all sources this session)

**Community / releases**
- Reddit PSA thread — r/StableDiffusion, u/-p-e-w-, ~2026-08-13 (full text captured in
  conversation; quotes in §2)
- p-e-w/Heretic — https://github.com/p-e-w/heretic
- DreamFast/Qwen3-VL-4b-Heretic-ComfyUI — https://huggingface.co/DreamFast/Qwen3-VL-4b-Heretic-ComfyUI
- DreamFast/Qwen3-VL-4b-Heretic — https://huggingface.co/DreamFast/Qwen3-VL-4b-Heretic
- DreamFast/Qwen3-VL-4b-Heretic-GGUF — (linked from card)
- DreamFast/qwen3-4b-heretic — https://huggingface.co/DreamFast/qwen3-4b-heretic
  (mirror README: https://dev.modelhub.org.cn/hf/DreamFast-qwen3-4b-heretic/src/branch/main/README.md)
- comfyui-wiki coverage — https://comfyui-wiki.com/en/news/2026-07-16-qwen3-vl-4b-heretic-comfyui
- silveroxides/convert-to-quant — https://github.com/silveroxides/convert_to_quant
- dreamfast/heretic-docker — https://github.com/dreamfast/heretic-docker
- city96/ComfyUI-GGUF — https://github.com/city96/ComfyUI-GGUF
- Z-Image resource page (Qwen3-4B GGUF TE) — https://z-image.me/en/resources/qwen3-4b-gguf
- Third-party civitai (opposite marketing claim) —
  https://civitai.red/models/2731465/ ("ensuring your unrestricted prompts remain highly
  detailed" — asserts benefit, unmeasured)
- Third-party civitai ("neither necessary nor sufficient") —
  https://civitai.red/models/2728378/
- Uncensored Qwen3-4B trained TE (civitai) — https://civitai.com/models/2373678/
- Nunchaku quantized TEs — https://deepwiki.com/mit-han-lab/nunchaku/2.4-text-encoder-quantization
- Qwen3.5-9B abliterated guide — https://codersera.com/blog/unrestricted-uncensored-qwen35-9b-abliterated-full-guide/

**Papers** — arXiv IDs: 2406.11717 (Arditi refusal direction) · 2606.03715 (Spingarn,
contextless T2I) · 2507.10340 (QLIP) · 2505.05215 (DM quantization review) ·
2503.19897 (scaling down TEs) · 2508.02324 (Qwen-Image report) · 2303.07345 (ESD) ·
2505.17013 (When Are Concepts Erased) · 2510.02768 (safety pretraining under
abliteration) · 2508.10029 (Latent Fusion Jailbreak) · Kornblith 2019 CKA
(cka-similarity.github.io) · Q-Diffusion (ICCV 2023) · DGQ (ICLR 2025,
openreview.net/forum?id=ZyNEr7Xw5L) · SD issue #35 (github.com/CompVis/stable-diffusion/issues/35)
· OpenReview x7kxQv0vJW (T2I textual robustness) · mlabonne abliteration blog
(huggingface.co/blog/mlabonne/abliteration) · ricyoung/abliteration-comparison (GitHub)
· nathan.sapwell.net/posts/hauhaucs-abliteration-analysis/ · officechai KL-0.27 article ·
riteshkhanna.com/blog/abliterate-gemma

**Internal**
- Journal: #113, #115 (prior-art), #116 (KL provenance), #117 (terminology), #118
  (KL-analog + dose-response) — `./abliterlitics.sh db journal list --comparison te-zimage`
- AGENTS.md lessons #19 (KL cross-tool validation), #20/#32 (empty-response forensics —
  pattern precedent for "3-number" reporting), #35 (sub-precision merges — precision-floor
  precedent)
- Code: `src/text_encoder/{metrics,compare,directional,run_analysis,run_dit_divergence,
  run_ab_test,suite}.py` · `comparisons/te-zimage/comparison.json`

---

## 11. Open items for the drafting phase

- [ ] Re-verify DreamFast card KLs (or strike from all drafts) per §5.2 retraction.
- [x] ~~H2 graded adherence~~ — done: CLIP-score in the ladder + dedicated P8 section
      (heretic −0.21; clean band ±1.5; 4-bit tier visible).
- [x] ~~H6 uncensoring study~~ — done (Phase 5): MIXED/effectively-no; base renders
      explicit tier at 100% NSFW rate — nothing to uncensor at the rendering stage.
- [x] ~~P5/P9 hidden-subspace + noise-equivalent dose~~ — done: heretic_a10 ≈ noise_1x
      (no subspace); 0.42× heretic-MSE (at the dose-independent floor).
- [ ] GPU-nondeterminism floor: two-process same-encode control (one line in report).
- [ ] H7 conversion-purity check (base TE file vs transformers BF16 weights).
- [ ] Decide whether to test H5 (re-measure LLM-side KL with frozen pipeline) —
      nice-to-have, not load-bearing.
- [ ] Drafting: HF blog (full, from §8 + §9 gifts) + Reddit condensed; identity
      disclosure (DreamFast) up front; terminology table verbatim (§9.6).
