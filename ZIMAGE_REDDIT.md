# Z-Image Reddit post draft

Target: r/StableDiffusion, as a follow-up to the Heretic PSA thread. Post Tue-Thu 8am-12pm EST, be in the comments for the first 2 hours. This file stays untracked, never commit.

## Title options

1. "I measured what an abliterated text encoder actually does to Z-Image. 27 encoders, 2240 same-seed images, real numbers." (110 chars, slightly long)
2. "The abliterated text encoder debate was pure vibes. I built the instruments and measured it. 27 encoders, here are the numbers." (127, long)
3. "I measured the abliterated text encoder debate: 27 encoders, one Z-Image, actual numbers" (89 chars, fits)
4. "27 text encoders, 2240 same-seed images: what abliteration actually does to Z-Image" (84 chars)

Recommendation: option 3 or 4.

**The post**

---

After [the PSA from Heretic's author](https://reddit.com/r/StableDiffusion/comments/1vmdxzk/psa_im_the_creator_of_heretic_and_i_advise_you_to/) the debate ran hot and both sides were pure vibes. Same-seed screenshots both ways, nobody measuring anything. The instruments did not exist. So I built them. They cover quants as well, so the encoder swap and the compression get read with the same rulers.

**TL;DR:** the abliterated encoder does not uncensor Z-Image, because the render stage never refused in the first place, base already renders the explicit tier at a 100% classifier rate. The swap itself is a visible but harmless image change, about the size of a stock nvfp4 quant, roughly half a seed change. Every functional metric stays clean.

Disclosure since it matters here: I release the heretic text-encoder for people to use, qwen3-4b-heretic included. My first release last year got replies that I didn't fully understand how text encoders work. They were right. I did my own deep dive and concluded that they are good for prompt enhancement and just change the image slightly, there's no harm in using them if you really want to. Lets see if my conclusion is correct, while also addressing with proof and data the experiences other people have had.

I've been comparing and benchmarking abliterated LLMs under the name Abliterlitics. And this is a first as we've delved into the ComfyUI world to get some solid data to cut through the nonsense.

**What I did**

Base Qwen3-4B and its heretic twin across 6 safetensors formats and 8 GGUF rungs, 27 encoders total, every heretic build matched to a base build at the same quant so the abliteration and the compression can be read separately. Then: conditioning tensors captured at three pipeline stages, paired sampling trajectories from identical noise, 2240 same-seed renders scored with LPIPS and CLIP, attention readouts, and a taboo comparison with sanitised-twin controls.

Two rulers make everything readable. Two encoders nobody argues about, int8 and fp8, differ by 0.19 LPIPS at the same seed. A seed change alone is 0.52. Any swap scoring under 0.19 is indistinguishable from ordinary compression. Near 0.52 is just a different picture.

An explanation of our measurements, metrics and the full report with an interactive A/B gallery can be found here [abliterlitics.dev/posts/z-image-text-encoder](https://abliterlitics.dev/posts/z-image-text-encoder/).

All of what u/-p-e-w- stated in his post is correct. He did hint that there may be degradation or damage, however it was framed as a *maybe* if I was reading correctly. So lets see what that damage is, if at all, and if it makes any difference.

**The questions people were actually arguing about**

**Does the base encoder refuse your prompt before the image model sees it?**

No. I encoded refused-vocabulary prompts to the exact tensor entering cross-attention and checked which base word each heretic vector lands closest to. All 12 test words decode to themselves, cosine floor 0.9967. Pornographic decodes to pornographic, beheading to beheading. The encoder hands the DiT the word intact. It was never the censor. An abliterated text encoder *does not* change the way the model understands the prompt at all. The base text encoder already knows these things.

**Do refused words, or any part of the prompt at all arrive corrupted?**

No. Worst sentence-level cosine between base and heretic on refused prompts is 0.9985. The shift is 3.3 to 6.6 times larger on refused prompts than innocent ones, so the edit concentrates where it acts, but the meaning survives it. Even int4 and Q3, visibly degraded, keep mean CLIP adherence in band. Across every encoder we tested, even the 4-bit tiers, mean CLIP adherence stays in band. The model understands the prompt throughout.

**Does it uncensor anything?**

No, and the reason is better than expected. The unmodified base stack already renders the explicit tier at a 100% NSFW-classifier rate, and the explicit tier owns the highest compliance gaps in the whole set. There is no render-stage censorship to remove. The debate argued about a lock on an open door. This matches where the research says engineered censorship lives, in the diffusion model's own weights: [ESD](https://arxiv.org/abs/2303.07345) and [MACE](https://arxiv.org/abs/2505.17013) erase concepts by fine-tuning the DiT, not the encoder.

**Does it damage outputs?**

The images change, the outputs do not degrade. Heretic vs base is 0.286 LPIPS, 1.5x the trusted band, but a stock nvfp4 quant of the base encoder moves images 0.274 and nobody calls that sabotage. Prompt adherence: -0.21 CLIP points, and the unmodified bf16 base itself reads -0.28 against the same reference. Attention readout moves 0.0031 vs int4's 0.0149. Output separation 1.049, no collapse. Different, not damaged.

**Why do people see differences then?**

Because seeing a difference is the default. Two trusted encoders already differ by 0.19 at the same seed, sampling is a butterfly machine, and below a threshold the response is dose-independent anyway. I also checked per-prompt: 71 of 540 CLIP rows cross the ±2 line on individual prompts while every mean stays in band. Single-prompt screenshots are real but they are noise, not signal.

**What about quantised encoders?**

The GGUF ladder is dose-ordered: the F16 container is a true round trip, 0.0008 quant units with cosine 1.0. Q8_0 costs 0.34. Q3 costs 83 and is visibly paying. Being precise about Q8_0 since the numbers deserve it: its conditioning perturbation is real and measurable, CI 0.29 to 0.39 quant units, but a third the size of what int8 ConvRot itself costs, and at the image level Q8_0 and bf16 are indistinguishable, 0.138 vs 0.152 LPIPS against the int8 reference with overlapping CIs. So the near-lossless claims for both hold where it shows, in the images. Q8_0's real cost is load time. One caution: stacking heretic on fp8 loses 20% of output separation, the only borderline damage in the whole study. Don't stack the abliteration on compression.

**So when should I use one?**

Anywhere the model writes text that feeds the next stage: prompt expansion, captioning, image description. Those are chat pathways and abliteration works on chat pathways. If a stage only embeds text, an abliterated encoder is at best a visible re-roll. In this case it changes the image about half of what a new seed would change.

**What's actually censored then?**

The knowledge, not the gate. The DiT doesn't refuse, it lacks the training data, and the fixes are LoRAs, reference images, or retraining. The PSA's framing about this is solid. Z Image itself though is mostly trained already on taboo things.

**What's next**

Krea 2, MiniMax H3 and LTX 2.5 are in the same pipeline. Krea 2 has a twelve-tap conditioning interface and the refusal-probe contrast works differently there. Also, it's more complicated to measure compared to Z-Image.

Happy to answer methodology questions in the comments.