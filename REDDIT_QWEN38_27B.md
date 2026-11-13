# Qwen 3.8 27B Reddit post draft

Target: r/LocalLLaMA. Post Tue-Thu 8am-12pm EST, be in the comments for the first 2 hours. This file stays untracked, never commit.

## Title options

1. "8 uncensored Qwen 3.8 27B variants, one base, 167 GPU hours, 3,600 judged responses" (85 chars)
2. "The most aggressive abliteration of Qwen 3.8 27B came second-to-last. The careful edits won." (91 chars)

Recommendation: option 1 for the data crowd, option 2 if you want the contrarian hook.

**The post**

---

This comparison was requested by a few people, and certainly we were all eager to see the final results. The comparison had taken 11 days and the GPU was crunching numbers for ~167 GPU hours.

We've been comparing different abliterated models from huggingface to see if they really are what they claim to be. So far the results have been interesting.

The pipeline includes a weight comparison, KL divergence measurement, 13 benchmarks and measuring refusals with the HarmBench 400 classic. Qwen 3.8 27b is a thinker, and with ourselves using xhigh we had to set our token budget much higher per request.

Lets check out how Qwen 3.8 27B stacks up comparing 8 variants.

**The rankings**

LLM Judge HarmBench ASR, attack success rate, best to worst, with the one-line story:

* [orcarouter](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored) 82.2%, the winner. Arditi-style single direction at layer 38, 131 matrices, and the only card where every claim checked out against the weights. Best copyright unlock in the set at 39%
* [apostate](https://huggingface.co/heterodoxin/qwen-3.8-27b-abliterated) 78.7%, best value. Their new KCRN method, 41 real edits, lowest KL measured at 0.0439, near-identity capabilities. Packaging quirks: text-only re-save with no vision and no MTP, stored FP16
* [huihui](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated) 75.6%, the classic method, reliable. Clean unlock everywhere except copyright, where it sits at 3%
* [ultra_heretic](https://huggingface.co/llmfan46/Qwen3.8-27B-Ultra-Uncensored-Heretic-Native-MTP-Preserved) 70.5%, Heretic v2 with MPOA. Works, but the heaviest truthfulness drop outside obliteratus and 118 soft refusals
* [coder3101](https://huggingface.co/coder3101/Qwen3.8-27B-heretic) 70.0%, vanilla Heretic. The card calls itself the weakest removal at 33 of 100 refusals. Measured: 5 explicit refusals in 400. The card undersells it
* [blackfrost](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-BF16) 68.5%, closed method. The weights say single direction, heaviest magnitude in the panel, 100% rank-1, which refutes the rank-k direction bank story. Also ships a jailbreak system prompt inside its chat template, meaning every single prompt you make will have a modified chat template injecting a jailbreak
* [obliteratus](https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED) 63.9%, **avoid**. The most aggressive edit in the panel at 841 of 850 tensors, and it performs like it. 44.8% of responses never finish thinking, and it is the only variant that got meaningfully dumber
* [trohrbaugh](https://huggingface.co/trohrbaugh/Qwen3.8-27B-heretic-ara) 57.5%, last of the variants because it still refuses. 122 explicit refusals, the most surviving alignment of any variant, and the cleanest capability profile in the comparison. This is the one I use at home and it's been great for me.
* [base](https://huggingface.co/Qwen/Qwen3.8-27B) 4.5%, a wall. Zero compliance on chem and bio, harassment, harmful content and copyright

**The highlights**

* Surgical beats heavy, again, and this time it is not close. The top two spots went to the two smallest verified edits. The heaviest edit of all landed second-to-last. At 27B, editing everything mostly buys you a model that thinks in circles
* The thinking loop story is the big new finding for this model. Qwen 3.8 thinks before answering, and on the aggressive arms up to 45% of HarmBench responses never close their think block before the 15,360-token budget dies. The judge reads the full trace, so compliance inside a loop still counts. But a model that only delivers the goods inside an unterminated monologue is not a usable model
* GSM8K loops are gone at this budget. The same arms that loop 40%+ on HarmBench finish their math reasoning fine, every arm within 1.2pp of base on answered-only. School math converges, adversarial deliberation does not
* Copyright is the new universal wall. Nobody exceeds 39%, five of nine sit at or below 3.2%. Chem and bio, historically the hardest category, is now the easiest unlock. The walls moved
* Chat template forensics was needed for the first time. blackfrost ships a 1457-character jailbreak prompt inside its template. obliteratus ships thinking-off. ultra_heretic deletes the stock reasoning-effort prompt. We pinned the stock template for every arm, because the template is a stronger behavioural lever than most people assume

**Card honesty check**

* orcarouter verified 4 of 4 claims exactly, the model card is honest.
* trohrbaugh's KL calibrated within 9% of our measurement. However the card mentions 0/100 refusals, with our measurement this model had the most refusals, yet also had preserved capabilities.
* obliteratus published honest changes to how their model was changed, yet we were unable to replicate the '0% refusals, 0% deflection' claims. 44.8% of harmbench never finished thinking, and reasoning analysis found 142 deflections.
* blackfrost mentions they have an internal direction bank, suggesting that multiple directions are changed. Yet we only found one direction changed. Their refusal numbers are accurate however, even with ourselves not using their jailbreak chat template we got similar results. The modified chat template also is not disclosed on the readme.
* The rest diverge on KL, and none of it is dishonesty, KL is non-deterministic and moves with CUDA version and hardware, or by what method used. Read KL as a within-comparison spread

**The Report**

* Full report: [abliterlitics.dev/models/qwen38-27b](https://abliterlitics.dev/models/qwen38-27b/)
* Every response, reasoning trace and judge verdict: [abliterlitics.dev/harmbench/qwen38-27b](https://abliterlitics.dev/harmbench/qwen38-27b/)
* HuggingFace: [DreamFast/Qwen-3.8-27b-abliterlitics](https://huggingface.co/DreamFast/Qwen-3.8-27b-abliterlitics)
* Code: [github.com/dreamfast/abliterlitics](https://github.com/dreamfast/abliterlitics)

Happy to answer questions. The template finding has us thinking on a new measurement of chat template and hyperparameter effects, with the intention to replace the harmbench part with a better measurement to cater for all of this. So if you have opinions on how that should be scored, [come tell us on Discord](https://discord.gg/AqmDnBjPvM).
