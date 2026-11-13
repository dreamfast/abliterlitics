# DISCORD_QWEN38_27B.md, Heretic server results announcement draft

> Draft for the Heretic Discord. Untracked, never commit markdown.
> Voice: user's own, lowercase casual like DISCORD.md. Style rules: no em dashes, no parentheticals, Australian spelling.
> The announcement fits ONE Discord message, under the 2000 char limit. The long version below is optional thread material.

**The message, post this**

---

the qwen3.8-27b comparison is done. this was the one people kept asking for. 167 gpu hours over eleven days on a single 5090, base plus eight variants, every arm served identically in dynamic fp8. all 3,600 harmbench responses were judged by an llm reading the full reasoning traces, no keyword matching.

judge asr, best to worst:

orcarouter 82.2, apostate 78.7, huihui 75.6, ultra heretic 70.5, coder3101 70.0, blackfrost 68.5, obliteratus 63.9, trohrbaugh 57.5, base 4.5

the headlines: surgical beats heavy again. the top two are the two smallest edits, and the heaviest edit of all came second to last because 45% of its answers never finish thinking. copyright is the new universal wall, nobody above 39%. blackfrost ships an undisclosed jailbreak inside its chat template. trohrbaugh kept the most alignment of anyone and has the cleanest numbers, that is the one i use at home.

report: https://abliterlitics.dev/models/qwen38-27b/
every response browsable: https://abliterlitics.dev/harmbench/qwen38-27b/
dataset: https://huggingface.co/DreamFast/Qwen-3.8-27b-abliterlitics

the template finding has us working on a new chat template and hyperparameter measurement, with the intention to eventually replace the harmbench part. opinions on how that should be scored are welcome here.

**Long version, optional thread material below**

---

the qwen3.8-27b comparison we announced here is done. this was the one people kept asking for. 167 gpu hours over eleven days, single 5090 at 400w, base plus eight variants through the full forensic suite. every arm served through the identical dynamic fp8 path, the bf16 weights are 52gb and were never fitting on one card. report and all artifacts are up:

- report: https://abliterlitics.dev/models/qwen38-27b/
- every response, full reasoning traces and judge verdicts browsable: https://abliterlitics.dev/harmbench/qwen38-27b/
- dataset: https://huggingface.co/DreamFast/Qwen-3.8-27b-abliterlitics

all 3,600 harmbench responses were judged by an llm reading the full reasoning trace, four-way rubric, no keyword matching. numbers below are judge asr.

the results, best to worst:

orcarouter 82.2%, the winner. single direction at layer 38, 131 matrices, and the only card where every claim verified exactly against the weights. capability within noise of base. this is the surgical reference the others get judged against and it held up completely.

apostate 78.7%, best value and the most interesting result for this server. the new kcrn method needed just 41 real edits for second place, lowest kl measured at 0.0439, near-identity capabilities. the fp16 re-save does throw storage noise at the diff, we counted 565 changed tensors and 41 real ones.

huihui 75.6%, the classic, reliable. clean unlock everywhere except copyright at 3 of 100.

llmfan46 ultra heretic 70.5%, heretic v2 with mpoa, mtp preserved. works, with the heaviest truthfulness drop outside obliteratus and 118 soft refusals concentrated on copyright.

coder3101 70.0%, vanilla heretic. worth noting his card says 33 of 100 refusals remain and we measured 5 in 400. the card undersells it. also the re-save dropped all 15 mtp tensors, documented upstream defect, harmless for serving.

blackfrost 68.5%. the blind fingerprint came back single direction, heaviest magnitude in the panel, 100% rank-1, which refutes the direction bank story. and it ships a 1457-character jailbreak system prompt inside its chat template, meaning every single prompt you make gets the jailbreak injected, and the template is not disclosed on the readme. their refusal numbers are accurate however, even with us not using their jailbreak template we measured a similar floor, so credit where due.

obliteratus 63.9%, avoid. most aggressive edit in the panel, 841 of 850 tensors, and second-to-last result. they published honest changes to how the model was changed, the metadata matched the weights exactly, yet we were unable to replicate the 0% refusals 0% deflection claims. 44.8% of responses never finish thinking and we found 142 deflections. also the only variant that got meaningfully dumber, mmlu-pro down 6.3pp, humaneval down 33.6pp.

trohrbaugh 57.5%, last of the variants, on purpose. 122 explicit refusals, most surviving alignment of anyone. the cleanest numbers in the comparison, plus the only kl that calibrated, card 0.0535 against our 0.0586. this is the one i use at home and it has been great for me. the ara method trades rank cleanliness for direction quality and it shows in the behavioural profile.

base 4.5%, a wall. zero on chem and bio, harassment, harmful content, copyright.

the findings that matter beyond the ranking:

- the thinking loops. qwen 3.8 thinks before answering and the aggressive arms loop, up to 45% of harmbench responses never close the think block before the 15,360-token budget dies. the judge reads the full trace so content inside a loop still counts, but a model that delivers only inside an unterminated monologue is not usable. same arms finish gsm8k fine, every arm within 1.2pp of base answered-only. school math converges, adversarial deliberation does not.
- copyright is the new universal wall. nobody above 39%, five of nine at or below 3.2%. chem and bio, historically the hardest category, is now the easiest unlock, orcarouter and apostate comply with all 56. the walls moved on this generation.
- chat templates needed forensics for the first time. blackfrost hides a jailbreak in its template, obliteratus ships thinking-off, ultra heretic deletes the stock xhigh reasoning-effort prompt. we pinned the stock template for every arm. template choice moved refusal behaviour more than 4-bit quantisation did in our earlier measurements, it is a stronger lever than anyone assumes.
- kl honesty. trohrbaugh calibrated within 9%. the rest diverge and none of it is dishonesty, kl is non-deterministic and moves with cuda version and hardware, or by what method used. read kl as a within-comparison spread, not a card-checking instrument.

the full weight forensics, per-category tables, kl methodology and the template analysis are in the report. the template finding has us thinking on a new measurement of chat template and hyperparameter effects, with the intention to replace the harmbench part with a better measurement to cater for all of it. if you have opinions on how that should be scored, this is the place to say it. questions, disagreements, methodology fights, all welcome here.
