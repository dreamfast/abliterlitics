# DISCORD.md, Heretic server announcement draft

> Draft for the Heretic Discord. Untracked, never commit markdown.
> Voice: user's own. Style rules: no em dashes, no parentheticals, Australian spelling.
> Subject: Qwen3.8-27B comparison roster, what is in and what is out.

---

next comparison is qwen3.8-27b, the new hybrid attention vl model. base plus six uncensored variants, the full forensic suite. lm-eval on 13 tasks, harmbench, kl divergence, weight analysis. every arm served identically in dynamic fp8 on a single 5090, that took some doing, the bf16 weights are 55.6 gb and do not fit. fp8 comes in at 27.6 gib with fp8 kv cache and room to spare, verified with a live server before committing.

the roster:

[qwen/qwen3.8-27b](https://huggingface.co/Qwen/Qwen3.8-27B) base, the reference arm. strongest safety alignment we have seen on a qwen, refuses 92 to 99 of 100 harm prompts depending who measured it.

[orcarouter/qwen3.8-27b-uncensored](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored) the textbook one. arditi style single direction at layer 38, 131 matrices edited, mtp head abliterated consistently, vision tower byte identical. their card is the most honest in the space, capability within 1.3 points of base across their benchs. this is the surgical reference the others get judged against.

[trohrbaugh/qwen3.8-27b-heretic-ara](https://huggingface.co/trohrbaugh/Qwen3.8-27B-heretic-ara) heretic with the ara method, arbitrary rank ablation, pr 211. custom fork, layers 26 to 56. zero refusals of 100 on his card, kl 0.0535. this is the ara first test in one of our comparisons.

[coder3101/qwen3.8-27b-heretic](https://huggingface.co/coder3101/Qwen3.8-27B-heretic) vanilla heretic v1.2.0, the same author we tested on 3.5-27b. interesting because his own card admits 33 of 100 refusals remain. the 27b scale alignment resists vanilla heretic. keeping it for the series continuity and as the under-abliterated control.

[huihui-ai/huihui-qwen3.8-27b-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated) the classic. layers 18 to 51 only. we are testing whether the 3.5-27b collapse repeats, huihui dropped to 88.8 percent asr there after being perfect on every smaller model.

[obliteratus/qwen3.8-27b-obliterated](https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED) pliny's, v3 deep liberation. novel recipe, svd and leace surgeries blended 60/40, iterative stacking, targeted corpus. zero hard refusals and zero soft deflections claimed, 2.1 points mmlu cost. also the drama pick, shipped a broken chat template that silently ignored enable_thinking, users report cot loops, and the repo has two generations of shards mixed together. broken and trending is exactly the kind of thing worth forensically examining.

[blackfrost-ai/qwen3.8-27b-abliterated-bf16](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-BF16) the blind fingerprint pick. the only arm with a fully closed method, the card says weight-level de-risked and nothing else, team evaluation status. no method, no layers, no direction count, no insights. so we will provide them. our weight analysis can determine which tensor types moved, the rank of the edits, the layer profile, and the cosine overlap against the directions every other technique here found. that identifies the method family or proves it is novel. the level1techs thread already measured it as heavy collateral at the logit level, 4 to 5 percent top-1 flips, p95 kld 0.31, which is our pre-registered prediction to test. if the fingerprint says surgical but the flips say damage, interesting. if the fingerprint says heavy recipe, case closed.

what is out and why:

- [aeon ultimate](https://huggingface.co/AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-BF16). the author labels it an early access draft and says wait for ga. the card admits long context loops. the level1techs thread measured literal copy failures, port numbers turning into mush mid generation. if the author says it is not ready, it is not ready. revisit when ga lands.
- [jonathancoletti](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored). it is a heretic run, bf16 merged, surgical scope. decent card honestly. but two heretic arms already cover the method and trohrbaugh's ara is the more interesting one. his big numbers are on the gguf side.
- [coder3101-heretic-ara](https://huggingface.co/coder3101/Qwen3.8-27B-heretic-ara). a third heretic roll. deferred, not dead. two independent ara runs on the same base is a heretic non-determinism study nobody has published, might add it later.
- [heretic-org heretic-ara](https://huggingface.co/heretic-org/Qwen3.8-27B-heretic-ara). byte identical mirror of trohrbaugh's, file sizes match exactly.
- [hauhaucs](https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF). discontinued by us after the 3.6 comparison, the tool is a heretic plagiarism rebranded as reaper.
- the gguf only ones, [0bserverx](https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF), [0xkitkat](https://huggingface.co/0xKitkat/Qwen3.8-27B-Uncensored-Aggressive), [zerodigest](https://huggingface.co/zerodigest/Qwen3.8-27B-Uncensored-YMQ-MTP-GGUF), the [empero distills](https://huggingface.co/empero-ai/Qwen3.8-9B-Distill). no bf16 source or a different modification class entirely.

worth noting, the only comparable independent work i have found is thr3e's series on level1techs, "why your local llm feels dumber than it is". post 3.11 ran four of these same abliterations against stock with fp64 full vocab kl, top-1 flip rates and teacher forced branch futures on a real 100k token workstream. surgical tier, heretic-ara and huihui. do-not-use tier, blackfrost and aeon. two of our exclusions land exactly on their measurements which is a good sanity check. they did not test orcarouter or obliteratus, and no benchmark deltas or harmbench, so our study extends that work rather than repeating it.

runs start once downloads finish. expect fp8 absolute numbers with base-relative deltas, every arm quantised through the identical path so the deltas stay clean.

---

## Numbers behind each claim, for follow-up questions

| Claim in message | Data | Where |
|---|---|---|
| fp8 fits a single 5090 | 27.64 gib weights with language-model-only, 0.97 gib kv pool, 22,341 fp8 tokens, live server verified, vllm 0.26.0 | smoke test 2026-08-26, journal + NOTES.md |
| base refuses 92 to 99 of 100 | coder3101 card 92/100, trohrbaugh card 99/100, orcarouter advbench 99 percent | model cards |
| orcarouter edits 131 matrices | o_proj 17, out_proj 48, down_proj 65, embed_tokens 1 | orcarouter card |
| trohrbaugh ara zero refusals | 0/100, kl 0.0535 | trohrbaugh card |
| coder3101 vanilla 33 refusals | 33/100, kl 0.0187 | coder3101 card |
| huihui 3.5-27b collapse | 88.8 percent asr, 45 refusals of 400 | older_comparisons/Qwen3.5-27B.md |
| obliteratus broken template | discussion 6, open fix pr, template always emitted a closed think block | hf discussions |
| obliteratus cot loops | discussion 7, "repeating cot and reasoning", multiple users confirm | hf discussions |
| obliteratus two shard generations | repo holds an orphaned 18 shard set plus the live 28 shard set and model-extra the index points at, 111 gb total | api file listing |
| blackfrost closed method | card states weight-level de-risked, no method, no parameters, team evaluation status | blackfrost card |
| blackfrost blind fingerprint plan | tensor type coverage, edit rank, layer profile, edit vector cosine vs all other arms, mtp and vision hash diff | our weights stage, lesson 22 taxonomy |
| blackfrost heavy collateral | 4 to 5 percent top-1 flips, p95 kld 0.31 | level1techs post 3.11 |
| aeon literal copy failures | 36 of 253 invalid branch futures, port 5432 became 543ql | level1techs post 3.11 |
| thr3e tier ranking | surgical, heretic-ara 0.7/1.3 percent flips p95 0.037, huihui 0.9/1.4 p95 0.045 | level1techs post 3.11 |

## If pressed on fp8 methodology

dynamic online fp8 from the bf16 checkpoint, not the official pre-quantised qwen fp8 release. every arm, base included, goes through the identical quantisation path so variant deltas measure the abliteration, not the quant. same reason we never mixed bnb4 arms with bf16 arms in the old comparisons. fp8 per-tensor online keeps mamba internals unquantised, only the linears quantise.

## If pressed on why gguf-only variants are excluded

our pipeline needs bf16 safetensors for weight forensics. gguf quants have the abliteration edit and the quantisation round trip superimposed, that is exactly the confound the 3.6 hauhaucs analysis had to untangle. not doing that again by choice.

## If pressed on obliteratus settings drama

the card mandates repetition penalty 1.15 because greedy decoding loops without it. we will evaluate with our standard settings, identical across arms, and report the loop behaviour as a finding. a model that needs a repetition penalty to function is a model with a measurable stability problem, that is the point of the study.

## Scope statement if pressed

all claims are qwen3.8-27b on identical fp8 serving, single 5090, thinking model with the template handled per phase. thinking on for gsm8k and harmbench, thinking suppressed for the loglikelihood benchs per our standard methodology. results are base-relative deltas first, absolute numbers second.
