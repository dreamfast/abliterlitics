"4 more uncensored Qwen 3.8 27B variants, 166 more GPU hours - Abliterlitics"

**The post**

---

Following up from our [previous comparison](https://www.reddit.com/r/LocalLLaMA/comments/1w8vx6w/8_uncensored_qwen_38_27b_variants_one_base), we ran four more uncensored Qwen 3.8 27B variants through the identical pipeline. Same FP8 serving, same 13 benchmarks, same HarmBench with an LLM judge.

* Full report: [abliterlitics.dev/models/qwen38-27b](https://abliterlitics.dev/models/qwen38-27b/)

**The four new arms**

* [ektome](https://huggingface.co/Zynerji/Ektome-Qwen3.8-27B-PristinelyUncensored) 98.3% harmful-only, the best uncensoring we have ever measured, zero explicit refusals including full chem weapons syntheses. But a third of its responses think in circles until the token budget dies and never ship an answer. It was made by an agentic search loop that graded its own homework, and its card ran a keyword classifier with thinking switched off, so the card could not see the loops. Overall given the readme, it just seems vibe coded with the agent brute forcing an approach. Despite the pristine name, its TruthfulQA dropped 7 points, the second worst drop of the whole panel. Other models in the comparison can be considered more pristine
* [davidau](https://huggingface.co/DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU) 92.7%, not really a pure abliteration, it's a finetune with a heretic pass on top. It scores 18 points ABOVE base on GPQA. It does not over think. Every knowledge benchmark went up, we can measure that. However the generative side took real damage, instruction following down 5 points and maths down 4.4, and it has the highest LAMBADA perplexity of the panel at 67% above base, meaning its raw next-token prediction is damaged
* [rvn](https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored) 92.0% harmful-only. Same 60 tensors as trohrbaugh from the first batch, one ARA pass refuses 122 of 400, two passes refuse 15, capabilities stay surgical
* [apostate_diode](https://huggingface.co/heterodoxin/qwen-3.8-27b-abliterated) 94.7%, the first conditional abliteration we have measured. It appends one gated neuron to each targeted MLP instead of editing weights, 658 tensors stay bit identical to base, KL is the lowest on the roster at 0.039, GSM8K is identical to base. This is the lowest KL and best preserved model with decent unlock. Given the extra addition to the weights, it did take slightly more VRAM for me to run. One thing to know, the author re-uploaded these weights over the earlier KCRN apostate release at the same URL, so the apostate link from our first post now serves diode

**What's new on the report**

* Total HuggingFace downloads column added to the model table. The uncomfortable stat, the most downloaded variant is obliteratus at 1.27M downloads, 14 times the next variant, and it is the one the report says avoid. Nearly half its answers never arrive and it is the only variant that got meaningfully dumber. Proving again the most popular model is not always the best
* Harmbench harmful-only scoring added to the judge table. The overall ASR includes 100 copyright behaviours which every model here struggles with, since verbatim recall is the bottleneck. We now have a measurement of just the 300 actual harmful questions if you are curious how they went
* A new recommended to use table, we determine the 'winner' on harmful unlock, whether the model actually finishes its answers rather than looping out, and benchmark capability preservation. Worth noting the looping was measured at temperature 0 which is the HarmBench standard, your recommended sampling settings may reduce it

**The Report**

* Full report: [abliterlitics.dev/models/qwen38-27b](https://abliterlitics.dev/models/qwen38-27b/)
* Every response, reasoning trace and judge verdict, all 5,200: [abliterlitics.dev/harmbench/qwen38-27b](https://abliterlitics.dev/harmbench/qwen38-27b/)
* HuggingFace dataset: [DreamFast/Qwen-3.8-27b-abliterlitics](https://huggingface.co/DreamFast/Qwen-3.8-27b-abliterlitics)
* Code: [github.com/dreamfast/abliterlitics](https://github.com/dreamfast/abliterlitics)

Happy to answer questions. We have a runtime refusal-removal technique in the works that needs no weight edits at all, this batch's loop findings are what pushed us that direction. [Discord](https://discord.gg/AqmDnBjPvM).
