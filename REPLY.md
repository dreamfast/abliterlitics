Hey, so I tested the kl stuff.

Tested with 100 prompts from `databricks/databricks-dolly-15k` dataset, and 1 token vs 32 tokens. Comparing to the original KL scores we had for gemma e4b.

Honestly it made no real difference. It changed the scores by about 10% more or less, but the order of best to worst remained the similar enough.

The kl measurement never uses harmful dataset too. So for measuring KL it doesn't matter what harmless dataset is used.

So overall I can't see any evidence the approach is bias towards existing heretic models.

Measuring with 198 questions from GPQA and it didn't affect the spread or order of best to worst. Although it had an interesting effect where the lower KL scores were higher, and the highest ones were a bit lower. So the spread narrowed, but still overall the same if we want to see what is best or worst.

And I confirmed thinking was enabled for this one too.

Overall for KL we just need harmless prompts, so it doesn't seem to matter what dataset is used, although testing with two so far that seems to be the case.

---

## Measured data (gemma4-e4b)

KL divergence vs the base model. Lower = less drift = better capability preservation. Both experiments are sorted by the original (`alpaca_1t`) score, best to worst.

### Experiment 1 — Dolly-15k (100 prompts, 1 vs 32 tokens)

| variant      | original (alpaca, 1 tok) | dolly-15k, 1 tok | dolly-15k, 32 tok |
|--------------|-------------------------:|-----------------:|------------------:|
| heretic      |                   0.0021 |           0.0038 |            0.0185 |
| nullpo       |                   0.0054 |           0.0169 |            0.0239 |
| trevorjs     |                   0.0145 |           0.0148 |            0.0224 |
| huihui       |                   0.0266 |           0.0201 |            0.0201 |
| abliterix    |                   0.0536 |           0.0553 |            0.0810 |
| treadon-combo|                   0.2683 |           0.2165 |            0.1933 |
| bendernina   |                   0.9232 |           1.0006 |            0.3319 |
| obliteratus  |                   1.1015 |           1.2435 |            0.4461 |

Rank correlation (Spearman ρ): original vs dolly 1-tok = **0.976** · original vs dolly 32-tok = **0.905** · dolly 1 vs 32 tok = 0.929

### Experiment 2 — GPQA Diamond (198 questions, 1 vs 32 tokens)

| variant      | original (alpaca, 1 tok) | gpqa, 1 tok | gpqa, 32 tok |
|--------------|-------------------------:|------------:|-------------:|
| heretic      |                   0.0021 |      0.0182 |       0.0212 |
| nullpo       |                   0.0054 |      0.0408 |       0.0224 |
| trevorjs     |                   0.0145 |      0.0166 |       0.0180 |
| huihui       |                   0.0266 |      0.0747 |       0.0375 |
| abliterix    |                   0.0536 |      0.0539 |       0.0723 |
| treadon-combo|                   0.2683 |      0.2794 |       0.3060 |
| bendernina   |                   0.9232 |      0.6192 |       0.4640 |
| obliteratus  |                   1.1015 |      0.2955 |       0.5894 |

Rank correlation (Spearman ρ): original vs gpqa 1-tok = **0.881** · original vs gpqa 32-tok = **0.929** · gpqa 1 vs 32 tok = 0.952

### Experiment 3 — Divergence trajectory over 1→32 tokens (GPQA, thinking ON vs OFF)

Cumulative mean KL over the first N generated tokens (N = 1, 2, 4, 8, 16, 32). Shows how the two trajectories diverge as generation proceeds. Rows sorted by thinking-ON @ 32 tok.

**Thinking ON** — note the dip at N=2–4 (near-deterministic thinking openers) before KL rises again:

| variant | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| trevorjs | 0.0168 | 0.0085 | 0.0046 | 0.0138 | 0.0151 | 0.0180 |
| heretic | 0.0177 | 0.0089 | 0.0053 | 0.0288 | 0.0273 | 0.0212 |
| nullpo | 0.0406 | 0.0203 | 0.0108 | 0.0244 | 0.0260 | 0.0224 |
| huihui | 0.0735 | 0.0369 | 0.0186 | 0.0177 | 0.0247 | 0.0375 |
| abliterix | 0.0528 | 0.0266 | 0.0137 | 0.0274 | 0.0343 | 0.0723 |
| treadon-combo | 0.2801 | 0.1413 | 0.0794 | 0.1971 | 0.2056 | 0.3060 |
| bendernina | 0.6192 | 0.3100 | 0.1647 | 0.1697 | 0.2801 | 0.4640 |
| obliteratus | 0.2925 | 0.1469 | 0.0815 | 0.1279 | 0.2118 | 0.5894 |

**Thinking OFF** (direct answers — dip disappears; heavy models diverge more):

| variant | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| trevorjs | 0.0253 | 0.0200 | 0.0322 | 0.0275 | 0.0275 | 0.0216 |
| heretic | 0.0200 | 0.0203 | 0.0300 | 0.0263 | 0.0270 | 0.0208 |
| nullpo | 0.1336 | 0.0858 | 0.0738 | 0.0481 | 0.0368 | 0.0279 |
| huihui | 0.0406 | 0.0379 | 0.0717 | 0.0857 | 0.0948 | 0.0824 |
| abliterix | 0.0424 | 0.0756 | 0.1534 | 0.1231 | 0.1308 | 0.1208 |
| treadon-combo | 1.4638 | 0.9193 | 0.7364 | 0.5846 | 0.5370 | 0.4402 |
| bendernina | 1.9464 | 1.1634 | 1.0002 | 0.9235 | 0.9776 | 0.9666 |
| obliteratus | 4.3137 | 2.5412 | 1.8273 | 1.4482 | 1.3484 | 1.2957 |

Rank correlation (Spearman ρ) @ 32 tokens: thinking ON vs OFF = **0.976** · no-think vs original alpaca = **0.976** · (at 1 token, ON vs OFF = 0.857 — first-token/boilerplate noise).

Takeaway: thinking on vs off barely moves the ranking at 32 tokens. The trajectory makes the boilerplate effect visible (the ON dip at N=2–4), and disabling thinking removes it — but the best-to-worst order is the same either way.

---

Notes:
- `original (alpaca, 1 tok)` = the production measurement (`mlabonne/harmless_alpaca`, `test[:100]`, first-token logits).
- The `*_32 tok` columns average over a greedy continuation (teacher-forced: the base generates the continuation, both models scored on that same shared path).
- Best-to-worst order is preserved across all datasets and token counts (ρ ≥ 0.88). GPQA was the tougher test (lower ρ, range compressed — low end up, high end down) but the ranking held.
- Thinking was enabled for the GPQA run (Gemma 4 thinks by default; the measured continuation begins `Here's a thinking process to arrive at the solution: 1. Analyze the Request:`).
