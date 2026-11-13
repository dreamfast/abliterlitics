# Metrics Methodology

This document defines *exactly* how we measure text-encoder damage. The core question is always: **"how much does a modified text encoder (abliterated / quantized / GGUF'd) perturb the conditioning that actually enters the diffusion transformer, relative to a baseline that means something?"**

---

## 1. Why KL divergence is the wrong primary metric

This is the most important methodological point in the project, so it gets stated first and completely.

### The naming collision

Heretic reports a "KL divergence" score per trial (the author's blog records `KL Divergence: 0.0826` for Gemma 3 12B trial 99). That number is **correct and meaningful** — but it measures something specific: the KL divergence between the base model's and the abliterated model's **next-token probability distributions** (softmax over the vocabulary logits) on a probe set.

Next-token distributions **are probability distributions** (non-negative, sum to 1), so KL is well-defined:

$$D_{KL}(P \| Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}$$

This is appropriate for measuring **chat / decoding damage** — does the model still produce sensible token distributions? Yes, use KL.

### Why that does not transfer to conditioning

The tensor that actually feeds a diffusion transformer's cross-attention is **not a probability distribution.** It is the text encoder's hidden states (or a projection thereof): an unbounded real-valued tensor in $\mathbb{R}^{L \times d}$, with negative entries, no normalization constraint, and arbitrary scale.

| Property | Probability distribution (KL input) | Conditioning tensor (what DiT sees) |
|---|---|---|
| Domain | $\Delta^{V-1}$ (simplex) | $\mathbb{R}^{L \times d}$ |
| Non-negative | yes | **no** |
| Sums to 1 | yes | **no** |
| Scale-invariant | yes | **no** (cross-attention uses magnitudes) |
| Symmetric metric needed | JSD, not KL | n/a |

To apply KL you must first force the tensor through a softmax (over which axis? the choice is arbitrary and lossy). That projection **destroys exactly the geometry the DiT cares about** — directions and magnitudes of conditioning vectors that cross-attention dot-products against. So you would be measuring damage to a *projection* of the signal, not the signal.

Additionally, KL is **asymmetric** ($D_{KL}(P\|Q) \ne D_{KL}(Q\|P)$); the symmetric fix (Jensen-Shannon) still operates on the wrong object.

**Conclusion:** KL is the right tool for Heretic's own chat-damage score, and the wrong tool for conditioning damage. Using it for conditioning would measure a lossy shadow of the thing we care about.

---

## 2. The primary metric: Centered Kernel Alignment (CKA)

### What it is

CKA (Kornblith et al., 2019, *"Similarity of Neural Network Representations Revisited"*) is **the** established method in the interpretability literature for answering "did these two networks (or two runs) produce similar representations?" It was designed precisely for our question.

### Definition (linear CKA)

Given two activation matrices $X, Y \in \mathbb{R}^{n \times p_1}$ and $\mathbb{R}^{n \times p_2}$ over the same $n$ samples (prompts), first **center** the columns (subtract per-dimension mean across the $n$ samples). Then:

$$\text{CKA}(X, Y) = \frac{\|Y^\top X\|_F^2}{\|X^\top X\|_F \cdot \|Y^\top Y\|_F}$$

Equivalently, via the Hilbert-Schmidt Independence Criterion (HSIC):

$$\text{CKA}(X, Y) = \frac{\text{HSIC}(X, Y)}{\sqrt{\text{HSIC}(X, X)\,\text{HSIC}(Y, Y)}}$$

### Properties that make it right for us

| Property | Why it matters here |
|---|---|
| **Bounded [0, 1]** | Interpretable, comparable across models/dims. |
| **Symmetric** | No base/variant ordering ambiguity. |
| **Invariant to orthogonal transforms** | A rotation of the embedding basis (irrelevant) doesn't change the score. |
| **Invariant to isotropic scaling** | A uniform rescale (irrelevant) doesn't change the score. |
| **Operates on the full $[n \times d]$ matrix** | Captures representational **geometry** across the whole prompt set, not pointwise distance. |
| **Cheap** | A few matrix mults; trivially GPU. |

### When it is *not* enough (and we add companions)

CKA can be **blind to anisotropic distortion**: if abliteration squashes one latent direction 10× more than others, CKA (which is scale-invariant in the aggregate) may barely move, while the DiT's attention is materially changed. That is why CKA is *primary but not alone* — we pair it with magnitude-aware metrics below.

### RBF CKA (secondary)

For a nonlinear comparison, replace the linear kernel with a Gaussian (RBF) kernel:

$$K_{ij} = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)$$

and apply the same HSIC normalization. We compute both; large linear-vs-RBF divergence itself is a signal of nonlinear warping.

### Reference implementations (do not reinvent)

- `RistoAle97/centered-kernel-alignment` (PyTorch, HSIC-based)
- `numpee/CKA.pytorch` (GPU, CVPR 2023 paper)
- `pytorch-cka`, `torch-cka` (PyPI)

We vendor or wrap one; the algorithm is ~20 lines.

---

## 3. Companion metrics

All operate **per-stream** (a stream = one named conditioning tensor, e.g. `video_proj`, `audio_proj`, `layer50`). See `ARCHITECTURE.md`.

### 3.1 Mean cosine similarity, all tokens

The author's prior metric, but **extended from last-token to the full sequence**, because the DiT attends to every token — last-token pooling discards most of the signal.

$$\bar{c} = \frac{1}{n} \sum_{s=1}^{n} \frac{1}{|L_s|} \sum_{t=1}^{L_s} \cos(x_{s,t},\, y_{s,t})$$

where $L_s$ is the non-padded length of prompt $s$. Reported per **category** (innocent / mild / violent / sexual / extreme / multilingual) to reproduce and extend the author's gradient finding.

### 3.2 L2 / MSE distance (magnitude-aware)

Cosine is magnitude-blind — abliteration can shift *along the same direction* but rescale it, leaving cosine = 1.0 while the DiT sees a different magnitude.

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n} \text{MSE}_i \quad (\text{per-prompt weighting; pad-masked. Equals element-weighted MSE only for equal-length prompts})$$

### 3.3 Pairwise-distance rank correlation (geometry preservation)

Does the modified encoder still rank "concept A closer to B than to C" the same way? Compute pairwise distances among prompts under each encoder, then **Spearman correlation** of the two distance vectors. Catches anisotropic warping CKA can miss.

$$\rho = \text{Spearman}\big(\{\|x_i - x_j\|\},\ \{\|y_i - y_j\|\}\big)$$

### 3.4 Per-dimension drift profile (diagnostic)

For each latent dimension $d_k$, compute the variance of $(X_{:,k} - Y_{:,k})$ across prompts. Identifies *which* directions abliteration perturbs (expected: the refusal/alignment subspace). Cheap and highly bloggable.

### 3.5 Downstream DiT-step divergence (optional, the mic-drop)

The only metric that directly answers "does it change the *output*": feed both conditionings into the DiT for $K$ steps (or just the first cross-attention block) from a fixed initial noise, measure divergence in predicted $x_0$ / attention maps. GPU-heavy; reserved for the headline model only. Needs the real DiT loader → couples to a model framework; treat as Phase 7 stretch.

---

## 4. The noise-floor reference (the part that makes numbers meaningful)

A raw `CKA = 0.94` or `cosine = 0.83` is uninterpretable in isolation. We always compute and report **reference baselines** and express the variant's divergence *relative* to them.

| Reference | What it bounds | How computed |
|---|---|---|
| **Numerical floor** | pure dtype noise | `div(base_bf16, base_fp32)` on the same ckpt |
| **Semantic ceiling** | "how different are two different prompts" | `div(prompt_set_A, prompt_set_B)` within the *base* model (paired shuffle) |
| **Quantization floor** | the cost of a known-good compression | `div(base_bf16, base_fp8)`, `div(base, base_gguf_Q6)` |
| **Author's prior** | cross-check | reproduce the blog's cosine table at raw-hidden-state stage |

### Relative reporting

For each variant $V$ and metric $m$ on stream $s$:

$$\text{damage}_m(V, s) = \frac{m(\text{base}, V) - m(\text{base}, \text{base}_\text{fp32})}{m(\text{base}, \text{base}_\text{fp8}) - m(\text{base}, \text{base}_\text{fp32})}$$

i.e. *fractional damage in units of "one fp8 quantization step."* A damage of `0.1` = "ten times smaller than the fp8 floor, almost certainly imperceptible." A damage of `3.4` = "three times worse than dropping to fp8." This is the headline number for the blog.

### The knowledge-level confound (why the floor matters)

The author flagged this in his own post: both base and abliterated models may have **weak** representations of taboo content, so *low divergence between them* does not mean "safe" — it might mean "both are bad." Two defenses:

1. The semantic ceiling (`div(prompt_A, prompt_B)`) calibrates what "a real difference" looks like in this model.
2. Per-category breakdown: if divergence tracks the innocent→extreme gradient (as his cosine did), that's evidence the encoder still *distinguishes* the concepts — i.e. it has *some* representation. A flat profile across categories would be the warning sign of "model doesn't know any of this."

---

## 5. Statistical rigor

### Bootstrap confidence intervals

Every aggregate metric is reported with a **95% CI** via nonparametric bootstrap over prompts ($B = 1000$ resamples). This prevents the cardinal sin of comparing two point estimates whose intervals overlap.

### Determinism contract

- Fixed seed for any stochastic op (none expected in a forward pass, but enforced).
- Identical dtype/device for base and variant in a comparison.
- **Tokenization-alignment assertion:** base and variant must produce identical token IDs for every prompt (they should — abliteration changes weights, not the tokenizer; GGUF round-trips *can* drift, which we assert and flag). Pad/truncate identically.

### Category stratification

Prompts are tagged by category. We report:
- per-category means + CIs (the gradient plot)
- pooled overall
- a "category-discrimination" score (between-category vs within-category divergence ratio) — a proxy for whether the encoder still separates the concepts at all.

---

## 6. What we report (per comparison)

For each `(base, variant, stream, stage)`:

```
stream: video_proj   stage: connector_out   n_prompts: 480
  CKA           0.913   [0.908, 0.917]      damage: 1.9× fp8-floor
  cos_all_tok   0.841   [0.836, 0.846]
  MSE           2.34e-2 [2.29e-2, 2.39e-2]
  rank_corr     0.977   [0.975, 0.979]
  per-category CKA:  innocent 0.94  mild 0.92  violent 0.90  sexual 0.88  extreme 0.87
```

And a one-line **fidelity index** per variant: the damage relative to fp8 floor, pooled across streams, with CI. Lower = more faithful. This is the number that goes in a table row.
