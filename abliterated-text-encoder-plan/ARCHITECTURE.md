# Architecture

Model-agnostic by construction: the **metrics core never imports a model**. Per-model knowledge lives in small `Backend` classes. The shared object that crosses the boundary is a `ConditioningArtifact` — a bag of named conditioning streams captured at a named pipeline stage.

---

## 1. Data flow

```
                 ┌──────────── backends/ ────────────┐
prompts ──► Backend.encode(prompts) ──► ConditioningArtifact (per stage)
   │         (loads ckpt, runs LLM fwd,         ▲
   │          extracts via hooks at each stage)  │ multiple stages:
   │                                            │ raw_hidden | projected | connector_out
   └──► same Backend, base ckpt + variant ckpt   │
                                                     │
                ┌──────────── core/ ────────────────┴────────────┐
                │  compare(base_art, variant_art, reference_art) │
                │     per-stream: CKA, cos_all_tok, L2/MSE,      │
                │     rank_corr  +  noise-floor normalization     │
                │     + bootstrap CIs + per-category              │
                └──────────────────────┬──────────────────────────┘
                                       ▼
                                  Report (markdown + json)
```

The core is pure-tensor: give it two artifacts, get numbers. No torch models, no tokenizers, no model repos. This is what makes it unit-testable without GPUs and reproducible.

---

## 2. Core types (`tebench/core/artifact.py`)

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import torch

class Stage(str, Enum):
    RAW_HIDDEN    = "raw_hidden"      # encoder hidden states, pre-projection
    PROJECTED     = "projected"       # after the linear projection(s)
    CONNECTOR_OUT = "connector_out"   # after any 1D connector transformer
    DIT_INPUT     = "dit_input"       # exactly what enters cross-attention

class Pooling(str, Enum):
    NONE = "none"
    MEAN = "mean"
    LAST = "last"
    CLS  = "cls"

@dataclass(frozen=True)
class StreamSpec:
    """Declared by a backend. Describes one conditioning stream it produces."""
    name: str                       # "video_proj", "audio_proj", "layer50"
    source_encoder: str             # "qwen3vl_lm", "gemma3", "gemma4"
    layers: tuple[int, ...] | str   # (48,) | (0,8,16,...,48) | "all"
    pooling: Pooling                # how multiple layers/tokens collapse
    stage: Stage                    # pipeline stage this stream is captured at
    dim: int                        # resulting feature dim d

@dataclass
class Stream:
    name: str
    spec: StreamSpec
    # activations: always [n_prompts, seq_len, d] (squeeze seq_len for pooled)
    activations: torch.Tensor
    # attention mask [n_prompts, seq_len] (1 = real token, 0 = pad); None if pooled
    mask: torch.Tensor | None = None

@dataclass
class ConditioningArtifact:
    model: str                      # "h3", "ltx2", "ltx25"
    backend: str                    # backend class name
    ckpt_id: str                    # path or repo id (hashed for reproducibility)
    stage: Stage
    streams: dict[str, Stream]
    prompts: list[str]
    categories: list[str]           # len == len(prompts), aligned
    dtype: str                      # "bf16" | "fp32" | "fp8" | "gguf_q6" ...
    device: str
    meta: dict = field(default_factory=dict)   # tokenization hash, layer count, etc.

    def cat(self, other: "ConditioningArtifact") -> torch.Tensor:
        """Sanity: same prompts/categories/tokenization before any compare."""
        ...
```

**Invariants enforced on construction:**
- `activations` dtype is fp32 internally for all metric math (cast on load; the *source* dtype is recorded in `self.dtype` for the noise-floor reference).
- `mask` is present iff `seq_len > 1`.
- `len(prompts) == len(categories) == activations.shape[0]`.

---

## 3. Backend protocol (`tebench/core/backend.py`)

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Backend(Protocol):
    name: str                       # "h3" | "ltx2" | "ltx25"
    streams: list[StreamSpec]       # declared up front

    def load(self, ckpt: str, dtype: str = "bf16", device: str = "cuda") -> None: ...

    # primary entry: encode a prompt batch, capture ALL declared stages
    def encode(self, prompts: list[str], categories: list[str],
               stages: list[Stage] | None = None) -> list[ConditioningArtifact]: ...

    # the variant loader MUST produce identical token ids to the base
    def tokenize(self, prompts: list[str]) -> dict: ...   # for the alignment assertion
```

### Backend registry

```python
BACKENDS: dict[str, type[Backend]] = {
    "h3":   H3Backend,
    "ltx2": LTX2Backend,
    "ltx25":LTX25Backend,
}

def get_backend(name: str) -> Backend: ...
```

Adding a model = writing one file in `backends/` + one registry line. Krea2 (if ever LLM-confirmed) is a single-file addition.

### Hook strategy (how backends capture stages)

Backends use forward hooks on the relevant modules, run **one** forward pass per (ckpt, prompt-batch), and capture tensors at each declared stage into a dict keyed by `Stage`:

```python
hooks = {
    Stage.RAW_HIDDEN:    register on <backbone last layers>,  store hidden_states[layer_set],
    Stage.PROJECTED:     register on <feature_extractor output>,
    Stage.CONNECTOR_OUT: register on <embeddings_connector output>,
}
```

One forward → all stages. This matters because loading a 32B model twice (base + variant) is the expensive part; we never want to re-run the forward per stage.

---

## 4. Metrics core (`tebench/core/metrics.py`)

Pure functions, fp32, no model imports. Each returns a `MetricResult(value, ci_low, ci_high)`.

```python
def cka_linear(X: Tensor, Y: Tensor) -> float: ...        # METRICS.md §2
def cka_rbf(X: Tensor, Y: Tensor, sigma: float | None = None) -> float: ...
def cosine_all_tokens(X, Y, masks) -> float: ...          # §3.1
def cosine_last_token(X, Y) -> float: ...                 # for blog comparability
def l2_distance(X, Y) -> float: ...                       # §3.2
def mse(X, Y) -> float: ...
def rank_correlation(X, Y) -> float: ...                  # §3.3 Spearman of pairwise dists
def per_dim_drift(X, Y) -> Tensor: ...                    # §3.4
```

### Linear CKA reference implementation (to vendor)

```python
def cka_linear(X: Tensor, Y: Tensor) -> float:
    # X,Y: [n, d]  (flattened across tokens for a stream, or pooled)
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    xtx = (X.T @ X).norm()           # ||X^T X||_F
    yty = (Y.T @ Y).norm()
    ytx = (Y.T @ X).norm()
    return (ytx * ytx / (xtx * yty)).clamp(min=0.0).item()
```

### Compare + noise floor (`tebench/core/compare.py`)

```python
@dataclass
class StreamReport:
    stream: str
    stage: Stage
    cka:        MetricResult
    cos_all:    MetricResult
    cos_last:   MetricResult
    mse:        MetricResult
    rank_corr:  MetricResult
    per_category: dict[str, MetricResult]      # category -> cka
    per_dim: Tensor                            # drift profile
    damage_fp8_units: MetricResult             # relative to noise floor (METRICS.md §4)

def compare(base: ConditioningArtifact,
            variant: ConditioningArtifact,
            numerical_floor: ConditioningArtifact | None,   # base fp32 vs base bf16
            quant_floor: ConditioningArtifact | None,       # base bf16 vs base fp8
            n_boot: int = 1000) -> list[StreamReport]:
    """Asserts base/variant share prompts, categories, tokenization; then per-stream metrics."""
```

### Bootstrap (`tebench/core/bootstrap.py`)

Nonparametric bootstrap over the prompt axis, $B=1000$, returns percentile CIs. Vectorized over the resample dimension.

---

## 5. Prompt suite (`tebench/core/suite.py` + `prompts/suite.yaml`)

```yaml
# each prompt tagged with a category; categories drive per-category metrics
- text: "a golden retriever catching a frisbee in a park"
  category: innocent
- text: "a woman in a bikini on a beach"
  category: mild
# ... violent / sexual / extreme / multilingual
```

- **Tokenization-alignment assertion:** `Backend.tokenize` must return identical ids for base & variant. GGUF round-trips *can* drift → caught and flagged (not silently fixed).
- **Determinism:** same pad/truncate policy; max_length from the model (H3: 1024, LTX: 1024).
- **Stratification:** ~80–100 prompts/category minimum for stable CIs; multilingual stratum (en + 2–3 others).

---

## 6. Report (`tebench/core/report.py`)

```python
@dataclass
class VariantReport:
    variant_id: str
    dtype: str
    streams: list[StreamReport]
    fidelity_index: MetricResult   # pooled damage in fp8-floor units, across streams

@dataclass
class Report:
    base_id: str
    reference_floors: dict          # numerical, quant
    variants: list[VariantReport]
    prompt_set_hash: str
    timestamp: str

    def to_markdown(self) -> str: ...   # the blog-ready table
    def to_json(self) -> dict: ...
    def to_html(self) -> str: ...       # optional, jinja2 template
```

The **fidelity index** is the one-number summary per variant: pooled damage in units of "one fp8 quantization step" (`METRICS.md` §4). Lower = more faithful.

---

## 7. Module layout

```
tebench/
  __init__.py
  cli.py                      # tebench compare --base ... --variant ... --backend h3
  core/
    __init__.py
    artifact.py               # Stage, Pooling, StreamSpec, Stream, ConditioningArtifact
    backend.py                # Backend protocol + BACKENDS registry + get_backend
    metrics.py                # cka_linear/rbf, cosine_all/last, l2, mse, rank_corr, per_dim
    compare.py                # compare() + StreamReport + damage_fp8_units
    bootstrap.py              # percentile CIs
    suite.py                  # prompt loading, stratification, tokenization alignment
    report.py                 # Report schema + markdown/json/html renderers
  backends/
    __init__.py
    base.py                   # shared LLM-forward helpers (hook mgr, dtype/device)
    h3.py                     # H3Backend — single stream, layer 50 (confirm)
    ltx2.py                   # LTX2Backend — V1 extractor, aggregate_embed + connector
    ltx25.py                  # LTX25Backend — V2 extractor, video + audio streams
prompts/
  suite.yaml
tests/
  conftest.py
  test_metrics.py             # CKA known-answer tests (see PLAN.md T1.2)
  test_artifact.py
  test_bootstrap.py
  test_compare.py
  backends/
    test_ltx2_acceptance.py   # reproduce the blog's cosine table
    test_h3_smoke.py          # tiny prompt set, shape checks (GPU gate)
docs/                         # = this planning folder, relocated
```

---

## 8. Reproducibility & determinism contract

- `Report` records: base/variant ckpt hash, prompt-set hash, dtype, device, torch/transformers versions, seed.
- All metric math in fp32 (cast on artifact construction; source dtype recorded).
- One forward pass per (ckpt, batch); stages captured via hooks, not re-runs.
- **Validation gate (do once per backend):** capture a conditioning tensor through ComfyUI's real pipeline for a fixed prompt, and assert our standalone forward reproduces it within fp tolerance (`atol=1e-3` bf16). This is the evidence that "standalone ≠ ComfyUI" is not silently wrong. Documented per backend in `MODELS.md`.

---

## 9. What is explicitly NOT in the architecture

- **No ComfyUI coupling.** Standalone `transformers` / `ltx-core` forward only. (ComfyUI is used solely as a one-time reference-tensor source for the validation gate.)
- **No private worker thread / async.** Batch job; emit a report file.
- **No claim of "uncensoring."** The tool measures conditioning perturbation, full stop (`RESEARCH.md` R5/R6).
- **No DiT loading in the core path.** Downstream DiT-step divergence (METRICS §3.5) is an optional Phase-7 stretch requiring a model-framework DiT loader; not in the core interface.
