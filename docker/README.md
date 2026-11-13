# Docker images

Build all documented images from the repo root:

```bash
./abliterlitics.sh build              # builds every image below except nightly
./abliterlitics.sh --dry-run build    # preview only
```

`cmd_build` (in `abliterlitics.sh`) builds every image in the table below
**except** the nightly one, which has no in-repo Dockerfile (see the gap
note at the bottom).

| Image (default tag) | Dockerfile | Base | Purpose |
|---|---|---|---|
| `abliterlitics-forensics:1.0.0` | `Dockerfile.forensics` | `nvcr.io/nvidia/pytorch:26.02-py3` | Weight forensics, KL divergence, graphs |
| `abliterlitics-lmeval:1.0.0` | `Dockerfile.lmeval` | `vllm/vllm-openai:v0.19.0` | lm-eval default (baseline stack) |
| `abliterlitics-lmeval-gemma4unified:1.0.0` | `Dockerfile.lmeval_gemma4_unified` | `vllm/vllm-openai:gemma4-unified` pinned **by digest** (dev snapshot on the vLLM 0.26 line) | `Gemma4UnifiedForConditionalGeneration` archs (AGENTS.md lesson #11) |
| `abliterlitics-llamacpp:1.0.0` | `Dockerfile.llamacpp` | `ghcr.io/ggml-org/llama.cpp:server-cuda` | llama.cpp backend |
| `abliterlitics-ik-llamacpp:1.0.0` | `Dockerfile.ik-llamacpp` | `nvidia/cuda:12.8.0-*` | ik_llamacpp backend (`IK_LLAMA_CPP_COMMIT` build-arg selects the source commit) |
| `abliterlitics-te:1.2.0` | `Dockerfile.te` | `nvcr.io/nvidia/pytorch:26.02-py3` | Text-encoder (tebench) analysis |

Per-comparison image selection is driven by `comparison.json` →
`settings.lmeval_image` / `settings.forensics_image`, exported by
`runners/_load_comparison.py` and consumed as env-var overrides in
`runners/common.sh` (AGENTS.md lesson #11).

## GAP: `abliterlitics-lmeval-nightly:1.0.0` has no Dockerfile

The nightly image (vLLM 0.20.1, with the `gemma4_reasoning_parser`) is
referenced by AGENTS.md lesson #11, `runners/common.sh`, and
`runners/_load_comparison.py` — but **cannot be rebuilt from this repo**:
no Dockerfile is checked in. It exists only as an image on the original
build host. Options if you need it:

- export/import the existing image: `docker save` / `docker load`
- author a new `Dockerfile.lmeval_nightly` based on `vllm/vllm-openai:v0.20.1`
  (mirroring `Dockerfile.lmeval`), then add it to `cmd_build`

`./abliterlitics.sh build` explicitly reports this image as skipped.

## Base-image pinning notes

- `Dockerfile.lmeval_gemma4_unified` pins its base **by digest** — the
  upstream `gemma4-unified` tag is a floating dev snapshot; the digest keeps
  rebuilds reproducible against the manifest the current image was built
  from.
- `Dockerfile.llamacpp` still floats on
  `ghcr.io/ggml-org/llama.cpp:server-cuda` (known gap, audit
  `.cache/audit/root-packaging.md` — no versioned replacement tag was
  available as evidence at fix time).
