#!/usr/bin/env python3
"""
KL Divergence measurement reimplementing the Heretic evaluator methodology.

Based on the Heretic abliteration tool by Philipp Emanuel Weidmann (p-e-w).
https://github.com/p-e-w/heretic

Heretic is licensed under AGPL-3.0. This module reimplements the KL measurement
methodology from heretic/src/heretic/model.py (lines 681-700) and evaluator.py
(lines 95-127) without copying source code. The original tool should be cited:

    @misc{heretic,
      author = {Weidmann, Philipp Emanuel},
      title = {Heretic: Fully automatic censorship removal for language models},
      year = {2025},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\\url{https://github.com/p-e-w/heretic}}
    }

Uses transformers model.generate() to get full-vocab first-token logits via
output_scores=True, then computes KL(P||Q) via F.kl_div:

    F.kl_div(logprobs_variant, logprobs_base, reduction="batchmean", log_target=True)

Where logprobs are F.log_softmax over the ENTIRE vocabulary at the first
generated token position, using 100 harmless prompts from mlabonne/harmless_alpaca
(split="test[:100]", column="text") with system prompt "You are a helpful assistant."

Key methodology points (matching Heretic reference):
  - model.generate(max_new_tokens=1, output_scores=True) — NOT direct forward pass
  - scores are float32 from generate internals — NO .float() cast needed
  - F.log_softmax(logits, dim=-1); non-finite logprobs (-inf/+inf/NaN) are clamped
    to LP_FLOOR=-100 before F.kl_div in the compute phase (see below)
   - Left-padding on tokenizer
   - System prompt + response_prefix appended before tokenization
   - Non-finite logprobs clamped to LP_FLOOR in the compute phase (lesson #37:
     F.kl_div(log_target=True) computes exp(-inf)*(target-input) = 0*-inf = NaN
     on PyTorch >= 2.11, which NaN'd every variant on 262K-vocab models)

Optional comparison-level chat template override (--chat-template-file):
   When variants ship modified bundled templates (jailbreak system prompts,
   deleted reasoning-effort prompts, thinking-off defaults), rendering each
   model's OWN template would measure template deltas, not weight deltas.
   Loading the comparison's stock template into the tokenizer before
   apply_chat_template makes every arm render identical prompts, so KL
   isolates the weight change. The KL math (dataset, system prompt,
   1-token logits, batchmean) is unchanged.
   NOTE (qwen38-27b, 2026-09-05): this override is SUPERSEDED for heretic-exact
   protocol runs — the runner mounts the BASE tokenizer for every arm, so the
   tokenizer's OWN template (heretic v1.2.0 main.py:371-399 semantics) already
   renders identical prompts across arms. The measurement position is set by
   the response prefix, not the template.

Response prefix modes (--response-prefix):
   - "auto": heretic main.py:371-399 — commonprefix(base responses).rstrip(" "),
     with CoT-init startswith replacement (CHAIN_OF_THOUGHT_SKIPS). Correct when
     the chat template leaves the think block CLOSED (raw response starts with
     "<think>").
   - "thinkclose" (qwen38-27b, user-directed 2026-09-05): DETERMINISTIC
     adaptation for templates that PREFILL an open "<think>\n" (Qwen3.x,
     AGENTS.md lesson #55c). There the raw response starts directly with
     reasoning-scaffold content and "auto" measures a scaffold position
     ("We" — journal kl-protocol-error, ~1000x below heretic card scale).
     thinkclose appends "</think>\n\n" when the render prefills an open
     <think> (empty CLOSED think block — heretic's "<think></think>" adapted
     to the prefilled format), else heretic's literal "<think></think>".
     First measured token = first CONTENT token. The exact appended string is
     recorded in provenance (response_prefix_mode key) and response_prefix.txt.

Strategy: load ONE model at a time, collect full logits, save to disk,
unload, repeat. KL is computed offline from saved tensors.

Model loading strategy (auto-detected by model size on disk):
  - <=26GB bf16: force onto cuda:0 (single 5090)
  - <=46GB bf16: split across cuda:0 (5090) + cuda:1 (4090)
  - >46GB bf16:  split across both GPUs + CPU RAM offload

Usage:
    # Phase 1a: Detect response prefix (optional, run on base model)
    python3 kl_divergence.py collect \\
        --model /models/base \\
        --output /results/logits_base.pt \\
        --response-prefix auto --save-prefix /results/response_prefix.txt

    # Phase 1b: Collect logits for variant (with known prefix)
    python3 kl_divergence.py collect \\
        --model /models/variant \\
        --output /results/logits_variant.pt \\
        --response-prefix "$(cat /results/response_prefix.txt)"

    # Phase 2: Compute KL from saved logits
    python3 kl_divergence.py compute \\
        --base-logits /results/logits_base.pt \\
        --variant-logits /results/logits_heretic.pt \\
        --variant-label heretic \\
        --output /results/kl_heretic.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from os.path import commonprefix
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src import RESULTS_VERSION

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROMPT_SOURCE = "mlabonne/harmless_alpaca"
DEFAULT_SPLIT = "test[:100]"
DEFAULT_COLUMN = "text"
DEFAULT_NUM_PROMPTS = 100
SYSTEM_PROMPT = "You are a helpful assistant."
METHODOLOGY_VERSION = "2.0"

# GPU memory budgets (conservative — leave headroom for KV cache + activations)
# GPU0_BUDGET MUST leave ~5 GiB above the weight placement: with accelerate
# big-model offload, modules placed on CPU are STREAMED through the GPU during
# forward (set_module_tensor_to_device per module). The largest single stream
# here is lm_head itself — 248,320 x 5120 x 2B = 2.37 GiB for Qwen3.8-27B —
# plus activations and CUDA context. A 30GiB budget filled VRAM to 30.86 GiB
# and OOM'd streaming lm_head (qwen38-27b, 2026-09-05). 26GiB leaves ~5 GiB.
GPU0_BUDGET = "26GiB"  # RTX 5090 (32GB physical)
GPU1_BUDGET = "22GiB"  # RTX 4090 (24GB physical)
CPU_BUDGET = "80GiB"

# Model size thresholds (bf16 safetensor bytes on disk)
SINGLE_GPU_MAX_GB = 29
DUAL_GPU_MAX_GB = 54

# Chain-of-thought skip patterns (from Heretic config.py lines 154-176)
# Each tuple is (cot_initializer, closed_cot_block)
CHAIN_OF_THOUGHT_SKIPS = [
    # Most thinking models (Qwen3, Qwen3.5, etc.)
    # Matching heretic/src/heretic/config.py lines 154-160
    (
        "<think>",
        "<think></think>",
    ),
    # gpt-oss
    (
        "<|channel|>analysis<|message|>",
        "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>",
    ),
    # Unknown, suggested patterns
    (
        "<thought>",
        "<thought></thought>",
    ),
    (
        "[THINK]",
        "[THINK][/THINK]",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_model_size_gb(model_path: str) -> float:
    """Sum safetensors/bin files to estimate bf16 model size in GB."""
    total = 0
    p = Path(model_path)
    for f in p.glob("*.safetensors"):
        total += f.stat().st_size
    for f in p.glob("*.bin"):
        total += f.stat().st_size
    return total / (1024**3)


def build_split_device_map(model_path: str) -> dict:
    """Build a manual device map splitting layers across GPUs and optionally CPU."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = getattr(config, "num_hidden_layers", None)
    if num_layers is None:
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            num_layers = getattr(text_config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError(f"Cannot determine number of layers from config: {type(config).__name__}")

    num_gpus = torch.cuda.device_count()
    gpu_memories = []
    for i in range(num_gpus):
        mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        gpu_memories.append(mem_gb)
    total_gpu_mem = sum(gpu_memories)

    # Estimate model weight size in GB (bf16 = 2 bytes per param, approximate)
    model_size_gb = get_model_size_gb(model_path)

    device_map: dict[str, str | int] = {}
    device_map["model.embed_tokens"] = 0
    device_map["lm_head"] = 0

    if model_size_gb > total_gpu_mem * 0.85:
        # Need CPU offload — put first layers on GPUs, last layers on CPU
        log.info(f"  Model {model_size_gb:.1f}GB exceeds {total_gpu_mem:.0f}GB VRAM, offloading tail layers to CPU")
        # Leave ~4GB headroom per GPU for activations
        usable_gpu = [m - 4 for m in gpu_memories]
        total_usable = sum(usable_gpu)
        gpu_fraction = total_usable / model_size_gb
        gpu_layers = max(1, int(num_layers * gpu_fraction))
        cpu_layers = num_layers - gpu_layers

        # Split GPU layers proportionally
        gpu_per = [int(u / total_usable * gpu_layers) for u in usable_gpu]
        remainder = gpu_layers - sum(gpu_per)
        for i in range(remainder):
            gpu_per[i] += 1

        layer_idx = 0
        for gpu_id, count in enumerate(gpu_per):
            for _ in range(count):
                device_map[f"model.layers.{layer_idx}"] = gpu_id
                layer_idx += 1
        for _ in range(cpu_layers):
            device_map[f"model.layers.{layer_idx}"] = "cpu"
            layer_idx += 1

        device_map["model.norm"] = "cpu"
        log.info(f"  Manual device map: {num_layers} layers -> GPUs {gpu_per} + {cpu_layers} on CPU")
    else:
        # Fit entirely on GPUs
        total_per_gpu = [m / total_gpu_mem * num_layers for m in gpu_memories]
        per_gpu_int = [int(t) for t in total_per_gpu]
        remainder = num_layers - sum(per_gpu_int)
        for i in range(remainder):
            per_gpu_int[i] += 1

        layer_idx = 0
        for gpu_id, count in enumerate(per_gpu_int):
            for _ in range(count):
                device_map[f"model.layers.{layer_idx}"] = gpu_id
                layer_idx += 1

        device_map["model.norm"] = num_gpus - 1
        device_map["lm_head"] = num_gpus - 1
        log.info(
            f"  Manual device map: {num_layers} layers split {per_gpu_int} across {num_gpus} GPUs "
            f"(memories: {[f'{m:.0f}GB' for m in gpu_memories]})"
        )
    return device_map


def get_loading_config(model_path: str, model_size_gb: float) -> dict:
    """Return kwargs for AutoModelForCausalLM.from_pretrained based on size."""
    num_gpus = torch.cuda.device_count()

    if model_size_gb <= SINGLE_GPU_MAX_GB:
        log.info(f"  Model size: {model_size_gb:.1f}GB -> single GPU (cuda:0)")
        return {"device_map": {"": "cuda:0"}}

    if num_gpus >= 2:
        total_vram = sum(torch.cuda.get_device_properties(i).total_memory / (1024**3) for i in range(num_gpus))
        if model_size_gb <= total_vram * 0.85:
            log.info(f"  Model size: {model_size_gb:.1f}GB -> multi GPU manual split")
            return {"device_map": build_split_device_map(model_path)}
        else:
            log.info(f"  Model size: {model_size_gb:.1f}GB exceeds {total_vram:.0f}GB VRAM -> dual GPU + CPU offload")
            return {
                "device_map": "sequential",
                "max_memory": {
                    0: f"{int(torch.cuda.get_device_properties(0).total_memory / (1024**3))}GiB",
                    1: f"{int(torch.cuda.get_device_properties(1).total_memory / (1024**3))}GiB",
                    "cpu": "100GiB",
                },
            }

    log.info(f"  Model size: {model_size_gb:.1f}GB -> single GPU + CPU offload")
    return {
        "device_map": "auto",
        "max_memory": {0: GPU0_BUDGET, "cpu": CPU_BUDGET},
    }


def get_input_device(model) -> torch.device:
    """Determine where to place tokenised inputs (= embedding device)."""
    for name, param in model.named_parameters():
        if "embed" in name.lower() and param.device.type != "meta":
            return param.device
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cuda:0")


def load_prompts(num_prompts: int) -> list[str]:
    """Load harmless prompts matching Heretic's good_evaluation_prompts config.

    Dataset: mlabonne/harmless_alpaca, split="test[:100]", column="text"
    Reference: heretic/src/heretic/config.py lines 410-416
    """
    from datasets import load_dataset

    ds = load_dataset(PROMPT_SOURCE, split=DEFAULT_SPLIT)
    prompts = [row[DEFAULT_COLUMN].strip() for row in ds if row[DEFAULT_COLUMN].strip()]

    if len(prompts) < num_prompts:
        log.info(f"  WARNING: requested {num_prompts} prompts but only {len(prompts)} available")

    return prompts[:num_prompts]


# ---------------------------------------------------------------------------
# Response prefix detection (matching Heretic main.py lines 396-431)
# ---------------------------------------------------------------------------


def detect_response_prefix(
    model,
    tokenizer,
    prompts: list[str],
    system_prompt: str,
    input_device: torch.device,
) -> str:
    """Auto-detect common response prefix by generating responses and finding common prefix.

    Matching Heretic main.py lines 396-431:
    1. Generate responses for prompts (only harmless prompts, unlike Heretic which
       uses good+bad — we don't load harmful prompts in this script)
    2. Find common prefix via os.path.commonprefix
    3. If prefix starts with a CoT initializer, replace with closed CoT block
    4. Re-generate to find additional prefix after CoT block

    NOTE: Heretic uses good_prompts[:100] + bad_prompts[:100] for prefix detection.
    We use only harmless prompts here as a deliberate simplification since we don't
    load harmful prompts. For thinking models (Qwen3/3.5), the CoT prefix (<think)
    is deterministic regardless of prompt content.
    """
    log.info("  Detecting response prefix...")

    # Helper: generate responses in small batches to avoid OOM on large models.
    # Uses batch_size=8 (Heretic uses model.batch_size, default=1, but we batch
    # for efficiency — the prefix detection result is identical since we just
    # need the common prefix across all responses).
    def _generate_responses(chat_prompts_list: list[str], max_tokens: int = 100) -> list[str]:
        all_responses = []
        batch_size = 8
        for start in range(0, len(chat_prompts_list), batch_size):
            batch = chat_prompts_list[start : start + batch_size]
            batch_inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
            ).to(input_device)

            with torch.no_grad():
                batch_outputs = model.generate(
                    **batch_inputs,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False,
                    max_new_tokens=max_tokens,
                )

            batch_input_len = batch_inputs["input_ids"].shape[1]
            batch_responses = tokenizer.batch_decode(
                batch_outputs[:, batch_input_len:],
                skip_special_tokens=False,
            )
            all_responses.extend(batch_responses)

            del batch_outputs, batch_inputs
            gc.collect()
            torch.cuda.empty_cache()

        return all_responses

    # Build chat prompts for first 100 prompts
    chat_prompts = []
    for prompt_text in prompts[:100]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ]
        chat_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        chat_prompts.append(chat_text)

    # Generate responses in small batches (avoids OOM on large models)
    responses = _generate_responses(chat_prompts, max_tokens=100)

    # Find common prefix (commonprefix is character-based, not path-based)
    prefix = commonprefix(responses).rstrip(" ")

    if not prefix:
        log.info("  No common response prefix detected")
        return ""

    log.info(f"  Detected prefix: {prefix!r}")

    # Check for Chain-of-Thought patterns
    for cot_initializer, closed_cot_block in CHAIN_OF_THOUGHT_SKIPS:
        if prefix.startswith(cot_initializer):
            log.info(f"  Found CoT initializer {cot_initializer!r}, replacing with closed block {closed_cot_block!r}")
            prefix = closed_cot_block

            # Re-generate with prefix to find additional common text after CoT block
            chat_prompts_with_prefix = [p + prefix for p in chat_prompts]
            responses2 = _generate_responses(chat_prompts_with_prefix, max_tokens=100)

            additional_prefix = commonprefix(responses2).rstrip(" ")
            if additional_prefix:
                prefix += additional_prefix
                log.info(f"  Extended prefix after CoT block: {prefix!r}")

            break

    log.info(f"  Final response prefix: {prefix!r}")
    return prefix


# ---------------------------------------------------------------------------
# Phase 1: collect
# ---------------------------------------------------------------------------


def _get_model_auto_class(model_path: str):
    """Return the appropriate AutoModel class based on the model architecture.

    VL/multimodal models need AutoModelForImageTextToText, others use
    AutoModelForCausalLM.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")
    # VL / multimodal model types that require ImageTextToText auto class
    vl_types = {
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3_vl",
        "qwen3_vl_moe",
        "llava",
        "llava_next",
        "llava_onevision",
        "mllama",
        "paligemma",
        "idefics",
        "idefics2",
        "idefics3",
        "phi3_v",
        "internvl",
        "pixtral",
        "glm4_1v",
        "qwen2_5_omni",
    }
    if model_type in vl_types:
        from transformers import AutoModelForImageTextToText

        log.info(f"  Detected VL model type '{model_type}', using AutoModelForImageTextToText")
        return AutoModelForImageTextToText
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM


def cmd_collect(args):
    from transformers import AutoTokenizer

    model_size = get_model_size_gb(args.model)

    # --device-map override bypasses auto-detection thresholds entirely
    if args.device_map:
        import json as _json

        try:
            parsed_map = _json.loads(args.device_map)
        except _json.JSONDecodeError:
            # If not JSON, treat as a single device string (e.g. "cuda:0")
            parsed_map = {"": args.device_map}
        load_cfg = {"device_map": parsed_map}
        log.info(f"  Using explicit device map: {parsed_map}")
    else:
        load_cfg = get_loading_config(args.model, model_size)
    load_cfg["torch_dtype"] = torch.bfloat16
    quant_note = ""

    response_prefix: str | None = None
    response_prefix_mode = ""
    if args.response_prefix == "auto":
        pass
    elif args.response_prefix == "thinkclose":
        # HERETIC-EXACT prefix mode for thinking models whose chat template
        # PREFILLS an open <think> in the generation prompt (Qwen3.x —
        # AGENTS.md lesson #55c). Resolved below, after the tokenizer loads.
        response_prefix_mode = "thinkclose"
    elif args.response_prefix == "none" or args.response_prefix == "":
        response_prefix = None
    else:
        response_prefix = args.response_prefix
        log.info(f"  Using explicit response prefix: {response_prefix!r}")

    log.info(f"Loading model: {args.model}")
    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    # Optional comparison-level chat template override (see module docstring).
    # Applied BEFORE any apply_chat_template call (prefix detection + collection),
    # so both phases render through the same stock template.
    chat_template_source = "(tokenizer default)"
    if args.chat_template_file:
        tpl_path = Path(args.chat_template_file)
        if not tpl_path.is_file():
            log.info(f"FATAL: --chat-template-file not found: {tpl_path}")
            sys.exit(1)
        tokenizer.chat_template = tpl_path.read_text()
        chat_template_source = str(tpl_path)
        log.info(
            f"  Chat template OVERRIDE applied: {tpl_path} "
            f"({len(tokenizer.chat_template)} bytes) — bundled template ignored"
        )
    else:
        log.info("  Chat template: tokenizer default (bundled)")

    # Resolve the thinkclose prefix mode (user-directed 2026-09-05, qwen38-27b).
    #
    # HERETIC-EXACT adaptation of "Suppress CoT output" (heretic v1.2.0
    # main.py:384-386): response_prefix = commonprefix(responses).rstrip(" "),
    # replaced by "<think></think>" when it starts with "<think>". That literal
    # algorithm assumes the template leaves the think block CLOSED so the raw
    # response begins with "<think>". Qwen3.x templates PREFILL an open
    # "<think>\n" in the generation prompt, so raw responses begin directly
    # with reasoning-scaffold content — heretic's literal algorithm would
    # measure a scaffold position ("We" — the kl-protocol-error, ~1000x scale
    # break vs heretic card numbers). Deterministic adaptation of the INTENT
    # (empty CLOSED think block at the CONTENT position), no auto-detection
    # (lessons #25/#61 confound discipline):
    #   - render ends with an open <think> prefill -> append "</think>\n\n"
    #   - render does NOT prefill                  -> append "<think></think>"
    # The exact appended string is recorded in provenance and in
    # response_prefix.txt for the variant collects.
    if response_prefix_mode == "thinkclose":
        probe_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": load_prompts(1)[0]},
        ]
        probe_render = tokenizer.apply_chat_template(probe_messages, add_generation_prompt=True, tokenize=False)
        if probe_render.endswith("<think>") or probe_render.endswith("<think>\n"):
            response_prefix = "</think>\n\n"
            log.info(
                "  thinkclose: render ends with OPEN <think> prefill -> "
                'appending "</think>\\n\\n" (empty closed think block; '
                'heretic "<think></think>" adapted to the prefilled format)'
            )
        else:
            response_prefix = "<think></think>"
            log.info(
                "  thinkclose: render does NOT prefill <think> -> appending "
                '"<think></think>" (heretic v1.2.0 main.py:385-386 literal)'
            )
        log.info(f"  Render tail (last 120 chars): ...{probe_render[-120:]!r}")

    AutoModelClass = _get_model_auto_class(args.model)
    try:
        model = AutoModelClass.from_pretrained(
            args.model,
            trust_remote_code=True,
            **load_cfg,
        )
    except RuntimeError as e:
        if "CONVERSION" in str(e) or "conversion" in str(e):
            log.info("  WARNING: weight conversion error, retrying with ignore_mismatched_sizes=True")
            model = AutoModelClass.from_pretrained(
                args.model,
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
                **load_cfg,
            )
        else:
            raise
    model.eval()

    # Apply PEFT LoRA adapter if provided (AGENTS.md lesson #35).
    # The adapter is applied dynamically — NOT merged into base weights — so
    # sub-precision edits are preserved at FP32 during the forward pass.
    if args.lora_adapter:
        from peft import PeftModel

        log.info(f"Applying LoRA adapter: {args.lora_adapter}")
        model = PeftModel.from_pretrained(model, args.lora_adapter)
        model.eval()
        adapter_params = sum(1 for n, _ in model.named_parameters() if "lora_" in n)
        log.info(f"  LoRA adapter applied: {adapter_params} adapter parameters active")

    meta_params = sum(1 for _, p in model.named_parameters() if p.device.type == "meta")
    total_params = sum(1 for _ in model.named_parameters())
    hf_map = getattr(model, "hf_device_map", None)
    has_accel_offload = isinstance(hf_map, dict) and any(d in ("cpu", "disk") for d in set(hf_map.values()))
    if meta_params > 0 and has_accel_offload:
        # transformers 5.x / accelerate: weights placed on cpu/disk by the device
        # map legitimately REMAIN ON THE META DEVICE and are streamed via dispatch
        # hooks during forward. This is the standard >VRAM load path (this file's
        # documented ">46GB -> CPU RAM offload" strategy), NOT a broken load.
        # The old unconditional abort was a false positive for it: it predates
        # transformers-5.x offload semantics and fired before any generation.
        # Numerics are unchanged (bf16 layers computed on their assigned device;
        # placement identical for every arm of the comparison = symmetric).
        n_offload = sum(1 for d in hf_map.values() if d in ("cpu", "disk"))
        log.info(
            f"  NOTE: {meta_params}/{total_params} params on meta device = accelerate "
            f"offload hooks (hf_device_map: {n_offload} cpu/disk entries). Weights "
            f"stream on demand during forward — standard >VRAM path, not a broken "
            f"load. Proceeding."
        )
    elif meta_params > 0:
        for name, param in model.named_parameters():
            if param.device.type == "meta":
                log.info(f"  FATAL meta param: {name}")
                break
        log.info(f"  FATAL: {meta_params}/{total_params} parameters on meta device!")
        log.info("ABORTING: model has parameters on meta device — logits would be garbage.")
        sys.exit(1)
    else:
        devices: dict[str, int] = {}
        for _, param in model.named_parameters():
            d = str(param.device)
            devices[d] = devices.get(d, 0) + 1
        log.info(f"  Device allocation: {devices}")
        has_cpu_params = any("cpu" in d for d in devices)
        if has_cpu_params:
            log.info(f"  WARNING: {devices.get('cpu', 0)} parameters on CPU — results may have precision artifacts")

    # Fix buffers that weren't moved by manual device map (e.g. rotary_emb inv_freq)
    if isinstance(load_cfg.get("device_map"), dict):
        fixed = 0
        for module in model.modules():
            try:
                param = next(module.parameters())
                target_device = param.device
                for buf in module.buffers():
                    if buf.device != target_device:
                        buf.data = buf.data.to(target_device)
                        fixed += 1
            except StopIteration:
                pass
        if fixed:
            log.info(f"  Fixed {fixed} buffers to match their module's device")

    input_device = get_input_device(model)
    log.info(f"  Input device: {input_device}")

    # Load prompts matching Heretic's good_evaluation_prompts
    log.info(f"Loading {args.num_prompts} harmless prompts from {PROMPT_SOURCE} (split={DEFAULT_SPLIT})...")
    raw_prompts = load_prompts(args.num_prompts)
    log.info(f"  Loaded {len(raw_prompts)} prompts")

    # Auto-detect response prefix if requested
    if args.response_prefix == "auto":
        response_prefix = detect_response_prefix(model, tokenizer, raw_prompts, SYSTEM_PROMPT, input_device)

    # Save detected prefix to file if requested
    if args.save_prefix and response_prefix is not None:
        prefix_path = Path(args.save_prefix)
        prefix_path.parent.mkdir(parents=True, exist_ok=True)
        prefix_path.write_text(response_prefix)
        log.info(f"  Saved response prefix to {args.save_prefix}")
    elif args.save_prefix:
        # Write empty file to indicate no prefix
        prefix_path = Path(args.save_prefix)
        prefix_path.parent.mkdir(parents=True, exist_ok=True)
        prefix_path.write_text("")
        log.info(f"  Saved empty response prefix to {args.save_prefix}")

    # Collect logits using model.generate() matching Heretic model.py lines 681-700
    all_logprobs: list[torch.Tensor] = []
    prompt_meta: list[dict] = []

    # Process prompts one at a time (matching Heretic's default batch_size=1)
    # to avoid padding-induced distribution shifts
    for i, prompt_text in enumerate(tqdm(raw_prompts, desc="Collecting logits")):
        # Build chat with system prompt (matching Heretic model.py lines 548-565)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]

        chat_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        # Append response prefix if set (matching Heretic model.py lines 567-572)
        if response_prefix:
            chat_text += response_prefix

        inputs = tokenizer(
            chat_text,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(input_device)

        # Use model.generate() to get logits (matching Heretic model.py lines 583-588, 684-689)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Extract logits from scores[0] (matching Heretic model.py lines 697-700)
        # scores are already float32 from generate internals
        logits = outputs.scores[0]
        log_probs = F.log_softmax(logits, dim=-1)

        # --- validate first prompt (fast-fail on garbage) ---
        if i == 0:
            finite_pct = torch.isfinite(log_probs).float().mean().item() * 100
            neginf_pct = (torch.isinf(log_probs) & (log_probs < 0)).float().mean().item() * 100
            log.info(f"  Prompt 0 sanity: finite={finite_pct:.1f}%, -inf={neginf_pct:.1f}%")
            log.info(f"  Logits dtype: {logits.dtype}, shape: {logits.shape}")
            # Measured-position sanity: log the top-5 candidate first tokens.
            # At the heretic CONTENT position these should be response openers
            # ("I", "Sure", ...); reasoning-scaffold boilerplate ("We need",
            # "Let's") dominating means the prefix puts us inside the think
            # block (kl-protocol-error detection gate).
            top_vals, top_ids = log_probs[0].topk(5)
            top_toks = [tokenizer.decode([int(t)], skip_special_tokens=False) for t in top_ids]
            log.info(
                "  Prompt 0 top-5 first tokens: "
                + ", ".join(f"{tok!r}({float(v.exp()):.3f})" for tok, v in zip(top_toks, top_vals, strict=True))
            )
            if finite_pct < 1.0:
                log.info("ABORTING: first prompt logprobs are ~100% -inf. Model did not load correctly.")
                sys.exit(1)

        all_logprobs.append(log_probs.cpu())
        prompt_meta.append(
            {
                "index": i,
                "preview": prompt_text[:80],
                "prompt_tokens": inputs["input_ids"].shape[-1],
            }
        )

        del outputs, logits, log_probs, inputs
        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    stacked = torch.cat(all_logprobs, dim=0)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Record device placement for post-hoc verification of no CPU offload.
    # With accelerate offload hooks, params stay on "meta" — summarise the
    # dispatch map instead so the metadata reflects the real placement.
    device_allocation: dict[str, int] = {}
    for _, param in model.named_parameters():
        d = str(param.device)
        device_allocation[d] = device_allocation.get(d, 0) + 1
    if isinstance(hf_map, dict) and any(d in ("cpu", "disk") for d in set(hf_map.values())):
        offload_summary: dict[str, int] = {}
        for d in hf_map.values():
            key = str(d)
            offload_summary[key] = offload_summary.get(key, 0) + 1
        device_allocation["hf_offload_summary"] = offload_summary

    # Save enriched metadata matching Heretic's reproducibility requirements
    # TODO: migrate to safetensors format
    torch.save(
        {
            "logprobs": stacked,
            "prompt_meta": prompt_meta,
            "model": args.model,
            "num_prompts": len(raw_prompts),
            "vocab_size": stacked.shape[-1],
            "system_prompt": SYSTEM_PROMPT,
            "chat_template": chat_template_source,
            "response_prefix": response_prefix or "",
            "response_prefix_mode": response_prefix_mode or ("auto" if args.response_prefix == "auto" else "explicit"),
            "split": DEFAULT_SPLIT,
            "column": DEFAULT_COLUMN,
            "methodology_version": METHODOLOGY_VERSION,
            "dtype": str(stacked.dtype),
            "quantization": quant_note,
            "device_allocation": device_allocation,
            "had_cpu_offload": "cpu" in device_allocation
            or (isinstance(hf_map, dict) and any(d in ("cpu", "disk") for d in hf_map.values())),
        },
        out,
    )
    log.info(f"Saved: {out} (shape={stacked.shape}, vocab={stacked.shape[-1]}, dtype={stacked.dtype})")

    del model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Phase 2: compute
# ---------------------------------------------------------------------------


def cmd_compute(args):
    log.info(f"Loading base logits: {args.base_logits}")
    # weights_only=False needed for dict structure; files are locally produced
    base = torch.load(args.base_logits, map_location="cpu", weights_only=False)
    base_lp = base["logprobs"]

    log.info(f"Loading variant logits: {args.variant_logits}")
    # weights_only=False needed for dict structure; files are locally produced
    variant = torch.load(args.variant_logits, map_location="cpu", weights_only=False)
    variant_lp = variant["logprobs"]

    # --- Strict metadata validation (matching reviewer recommendations) ---
    base_prompts = base.get("num_prompts", base_lp.shape[0])
    var_prompts = variant.get("num_prompts", variant_lp.shape[0])
    base_vocab = base_lp.shape[-1]
    var_vocab = variant_lp.shape[-1]

    if base_prompts != var_prompts:
        log.info(f"  FATAL: num_prompts mismatch: base={base_prompts}, variant={var_prompts}")
        sys.exit(1)
    if base_vocab != var_vocab:
        log.info(f"  FATAL: vocab_size mismatch: base={base_vocab}, variant={var_vocab}")
        sys.exit(1)

    # Warn on metadata differences
    base_prefix = base.get("response_prefix", "")
    var_prefix = variant.get("response_prefix", "")
    if base_prefix != var_prefix:
        log.info(f"  WARNING: response_prefix mismatch: base={base_prefix!r}, variant={var_prefix!r}")

    base_sys = base.get("system_prompt", "")
    var_sys = variant.get("system_prompt", "")
    if base_sys != var_sys:
        log.info(f"  WARNING: system_prompt mismatch: base={base_sys!r}, variant={var_sys!r}")

    # Chat-template provenance: arms compared for KL MUST have been rendered
    # through the same template. A mismatch means one collect ran without the
    # comparison-level override — the KL would conflate template + weight deltas.
    base_tpl = base.get("chat_template", "(pre-override run)")
    var_tpl = variant.get("chat_template", "(pre-override run)")
    if base_tpl != var_tpl:
        log.info(
            f"  WARNING: chat_template mismatch: base={base_tpl!r}, variant={var_tpl!r} — "
            f"KL reflects template deltas too!"
        )

    n = min(base_lp.shape[0], variant_lp.shape[0])
    base_lp = base_lp[:n]
    variant_lp = variant_lp[:n]
    log.info(f"  Comparing {n} prompts, vocab_size={base_lp.shape[-1]}")

    # Report -inf statistics BEFORE clamping (informational); clamped below per
    # lesson #37 (F.kl_div log_target=True produces 0*-inf=NaN on PyTorch >= 2.11).
    base_inf_pct = (torch.isinf(base_lp) & (base_lp < 0)).float().mean().item() * 100
    var_inf_pct = (torch.isinf(variant_lp) & (variant_lp < 0)).float().mean().item() * 100
    log.info(f"  Base -inf: {base_inf_pct:.2f}%, Variant -inf: {var_inf_pct:.2f}%")

    if base_inf_pct > 99 or var_inf_pct > 99:
        log.info(
            "  WARNING: logprobs are nearly all -inf — model was likely "
            "broken during collection. Results will be unreliable!"
        )

    # Clamp -inf/+inf/NaN in logprobs BEFORE F.kl_div to avoid 0*-inf=NaN.
    # In PyTorch (esp. >=2.11), F.kl_div(input, target, log_target=True) computes
    # exp(target)*(target-input); when target=-inf, exp(-inf)=0 and 0*(-inf)=NaN.
    # For a 262K vocab, ~200 tokens/prompt underflow to -inf log-prob, so EVERY
    # variant produced NaN before this fix. Clamp to FLOOR=-100 (well below any
    # observed finite log-prob of -57.5; exp(-100)~3.7e-44~0 in float32). This is
    # mathematically equivalent for softmax (still ~0 probability) but keeps
    # arithmetic finite. +inf (rare, from extreme weight mods) -> 0 (prob 1).
    LP_FLOOR = -100.0
    var_neginf = int((torch.isinf(variant_lp) & (variant_lp < 0)).sum())
    var_posinf = int((torch.isinf(variant_lp) & (variant_lp > 0)).sum())
    var_nan = int(torch.isnan(variant_lp).sum())
    base_neginf = int((torch.isinf(base_lp) & (base_lp < 0)).sum())
    base_posinf = int((torch.isinf(base_lp) & (base_lp > 0)).sum())
    base_nan = int(torch.isnan(base_lp).sum())
    if var_neginf or var_posinf or var_nan or base_neginf or base_posinf or base_nan:
        log.info(
            f"  Clamping non-finite logprobs to [{LP_FLOOR}, 0]: "
            f"variant(-inf={var_neginf},+inf={var_posinf},nan={var_nan}), "
            f"base(-inf={base_neginf},+inf={base_posinf},nan={base_nan})"
        )
    variant_lp = torch.nan_to_num(variant_lp, nan=LP_FLOOR, posinf=0.0, neginf=LP_FLOOR)
    base_lp = torch.nan_to_num(base_lp, nan=LP_FLOOR, posinf=0.0, neginf=LP_FLOOR)

    # Compute KL divergence (matching Heretic evaluator.py lines 98-103)
    kl_div = F.kl_div(
        variant_lp,
        base_lp,
        reduction="batchmean",
        log_target=True,
    ).item()

    per_prompt_kl = []
    for i in range(n):
        kl_i = F.kl_div(
            variant_lp[i : i + 1],
            base_lp[i : i + 1],
            reduction="batchmean",
            log_target=True,
        ).item()
        per_prompt_kl.append(kl_i)

    prompt_meta = base.get("prompt_meta", [])
    per_prompt = []
    for i in range(n):
        entry = {
            "prompt_index": i,
            "kl_divergence": per_prompt_kl[i],
        }
        if i < len(prompt_meta):
            entry["prompt_preview"] = prompt_meta[i].get("preview", "")
            entry["prompt_tokens"] = prompt_meta[i].get("prompt_tokens", 0)
        per_prompt.append(entry)

    mean_kl = sum(per_prompt_kl) / len(per_prompt_kl)
    std_kl = (sum((x - mean_kl) ** 2 for x in per_prompt_kl) / len(per_prompt_kl)) ** 0.5

    sorted_kl = sorted(per_prompt_kl)
    median_kl = (sorted_kl[n // 2 - 1] + sorted_kl[n // 2]) / 2 if n % 2 == 0 else sorted_kl[n // 2]

    if mean_kl < 0.01:
        interp = "excellent"
    elif mean_kl < 0.1:
        interp = "very good"
    elif mean_kl < 0.5:
        interp = "moderate"
    elif mean_kl < 1.0:
        interp = "significant"
    else:
        interp = "heavy"

    summary = {
        "results_version": RESULTS_VERSION,
        "variant_label": args.variant_label,
        "base_model": base.get("model", ""),
        "variant_model": variant.get("model", ""),
        "num_prompts": n,
        "vocab_size": int(base_lp.shape[-1]),
        "kl_divergence_batchmean": kl_div,
        "kl_divergence_per_prompt_mean": mean_kl,
        "kl_divergence_per_prompt_std": std_kl,
        "kl_divergence_per_prompt_median": median_kl,
        "kl_divergence_per_prompt_min": min(per_prompt_kl),
        "kl_divergence_per_prompt_max": max(per_prompt_kl),
        "interpretation": interp,
        "chat_template": base_tpl,
        "methodology": (
            f"F.kl_div(logprobs_variant, logprobs_base, reduction='batchmean', "
            f"log_target=True) on full vocab first-token log_softmax via "
            f"model.generate(max_new_tokens=1, output_scores=True), "
            f"matching Heretic evaluator. Dataset: {PROMPT_SOURCE} split={DEFAULT_SPLIT} "
            f"column={DEFAULT_COLUMN}. System prompt: '{SYSTEM_PROMPT}'. "
            f"Response prefix: {base.get('response_prefix', '')!r}. "
            f"Methodology version: {METHODOLOGY_VERSION}."
        ),
        "per_prompt_results": per_prompt,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    log.info(f"\n=== KL DIVERGENCE: {args.variant_label} ===")
    log.info(f"  Batchmean KL:          {kl_div:.4f}")
    log.info(f"  Per-prompt mean:       {mean_kl:.4f}")
    log.info(f"  Per-prompt median:     {summary['kl_divergence_per_prompt_median']:.4f}")
    log.info(f"  Per-prompt std:        {std_kl:.4f}")
    log.info(f"  Interpretation:        {interp}")
    log.info(f"Saved: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="KL Divergence measurement matching Heretic methodology")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- collect subcommand ---
    p_collect = sub.add_parser("collect", help="Collect full-vocab logits from a model")
    p_collect.add_argument("--model", required=True, help="Path to model directory")
    p_collect.add_argument("--tokenizer", default=None, help="Path to tokenizer (defaults to --model)")
    p_collect.add_argument("--output", required=True, help="Path to save logits .pt file")
    p_collect.add_argument(
        "--num-prompts",
        type=int,
        default=DEFAULT_NUM_PROMPTS,
        help=f"Number of prompts to use (default: {DEFAULT_NUM_PROMPTS})",
    )
    p_collect.add_argument(
        "--response-prefix",
        default="none",
        help=(
            'Response prefix: "auto" (detect via generation), "thinkclose" '
            "(deterministic heretic-adapted CoT suppression: '</think>\\n\\n' "
            "when the render prefills an open <think>, '<think></think>' "
            'otherwise), "none" (no prefix), or literal string'
        ),
    )
    p_collect.add_argument(
        "--save-prefix",
        default=None,
        help="Path to save detected response prefix (for passing to variant collects)",
    )
    p_collect.add_argument(
        "--chat-template-file",
        default=None,
        help=(
            "Path to a chat template .jinja file loaded into the tokenizer BEFORE "
            "rendering (overrides the model's bundled template). Pass the "
            "comparison's stock template so every arm renders identical prompts — "
            "KL then measures weight deltas, not template deltas."
        ),
    )
    p_collect.add_argument(
        "--device-map",
        default=None,
        help=(
            'Override auto-detection. Pass a JSON string like \'{"": "cuda:0"}\' '
            "to force all params onto a specific device. Bypasses size thresholds."
        ),
    )
    p_collect.add_argument(
        "--lora-adapter",
        default=None,
        help=(
            "Path to a PEFT LoRA adapter directory. When set, the base model is "
            "loaded normally and the adapter is applied dynamically at FP32 "
            "precision (NOT merged into BF16 — preserves sub-precision edits, "
            "AGENTS.md lesson #35)."
        ),
    )
    # --- compute subcommand ---
    p_compute = sub.add_parser("compute", help="Compute KL from saved logits")
    p_compute.add_argument("--base-logits", required=True, help="Path to base logits .pt file")
    p_compute.add_argument("--variant-logits", required=True, help="Path to variant logits .pt file")
    p_compute.add_argument("--variant-label", required=True, help="Label for the variant (e.g. heretic)")
    p_compute.add_argument("--output", required=True, help="Path to save KL results .json file")

    args = parser.parse_args()
    if args.command == "collect":
        if args.num_prompts <= 0:
            parser.error("--num-prompts must be positive")
        cmd_collect(args)
    elif args.command == "compute":
        cmd_compute(args)


if __name__ == "__main__":
    main()
