#!/usr/bin/env python3
"""
KL Divergence measurement matching Heretic's exact methodology.

Uses transformers model.generate() to get full-vocab first-token logits via
output_scores=True, then computes KL(P||Q) via F.kl_div:

    F.kl_div(logprobs_variant, logprobs_base, reduction="batchmean", log_target=True)

Where logprobs are F.log_softmax over the ENTIRE vocabulary at the first
generated token position, using 100 harmless prompts from mlabonne/harmless_alpaca
(split="test[:100]", column="text") with system prompt "You are a helpful assistant."

Key methodology points (matching Heretic reference):
  - model.generate(max_new_tokens=1, output_scores=True) — NOT direct forward pass
  - scores are float32 from generate internals — NO .float() cast needed
  - F.log_softmax(logits, dim=-1) — no clamping of -inf values
  - Left-padding on tokenizer
  - System prompt + response_prefix appended before tokenization
  - No -inf clamping in compute phase

Reference: heretic/src/heretic/model.py lines 681-700, evaluator.py lines 95-127

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
GPU0_BUDGET = "30GiB"  # RTX 5090 (32GB physical)
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


def validate_model_loaded(model) -> bool:
    """Abort if any parameters are still on meta device (= not loaded)."""
    meta_params = []
    total = 0
    for name, param in model.named_parameters():
        total += 1
        if param.device.type == "meta":
            meta_params.append(name)

    if meta_params:
        for name in meta_params[:10]:
            log.info(f"  FATAL meta param: {name}")
        if len(meta_params) > 10:
            log.info(f"  ... and {len(meta_params) - 10} more")
        log.info(f"  FATAL: {len(meta_params)}/{total} parameters on meta device!")
        return False

    # Summarise device allocation
    devices: dict[str, int] = {}
    for _, param in model.named_parameters():
        d = str(param.device)
        devices[d] = devices.get(d, 0) + 1
    log.info(f"  Device allocation: {devices}")
    return True


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


def cmd_collect(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    if args.response_prefix == "auto":
        pass
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

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            **load_cfg,
        )
    except RuntimeError as e:
        if "CONVERSION" in str(e) or "conversion" in str(e):
            log.info("  WARNING: weight conversion error, retrying with ignore_mismatched_sizes=True")
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
                **load_cfg,
            )
        else:
            raise
    model.eval()

    meta_params = sum(1 for _, p in model.named_parameters() if p.device.type == "meta")
    total_params = sum(1 for _ in model.named_parameters())
    if meta_params > 0:
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

    # Record device placement for post-hoc verification of no CPU offload
    device_allocation: dict[str, int] = {}
    for _, param in model.named_parameters():
        d = str(param.device)
        device_allocation[d] = device_allocation.get(d, 0) + 1

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
            "response_prefix": response_prefix or "",
            "split": DEFAULT_SPLIT,
            "column": DEFAULT_COLUMN,
            "methodology_version": METHODOLOGY_VERSION,
            "dtype": str(stacked.dtype),
            "quantization": quant_note,
            "device_allocation": device_allocation,
            "had_cpu_offload": "cpu" in device_allocation,
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

    n = min(base_lp.shape[0], variant_lp.shape[0])
    base_lp = base_lp[:n]
    variant_lp = variant_lp[:n]
    log.info(f"  Comparing {n} prompts, vocab_size={base_lp.shape[-1]}")

    # Report -inf statistics (informational only — NO clamping)
    base_inf_pct = (torch.isinf(base_lp) & (base_lp < 0)).float().mean().item() * 100
    var_inf_pct = (torch.isinf(variant_lp) & (variant_lp < 0)).float().mean().item() * 100
    log.info(f"  Base -inf: {base_inf_pct:.2f}%, Variant -inf: {var_inf_pct:.2f}%")

    if base_inf_pct > 99 or var_inf_pct > 99:
        log.info(
            "  WARNING: logprobs are nearly all -inf — model was likely "
            "broken during collection. Results will be unreliable!"
        )

    # NO clamping — matching Heretic which does no clamping
    # F.kl_div with log_target=True handles -inf correctly:
    # exp(-inf) = 0, so those tokens contribute 0 to KL divergence

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
    p_collect.add_argument("--comparison", help="Path to comparison.json dir (resolves --model)")
    p_collect.add_argument(
        "--num-prompts",
        type=int,
        default=DEFAULT_NUM_PROMPTS,
        help=f"Number of prompts to use (default: {DEFAULT_NUM_PROMPTS})",
    )
    p_collect.add_argument(
        "--response-prefix",
        default="none",
        help='Response prefix: "auto" (detect), "none" (no prefix), or literal string',
    )
    p_collect.add_argument(
        "--save-prefix",
        default=None,
        help="Path to save detected response prefix (for passing to variant collects)",
    )
    p_collect.add_argument(
        "--device-map",
        default=None,
        help=(
            'Override auto-detection. Pass a JSON string like \'{"": "cuda:0"}\' '
            "to force all params onto a specific device. Bypasses size thresholds."
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
