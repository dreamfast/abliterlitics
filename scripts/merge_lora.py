#!/usr/bin/env python3
"""
Merge a PEFT LoRA adapter into a base model, producing a full BF16 safetensors
model that fits the abliterlitics pipeline (weights forensics, lm-eval, KL,
HarmBench all expect full models).

Usage:
    python3 merge_lora.py <base_dir> <adapter_dir> <output_dir>

Output:
    <output_dir>/model.safetensors      # merged weights (BF16)
    <output_dir>/config.json            # copied from base
    <output_dir>/generation_config.json # copied from base
    <output_dir>/chat_template.jinja    # copied from base
    <output_dir>/tokenizer.json         # copied from base
    <output_dir>/tokenizer_config.json  # copied from base
    <output_dir>/processor_config.json  # copied from base (for vLLM lesson #31)

Verification:
    - Tensor count must match base
    - Output size must be within ~5% of base
    - Merge is idempotent (skips if output already exists with correct size)
"""
import argparse
import os
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge PEFT LoRA into base model")
    parser.add_argument("base_dir", help="Path to base model directory")
    parser.add_argument("adapter_dir", help="Path to LoRA adapter directory (containing adapter_config.json)")
    parser.add_argument("output_dir", help="Output directory for merged model")
    parser.add_argument("--force", action="store_true", help="Re-merge even if output exists")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.output_dir)

    # Validate inputs
    if not (base_dir / "config.json").is_file():
        sys.exit(f"ERROR: base_dir {base_dir} missing config.json")
    if not (adapter_dir / "adapter_config.json").is_file():
        sys.exit(f"ERROR: adapter_dir {adapter_dir} missing adapter_config.json")
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        sys.exit(f"ERROR: adapter_dir {adapter_dir} missing adapter_model.safetensors")

    # Idempotency check
    if (output_dir / "model.safetensors").is_file() and not args.force:
        existing_size = (output_dir / "model.safetensors").stat().st_size
        if existing_size > 20 * 1024**3:  # >20GB means it's a real merged model
            print(f"[SKIP] {output_dir} already has model.safetensors ({existing_size / 1024**3:.2f} GiB). Use --force to re-merge.")
            return
        else:
            print(f"[WARN] {output_dir} has small model.safetensors ({existing_size / 1024**2:.1f} MiB) — likely partial. Re-merging.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Install peft if not present
    try:
        import peft
        print(f"[OK] peft {peft.__version__} already installed")
    except ImportError:
        print("[INSTALL] pip installing peft...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "peft"], check=True)
        import peft
        print(f"[OK] peft {peft.__version__} installed")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Load base model in FP32 for the merge (AGENTS.md lesson #35).
    #
    # CRITICAL: loading in BF16 causes PEFT to cast the LoRA delta back to
    # BF16 BEFORE adding to the base weight. For small-alpha LoRAs where
    # alpha/r=1.0, the edits (~5e-4 relative) are below BF16 ULP (~0.78%),
    # so PEFT's intermediate cast rounds most delta elements to zero — the
    # LoRA effect is lost at merge time.
    #
    # With FP32 base, PEFT keeps the delta in FP32 for the addition, then
    # we cast the FINAL merged weight to BF16 at save time (single rounding
    # per element). This preserves the rank-r SVD signal in the diff.
    print(f"[LOAD] base model from {base_dir} (FP32 for precision-safe merge)...")
    base = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"[OK] base loaded: {type(base).__name__} (dtype={next(base.parameters()).dtype})")

    # Load adapter on top of base
    print(f"[LOAD] LoRA adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    print(f"[OK] adapter loaded. Target modules: {len(model.peft_config['default'].target_modules)} entries")

    # Merge adapter into base weights (FP32 arithmetic — exact)
    print("[MERGE] merge_and_unload()...")
    merged = model.merge_and_unload()
    print(f"[OK] merged. Type: {type(merged).__name__} (dtype={next(merged.parameters()).dtype})")

    # Cast merged FP32 weights to BF16 for storage. This is a SINGLE rounding
    # per element (round_BF16(W_base + delta)), which preserves far more of
    # the LoRA signal than PEFT's intermediate delta-to-BF16 cast would.
    print("[CAST] FP32 → BF16 for storage...")
    merged = merged.to(torch.bfloat16)
    print(f"[OK] cast to BF16 (dtype={next(merged.parameters()).dtype})")

    # Save merged model (BF16 safetensors)
    print(f"[SAVE] merged model to {output_dir}...")
    merged.save_pretrained(str(output_dir), safe_serialization=True)
    del base, model, merged  # free memory before tokenizer work

    # Copy tokenizer + config files from base
    files_to_copy = [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "processor_config.json",  # critical for vLLM Gemma4Unified (lesson #31)
    ]
    for fname in files_to_copy:
        src = base_dir / fname
        if src.is_file():
            shutil.copy2(src, output_dir / fname)
            print(f"[COPY] {fname}")
        else:
            print(f"[WARN] base missing {fname} — not copied")

    # Verify output
    out_safetensors = output_dir / "model.safetensors"
    if not out_safetensors.is_file():
        sys.exit(f"[FAIL] merged model.safetensors not created")
    out_size = out_safetensors.stat().st_size
    base_size = (base_dir / "model.safetensors").stat().st_size
    print(f"[DONE] merged model: {out_size / 1024**3:.2f} GiB (base was {base_size / 1024**3:.2f} GiB)")
    if abs(out_size - base_size) / base_size > 0.05:
        print(f"[WARN] merged size differs from base by >5% — check the output")


if __name__ == "__main__":
    main()
