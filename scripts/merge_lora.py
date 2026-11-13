#!/usr/bin/env python3
"""
Merge a PEFT LoRA adapter into a base model, producing a full BF16 safetensors
model that fits the abliterlitics pipeline (weights forensics, lm-eval, KL,
HarmBench all expect full models).

Usage:
    python3 merge_lora.py <base_dir> <adapter_dir> <output_dir>

Output:
    <output_dir>/model.safetensors          # merged weights (BF16), single file
                                            # for small models, OR
    <output_dir>/model-0000N-of-0000M.safetensors + model.safetensors.index.json
                                            # sharded — save_pretrained shards
                                            # at 5GB by default, so any merge of
                                            # a >5GB base lands here
    <output_dir>/config.json            # copied from base
    <output_dir>/generation_config.json # copied from base
    <output_dir>/chat_template.jinja    # copied from base
    <output_dir>/tokenizer.json         # copied from base
    <output_dir>/tokenizer_config.json  # copied from base
    <output_dir>/processor_config.json  # copied from base (for vLLM lesson #31)

Verification:
    - Output safetensors (single file OR shard set per the index) must exist
    - Output size must be within ~5% of base (base may itself be sharded)
    - Merge is idempotent (skips if output already exists and is not a
      partial file — measured against the BASE size, not a hardcoded byte
      threshold)
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


def safetensors_files(d: Path) -> list[Path]:
    """All safetensors weight files of a model dir (single-file OR sharded).

    save_pretrained shards >5GB models into model-0000N-of-0000M.safetensors
    plus a model.safetensors.index.json — a bare `model.safetensors` check
    misses those entirely. Falls back to globbing `model*.safetensors` if the
    index is absent (single-file convention).
    """
    single = d / "model.safetensors"
    if single.is_file():
        return [single]
    index = d / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text()).get("weight_map", {})
        shards = sorted({d / fname for fname in weight_map.values()})
        if shards and all(s.is_file() for s in shards):
            return shards
    return sorted(d.glob("model*.safetensors"))


def has_weights(d: Path) -> bool:
    """True if the dir holds a complete-looking safetensors weight set."""
    return bool(safetensors_files(d))


def total_model_size(d: Path) -> int:
    """Total bytes of the safetensors weight set (0 if none found)."""
    return sum(f.stat().st_size for f in safetensors_files(d))


def skip_reason(output_dir: Path, base_dir: Path) -> str | None:
    """Reason to skip re-merging, or None if a (re-)merge should run.

    A previously merged output is recognized by weight-set presence and a
    total size >0.5x the base model size (BF16 merged ≈ base size; a partial
    or placeholder file is far smaller). The old check (bare model.safetensors
    + hardcoded >20GiB) both missed sharded outputs and re-merged every
    7-13B single-file merge.
    """
    if not has_weights(output_dir):
        return None
    existing = total_model_size(output_dir)
    base_size = total_model_size(base_dir)
    # 0.5x base: a real merged BF16 model is ~= base size; a partial/placeholder
    # file is far smaller. (The old hardcoded >20GiB re-merged every 7-13B merge.)
    # Fallback only when the base itself has no readable safetensors.
    threshold = (base_size // 2) if base_size else 20 * 1024**3
    if existing > threshold:
        return (
            f"[SKIP] {output_dir} already has merged weights "
            f"({existing / 1024**3:.2f} GiB, base {base_size / 1024**3:.2f} GiB). "
            "Use --force to re-merge."
        )
    print(f"[WARN] {output_dir} has small weight set ({existing / 1024**2:.1f} MiB) — likely partial. Re-merging.")
    return None


def main() -> None:
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

    # Idempotency check (shard-aware, base-relative threshold)
    if not args.force:
        reason = skip_reason(output_dir, base_dir)
        if reason:
            print(reason)
            return

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
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

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

    # Verify output (single-file OR sharded; base may be sharded too)
    out_files = safetensors_files(output_dir)
    if not out_files:
        sys.exit("[FAIL] merged safetensors not created (no model*.safetensors found)")
    out_size = total_model_size(output_dir)
    base_size = total_model_size(base_dir)
    if not base_size:
        sys.exit(f"[FAIL] base {base_dir} has no readable safetensors to compare against")
    print(
        f"[DONE] merged model: {out_size / 1024**3:.2f} GiB in {len(out_files)} file(s) "
        f"(base was {base_size / 1024**3:.2f} GiB)"
    )
    if abs(out_size - base_size) / base_size > 0.05:
        print("[WARN] merged size differs from base by >5% — check the output")


if __name__ == "__main__":
    main()
