# Examples

Step-by-step workflows for common Abliterlitics tasks.

---

## Example 1: Setting Up a New Comparison

### Directory Structure

```bash
mkdir -p ~/comparisons/qwen35-4b
cd ~/comparisons/qwen35-4b

# Place your model safetensors directories here
# (Each directory should contain .safetensors files + config.json + tokenizer)
ls
# Qwen3.5-4B/
# Qwen3.5-4B-heretic/
# Qwen3.5-4B-hauhau/
# Qwen3.5-4B-huihui/
```

### Create comparison.json

```bash
cat > comparison.json << 'EOF'
{
  "name": "qwen35-4b",
  "base": "Qwen3.5-4B",
  "variants": {
    "heretic": { "path": "Qwen3.5-4B-heretic" },
    "hauhau": { "path": "Qwen3.5-4B-hauhau" },
    "huihui": { "path": "Qwen3.5-4B-huihui" }
  }
}
EOF
```

### Validate

```bash
cd ~/abliterlitics/
./abliterlitics.sh validate ~/comparisons/qwen35-4b/
# Output: "Validation passed: qwen35-4b (3 variants)"
```

---

## Example 2: Running Weight Analysis Only

```bash
./abliterlitics.sh weights ~/comparisons/qwen35-4b/
```

This runs all 11 weight analysis scripts:
1. **Panel comparison** — Which tensors changed, by how much
2. **Edit vector analysis** — Direction and magnitude of modifications
3. **SVD analysis** — Decomposition of edit vectors
4. **Technique fingerprint** — What pattern each technique uses
5. **Technique correlation** — Do techniques converge?
6. **Subspace alignment** — Principal component overlap
7. **Layer analysis** — Depth-wise distribution of changes
8. **Low-rank reconstruction** — Can edits be compressed?
9. **Expert analysis** — MoE-specific analysis (GLM only)
10. **Cross-architecture comparison** — Same technique, different models
11. **Stacking analysis** — Additive vs. diminishing returns

Results are saved to `~/comparisons/qwen35-4b/results/` with canonical filenames like:
- `panel_comparison.json`
- `edit_vector_heretic.json`
- `svd_hauhau.json`
- `correlation_heretic_vs_hauhau.json`

---

## Example 3: Running the Full Pipeline

```bash
./abliterlitics.sh auto ~/comparisons/qwen35-4b/
```

This runs all phases in order:
1. Weight analysis (~5-15 minutes per variant)
2. KL divergence (~10-30 minutes per variant, requires GPU)
3. lm-evaluation-harness (~2-6 hours per variant, requires GPU)
4. HarmBench (~30-60 minutes per variant, requires GPU)
5. Graph + report generation

**Tip:** Use `--skip-existing` (default) to resume interrupted runs.

---

## Example 4: Adding a New Variant to an Existing Comparison

```bash
# 1. Place the new model directory in the comparison folder
cp -r ~/models/Qwen3.5-4B-custom ~/comparisons/qwen35-4b/

# 2. Edit comparison.json to add the variant
# Add under "variants":
#   "custom": { "path": "Qwen3.5-4B-custom" }

# 3. Run only the new variant (--skip-existing preserves old results)
./abliterlitics.sh auto ~/comparisons/qwen35-4b/

# The tool skips existing results and only processes the new variant
```

---

## Example 5: Cross-Architecture Comparison

Compare the same abliteration technique across different model sizes:

```bash
# Set up separate comparisons for each size
for size in 2b 4b 9b 27b; do
  mkdir -p ~/comparisons/qwen35-${size}
  cat > ~/comparisons/qwen35-${size}/comparison.json << EOF
{
  "name": "qwen35-${size}",
  "base": "Qwen3.5-${size^^}/",
  "variants": {
    "heretic": { "path": "Qwen3.5-${size^^}-heretic/" }
  }
}
EOF
  ./abliterlitics.sh weights ~/comparisons/qwen35-${size}/
done

# Then use cross_arch_comparison.py to compare across sizes
docker run --rm --runtime=nvidia \
  -v ~/comparisons:/comparisons:ro \
  -v ~/comparisons/cross_arch_results:/results \
  -e PYTHONPATH=/app/src \
  abliterlitics-forensics:1.0.0 \
  python3 /app/src/weight/cross_arch_comparison.py \
    --results-dirs /comparisons/qwen35-2b/results /comparisons/qwen35-4b/results \
                    /comparisons/qwen35-9b/results /comparisons/qwen35-27b/results \
    --output /results/cross_arch_qwen35.json
```

---

## Example 6: Provenance Investigation (Stacking Analysis)

Investigate whether applying HauhauCS on top of Heretic produces additive effects:

```bash
# This is an investigation-specific tool — uses explicit CLI flags, not comparison.json
docker run --rm --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0 \
  -v ~/comparisons/qwen35-4b:/models:ro \
  -v ~/comparisons/qwen35-4b/results:/results \
  -e PYTHONPATH=/app/src \
  abliterlitics-forensics:1.0.0 \
  python3 /app/src/weight/stacking_analysis.py \
    --base /models/Qwen3.5-4B \
    --variant-a /models/Qwen3.5-4B-heretic \
    --variant-b /models/Qwen3.5-4B-hauhau \
    --output /results/stacking_heretic_vs_hauhau.json
```

---

## Example 7: Custom Inference Backend

Force GGUF fallback for a model that doesn't fit in VRAM:

```bash
./abliterlitics.sh --backend llamacpp lm-eval ~/comparisons/glm-47/
```

Or use `ik_llamacpp` for faster inference:

```bash
./abliterlitics.sh --backend ik_llamacpp harmbench ~/comparisons/glm-47/
```

With `--backend auto` (default), the tool selects the best backend based on:
- Model size vs. available GPU VRAM
- Whether BitsAndBytes 4-bit quantization can fit the model
- Whether GGUF conversion is needed

---

## Example 8: Generating Graphs

After running any analysis, generate SVG visualizations:

```bash
./abliterlitics.sh graphs ~/comparisons/qwen35-4b/
```

Output graphs include:
- `technique_venn.svg` — Venn diagram of overlapping changed tensors
- `aggressiveness_bar.svg` — Per-technique edit magnitude
- `layer_heatmap.svg` — Layer-wise edit magnitude heatmap
- `svd_spectrum.svg` — Singular value spectrum comparison
- `expert_heatmap.svg` — Expert-level changes (MoE models)
- `edit_progression.svg` — Edit norm vs. layer depth
- `technique_radar.svg` — Technique fingerprint radar chart

---

## Example 9: Running on Specific GPU

On a multi-GPU system, select which GPU to use:

```bash
# Use GPU 0 (the tool auto-detects which GPU this maps to in Docker)
./abliterlitics.sh --gpu 0 kl ~/comparisons/qwen35-4b/

# Use GPU 1
./abliterlitics.sh --gpu 1 lm-eval ~/comparisons/qwen35-4b/
```

See [GPU_SETUP.md](GPU_SETUP.md) for details on Docker GPU index mapping.

---

## Example 10: Dry Run

Preview what commands would execute without running them:

```bash
./abliterlitics.sh --dry-run auto ~/comparisons/qwen35-4b/

# Output:
# [DRY-RUN] Would run: docker run --rm --runtime=nvidia ...
# [DRY-RUN] Would run: docker run --rm --runtime=nvidia ...
```
