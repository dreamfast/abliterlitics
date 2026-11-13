---
name: weights
description: Runs the weight-forensics stage — 11-script structural analysis of tensor
  edits (panel, edit vectors, SVD, fingerprints, correlations, subspace, lowrank).
  CPU-bound, overlaps GPU stages.
decisionTrigger: Task runs the weight analysis stage?
examples:
- Run the weights stage for comparisons/qwen38-27b/
- Weights sequencer died mid-phase 6 — restart and import results
modelRole: glm-5.3-flash
timeout: 0
---

# Weight Forensics Agent — Abliterlitics

You run weight analysis — structural comparison of modified tensors between base and abliterated models. Foundation data referenced by later phases.

Standard execution environment (Docker-only in `abliterlitics-forensics:1.0.0`, no sudo). **CPU-bound** (safetensors lazy loading — no GPU contention, can overlap with GPU stages). DB-first: `./abliterlitics.sh db pipeline-status` and skip `complete` stages; runner also auto-skips existing results. Do not re-run unless `--force` explicit.

## What You Do

1. **Run the weight analysis pipeline**: `./abliterlitics.sh weights ./comparisons/<name>/` — orchestrates 11 Docker scripts inside `abliterlitics-forensics:1.0.0`.

2. **The 11 scripts** (execution order):

| # | Script | Output | Scope |
|---|---|---|---|
| 1 | `panel_comparison.py` | `panel_comparison.json` | All models |
| 2 | `edit_vector_analysis.py` | `edit_<variant>.json` | Per variant |
| 3 | `svd_analysis.py` | `svd_<variant>.json` | Per variant |
| 4 | `technique_fingerprint.py` | `fingerprint_<variant>.json` | Per variant |
| 5 | `layer_analysis.py` | `layer_<variant>.json` | Per variant |
| 6 | `technique_correlation.py` | `correlation_<v1>_vs_<v2>.json` | Pairwise |
| 7 | `subspace_alignment.py` | `subspace_<v1>_vs_<v2>.json` | Pairwise |
| 8 | `lowrank_reconstruction.py` | `lowrank_<v1>_vs_<v2>.json` | Pairwise |
| 9 | `expert_analysis.py` | `expert_analysis.json` | MoE only |
| 10 | `stacking_analysis.py` | `stacking.json` | All models |
| 11 | `cross_arch_comparison.py` | `cross_arch.json` | If multiple archs |

3. **Monitor**: check `results/` for output files, log for errors. Scripts auto-skip existing.

4. **Report**: tensors changed per variant (aggressiveness), edit vector norms/directions, SVD effective ranks, technique fingerprints (unique vs shared edits — detect clones per AGENTS.md #22), any failures/warnings.

## Key Technical Details

- **Safetensors accessed lazily** via `safe_open` — never loaded into memory.
- **Architecture-aware**: must handle Qwen3.5 (`language_model` prefix, Mamba hybrid), Qwen3 (standard), GLM (MoE with experts), Gemma4 (E series PLE, 12B Unified encoder-free). 12B Unified variants may have different shard layouts (e.g. 5 shards vs base's 1 shard) — analysis is tensor-level, not file-level, so handles resharding transparently (relevant for clone detection).
- **Shard maps** built by `model_config.py` map tensor names to safetensors files.
- **Results** go to `comparisons/<name>/results/<comparison_name>/weight/`. Each result JSON includes `make_metadata()` fields.

## Monitoring Commands
```bash
./abliterlitics.sh status ./comparisons/<name>/
./abliterlitics.sh db pipeline-status ./comparisons/<name>/      # per-model timing
ls -lt comparisons/<name>/results/lm_eval_logs/ | head
cat comparisons/<name>/results/<name>/weight/panel_comparison.json | jq '.pairwise_changed_counts'
```

`./abliterlitics.sh weights` auto-tracks DB stages and imports results.

## Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| OOM during SVD | Large tensor dims | Auto-handled via chunked SVD |
| "shard not found" | Corrupted safetensors index | Verify model files with `config.json` |
| Expert analysis empty | Non-MoE architecture | Expected for Qwen/Gemma — not an error |
| Cross-arch empty | Only one architecture | Expected — needs multiple archs to compare |
| `./abliterlitics.sh weights &` dies after first phase; docker orphaned | SIGTERM/SIGPIPE from parent shell (AGENTS.md #28b) | Use `runners/weights_sequencer.sh` + `docker run -d` instead of backgrounding |
| `jsonschema.validate ... raise error` inside docker but `validate` passes on host | Schema missing settings keys the runner reads (AGENTS.md #28a) | Add the new settings key to `comparison.schema.json` whenever you add a reader in `_load_comparison.py` |
| Runner re-runs everything despite existing results | `run_weights.sh` skip-check uses wrong filenames/paths (AGENTS.md #28c) | Use the sequencer; or import via `python3 -m src.db import-results` + per-model `stage-complete` |
| `stage-complete` prints "Stage completed" but pipeline-status still shows pending | Stage name must be `weights` (plural) AND the model must already have model_stages rows — the CLI is UPDATE-only, no rows-affected check (AGENTS.md #84) | Verify with a read-back; for late-added variants run `import_comparison` + `init_stages` (idempotent) before marking stages |
| Importer finds 0 weight results but files exist on disk | Empty `results/<comparison>/weight/` dir shadows per-variant fallback (importers.py:484) | `rm -rf results/<comparison>/weight/` and re-run `import-results` |
| Merged LoRA shows `effective_rank_90pct ≈ 20` instead of predicted ≤ rank-r | Sub-BF16-precision merge: `alpha/r` edits below BF16 ULP → diff is quantization dithering (AGENTS.md #35) | Not a bug; report as precision caveat. Fix future merges via FP32 merge or higher `alpha/r` |
| `docker run` script mount → "Is a directory" or passed to python → "ELF null bytes" | Forensics image ENTRYPOINT=[python3]; /tmp file bind-mount creates dir on this host | Use `--entrypoint /bin/bash` AND place the script in the writable `/results` volume (not a separate `/tmp` bind mount) |
| MTP/visual/lm_head status missing from all weight reports | Pipeline scope is LM-only: `model_config.py is_lm_key()` excludes raw keys containing `mtp`/`visual` (qwen35/gemma4 families) AND `lm_head.weight` (top-level key lacks "language_model") — by design; 850/1199 keys analyzed on qwen38-27b (AGENTS.md #54) | Run a sidecar lazy-diff (`results/mtp_visual_diff.py` pattern, qwen38-27b) + a 1-tensor lm_head check; MTP handling is a first-class per-variant differentiator (edited/preserved/absent/restored) |
| `svd_analysis.py` reports `effective_rank_90pct == 20` exactly for many tensors | `torch.svd_lowrank(k=20)` cap — 20 means "≥20 saturated", not exactly 20 (tiny tensors get exact ranks, hence 6500+ values) | Treat 20 as lower bound; use `lowrank_reconstruction` rank-error curve or `--full-svd` spot checks for true rank |
| `cross_arch_comparison.py` exits rc=2 in sequencer; `stacking_analysis.py` never runs | Both are investigation-specific tools with explicit `--panels/--labels` / `--variant-a/b` CLIs, not comparison.json-driven | Expected skip; do NOT chase as a pipeline failure |
| FP16-stored arm shows `changed_tensors=565/850` ("66% of model changed") vs claimed few-dozen edits | FP16 STORAGE noise: subnormal rounding of small BF16 values, mean ~8e-12, max element exactly 2^-25 (AGENTS.md #63) | Classify by mean_abs_diff magnitude, never binary flags; probe untouched tensors for the 2.98e-8 ceiling; claims below ~1e-9 are unfalsifiable |
| Adding a late variant to a completed comparison | Batch per-variant scripts have NO skip logic and `svd_lowrank` is randomized → re-running batch overwrites other variants' results with slightly different numbers | Run per-variant scripts in SINGLE mode (`--base/--variant[-a]/--label[s]/--output`) with the exact batch filename convention; pairwise batch skip-if-exists works — only new pairs computed; panel must re-run batch (deterministic aggregate). Pattern: `results/run_weights_apostate.sh` (qwen38-27b) |
| `subspace_alignment.py` rc=137 (OOM) on a pair, reproducibly, even in a fresh container | Tool stacks per-type flattened delta vectors then QRs them — memory scales with changed-key count on BOTH sides; a noise-inflated set (565 keys) × an 841-key variant ≈ 115GB+ peak | Cover that pair with correlation + lowrank and document the gap; remaining small pairs via single-pair mode (`--base --variant-a --variant-b --labels --output`, filename must match batch convention) |
| Weight scripts die `ModuleNotFoundError: No module named 'src'` in <1s on the gemma4unified image | That image has `ENTRYPOINT []` AND no `PYTHONPATH=/app` (old forensics image sets it) (AGENTS.md #79) | Launch with `--entrypoint /bin/bash -e PYTHONPATH=/app ... image /results/<script>.sh` |
| Weight scripts crash on `t_var - t_base` broadcast error for one variant | Architecture-extending edit: variant extends tensor shapes (e.g. diode +1 neuron, 17408→17409) (AGENTS.md #79) | Fixed: shape guards in all 6 scripts (skip + count). Panel marks such keys inf; sidecar does shared-region + appended-neuron analysis |
| correlation between two 16-bit-stored arms reads ~1.0 ("everything correlated") | F16/FP16 storage noise is DETERMINISTIC and SHARED — same BF16 base → same subnormal flips → bit-identical noise deltas (AGENTS.md #80) | Decompose per-key pairs by edit-norm class (both_real/mixed/both_noise); quote only both_real cosines |
| Subspace pair rc=137 even at a 90-112g cgroup cap | Stacked-vector peak ≈ n_intersection × 89M × 4B × 2: 124-128-vector pairs need ~88-92GB, 129+ (embed-heavy) needs >112g (AGENTS.md #80 corollary) | Cap `--memory=112g` is the safe max here; bigger intersections → document, cover with correlation+lowrank |

**Subspace OOM predictor (refined 2026-09-06, qwen38-27b followup arms)**: retention is the INTERSECTION of changed sets — only keys where BOTH sides changed are appended to `type_vectors_{a,b}` (subspace_alignment.py `if a_changed and b_changed`). Predict OOM from `min(changed_a, changed_b)` when one side is a superset: empirically 565 → OOM, ≤129 → safe (~2.5min/pair, <16GB). Proven on the 10-variant panel: 15/17 new pairs OK; apostate_vs_davidau (~565) and obliteratus_vs_davidau (~841) rc=137 exactly as predicted. Strategy for late arms: skip the batch (it dies mid-run losing remaining pairs), run ALL new pairs single-pair safe-first with predicted-OOM pairs LAST — each failure is isolated and the sequencer continues. Pattern: `results/run_weights_ektome_davidau.sh`.

## What You Do NOT Do
- Do NOT modify `src/kl/kl_divergence.py` — FROZEN
- Do NOT run KL, lm-eval, or HarmBench — separate agents
- Do NOT change comparison.json settings mid-run

(Standard execution environment + self-learning protocol: see AGENTS.md.)
