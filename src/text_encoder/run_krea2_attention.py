"""Attention-readout battery for Krea2 — which prompt WORDS does the DiT read,
base vs variant TE? (phase 9; mirrors run_attention.py for z-image)

Encodes each prompt with the krea2 backend (12-tap stack, the literal DiT
context), runs paired Euler denoise trajectories with the capture override,
and records per-(step, layer) JSD of the text-readout distributions plus the
mean per-token readout per side (word-level diff in the report layer).

Krea2 is a text-rendering model — this is the instrument for "does the
abliterated TE shift which words get emphasized when the model draws text
into images".

Usage (container):
    python3 -m src.text_encoder.run_krea2_attention \
        --dit /home/maxx/extra/comfyui-models/diffusion_models/redcraft22INT8INT4_2Krea2Edition.safetensors \
        --base /comp/models/te/qwen3vl_4b_base.safetensors \
        --variants /home/maxx/extra/comfyui-models/text_encoders/qwen3-vl-4b-heretic.safetensors \
        --variant-names heretic \
        --suite prompts/te_krea2_prompts.yaml --n-prompts 2 \
        --results-dir /comp/results/te-krea2/text_encoder
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.text_encoder import TE_RESULTS_VERSION
from src.text_encoder.artifact import Stage
from src.text_encoder.attention_capture import AttentionCapture
from src.text_encoder.attention_metrics import summarize_readout

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Krea2 attention readout (base vs variant TE)")
    p.add_argument("--dit", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--variants", required=True, help="Comma-separated variant TE files")
    p.add_argument("--variant-names", default="")
    p.add_argument("--comfyui", default="/comfyui")
    p.add_argument("--suite", default="prompts/te_krea2_prompts.yaml")
    p.add_argument("--n-prompts", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-steps", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--latent-hw", type=int, default=32, help="latent spatial size (32 = 256px)")
    p.add_argument("--layer-sample", type=int, default=1, help="capture every Nth step")
    p.add_argument("--results-dir", required=True)
    return p.parse_args()


def _suite_entries(suite_path: str) -> tuple[list[str], list[str], list[str]]:
    import yaml

    raw = yaml.safe_load(Path(suite_path).read_text())
    if isinstance(raw, dict):
        raw = raw.get("prompts", [])
    texts = [str(p["text"]) for p in raw]
    cats = [str(p.get("category", "x")) for p in raw]
    labels = [str(p.get("label", "")) for p in raw]
    return texts, cats, labels


def main() -> int:
    args = parse_args()
    from src.text_encoder.backends import comfyui_loader
    from src.text_encoder.backends.krea2 import Krea2Backend
    from src.text_encoder.backends.krea2_dit import _sigma_schedule, forward_krea2, load_krea2_dit

    comfyui_loader.import_comfy(args.comfyui, cpu=(args.device == "cpu"))
    prompts, cats, labels = _suite_entries(args.suite)
    if 0 < args.n_prompts < len(prompts):
        seen: set[str] = set()
        idxs: list[int] = []
        for i, c in enumerate(cats):
            if c not in seen:
                idxs.append(i)
                seen.add(c)
        prompts = [prompts[i] for i in idxs[: args.n_prompts]]
        cats = [cats[i] for i in idxs[: args.n_prompts]]
        labels = [labels[i] for i in idxs[: args.n_prompts]]

    # --- conditioning: 12-tap stacks via the krea2 backend (CPU TE) --------
    def _encode(te_file: str) -> tuple[list[torch.Tensor], list[list[str]]]:
        backend = Krea2Backend(comfyui_path=args.comfyui)
        backend.load(te_file, device="cpu", dit_file=args.dit)
        arts = backend.encode(prompts, cats)
        raw = next(a for a in arts if a.stage == Stage.RAW_HIDDEN)
        stack = raw.streams["tap_stack"]
        mask = stack.mask
        if mask is None:
            raise RuntimeError("tap_stack stream lacks mask — backend contract broken")
        ctxs: list[torch.Tensor] = []
        for i in range(len(prompts)):
            n = int(mask[i].sum())
            ctxs.append(stack.activations[i, :n, :].float())
        ids_all = backend.tokenize(prompts)["input_ids"]
        tok_strings: list[list[str]] = []
        inner = backend.tokenizer.qwen3vl_4b.tokenizer
        for ids in ids_all:
            toks = inner.convert_ids_to_tokens(ids)
            # HF BPE markers -> whitespace (group_words expects decoded form)
            toks = [t.replace("Ġ", " ").replace("Ċ", "\n") for t in toks]
            # the tap stack is post-template-strip: drop the stripped prefix
            # tokens (ids before the user text) so readout aligns with rows
            tok_strings.append(_strip_prefix_tokens(toks, ids))
        backend.close()
        return ctxs, tok_strings

    def _strip_prefix_tokens(toks: list[str], ids: list[int]) -> list[str]:
        # krea2 encode strips the system+user template prefix (2nd im_start,
        # 'user', '\n' + 3). Mirror: find index after the second im_start
        # triplet.
        im_start = 151644
        starts = [i for i, t in enumerate(ids) if t == im_start]
        cut = starts[1] + 3 if len(starts) >= 2 and len(starts) > 1 else 0
        return toks[cut:] if cut < len(toks) else toks

    base_ctxs, token_strings = _encode(args.base)
    log.info("encoded %d base contexts (krea2 12-tap)", len(base_ctxs))

    dit = load_krea2_dit(args.dit, device=args.device)
    capture = AttentionCapture(device=args.device)
    sigmas = _sigma_schedule(args.n_steps).to(args.device)
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    img_tokens = (args.latent_hw // 2) ** 2  # patch 2

    def _forward_capture(x: torch.Tensor, ctx: torch.Tensor, sigma: float, step: int, side: str) -> torch.Tensor:
        capture.reset(step, side)
        out = forward_krea2(
            dit, x, ctx.unsqueeze(0), sigma, args.device, transformer_options=capture.transformer_options
        )
        _forward_capture.last_records = capture.take_records()  # type: ignore[attr-defined]
        return out.float()

    variant_files = [f for f in args.variants.split(",") if f.strip()]
    names = (
        [n.strip() for n in args.variant_names.split(",")]
        if args.variant_names
        else [Path(f).stem for f in variant_files]
    )
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for vfile, vname in zip(variant_files, names, strict=True):
        var_ctxs, _tok_v = _encode(vfile)
        per_prompt: list[dict[str, Any]] = []
        for pi, (ctx_b, ctx_v) in enumerate(zip(base_ctxs, var_ctxs, strict=True)):
            text_len = int(ctx_b.shape[0])
            capture.configure(text_len, {"img": (text_len, text_len + img_tokens)})
            x0 = torch.randn(1, 16, args.latent_hw, args.latent_hw, generator=g).to(args.device).to(torch.bfloat16)
            xb, xv = x0.clone(), x0.clone()
            step_layer: dict[tuple[int, int], dict[str, float]] = {}
            recs_b_all: list[dict[str, Any]] = []
            recs_v_all: list[dict[str, Any]] = []
            for i in range(args.n_steps):
                s, s_next = float(sigmas[i]), float(sigmas[i + 1])
                vb = _forward_capture(xb, ctx_b, s, i, "base")
                recs_b = list(_forward_capture.last_records)  # type: ignore[attr-defined]
                vv = _forward_capture(xv, ctx_v, s, i, "variant")
                recs_v = list(_forward_capture.last_records)  # type: ignore[attr-defined]
                dt = s_next - s
                xb = (xb.float() + vb * dt).to(torch.bfloat16)
                xv = (xv.float() + vv * dt).to(torch.bfloat16)
                recs_b_all.extend(recs_b)
                recs_v_all.extend(recs_v)
                if i % args.layer_sample and i != args.n_steps - 1:
                    continue
                for rb, rv in zip(recs_b, recs_v, strict=False):
                    if rb["text_len"] != rv["text_len"]:
                        continue
                    step_layer[(i, rb["call"])] = summarize_readout(rb, rv)
            base_mean = torch.stack([r["text_readout"] for r in recs_b_all]).mean(dim=0) if recs_b_all else None
            var_mean = torch.stack([r["text_readout"] for r in recs_v_all]).mean(dim=0) if recs_v_all else None
            per_prompt.append(
                {
                    "prompt": prompts[pi],
                    "category": cats[pi],
                    "label": labels[pi],
                    "step_layer": {f"{s}_{ly}": m for (s, ly), m in sorted(step_layer.items())},
                    "base_mean_readout": [round(float(x), 6) for x in base_mean] if base_mean is not None else [],
                    "variant_mean_readout": [round(float(x), 6) for x in var_mean] if var_mean is not None else [],
                    "tokens": token_strings[pi],
                }
            )
            log.info("%s prompt %d/%d: %d (step,layer) pairs", vname, pi + 1, len(prompts), len(step_layer))

        layer_acc: dict[int, list[float]] = {}
        for pp in per_prompt:
            for key, m in pp["step_layer"].items():
                ly = int(key.split("_")[1])
                layer_acc.setdefault(ly, []).append(m["jsd"])
        doc = {
            "te_results_version": TE_RESULTS_VERSION,
            "analysis": "attention_readout",
            "backend": "krea2",
            "base": args.base,
            "variant": vfile,
            "variant_name": vname,
            "dit": args.dit,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_steps": args.n_steps,
            "seed": args.seed,
            "latent_hw": args.latent_hw,
            "layer_sample": args.layer_sample,
            "jsd_per_layer": {str(ly): sum(v) / len(v) for ly, v in sorted(layer_acc.items())},
            "jsd_mean_all": sum(sum(v) for v in layer_acc.values()) / max(sum(len(v) for v in layer_acc.values()), 1),
            "per_prompt": [
                {k: v for k, v in pp.items() if k != "step_layer"} | {"n_pairs": len(pp["step_layer"])}
                for pp in per_prompt
            ],
            "has_word_readout": bool(token_strings),
        }
        out_path = results_dir / f"tebench_attention_{vname}.json"
        out_path.write_text(json.dumps(doc, indent=2))
        log.info("%s: mean JSD %.5f -> %s", vname, doc["jsd_mean_all"], out_path)

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
