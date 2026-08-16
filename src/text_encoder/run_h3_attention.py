"""Attention-readout battery for MiniMax H3 — what do the AUDIO and VIDEO
streams read from the text conditioning, base vs variant TE? (phase 9)

Encodes prompts with the h3 backend (consumed_hidden, the literal DiT
context), runs paired Euler trajectories on the fl2va DiT with the capture
override, and records per-(step, layer) JSD for the AGGREGATE text readout
plus per-modality (audio-stream vs video-stream) readouts — the conditioning-
level companion to the audio-damage investigation: if the abliterated TE
shifts which words the audio stream latches onto, it shows here even when
WER/MOS stay clean.

Usage (container):
    python3 -m src.text_encoder.run_h3_attention \
        --dit /home/maxx/extra/comfyui-models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
        --base /home/maxx/extra/comfyui-models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
        --variants /home/maxx/extra/comfyui-models/text_encoders/\
qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors \
        --variant-names heretic \
        --suite prompts/te_h3_suite.yaml --n-prompts 2 \
        --results-dir /comp/results/te-h3/text_encoder
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

H3_TOKENIZER_LAYER = "qwen3vl_32b"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H3 attention readout (base vs variant TE, per modality)")
    p.add_argument("--dit", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--variants", required=True, help="Comma-separated variant TE files")
    p.add_argument("--variant-names", default="")
    p.add_argument("--comfyui", default="/comfyui")
    p.add_argument("--suite", default="prompts/te_h3_suite.yaml")
    p.add_argument("--n-prompts", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--latent-thw", default="5,32,32", help="video latent (T,H,W); 32x32 = 256px")
    p.add_argument("--audio-t", type=int, default=80, help="audio latent frames (40 fps)")
    p.add_argument("--layer-sample", type=int, default=1)
    p.add_argument("--results-dir", required=True)
    return p.parse_args()


def _suite_entries(suite_path: str) -> tuple[list[str], list[str], dict[str, str]]:
    from src.text_encoder.suite import load_suite

    prompts, cats = load_suite(suite_path)
    # label map (audio/video suites carry labels; plain te_suite does not)
    import yaml

    raw = yaml.safe_load(Path(suite_path).read_text())
    if isinstance(raw, dict):
        raw = raw.get("prompts", [])
    labels = {str(e["text"]): str(e.get("label", "")) for e in raw if "text" in e}
    return prompts, cats, labels


def main() -> int:
    args = parse_args()
    from src.text_encoder.backends import comfyui_loader
    from src.text_encoder.backends.h3 import H3Backend
    from src.text_encoder.backends.h3_dit import (
        build_layout,
        forward_h3,
        load_h3_dit,
        time_shift_sigma,
    )
    from src.text_encoder.backends.krea2_dit import sigma_schedule_flow

    comfyui_loader.import_comfy(args.comfyui, cpu=(args.device == "cpu"))
    prompts, cats, label_map = _suite_entries(args.suite)
    if 0 < args.n_prompts < len(prompts):
        seen: set[str] = set()
        idxs: list[int] = []
        for i, c in enumerate(cats):
            if c not in seen:
                idxs.append(i)
                seen.add(c)
        prompts = [prompts[i] for i in idxs[: args.n_prompts]]
        cats = [cats[i] for i in idxs[: args.n_prompts]]

    lat_t, lat_h, lat_w = (int(x) for x in args.latent_thw.split(","))

    # --- conditioning: consumed_hidden via the h3 backend (CPU TE) ----------
    def _encode(te_file: str) -> tuple[list[torch.Tensor], list[list[str]]]:
        backend = H3Backend(comfyui_path=args.comfyui)
        backend.load(te_file, device="cpu", dit_file=args.dit)
        arts = backend.encode(prompts, cats)
        raw = next(a for a in arts if a.stage == Stage.RAW_HIDDEN)
        consumed = raw.streams["consumed_hidden"]
        mask = consumed.mask
        if mask is None:
            raise RuntimeError("consumed_hidden stream lacks mask — backend contract broken")
        ctxs: list[torch.Tensor] = []
        for i in range(len(prompts)):
            n = int(mask[i].sum())
            ctxs.append(consumed.activations[i, :n, :].float())
        ids_all = backend.tokenize(prompts)["input_ids"]
        inner = getattr(backend.tokenizer, H3_TOKENIZER_LAYER).tokenizer
        tok_strings = [
            [t.replace("Ġ", " ").replace("Ċ", "\n") for t in inner.convert_ids_to_tokens(ids)] for ids in ids_all
        ]
        backend.close()
        return ctxs, tok_strings

    base_ctxs, token_strings = _encode(args.base)
    log.info("encoded %d base contexts (h3 consumed_hidden)", len(base_ctxs))

    dit = load_h3_dit(args.dit, device=args.device)
    shift_v = float(getattr(dit, "sigma_shift_video", 12.0))
    shift_a = float(getattr(dit, "sigma_shift_audio", 3.0))
    capture = AttentionCapture(device=args.device)
    sigmas_v = sigma_schedule_flow(args.n_steps, shift=shift_v).to(args.device)
    g = torch.Generator(device="cpu").manual_seed(args.seed)

    def _forward_capture(
        xv_: torch.Tensor,
        xa_: torch.Tensor,
        ctx: torch.Tensor,
        sigma_v: float,
        step: int,
        side: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        capture.reset(step, side)
        v_video, v_audio = forward_h3(
            dit,
            xv_,
            xa_,
            ctx.unsqueeze(0),
            sigma_v,
            args.device,
            transformer_options=capture.transformer_options,
            payload=_forward_capture.payload,  # type: ignore[attr-defined]
        )
        _forward_capture.last_records = capture.take_records()  # type: ignore[attr-defined]
        return v_video.float(), v_audio.float()

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
            if ctx_b.shape[0] != ctx_v.shape[0]:
                raise RuntimeError(
                    f"prompt {pi!r}: base/variant context lengths differ "
                    f"({ctx_b.shape[0]} vs {ctx_v.shape[0]}) — capture cannot pair"
                )
            text_len = int(ctx_b.shape[0])
            layout, spans = build_layout(dit, text_len, lat_t, lat_h, lat_w, args.audio_t)
            capture.configure(text_len, spans)
            payload = {"layout": layout, "text_token_tags": torch.ones(text_len, dtype=torch.int)}
            _forward_capture.payload = payload  # type: ignore[attr-defined]

            x0v = torch.randn(1, 24, lat_t, lat_h, lat_w, generator=g).to(args.device).to(torch.bfloat16)
            x0a = torch.randn(1, 32, 2, args.audio_t, generator=g).to(args.device).to(torch.bfloat16)
            xbv, xav = x0v.clone(), x0a.clone()
            xvv, xva = x0v.clone(), x0a.clone()
            step_layer: dict[tuple[int, int], dict[str, float]] = {}
            step_layer_seg: dict[tuple[int, int, str], dict[str, float]] = {}
            recs_b_all: list[dict[str, Any]] = []
            recs_v_all: list[dict[str, Any]] = []
            for i in range(args.n_steps):
                s_v, s_v_next = float(sigmas_v[i]), float(sigmas_v[i + 1])
                s_a = time_shift_sigma(s_v, shift_v, shift_a)
                s_a_next = time_shift_sigma(s_v_next, shift_v, shift_a)
                vb_v, vb_a = _forward_capture(xbv, xav, ctx_b, s_v, i, "base")
                recs_b = list(_forward_capture.last_records)  # type: ignore[attr-defined]
                vv_v, vv_a = _forward_capture(xvv, xva, ctx_v, s_v, i, "variant")
                recs_v = list(_forward_capture.last_records)  # type: ignore[attr-defined]
                xbv = (xbv.float() + vb_v * (s_v_next - s_v)).to(torch.bfloat16)
                xav = (xav.float() + vb_a * (s_a_next - s_a)).to(torch.bfloat16)
                xvv = (xvv.float() + vv_v * (s_v_next - s_v)).to(torch.bfloat16)
                xva = (xva.float() + vv_a * (s_a_next - s_a)).to(torch.bfloat16)
                recs_b_all.extend(recs_b)
                recs_v_all.extend(recs_v)
                if i % args.layer_sample and i != args.n_steps - 1:
                    continue
                for rb, rv in zip(recs_b, recs_v, strict=False):
                    if rb["text_len"] != rv["text_len"]:
                        continue
                    step_layer[(i, rb["call"])] = summarize_readout(rb, rv)
                    for seg in ("audio", "video"):
                        pb = rb["segment_readouts"].get(seg)
                        pv = rv["segment_readouts"].get(seg)
                        if pb is not None and pv is not None:
                            step_layer_seg[(i, rb["call"], seg)] = summarize_readout(
                                {"text_readout": pb}, {"text_readout": pv}
                            )
            base_mean = torch.stack([r["text_readout"] for r in recs_b_all]).mean(dim=0) if recs_b_all else None
            var_mean = torch.stack([r["text_readout"] for r in recs_v_all]).mean(dim=0) if recs_v_all else None
            seg_means: dict[str, dict[str, Any]] = {}
            for seg in ("audio", "video"):
                pb = torch.stack([r["segment_readouts"][seg] for r in recs_b_all if seg in r["segment_readouts"]])
                pv = torch.stack([r["segment_readouts"][seg] for r in recs_v_all if seg in r["segment_readouts"]])
                seg_means[seg] = {
                    "base_mean_readout": [round(float(x), 6) for x in pb.mean(dim=0)] if pb.numel() else [],
                    "variant_mean_readout": [round(float(x), 6) for x in pv.mean(dim=0)] if pv.numel() else [],
                }
            per_prompt.append(
                {
                    "prompt": prompts[pi],
                    "category": cats[pi],
                    "label": label_map.get(prompts[pi], ""),
                    "step_layer": {f"{s}_{ly}": m for (s, ly), m in sorted(step_layer.items())},
                    "step_layer_segment": {f"{s}_{ly}_{sg}": m for (s, ly, sg), m in sorted(step_layer_seg.items())},
                    "base_mean_readout": [round(float(x), 6) for x in base_mean] if base_mean is not None else [],
                    "variant_mean_readout": [round(float(x), 6) for x in var_mean] if var_mean is not None else [],
                    "segment_mean_readouts": seg_means,
                    "tokens": token_strings[pi],
                }
            )
            log.info(
                "%s prompt %d/%d: %d pairs, %d segment pairs",
                vname,
                pi + 1,
                len(prompts),
                len(step_layer),
                len(step_layer_seg),
            )

        layer_acc: dict[int, list[float]] = {}
        seg_acc: dict[str, list[float]] = {}
        for pp in per_prompt:
            for key, m in pp["step_layer"].items():
                ly = int(key.split("_")[1])
                layer_acc.setdefault(ly, []).append(m["jsd"])
            for key, m in pp["step_layer_segment"].items():
                seg = key.split("_")[-1]
                seg_acc.setdefault(seg, []).append(m["jsd"])
        doc = {
            "te_results_version": TE_RESULTS_VERSION,
            "analysis": "attention_readout",
            "backend": "h3",
            "base": args.base,
            "variant": vfile,
            "variant_name": vname,
            "dit": args.dit,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_steps": args.n_steps,
            "seed": args.seed,
            "latent_thw": [lat_t, lat_h, lat_w],
            "audio_t": args.audio_t,
            "layer_sample": args.layer_sample,
            "jsd_per_layer": {str(ly): sum(v) / len(v) for ly, v in sorted(layer_acc.items())},
            "jsd_per_segment": {seg: sum(v) / len(v) for seg, v in sorted(seg_acc.items())},
            "jsd_mean_all": sum(sum(v) for v in layer_acc.values()) / max(sum(len(v) for v in layer_acc.values()), 1),
            "per_prompt": [
                {k: v for k, v in pp.items() if k not in ("step_layer", "step_layer_segment")}
                | {"n_pairs": len(pp["step_layer"]), "n_segment_pairs": len(pp["step_layer_segment"])}
                for pp in per_prompt
            ],
            "has_word_readout": bool(token_strings),
            "has_segment_readout": True,
        }
        out_path = results_dir / f"tebench_attention_{vname}.json"
        out_path.write_text(json.dumps(doc, indent=2))
        log.info(
            "%s: mean JSD %.5f (audio %.5f / video %.5f) -> %s",
            vname,
            doc["jsd_mean_all"],
            doc["jsd_per_segment"].get("audio", float("nan")),
            doc["jsd_per_segment"].get("video", float("nan")),
            out_path,
        )

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
