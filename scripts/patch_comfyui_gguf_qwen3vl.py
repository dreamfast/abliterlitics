"""Patch ComfyUI-GGUF loader.py: qwen3vl mmproj sidecar merge + key rename.

Context (te-krea2, 2026-08-16): llama.cpp text-encoder GGUFs of Qwen3-VL-4B
(arch ``qwen3vl``) never worked for krea2 conditioning. Two coupled gaps:

1. ComfyUI-GGUF ``gguf_clip_loader`` lists ``qwen3vl`` in its arch set but
   only loads the mmproj vision-tower sidecar for ``qwen2vl`` — a text-only
   VL GGUF therefore carries no ``visual`` keys.
2. Core comfy ``detect_te_model`` classifies VL by the vision tower's
   deepstack keys; without them the file is indistinguishable from plain
   Qwen3-4B (identical 36x2560 LLM stack) and routes to ``z_image.te`` —
   single-layer 2560 conditioning, which the krea2 DiT rejects loudly
   ("expects 12x2560=30720 ... got 2560").

Fix: for ``arch == "qwen3vl"``, load the mmproj sidecar (llama.cpp naming:
``*mmproj*<stripped-main-name>*.gguf`` in the same dir) and rename its 315
llama.cpp keys to comfy's layout so detection routes to QWEN3VL_4B and the
krea2 branch (CLIPType.KREA2) builds the 12-tap Krea2TEModel.

Key map (verified 315/315 against the Qwen3-VL-4B safetensors):
    v.position_embd.weight            -> model.visual.pos_embed.weight
    v.deepstack.{5,11,17}.{fc1,fc2,norm}.* -> model.visual.deepstack_merger_list.{0,1,2}.{linear_fc1,linear_fc2,norm}.*
    visual.blocks.N.attn.attn_qkv.*    -> model.visual.blocks.N.attn.qkv.*
    visual.blocks.N.mlp.up_proj.*      -> model.visual.blocks.N.mlp.linear_fc1.*
    visual.blocks.N.mlp.down_proj.*    -> model.visual.blocks.N.mlp.linear_fc2.*
    visual.merger.ln_q.*               -> model.visual.merger.norm.*
    visual.merger.mlp.{0,1}.*          -> model.visual.merger.{linear_fc1,linear_fc2}.*
    (all other visual.*)               -> model.visual.* (prefix only)

Usage (run INSIDE the comfyui container, then restart it):
    docker cp scripts/patch_comfyui_gguf_qwen3vl.py comfyui-5090:/tmp/patch.py
    docker exec comfyui-5090 python /tmp/patch.py
    docker restart comfyui-5090

Idempotent (no-op when already applied); keeps ``loader.py.abliterlitics.bak``.
Note: both 5090/4090 containers mount the same checkout — one patch serves both.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

log = logging.getLogger(__name__)

PATH = Path("/opt/comfyui/custom_nodes/ComfyUI-GGUF/loader.py")

HELPER = """def qwen3vl_mmproj_rename(vsd):
    # llama.cpp mmproj key layout -> comfy model.visual.* layout
    # (verified against Qwen3-VL-4B safetensors: 315/315 keys map 1:1)
    deepstack_layers = [5, 11, 17]
    out = {}
    for k, v in vsd.items():
        if k == "v.position_embd.weight":
            nk = "model.visual.pos_embed.weight"
        elif k.startswith("v.deepstack."):
            parts = k.split(".")
            layer, sub, param = int(parts[2]), parts[3], parts[4]
            sub = {"fc1": "linear_fc1", "fc2": "linear_fc2"}.get(sub, sub)
            nk = f"model.visual.deepstack_merger_list.{deepstack_layers.index(layer)}.{sub}.{param}"
        elif k.startswith("visual.merger.mlp."):
            parts = k.split(".")
            fc = "linear_fc1" if parts[3] == "0" else "linear_fc2"
            nk = f"model.visual.merger.{fc}.{parts[4]}"
        elif k.startswith("visual.merger.ln_q."):
            nk = "model.visual.merger.norm." + k.split(".")[-1]
        elif k.startswith("visual."):
            nk = "model." + k
            nk = nk.replace(".mlp.up_proj.", ".mlp.linear_fc1.")
            nk = nk.replace(".mlp.down_proj.", ".mlp.linear_fc2.")
            nk = nk.replace(".attn.attn_qkv.", ".attn.qkv.")
        else:
            nk = k
        out[nk] = v
    return out


"""

ANCHOR_OLD = """        if arch == "qwen2vl":
            vsd = gguf_mmproj_loader(path)
            sd.update(vsd)
"""

ANCHOR_NEW = """        if arch == "qwen2vl":
            vsd = gguf_mmproj_loader(path)
            sd.update(vsd)
        elif arch == "qwen3vl":
            # krea2 conditioning needs the full VL model: visual keys drive
            # detect_te_model -> QWEN3VL_4B routing. Merge the mmproj vision
            # tower and rename llama.cpp keys to comfy's layout.
            vsd = gguf_mmproj_loader(path)
            sd.update(qwen3vl_mmproj_rename(vsd))
"""


def main() -> None:
    src = PATH.read_text()
    if "qwen3vl_mmproj_rename" in src:
        log.info("already applied, nothing to do")
        return
    if ANCHOR_OLD not in src:
        log.error("qwen2vl anchor not found — upstream changed? refusing to patch")
        sys.exit(1)
    if "def gguf_clip_loader(path):" not in src:
        log.error("gguf_clip_loader def not found — refusing to patch")
        sys.exit(1)
    shutil.copyfile(PATH, str(PATH) + ".abliterlitics.bak")
    patched = src.replace("def gguf_clip_loader(path):", HELPER + "def gguf_clip_loader(path):", 1)
    patched = patched.replace(ANCHOR_OLD, ANCHOR_NEW, 1)
    PATH.write_text(patched)
    log.info("applied (backup at loader.py.abliterlitics.bak)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    main()
