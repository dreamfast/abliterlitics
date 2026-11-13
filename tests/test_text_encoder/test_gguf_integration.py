"""Tests for GGUF seam wiring: artifact labeling + JSON data path.

Covers the GGUFMeta -> ConditioningArtifact.meta -> VariantReport.gguf ->
Report.to_json -> generate_report.load_results threading (reviewer M1/M2).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.text_encoder.artifact import Pooling, Stage, Stream, StreamSpec
from src.text_encoder.backends.gguf_loader import GGUFMeta
from src.text_encoder.backends.z_image import ZImageBackend
from src.text_encoder.compare import StreamReport, VariantReport
from src.text_encoder.report import Report


def _meta() -> GGUFMeta:
    return GGUFMeta(
        quant_label="q4_k_m",
        bpw=4.97,
        qtype_histogram={"Q4_K": 216, "Q6_K": 37, "F32": 145},
        arch="qwen3",
        n_tensors=398,
        n_params=4_022_468_096,
    )


def _stream() -> Stream:
    spec = StreamSpec(
        name="s", source_encoder="qwen3", layers=(0,), pooling=Pooling.MEAN, stage=Stage.RAW_HIDDEN, dim=4
    )
    return Stream(name="s", spec=spec, activations=torch.zeros(2, 4))


class TestArtifactLabeling:
    def test_gguf_label_takes_precedence(self) -> None:
        b = ZImageBackend(comfyui_path="/nonexistent")
        b._gguf_meta = _meta()
        art = b._make_artifact(Stage.RAW_HIDDEN, {"s": _stream()}, ["a", "b"], ["innocent", "innocent"])
        assert art.dtype == "gguf_q4_k_m"
        assert art.meta["gguf"]["bpw"] == 4.97
        assert art.meta["gguf"]["quant_label"] == "q4_k_m"

    def test_no_gguf_stays_comfy_bf16(self) -> None:
        b = ZImageBackend(comfyui_path="/nonexistent")
        art = b._make_artifact(Stage.RAW_HIDDEN, {"s": _stream()}, ["a", "b"], ["innocent", "innocent"])
        assert art.dtype == "comfy_bf16"
        assert "gguf" not in art.meta

    def test_fake_quant_beats_comfy_and_gguf(self) -> None:
        # unified precedence (shared dtype_label helper): fakequant describes
        # this run's numerics and outranks storage provenance; gguf provenance
        # stays in meta["gguf"] either way. Before the unification z_image
        # ranked gguf first here while krea2 ranked fakequant first.
        b = ZImageBackend(comfyui_path="/nonexistent")
        b._te_fake_quant = "int8"
        art = b._make_artifact(Stage.RAW_HIDDEN, {"s": _stream()}, ["a", "b"], ["innocent", "innocent"])
        assert art.dtype == "fakequant_int8"
        b._gguf_meta = _meta()
        art2 = b._make_artifact(Stage.RAW_HIDDEN, {"s": _stream()}, ["a", "b"], ["innocent", "innocent"])
        assert art2.dtype == "fakequant_int8"
        assert art2.meta["gguf"]["quant_label"] == "q4_k_m"  # provenance preserved


class TestJsonThreading:
    def _report(self, gguf: dict | None) -> Report:
        vr = VariantReport(
            variant_id="/models/x.gguf",
            dtype="gguf_q4_k_m" if gguf else "comfy_bf16",
            streams=[
                StreamReport(
                    stream="refined_context",
                    stage="CONNECTOR_OUT",
                    n_prompts=2,
                    cka_linear=_m(1.0),
                    cka_rbf=_m(1.0),
                    cos_all=_m(1.0),
                    cos_last=_m(1.0),
                    mse=_m(0.0),
                    l2=_m(0.0),
                    rank_corr=_m(1.0),
                )
            ],
            gguf=gguf,
        )
        return Report(
            base_id="/models/base.safetensors",
            backend="ZImageBackend",
            reference_floors={},
            variants=[vr],
            prompt_set_hash="abc",
        )

    def test_to_json_emits_gguf_field_only_when_present(self) -> None:
        with_gguf = self._report(_meta().to_dict()).to_json()
        assert with_gguf["variants"][0]["gguf"]["quant_label"] == "q4_k_m"
        without = self._report(None).to_json()
        assert "gguf" not in without["variants"][0]

    def test_load_results_carries_gguf(self, tmp_path: Path) -> None:
        data = self._report(_meta().to_dict()).to_json()
        (tmp_path / "te_base_gguf_q4_k_m.json").write_text(json.dumps(data))
        from src.text_encoder.generate_report import load_results

        variants = load_results(tmp_path)
        entry = next(v for v in variants if v["name"] == "base_gguf_q4_k_m")
        assert entry["gguf"]["bpw"] == 4.97
        assert entry["dtype"] == "gguf_q4_k_m"


def _m(value: float):  # type: ignore[no-untyped-def] # noqa: ANN202
    from src.text_encoder.compare import MetricResult

    return MetricResult(value=value)
