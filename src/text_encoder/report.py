"""Report schema + markdown/json rendering for text-encoder analysis.

Produces blog-ready output with MSE-primary divergence, CKA-secondary
similarity, and per-category breakdown.

Terminology (journal #117/#118): "divergence" is the value-neutral
measurement reported here; "quantization error" and "abliteration shift"
type it by intent; "damage" is reserved for collateral unintended
divergence and never appears as a column label. JSON field names
(damage_quant_units, damage_semantic_units) are schema-stable and
intentionally unchanged.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.text_encoder import TE_RESULTS_VERSION
from src.text_encoder.artifact import ConditioningArtifact
from src.text_encoder.compare import VariantReport

log = logging.getLogger(__name__)


@dataclass
class Report:
    """Full comparison report for one base vs one or more variants."""

    base_id: str
    backend: str
    reference_floors: dict[str, str]  # {"numerical": "bf16_vs_fp16", "quant": "fakequant_fp8"}
    variants: list[VariantReport]
    prompt_set_hash: str
    timestamp: str = ""
    env: dict[str, str] = field(default_factory=dict)  # torch/transformers versions, seed, device

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_json(self) -> dict[str, Any]:
        """Serialize to a JSON-serializable dict (for DB import + reproducibility)."""
        return {
            "te_results_version": TE_RESULTS_VERSION,
            "base_id": self.base_id,
            "backend": self.backend,
            "reference_floors": self.reference_floors,
            "prompt_set_hash": self.prompt_set_hash,
            "timestamp": self.timestamp,
            "env": self.env,
            "variants": [
                {
                    "variant_id": v.variant_id,
                    "dtype": v.dtype,
                    **({"gguf": v.gguf} if v.gguf else {}),
                    "streams": [
                        {
                            "stream": sr.stream,
                            "stage": sr.stage,
                            "n_prompts": sr.n_prompts,
                            "metrics": {
                                "cka_linear": sr.cka_linear.value,
                                "cka_rbf": sr.cka_rbf.value,
                                "cos_all": sr.cos_all.value,
                                "cos_all_ci": [sr.cos_all.ci_low, sr.cos_all.ci_high]
                                if sr.cos_all.ci_low is not None
                                else None,
                                "cos_last": sr.cos_last.value,
                                "mse": sr.mse.value,
                                "mse_ci": [sr.mse.ci_low, sr.mse.ci_high] if sr.mse.ci_low is not None else None,
                                "l2": sr.l2.value,
                                "rank_corr": sr.rank_corr.value,
                            },
                            "damage_quant_units": (sr.damage_quant_units.value if sr.damage_quant_units else None),
                            "damage_quant_ci": [
                                sr.damage_quant_units.ci_low,
                                sr.damage_quant_units.ci_high,
                            ]
                            if sr.damage_quant_units and sr.damage_quant_units.ci_low is not None
                            else None,
                            "semantic_ceiling_mse": sr.semantic_ceiling_mse,
                            "damage_semantic_units": (
                                sr.damage_semantic_units.value if sr.damage_semantic_units else None
                            ),
                            "per_category": sr.per_category,
                        }
                        for sr in v.streams
                    ],
                }
                for v in self.variants
            ],
        }

    def to_markdown(self) -> str:
        """Render as a markdown table — blog-ready divergence report."""
        lines: list[str] = []
        lines.append("# Text-Encoder Conditioning Fidelity Report")
        lines.append("")
        lines.append(f"- **Base:** `{self.base_id}`")
        lines.append(f"- **Backend:** `{self.backend}`")
        lines.append(f"- **Timestamp:** {self.timestamp}")
        lines.append(f"- **Prompt set hash:** `{self.prompt_set_hash}`")
        if self.reference_floors:
            lines.append(f"- **Reference floors:** {self.reference_floors}")
        lines.append("")

        for v in self.variants:
            lines.append(f"## Variant: `{v.variant_id}` ({v.dtype})")
            lines.append("")
            lines.append(
                "| Stream | Stage | CKA (linear) | CKA (RBF) | Cos (all) | "
                "MSE | L2 | Rank Corr | Divergence (quant units) | Div (semantic) |"
            )
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for sr in v.streams:
                if sr.damage_quant_units is not None:
                    dmg = f"{sr.damage_quant_units.value:.4f}"
                    if sr.damage_quant_units.ci_low is not None and sr.damage_quant_units.ci_high is not None:
                        dmg += f" [{sr.damage_quant_units.ci_low:.3f}, {sr.damage_quant_units.ci_high:.3f}]"
                else:
                    dmg = "N/A"
                sem = f"{sr.damage_semantic_units.value:.2%}" if sr.damage_semantic_units else "N/A"
                lines.append(
                    f"| {sr.stream} | {sr.stage} | {sr.cka_linear.value:.6f} | "
                    f"{sr.cka_rbf.value:.6f} | {sr.cos_all.value:.6f} | "
                    f"{sr.mse.value:.6e} | {sr.l2.value:.6e} | "
                    f"{sr.rank_corr.value:.6f} | {dmg} | {sem} |"
                )

            # Per-category breakdown for streams that have it
            for sr in v.streams:
                if sr.per_category:
                    lines.append("")
                    lines.append(f"### Per-category: stream `{sr.stream}` ({sr.stage})")
                    lines.append("")
                    cats = sorted(sr.per_category.keys())
                    lines.append("| Category | CKA (linear) | MSE | Cos (all) |")
                    lines.append("|---|---:|---:|---:|")
                    for cat in cats:
                        m = sr.per_category[cat]
                        lines.append(
                            f"| {cat} | {m.get('cka_linear', 0):.6f} | "
                            f"{m.get('mse', 0):.6e} | {m.get('cos_all', 0):.6f} |"
                        )
            lines.append("")

        return "\n".join(lines)

    def save(self, output_dir: Path | str, variant_name: str = "") -> list[Path]:
        """Save report as JSON + markdown to the given directory.

        Returns list of written file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        json_path = output_dir / f"te_{variant_name or 'report'}.json"
        json_path.write_text(json.dumps(self.to_json(), indent=2))
        written.append(json_path)

        md_path = output_dir / f"te_{variant_name or 'report'}.md"
        md_path.write_text(self.to_markdown())
        written.append(md_path)

        log.info("Saved TE report to %s", output_dir)
        return written


def make_report(
    base: ConditioningArtifact,
    variant_reports: list[VariantReport],
    reference_floors: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
) -> Report:
    """Build a Report from artifacts + variant reports."""
    return Report(
        base_id=base.ckpt_id,
        backend=base.backend,
        reference_floors=reference_floors or {},
        variants=variant_reports,
        prompt_set_hash=base.prompt_set_hash,
        env=env or {},
    )
