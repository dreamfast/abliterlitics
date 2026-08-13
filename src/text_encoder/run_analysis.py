#!/usr/bin/env python3
"""
Text-encoder conditioning fidelity analysis — main entry point.

Loads base + variant text encoders, captures conditioning at multiple
pipeline stages, computes CKA/MSE/damage metrics, and writes report.

Usage:
    python3 run_analysis.py --base /models/base --variant /models/heretic \
        --backend ltx2 --prompts /prompts/te_suite.yaml \
        --results-dir /results --cache-dir /te_cache
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Text-encoder conditioning fidelity analysis")
    p.add_argument("--base", required=True, help="Path to base text encoder checkpoint")
    p.add_argument("--variant", required=True, help="Path to variant text encoder checkpoint")
    p.add_argument("--backend", required=True, choices=["ltx2", "h3", "ltx25"], help="Backend name")
    p.add_argument("--variant-name", default="variant", help="Variant display name")
    p.add_argument("--prompts", default="/prompts/te_suite.yaml", help="Path to prompt suite YAML")
    p.add_argument("--results-dir", default="/results", help="Output directory for results")
    p.add_argument("--cache-dir", default="/te_cache", help="Activation cache directory")
    p.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size for encoding")
    p.add_argument("--max-length", type=int, default=1024, help="Max sequence length")
    p.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples for CIs")
    p.add_argument("--dtype", default="bf16", help="Data type for model loading")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from src.text_encoder.backends import get_backend
    from src.text_encoder.suite import load_suite

    # Load prompt suite
    prompts, categories = load_suite(args.prompts)
    log.info("Loaded %d prompts (%d categories)", len(prompts), len(set(categories)))

    # Get backend
    backend = get_backend(args.backend)
    log.info("Using backend: %s", backend.name)

    # This is where the real work happens — model loading, encoding, comparing.
    # The full pipeline is implemented in T2.3-T2.4 (LTX2Backend) and T4.2 (H3Backend).
    # For now, this script is the wiring point — backends will be plugged in as they're completed.
    log.warning("Backend %s not yet fully implemented — this is the wiring point", args.backend)
    log.info("Would: load base=%s, variant=%s, device=%s", args.base, args.variant, args.device)

    # Write a placeholder so the runner doesn't fail
    import json
    from datetime import datetime, timezone

    from src.text_encoder import TE_RESULTS_VERSION

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "te_results_version": TE_RESULTS_VERSION,
        "backend": args.backend,
        "base_id": args.base,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "not_implemented",
        "message": f"Backend {args.backend} pending implementation (T2.3/T4.2)",
        "variant_id": args.variant,
    }

    output_path = results_dir / f"te_{args.variant_name}.json"
    output_path.write_text(json.dumps(output, indent=2))
    log.info("Wrote placeholder: %s", output_path)

    backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
