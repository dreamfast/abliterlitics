"""Split an export-review batch into per-item JSON files the reviewer can fully read.

Why: JSON long fields sit on ONE line, and the Read tool truncates lines at
~2,000 chars — so a 72K-char reasoning_text is invisible to the judge, which
then improvises bash/grep workarounds (the 2026-09-05 failure). This script
soft-wraps long text fields to <=1,500-char lines so Read shows the COMPLETE
trace, and writes one file per item so a dispatch only reads what it judges.

Usage: python3 split_review_batch.py <batch.json> <outdir>
"""
import json
import sys
from pathlib import Path


def wrap(text: str, width: int = 1500) -> str:
    if not text:
        return text
    return "\n".join(text[i : i + width] for i in range(0, len(text), width))


def main() -> None:
    batch_file, outdir = sys.argv[1], Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    items = json.loads(Path(batch_file).read_text())
    for n, item in enumerate(items):
        wrapped = dict(item)
        wrapped["reasoning_text"] = wrap(item.get("reasoning_text") or "")
        wrapped["response_text"] = wrap(item.get("response_text") or "")
        (outdir / f"item_{n:02d}.json").write_text(
            json.dumps(wrapped, indent=1, ensure_ascii=False)
        )
        # json.dumps re-escapes the wrap() newlines back to literal \n, so the
        # .json alone still shows ONE truncated physical line in Read. Emit the
        # wrapped long fields as plain-text sidecars (raw newlines survive) and
        # point at them from the JSON.
        (outdir / f"item_{n:02d}_reasoning.txt").write_text(
            "REASONING TEXT (wrapped, full)\n" + (wrapped["reasoning_text"] or "(empty)")
        )
        (outdir / f"item_{n:02d}_response.txt").write_text(
            "RESPONSE TEXT (wrapped, full)\n" + (wrapped["response_text"] or "(empty)")
        )
        wrapped["reasoning_text_sidecar"] = f"item_{n:02d}_reasoning.txt"
        wrapped["response_text_sidecar"] = f"item_{n:02d}_response.txt"
        (outdir / f"item_{n:02d}.json").write_text(
            json.dumps(wrapped, indent=1, ensure_ascii=False)
        )
    print(f"wrote {len(items)} item files to {outdir}")


if __name__ == "__main__":
    main()
