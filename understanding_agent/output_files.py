"""Readable rollout outputs, independent of report-schema validation."""
import json
from pathlib import Path


def save_readable_output(state: dict, path: Path) -> None:
    raw = state.get("raw_output")
    if raw is None:
        messages = state.get("messages") or []
        last = messages[-1] if messages else None
        raw = last.get("content", "") if isinstance(last, dict) else getattr(last, "content", "")
    if isinstance(raw, str):
        # Pretty-print JSON even if it violates the report schema. Other text is
        # preserved verbatim; the final-state JSON retains the exact raw string.
        try:
            text = json.dumps(json.loads(raw), ensure_ascii=False, indent=2)
        except ValueError:
            text = raw
    elif isinstance(raw, list):
        text = "\n\n".join(
            block if isinstance(block, str) else
            block["text"] if isinstance(block, dict) and isinstance(block.get("text"), str)
            else json.dumps(block, ensure_ascii=False, indent=2, default=str)
            for block in raw
        )
    else:
        text = json.dumps(raw, ensure_ascii=False, indent=2, default=str)
    path.write_text(text, encoding="utf-8")
