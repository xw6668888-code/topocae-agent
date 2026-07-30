"""Compare two review-aware residual atlases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def delta_counts(old: dict[str, int], new: dict[str, int]) -> dict[str, int]:
    keys = sorted(set(old) | set(new))
    return {key: int(new.get(key, 0)) - int(old.get(key, 0)) for key in keys}


def compare_summaries(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    old_summary = old.get("summary", {})
    new_summary = new.get("summary", {})
    return {
        "old_recommendation": old.get("recommendation", ""),
        "new_recommendation": new.get("recommendation", ""),
        "macro_delta": delta_counts(old_summary.get("by_macro", {}), new_summary.get("by_macro", {})),
        "review_status_delta": delta_counts(old_summary.get("by_review_status", {}), new_summary.get("by_review_status", {})),
        "miss_count_delta": int(new_summary.get("miss_count", 0)) - int(old_summary.get("miss_count", 0)),
    }


def write_markdown(path: Path, comparison: dict[str, Any]) -> None:
    lines = [
        "# v5.23 vs v5.22 Review-Aware Residual Comparison",
        "",
        f"- old_recommendation: `{comparison['old_recommendation']}`",
        f"- new_recommendation: `{comparison['new_recommendation']}`",
        f"- miss_count_delta: `{comparison['miss_count_delta']}`",
        "",
        "## Macro Delta",
        "",
        "| Macro | Delta |",
        "|-------|-------|",
    ]
    for macro, delta in comparison["macro_delta"].items():
        lines.append(f"| {macro} | {delta} |")
    lines.extend(["", "## Review Status Delta", "", "| Status | Delta |", "|--------|-------|"])
    for status, delta in comparison["review_status_delta"].items():
        lines.append(f"| {status} | {delta} |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two review-aware residual atlases.")
    parser.add_argument("--old-json", type=Path, required=True)
    parser.add_argument("--new-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comparison = compare_summaries(load_json(args.old_json), load_json(args.new_json))
    write_markdown(args.output_md, comparison)
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
