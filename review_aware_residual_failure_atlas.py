"""v5.22 Review-Aware Residual Failure Atlas.

Join v5.21 proposal-miss data with v5.20 user review decisions to produce
a review-aware residual atlas. Separates semantic-policy-accepted rows
from confirmed strict and engineering failures. Guides the next code step.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


MACRO_ALIASES = {
    "through_hole": "through_hole_round",
    "blind_hole": "blind_hole_round",
    "stepped_hole": "stepped_hole_round",
    "cylindrical_boss": "cylinder_boss",
    "rectangle_boss": "rectangular_boss",
}

AREA_TOLERANCE = 0.02


def normalize_macro(value: str) -> str:
    token = str(value or "").strip()
    return MACRO_ALIASES.get(token, token)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read_review_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def area_close(left: Any, right: Any, tolerance: float = AREA_TOLERANCE) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def review_row_matches_macro_and_area(miss: dict[str, Any], review: dict[str, str]) -> bool:
    return (
        str(miss.get("sample", "")) == str(review.get("sample", ""))
        and normalize_macro(str(miss.get("macro", ""))) == normalize_macro(str(review.get("truth_macro", "")))
        and area_close(miss.get("area", 0.0), review.get("truth_area", 0.0))
    )


def review_row_matches_area_only(miss: dict[str, Any], review: dict[str, str]) -> bool:
    return (
        str(miss.get("sample", "")) == str(review.get("sample", ""))
        and area_close(miss.get("area", 0.0), review.get("truth_area", 0.0))
    )


def find_matching_review(miss: dict[str, Any], review_rows: list[dict[str, str]]) -> dict[str, str] | None:
    for review in review_rows:
        decision = str(review.get("user_decision", "")).strip()
        if decision in {"accept_family_match", "keep_strict"} and review_row_matches_macro_and_area(miss, review):
            return review
    for review in review_rows:
        decision = str(review.get("user_decision", "")).strip()
        if decision == "truth_label_should_change" and review_row_matches_area_only(miss, review):
            return review
    return None


def classify_miss_with_review(miss: dict[str, Any], review_rows: list[dict[str, str]]) -> dict[str, Any]:
    row = dict(miss)
    review = find_matching_review(miss, review_rows)
    if review is not None:
        decision = str(review.get("user_decision", "")).strip()
        row["user_decision"] = decision
        row["user_note"] = str(review.get("user_note", ""))
        row["review_row_id"] = str(review.get("row_id", ""))
        if decision == "accept_family_match":
            row["review_status"] = "semantic_policy_accepted_auxiliary"
        elif decision == "keep_strict":
            row["review_status"] = "confirmed_strict_cross_macro"
        elif decision == "truth_label_should_change":
            row["review_status"] = "truth_corrected_strict_failure"
        else:
            row["review_status"] = "unreviewed_cross_macro"
        return row
    row["user_decision"] = ""
    row["user_note"] = ""
    row["review_row_id"] = ""
    if str(miss.get("miss_cause", "")) == "cross_macro_area_compatible":
        row["review_status"] = "unreviewed_cross_macro"
    else:
        row["review_status"] = "non_cross_macro_engineering_failure"
    return row


def summarize_review_aware_misses(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "miss_count": len(rows),
        "by_review_status": dict(Counter(str(row.get("review_status", "")) for row in rows).most_common()),
        "by_macro": dict(Counter(str(row.get("macro", "")) for row in rows).most_common()),
        "by_miss_cause": dict(Counter(str(row.get("miss_cause", "")) for row in rows).most_common()),
        "by_status_macro": dict(Counter(f"{row.get('review_status', '')}|{row.get('macro', '')}" for row in rows).most_common()),
        "by_part": dict(Counter(str(row.get("sample", "")) for row in rows).most_common()),
    }


def choose_next_action(summary: dict[str, Any]) -> tuple[str, str]:
    status = summary.get("by_review_status", {})
    unreviewed = int(status.get("unreviewed_cross_macro", 0))
    if unreviewed > 0:
        return "user_review_new_cross_macro_rows", "There are unreviewed_cross_macro rows after joining v5.20 decisions; ask the user to review only those rows."
    strict_residual = (
        int(status.get("confirmed_strict_cross_macro", 0))
        + int(status.get("truth_corrected_strict_failure", 0))
        + int(status.get("non_cross_macro_engineering_failure", 0))
    )
    if strict_residual > 0:
        return "write_targeted_proposer_repair_plan", "All cross-macro rows are reviewed; strict residual failures should drive targeted proposer/classifier repair."
    return "stop_engineering_and_write_paper_results", "No strict residual failures remain in the review-aware atlas."


def build_review_aware_atlas(atlas: dict[str, Any], review_rows: list[dict[str, str]], semantic_policy: dict[str, Any] | None = None) -> dict[str, Any]:
    source_misses = atlas.get("misses", [])
    rows = [classify_miss_with_review(miss, review_rows) for miss in source_misses]
    summary = summarize_review_aware_misses(rows)
    recommendation, reason = choose_next_action(summary)
    return {
        "schema": "review_aware_residual_failure_atlas_v1",
        "source_atlas_schema": atlas.get("schema", "unknown"),
        "semantic_policy_recall": semantic_policy or {},
        "summary": summary,
        "recommendation": recommendation,
        "reason": reason,
        "misses": rows,
    }


def write_markdown(path: Path, payload: dict[str, Any], *, title: str = "Review-Aware Residual Failure Atlas") -> None:
    summary = payload["summary"]
    lines = [
        f"# {title}",
        "",
        "## Recommendation",
        "",
        f"**{payload['recommendation']}**",
        "",
        "## Reasoning",
        "",
        payload["reason"],
        "",
        "## Summary",
        "",
        f"- miss_count: `{summary['miss_count']}`",
        "",
        "## Review Status Counts",
        "",
        "| Status | Count |",
        "|--------|-------|",
    ]
    for status, count in summary["by_review_status"].items():
        lines.append(f"| {status} | {count} |")
    lines.extend(["", "## Macro Counts", "", "| Macro | Count |", "|-------|-------|"])
    for macro, count in summary["by_macro"].items():
        lines.append(f"| {macro} | {count} |")
    lines.extend(["", "## Status by Macro", "", "| Status | Macro | Count |", "|--------|-------|-------|"])
    for key, count in summary["by_status_macro"].items():
        status, macro = key.split("|", 1)
        lines.append(f"| {status} | {macro} | {count} |")
    lines.extend([
        "",
        "## Top Residual Misses",
        "",
        "| Sample | Macro | Area | Faces | Miss Cause | Review Status | User Decision |",
        "|--------|-------|------|-------|------------|---------------|---------------|",
    ])
    for row in payload["misses"][:80]:
        lines.append(
            f"| {row.get('sample', '')} | {row.get('macro', '')} | {float(row.get('area', 0.0)):.4f} | "
            f"{row.get('face_count', '')} | {row.get('miss_cause', '')} | {row.get('review_status', '')} | "
            f"{row.get('user_decision', '')} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_decision(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# v5.22 Decision",
        "",
        "## Recommendation",
        "",
        f"**{payload['recommendation']}**",
        "",
        "## Reasoning",
        "",
        payload["reason"],
        "",
        "## Review-Aware Counts",
        "",
        "| Status | Count |",
        "|--------|-------|",
    ]
    for status, count in summary["by_review_status"].items():
        lines.append(f"| {status} | {count} |")
    lines.extend(["", "## Top Macros for Next Repair", "", "| Macro | Count |", "|-------|-------|"])
    for macro, count in summary["by_macro"].items():
        lines.append(f"| {macro} | {count} |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a review-aware residual failure atlas.")
    parser.add_argument("--atlas-json", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--semantic-policy-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Review-Aware Residual Failure Atlas")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    atlas = load_json(args.atlas_json)
    review_rows = read_review_rows(args.review_csv)
    semantic_policy = load_json(args.semantic_policy_json)
    payload = build_review_aware_atlas(atlas, review_rows, semantic_policy)
    write_json(args.output_dir / "review_aware_residual_atlas.json", payload)
    write_markdown(args.output_dir / "review_aware_residual_atlas.md", payload, title=args.title)
    write_decision(args.output_dir / "v5_22_decision.md", payload)
    print(json.dumps({"summary": payload["summary"], "recommendation": payload["recommendation"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
