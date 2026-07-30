"""Write v5.21 decision memo from reviewed-truth recall, semantic policy, and miss atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def choose_next_recommendation(atlas_summary: dict[str, Any]) -> tuple[str, str]:
    causes = atlas_summary.get("by_miss_cause", atlas_summary.get("by_cause", {}))
    no_same = int(causes.get("no_same_macro_candidates", 0))
    cross = int(causes.get("cross_macro_area_compatible", 0))
    near = int(causes.get("same_macro_area_near_miss", 0)) + int(causes.get("cross_macro_area_near_miss", 0))
    if no_same >= max(cross, near) and no_same > 0:
        return "targeted_proposer_repair", "Reviewed truth and semantic policy leave no_same_macro_candidates as the dominant strict failure mode."
    if cross >= max(no_same, near) and cross > 0:
        return "inspect_remaining_cross_macro_semantics", "cross_macro_area_compatible remains dominant after v5.20 review, so inspect remaining macro semantics before coding."
    if near > 0:
        return "area_tolerance_or_boundary_refinement", "Near misses dominate, so inspect boundary grouping and area tolerance before new model work."
    return "hold_current_pipeline_and_write_results", "No dominant repair category remains."


def write_decision(path: Path, strict_report: dict[str, Any], policy_report: dict[str, Any], atlas: dict[str, Any]) -> None:
    best = strict_report.get("best_by_final_survival_recall") or {}
    stage = best.get("final_survival_stage") or {}
    atlas_summary = atlas.get("summary") or {}
    causes = atlas_summary.get("by_miss_cause", atlas_summary.get("by_cause", {}))
    recommendation, reason = choose_next_recommendation({"by_miss_cause": causes, "by_macro": atlas_summary.get("by_macro", {})})
    lines = [
        "# v5.21 Decision",
        "",
        "## Recommendation",
        "",
        f"**{recommendation}**",
        "",
        "## Reasoning",
        "",
        reason,
        "",
        "## Strict Recall",
        "",
        f"- truth_count: `{best.get('truth_count', strict_report.get('truth_count', 'N/A'))}`",
        f"- matched_truth_count: `{stage.get('matched_truth_count', 'N/A')}`",
        f"- recall: `{float(stage.get('recall', 0.0)):.4f}`",
        "",
        "## Semantic Policy Recall",
        "",
        f"- policy_recovered_count: `{policy_report.get('policy_recovered_count', 0)}`",
        f"- semantic_policy_recall: `{float(policy_report.get('semantic_policy_recall', 0.0)):.4f}`",
        "",
        "## Remaining Miss Causes",
        "",
        "| Cause | Count |",
        "|-------|-------|",
    ]
    for cause, count in causes.items():
        lines.append(f"| {cause} | {count} |")
    lines.extend(["", "## Remaining Missed Macros", "", "| Macro | Count |", "|-------|-------|"])
    for macro, count in (atlas_summary.get("by_macro") or {}).items():
        lines.append(f"| {macro} | {count} |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write v5.21 decision memo.")
    parser.add_argument("--strict-report-json", type=Path, required=True)
    parser.add_argument("--semantic-policy-json", type=Path, required=True)
    parser.add_argument("--atlas-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    strict_report = load_json(args.strict_report_json)
    policy_report = load_json(args.semantic_policy_json)
    atlas = load_json(args.atlas_json)
    write_decision(args.output_md, strict_report, policy_report, atlas)
    print(f"Decision written to {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
