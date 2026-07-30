"""v5.18 Proposal Miss Atlas.

Classify every no_area_compatible_proposal failure by analyzing nearest
same-macro and cross-macro candidates. Output a repair gate recommending
the next code-changing step.

Diagnostic-only — does not modify model, proposer, classifier, or inference.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

from utils.area_truth_normalization import normalize_truth_macro


MISS_CAUSES = (
    "no_same_macro_candidates",
    "same_macro_area_near_miss",
    "same_macro_split_or_merge_suspected",
    "cross_macro_area_compatible",
    "cross_macro_area_near_miss",
    "no_related_candidate",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


def normalized_macro(value: Any, *, sample: str | None = None, area: float | None = None, face_count: int | None = None) -> str:
    return normalize_truth_macro(
        str(value or ""),
        sample_id=sample,
        area=area,
        face_count=face_count,
        truth_source_name="Feature_Comparison_Table.xlsx",
    )


def area_compatible(
    truth_area: float,
    candidate_area: float,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> bool:
    truth_area = finite_float(truth_area)
    candidate_area = finite_float(candidate_area)
    diff = abs(truth_area - candidate_area)
    if diff <= absolute_tolerance:
        return True
    denominator = max(abs(truth_area), 1e-9)
    return diff / denominator <= relative_tolerance


def relative_error(truth_area: float, candidate_area: float) -> float:
    truth_area = finite_float(truth_area)
    candidate_area = finite_float(candidate_area)
    return abs(truth_area - candidate_area) / max(abs(truth_area), 1e-9)


def candidate_macro(candidate: dict[str, Any]) -> str:
    for key in ("proposal_macro", "classified_macro", "macro"):
        if candidate.get(key) is not None:
            return normalized_macro(candidate[key])
    return "unknown"


def candidate_area(candidate: dict[str, Any]) -> float:
    return finite_float(candidate.get("area", 0.0))


def compact_candidate(candidate: dict[str, Any], truth_area: float) -> dict[str, Any]:
    area = candidate_area(candidate)
    return {
        "source": str(candidate.get("source", "unknown")),
        "proposal_macro": candidate_macro(candidate),
        "classified_macro": normalized_macro(candidate.get("classified_macro", candidate.get("proposal_macro", ""))),
        "action": str(candidate.get("action", "")),
        "confidence": candidate.get("confidence"),
        "face_indices": list(candidate.get("face_indices", [])),
        "face_count": int(candidate.get("face_count", len(candidate.get("face_indices", [])))),
        "area": area,
        "absolute_error": abs(finite_float(truth_area) - area),
        "relative_error": relative_error(truth_area, area),
    }


def nearest_candidates(
    truth_area: float,
    candidates: list[dict[str, Any]],
    *,
    macro: str | None = None,
    exclude_macro: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        macro_value = candidate_macro(candidate)
        if macro is not None and macro_value != macro:
            continue
        if exclude_macro is not None and macro_value == exclude_macro:
            continue
        rows.append(compact_candidate(candidate, truth_area))
    rows.sort(key=lambda item: (item["relative_error"], item["absolute_error"], item["proposal_macro"]))
    return rows[:limit]


def closest_same_macro_area_sum(
    truth_area: float,
    same_macro_candidates: list[dict[str, Any]],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    max_group_size: int = 4,
) -> tuple[float, list[dict[str, Any]]]:
    best_sum = 0.0
    best_group: list[dict[str, Any]] = []
    candidates = sorted(
        same_macro_candidates,
        key=lambda item: relative_error(truth_area, candidate_area(item)),
    )[:10]
    for group_size in range(2, min(max_group_size, len(candidates)) + 1):
        for group in combinations(candidates, group_size):
            area_sum = sum(candidate_area(item) for item in group)
            if not best_group or relative_error(truth_area, area_sum) < relative_error(truth_area, best_sum):
                best_sum = area_sum
                best_group = list(group)
            if area_compatible(
                truth_area,
                area_sum,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            ):
                return area_sum, [compact_candidate(item, truth_area) for item in group]
    return best_sum, [compact_candidate(item, truth_area) for item in best_group]


def classify_no_proposal_failure(
    failure: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    sample = str(failure.get("sample", ""))
    truth_area = finite_float(failure.get("area", 0.0))
    face_count = int(failure.get("face_count", 0) or 0)
    macro = normalized_macro(failure.get("macro", ""), sample=sample, area=truth_area, face_count=face_count)
    same_macro_candidates = [candidate for candidate in candidates if candidate_macro(candidate) == macro]
    cross_macro_candidates = [candidate for candidate in candidates if candidate_macro(candidate) != macro]
    nearest_same = nearest_candidates(truth_area, same_macro_candidates, limit=5)
    nearest_cross = nearest_candidates(truth_area, cross_macro_candidates, limit=5)
    compatible_cross = [
        item for item in nearest_cross
        if area_compatible(
            truth_area,
            item["area"],
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
    ]
    same_area_sum, same_area_group = closest_same_macro_area_sum(
        truth_area,
        same_macro_candidates,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )

    if same_macro_candidates and same_area_group and area_compatible(
        truth_area,
        same_area_sum,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    ):
        miss_cause = "same_macro_split_or_merge_suspected"
    elif same_macro_candidates and nearest_same and nearest_same[0]["relative_error"] <= 0.25:
        miss_cause = "same_macro_area_near_miss"
    elif compatible_cross:
        miss_cause = "cross_macro_area_compatible"
    elif same_macro_candidates and nearest_cross and nearest_cross[0]["relative_error"] <= 0.25:
        miss_cause = "cross_macro_area_near_miss"
    elif not same_macro_candidates:
        miss_cause = "no_same_macro_candidates"
    elif nearest_cross and nearest_cross[0]["relative_error"] <= 0.25:
        miss_cause = "cross_macro_area_near_miss"
    else:
        miss_cause = "no_related_candidate"

    return {
        "sample": sample,
        "truth_index": failure.get("truth_index"),
        "macro": macro,
        "area": truth_area,
        "face_count": face_count,
        "failure_stage": failure.get("failure_stage"),
        "miss_cause": miss_cause,
        "candidate_count": len(candidates),
        "same_macro_candidate_count": len(same_macro_candidates),
        "cross_macro_candidate_count": len(cross_macro_candidates),
        "nearest_same_macro": nearest_same,
        "nearest_cross_macro": nearest_cross,
        "same_macro_area_sum": same_area_sum,
        "same_macro_area_group": same_area_group,
    }


def summarize_misses(misses: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "miss_count": len(misses),
        "by_macro": dict(Counter(item["macro"] for item in misses).most_common()),
        "by_part": dict(Counter(item["sample"] for item in misses).most_common()),
        "by_cause": dict(Counter(item["miss_cause"] for item in misses).most_common()),
        "by_macro_cause": dict(
            Counter(f"{item['macro']}|{item['miss_cause']}" for item in misses).most_common()
        ),
        "by_part_macro": dict(
            Counter(f"{item['sample']}|{item['macro']}" for item in misses).most_common()
        ),
    }


def build_atlas(report: dict[str, Any]) -> dict[str, Any]:
    best = report.get("best_by_final_survival_recall") or {}
    absolute_tolerance = finite_float(report.get("absolute_area_tolerance", 10.0), 10.0)
    relative_tolerance = finite_float(report.get("relative_area_tolerance", 0.02), 0.02)
    misses: list[dict[str, Any]] = []
    for sample in best.get("samples", []):
        sample_name = str(sample.get("sample", ""))
        candidates = list(sample.get("candidates", []))
        for failure in sample.get("truth_failures", []):
            if failure.get("failure_stage") != "no_area_compatible_proposal":
                continue
            enriched_failure = dict(failure)
            enriched_failure["sample"] = sample_name
            misses.append(
                classify_no_proposal_failure(
                    enriched_failure,
                    candidates,
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                )
            )

    return {
        "schema": "proposal_miss_atlas_v1",
        "source_report": report.get("classifier", report.get("schema", "unknown")),
        "absolute_area_tolerance": absolute_tolerance,
        "relative_area_tolerance": relative_tolerance,
        "summary": summarize_misses(misses),
        "misses": misses,
    }


def recommend_next_action(summary: dict[str, Any]) -> tuple[str, str]:
    miss_count = int(summary.get("miss_count", 0))
    by_cause = summary.get("by_cause", {})
    no_same = int(by_cause.get("no_same_macro_candidates", 0))
    split_merge = int(by_cause.get("same_macro_split_or_merge_suspected", 0))
    cross_compatible = int(by_cause.get("cross_macro_area_compatible", 0))
    near_miss = int(by_cause.get("same_macro_area_near_miss", 0))
    if miss_count == 0:
        return "no_proposal_repair_needed", "No no-area-compatible proposal failures remain."
    if no_same >= max(split_merge, cross_compatible, near_miss) and no_same / miss_count >= 0.35:
        return "targeted_proposal_generation_repair", "Most misses have no same-macro candidate, so repair proposal generators for dominant missed macros."
    if split_merge >= max(no_same, cross_compatible, near_miss) and split_merge / miss_count >= 0.25:
        return "candidate_merge_split_repair", "Many same-macro candidates exist only as split/merged fragments, so repair candidate assembly before adding new feature types."
    if cross_compatible >= max(no_same, split_merge, near_miss) and cross_compatible / miss_count >= 0.25:
        return "truth_or_macro_semantic_audit", "Many misses have area-compatible cross-macro candidates, so inspect macro semantics and truth labels before coding proposal repairs."
    if near_miss / miss_count >= 0.25:
        return "area_tolerance_or_boundary_refinement_audit", "Many misses are close in area but outside tolerance, so inspect area truth precision and boundary grouping."
    return "manual_review_then_targeted_repair", "No single miss cause dominates; manually review the top macro/part clusters before changing proposal code."


def write_markdown(path: Path, atlas: dict[str, Any], title: str) -> None:
    summary = atlas["summary"]
    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        f"- miss_count: `{summary['miss_count']}`",
        f"- absolute_area_tolerance: `{atlas['absolute_area_tolerance']}`",
        f"- relative_area_tolerance: `{atlas['relative_area_tolerance']}`",
        "",
        "## Miss Cause Counts",
        "",
        "| Cause | Count |",
        "|-------|-------|",
    ]
    for cause, count in summary["by_cause"].items():
        lines.append(f"| {cause} | {count} |")
    lines.extend(["", "## Macro Counts", "", "| Macro | Count |", "|-------|-------|"])
    for macro, count in summary["by_macro"].items():
        lines.append(f"| {macro} | {count} |")
    lines.extend(["", "## Part Counts", "", "| Part | Count |", "|------|-------|"])
    for part, count in summary["by_part"].items():
        lines.append(f"| {part} | {count} |")
    lines.extend(["", "## Top Miss Details", "", "| Sample | Macro | Area | Faces | Cause | Same Macro Candidates | Nearest Same | Nearest Cross |", "|--------|-------|------|-------|-------|-----------------------|--------------|---------------|"])
    for miss in atlas["misses"][:80]:
        nearest_same = miss["nearest_same_macro"][0] if miss["nearest_same_macro"] else {}
        nearest_cross = miss["nearest_cross_macro"][0] if miss["nearest_cross_macro"] else {}
        nearest_same_text = f"{nearest_same.get('proposal_macro', '')}:{nearest_same.get('area', ''):.2f}:{nearest_same.get('relative_error', ''):.3f}" if nearest_same else ""
        nearest_cross_text = f"{nearest_cross.get('proposal_macro', '')}:{nearest_cross.get('area', ''):.2f}:{nearest_cross.get('relative_error', ''):.3f}" if nearest_cross else ""
        lines.append(
            f"| {miss['sample']} | {miss['macro']} | {miss['area']:.2f} | {miss['face_count']} | "
            f"{miss['miss_cause']} | {miss['same_macro_candidate_count']} | {nearest_same_text} | {nearest_cross_text} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_repair_gate(path: Path, named_atlases: list[tuple[str, dict[str, Any]]]) -> None:
    lines = [
        "# v5.18 Repair Gate",
        "",
        "## Recommendation By Report",
        "",
        "| Report | Miss Count | Recommendation | Reason |",
        "|--------|------------|----------------|--------|",
    ]
    for name, atlas in named_atlases:
        recommendation, reason = recommend_next_action(atlas["summary"])
        lines.append(f"| {name} | {atlas['summary']['miss_count']} | {recommendation} | {reason} |")
    lines.extend([
        "",
        "## Required Next Step",
        "",
        "Use the v5.13 atlas as the primary decision source because v5.13 is still the current best real-S3 baseline. Use v5.17_plain only as supporting evidence for whether role-aware features changed the failure mix.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_next_step_recommendation(path: Path, primary_name: str, atlas: dict[str, Any]) -> None:
    recommendation, reason = recommend_next_action(atlas["summary"])
    summary = atlas["summary"]
    top_macros = list(summary["by_macro"].items())[:8]
    top_parts = list(summary["by_part"].items())[:8]
    lines = [
        "# v5.18 Next Step Recommendation",
        "",
        f"Primary report: `{primary_name}`",
        "",
        "## Recommendation",
        "",
        f"**{recommendation}**",
        "",
        "## Reasoning",
        "",
        reason,
        "",
        "## Top Missed Macros",
        "",
        "| Macro | Count |",
        "|-------|-------|",
    ]
    for macro, count in top_macros:
        lines.append(f"| {macro} | {count} |")
    lines.extend(["", "## Top Missed Parts", "", "| Part | Count |", "|------|-------|"])
    for part, count in top_parts:
        lines.append(f"| {part} | {count} |")
    lines.extend([
        "",
        "## Execution Rule For Next Agent",
        "",
        "If the recommendation is `targeted_proposal_generation_repair`, write the next plan against `utils/object_candidate_proposer.py` and the topology candidate functions it imports. If the recommendation is `truth_or_macro_semantic_audit`, do not change proposer code until the area truth table or macro semantics are reviewed. If the recommendation is `candidate_merge_split_repair`, write the next plan against candidate assembly and same-instance grouping.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_named_report(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"report must use name=path format, got {value!r}")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    raw_path = raw_path.strip()
    if not name or not raw_path:
        raise ValueError(f"report must use name=path format, got {value!r}")
    return name, Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build no-compatible-proposal failure atlas.")
    parser.add_argument("--report", action="append", required=True, help="Named report in name=path format.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    named_atlases: list[tuple[str, dict[str, Any]]] = []
    for name, report_path in [parse_named_report(item) for item in args.report]:
        atlas = build_atlas(load_json(report_path))
        named_atlases.append((name, atlas))
        write_json(args.output_dir / f"{name}_proposal_miss_atlas.json", atlas)
        write_markdown(args.output_dir / f"{name}_proposal_miss_atlas.md", atlas, f"{name} Proposal Miss Atlas")
    write_repair_gate(args.output_dir / "v5_18_repair_gate.md", named_atlases)
    if named_atlases:
        write_next_step_recommendation(
            args.output_dir / "v5_18_next_step_recommendation.md",
            named_atlases[0][0],
            named_atlases[0][1],
        )
    print(json.dumps({name: atlas["summary"] for name, atlas in named_atlases}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
