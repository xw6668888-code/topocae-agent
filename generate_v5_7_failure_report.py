from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CORE_FIX_MACROS = {
    "rib",
    "slot_u",
    "slot_o",
    "slot_rect",
    "slot_v",
    "pocket_rect",
    "pocket_circular",
    "through_hole_round",
    "blind_hole_round",
    "counterbore_hole",
    "countersink_hole",
    "rectangular_boss",
    "cylinder_boss",
    "obround_hole",
}
TRANSITION_MACROS = {"fillet", "chamfer"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v5.7 formal failure atlas and fix-priority report.")
    parser.add_argument("--area-review-json", type=Path, required=True)
    parser.add_argument("--prediction-path", type=Path, required=True)
    parser.add_argument("--feature-objects-path", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--dataset-name", default="v5.7")
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def collect_json_files(path: Path, recursive: bool, ignored_names: set[str]) -> list[Path]:
    if path.is_file():
        return [path]
    globber = path.rglob if recursive else path.glob
    return sorted(p for p in globber("*.json") if p.name not in ignored_names)


def macro_counter(items: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for item in items:
        macro = item.get("macro")
        if macro:
            counter[str(macro)] += 1
    return counter


def summarize_area_review(area_review: dict[str, Any]) -> dict[str, Any]:
    missed_counter: Counter[str] = Counter()
    false_positive_counter: Counter[str] = Counter()
    sample_summaries = []
    for sample in area_review.get("samples", []):
        missed = list(sample.get("missed_truth", []))
        false_positives = list(sample.get("false_positives", []))
        matched = list(sample.get("matches", []))
        missed_counter.update(macro_counter(missed))
        false_positive_counter.update(macro_counter(false_positives))
        sample_summaries.append(
            {
                "sample_id": sample.get("sample_id"),
                "recall": sample.get("recall"),
                "precision": sample.get("precision"),
                "missed_count": len(missed),
                "false_positive_count": len(false_positives),
                "match_count": len(matched),
                "missed_macro_counts": dict(sorted(macro_counter(missed).items())),
                "false_positive_macro_counts": dict(sorted(macro_counter(false_positives).items())),
                "missed_truth": missed,
                "false_positives": false_positives,
            }
        )
    sample_summaries.sort(key=lambda item: (-(item["missed_count"] + item["false_positive_count"]), str(item["sample_id"])))
    return {
        "missed_macro_counts": dict(sorted(missed_counter.items())),
        "false_positive_macro_counts": dict(sorted(false_positive_counter.items())),
        "sample_summaries": sample_summaries,
    }


def summarize_rib_decisions(prediction_path: Path, recursive: bool) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    sample_summaries = {}
    total_decisions = 0
    total_accepted = 0
    for path in collect_json_files(prediction_path, recursive, {"prediction_manifest.json"}):
        doc = load_json(path)
        post = doc.get("rib_boundary_classifier_postprocess", {}) or {}
        decisions = list(post.get("decision_records", []))
        accepted = list(post.get("records", []))
        if not decisions and accepted:
            decisions = accepted
        sample_id = str(doc.get("sample_id") or path.stem)
        sample_action_counts: Counter[str] = Counter()
        for decision in decisions:
            action = str(decision.get("action", "unknown"))
            stage = str(decision.get("stage", "unknown"))
            action_counts[action] += 1
            stage_counts[stage] += 1
            sample_action_counts[action] += 1
        total_decisions += len(decisions)
        total_accepted += len(accepted)
        if decisions or accepted:
            sample_summaries[sample_id] = {
                "decision_count": len(decisions),
                "accepted_count": len(accepted),
                "action_counts": dict(sorted(sample_action_counts.items())),
                "pruned_count": int(post.get("pruned_count", 0) or 0),
                "expanded_count": int(post.get("expanded_count", 0) or 0),
            }
    return {
        "decision_count": total_decisions,
        "accepted_count": total_accepted,
        "action_counts": dict(sorted(action_counts.items())),
        "stage_counts": dict(sorted(stage_counts.items())),
        "samples": dict(sorted(sample_summaries.items())),
    }


def summarize_feature_objects(feature_objects_path: Path | None, recursive: bool) -> dict[str, Any]:
    if feature_objects_path is None:
        return {}
    macro_counts: Counter[str] = Counter()
    sample_counts = {}
    for path in collect_json_files(feature_objects_path, recursive, {"feature_object_manifest.json"}):
        doc = load_json(path)
        sample_id = str(doc.get("sample_id") or path.stem)
        sample_macro_counts: Counter[str] = Counter()
        for obj in doc.get("feature_objects", []):
            macro = str(obj.get("macro_feature", "unknown"))
            macro_counts[macro] += 1
            sample_macro_counts[macro] += 1
        sample_counts[sample_id] = dict(sorted(sample_macro_counts.items()))
    return {
        "macro_counts": dict(sorted(macro_counts.items())),
        "sample_macro_counts": dict(sorted(sample_counts.items())),
    }


def build_fix_recommendations(residual_summary: dict[str, Any], rib_decision_summary: dict[str, Any]) -> list[dict[str, Any]]:
    missed = Counter(residual_summary.get("missed_macro_counts", {}))
    false_pos = Counter(residual_summary.get("false_positive_macro_counts", {}))
    macros = sorted(set(missed) | set(false_pos))
    recommendations = []
    for macro in macros:
        missed_count = int(missed.get(macro, 0))
        false_positive_count = int(false_pos.get(macro, 0))
        total = missed_count + false_positive_count
        if macro == "rib":
            priority = "high" if total >= 3 else "medium"
            action = "continue_with_reviewed_face_truth"
            rationale = "Rib errors affect CAE idealization and boundary/load region selection; area-only truth is not enough for boundary membership."
        elif macro in CORE_FIX_MACROS and missed_count >= 3:
            priority = "high"
            action = "targeted_data_and_rule_fix"
            rationale = "Repeated missed target macro in the first-paper scope."
        elif macro in CORE_FIX_MACROS and false_positive_count >= 5:
            priority = "medium"
            action = "classifier_calibration_or_candidate_filter"
            rationale = "Repeated false positives can reduce precision, but may not require new geometry generation."
        elif macro in TRANSITION_MACROS:
            priority = "low"
            action = "defer_unless_transition_is_core_to_case"
            rationale = "Transition surplus often reflects truth granularity; fix only if it harms selected CAE regions."
        else:
            priority = "low"
            action = "monitor"
            rationale = "Sparse or lower-priority residual error."
        recommendations.append(
            {
                "macro": macro,
                "priority": priority,
                "recommended_action": action,
                "missed_count": missed_count,
                "false_positive_count": false_positive_count,
                "rationale": rationale,
            }
        )

    action_counts = Counter(rib_decision_summary.get("action_counts", {}))
    if action_counts.get("skipped_low_confidence", 0) > 0:
        recommendations.append(
            {
                "macro": "rib_boundary_classifier",
                "priority": "medium",
                "recommended_action": "calibrate_threshold_or_add_reviewed_hard_cases",
                "missed_count": 0,
                "false_positive_count": 0,
                "rationale": "Low-confidence skipped rib decisions indicate uncertainty rather than absent candidates.",
            }
        )
    priority_order = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(key=lambda item: (priority_order[item["priority"]], item["macro"]))
    return recommendations


def build_v5_7_report(
    *,
    area_review_json: Path,
    prediction_path: Path,
    feature_objects_path: Path | None,
    recursive: bool,
    dataset_name: str,
) -> dict[str, Any]:
    area_review = load_json(area_review_json)
    residual_summary = summarize_area_review(area_review)
    rib_decision_summary = summarize_rib_decisions(prediction_path, recursive)
    feature_object_summary = summarize_feature_objects(feature_objects_path, recursive)
    return {
        "schema": "cae_v5_7_failure_report_v1",
        "dataset_name": dataset_name,
        "area_review_json": str(area_review_json.resolve()),
        "prediction_path": str(prediction_path.resolve()),
        "feature_objects_path": str(feature_objects_path.resolve()) if feature_objects_path else None,
        "overall": {
            "sample_count": area_review.get("sample_count"),
            "truth_count": area_review.get("truth_count"),
            "prediction_count": area_review.get("prediction_count"),
            "matched_truth_count": area_review.get("matched_truth_count"),
            "matched_prediction_count": area_review.get("matched_prediction_count"),
            "recall": area_review.get("recall"),
            "precision": area_review.get("precision"),
            "matched_macro_counts": area_review.get("matched_macro_counts", {}),
            "predicted_macro_counts": area_review.get("predicted_macro_counts", {}),
            "truth_macro_counts": area_review.get("truth_macro_counts", {}),
        },
        "residual_error_summary": residual_summary,
        "rib_boundary_decision_summary": rib_decision_summary,
        "feature_object_summary": feature_object_summary,
        "fix_recommendations": build_fix_recommendations(residual_summary, rib_decision_summary),
    }


def format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "{}"
    return ", ".join(f"`{key}`: {value}" for key, value in counts.items())


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overall = report["overall"]
    residual = report["residual_error_summary"]
    rib = report["rib_boundary_decision_summary"]
    lines = [
        "# v5.7 Failure Atlas And Fix Evaluation",
        "",
        f"Dataset: `{report['dataset_name']}`",
        f"Area review: `{report['area_review_json']}`",
        f"Predictions: `{report['prediction_path']}`",
        "",
        "## Overall",
        "",
        f"- Samples: `{overall.get('sample_count')}`",
        f"- Truth / predictions / matched truth: `{overall.get('truth_count')}` / `{overall.get('prediction_count')}` / `{overall.get('matched_truth_count')}`",
        f"- Recall / precision: `{float(overall.get('recall') or 0.0):.4f}` / `{float(overall.get('precision') or 0.0):.4f}`",
        f"- Matched macro counts: {format_counts(overall.get('matched_macro_counts', {}))}",
        "",
        "## Residual Errors",
        "",
        f"- Missed macro counts: {format_counts(residual.get('missed_macro_counts', {}))}",
        f"- False-positive macro counts: {format_counts(residual.get('false_positive_macro_counts', {}))}",
        "",
        "## Rib Boundary Decisions",
        "",
        f"- Decisions: `{rib.get('decision_count', 0)}`",
        f"- Accepted actions: `{rib.get('accepted_count', 0)}`",
        f"- Action counts: {format_counts(rib.get('action_counts', {}))}",
        "",
        "## Fix Recommendations",
        "",
        "| Priority | Macro | Missed | False positives | Recommended action |",
        "|---|---|---:|---:|---|",
    ]
    for item in report["fix_recommendations"]:
        lines.append(
            f"| `{item['priority']}` | `{item['macro']}` | {item['missed_count']} | "
            f"{item['false_positive_count']} | `{item['recommended_action']}` |"
        )
    lines.extend(["", "## Worst Samples", ""])
    lines.append("| Sample | Missed | False positives | Missed macros | False-positive macros |")
    lines.append("|---|---:|---:|---|---|")
    for sample in residual.get("sample_summaries", [])[:12]:
        lines.append(
            f"| `{sample['sample_id']}` | {sample['missed_count']} | {sample['false_positive_count']} | "
            f"{format_counts(sample.get('missed_macro_counts', {}))} | "
            f"{format_counts(sample.get('false_positive_macro_counts', {}))} |"
        )
    lines.extend(["", "## Assessment", ""])
    lines.append("- Continue fixing `high` priority items only when they repeat across samples or directly affect CAE selectable regions.")
    lines.append("- Defer low-priority transition surplus unless manual inspection shows it changes the intended CAE operation.")
    lines.append("- For rib boundary, the next reliable improvement requires reviewed face-level truth on the flagged samples; area-only truth is too coarse.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    report = build_v5_7_report(
        area_review_json=args.area_review_json,
        prediction_path=args.prediction_path,
        feature_objects_path=args.feature_objects_path,
        recursive=args.recursive,
        dataset_name=args.dataset_name,
    )
    write_json(args.output_json, report)
    write_markdown(args.output_md, report)
    print(f"v5.7 failure report json written: {args.output_json}")
    print(f"v5.7 failure report markdown written: {args.output_md}")


if __name__ == "__main__":
    main()
