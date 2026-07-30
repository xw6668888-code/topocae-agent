"""v5.15 Classifier Error Atlas.

Compare old v5.10 and new v5.14 candidate classifiers on identical v5.13 hybrid
proposals. Identify per-candidate error categories and audit training label
distribution by source/macro.

Diagnostic-only — does not modify model, training, or inference behavior.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from object_proposal_recall_report import (
    candidate_area,
    candidate_matches_truth,
    load_json,
    normalize_truth_entries,
)
from utils.instance_candidate_assembly import assemble_instance_candidates
from utils.instance_candidate_postprocess import (
    classify_instance_candidate_proposals,
    load_instance_candidate_classifier_bundle,
)
from utils.object_candidate_proposer import generate_hybrid_object_proposals


REJECT_LABEL = "reject"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def face_key(record: dict[str, Any]) -> tuple[int, ...]:
    return tuple(sorted(int(idx) for idx in record.get("face_indices", [])))


def proposal_key(record: dict[str, Any]) -> tuple[str, tuple[int, ...]]:
    return (str(record.get("proposal_macro", "")), face_key(record))


def apply_action_policy(records: list[dict[str, Any]], min_confidence: float) -> list[dict[str, Any]]:
    occupied_faces: set[int] = set()
    enriched = [dict(record) for record in records]
    for record in enriched:
        record.setdefault("action", "unprocessed")
    for record in sorted(enriched, key=lambda item: float(item.get("confidence", 0.0)), reverse=True):
        faces = set(int(idx) for idx in record.get("face_indices", []))
        if str(record.get("predicted_label", "")) == REJECT_LABEL:
            record["action"] = "rejected_label"
        elif float(record.get("confidence", 0.0)) < float(min_confidence):
            record["action"] = "skipped_low_confidence"
        elif faces & occupied_faces:
            record["action"] = "skipped_overlap"
        else:
            occupied_faces.update(faces)
            record["action"] = "accepted_remap"
    return enriched


def classifier_record_by_key(records: list[dict[str, Any]]) -> dict[tuple[str, tuple[int, ...]], dict[str, Any]]:
    return {proposal_key(record): record for record in records}


def is_correct_accept(record: dict[str, Any] | None, matched_truth_macros: list[str]) -> bool:
    if record is None:
        return False
    if record.get("action") != "accepted_remap":
        return False
    return str(record.get("predicted_label", "")) in set(matched_truth_macros)


def compare_candidate_decision(
    candidate: dict[str, Any],
    old_record: dict[str, Any] | None,
    new_record: dict[str, Any] | None,
) -> dict[str, Any]:
    truth_macros = list(candidate.get("matched_truth_macros", []))
    old_correct = is_correct_accept(old_record, truth_macros)
    new_correct = is_correct_accept(new_record, truth_macros)
    old_action = old_record.get("action") if old_record else "missing"
    new_action = new_record.get("action") if new_record else "missing"

    if old_correct and not new_correct:
        if new_action == "rejected_label":
            category = "old_correct_accept_new_reject"
        elif new_action == "skipped_low_confidence":
            category = "old_correct_accept_new_low_confidence"
        elif new_action == "skipped_overlap":
            category = "old_correct_accept_new_overlap"
        else:
            category = "old_correct_accept_new_wrong_label"
    elif new_correct and not old_correct:
        category = "new_correct_old_wrong"
    elif old_correct and new_correct:
        category = "both_correct_accept"
    elif truth_macros:
        category = "both_wrong_on_matched_proposal"
    else:
        category = "unmatched_proposal"

    return {
        "sample": candidate.get("sample"),
        "proposal_macro": candidate.get("proposal_macro"),
        "proposal_sources": list(candidate.get("proposal_sources", [])),
        "face_indices": list(candidate.get("face_indices", [])),
        "area": float(candidate.get("area", 0.0)),
        "matched_truth_macros": truth_macros,
        "old_predicted_label": old_record.get("predicted_label") if old_record else None,
        "old_confidence": old_record.get("confidence") if old_record else None,
        "old_action": old_action,
        "new_predicted_label": new_record.get("predicted_label") if new_record else None,
        "new_confidence": new_record.get("confidence") if new_record else None,
        "new_action": new_action,
        "old_correct": old_correct,
        "new_correct": new_correct,
        "error_category": category,
    }


def audit_training_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_source_macro: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    by_label = Counter()
    by_source = Counter()
    for record in records:
        source = str(record.get("source", "unknown"))
        macro = str(record.get("proposal_macro", "unknown"))
        label = str(record.get("label", "unknown"))
        by_label[label] += 1
        by_source[source] += 1
        bucket = by_source_macro[source][macro]
        bucket["record_count"] += 1
        if label == REJECT_LABEL:
            bucket["reject_count"] += 1
        else:
            bucket["positive_count"] += 1
            if label == macro:
                bucket["same_macro_positive_count"] += 1
            else:
                bucket["cross_macro_positive_count"] += 1

    normalized: dict[str, dict[str, dict[str, float]]] = {}
    for source, macro_map in by_source_macro.items():
        normalized[source] = {}
        for macro, counter in macro_map.items():
            total = int(counter["record_count"])
            positive = int(counter["positive_count"])
            reject = int(counter["reject_count"])
            normalized[source][macro] = {
                "record_count": total,
                "positive_count": positive,
                "reject_count": reject,
                "same_macro_positive_count": int(counter["same_macro_positive_count"]),
                "cross_macro_positive_count": int(counter["cross_macro_positive_count"]),
                "positive_rate": positive / total if total else 0.0,
                "reject_rate": reject / total if total else 0.0,
            }

    return {
        "record_count": len(records),
        "label_counts": dict(sorted(by_label.items())),
        "source_counts": dict(sorted(by_source.items())),
        "by_source_macro": normalized,
    }


# ---------------------------------------------------------------------------
# s3 comparison
# ---------------------------------------------------------------------------


def collect_json_by_stem(directory: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted(directory.glob("*.json")):
        if "manifest" in path.stem.lower():
            continue
        result[path.stem] = load_json(path)
    return result


def matched_truth_macros_for_candidate(
    candidate: dict[str, Any],
    truth_entries: list[dict[str, Any]],
    absolute_area_tolerance: float,
    relative_area_tolerance: float,
) -> list[str]:
    macros = []
    for truth in truth_entries:
        if candidate_matches_truth(
            candidate,
            truth,
            "proposal",
            absolute_area_tolerance=absolute_area_tolerance,
            relative_area_tolerance=relative_area_tolerance,
        ):
            macros.append(str(truth["macro"]))
    return sorted(set(macros))


def classify_with_bundle(
    graph: dict[str, Any],
    raw_prediction: dict[str, Any],
    proposals: list[dict[str, Any]],
    bundle: dict[str, Any],
    min_confidence: float,
) -> dict[tuple[str, tuple[int, ...]], dict[str, Any]]:
    records = classify_instance_candidate_proposals(graph, raw_prediction, proposals, bundle)
    records = apply_action_policy(records, min_confidence=min_confidence)
    return classifier_record_by_key(records)


def build_sample_comparisons(
    sample: str,
    graph: dict[str, Any],
    raw_prediction: dict[str, Any],
    truth_entries: list[dict[str, Any]],
    old_bundle: dict[str, Any],
    new_bundle: dict[str, Any],
    min_confidence: float,
    absolute_area_tolerance: float,
    relative_area_tolerance: float,
) -> list[dict[str, Any]]:
    same_instance = assemble_instance_candidates(
        graph,
        raw_prediction,
        strong_threshold=0.30,
        weak_threshold=0.20,
    )
    proposals = generate_hybrid_object_proposals(
        graph,
        raw_prediction,
        same_instance_proposals=same_instance,
    )
    old_by_key = classify_with_bundle(graph, raw_prediction, proposals, old_bundle, min_confidence)
    new_by_key = classify_with_bundle(graph, raw_prediction, proposals, new_bundle, min_confidence)

    comparisons = []
    for proposal in proposals:
        candidate = {
            "sample": sample,
            "proposal_macro": str(proposal.get("proposal_macro", "")),
            "proposal_sources": list(proposal.get("proposal_sources", [proposal.get("source", "unknown")])),
            "face_indices": sorted(int(idx) for idx in proposal.get("face_indices", [])),
            "area": candidate_area(proposal, graph),
        }
        candidate["matched_truth_macros"] = matched_truth_macros_for_candidate(
            candidate,
            truth_entries,
            absolute_area_tolerance=absolute_area_tolerance,
            relative_area_tolerance=relative_area_tolerance,
        )
        key = proposal_key(proposal)
        comparisons.append(compare_candidate_decision(candidate, old_by_key.get(key), new_by_key.get(key)))
    return comparisons


def summarize_comparisons(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    by_category = Counter(record["error_category"] for record in comparisons)
    by_macro_category = Counter((record["proposal_macro"], record["error_category"]) for record in comparisons)
    by_source_category = Counter()
    by_source_macro_category = Counter()
    matched_records = [record for record in comparisons if record.get("matched_truth_macros")]
    for record in comparisons:
        for source in record.get("proposal_sources", []):
            by_source_category[(source, record["error_category"])] += 1
            by_source_macro_category[(source, record["proposal_macro"], record["error_category"])] += 1
    return {
        "candidate_count": len(comparisons),
        "matched_candidate_count": len(matched_records),
        "by_category": dict(sorted(by_category.items())),
        "by_macro_category": [
            {"macro": macro, "category": category, "count": count}
            for (macro, category), count in by_macro_category.most_common()
        ],
        "by_source_category": [
            {"source": source, "category": category, "count": count}
            for (source, category), count in by_source_category.most_common()
        ],
        "by_source_macro_category": [
            {"source": source, "macro": macro, "category": category, "count": count}
            for (source, macro, category), count in by_source_macro_category.most_common()
        ],
    }


# ---------------------------------------------------------------------------
# Markdown writers
# ---------------------------------------------------------------------------


def write_error_atlas_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# v5.15 Classifier Error Atlas",
        "",
        "## Summary",
        "",
        f"- candidate_count: `{summary['candidate_count']}`",
        f"- matched_candidate_count: `{summary['matched_candidate_count']}`",
        "",
        "## Error Categories",
        "",
        "| Category | Count |",
        "|----------|-------|",
    ]
    for category, count in summary["by_category"].items():
        lines.append(f"| {category} | {count} |")
    lines.extend(["", "## Macro x Category", "", "| Macro | Category | Count |", "|-------|----------|-------|"])
    for row in summary["by_macro_category"][:60]:
        lines.append(f"| {row['macro']} | {row['category']} | {row['count']} |")
    lines.extend(["", "## Source x Macro x Category", "", "| Source | Macro | Category | Count |", "|--------|-------|----------|-------|"])
    for row in summary["by_source_macro_category"][:80]:
        lines.append(f"| {row['source']} | {row['macro']} | {row['category']} | {row['count']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_training_audit_markdown(path: Path, audit: dict[str, Any], comparisons: list[dict[str, Any]]) -> None:
    lost = [
        record for record in comparisons
        if record["error_category"].startswith("old_correct_accept_new")
    ]
    lost_by_source_macro = Counter()
    for record in lost:
        for source in record.get("proposal_sources", []):
            lost_by_source_macro[(source, record["proposal_macro"])] += 1

    lines = [
        "# v5.15 Training Label Audit",
        "",
        f"- training_record_count: `{audit['record_count']}`",
        "",
        "## Source Counts",
        "",
        "| Source | Count |",
        "|--------|-------|",
    ]
    for source, count in sorted(audit["source_counts"].items()):
        lines.append(f"| {source} | {count} |")
    lines.extend([
        "",
        "## Source/Macro Label Rates for Lost Real Candidates",
        "",
        "| Source | Proposal Macro | Lost Old-Correct Count | Train Records | Train Positive Rate | Train Reject Rate |",
        "|--------|----------------|------------------------|---------------|---------------------|-------------------|",
    ])
    for (source, macro), lost_count in lost_by_source_macro.most_common():
        stats = audit["by_source_macro"].get(source, {}).get(macro, {})
        lines.append(
            f"| {source} | {macro} | {lost_count} | {stats.get('record_count', 0)} | "
            f"{stats.get('positive_rate', 0.0):.4f} | {stats.get('reject_rate', 0.0):.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_decision(report: dict[str, Any], audit: dict[str, Any]) -> tuple[str, str]:
    categories = report["summary"]["by_category"]
    old_lost = sum(count for category, count in categories.items() if category.startswith("old_correct_accept_new"))
    new_gain = int(categories.get("new_correct_old_wrong", 0))
    both_wrong = int(categories.get("both_wrong_on_matched_proposal", 0))

    if old_lost > new_gain * 2 and old_lost >= 10:
        return (
            "keep_v5_13_classifier_and_build_source_macro_label_rebalance",
            "The v5.14 classifier loses many candidates the old classifier accepted correctly. Keep v5.13 as current best and rebalance hybrid source/macro labels before retraining.",
        )
    if both_wrong >= old_lost and both_wrong >= 10:
        return (
            "candidate_feature_revision",
            "Both classifiers fail many matched candidates. Add discriminative candidate features before retraining.",
        )
    return (
        "classifier_threshold_scan",
        "The error split does not show a dominant label-distribution failure. Scan confidence thresholds before changing labels or features.",
    )


def write_decision(path: Path, report: dict[str, Any], audit: dict[str, Any]) -> None:
    recommendation, reason = build_decision(report, audit)
    lines = [
        "# v5.15 Classifier Error Atlas Decision",
        "",
        "## Recommendation",
        "",
        f"**{recommendation}**",
        "",
        "## Reasoning",
        "",
        reason,
        "",
        "## Supporting Counts",
        "",
    ]
    for category, count in report["summary"]["by_category"].items():
        lines.append(f"- {category}: {count}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare old/new candidate classifiers on identical hybrid proposals.")
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--raw-prediction-dir", type=Path, required=True)
    parser.add_argument("--truth-json", type=Path, required=True)
    parser.add_argument("--old-classifier", type=Path, required=True)
    parser.add_argument("--new-classifier", type=Path, required=True)
    parser.add_argument("--training-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument("--absolute-area-tolerance", type=float, default=10.0)
    parser.add_argument("--relative-area-tolerance", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_by_sample = collect_json_by_stem(args.graph_dir)
    prediction_by_sample = collect_json_by_stem(args.raw_prediction_dir)
    truth_by_sample = normalize_truth_entries(load_json(args.truth_json))
    old_bundle = load_instance_candidate_classifier_bundle(args.old_classifier)
    new_bundle = load_instance_candidate_classifier_bundle(args.new_classifier)

    comparisons = []
    for sample in sorted(graph_by_sample):
        if sample not in prediction_by_sample:
            continue
        comparisons.extend(
            build_sample_comparisons(
                sample,
                graph_by_sample[sample],
                prediction_by_sample[sample],
                truth_by_sample.get(sample, []),
                old_bundle,
                new_bundle,
                min_confidence=args.min_confidence,
                absolute_area_tolerance=args.absolute_area_tolerance,
                relative_area_tolerance=args.relative_area_tolerance,
            )
        )

    report = {
        "schema": "candidate_classifier_error_atlas_v1",
        "old_classifier": str(args.old_classifier),
        "new_classifier": str(args.new_classifier),
        "summary": summarize_comparisons(comparisons),
        "candidates": comparisons,
    }
    audit = audit_training_records(load_jsonl(args.training_records))

    output_dir = args.output_dir
    write_json(output_dir / "classifier_error_atlas_s3.json", report)
    write_json(output_dir / "training_label_audit_v5_14.json", audit)
    write_error_atlas_markdown(output_dir / "classifier_error_atlas_s3.md", report)
    write_training_audit_markdown(output_dir / "training_label_audit_v5_14.md", audit, comparisons)
    write_decision(output_dir / "v5_15_decision.md", report, audit)

    print(json.dumps({
        "output_dir": str(output_dir),
        "candidate_count": report["summary"]["candidate_count"],
        "matched_candidate_count": report["summary"]["matched_candidate_count"],
        "decision_path": str(output_dir / "v5_15_decision.md"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
