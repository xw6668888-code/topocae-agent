from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

from utils.area_truth_normalization import normalize_truth_macro

MERGE_EQUIVALENT_TRUTH_MACROS = {"slot_o"}
MAX_MERGED_TRUTH_OBJECTS = 6


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def collect_feature_object_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    globber = path.rglob if recursive else path.glob
    return sorted(p for p in globber("*.json") if p.name != "feature_object_manifest.json")


def normalize_macro(value: str) -> str:
    return normalize_truth_macro(value)


def feature_area(feature_object: dict[str, Any]) -> float:
    fingerprint = feature_object.get("fingerprint", {})
    if "area_total" in fingerprint:
        return float(fingerprint["area_total"])
    return float(feature_object.get("area_total", 0.0) or 0.0)


def area_error(predicted_area: float, truth_area: float) -> dict[str, float]:
    abs_error = abs(predicted_area - truth_area)
    rel_error = abs_error / max(abs(truth_area), 1e-9)
    return {"absolute": abs_error, "relative": rel_error}


def within_tolerance(predicted_area: float, truth_area: float, abs_tol: float, rel_tol: float) -> bool:
    error = area_error(predicted_area, truth_area)
    return error["absolute"] <= abs_tol or error["relative"] <= rel_tol


def predicted_summary(feature_object: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "index": index,
        "feature_object_id": feature_object.get("feature_object_id", f"feature_{index:03d}"),
        "macro": normalize_macro(str(feature_object.get("macro_feature", "unknown"))),
        "face_count": int(feature_object.get("face_count", len(feature_object.get("face_indices", [])))),
        "face_indices": list(feature_object.get("face_indices", [])),
        "area": feature_area(feature_object),
        "confidence": feature_object.get("confidence", {}),
    }


def truth_summary(raw_truth: dict[str, Any], index: int, sample_id: str, truth_source_name: str) -> dict[str, Any]:
    area = float(raw_truth["area"])
    face_count = int(raw_truth["face_count"])
    return {
        "index": index,
        "macro": normalize_truth_macro(
            str(raw_truth["macro"]),
            sample_id=sample_id,
            area=area,
            face_count=face_count,
            truth_source_name=truth_source_name,
        ),
        "face_count": face_count,
        "area": area,
    }


def best_match(
    prediction: dict[str, Any],
    unmatched_truth: list[dict[str, Any]],
    abs_tol: float,
    rel_tol: float,
) -> tuple[int | None, dict[str, float] | None]:
    candidates: list[tuple[float, int, dict[str, float]]] = []
    for idx, truth in enumerate(unmatched_truth):
        if prediction["macro"] != truth["macro"]:
            continue
        if prediction["face_count"] != truth["face_count"]:
            continue
        if not within_tolerance(prediction["area"], truth["area"], abs_tol, rel_tol):
            continue
        error = area_error(prediction["area"], truth["area"])
        candidates.append((error["relative"], idx, error))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: (item[0], item[2]["absolute"]))
    _, idx, error = candidates[0]
    return idx, error


def merged_truth_summary(truth_entries: list[dict[str, Any]]) -> dict[str, Any]:
    macro = truth_entries[0]["macro"]
    area = sum(float(truth["area"]) for truth in truth_entries)
    face_count = sum(int(truth["face_count"]) for truth in truth_entries)
    return {
        "index": "+".join(str(truth["index"]) for truth in truth_entries),
        "macro": macro,
        "face_count": face_count,
        "area": area,
        "merged_truth_count": len(truth_entries),
        "merged_truth_entries": truth_entries,
    }


def best_merged_truth_match(
    prediction: dict[str, Any],
    unmatched_truth: list[dict[str, Any]],
    abs_tol: float,
    rel_tol: float,
) -> tuple[list[int] | None, dict[str, Any] | None, dict[str, float] | None]:
    if prediction["macro"] not in MERGE_EQUIVALENT_TRUTH_MACROS:
        return None, None, None

    matching_indices = [
        idx for idx, truth in enumerate(unmatched_truth)
        if truth["macro"] == prediction["macro"]
    ]
    if len(matching_indices) < 2:
        return None, None, None

    candidates: list[tuple[float, float, int, tuple[int, ...], dict[str, Any], dict[str, float]]] = []
    max_group_size = min(MAX_MERGED_TRUTH_OBJECTS, len(matching_indices))
    for group_size in range(2, max_group_size + 1):
        for index_group in combinations(matching_indices, group_size):
            truth_group = [unmatched_truth[idx] for idx in index_group]
            summary = merged_truth_summary(truth_group)
            if prediction["face_count"] != summary["face_count"]:
                continue
            if not within_tolerance(prediction["area"], summary["area"], abs_tol, rel_tol):
                continue
            error = area_error(prediction["area"], summary["area"])
            candidates.append((error["relative"], error["absolute"], group_size, index_group, summary, error))

    if not candidates:
        return None, None, None

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    _, _, _, index_group, summary, error = candidates[0]
    return list(index_group), summary, error


def matched_truth_count(match: dict[str, Any]) -> int:
    return int(match["truth"].get("merged_truth_count", 1))


def matched_macro_counter(matches: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for match in matches:
        counter.update({match["truth"]["macro"]: matched_truth_count(match)})
    return counter


def evaluate_sample(
    sample_id: str,
    feature_doc: dict[str, Any],
    truth_entries: list[dict[str, Any]],
    abs_tol: float,
    rel_tol: float,
) -> dict[str, Any]:
    predictions = [
        predicted_summary(feature_object, index)
        for index, feature_object in enumerate(feature_doc.get("feature_objects", []))
    ]
    truth_source_name = str(feature_doc.get("truth_source_name", ""))
    unmatched_truth = [
        truth_summary(entry, index, sample_id, truth_source_name)
        for index, entry in enumerate(truth_entries)
    ]
    matches: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []

    for prediction in predictions:
        match_index, error = best_match(prediction, unmatched_truth, abs_tol, rel_tol)
        if match_index is not None:
            truth = unmatched_truth.pop(match_index)
            matches.append({
                "prediction": prediction,
                "truth": truth,
                "area_error": error,
                "match_type": "one_to_one",
            })
            continue

        merged_indices, merged_truth, merged_error = best_merged_truth_match(
            prediction, unmatched_truth, abs_tol, rel_tol
        )
        if merged_indices is None or merged_truth is None:
            false_positives.append(prediction)
            continue

        for idx in sorted(merged_indices, reverse=True):
            unmatched_truth.pop(idx)
        matches.append({
            "prediction": prediction,
            "truth": merged_truth,
            "area_error": merged_error,
            "match_type": "merged_truth",
        })

    truth_counter = Counter(
        truth["macro"]
        for truth in [
            truth_summary(entry, i, sample_id, truth_source_name)
            for i, entry in enumerate(truth_entries)
        ]
    )
    pred_counter = Counter(prediction["macro"] for prediction in predictions)
    matched_counter = matched_macro_counter(matches)
    matched_truth_total = sum(matched_truth_count(match) for match in matches)
    matched_prediction_total = len(matches)

    return {
        "sample_id": sample_id,
        "source_step_path": feature_doc.get("source_step_path"),
        "truth_count": len(truth_entries),
        "prediction_count": len(predictions),
        "matched_count": matched_truth_total,
        "matched_truth_count": matched_truth_total,
        "matched_prediction_count": matched_prediction_total,
        "recall": matched_truth_total / len(truth_entries) if truth_entries else 1.0,
        "precision": matched_prediction_total / len(predictions) if predictions else 0.0,
        "truth_macro_counts": dict(sorted(truth_counter.items())),
        "predicted_macro_counts": dict(sorted(pred_counter.items())),
        "matched_macro_counts": dict(sorted(matched_counter.items())),
        "matches": matches,
        "false_positives": false_positives,
        "missed_truth": unmatched_truth,
    }


def build_review(
    feature_objects_path: Path,
    truth_path: Path,
    recursive: bool,
    abs_tol: float,
    rel_tol: float,
) -> dict[str, Any]:
    truth_doc = load_json(truth_path)
    truth_by_sample = truth_doc.get("samples", {})
    sample_reports = []
    for path in collect_feature_object_files(feature_objects_path, recursive):
        feature_doc = load_json(path)
        sample_id = str(feature_doc.get("sample_id", path.stem))
        if sample_id not in truth_by_sample:
            continue
        sample_reports.append(
            evaluate_sample(sample_id, feature_doc, truth_by_sample[sample_id], abs_tol, rel_tol)
        )

    if not sample_reports:
        raise RuntimeError(f"No feature-object files matched truth samples in {truth_path}")

    truth_total = sum(sample["truth_count"] for sample in sample_reports)
    pred_total = sum(sample["prediction_count"] for sample in sample_reports)
    matched_truth_total = sum(sample["matched_truth_count"] for sample in sample_reports)
    matched_prediction_total = sum(sample["matched_prediction_count"] for sample in sample_reports)
    truth_counter: Counter[str] = Counter()
    pred_counter: Counter[str] = Counter()
    matched_counter: Counter[str] = Counter()
    for sample in sample_reports:
        truth_counter.update(sample["truth_macro_counts"])
        pred_counter.update(sample["predicted_macro_counts"])
        matched_counter.update(sample["matched_macro_counts"])

    return {
        "schema": "cae_area_truth_review_v1",
        "feature_objects_path": str(feature_objects_path.resolve()),
        "truth_path": str(truth_path.resolve()),
        "absolute_area_tolerance": abs_tol,
        "relative_area_tolerance": rel_tol,
        "sample_count": len(sample_reports),
        "truth_count": truth_total,
        "prediction_count": pred_total,
        "matched_count": matched_truth_total,
        "matched_truth_count": matched_truth_total,
        "matched_prediction_count": matched_prediction_total,
        "recall": matched_truth_total / truth_total if truth_total else 1.0,
        "precision": matched_prediction_total / pred_total if pred_total else 0.0,
        "truth_macro_counts": dict(sorted(truth_counter.items())),
        "predicted_macro_counts": dict(sorted(pred_counter.items())),
        "matched_macro_counts": dict(sorted(matched_counter.items())),
        "samples": sample_reports,
    }


def format_macro_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "{}"
    return ", ".join(f"`{macro}`: {count}" for macro, count in counts.items())


def write_markdown(path: Path, review: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# v4.4 Area Truth Baseline Review",
        "",
        f"Feature objects: `{review['feature_objects_path']}`",
        f"Truth table: `{review['truth_path']}`",
        f"Area tolerance: absolute `{review['absolute_area_tolerance']}`, relative `{review['relative_area_tolerance']:.4f}`",
        "",
        "## Overall",
        "",
        f"- Samples: `{review['sample_count']}`",
        f"- Truth feature objects: `{review['truth_count']}`",
        f"- Predicted feature objects: `{review['prediction_count']}`",
        f"- Matched truth feature objects: `{review['matched_truth_count']}`",
        f"- Matched predicted feature objects: `{review['matched_prediction_count']}`",
        f"- Area-truth recall: `{review['recall']:.4f}`",
        f"- Area-truth precision: `{review['precision']:.4f}`",
        f"- Truth macro counts: {format_macro_counts(review['truth_macro_counts'])}",
        f"- Predicted macro counts: {format_macro_counts(review['predicted_macro_counts'])}",
        f"- Matched macro counts: {format_macro_counts(review['matched_macro_counts'])}",
        "",
    ]

    for sample in review["samples"]:
        lines.extend(
            [
                f"## {sample['sample_id']}",
                "",
                f"- STEP: `{sample.get('source_step_path')}`",
                f"- Truth / predicted / matched truth / matched predictions: `{sample['truth_count']}` / `{sample['prediction_count']}` / `{sample['matched_truth_count']}` / `{sample['matched_prediction_count']}`",
                f"- Recall / precision: `{sample['recall']:.4f}` / `{sample['precision']:.4f}`",
                f"- Truth macro counts: {format_macro_counts(sample['truth_macro_counts'])}",
                f"- Predicted macro counts: {format_macro_counts(sample['predicted_macro_counts'])}",
                "",
                "### Matches",
                "",
            ]
        )
        if sample["matches"]:
            lines.append("| Match type | Truth macro | Truth faces | Truth area | Pred macro | Pred faces | Pred area | Face ids | Relative error |")
            lines.append("|---|---|---:|---:|---|---:|---:|---|---:|")
            for match in sample["matches"]:
                truth = match["truth"]
                prediction = match["prediction"]
                lines.append(
                    "| "
                    f"`{match.get('match_type', 'one_to_one')}` | "
                    f"`{truth['macro']}` | {truth['face_count']} | {truth['area']:.4f} | "
                    f"`{prediction['macro']}` | {prediction['face_count']} | {prediction['area']:.4f} | "
                    f"`{prediction['face_indices']}` | {match['area_error']['relative']:.6f} |"
                )
        else:
            lines.append("No matched feature objects.")

        lines.extend(["", "### Missed Truth", ""])
        if sample["missed_truth"]:
            lines.append("| Macro | Faces | Area |")
            lines.append("|---|---:|---:|")
            for truth in sample["missed_truth"]:
                lines.append(f"| `{truth['macro']}` | {truth['face_count']} | {truth['area']:.4f} |")
        else:
            lines.append("No missed truth entries.")

        lines.extend(["", "### False Positives", ""])
        if sample["false_positives"]:
            lines.append("| Macro | Faces | Area | Face ids |")
            lines.append("|---|---:|---:|---|")
            for prediction in sample["false_positives"]:
                lines.append(
                    f"| `{prediction['macro']}` | {prediction['face_count']} | "
                    f"{prediction['area']:.4f} | `{prediction['face_indices']}` |"
                )
        else:
            lines.append("No false-positive predictions.")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare predicted feature objects with area-based manual truth.")
    parser.add_argument("--feature-objects-path", required=True, type=Path)
    parser.add_argument("--truth-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--absolute-area-tolerance", type=float, default=5.0)
    parser.add_argument("--relative-area-tolerance", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review = build_review(
        args.feature_objects_path,
        args.truth_json,
        args.recursive,
        args.absolute_area_tolerance,
        args.relative_area_tolerance,
    )
    write_json(args.output_json, review)
    write_markdown(args.output_md, review)
    print(f"area truth review json written: {args.output_json}")
    print(f"area truth review markdown written: {args.output_md}")


if __name__ == "__main__":
    main()



