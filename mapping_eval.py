from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Abaqus face mapping outputs with optional reference mapping and action plan files."
    )
    parser.add_argument("--mapping-path", type=Path, required=True)
    parser.add_argument("--reference-mapping-path", type=Path, default=None)
    parser.add_argument("--action-plan-path", type=Path, default=None)
    parser.add_argument("--reference-action-plan-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def collect_mapping_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    globber = path.rglob if recursive else path.glob
    return sorted(
        item for item in globber("*.json")
        if item.is_file() and item.name != "abaqus_mapping_manifest.json"
    )


def resolve_matching_file(sample_file: Path, root: Path | None) -> Path | None:
    if root is None:
        return None
    if root.is_file():
        return root
    candidate = root / sample_file.name
    if candidate.exists():
        return candidate
    matches = list(root.rglob(sample_file.name))
    if len(matches) == 1:
        return matches[0]
    return None


def set_metrics(predicted: list[int], reference: list[int]) -> dict:
    pred = set(int(value) for value in predicted)
    ref = set(int(value) for value in reference)
    intersection = len(pred & ref)
    union = len(pred | ref)
    precision = intersection / len(pred) if pred else 0.0
    recall = intersection / len(ref) if ref else 0.0
    iou = intersection / union if union else 0.0
    exact = pred == ref
    return {
        "pred_count": len(pred),
        "ref_count": len(ref),
        "intersection": intersection,
        "union": union,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "exact_match": exact,
    }


def extract_faces_from_action_plan(action_plan: dict, op_type: str) -> list[int]:
    faces = []
    for op in action_plan.get("operations", []):
        if op.get("type") != op_type:
            continue
        faces.extend(int(face_id) for face_id in op.get("faces", []))
    return sorted(set(faces))


def evaluate_structure(mapping_data: dict) -> dict:
    feature_mappings = mapping_data.get("feature_mappings", [])
    status_counts = {}
    mapped_face_ids = []
    source_face_count = 0
    scores = []

    for mapping in feature_mappings:
        status = mapping.get("mapping_status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        source_face_count += len(mapping.get("source_face_indices", []))
        mapped_face_ids.extend(int(face_id) for face_id in mapping.get("mapped_abaqus_face_ids", []))
        scores.append(float(mapping.get("mapping_score_mean", 0.0)))

    unique_mapped = sorted(set(mapped_face_ids))
    return {
        "feature_mapping_count": len(feature_mappings),
        "source_face_count": source_face_count,
        "mapped_face_count": len(mapped_face_ids),
        "unique_mapped_face_count": len(unique_mapped),
        "reused_mapped_face_count": len(mapped_face_ids) - len(unique_mapped),
        "mean_mapping_score": sum(scores) / len(scores) if scores else 0.0,
        "status_counts": status_counts,
        "feature_region_count": len(mapping_data.get("suggested_abaqus_regions", {}).get("feature_object_regions", [])),
        "stock_region_count": len(mapping_data.get("suggested_abaqus_regions", {}).get("stock_planar_regions", [])),
    }


def evaluate_against_reference(mapping_data: dict, reference_data: dict) -> dict:
    pred_map = {item["feature_object_id"]: item for item in mapping_data.get("feature_mappings", [])}
    ref_map = {item["feature_object_id"]: item for item in reference_data.get("feature_mappings", [])}

    shared_ids = sorted(set(pred_map) & set(ref_map))
    missing_prediction = sorted(set(ref_map) - set(pred_map))
    extra_prediction = sorted(set(pred_map) - set(ref_map))

    per_feature = []
    for feature_object_id in shared_ids:
        metrics = set_metrics(
            pred_map[feature_object_id].get("mapped_abaqus_face_ids", []),
            ref_map[feature_object_id].get("mapped_abaqus_face_ids", []),
        )
        metrics["feature_object_id"] = feature_object_id
        metrics["macro_feature"] = pred_map[feature_object_id].get("macro_feature")
        per_feature.append(metrics)

    exact_ratio = (
        sum(1 for item in per_feature if item["exact_match"]) / len(per_feature)
        if per_feature else 0.0
    )
    mean_iou = sum(item["iou"] for item in per_feature) / len(per_feature) if per_feature else 0.0
    mean_precision = sum(item["precision"] for item in per_feature) / len(per_feature) if per_feature else 0.0
    mean_recall = sum(item["recall"] for item in per_feature) / len(per_feature) if per_feature else 0.0

    return {
        "matched_feature_count": len(shared_ids),
        "missing_prediction_count": len(missing_prediction),
        "extra_prediction_count": len(extra_prediction),
        "exact_feature_ratio": exact_ratio,
        "mean_feature_iou": mean_iou,
        "mean_feature_precision": mean_precision,
        "mean_feature_recall": mean_recall,
        "missing_prediction_feature_ids": missing_prediction,
        "extra_prediction_feature_ids": extra_prediction,
        "per_feature": per_feature,
    }


def evaluate_action_plan(action_plan: dict, reference_action_plan: dict) -> dict:
    bc_pred = extract_faces_from_action_plan(action_plan, "BoundaryCondition")
    bc_ref = extract_faces_from_action_plan(reference_action_plan, "BoundaryCondition")
    load_pred = extract_faces_from_action_plan(action_plan, "Load")
    load_ref = extract_faces_from_action_plan(reference_action_plan, "Load")
    return {
        "boundary_condition": set_metrics(bc_pred, bc_ref),
        "load": set_metrics(load_pred, load_ref),
    }


def evaluate_one_sample(
    mapping_file: Path,
    reference_mapping_root: Path | None,
    action_plan_root: Path | None,
    reference_action_plan_root: Path | None,
) -> dict:
    mapping_data = load_json(mapping_file)
    report = {
        "sample_id": mapping_data.get("sample_id", mapping_file.stem),
        "mapping_path": str(mapping_file.resolve()),
        "structure": evaluate_structure(mapping_data),
    }

    reference_mapping_file = resolve_matching_file(mapping_file, reference_mapping_root)
    if reference_mapping_file is not None:
        report["reference_mapping_path"] = str(reference_mapping_file.resolve())
        report["reference_mapping_eval"] = evaluate_against_reference(
            mapping_data,
            load_json(reference_mapping_file),
        )

    action_plan_file = resolve_matching_file(mapping_file, action_plan_root)
    reference_action_plan_file = resolve_matching_file(mapping_file, reference_action_plan_root)
    if action_plan_file is not None and reference_action_plan_file is not None:
        report["action_plan_path"] = str(action_plan_file.resolve())
        report["reference_action_plan_path"] = str(reference_action_plan_file.resolve())
        report["action_plan_eval"] = evaluate_action_plan(
            load_json(action_plan_file),
            load_json(reference_action_plan_file),
        )

    return report


def aggregate_reports(sample_reports: list[dict]) -> dict:
    sample_count = len(sample_reports)
    if sample_count == 0:
        return {"sample_count": 0, "samples": []}

    mean_mapping_score = sum(
        report["structure"]["mean_mapping_score"]
        for report in sample_reports
    ) / sample_count

    total_status_counts = {}
    for report in sample_reports:
        for status, count in report["structure"]["status_counts"].items():
            total_status_counts[status] = total_status_counts.get(status, 0) + int(count)

    summary = {
        "sample_count": sample_count,
        "mean_mapping_score": mean_mapping_score,
        "total_status_counts": total_status_counts,
        "samples": sample_reports,
    }

    reference_reports = [item["reference_mapping_eval"] for item in sample_reports if "reference_mapping_eval" in item]
    if reference_reports:
        summary["reference_mapping_summary"] = {
            "evaluated_sample_count": len(reference_reports),
            "mean_exact_feature_ratio": sum(item["exact_feature_ratio"] for item in reference_reports) / len(reference_reports),
            "mean_feature_iou": sum(item["mean_feature_iou"] for item in reference_reports) / len(reference_reports),
            "mean_feature_precision": sum(item["mean_feature_precision"] for item in reference_reports) / len(reference_reports),
            "mean_feature_recall": sum(item["mean_feature_recall"] for item in reference_reports) / len(reference_reports),
        }

    action_reports = [item["action_plan_eval"] for item in sample_reports if "action_plan_eval" in item]
    if action_reports:
        summary["action_plan_summary"] = {
            "evaluated_sample_count": len(action_reports),
            "mean_bc_iou": sum(item["boundary_condition"]["iou"] for item in action_reports) / len(action_reports),
            "mean_load_iou": sum(item["load"]["iou"] for item in action_reports) / len(action_reports),
        }

    return summary


def main() -> None:
    args = parse_args()
    mapping_files = collect_mapping_files(args.mapping_path, args.recursive)
    if not mapping_files:
        raise RuntimeError(f"No mapping files found under: {args.mapping_path}")

    sample_reports = [
        evaluate_one_sample(
            mapping_file=mapping_file,
            reference_mapping_root=args.reference_mapping_path,
            action_plan_root=args.action_plan_path,
            reference_action_plan_root=args.reference_action_plan_path,
        )
        for mapping_file in mapping_files
    ]

    report = aggregate_reports(sample_reports)
    if args.output_path is not None:
        write_json(args.output_path, report)
        print(f"mapping evaluation written: {args.output_path}")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
