from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_rib_boundary_classifier_dataset import (
    NOT_RIB_MEMBER_LABEL,
    RIB_MEMBER_LABEL,
    adjacent_faces,
    candidate_to_seed_edges,
    make_record,
    split_rib_boundary_records,
    write_rib_boundary_dataset,
)


RIB_MACROS = {"rib", "web", "rib_web_candidate"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_rib_face_truth_template(
    *,
    area_review: dict[str, Any],
    graph_dir: Path,
    prediction_dir: Path,
    feature_objects_dir: Path,
    sample_ids: list[str] | None = None,
    high_aspect_threshold: float = 4.0,
    max_candidates_per_sample: int = 80,
) -> dict[str, Any]:
    selected_sample_ids = set(sample_ids or [])
    samples = []
    for sample in area_review.get("samples", []):
        sample_id = str(sample.get("sample_id", ""))
        if not sample_id:
            continue
        if selected_sample_ids and sample_id not in selected_sample_ids:
            continue
        if not selected_sample_ids and not sample_has_rib_residual(sample):
            continue
        graph_path = graph_dir / f"{sample_id}.json"
        prediction_path = prediction_dir / f"{sample_id}.json"
        feature_path = feature_objects_dir / f"{sample_id}.json"
        if not graph_path.exists() or not prediction_path.exists():
            continue
        graph_data = load_json(graph_path)
        prediction_data = load_json(prediction_path)
        feature_data = load_json(feature_path) if feature_path.exists() else {}
        samples.append(
            build_sample_template(
                sample_id=sample_id,
                area_sample=sample,
                graph_data=graph_data,
                prediction_data=prediction_data,
                feature_data=feature_data,
                graph_path=graph_path,
                prediction_path=prediction_path,
                feature_path=feature_path,
                high_aspect_threshold=high_aspect_threshold,
                max_candidates=max_candidates_per_sample,
            )
        )
    return {
        "schema": "rib_face_truth_template_v1",
        "instructions": [
            "Only edit the review fields under candidate_faces.",
            "Set review.is_rib_face to true for faces that belong to a rib/web target instance.",
            "Use the same review.rib_instance_id for faces belonging to the same rib instance, for example rib_1.",
            "Set review.is_rib_face to false for reviewed nearby non-rib faces.",
            "Leave uncertain faces as null and add review.notes if needed.",
        ],
        "source": {
            "area_review_schema": area_review.get("schema"),
            "graph_dir": str(graph_dir),
            "prediction_dir": str(prediction_dir),
            "feature_objects_dir": str(feature_objects_dir),
        },
        "sample_count": len(samples),
        "samples": samples,
    }


def sample_has_rib_residual(sample: dict[str, Any]) -> bool:
    for key in ("missed_truth", "false_positives"):
        for item in sample.get(key, []) or []:
            if str(item.get("macro")) in RIB_MACROS:
                return True
    return False


def build_sample_template(
    *,
    sample_id: str,
    area_sample: dict[str, Any],
    graph_data: dict[str, Any],
    prediction_data: dict[str, Any],
    feature_data: dict[str, Any],
    graph_path: Path,
    prediction_path: Path,
    feature_path: Path,
    high_aspect_threshold: float,
    max_candidates: int,
) -> dict[str, Any]:
    reasons = collect_rib_candidate_reasons(
        graph_data=graph_data,
        prediction_data=prediction_data,
        feature_data=feature_data,
        high_aspect_threshold=high_aspect_threshold,
    )
    ordered_face_indices = sorted(
        reasons,
        key=lambda idx: (
            candidate_priority(reasons[idx]),
            idx,
        ),
    )[:max_candidates]
    candidate_faces = [
        describe_candidate_face(
            face_idx=face_idx,
            reasons=sorted(reasons[face_idx]),
            graph_data=graph_data,
            prediction_data=prediction_data,
        )
        for face_idx in ordered_face_indices
    ]
    return {
        "sample_id": sample_id,
        "source_step_path": area_sample.get("source_step_path") or feature_data.get("source_step_path"),
        "graph_path": str(graph_path),
        "prediction_path": str(prediction_path),
        "feature_objects_path": str(feature_path) if feature_path.exists() else "",
        "rib_truth_rows": [
            item for item in area_sample.get("missed_truth", []) + area_sample.get("matches", [])
            if str(item.get("macro") or item.get("truth", {}).get("macro")) in RIB_MACROS
        ],
        "rib_false_positive_rows": [
            item for item in area_sample.get("false_positives", [])
            if str(item.get("macro")) in RIB_MACROS
        ],
        "candidate_face_count": len(candidate_faces),
        "candidate_faces": candidate_faces,
    }


def collect_rib_candidate_reasons(
    *,
    graph_data: dict[str, Any],
    prediction_data: dict[str, Any],
    feature_data: dict[str, Any],
    high_aspect_threshold: float,
) -> dict[int, set[str]]:
    reasons: dict[int, set[str]] = defaultdict(set)
    face_by_idx = prediction_face_by_idx(prediction_data)
    for face_idx, face in face_by_idx.items():
        if str(face.get("macro_feature")) in RIB_MACROS:
            reasons[face_idx].add("predicted_rib_face")
        if str(face.get("raw_macro_feature")) in RIB_MACROS:
            reasons[face_idx].add("raw_rib_face")

    for feature_object in feature_data.get("feature_objects", []) or []:
        if str(feature_object.get("macro_feature")) not in RIB_MACROS:
            continue
        for face_idx in feature_object.get("face_indices", []) or []:
            reasons[int(face_idx)].add("feature_object_rib_face")

    for record in prediction_data.get("rib_boundary_classifier_postprocess", {}).get("decision_records", []) or []:
        candidate_face_idx = record.get("candidate_face_idx")
        if candidate_face_idx is not None:
            reasons[int(candidate_face_idx)].add("rib_boundary_decision_face")
        for seed_face_idx in record.get("seed_face_indices", []) or []:
            reasons[int(seed_face_idx)].add("rib_boundary_seed_face")

    for record in prediction_data.get("topology_postprocess", {}).get("repaired_components", []) or []:
        if str(record.get("to_macro_feature")) not in RIB_MACROS and str(record.get("from_macro_feature")) not in RIB_MACROS:
            continue
        for face_idx in record.get("face_indices", []) or []:
            reasons[int(face_idx)].add("topology_rib_repair_face")

    rib_seed_faces = {idx for idx, item_reasons in reasons.items() if any("rib" in reason for reason in item_reasons)}
    for neighbor in sorted(adjacent_faces(graph_data, rib_seed_faces)) if rib_seed_faces else []:
        reasons[int(neighbor)].add("adjacent_to_rib_candidate")

    for node in graph_data.get("nodes", []) or []:
        face_idx = int(node.get("idx", 0))
        if node.get("primitive_surface") != "plane":
            continue
        descriptors = node.get("attributes", {}).get("shape_descriptors", {}) or {}
        projected_aspect = float(descriptors.get("projected_aspect_ratio", 0.0) or 0.0)
        bbox_aspect = float(descriptors.get("bbox_aspect_ratio", 0.0) or 0.0)
        if max(projected_aspect, bbox_aspect) >= high_aspect_threshold:
            reasons[face_idx].add("high_aspect_plane_candidate")

    return reasons


def candidate_priority(reasons: set[str]) -> tuple[int, int, int]:
    priority_order = [
        "predicted_rib_face",
        "feature_object_rib_face",
        "rib_boundary_decision_face",
        "topology_rib_repair_face",
        "rib_boundary_seed_face",
        "adjacent_to_rib_candidate",
        "high_aspect_plane_candidate",
    ]
    scores = [priority_order.index(reason) for reason in reasons if reason in priority_order]
    return (min(scores) if scores else 99, -len(reasons), 0)


def describe_candidate_face(
    *,
    face_idx: int,
    reasons: list[str],
    graph_data: dict[str, Any],
    prediction_data: dict[str, Any],
) -> dict[str, Any]:
    node = node_by_idx(graph_data).get(face_idx, {})
    physical = node.get("attributes", {}).get("physical", {}) or {}
    specific = node.get("attributes", {}).get("specific", {}) or {}
    descriptors = node.get("attributes", {}).get("shape_descriptors", {}) or {}
    pred_face = prediction_face_by_idx(prediction_data).get(face_idx, {})
    return {
        "face_idx": int(face_idx),
        "candidate_reasons": reasons,
        "primitive_surface": node.get("primitive_surface", pred_face.get("primitive_surface")),
        "area": round_float(physical.get("area", descriptors.get("area", 0.0))),
        "center": round_list(physical.get("center_of_mass", [])),
        "normal": round_list(specific.get("normal", [])),
        "axis": round_list(specific.get("axis", [])),
        "radius": round_float(specific.get("radius", 0.0)),
        "shape_descriptors": {
            key: round_float(descriptors.get(key, 0.0))
            for key in [
                "projected_aspect_ratio",
                "bbox_aspect_ratio",
                "bbox_longest_extent",
                "bbox_shortest_extent",
                "neighbor_count",
                "neighbor_area_max",
            ]
            if key in descriptors
        },
        "current_prediction": {
            "macro_feature": pred_face.get("macro_feature", ""),
            "semantic_subtype": pred_face.get("semantic_subtype", ""),
            "macro_confidence": round_float(pred_face.get("macro_confidence", 0.0)),
            "raw_macro_feature": pred_face.get("raw_macro_feature", ""),
            "raw_semantic_subtype": pred_face.get("raw_semantic_subtype", ""),
        },
        "adjacent_faces": sorted(adjacent_faces(graph_data, {int(face_idx)})),
        "review": {
            "is_rib_face": None,
            "rib_instance_id": "",
            "notes": "",
        },
    }


def build_rib_boundary_records_from_review_template(template: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[int, ...], int]] = set()
    for sample in template.get("samples", []) or []:
        sample_id = str(sample.get("sample_id", ""))
        graph_path = Path(sample.get("graph_path", ""))
        if not graph_path.exists():
            continue
        graph_data = load_json(graph_path)
        reviewed_faces = reviewed_face_labels(sample)
        instances: dict[str, list[int]] = defaultdict(list)
        negative_faces = []
        for face_idx, review in reviewed_faces.items():
            if review.get("is_rib_face") is True:
                instance_id = str(review.get("rib_instance_id") or "rib_1")
                instances[instance_id].append(face_idx)
            elif review.get("is_rib_face") is False:
                negative_faces.append(face_idx)

        for instance_id, rib_faces in sorted(instances.items()):
            rib_faces = sorted(set(rib_faces))
            if len(rib_faces) < 2:
                continue
            instance = {"macro_feature": "rib", "instance_id": instance_id, "face_indices": rib_faces}
            rib_set = set(rib_faces)
            for candidate_face_idx in rib_faces:
                seed_faces = sorted(rib_set - {candidate_face_idx})
                add_review_record(
                    records=records,
                    seen=seen,
                    sample_id=sample_id,
                    graph_path=graph_path,
                    graph_data=graph_data,
                    instance=instance,
                    label=RIB_MEMBER_LABEL,
                    seed_face_indices=seed_faces,
                    candidate_face_idx=candidate_face_idx,
                )
            for candidate_face_idx in sorted(set(negative_faces) | (adjacent_faces(graph_data, rib_set) - rib_set)):
                if candidate_face_idx in rib_set:
                    continue
                if not candidate_to_seed_edges(graph_data, candidate_face_idx, rib_set):
                    continue
                add_review_record(
                    records=records,
                    seen=seen,
                    sample_id=sample_id,
                    graph_path=graph_path,
                    graph_data=graph_data,
                    instance=instance,
                    label=NOT_RIB_MEMBER_LABEL,
                    seed_face_indices=rib_faces,
                    candidate_face_idx=candidate_face_idx,
                )
    records.sort(key=lambda item: (item["sample_id"], item["label"], item["seed_face_indices"], item["candidate_face_idx"]))
    return records


def reviewed_face_labels(sample: dict[str, Any]) -> dict[int, dict[str, Any]]:
    labels = {}
    for candidate in sample.get("candidate_faces", []) or []:
        review = candidate.get("review", {}) or {}
        if review.get("is_rib_face") is None:
            continue
        labels[int(candidate["face_idx"])] = review
    return labels


def add_review_record(
    *,
    records: list[dict[str, Any]],
    seen: set[tuple[str, str, tuple[int, ...], int]],
    sample_id: str,
    graph_path: Path,
    graph_data: dict[str, Any],
    instance: dict[str, Any],
    label: str,
    seed_face_indices: list[int],
    candidate_face_idx: int,
) -> None:
    key = (sample_id, label, tuple(seed_face_indices), int(candidate_face_idx))
    if key in seen:
        return
    seen.add(key)
    record = make_record(
        sample_id=sample_id,
        graph_path=graph_path,
        source="reviewed_rib_face_truth",
        label=label,
        seed_face_indices=seed_face_indices,
        candidate_face_idx=int(candidate_face_idx),
        graph_data=graph_data,
        instance=instance,
    )
    records.append(record)


def write_markdown(path: Path, template: dict[str, Any]) -> None:
    lines = [
        "# Rib Face-Level Truth Template",
        "",
        "Edit the JSON review fields for actual training. This Markdown is for visual checking.",
        "",
    ]
    for sample in template.get("samples", []):
        lines.extend([
            f"## {sample['sample_id']}",
            "",
            f"STEP: `{sample.get('source_step_path')}`",
            "",
            "### Rib Truth Rows",
            "",
        ])
        truth_rows = sample.get("rib_truth_rows", [])
        if truth_rows:
            lines.append("| Macro | Faces | Area |")
            lines.append("|---|---:|---:|")
            for row in truth_rows:
                macro = row.get("macro") or row.get("truth", {}).get("macro")
                face_count = row.get("face_count") or row.get("truth", {}).get("face_count", "")
                area = row.get("area") or row.get("truth", {}).get("area", 0.0)
                lines.append(f"| `{macro}` | {face_count} | {float(area):.4f} |")
        else:
            lines.append("No rib truth rows in area review.")
        lines.extend(["", "### Candidate Faces", ""])
        lines.append("| Face | Reasons | Primitive | Area | Prediction | Review is_rib_face | Review instance |")
        lines.append("|---:|---|---|---:|---|---|---|")
        for face in sample.get("candidate_faces", []):
            prediction = face.get("current_prediction", {})
            lines.append(
                f"| {face['face_idx']} | `{face.get('candidate_reasons', [])}` | "
                f"`{face.get('primitive_surface')}` | {float(face.get('area', 0.0)):.4f} | "
                f"`{prediction.get('macro_feature', '')}/{prediction.get('semantic_subtype', '')}` | "
                f"{face.get('review', {}).get('is_rib_face')} | `{face.get('review', {}).get('rib_instance_id', '')}` |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def node_by_idx(graph_data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(node.get("idx", idx)): node for idx, node in enumerate(graph_data.get("nodes", []) or [])}


def prediction_face_by_idx(prediction_data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(face.get("idx", idx)): face for idx, face in enumerate(prediction_data.get("faces", []) or [])}


def round_float(value: Any, ndigits: int = 4) -> float:
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return 0.0


def round_list(values: Any, ndigits: int = 4) -> list[float]:
    if not isinstance(values, list):
        return []
    return [round_float(value, ndigits) for value in values[:3]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or consume face-level rib truth review templates.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    template = subparsers.add_parser("template")
    template.add_argument("--area-review-json", type=Path, required=True)
    template.add_argument("--graph-dir", type=Path, required=True)
    template.add_argument("--prediction-dir", type=Path, required=True)
    template.add_argument("--feature-objects-dir", type=Path, required=True)
    template.add_argument("--output-json", type=Path, required=True)
    template.add_argument("--output-md", type=Path, required=True)
    template.add_argument("--sample-ids", default="")
    template.add_argument("--max-candidates-per-sample", type=int, default=80)
    template.add_argument("--high-aspect-threshold", type=float, default=4.0)

    dataset = subparsers.add_parser("dataset")
    dataset.add_argument("--review-template-json", type=Path, required=True)
    dataset.add_argument("--output-dir", type=Path, required=True)
    dataset.add_argument("--seed", type=int, default=20260629)
    dataset.add_argument("--train-ratio", type=float, default=0.8)
    dataset.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "template":
        area_review = load_json(args.area_review_json)
        sample_ids = [item.strip() for item in args.sample_ids.split(",") if item.strip()] or None
        template = build_rib_face_truth_template(
            area_review=area_review,
            graph_dir=args.graph_dir,
            prediction_dir=args.prediction_dir,
            feature_objects_dir=args.feature_objects_dir,
            sample_ids=sample_ids,
            high_aspect_threshold=args.high_aspect_threshold,
            max_candidates_per_sample=args.max_candidates_per_sample,
        )
        write_json(args.output_json, template)
        write_markdown(args.output_md, template)
        print(f"rib face truth template json written: {args.output_json}")
        print(f"rib face truth template markdown written: {args.output_md}")
        return

    template = load_json(args.review_template_json)
    records = build_rib_boundary_records_from_review_template(template)
    splits = split_rib_boundary_records(records, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    metadata = write_rib_boundary_dataset(records, splits, args.output_dir)
    metadata["source_review_template_json"] = str(args.review_template_json)
    write_json(args.output_dir / "metadata.json", metadata)
    print(json.dumps({"output_dir": str(args.output_dir), "record_count": len(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
