from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any


REVIEW_STATUSES = [
    "unreviewed",
    "correct",
    "wrong_class",
    "wrong_faces",
    "partial",
    "out_of_scope",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def collect_feature_object_files(feature_objects_path: Path, recursive: bool = False) -> list[Path]:
    if feature_objects_path.is_file():
        return [feature_objects_path]

    globber = feature_objects_path.rglob if recursive else feature_objects_path.glob
    return sorted(
        path for path in globber("*.json")
        if path.is_file() and path.name != "feature_object_manifest.json"
    )


def compact_selectable_region(region: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": region.get("role", "unknown_region"),
        "face_indices": list(region.get("face_indices", [])),
        "source_semantic_subtypes": list(region.get("source_semantic_subtypes", [])),
        "intended_uses": list(region.get("intended_uses", [])),
        "is_default_for": list(region.get("is_default_for", [])),
        "selection_priority": region.get("selection_priority"),
    }


def editable_selectable_region(region: dict[str, Any]) -> dict[str, Any]:
    compact_region = compact_selectable_region(region)
    return {
        "role": compact_region["role"],
        "face_indices": compact_region["face_indices"],
        "review_status": "unreviewed",
        "corrected_role": compact_region["role"],
        "corrected_face_indices": compact_region["face_indices"],
        "notes": "",
    }


def geometry_summary(feature_object: dict[str, Any]) -> dict[str, Any]:
    fingerprint = feature_object.get("fingerprint", {})
    return {
        "area_total": fingerprint.get("area_total"),
        "area_weighted_centroid": fingerprint.get("area_weighted_centroid", []),
        "center_bbox_size": fingerprint.get("center_bbox_size", []),
        "radius_stats": fingerprint.get("radius_stats", {}),
        "primitive_surface_histogram": fingerprint.get("primitive_surface_histogram", {}),
        "semantic_subtype_histogram": fingerprint.get("semantic_subtype_histogram", {}),
        "derived_macro_feature_histogram": fingerprint.get("derived_macro_feature_histogram", {}),
        "idealization_tag_histogram": fingerprint.get("idealization_tag_histogram", {}),
    }


def build_predicted_feature_template(feature_object: dict[str, Any], index: int) -> dict[str, Any]:
    macro_feature = feature_object.get("macro_feature", "unknown_feature")
    face_indices = list(feature_object.get("face_indices", []))
    selectable_regions = [
        compact_selectable_region(region)
        for region in feature_object.get("selectable_regions", [])
    ]

    return {
        "prediction_index": index,
        "predicted_feature_object_id": feature_object.get("feature_object_id"),
        "review_status": "unreviewed",
        "is_in_scope": True,
        "predicted_macro_feature": macro_feature,
        "ground_truth_macro_feature": macro_feature,
        "predicted_face_indices": face_indices,
        "ground_truth_face_indices": face_indices,
        "predicted_selectable_regions": selectable_regions,
        "ground_truth_selectable_regions": [
            editable_selectable_region(region)
            for region in feature_object.get("selectable_regions", [])
        ],
        "prediction_confidence": deepcopy(feature_object.get("confidence", {})),
        "geometry_summary": geometry_summary(feature_object),
        "review_notes": "",
    }


def build_sample_template(feature_object_doc: dict[str, Any]) -> dict[str, Any]:
    feature_objects = feature_object_doc.get("feature_objects", [])
    return {
        "sample_id": feature_object_doc.get("sample_id"),
        "source_step_path": feature_object_doc.get("source_step_path"),
        "source_graph_path": feature_object_doc.get("source_graph_path"),
        "source_prediction_path": feature_object_doc.get("source_prediction_path"),
        "source_feature_objects_path": feature_object_doc.get("source_feature_objects_path"),
        "predicted_feature_count": len(feature_objects),
        "stock_face_indices": list(feature_object_doc.get("stock_face_indices", [])),
        "predicted_features": [
            build_predicted_feature_template(feature_object, index)
            for index, feature_object in enumerate(feature_objects)
        ],
        "missing_ground_truth_features": [],
        "part_review_notes": "",
    }


def build_annotation_template(feature_object_docs: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [build_sample_template(feature_object_doc) for feature_object_doc in feature_object_docs]
    return {
        "annotation_schema": "cae_real_step_annotation_template_v1",
        "review_status_values": REVIEW_STATUSES,
        "instructions": {
            "predicted_features": (
                "Review each predicted feature. Keep or edit ground_truth_macro_feature, "
                "ground_truth_face_indices, and ground_truth_selectable_regions."
            ),
            "missing_ground_truth_features": (
                "Add features that are visible in the STEP part but missing from predicted_features."
            ),
            "out_of_scope": (
                "Set is_in_scope=false and review_status=out_of_scope for features outside the paper scope."
            ),
        },
        "sample_count": len(samples),
        "samples": samples,
    }


def build_template_from_paths(feature_object_files: list[Path]) -> dict[str, Any]:
    docs = []
    for path in feature_object_files:
        doc = load_json(path)
        doc["source_feature_objects_path"] = str(path.resolve())
        docs.append(doc)
    return build_annotation_template(docs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an editable annotation template from real STEP feature-object predictions."
    )
    parser.add_argument("--feature-objects-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_object_files = collect_feature_object_files(args.feature_objects_path, args.recursive)
    if not feature_object_files:
        raise RuntimeError(f"No feature-object json files found under: {args.feature_objects_path}")

    template = build_template_from_paths(feature_object_files)
    write_json(args.output_path, template)
    print(f"real STEP annotation template written: {args.output_path}")


if __name__ == "__main__":
    main()
