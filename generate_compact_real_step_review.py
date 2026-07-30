from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


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


def build_compact_review(feature_objects_path: Path, recursive: bool = False) -> dict[str, Any]:
    files = collect_feature_object_files(Path(feature_objects_path), recursive=recursive)
    samples = []
    total_macro_counts = Counter()

    for path in files:
        doc = load_json(path)
        sample = compact_sample(doc, path)
        samples.append(sample)
        total_macro_counts.update(sample["macro_counts"])

    return {
        "schema": "cae_real_step_compact_review_v1",
        "sample_count": len(samples),
        "total_macro_counts": dict(sorted(total_macro_counts.items())),
        "review_status_values": [
            "unreviewed",
            "correct",
            "wrong_class",
            "wrong_faces",
            "partial",
            "out_of_scope",
            "missed_feature",
        ],
        "instructions": {
            "review_status": "Set a compact judgement per predicted feature object.",
            "corrected_macro_feature": "Edit only when the predicted class is wrong.",
            "corrected_face_indices": "Edit only when the predicted face set is wrong.",
            "missing_features": "Add only obvious missing in-scope features for the part.",
        },
        "samples": samples,
    }


def compact_sample(doc: dict[str, Any], source_path: Path) -> dict[str, Any]:
    feature_objects = doc.get("feature_objects", [])
    macro_counts = Counter(obj.get("macro_feature", "unknown") for obj in feature_objects)
    return {
        "sample_id": doc.get("sample_id", source_path.stem),
        "source_step_path": doc.get("source_step_path"),
        "source_feature_objects_path": str(source_path.resolve()),
        "stock_face_count": len(doc.get("stock_face_indices", [])),
        "feature_object_count": len(feature_objects),
        "macro_counts": dict(sorted(macro_counts.items())),
        "features": [
            compact_feature_object(obj, index)
            for index, obj in enumerate(feature_objects)
        ],
        "missing_features": [],
        "part_review_notes": "",
    }


def compact_feature_object(obj: dict[str, Any], index: int) -> dict[str, Any]:
    macro_feature = obj.get("macro_feature", "unknown")
    fingerprint = obj.get("fingerprint", {})
    confidence = obj.get("confidence", {})
    return {
        "index": index,
        "feature_object_id": obj.get("feature_object_id"),
        "review_status": "unreviewed",
        "macro_feature": macro_feature,
        "corrected_macro_feature": macro_feature,
        "face_indices": list(obj.get("face_indices", [])),
        "corrected_face_indices": list(obj.get("face_indices", [])),
        "macro_confidence_mean": _round_optional(confidence.get("macro_confidence_mean")),
        "semantic_confidence_mean": _round_optional(confidence.get("semantic_confidence_mean")),
        "area_total": _round_optional(fingerprint.get("area_total")),
        "centroid": _round_list(fingerprint.get("area_weighted_centroid", []), 3),
        "bbox_size": _round_list(fingerprint.get("center_bbox_size", []), 3),
        "primitive_surface_histogram": dict(fingerprint.get("primitive_surface_histogram", {})),
        "selectable_regions": [
            f"{region.get('role', 'unknown')}:{list(region.get('face_indices', []))}"
            for region in obj.get("selectable_regions", [])
        ],
        "review_notes": "",
    }


def render_markdown(review: dict[str, Any]) -> str:
    lines = [
        "# Compact Real STEP Review",
        "",
        f"Sample count: `{review['sample_count']}`",
        "",
        "## Total Macro Counts",
        "",
    ]
    for macro, count in review.get("total_macro_counts", {}).items():
        lines.append(f"- `{macro}`: {count}")
    lines.append("")

    for sample in review.get("samples", []):
        lines.extend([
            f"## {sample['sample_id']}",
            "",
            f"- STEP: `{sample.get('source_step_path')}`",
            f"- Feature objects: `{sample.get('feature_object_count', 0)}`",
            f"- Macro counts: `{sample.get('macro_counts', {})}`",
            "",
            "| # | Status | Macro | Faces | Conf | Area | Centroid | BBox | Regions |",
            "|---:|---|---|---|---:|---:|---|---|---|",
        ])
        for item in sample.get("features", []):
            lines.append(
                "| {index} | {status} | `{macro}` | `{faces}` | {conf} | {area} | `{centroid}` | `{bbox}` | `{regions}` |".format(
                    index=item["index"],
                    status=item["review_status"],
                    macro=item["macro_feature"],
                    faces=item["face_indices"],
                    conf=_format_optional(item["macro_confidence_mean"]),
                    area=_format_optional(item["area_total"]),
                    centroid=item["centroid"],
                    bbox=item["bbox_size"],
                    regions=item["selectable_regions"],
                )
            )
        lines.append("")
    return "\n".join(lines)


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _round_optional(value: Any, ndigits: int = 4) -> float | None:
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def _round_list(values: Any, ndigits: int) -> list[float]:
    if not isinstance(values, list):
        return []
    rounded = []
    for value in values:
        try:
            rounded.append(round(float(value), ndigits))
        except (TypeError, ValueError):
            continue
    return rounded


def _format_optional(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate compact review files from real STEP feature-object predictions.")
    parser.add_argument("--feature-objects-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review = build_compact_review(args.feature_objects_path, recursive=args.recursive)
    write_json(args.output_json, review)
    write_markdown(args.output_md, render_markdown(review))
    print(f"compact review json written: {args.output_json}")
    print(f"compact review markdown written: {args.output_md}")


if __name__ == "__main__":
    main()
