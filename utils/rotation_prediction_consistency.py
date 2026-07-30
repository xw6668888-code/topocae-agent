from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize prediction consistency across rotated variants such as part11/part12/part13."
    )
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--feature-object-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def collect_samples(prediction_dir: Path, feature_object_dir: Path) -> dict[str, dict[str, Any]]:
    samples = {}
    for prediction_path in sorted(prediction_dir.glob("*.json")):
        if prediction_path.name == "prediction_manifest.json":
            continue
        feature_object_path = feature_object_dir / prediction_path.name
        if not feature_object_path.exists():
            continue
        prediction = load_json(prediction_path)
        feature_objects = load_json(feature_object_path)
        samples[prediction_path.stem] = summarize_sample(prediction, feature_objects)
    return samples


def summarize_sample(prediction: dict[str, Any], feature_objects_doc: dict[str, Any]) -> dict[str, Any]:
    face_macro_counts = Counter(face.get("macro_feature", "unknown") for face in prediction.get("faces", []))
    object_macro_counts = Counter(
        obj.get("macro_feature", "unknown")
        for obj in feature_objects_doc.get("feature_objects", [])
    )
    object_signatures = sorted(
        (
            feature_object_signature(obj)
            for obj in feature_objects_doc.get("feature_objects", [])
        ),
        key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=True),
    )
    return {
        "sample_id": prediction.get("sample_id"),
        "face_count": len(prediction.get("faces", [])),
        "feature_object_count": len(feature_objects_doc.get("feature_objects", [])),
        "face_macro_counts": dict(sorted(face_macro_counts.items())),
        "feature_object_macro_counts": dict(sorted(object_macro_counts.items())),
        "feature_object_signatures": object_signatures,
    }


def feature_object_signature(obj: dict[str, Any]) -> dict[str, Any]:
    fingerprint = obj.get("fingerprint", {})
    primitive_hist = fingerprint.get("primitive_surface_histogram", {})
    regions = obj.get("selectable_regions", [])
    region_summary = sorted(
        (
            {
                "role": region.get("role", "unknown"),
                "face_count": len(region.get("face_indices", [])),
            }
            for region in regions
        ),
        key=lambda item: (item["role"], item["face_count"]),
    )
    return {
        "macro_feature": obj.get("macro_feature", "unknown"),
        "face_count": len(obj.get("face_indices", [])),
        "primitive_surface_histogram": dict(sorted(primitive_hist.items())),
        "region_summary": region_summary,
        "area_total_rounded": round(float(fingerprint.get("area_total", 0.0)), 3),
    }


def infer_group_id(sample_id: str) -> str:
    if len(sample_id) > 1 and sample_id[-1].isdigit():
        return sample_id[:-1]
    return sample_id


def build_report(samples: dict[str, dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for sample_id, sample in samples.items():
        groups.setdefault(infer_group_id(sample_id), []).append(sample)

    group_reports = []
    for group_id, group_samples in sorted(groups.items()):
        group_samples = sorted(group_samples, key=lambda item: item["sample_id"])
        reference = group_samples[0] if group_samples else {}
        face_macro_consistent = all(
            sample["face_macro_counts"] == reference.get("face_macro_counts", {})
            for sample in group_samples
        )
        object_macro_consistent = all(
            sample["feature_object_macro_counts"] == reference.get("feature_object_macro_counts", {})
            for sample in group_samples
        )
        object_signature_consistent = all(
            sample["feature_object_signatures"] == reference.get("feature_object_signatures", [])
            for sample in group_samples
        )
        group_reports.append({
            "group_id": group_id,
            "sample_ids": [sample["sample_id"] for sample in group_samples],
            "variant_count": len(group_samples),
            "face_macro_consistent": face_macro_consistent,
            "feature_object_macro_consistent": object_macro_consistent,
            "feature_object_signature_consistent": object_signature_consistent,
            "overall_consistent": (
                face_macro_consistent
                and object_macro_consistent
                and object_signature_consistent
            ),
            "samples": group_samples,
        })

    return {
        "schema": "cae_rotation_prediction_consistency_v1",
        "sample_count": len(samples),
        "group_count": len(group_reports),
        "overall_consistent_group_count": sum(1 for group in group_reports if group["overall_consistent"]),
        "groups": group_reports,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Rotation Prediction Consistency",
        "",
        f"Sample count: `{report['sample_count']}`",
        f"Group count: `{report['group_count']}`",
        f"Overall consistent groups: `{report['overall_consistent_group_count']}`",
        "",
        "| Group | Samples | Face Macro | Object Macro | Object Signature | Overall |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for group in report.get("groups", []):
        lines.append(
            "| {group} | `{samples}` | {face} | {obj} | {sig} | {overall} |".format(
                group=group["group_id"],
                samples=", ".join(group["sample_ids"]),
                face=_yes_no(group["face_macro_consistent"]),
                obj=_yes_no(group["feature_object_macro_consistent"]),
                sig=_yes_no(group["feature_object_signature_consistent"]),
                overall=_yes_no(group["overall_consistent"]),
            )
        )

    for group in report.get("groups", []):
        lines.extend(["", f"## {group['group_id']}", ""])
        for sample in group.get("samples", []):
            lines.append(
                "- `{sample_id}` faces={face_count} objects={object_count} face_macros=`{face_macros}` object_macros=`{object_macros}`".format(
                    sample_id=sample["sample_id"],
                    face_count=sample["face_count"],
                    object_count=sample["feature_object_count"],
                    face_macros=sample["face_macro_counts"],
                    object_macros=sample["feature_object_macro_counts"],
                )
            )
    lines.append("")
    return "\n".join(lines)


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def main() -> None:
    args = parse_args()
    samples = collect_samples(args.prediction_dir, args.feature_object_dir)
    report = build_report(samples)
    write_json(args.output_json, report)
    write_text(args.output_md, render_markdown(report))
    print(f"rotation consistency json written: {args.output_json}")
    print(f"rotation consistency markdown written: {args.output_md}")


if __name__ == "__main__":
    main()
