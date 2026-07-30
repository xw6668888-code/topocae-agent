from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def collect_sample_ids(graph_dir: Path, requested_ids: list[str] | None) -> list[str]:
    if requested_ids:
        return requested_ids
    return sorted(path.stem for path in graph_dir.glob("*.json"))


def node_area(node: dict[str, Any]) -> float:
    return float(node.get("attributes", {}).get("physical", {}).get("area", 0.0) or 0.0)


def node_specific(node: dict[str, Any], key: str, default: Any = None) -> Any:
    return node.get("attributes", {}).get("specific", {}).get(key, default)


def compact_face_summary(node: dict[str, Any]) -> dict[str, Any]:
    descriptors = node.get("descriptors", {})
    return {
        "idx": int(node.get("idx", -1)),
        "primitive_surface": node.get("primitive_surface", "unknown_surface"),
        "area": node_area(node),
        "center_of_mass": node.get("attributes", {}).get("physical", {}).get("center_of_mass", []),
        "normal": node_specific(node, "normal", []),
        "axis": node_specific(node, "axis", []),
        "radius": node_specific(node, "radius", 0.0),
        "shape_descriptors": {
            key: descriptors.get(key)
            for key in [
                "bbox_size_x",
                "bbox_size_y",
                "bbox_size_z",
                "face_area_ratio",
                "projected_aspect_ratio",
                "uv_aspect_ratio",
            ]
            if key in descriptors
        },
        "stage_predictions": {},
    }


def build_prediction_lookup(prediction_doc: dict[str, Any]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for face in prediction_doc.get("faces", []):
        idx = int(face.get("idx", -1))
        lookup[idx] = {
            "macro_feature": face.get("macro_feature", "unknown_feature"),
            "semantic_subtype": face.get("semantic_subtype", "unknown_subtype"),
            "macro_confidence": face.get("macro_confidence"),
            "semantic_confidence": face.get("semantic_confidence"),
            "derived_macro_features": face.get("derived_macro_features", []),
            "idealization_tags": face.get("idealization_tags", []),
            "topology_repair": face.get("topology_repair"),
        }
    return lookup


def compact_feature_object(feature_object: dict[str, Any]) -> dict[str, Any]:
    fingerprint = feature_object.get("fingerprint", {})
    return {
        "feature_object_id": feature_object.get("feature_object_id"),
        "macro_feature": feature_object.get("macro_feature", "unknown_feature"),
        "face_indices": list(feature_object.get("face_indices", [])),
        "face_count": int(feature_object.get("face_count", len(feature_object.get("face_indices", [])))),
        "area_total": float(fingerprint.get("area_total", feature_object.get("area_total", 0.0)) or 0.0),
        "primitive_surface_histogram": fingerprint.get("primitive_surface_histogram", {}),
        "semantic_subtype_histogram": fingerprint.get("semantic_subtype_histogram", {}),
        "adjacent_stock_face_count": fingerprint.get("adjacent_stock_face_count"),
        "adjacent_nonstock_face_count": fingerprint.get("adjacent_nonstock_face_count"),
        "confidence": feature_object.get("confidence", {}),
    }


def edge_summary(graph_doc: dict[str, Any]) -> list[dict[str, Any]]:
    edges = graph_doc.get("edges", [])
    edge_attrs = graph_doc.get("edge_attributes", [])
    summaries: list[dict[str, Any]] = []
    for index, edge in enumerate(edges):
        if not isinstance(edge, list) or len(edge) != 2:
            continue
        attr = edge_attrs[index] if index < len(edge_attrs) and isinstance(edge_attrs[index], dict) else {}
        summaries.append(
            {
                "src": int(edge[0]),
                "dst": int(edge[1]),
                "curve_type": attr.get("curve_type"),
                "convexity": attr.get("convexity"),
                "length": attr.get("length"),
                "dihedral_angle": attr.get("dihedral_angle"),
            }
        )
    return summaries


def load_stage_doc(stage_dir: Path, sample_id: str) -> dict[str, Any]:
    path = stage_dir / f"{sample_id}.json"
    if not path.exists():
        return {}
    return load_json(path)


def build_sample_atlas(
    sample_id: str,
    graph_dir: Path,
    prediction_stages: dict[str, Path],
    feature_object_stages: dict[str, Path],
    truth_by_sample: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    graph_path = graph_dir / f"{sample_id}.json"
    graph_doc = load_json(graph_path)

    faces = [compact_face_summary(node) for node in graph_doc.get("nodes", [])]
    faces.sort(key=lambda face: int(face["idx"]))
    face_by_idx = {int(face["idx"]): face for face in faces}

    for stage_name, stage_dir in prediction_stages.items():
        prediction_doc = load_stage_doc(stage_dir, sample_id)
        for idx, prediction in build_prediction_lookup(prediction_doc).items():
            if idx in face_by_idx:
                face_by_idx[idx]["stage_predictions"][stage_name] = prediction

    feature_objects: dict[str, list[dict[str, Any]]] = {}
    for stage_name, stage_dir in feature_object_stages.items():
        feature_doc = load_stage_doc(stage_dir, sample_id)
        feature_objects[stage_name] = [
            compact_feature_object(feature_object)
            for feature_object in feature_doc.get("feature_objects", [])
        ]

    return {
        "sample_id": sample_id,
        "source_step_path": graph_doc.get("metadata", {}).get("source_step_path")
        or graph_doc.get("source_step_path"),
        "graph_path": str(graph_path.resolve()),
        "truth": truth_by_sample.get(sample_id, []),
        "faces": faces,
        "edges": edge_summary(graph_doc),
        "feature_objects": feature_objects,
    }


def build_failure_atlas(
    graph_dir: Path,
    prediction_stages: dict[str, Path],
    feature_object_stages: dict[str, Path],
    truth_json: Path,
    sample_ids: list[str] | None = None,
) -> dict[str, Any]:
    truth_doc = load_json(truth_json)
    truth_by_sample = truth_doc.get("samples", {})
    selected_ids = collect_sample_ids(graph_dir, sample_ids)
    samples = [
        build_sample_atlas(
            sample_id=sample_id,
            graph_dir=graph_dir,
            prediction_stages=prediction_stages,
            feature_object_stages=feature_object_stages,
            truth_by_sample=truth_by_sample,
        )
        for sample_id in selected_ids
    ]
    return {
        "schema": "cae_failure_atlas_v1",
        "graph_dir": str(graph_dir.resolve()),
        "truth_json": str(truth_json.resolve()),
        "prediction_stages": {name: str(path.resolve()) for name, path in prediction_stages.items()},
        "feature_object_stages": {name: str(path.resolve()) for name, path in feature_object_stages.items()},
        "sample_count": len(samples),
        "samples": samples,
    }


def format_stage_predictions(stage_predictions: dict[str, dict[str, Any]]) -> str:
    if not stage_predictions:
        return ""
    parts = []
    for stage_name, prediction in stage_predictions.items():
        macro = prediction.get("macro_feature", "unknown_feature")
        subtype = prediction.get("semantic_subtype", "unknown_subtype")
        confidence = prediction.get("macro_confidence")
        if confidence is None:
            parts.append(f"{stage_name}: `{macro}` / `{subtype}`")
        else:
            parts.append(f"{stage_name}: `{macro}` / `{subtype}` ({float(confidence):.3f})")
    return "<br>".join(parts)


def write_markdown(path: Path, atlas: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Failure Atlas",
        "",
        f"Graph dir: `{atlas['graph_dir']}`",
        f"Truth json: `{atlas['truth_json']}`",
        "",
    ]
    for sample in atlas["samples"]:
        lines.extend(
            [
                f"## {sample['sample_id']}",
                "",
                f"STEP: `{sample.get('source_step_path')}`",
                "",
                "### Truth",
                "",
            ]
        )
        truth = sample.get("truth", [])
        if truth:
            lines.append("| Macro | Faces | Area |")
            lines.append("|---|---:|---:|")
            for item in truth:
                lines.append(
                    f"| `{item.get('macro')}` | {int(item.get('face_count', 0))} | {float(item.get('area', 0.0)):.4f} |"
                )
        else:
            lines.append("No truth entries.")
        lines.extend(["", "### Feature Objects", ""])
        for stage_name, objects in sample.get("feature_objects", {}).items():
            lines.append(f"#### {stage_name}")
            if objects:
                lines.append("| Macro | Faces | Area | Face ids |")
                lines.append("|---|---:|---:|---|")
                for obj in objects:
                    lines.append(
                        f"| `{obj['macro_feature']}` | {obj['face_count']} | {obj['area_total']:.4f} | `{obj['face_indices']}` |"
                    )
            else:
                lines.append("No feature objects.")
            lines.append("")
        lines.extend(["### Faces", ""])
        lines.append("| Face | Primitive | Area | Radius | Predictions |")
        lines.append("|---:|---|---:|---:|---|")
        for face in sample.get("faces", []):
            predictions = format_stage_predictions(face.get("stage_predictions", {}))
            lines.append(
                f"| {int(face['idx'])} | `{face['primitive_surface']}` | "
                f"{float(face['area']):.4f} | {float(face.get('radius') or 0.0):.4f} | {predictions} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("stage paths must use name=path")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("stage name cannot be empty")
    return name, Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a face-level failure atlas across prediction stages.")
    parser.add_argument("--graph-dir", required=True, type=Path)
    parser.add_argument("--truth-json", required=True, type=Path)
    parser.add_argument("--sample-ids", default="", help="Comma-separated sample ids. Defaults to all graph json files.")
    parser.add_argument("--prediction-stage", action="append", default=[], type=parse_named_path, help="name=path")
    parser.add_argument("--feature-object-stage", action="append", default=[], type=parse_named_path, help="name=path")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_ids = [item.strip() for item in args.sample_ids.split(",") if item.strip()] or None
    atlas = build_failure_atlas(
        graph_dir=args.graph_dir,
        prediction_stages=dict(args.prediction_stage),
        feature_object_stages=dict(args.feature_object_stage),
        truth_json=args.truth_json,
        sample_ids=sample_ids,
    )
    write_json(args.output_json, atlas)
    write_markdown(args.output_md, atlas)
    print(f"failure atlas json written: {args.output_json}")
    print(f"failure atlas markdown written: {args.output_md}")


if __name__ == "__main__":
    main()
