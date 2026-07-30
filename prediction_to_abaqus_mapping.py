from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


TOPOLOGY_TYPE_TO_PRIMITIVES = {
    "planar": {"plane"},
    "cylindrical/curved": {"cylinder"},
    "convex cylinder": {"cylinder"},
    "concave cylinder": {"cylinder"},
    "conical/curved": {"cone"},
    "curved": {"cylinder", "cone", "sphere", "torus", "bspline_surface", "surface_of_revolution", "surface_of_extrusion", "offset_surface", "other_surface"},
    "spherical/curved": {"sphere"},
    "toroidal/curved": {"torus"},
    "freeform/curved": {"bspline_surface", "other_surface"},
    "unknown": {"unknown_surface"},
}


def parse_args() -> argparse.Namespace:
    current_dir = Path(__file__).resolve().parent
    output_root = current_dir / "output"
    parser = argparse.ArgumentParser(
        description="Map predicted feature objects onto Abaqus topology face ids using geometry fingerprints."
    )
    parser.add_argument("--feature-object-path", type=Path, required=True)
    parser.add_argument("--topology-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--center-distance-weight", type=float, default=0.35)
    parser.add_argument("--area-weight", type=float, default=0.25)
    parser.add_argument("--direction-weight", type=float, default=0.20)
    parser.add_argument("--radius-weight", type=float, default=0.10)
    parser.add_argument("--type-weight", type=float, default=0.10)
    parser.add_argument("--score-threshold-high", type=float, default=0.75)
    parser.add_argument("--score-threshold-medium", type=float, default=0.55)
    parser.add_argument("--max-stock-planar-regions", type=int, default=6)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def collect_input_files(path: Path, recursive: bool, skip_names: set[str]) -> list[Path]:
    if path.is_file():
        return [path]
    globber = path.rglob if recursive else path.glob
    return sorted(
        item for item in globber("*.json")
        if item.is_file() and item.name not in skip_names
    )


def resolve_matching_topology_file(feature_object_file: Path, topology_root: Path) -> Path:
    if topology_root.is_file():
        return topology_root
    candidate = topology_root / feature_object_file.name
    if candidate.exists():
        return candidate
    matches = list(topology_root.rglob(feature_object_file.name))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Cannot resolve topology snapshot for {feature_object_file}")


def vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def normalize(vector: list[float]) -> list[float]:
    norm = vector_norm(vector)
    if norm <= 1e-12:
        return []
    return [value / norm for value in vector]


def dot_abs(lhs: list[float], rhs: list[float]) -> float:
    if not lhs or not rhs:
        return 0.0
    lhs_n = normalize(lhs)
    rhs_n = normalize(rhs)
    if not lhs_n or not rhs_n:
        return 0.0
    return abs(sum(a * b for a, b in zip(lhs_n, rhs_n)))


def bbox_diagonal(topology_data: dict) -> float:
    if "bbox" in topology_data:
        bbox = topology_data["bbox"]
        dx = float(bbox.get("x_max", 0.0)) - float(bbox.get("x_min", 0.0))
        dy = float(bbox.get("y_max", 0.0)) - float(bbox.get("y_min", 0.0))
        dz = float(bbox.get("z_max", 0.0)) - float(bbox.get("z_min", 0.0))
        return max(math.sqrt(dx * dx + dy * dy + dz * dz), 1e-6)

    centers = [face.get("center", [0.0, 0.0, 0.0]) for face in topology_data.get("faces", [])]
    if not centers:
        return 1.0
    xs = [center[0] for center in centers]
    ys = [center[1] for center in centers]
    zs = [center[2] for center in centers]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    dz = max(zs) - min(zs)
    return max(math.sqrt(dx * dx + dy * dy + dz * dz), 1e-6)


def normalize_topology_face(raw_face: dict) -> dict:
    axis = raw_face.get("axis_vector") or raw_face.get("axis") or [0.0, 0.0, 0.0]
    normal = raw_face.get("normal") or [0.0, 0.0, 0.0]
    return {
        "id": int(raw_face["id"]),
        "type": str(raw_face.get("type", "Unknown")),
        "type_norm": str(raw_face.get("type", "Unknown")).lower(),
        "area": float(raw_face.get("area", 0.0)),
        "center": list(raw_face.get("center", [0.0, 0.0, 0.0])),
        "normal": list(normal),
        "axis_vector": list(axis),
        "radius": float(raw_face.get("radius", 0.0)),
    }


def source_face_record(face_idx: int, prediction_data: dict, graph_data: dict) -> dict:
    pred_face = next(face for face in prediction_data["faces"] if face["idx"] == face_idx)
    graph_node = next(node for node in graph_data["nodes"] if node["idx"] == face_idx)
    specific = graph_node["attributes"].get("specific", {})
    return {
        "idx": face_idx,
        "primitive_surface": graph_node.get("primitive_surface", "unknown_surface"),
        "area": float(graph_node["attributes"]["physical"].get("area", 0.0)),
        "center": list(graph_node["attributes"]["physical"].get("center_of_mass", [0.0, 0.0, 0.0])),
        "normal": list(specific.get("normal", [])),
        "axis_vector": list(specific.get("axis", [])),
        "radius": float(specific.get("radius", 0.0)),
        "macro_feature": pred_face.get("macro_feature", "stock"),
        "semantic_subtype": pred_face.get("semantic_subtype", "unknown_subtype"),
        "macro_confidence": float(pred_face.get("macro_confidence", 0.0)),
        "semantic_confidence": float(pred_face.get("semantic_confidence", 0.0)),
    }


def type_compatibility_score(source_primitive: str, topology_type_norm: str) -> float:
    allowed = TOPOLOGY_TYPE_TO_PRIMITIVES.get(topology_type_norm)
    if allowed is None:
        return 0.5
    return 1.0 if source_primitive in allowed else 0.0


def area_similarity(source_area: float, target_area: float) -> float:
    denom = max(source_area, target_area, 1e-6)
    return max(0.0, 1.0 - abs(source_area - target_area) / denom)


def center_similarity(source_center: list[float], target_center: list[float], diag: float) -> float:
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(source_center, target_center)))
    return max(0.0, 1.0 - distance / max(diag, 1e-6))


def radius_similarity(source_radius: float, target_radius: float) -> float:
    if source_radius <= 1e-6 and target_radius <= 1e-6:
        return 1.0
    if source_radius <= 1e-6 or target_radius <= 1e-6:
        return 0.0
    denom = max(source_radius, target_radius, 1e-6)
    return max(0.0, 1.0 - abs(source_radius - target_radius) / denom)


def direction_similarity(source_face: dict, topology_face: dict) -> float:
    if source_face["primitive_surface"] == "plane":
        return dot_abs(source_face.get("normal", []), topology_face.get("normal", []))
    return max(
        dot_abs(source_face.get("axis_vector", []), topology_face.get("axis_vector", [])),
        dot_abs(source_face.get("normal", []), topology_face.get("normal", [])),
    )


def score_face_match(source_face: dict, topology_face: dict, diag: float, weights: dict[str, float]) -> dict:
    scores = {
        "center": center_similarity(source_face["center"], topology_face["center"], diag),
        "area": area_similarity(source_face["area"], topology_face["area"]),
        "direction": direction_similarity(source_face, topology_face),
        "radius": radius_similarity(source_face["radius"], topology_face["radius"]),
        "type": type_compatibility_score(source_face["primitive_surface"], topology_face["type_norm"]),
    }
    score = sum(weights[key] * scores[key] for key in weights)
    return {
        "score": score,
        "subscores": scores,
    }


def mapping_status(score_mean: float, high_threshold: float, medium_threshold: float) -> str:
    if score_mean >= high_threshold:
        return "high_confidence"
    if score_mean >= medium_threshold:
        return "medium_confidence"
    return "low_confidence"


def map_feature_object(
    feature_object: dict,
    prediction_data: dict,
    graph_data: dict,
    topology_faces: list[dict],
    used_topology_face_ids: set[int],
    diag: float,
    weights: dict[str, float],
    high_threshold: float,
    medium_threshold: float,
) -> dict:
    source_faces = [
        source_face_record(face_idx, prediction_data, graph_data)
        for face_idx in feature_object.get("face_indices", [])
    ]
    source_faces.sort(key=lambda item: item["area"], reverse=True)

    face_matches = []
    local_used = set()
    for source_face in source_faces:
        scored_candidates = []
        for topology_face in topology_faces:
            if topology_face["id"] in used_topology_face_ids or topology_face["id"] in local_used:
                continue
            match = score_face_match(source_face, topology_face, diag, weights)
            scored_candidates.append({
                "abaqus_face_id": topology_face["id"],
                "topology_type": topology_face["type"],
                "score": match["score"],
                "subscores": match["subscores"],
            })

        scored_candidates.sort(key=lambda item: item["score"], reverse=True)
        if not scored_candidates:
            face_matches.append({
                "source_face_idx": source_face["idx"],
                "abaqus_face_id": None,
                "score": 0.0,
                "subscores": {},
            })
            continue

        best = scored_candidates[0]
        local_used.add(best["abaqus_face_id"])
        face_matches.append({
            "source_face_idx": source_face["idx"],
            "abaqus_face_id": best["abaqus_face_id"],
            "topology_type": best["topology_type"],
            "score": best["score"],
            "subscores": best["subscores"],
        })

    mapped_ids = [item["abaqus_face_id"] for item in face_matches if item["abaqus_face_id"] is not None]
    used_topology_face_ids.update(mapped_ids)
    score_mean = sum(item["score"] for item in face_matches) / max(len(face_matches), 1)

    return {
        "feature_object_id": feature_object["feature_object_id"],
        "macro_feature_id": feature_object["macro_feature_id"],
        "macro_feature": feature_object["macro_feature"],
        "source_face_indices": feature_object.get("face_indices", []),
        "mapped_abaqus_face_ids": mapped_ids,
        "mapping_score_mean": score_mean,
        "mapping_status": mapping_status(score_mean, high_threshold, medium_threshold),
        "face_matches": face_matches,
        "source_confidence": feature_object.get("confidence", {}),
    }


def suggested_stock_regions(topology_faces: list[dict], used_ids: set[int], max_regions: int) -> list[dict]:
    candidates = [
        face for face in topology_faces
        if face["id"] not in used_ids and face["type_norm"] == "planar"
    ]
    candidates.sort(key=lambda face: face["area"], reverse=True)
    result = []
    for idx, face in enumerate(candidates[:max_regions]):
        result.append({
            "region_name": f"stock_planar_face_{idx:03d}",
            "abaqus_face_ids": [face["id"]],
            "area": face["area"],
            "center": face["center"],
            "normal": face["normal"],
        })
    return result


def suggested_feature_regions(feature_mappings: list[dict]) -> list[dict]:
    regions = []
    for idx, mapping in enumerate(feature_mappings):
        if not mapping["mapped_abaqus_face_ids"]:
            continue
        regions.append({
            "region_name": f"feature_{idx:03d}_{mapping['macro_feature']}",
            "feature_object_id": mapping["feature_object_id"],
            "macro_feature": mapping["macro_feature"],
            "abaqus_face_ids": mapping["mapped_abaqus_face_ids"],
            "mapping_status": mapping["mapping_status"],
            "mapping_score_mean": mapping["mapping_score_mean"],
        })
    return regions


def build_mapping_output(
    feature_object_data: dict,
    feature_object_path: Path,
    topology_data: dict,
    topology_path: Path,
    prediction_data: dict,
    graph_data: dict,
    args: argparse.Namespace,
) -> dict:
    topology_faces = [normalize_topology_face(face) for face in topology_data.get("faces", [])]
    diag = bbox_diagonal(topology_data)
    weights = {
        "center": args.center_distance_weight,
        "area": args.area_weight,
        "direction": args.direction_weight,
        "radius": args.radius_weight,
        "type": args.type_weight,
    }

    used_topology_face_ids = set()
    feature_mappings = []
    for feature_object in feature_object_data.get("feature_objects", []):
        mapping = map_feature_object(
            feature_object=feature_object,
            prediction_data=prediction_data,
            graph_data=graph_data,
            topology_faces=topology_faces,
            used_topology_face_ids=used_topology_face_ids,
            diag=diag,
            weights=weights,
            high_threshold=args.score_threshold_high,
            medium_threshold=args.score_threshold_medium,
        )
        feature_mappings.append(mapping)

    return {
        "sample_id": feature_object_data.get("sample_id", feature_object_path.stem),
        "mapping_mode": "feature_objects_to_abaqus_faces",
        "source_step_path": feature_object_data.get("source_step_path"),
        "source_graph_path": feature_object_data.get("source_graph_path"),
        "source_prediction_path": feature_object_data.get("source_prediction_path"),
        "source_feature_object_path": str(feature_object_path.resolve()),
        "source_topology_path": str(topology_path.resolve()),
        "topology_face_count": len(topology_faces),
        "feature_mapping_count": len(feature_mappings),
        "score_weights": weights,
        "feature_mappings": feature_mappings,
        "suggested_abaqus_regions": {
            "feature_object_regions": suggested_feature_regions(feature_mappings),
            "stock_planar_regions": suggested_stock_regions(
                topology_faces,
                used_ids={face_id for mapping in feature_mappings for face_id in mapping["mapped_abaqus_face_ids"]},
                max_regions=args.max_stock_planar_regions,
            ),
        },
    }


def main() -> None:
    args = parse_args()
    feature_object_files = collect_input_files(
        args.feature_object_path,
        args.recursive,
        skip_names={"feature_object_manifest.json"},
    )
    if not feature_object_files:
        raise RuntimeError(f"No feature object files found under: {args.feature_object_path}")

    manifest_entries = []
    for feature_object_file in feature_object_files:
        feature_object_data = load_json(feature_object_file)
        if "feature_objects" not in feature_object_data:
            continue

        topology_path = resolve_matching_topology_file(feature_object_file, args.topology_path)
        topology_data = load_json(topology_path)
        prediction_data = load_json(Path(feature_object_data["source_prediction_path"]))
        graph_data = load_json(Path(feature_object_data["source_graph_path"]))

        mapping_output = build_mapping_output(
            feature_object_data=feature_object_data,
            feature_object_path=feature_object_file,
            topology_data=topology_data,
            topology_path=topology_path,
            prediction_data=prediction_data,
            graph_data=graph_data,
            args=args,
        )

        output_path = (
            args.output_dir / feature_object_file.relative_to(args.feature_object_path)
            if args.feature_object_path.is_dir()
            else args.output_dir / feature_object_file.name
        )
        write_json(output_path, mapping_output)
        manifest_entries.append({
            "sample_id": mapping_output["sample_id"],
            "feature_object_path": str(feature_object_file.resolve()),
            "topology_path": str(topology_path.resolve()),
            "mapping_output_path": str(output_path.resolve()),
            "feature_mapping_count": mapping_output["feature_mapping_count"],
        })
        print(f"abaqus mapping written: {output_path}")

    write_json(
        args.output_dir / "abaqus_mapping_manifest.json",
        {
            "mapping_mode": "feature_objects_to_abaqus_faces",
            "sample_count": len(manifest_entries),
            "samples": manifest_entries,
        },
    )


if __name__ == "__main__":
    main()
