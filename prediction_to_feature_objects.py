from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path

from utils.selectable_regions import build_selectable_regions


SINGLE_FACE_TRANSITION_MACROS = {"fillet", "chamfer"}
CONNECTED_COMPONENT_GROUPING_METHOD = "connected_components_same_macro_feature"
ASSEMBLED_INSTANCE_CANDIDATE_GROUPING_METHOD = "assembled_instance_candidates"


def parse_args() -> argparse.Namespace:
    current_dir = Path(__file__).resolve().parent
    output_root = current_dir / "output"
    parser = argparse.ArgumentParser(
        description="Convert face-level prediction json files into connected feature-object records."
    )
    parser.add_argument("--prediction-path", type=Path, required=True, help="Prediction json file or directory.")
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=output_root / "inference_graphs",
        help="Directory used to resolve source inference graphs when prediction json lacks a source_graph_path.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for feature-object json outputs.")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--fingerprint-round-ndigits",
        type=int,
        default=6,
        help="Decimal places used for stable fingerprint signatures.",
    )
    parser.add_argument(
        "--grouping-mode",
        choices=[CONNECTED_COMPONENT_GROUPING_METHOD, ASSEMBLED_INSTANCE_CANDIDATE_GROUPING_METHOD],
        default=CONNECTED_COMPONENT_GROUPING_METHOD,
        help="How face predictions are grouped into feature objects.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def collect_prediction_files(prediction_path: Path, recursive: bool) -> list[Path]:
    if prediction_path.is_file():
        return [prediction_path]
    globber = prediction_path.rglob if recursive else prediction_path.glob
    return sorted(
        path for path in globber("*.json")
        if path.is_file() and path.name != "prediction_manifest.json"
    )


def resolve_graph_path(prediction_data: dict, prediction_file: Path, graph_dir: Path) -> Path:
    source_graph_path = prediction_data.get("source_graph_path")
    if source_graph_path:
        return Path(source_graph_path)
    candidate = graph_dir / prediction_file.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Cannot resolve source graph for prediction: {prediction_file}")


def build_undirected_adjacency(edges: list[list[int]], node_count: int) -> dict[int, set[int]]:
    adjacency = {idx: set() for idx in range(node_count)}
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2:
            continue
        src, dst = edge
        if 0 <= src < node_count and 0 <= dst < node_count:
            adjacency[src].add(dst)
            adjacency[dst].add(src)
    return adjacency


def connected_components_by_macro(pred_faces: list[dict], adjacency: dict[int, set[int]]) -> list[list[int]]:
    face_by_idx = {face["idx"]: face for face in pred_faces}
    visited = set()
    components = []

    for face in pred_faces:
        start_idx = face["idx"]
        if start_idx in visited:
            continue
        macro_feature_id = int(face.get("macro_feature_id", 0))
        if macro_feature_id <= 0:
            visited.add(start_idx)
            continue
        if face.get("macro_feature") in SINGLE_FACE_TRANSITION_MACROS:
            visited.add(start_idx)
            components.append([start_idx])
            continue

        component = []
        queue = deque([start_idx])
        visited.add(start_idx)
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor in visited:
                    continue
                neighbor_face = face_by_idx.get(neighbor)
                if neighbor_face is None:
                    continue
                if int(neighbor_face.get("macro_feature_id", 0)) != macro_feature_id:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)

        components.append(sorted(component))
    return components



def accepted_instance_candidate_components(prediction_data: dict, pred_faces: list[dict]) -> list[dict]:
    records = prediction_data.get("instance_candidate_postprocess", {}).get("records", [])
    face_by_idx = {int(face["idx"]): face for face in pred_faces}
    components = []
    seen = set()
    for record in records:
        if record.get("action") != "accepted_remap":
            continue
        face_indices = sorted({int(idx) for idx in record.get("face_indices", [])})
        if not face_indices:
            continue
        current_faces = [face_by_idx.get(idx) for idx in face_indices]
        if any(face is None for face in current_faces):
            continue
        current_macros = {str(face.get("macro_feature", "stock")) for face in current_faces}
        if any(int(face.get("macro_feature_id", 0) or 0) <= 0 for face in current_faces):
            continue
        if len(current_macros) != 1:
            continue
        key = tuple(face_indices)
        if key in seen:
            continue
        seen.add(key)
        components.append({
            "component": face_indices,
            "grouping_method": ASSEMBLED_INSTANCE_CANDIDATE_GROUPING_METHOD,
            "source_assembly_record": deepcopy(record),
        })
    return components


def component_entries_from_grouping_mode(
    prediction_data: dict,
    pred_faces: list[dict],
    adjacency: dict[int, set[int]],
    grouping_mode: str,
) -> list[dict]:
    if grouping_mode == CONNECTED_COMPONENT_GROUPING_METHOD:
        return [
            {
                "component": component,
                "grouping_method": CONNECTED_COMPONENT_GROUPING_METHOD,
                "source_assembly_record": None,
            }
            for component in connected_components_by_macro(pred_faces, adjacency)
        ]
    if grouping_mode == ASSEMBLED_INSTANCE_CANDIDATE_GROUPING_METHOD:
        return accepted_instance_candidate_components(prediction_data, pred_faces)
    raise ValueError(f"Unsupported grouping_mode: {grouping_mode}")
def weighted_centroid(nodes: list[dict]) -> list[float]:
    total_area = sum(float(node["attributes"]["physical"].get("area", 0.0)) for node in nodes)
    if total_area <= 0:
        count = max(len(nodes), 1)
        sums = [0.0, 0.0, 0.0]
        for node in nodes:
            center = node["attributes"]["physical"].get("center_of_mass", [0.0, 0.0, 0.0])
            sums[0] += center[0]
            sums[1] += center[1]
            sums[2] += center[2]
        return [value / count for value in sums]

    weighted = [0.0, 0.0, 0.0]
    for node in nodes:
        area = float(node["attributes"]["physical"].get("area", 0.0))
        center = node["attributes"]["physical"].get("center_of_mass", [0.0, 0.0, 0.0])
        weighted[0] += area * center[0]
        weighted[1] += area * center[1]
        weighted[2] += area * center[2]
    return [value / total_area for value in weighted]


def bbox_from_centers(nodes: list[dict]) -> tuple[list[float], list[float], list[float]]:
    centers = [node["attributes"]["physical"].get("center_of_mass", [0.0, 0.0, 0.0]) for node in nodes]
    mins = [min(center[axis] for center in centers) for axis in range(3)]
    maxs = [max(center[axis] for center in centers) for axis in range(3)]
    size = [maxs[axis] - mins[axis] for axis in range(3)]
    return mins, maxs, size


def normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-12:
        return []
    return [value / norm for value in vector]


def round_float(value: float, ndigits: int) -> float:
    return round(float(value), ndigits)


def round_vector(vector: list[float], ndigits: int) -> list[float]:
    return [round_float(value, ndigits) for value in vector]


def mean_direction(nodes: list[dict], key: str) -> list[float]:
    summed = [0.0, 0.0, 0.0]
    count = 0
    for node in nodes:
        direction = node["attributes"].get("specific", {}).get(key)
        if not direction or len(direction) != 3:
            continue
        summed[0] += float(direction[0])
        summed[1] += float(direction[1])
        summed[2] += float(direction[2])
        count += 1
    if count == 0:
        return []
    return normalize_vector(summed)


def radius_stats(nodes: list[dict]) -> dict:
    radii = [
        float(node["attributes"].get("specific", {}).get("radius", 0.0))
        for node in nodes
        if float(node["attributes"].get("specific", {}).get("radius", 0.0)) > 1e-9
    ]
    if not radii:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
    return {
        "count": len(radii),
        "min": min(radii),
        "max": max(radii),
        "mean": sum(radii) / len(radii),
    }


def face_descriptor(face_idx: int, pred_face: dict, graph_node: dict, ndigits: int) -> dict:
    physical = graph_node["attributes"].get("physical", {})
    specific = graph_node["attributes"].get("specific", {})
    return {
        "face_idx": int(face_idx),
        "primitive_surface": graph_node.get("primitive_surface", "unknown_surface"),
        "primitive_surface_id": int(graph_node.get("primitive_surface_id", -1)),
        "area": round_float(physical.get("area", 0.0), ndigits),
        "center": round_vector(physical.get("center_of_mass", [0.0, 0.0, 0.0]), ndigits),
        "normal": round_vector(specific.get("normal", []), ndigits),
        "axis": round_vector(specific.get("axis", []), ndigits),
        "radius": round_float(specific.get("radius", 0.0), ndigits),
        "macro_feature_id": int(pred_face.get("macro_feature_id", -1)),
        "macro_feature": pred_face.get("macro_feature", "unknown_feature"),
        "semantic_subtype_id": int(pred_face.get("semantic_subtype_id", -1)),
        "semantic_subtype": pred_face.get("semantic_subtype", "unknown_subtype"),
        "macro_confidence": round_float(pred_face.get("macro_confidence", 0.0), ndigits),
        "semantic_confidence": round_float(pred_face.get("semantic_confidence", 0.0), ndigits),
    }


def stable_signature(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def build_signature_payload(
    macro_feature_id: int,
    macro_feature: str,
    face_count: int,
    fingerprint: dict,
    ndigits: int,
) -> dict:
    radius = fingerprint["radius_stats"]
    return {
        "schema": "cae_feature_fingerprint_v1",
        "macro_feature_id": macro_feature_id,
        "macro_feature": macro_feature,
        "face_count": face_count,
        "area_total": round_float(fingerprint["area_total"], ndigits),
        "area_mean": round_float(fingerprint["area_mean"], ndigits),
        "bbox_size": round_vector(fingerprint["center_bbox_size"], ndigits),
        "mean_normal": round_vector(fingerprint["mean_normal"], ndigits),
        "mean_axis": round_vector(fingerprint["mean_axis"], ndigits),
        "radius_stats": {
            "count": int(radius["count"]),
            "min": round_float(radius["min"], ndigits),
            "max": round_float(radius["max"], ndigits),
            "mean": round_float(radius["mean"], ndigits),
        },
        "primitive_surface_histogram": fingerprint["primitive_surface_histogram"],
        "semantic_subtype_histogram": fingerprint["semantic_subtype_histogram"],
        "derived_macro_feature_histogram": fingerprint["derived_macro_feature_histogram"],
        "idealization_tag_histogram": fingerprint["idealization_tag_histogram"],
        "adjacent_macro_feature_histogram": fingerprint["adjacent_macro_feature_histogram"],
    }


def matching_hints(macro_feature: str, fingerprint: dict) -> dict:
    hints = [
        "macro_feature",
        "semantic_subtype_histogram",
        "primitive_surface_histogram",
        "face_count",
        "area_total",
        "area_weighted_centroid",
        "center_bbox_size",
        "mean_normal",
        "mean_axis",
        "radius_stats",
        "adjacent_macro_feature_histogram",
    ]
    geometry_targets = []
    primitive_histogram = fingerprint.get("primitive_surface_histogram", {})
    if primitive_histogram.get("cylinder", 0) > 0:
        geometry_targets.append("cylindrical_faces")
    if primitive_histogram.get("plane", 0) > 0:
        geometry_targets.append("planar_faces")
    if primitive_histogram.get("cone", 0) > 0:
        geometry_targets.append("conical_faces")
    if "hole" in macro_feature:
        geometry_targets.append("concave_or_internal_faces")
    if "boss" in macro_feature or macro_feature in {"rib", "web", "rectangular_boss"}:
        geometry_targets.append("external_or_protruding_faces")

    return {
        "intended_downstream_use": [
            "face_selection",
            "region_creation",
            "mesh_refinement",
            "bc_or_load_candidate_selection",
            "cad_edit_candidate_selection",
        ],
        "matching_priority": hints,
        "geometry_targets": sorted(set(geometry_targets)),
    }


def build_feature_set_lookup(feature_sets: list[dict]) -> dict[tuple[int, ...], dict]:
    lookup = {}
    for feature_set in feature_sets:
        node_indices = feature_set.get("node_indices", [])
        if not node_indices:
            continue
        lookup[tuple(sorted(int(idx) for idx in node_indices))] = feature_set
    return lookup


def source_feature_metadata(component: list[int], feature_set_lookup: dict[tuple[int, ...], dict]) -> dict:
    feature_set = feature_set_lookup.get(tuple(sorted(component)))
    if not feature_set:
        return {
            "source_instance_id": None,
            "source_generator": None,
            "geometry_parameters": {},
            "raw_parameters": {},
        }
    return {
        "source_instance_id": feature_set.get("instance_id"),
        "source_generator": feature_set.get("source_generator"),
        "geometry_parameters": deepcopy(feature_set.get("geometry_parameters", {})),
        "raw_parameters": deepcopy(feature_set.get("raw_parameters", {})),
    }


def summarize_component(
    sample_id: str,
    object_index: int,
    component: list[int],
    pred_face_map: dict[int, dict],
    graph_node_map: dict[int, dict],
    adjacency: dict[int, set[int]],
    feature_set_lookup: dict[tuple[int, ...], dict],
    ndigits: int,
    grouping_method: str = CONNECTED_COMPONENT_GROUPING_METHOD,
    source_assembly_record: dict | None = None,
) -> dict:
    pred_faces = [pred_face_map[idx] for idx in component]
    graph_nodes = [graph_node_map[idx] for idx in component]

    macro_feature_id = int(pred_faces[0]["macro_feature_id"])
    macro_feature = pred_faces[0]["macro_feature"]
    face_areas = [float(node["attributes"]["physical"].get("area", 0.0)) for node in graph_nodes]
    total_area = sum(face_areas)

    primitive_counter = Counter(node.get("primitive_surface", "unknown_surface") for node in graph_nodes)
    semantic_counter = Counter(face.get("semantic_subtype", "unknown_subtype") for face in pred_faces)
    derived_counter = Counter(
        derived_key
        for face in pred_faces
        for derived_key in face.get("derived_macro_features", [])
    )
    idealization_counter = Counter(
        tag_key
        for face in pred_faces
        for tag_key in face.get("idealization_tags", [])
    )

    boundary_neighbors = sorted({
        neighbor
        for idx in component
        for neighbor in adjacency.get(idx, set())
        if neighbor not in component
    })
    adjacent_stock_faces = sorted([
        neighbor for neighbor in boundary_neighbors
        if int(pred_face_map.get(neighbor, {}).get("macro_feature_id", 0)) == 0
    ])
    adjacent_nonstock_faces = sorted([
        neighbor for neighbor in boundary_neighbors
        if int(pred_face_map.get(neighbor, {}).get("macro_feature_id", 0)) > 0
    ])
    adjacent_macro_counter = Counter(
        pred_face_map[neighbor].get("macro_feature", "unknown_feature")
        for neighbor in adjacent_nonstock_faces
        if neighbor in pred_face_map
    )

    bbox_min, bbox_max, bbox_size = bbox_from_centers(graph_nodes)
    centroid = weighted_centroid(graph_nodes)
    macro_conf_mean = sum(float(face.get("macro_confidence", 0.0)) for face in pred_faces) / len(pred_faces)
    semantic_conf_mean = sum(float(face.get("semantic_confidence", 0.0)) for face in pred_faces) / len(pred_faces)
    radii = radius_stats(graph_nodes)
    face_descriptors = [
        face_descriptor(idx, pred_face_map[idx], graph_node_map[idx], ndigits)
        for idx in component
    ]
    selectable_regions = build_selectable_regions(macro_feature, face_descriptors)
    source_metadata = source_feature_metadata(component, feature_set_lookup)
    geometry_parameters = source_metadata["geometry_parameters"]
    fingerprint = {
        "schema_version": "cae_feature_fingerprint_v1",
        "coordinate_frame": "source_step_global",
        "units": "source_model_units",
        "round_ndigits": ndigits,
        "area_total": total_area,
        "area_mean": total_area / max(len(face_areas), 1),
        "area_min": min(face_areas) if face_areas else 0.0,
        "area_max": max(face_areas) if face_areas else 0.0,
        "area_weighted_centroid": centroid,
        "center_bbox_min": bbox_min,
        "center_bbox_max": bbox_max,
        "center_bbox_size": bbox_size,
        "mean_normal": mean_direction(graph_nodes, "normal"),
        "mean_axis": mean_direction(graph_nodes, "axis"),
        "radius_stats": radii,
        "primitive_surface_histogram": dict(primitive_counter),
        "semantic_subtype_histogram": dict(semantic_counter),
        "derived_macro_feature_histogram": dict(derived_counter),
        "idealization_tag_histogram": dict(idealization_counter),
        "adjacent_stock_face_count": len(adjacent_stock_faces),
        "adjacent_nonstock_face_count": len(adjacent_nonstock_faces),
        "adjacent_macro_feature_histogram": dict(adjacent_macro_counter),
        "geometry_parameters": geometry_parameters,
        "face_descriptors": face_descriptors,
    }
    signature_payload = build_signature_payload(
        macro_feature_id=macro_feature_id,
        macro_feature=macro_feature,
        face_count=len(component),
        fingerprint=fingerprint,
        ndigits=ndigits,
    )
    fingerprint["software_agnostic_signature"] = stable_signature(signature_payload)
    fingerprint["software_agnostic_signature_payload"] = signature_payload
    fingerprint["matching_hints"] = matching_hints(macro_feature, fingerprint)

    return {
        "feature_object_id": f"{sample_id}__feature_{object_index:03d}",
        "grouping_method": grouping_method,
        "macro_feature_id": macro_feature_id,
        "macro_feature": macro_feature,
        "source_instance_id": source_metadata["source_instance_id"],
        "source_generator": source_metadata["source_generator"],
        "source_assembly_record": deepcopy(source_assembly_record) if source_assembly_record is not None else None,
        "face_indices": component,
        "face_count": len(component),
        "geometry_parameters": geometry_parameters,
        "raw_parameters": source_metadata["raw_parameters"],
        "selectable_regions": selectable_regions,
        "adjacent_stock_face_indices": adjacent_stock_faces,
        "adjacent_nonstock_face_indices": adjacent_nonstock_faces,
        "confidence": {
            "macro_confidence_mean": macro_conf_mean,
            "semantic_confidence_mean": semantic_conf_mean,
        },
        "fingerprint": fingerprint,
    }


def build_feature_objects(
    prediction_data: dict,
    graph_data: dict,
    ndigits: int = 6,
    grouping_mode: str = CONNECTED_COMPONENT_GROUPING_METHOD,
) -> dict:
    pred_faces = prediction_data.get("faces", [])
    graph_nodes = graph_data.get("nodes", [])
    pred_face_map = {face["idx"]: face for face in pred_faces}
    graph_node_map = {node["idx"]: node for node in graph_nodes}
    adjacency = build_undirected_adjacency(graph_data.get("edges", []), len(graph_nodes))
    feature_set_lookup = build_feature_set_lookup(graph_data.get("feature_sets", []))

    component_entries = component_entries_from_grouping_mode(
        prediction_data,
        pred_faces,
        adjacency,
        grouping_mode,
    )
    sample_id = prediction_data.get("sample_id", Path(prediction_data.get("source_graph_path", "sample")).stem)

    feature_objects = [
        summarize_component(
            sample_id=sample_id,
            object_index=index,
            component=entry["component"],
            pred_face_map=pred_face_map,
            graph_node_map=graph_node_map,
            adjacency=adjacency,
            feature_set_lookup=feature_set_lookup,
            ndigits=ndigits,
            grouping_method=entry["grouping_method"],
            source_assembly_record=entry.get("source_assembly_record"),
        )
        for index, entry in enumerate(component_entries)
    ]
    stock_face_indices = sorted([
        face["idx"] for face in pred_faces
        if int(face.get("macro_feature_id", 0)) == 0
    ])
    return {
        "sample_id": sample_id,
        "graph_mode": "feature_objects",
        "source_step_path": prediction_data.get("source_step_path"),
        "source_graph_path": prediction_data.get("source_graph_path"),
        "source_prediction_path": None,
        "fingerprint_schema": "cae_feature_fingerprint_v1",
        "grouping_mode": grouping_mode,
        "feature_object_count": len(feature_objects),
        "stock_face_indices": stock_face_indices,
        "feature_objects": feature_objects,
    }


def main() -> None:
    args = parse_args()
    prediction_files = collect_prediction_files(args.prediction_path, args.recursive)
    if not prediction_files:
        raise RuntimeError(f"No prediction json files found under: {args.prediction_path}")

    manifest_entries = []
    for prediction_file in prediction_files:
        prediction_data = load_json(prediction_file)
        if "faces" not in prediction_data:
            continue
        graph_path = resolve_graph_path(prediction_data, prediction_file, args.graph_dir)
        graph_data = load_json(graph_path)
        feature_objects = build_feature_objects(
            prediction_data,
            graph_data,
            args.fingerprint_round_ndigits,
            grouping_mode=args.grouping_mode,
        )
        feature_objects["source_prediction_path"] = str(prediction_file.resolve())

        output_path = (
            args.output_dir / prediction_file.relative_to(args.prediction_path)
            if args.prediction_path.is_dir()
            else args.output_dir / prediction_file.name
        )
        write_json(output_path, feature_objects)
        manifest_entries.append({
            "sample_id": feature_objects["sample_id"],
            "prediction_path": str(prediction_file.resolve()),
            "graph_path": str(graph_path.resolve()),
            "feature_object_path": str(output_path.resolve()),
            "feature_object_count": feature_objects["feature_object_count"],
        })
        print(f"feature objects written: {output_path}")

    write_json(
        args.output_dir / "feature_object_manifest.json",
        {
            "sample_count": len(manifest_entries),
            "samples": manifest_entries,
        },
    )


if __name__ == "__main__":
    main()



