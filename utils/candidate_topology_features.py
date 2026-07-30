from __future__ import annotations

import math
from itertools import combinations

from utils.topology_rules import validate_feature_topology


PRIMITIVE_KEYS = {
    "plane": "plane_count",
    "cylinder": "cylinder_count",
    "cone": "cone_count",
    "toroid": "toroid_count",
    "surface_of_revolution": "surface_of_revolution_count",
    "surface_of_extrusion": "surface_of_extrusion_count",
    "bspline_surface": "bspline_surface_count",
}


def extract_candidate_topology_features(
    graph_data: dict,
    face_indices: list[int],
    *,
    prediction_faces: list[dict] | None = None,
    candidate_macro: str | None = None,
) -> dict[str, float]:
    nodes = _nodes_by_idx(graph_data)
    face_set = {int(idx) for idx in face_indices}
    prediction_by_idx = {int(face["idx"]): face for face in prediction_faces or []}
    candidate_nodes = [nodes[idx] for idx in sorted(face_set) if idx in nodes]
    total_area = sum(_node_area(node) for node in nodes.values())
    candidate_area = sum(_node_area(node) for node in candidate_nodes)

    features = {
        "face_count": float(len(face_set)),
        "candidate_area": candidate_area,
        "model_area": total_area,
        "area_ratio": candidate_area / total_area if total_area > 1e-9 else 0.0,
        "boundary_neighbor_count": 0.0,
        "topology_valid": 0.0,
        "raw_stock_face_fraction": _raw_macro_fraction(prediction_by_idx, face_set, "stock"),
        "candidate_macro_face_fraction": _macro_fraction(prediction_by_idx, face_set, candidate_macro),
        "cylindrical_angle_estimate_max": 0.0,
    }
    for key in set(PRIMITIVE_KEYS.values()):
        features[key] = 0.0
    for node in candidate_nodes:
        primitive = node.get("primitive_surface", "")
        if primitive in PRIMITIVE_KEYS:
            features[PRIMITIVE_KEYS[primitive]] += 1.0
        if primitive in {"cylinder", "surface_of_revolution", "surface_of_extrusion", "bspline_surface", "toroid"}:
            features["curved_surface_count"] = features.get("curved_surface_count", 0.0) + 1.0

    internal_edges, boundary_edges, boundary_neighbors = _split_unique_edges(graph_data, face_set)
    _add_edge_features(features, internal_edges, "internal")
    _add_edge_features(features, boundary_edges, "boundary")
    features["boundary_neighbor_count"] = float(len(boundary_neighbors))
    features.update(_boundary_parent_plane_relation_features(nodes, boundary_neighbors))
    features["cylindrical_angle_estimate_max"] = _max_cylindrical_angle_estimate(nodes, face_set, graph_data)
    if candidate_macro:
        features["topology_valid"] = 1.0 if validate_feature_topology(candidate_macro, sorted(face_set), graph_data)["valid"] else 0.0
    return features


def _nodes_by_idx(graph_data: dict) -> dict[int, dict]:
    return {int(node["idx"]): node for node in graph_data.get("nodes", [])}


def _node_area(node: dict) -> float:
    return float(node.get("attributes", {}).get("physical", {}).get("area", 0.0) or 0.0)


def _node_radius(node: dict) -> float:
    return float(node.get("attributes", {}).get("specific", {}).get("radius", 0.0) or 0.0)


def _node_normal(node: dict) -> list[float]:
    normal = node.get("attributes", {}).get("specific", {}).get("normal", [0.0, 0.0, 0.0])
    return _normalize_vector([float(value) for value in normal[:3]])


def _raw_macro_fraction(prediction_by_idx: dict[int, dict], face_set: set[int], macro: str) -> float:
    if not face_set:
        return 0.0
    count = 0
    for idx in face_set:
        face = prediction_by_idx.get(idx, {})
        if face.get("raw_macro_feature") == macro:
            count += 1
    return count / len(face_set)


def _macro_fraction(prediction_by_idx: dict[int, dict], face_set: set[int], macro: str | None) -> float:
    if not face_set or not macro:
        return 0.0
    count = 0
    for idx in face_set:
        face = prediction_by_idx.get(idx, {})
        if face.get("macro_feature") == macro or face.get("raw_macro_feature") == macro:
            count += 1
    return count / len(face_set)


def _split_unique_edges(graph_data: dict, face_set: set[int]) -> tuple[list[dict], list[dict], set[int]]:
    internal = []
    boundary = []
    boundary_neighbors = set()
    seen = set()
    for edge, attr in zip(graph_data.get("edges", []), graph_data.get("edge_attributes", [])):
        if len(edge) < 2:
            continue
        src, dst = int(edge[0]), int(edge[1])
        src_in = src in face_set
        dst_in = dst in face_set
        if not src_in and not dst_in:
            continue
        key = (
            min(src, dst),
            max(src, dst),
            str(attr.get("curve_type", "")),
            str(attr.get("convexity", "")),
            round(float(attr.get("length", 0.0) or 0.0), 6),
        )
        if key in seen:
            continue
        seen.add(key)
        item = dict(attr)
        item.setdefault("source_face", src)
        item.setdefault("target_face", dst)
        if src_in and dst_in:
            internal.append(item)
        else:
            boundary.append(item)
            boundary_neighbors.add(dst if src_in else src)
    return internal, boundary, boundary_neighbors


def _add_edge_features(features: dict[str, float], edges: list[dict], prefix: str) -> None:
    features[f"{prefix}_edge_count"] = float(len(edges))
    line_lengths = [float(edge.get("length", 0.0) or 0.0) for edge in edges if _curve_is(edge, "line")]
    positive_line_lengths = [length for length in line_lengths if length > 1e-9]
    longest_line = max(positive_line_lengths) if positive_line_lengths else 0.0
    features[f"{prefix}_line_edge_count"] = float(sum(1 for edge in edges if _curve_is(edge, "line")))
    features[f"{prefix}_circle_edge_count"] = float(sum(1 for edge in edges if _curve_is(edge, "circle")))
    features[f"{prefix}_convex_edge_count"] = float(sum(1 for edge in edges if _convexity_is(edge, "convex")))
    features[f"{prefix}_concave_edge_count"] = float(sum(1 for edge in edges if _convexity_is(edge, "concave")))
    features[f"{prefix}_smooth_edge_count"] = float(sum(1 for edge in edges if _convexity_is(edge, "smooth")))
    features[f"{prefix}_max_line_length"] = longest_line
    features[f"{prefix}_long_line_edge_count"] = float(
        sum(1 for length in positive_line_lengths if longest_line > 0.0 and length >= 0.75 * longest_line)
    )


def _boundary_parent_plane_relation_features(nodes: dict[int, dict], boundary_neighbors: set[int]) -> dict[str, float]:
    plane_normals = [
        _node_normal(nodes[idx])
        for idx in sorted(boundary_neighbors)
        if nodes.get(idx, {}).get("primitive_surface") == "plane"
    ]
    features = {
        "boundary_parent_plane_count": float(len(plane_normals)),
        "boundary_parent_plane_parallel_pair_count": 0.0,
        "boundary_parent_plane_orthogonal_pair_count": 0.0,
        "boundary_parent_plane_intersecting_pair_count": 0.0,
    }
    for left, right in combinations(plane_normals, 2):
        angle = _angle_between_abs(left, right)
        if angle <= 0.18:
            features["boundary_parent_plane_parallel_pair_count"] += 1.0
        elif abs(angle - math.pi / 2.0) <= 0.25:
            features["boundary_parent_plane_orthogonal_pair_count"] += 1.0
        else:
            features["boundary_parent_plane_intersecting_pair_count"] += 1.0
    return features


def _max_cylindrical_angle_estimate(nodes: dict[int, dict], face_set: set[int], graph_data: dict) -> float:
    max_angle = 0.0
    for idx in face_set:
        node = nodes.get(idx, {})
        if node.get("primitive_surface") != "cylinder":
            continue
        radius = _node_radius(node)
        area = _node_area(node)
        if radius <= 1e-9 or area <= 1e-9:
            continue
        line_lengths = []
        for edge, attr in zip(graph_data.get("edges", []), graph_data.get("edge_attributes", [])):
            if idx not in {int(edge[0]), int(edge[1])} or not _curve_is(attr, "line"):
                continue
            length = float(attr.get("length", 0.0) or 0.0)
            if length > 1e-9:
                line_lengths.append(length)
        if line_lengths:
            max_angle = max(max_angle, area / (radius * max(line_lengths)))
    return max_angle


def _curve_is(edge: dict, curve_type: str) -> bool:
    return edge.get("curve_type") == curve_type


def _convexity_is(edge: dict, convexity: str) -> bool:
    return edge.get("convexity") == convexity


def _angle_between_abs(left: list[float], right: list[float]) -> float:
    if _vector_norm(left) <= 1e-12 or _vector_norm(right) <= 1e-12:
        return math.pi
    dot = max(-1.0, min(1.0, abs(sum(a * b for a, b in zip(left, right)))))
    return math.acos(dot)


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = _vector_norm(vector)
    if norm <= 1e-12:
        return [0.0, 0.0, 0.0]
    return [value / norm for value in vector]


def _vector_norm(vector: list[float]) -> float:
    return sum(value * value for value in vector) ** 0.5
