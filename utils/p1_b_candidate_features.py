"""Candidate tensors combining face-GNN embeddings and Graph v2 topology."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Mapping

from utils.training_export_v2_1 import (
    build_graph_v2_edge_feature_vector,
    build_graph_v2_node_feature_vector,
)


MACRO_CLASSES = [
    "through_hole_round",
    "blind_hole_round",
    "stepped_hole_round",
    "counterbore_hole",
    "countersink_hole",
    "obround_hole",
    "slot_u",
    "slot_rect",
    "pocket_rect",
    "pocket_circular",
    "cylinder_boss",
    "rib",
    "web",
    "rectangular_boss",
    "fillet",
    "chamfer",
    "slot_v",
    "slot_o",
]
PRIMITIVES = [
    "plane",
    "cylinder",
    "cone",
    "sphere",
    "toroid",
    "bezier_surface",
    "bspline_surface",
    "surface_of_revolution",
    "surface_of_extrusion",
    "offset_surface",
    "unknown_surface",
]
CONVEXITIES = ["concave", "convex", "smooth", "flat", "unknown"]
PROPOSAL_SOURCES = [
    "same_instance_strong_component",
    "coaxial_cylinder_cone_chain",
    "loop_material_side_guided",
    "concave_boundary_region",
    "elongated_swept_channel",
    "thin_wall_rib_web",
    "parent_surface_transition",
]
BASE_ENGINEERED_FEATURES = [
    "log1p_face_count",
    "log1p_canonical_patch_count",
    "candidate_area_over_part_total_area",
    "log1p_boundary_neighbor_count",
    "log1p_internal_edge_count",
    "log1p_boundary_edge_count",
    "inner_loop_face_fraction",
    "outer_loop_face_fraction",
    "mean_inner_loop_count",
    "mean_outer_loop_count",
    "boundary_circle_edge_fraction",
    "internal_concave_edge_fraction",
    "internal_convex_edge_fraction",
    "internal_smooth_edge_fraction",
    "boundary_material_inner_mean",
    "boundary_material_outer_mean",
    "boundary_material_ambiguous_mean",
    "coaxial_curved_pair_fraction",
    "mean_face_area_over_part_total_area",
    "max_face_area_over_part_total_area",
    "mean_pca_long_over_part_scale",
    "max_pca_long_over_part_scale",
    "mean_local_extent_long_middle_ratio_log1p",
    "max_local_extent_long_middle_ratio_log1p",
]
ENGINEERED_FEATURE_SCHEMA = (
    BASE_ENGINEERED_FEATURES
    + [f"primitive_fraction_{name}" for name in PRIMITIVES]
    + [f"provisional_macro_fraction_{name}" for name in MACRO_CLASSES]
    + [f"internal_convexity_fraction_{name}" for name in CONVEXITIES]
    + [f"boundary_convexity_fraction_{name}" for name in CONVEXITIES]
    + [f"proposal_source_{name}" for name in PROPOSAL_SOURCES]
)
MAX_ABS = 32.0


def build_candidate_tensor_record(
    graph_data: Mapping[str, Any],
    prediction_data: Mapping[str, Any],
    proposal: Mapping[str, Any],
) -> dict[str, Any]:
    nodes = _nodes(graph_data)
    face_indices = sorted(set(int(index) for index in proposal.get("face_indices", [])))
    face_set = set(face_indices)
    prediction_faces = _prediction_faces(prediction_data)
    face_embeddings, embedding_source = _candidate_face_embeddings(
        face_indices, nodes, prediction_faces, prediction_data
    )
    internal_edges, boundary_edges, boundary_neighbors = _candidate_edges(graph_data, face_set, prediction_data)
    engineered = _engineered_features(
        graph_data,
        proposal,
        nodes,
        prediction_faces,
        face_set,
        internal_edges,
        boundary_edges,
        boundary_neighbors,
    )
    vector = [_finite_bounded(engineered.get(name, 0.0)) for name in ENGINEERED_FEATURE_SCHEMA]
    return {
        "schema_version": "topocae_p1_b_candidate_tensor_record_1",
        "candidate_id": proposal.get("candidate_id"),
        "face_indices": face_indices,
        "face_embedding_source": embedding_source,
        "face_embedding_dim": len(face_embeddings[0]) if face_embeddings else 0,
        "edge_embedding_source": "face_gnn_edge_embedding" if _has_edge_embeddings(prediction_data) else "graph_v2_edge_features",
        "edge_embedding_dim": len(internal_edges[0]["embedding"]) if internal_edges else (
            len(boundary_edges[0]["embedding"]) if boundary_edges else 0
        ),
        "face_embeddings": face_embeddings,
        "internal_edge_embeddings": [record["embedding"] for record in internal_edges],
        "boundary_edge_embeddings": [record["embedding"] for record in boundary_edges],
        "engineered_feature_schema": list(ENGINEERED_FEATURE_SCHEMA),
        "engineered_features": vector,
        "engineered_feature_values": engineered,
        "boundary_neighbor_indices": sorted(boundary_neighbors),
        "trace": {
            "proposal_sources": list(proposal.get("proposal_sources", [])),
            "original_face_ids_preserved": True,
            "all_features_finite": all(math.isfinite(value) for value in vector),
            "max_abs": max((abs(value) for value in vector), default=0.0),
        },
    }


def _candidate_face_embeddings(
    face_indices: list[int],
    nodes: Mapping[int, Mapping[str, Any]],
    prediction_faces: Mapping[int, Mapping[str, Any]],
    prediction_data: Mapping[str, Any],
) -> tuple[list[list[float]], str]:
    aligned = prediction_data.get("node_embeddings")
    result = []
    source = "graph_v2_node_features"
    for index in face_indices:
        embedding = prediction_faces.get(index, {}).get("embedding")
        if embedding is None and isinstance(aligned, (list, tuple)) and 0 <= index < len(aligned):
            embedding = aligned[index]
        if isinstance(embedding, (list, tuple)):
            source = "face_gnn_node_embedding"
            result.append([_finite_bounded(value) for value in embedding])
        else:
            result.append([_finite_bounded(value) for value in build_graph_v2_node_feature_vector(dict(nodes[index]))])
    if result:
        dimensions = {len(vector) for vector in result}
        if len(dimensions) != 1:
            raise ValueError("candidate face embeddings must use one fixed dimension")
    return result, source


def _candidate_edges(
    graph: Mapping[str, Any],
    face_set: set[int],
    prediction: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[int]]:
    edge_embeddings = prediction.get("edge_embeddings")
    by_pair: dict[tuple[int, int], list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for position, (edge, attribute) in enumerate(zip(graph.get("edges", []), graph.get("edge_attributes", []))):
        if not isinstance(edge, (list, tuple)) or len(edge) < 2:
            continue
        pair = tuple(sorted((int(edge[0]), int(edge[1]))))
        by_pair[pair].append((position, attribute))
    internal, boundary = [], []
    boundary_neighbors = set()
    for pair, records in sorted(by_pair.items()):
        left_in, right_in = pair[0] in face_set, pair[1] in face_set
        if not left_in and not right_in:
            continue
        embeddings = []
        for position, attribute in records:
            if isinstance(edge_embeddings, (list, tuple)) and position < len(edge_embeddings):
                raw_embedding = edge_embeddings[position]
                if isinstance(raw_embedding, (list, tuple)):
                    embeddings.append([_finite_bounded(value) for value in raw_embedding])
                    continue
            embeddings.append([_finite_bounded(value) for value in build_graph_v2_edge_feature_vector(dict(attribute))])
        embedding = _mean_vectors(embeddings)
        item = {"pair": pair, "attributes": [record[1] for record in records], "embedding": embedding}
        if left_in and right_in:
            internal.append(item)
        else:
            boundary.append(item)
            boundary_neighbors.add(pair[1] if left_in else pair[0])
    return internal, boundary, boundary_neighbors


def _engineered_features(
    graph: Mapping[str, Any],
    proposal: Mapping[str, Any],
    nodes: Mapping[int, Mapping[str, Any]],
    prediction_faces: Mapping[int, Mapping[str, Any]],
    face_set: set[int],
    internal_edges: list[dict[str, Any]],
    boundary_edges: list[dict[str, Any]],
    boundary_neighbors: set[int],
) -> dict[str, float]:
    candidate_nodes = [nodes[index] for index in sorted(face_set) if index in nodes]
    total_area = float(graph.get("part_metadata_v2", {}).get("normalization", {}).get("part_total_area", 0.0) or 0.0)
    areas = [float(node.get("attributes", {}).get("raw_physical_v2", {}).get("area", 0.0) or 0.0) for node in candidate_nodes]
    recognitions = [node.get("attributes", {}).get("recognition_features_v2", {}) for node in candidate_nodes]
    area_ratios = [float(record.get("face_area_over_part_total_area", 0.0) or 0.0) for record in recognitions]
    pca_longs = [float(record.get("pca_long_over_part_scale", 0.0) or 0.0) for record in recognitions]
    aspect_logs = [math.log1p(max(0.0, float(record.get("local_extent_long_middle_ratio", 0.0) or 0.0))) for record in recognitions]
    topology = [node.get("attributes", {}).get("topology_v2", {}) for node in candidate_nodes]
    patch_map = graph.get("face_to_canonical_patch", {})
    patch_ids = {patch_map.get(str(index), patch_map.get(index, index)) for index in face_set}
    primitive_counts = Counter(str(node.get("primitive_surface", "unknown_surface")) for node in candidate_nodes)
    macro_counts = Counter(
        str(prediction_faces.get(index, {}).get("macro_feature", prediction_faces.get(index, {}).get("raw_macro_feature", "stock")))
        for index in face_set
    )
    internal_convexity = _edge_convexity_counts(internal_edges)
    boundary_convexity = _edge_convexity_counts(boundary_edges)
    material_inner = _edge_recognition_values(boundary_edges, "material_side_inner_evidence")
    material_outer = _edge_recognition_values(boundary_edges, "material_side_outer_evidence")
    material_ambiguous = _edge_recognition_values(boundary_edges, "material_side_ambiguous_evidence")
    result = {
        "log1p_face_count": math.log1p(len(face_set)),
        "log1p_canonical_patch_count": math.log1p(len(patch_ids)),
        "candidate_area_over_part_total_area": sum(areas) / total_area if total_area > 1e-12 else 0.0,
        "log1p_boundary_neighbor_count": math.log1p(len(boundary_neighbors)),
        "log1p_internal_edge_count": math.log1p(len(internal_edges)),
        "log1p_boundary_edge_count": math.log1p(len(boundary_edges)),
        "inner_loop_face_fraction": _fraction(sum(int(record.get("inner_loop_count", 0) or 0) > 0 for record in topology), len(topology)),
        "outer_loop_face_fraction": _fraction(sum(int(record.get("outer_loop_count", 0) or 0) > 0 for record in topology), len(topology)),
        "mean_inner_loop_count": _mean([float(record.get("inner_loop_count", 0) or 0) for record in topology]),
        "mean_outer_loop_count": _mean([float(record.get("outer_loop_count", 0) or 0) for record in topology]),
        "boundary_circle_edge_fraction": _edge_curve_fraction(boundary_edges, "circle"),
        "internal_concave_edge_fraction": _fraction(internal_convexity["concave"], len(internal_edges)),
        "internal_convex_edge_fraction": _fraction(internal_convexity["convex"], len(internal_edges)),
        "internal_smooth_edge_fraction": _fraction(internal_convexity["smooth"], len(internal_edges)),
        "boundary_material_inner_mean": _mean(material_inner),
        "boundary_material_outer_mean": _mean(material_outer),
        "boundary_material_ambiguous_mean": _mean(material_ambiguous),
        "coaxial_curved_pair_fraction": _coaxial_pair_fraction(candidate_nodes, _part_scale(graph)),
        "mean_face_area_over_part_total_area": _mean(area_ratios),
        "max_face_area_over_part_total_area": max(area_ratios, default=0.0),
        "mean_pca_long_over_part_scale": _mean(pca_longs),
        "max_pca_long_over_part_scale": max(pca_longs, default=0.0),
        "mean_local_extent_long_middle_ratio_log1p": _mean(aspect_logs),
        "max_local_extent_long_middle_ratio_log1p": max(aspect_logs, default=0.0),
    }
    denominator = max(1, len(candidate_nodes))
    for primitive in PRIMITIVES:
        result[f"primitive_fraction_{primitive}"] = primitive_counts[primitive] / denominator
    for macro in MACRO_CLASSES:
        result[f"provisional_macro_fraction_{macro}"] = macro_counts[macro] / max(1, len(face_set))
    for convexity in CONVEXITIES:
        result[f"internal_convexity_fraction_{convexity}"] = _fraction(internal_convexity[convexity], len(internal_edges))
        result[f"boundary_convexity_fraction_{convexity}"] = _fraction(boundary_convexity[convexity], len(boundary_edges))
    sources = set(str(value) for value in proposal.get("proposal_sources", []))
    for source in PROPOSAL_SOURCES:
        result[f"proposal_source_{source}"] = float(source in sources)
    return {key: _finite_bounded(value) for key, value in result.items()}


def _edge_convexity_counts(edges: list[dict[str, Any]]) -> Counter:
    counts = Counter()
    for edge in edges:
        roles = {
            str(attribute.get("convexity", "unknown"))
            for attribute in edge.get("attributes", [])
            if isinstance(attribute, Mapping)
        }
        for role in roles or {"unknown"}:
            counts[role if role in CONVEXITIES else "unknown"] += 1
    return counts


def _edge_recognition_values(edges: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for edge in edges:
        for attribute in edge.get("attributes", []):
            values.append(float(attribute.get("recognition_features_v2", {}).get(key, 0.0) or 0.0))
    return values


def _edge_curve_fraction(edges: list[dict[str, Any]], curve: str) -> float:
    count = sum(
        any(str(attribute.get("curve_type", "")) == curve for attribute in edge.get("attributes", []))
        for edge in edges
    )
    return _fraction(count, len(edges))


def _coaxial_pair_fraction(nodes: list[Mapping[str, Any]], scale: float) -> float:
    curved = [node for node in nodes if str(node.get("primitive_surface", "")) in {"cylinder", "cone"}]
    pair_count = coaxial_count = 0
    for left_position, left in enumerate(curved):
        for right in curved[left_position + 1:]:
            pair_count += 1
            coaxial_count += int(_coaxial(left, right, scale))
    return _fraction(coaxial_count, pair_count)


def _coaxial(left: Mapping[str, Any], right: Mapping[str, Any], scale: float) -> bool:
    left_axis = _axis(left)
    right_axis = _axis(right)
    if left_axis is None or right_axis is None:
        return False
    left_direction, left_location = left_axis
    right_direction, right_location = right_axis
    dot = abs(sum(a * b for a, b in zip(left_direction, right_direction)))
    if dot < math.cos(math.radians(2.0)):
        return False
    delta = [b - a for a, b in zip(left_location, right_location)]
    cross = [
        delta[1] * left_direction[2] - delta[2] * left_direction[1],
        delta[2] * left_direction[0] - delta[0] * left_direction[2],
        delta[0] * left_direction[1] - delta[1] * left_direction[0],
    ]
    return math.sqrt(sum(value * value for value in cross)) / max(scale, 1e-12) <= 1e-5


def _axis(node: Mapping[str, Any]) -> tuple[list[float], list[float]] | None:
    geometry = node.get("attributes", {}).get("geometry_v2", {})
    direction = geometry.get("axis_direction")
    location = geometry.get("axis_location")
    if not isinstance(direction, (list, tuple)) or len(direction) < 3:
        return None
    if not isinstance(location, (list, tuple)) or len(location) < 3:
        return None
    values = [float(value) for value in direction[:3]]
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        return None
    return [value / norm for value in values], [float(value) for value in location[:3]]


def _part_scale(graph: Mapping[str, Any]) -> float:
    return float(graph.get("part_metadata_v2", {}).get("normalization", {}).get("part_characteristic_length", 1.0) or 1.0)


def _nodes(graph: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {int(node.get("idx", position)): node for position, node in enumerate(graph.get("nodes", []))}


def _prediction_faces(prediction: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {int(face.get("idx", position)): face for position, face in enumerate(prediction.get("faces", []))}


def _has_edge_embeddings(prediction: Mapping[str, Any]) -> bool:
    return isinstance(prediction.get("edge_embeddings"), (list, tuple)) and bool(prediction.get("edge_embeddings"))


def _mean_vectors(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError("edge embeddings must use one fixed dimension")
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(len(vectors[0]))]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _fraction(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _finite_bounded(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return min(MAX_ABS, max(-MAX_ABS, number))


__all__ = [
    "ENGINEERED_FEATURE_SCHEMA",
    "MACRO_CLASSES",
    "MAX_ABS",
    "build_candidate_tensor_record",
]
