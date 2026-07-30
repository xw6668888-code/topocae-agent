from __future__ import annotations

from build_rib_boundary_classifier_dataset import (
    adjacent_faces,
    count_edges,
    edge_curve,
    edge_length,
    face_area,
    iter_edge_attributes,
    node_index,
    normal_vector,
    safe_div,
    shape_descriptors,
)


PROPOSAL_MACROS = [
    "stock",
    "rib",
    "web",
    "pocket_rect",
    "pocket_circular",
    "rectangular_boss",
    "cylinder_boss",
    "slot_u",
    "slot_o",
    "fillet",
    "chamfer",
]


def extract_rib_candidate_features(
    graph_data: dict,
    candidate_face_idx: int,
    *,
    prediction_face: dict | None = None,
) -> dict[str, float]:
    node_by_idx = node_index(graph_data)
    candidate_idx = int(candidate_face_idx)
    candidate = node_by_idx.get(candidate_idx, {})
    model_area = sum(max(face_area(node), 0.0) for node in node_by_idx.values())
    candidate_area = face_area(candidate)
    descriptors = shape_descriptors(candidate)
    primitive = str(candidate.get("primitive_surface") or "").lower()
    adjacent = adjacent_faces(graph_data, {candidate_idx})
    adjacent_nodes = [node_by_idx[idx] for idx in sorted(adjacent) if idx in node_by_idx]
    adjacent_areas = [face_area(node) for node in adjacent_nodes]
    incident_edges = face_incident_edges(graph_data, candidate_idx)

    normal = normal_vector(candidate)
    neighbor_parallel_count = 0
    neighbor_orthogonal_count = 0
    if normal is not None:
        for neighbor in adjacent_nodes:
            neighbor_normal = normal_vector(neighbor)
            if neighbor_normal is None:
                continue
            dot = abs(sum(left * right for left, right in zip(normal, neighbor_normal)))
            if dot >= 0.92:
                neighbor_parallel_count += 1
            if dot <= 0.18:
                neighbor_orthogonal_count += 1

    max_adjacent_area = max(adjacent_areas, default=0.0)
    mean_adjacent_area = safe_div(sum(adjacent_areas), len(adjacent_areas))
    feature = {
        "candidate_area": float(candidate_area),
        "candidate_area_ratio": safe_div(candidate_area, model_area),
        "candidate_is_plane": 1.0 if primitive == "plane" else 0.0,
        "candidate_is_cylinder": 1.0 if primitive == "cylinder" else 0.0,
        "candidate_is_cone": 1.0 if primitive == "cone" else 0.0,
        "candidate_is_curved": 1.0 if primitive in {"cylinder", "cone", "sphere", "torus"} else 0.0,
        "candidate_projected_aspect_ratio": float(descriptors.get("projected_aspect_ratio", 0.0) or 0.0),
        "candidate_bbox_aspect_ratio": float(descriptors.get("bbox_aspect_ratio", 0.0) or 0.0),
        "candidate_bbox_longest_extent": float(descriptors.get("bbox_longest_extent", 0.0) or 0.0),
        "candidate_bbox_shortest_extent": float(descriptors.get("bbox_shortest_extent", 0.0) or 0.0),
        "candidate_neighbor_count": float(len(adjacent)),
        "candidate_neighbor_area_max": float(max_adjacent_area),
        "candidate_neighbor_area_mean": float(mean_adjacent_area),
        "candidate_neighbor_area_max_ratio": safe_div(max_adjacent_area, model_area),
        "candidate_neighbor_area_mean_ratio": safe_div(mean_adjacent_area, model_area),
        "candidate_to_largest_neighbor_area_ratio": safe_div(candidate_area, max_adjacent_area),
        "candidate_large_neighbor_count": float(sum(1 for area in adjacent_areas if area >= candidate_area * 4.0)),
        "candidate_parallel_neighbor_plane_count": float(neighbor_parallel_count),
        "candidate_orthogonal_neighbor_plane_count": float(neighbor_orthogonal_count),
        "candidate_incident_edge_count": float(len(incident_edges)),
        "candidate_line_edge_count": float(count_edges(incident_edges, "curve_type", "line")),
        "candidate_circle_edge_count": float(count_edges(incident_edges, "curve_type", "circle")),
        "candidate_convex_edge_count": float(count_edges(incident_edges, "convexity", "convex")),
        "candidate_concave_edge_count": float(count_edges(incident_edges, "convexity", "concave")),
        "candidate_smooth_edge_count": float(count_edges(incident_edges, "convexity", "smooth")),
        "candidate_total_edge_length": float(sum(edge_length(edge) for edge in incident_edges)),
        "candidate_max_edge_length": float(max((edge_length(edge) for edge in incident_edges), default=0.0)),
        "candidate_max_line_edge_length": float(max((edge_length(edge) for edge in incident_edges if edge_curve(edge) == "line"), default=0.0)),
    }
    feature["candidate_edge_length_to_area_ratio"] = safe_div(
        feature["candidate_total_edge_length"],
        max(candidate_area, 1e-12) ** 0.5,
    )
    proposal_macro = str((prediction_face or {}).get("macro_feature") or "")
    for macro in PROPOSAL_MACROS:
        feature[f"proposal_macro_{macro}"] = 1.0 if proposal_macro == macro else 0.0
    return feature


def face_incident_edges(graph_data: dict, candidate_face_idx: int) -> list[dict]:
    candidate_face_idx = int(candidate_face_idx)
    return [
        attr
        for attr in iter_edge_attributes(graph_data)
        if int(attr.get("source_face", -1)) == candidate_face_idx
        or int(attr.get("target_face", -1)) == candidate_face_idx
    ]
