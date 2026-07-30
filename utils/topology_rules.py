from __future__ import annotations

import math
from itertools import combinations


COAXIAL_MULTI_RADIUS_HOLE_MACROS = {
    "stepped_hole_round",
    "counterbore_hole",
}

TRANSITION_MACROS = {"fillet", "chamfer"}

RIB_WEB_MACROS = {"rib", "web", "rib_web_candidate"}


def validate_feature_topology(macro_feature: str, face_indices: list[int], graph_data: dict) -> dict:
    face_indices = sorted({int(idx) for idx in face_indices})
    nodes = {int(node["idx"]): node for node in graph_data.get("nodes", [])}
    internal_edges = _internal_edge_attributes(face_indices, graph_data)
    boundary_edges = _boundary_edge_attributes(face_indices, graph_data)
    result = {
        "macro_feature": macro_feature,
        "normalized_family": _normalized_family(macro_feature),
        "face_indices": face_indices,
        "valid": True,
        "reasons": [],
        "evidence": [],
        "may_attach_to_parent_feature": macro_feature in TRANSITION_MACROS,
    }

    if not face_indices:
        return _invalid(result, "empty_feature")

    if macro_feature == "through_hole_round":
        _validate_through_hole(result, face_indices, nodes, boundary_edges)
    elif macro_feature == "blind_hole_round":
        _validate_blind_hole(result, face_indices, nodes, internal_edges)
    elif macro_feature == "pocket_circular":
        _validate_blind_hole(result, face_indices, nodes, internal_edges, allow_shallow=True)
    elif macro_feature == "countersink_hole":
        _validate_countersink_hole(result, face_indices, nodes, internal_edges)
    elif macro_feature == "obround_hole":
        _validate_obround_hole(result, face_indices, nodes, internal_edges)
    elif macro_feature == "counterbore_hole":
        _validate_counterbore_hole(result, face_indices, nodes, internal_edges)
    elif macro_feature in COAXIAL_MULTI_RADIUS_HOLE_MACROS:
        _validate_coaxial_multi_radius_hole(result, face_indices, nodes, internal_edges)
    elif macro_feature == "slot_rect":
        _validate_rectangular_slot(result, face_indices, nodes, internal_edges)
    elif macro_feature == "slot_v":
        _validate_v_slot(result, face_indices, nodes, internal_edges)
    elif macro_feature == "pocket_rect":
        _validate_rectangular_pocket(result, face_indices, nodes, internal_edges)
    elif macro_feature == "rectangular_boss":
        _validate_rectangular_boss(result, face_indices, nodes, internal_edges)
    elif macro_feature == "cylinder_boss":
        _validate_cylinder_boss(result, face_indices, nodes, internal_edges)
    elif macro_feature in RIB_WEB_MACROS:
        _validate_rib_web(result, face_indices, nodes)
    elif macro_feature in TRANSITION_MACROS:
        if len(face_indices) >= 1:
            result["evidence"].append("transition_face_candidate")

    return result


def _normalized_family(macro_feature: str) -> str:
    if macro_feature in COAXIAL_MULTI_RADIUS_HOLE_MACROS:
        return "coaxial_multi_radius_hole"
    if macro_feature in RIB_WEB_MACROS:
        return "rib_web_candidate"
    return macro_feature


def _invalid(result: dict, reason: str) -> dict:
    result["valid"] = False
    if reason not in result["reasons"]:
        result["reasons"].append(reason)
    return result


def _internal_edge_attributes(face_indices: list[int], graph_data: dict) -> list[dict]:
    face_set = set(face_indices)
    attrs = []
    edges = graph_data.get("edges", [])
    edge_attrs = graph_data.get("edge_attributes", [])
    for index, edge in enumerate(edges):
        if len(edge) != 2:
            continue
        src, dst = int(edge[0]), int(edge[1])
        if src not in face_set or dst not in face_set:
            continue
        attr = dict(edge_attrs[index]) if index < len(edge_attrs) else {}
        attr.setdefault("source_face", src)
        attr.setdefault("target_face", dst)
        attrs.append(attr)
    return attrs


def _boundary_edge_attributes(face_indices: list[int], graph_data: dict) -> list[dict]:
    face_set = set(face_indices)
    attrs = []
    edges = graph_data.get("edges", [])
    edge_attrs = graph_data.get("edge_attributes", [])
    for index, edge in enumerate(edges):
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            continue
        src, dst = int(edge[0]), int(edge[1])
        if (src in face_set) == (dst in face_set):
            continue
        attr = dict(edge_attrs[index]) if index < len(edge_attrs) else {}
        attr.setdefault("source_face", src)
        attr.setdefault("target_face", dst)
        attrs.append(attr)
    return attrs


def _primitive_counts(face_indices: list[int], nodes: dict[int, dict]) -> dict[str, int]:
    counts = {}
    for idx in face_indices:
        primitive = nodes.get(idx, {}).get("primitive_surface", "unknown_surface")
        counts[primitive] = counts.get(primitive, 0) + 1
    return counts


def _node_specific(node: dict, key: str, default=None):
    return node.get("attributes", {}).get("specific", {}).get(key, default)


def _node_descriptor(node: dict, key: str, default=0.0) -> float:
    return float(node.get("attributes", {}).get("shape_descriptors", {}).get(key, default))


def _node_center(node: dict) -> list[float]:
    return list(node.get("attributes", {}).get("physical", {}).get("center_of_mass", [0.0, 0.0, 0.0]))


def _dot(a, b) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _norm(v) -> float:
    return math.sqrt(_dot(v, v))


def _normalize(v) -> list[float]:
    length = _norm(v)
    if length <= 1e-12:
        return [0.0, 0.0, 0.0]
    return [float(x) / length for x in v]


def _angle_between(a, b) -> float:
    a = _normalize(a)
    b = _normalize(b)
    if _norm(a) <= 1e-12 or _norm(b) <= 1e-12:
        return math.pi
    return math.acos(max(-1.0, min(1.0, abs(_dot(a, b)))))


def _validate_blind_hole(
    result: dict,
    face_indices: list[int],
    nodes: dict[int, dict],
    internal_edges: list[dict],
    allow_shallow: bool = False,
) -> None:
    counts = _primitive_counts(face_indices, nodes)
    if counts.get("cylinder", 0) < 1:
        _invalid(result, "missing_cylindrical_wall")
    if counts.get("plane", 0) < 1:
        _invalid(result, "missing_bottom_face")
    if not _has_internal_circular_edge(internal_edges):
        _invalid(result, "missing_wall_bottom_circular_edge")
    if result["valid"]:
        result["evidence"].append("cylinder_wall_plus_bottom_face")
        if allow_shallow:
            result["evidence"].append("shallow_circular_recess_family")


def _validate_through_hole(
    result: dict,
    face_indices: list[int],
    nodes: dict[int, dict],
    boundary_edges: list[dict],
) -> None:
    counts = _primitive_counts(face_indices, nodes)
    if counts.get("cylinder", 0) != 1 or len(face_indices) != 1:
        _invalid(result, "single_cylindrical_wall_required")
    if counts.get("plane", 0) or counts.get("cone", 0):
        _invalid(result, "through_hole_should_not_include_bottom_or_countersink_faces")
    circular_openings = _count_unique_edges(boundary_edges, _is_circular_edge)
    if circular_openings < 2:
        _invalid(result, "requires_two_circular_opening_edges")
    if result["valid"]:
        result["evidence"].append("single_cylindrical_wall_with_two_openings")


def _validate_countersink_hole(
    result: dict,
    face_indices: list[int],
    nodes: dict[int, dict],
    internal_edges: list[dict],
) -> None:
    counts = _primitive_counts(face_indices, nodes)
    if counts.get("cone", 0) < 1:
        _invalid(result, "missing_conical_countersink_face")
    if counts.get("cylinder", 0) < 1:
        _invalid(result, "missing_small_cylindrical_wall")
    if not _has_internal_circular_edge(internal_edges):
        _invalid(result, "missing_cone_cylinder_circular_edge")
    if result["valid"]:
        result["evidence"].append("cone_plus_small_cylindrical_wall")


def _validate_obround_hole(
    result: dict,
    face_indices: list[int],
    nodes: dict[int, dict],
    internal_edges: list[dict],
) -> None:
    counts = _primitive_counts(face_indices, nodes)
    if counts.get("plane", 0) < 2:
        _invalid(result, "missing_obround_side_planes")
    if counts.get("cylinder", 0) < 2:
        _invalid(result, "missing_obround_end_cylinders")

    concave_wall_junctions = _count_unique_edges(
        internal_edges,
        lambda edge: _is_line_edge(edge) and _is_concave_edge(edge),
    )
    if concave_wall_junctions < 4:
        _invalid(result, "missing_concave_obround_wall_junctions")

    cylinders = [
        nodes[idx]
        for idx in face_indices
        if nodes.get(idx, {}).get("primitive_surface") == "cylinder"
    ]
    if len(cylinders) >= 2:
        axes = [_node_specific(node, "axis", [0.0, 0.0, 0.0]) for node in cylinders]
        reference = axes[0]
        if any(_angle_between(reference, axis) > math.radians(8.0) for axis in axes[1:]):
            _invalid(result, "obround_end_cylinder_axes_not_parallel")
        radii = [float(_node_specific(node, "radius", 0.0) or 0.0) for node in cylinders]
        positive_radii = [radius for radius in radii if radius > 1e-9]
        if positive_radii and max(positive_radii) / max(min(positive_radii), 1e-9) > 1.2:
            _invalid(result, "obround_end_cylinder_radii_not_consistent")

    if result["valid"]:
        result["evidence"].append("planar_sides_and_cylindrical_ends_with_concave_wall_junctions")


def _validate_coaxial_multi_radius_hole(
    result: dict,
    face_indices: list[int],
    nodes: dict[int, dict],
    internal_edges: list[dict],
) -> None:
    cylinders = [
        nodes[idx]
        for idx in face_indices
        if nodes.get(idx, {}).get("primitive_surface") == "cylinder"
    ]
    planes = [
        nodes[idx]
        for idx in face_indices
        if nodes.get(idx, {}).get("primitive_surface") == "plane"
    ]
    if len(cylinders) < 2:
        _invalid(result, "min_2_coaxial_cylinders_required")
        return
    if not planes:
        _invalid(result, "missing_shoulder_or_transition_plane")

    axes = [_node_specific(node, "axis", [0.0, 0.0, 0.0]) for node in cylinders]
    reference = axes[0]
    if any(_angle_between(reference, axis) > math.radians(8.0) for axis in axes[1:]):
        _invalid(result, "cylinder_axes_not_coaxial")

    radii = sorted({
        round(float(_node_specific(node, "radius", 0.0)), 6)
        for node in cylinders
    })
    if len(radii) < 2:
        _invalid(result, "multi_radius_difference_missing")
    if not _has_internal_circular_edge(internal_edges):
        _invalid(result, "missing_circular_step_edges")
    if result["valid"]:
        result["evidence"].append("coaxial_multi_radius_cylinders")


def _validate_counterbore_hole(
    result: dict,
    face_indices: list[int],
    nodes: dict[int, dict],
    internal_edges: list[dict],
) -> None:
    cylinders = [
        nodes[idx]
        for idx in face_indices
        if nodes.get(idx, {}).get("primitive_surface") == "cylinder"
    ]
    planes = [
        nodes[idx]
        for idx in face_indices
        if nodes.get(idx, {}).get("primitive_surface") == "plane"
    ]
    unsupported = [
        idx
        for idx in face_indices
        if nodes.get(idx, {}).get("primitive_surface") not in {"cylinder", "plane"}
    ]
    if unsupported:
        _invalid(result, "counterbore_contains_unsupported_surfaces")
    if len(cylinders) < 2:
        _invalid(result, "min_2_coaxial_cylinders_required")
        return
    if len(planes) != 1:
        if len(planes) > 1:
            _invalid(result, "counterbore_should_not_include_bottom_face")
        else:
            _invalid(result, "missing_shoulder_plane")

    axes = [_node_specific(node, "axis", [0.0, 0.0, 0.0]) for node in cylinders]
    reference = axes[0]
    if any(_angle_between(reference, axis) > math.radians(8.0) for axis in axes[1:]):
        _invalid(result, "cylinder_axes_not_coaxial")

    radii = sorted({
        round(float(_node_specific(node, "radius", 0.0)), 6)
        for node in cylinders
    })
    if len(radii) < 2:
        _invalid(result, "multi_radius_difference_missing")

    if len(planes) == 1:
        shoulder_idx = int(planes[0]["idx"])
        connected_cylinders = set()
        for edge in internal_edges:
            if not _is_circular_edge(edge):
                continue
            src = int(edge.get("source_face", -1))
            dst = int(edge.get("target_face", -1))
            if shoulder_idx not in {src, dst}:
                continue
            neighbor = dst if src == shoulder_idx else src
            if nodes.get(neighbor, {}).get("primitive_surface") == "cylinder":
                connected_cylinders.add(neighbor)
        if len(connected_cylinders) < 2:
            _invalid(result, "shoulder_plane_must_connect_large_and_small_cylinders")

    if result["valid"]:
        result["evidence"].append("large_cylinder_shoulder_small_through_cylinder")


def _validate_rectangular_pocket(
    result: dict,
    face_indices: list[int],
    nodes: dict[int, dict],
    internal_edges: list[dict],
) -> None:
    if len(face_indices) < 5:
        _invalid(result, "min_5_faces_required")
    counts = _primitive_counts(face_indices, nodes)
    if counts.get("plane", 0) < 5:
        _invalid(result, "requires_planar_bottom_and_side_faces")
    concave_edges = [
        edge for edge in internal_edges
        if edge.get("convexity") == "concave" or int(edge.get("convexity_id", 0)) == 2
    ]
    # Directed edges are counted twice, so four bottom-side edges appear as at
    # least four in minimal synthetic tests and usually eight in real graphs.
    if len(concave_edges) < 4:
        _invalid(result, "insufficient_concave_internal_edges")
    if result["valid"]:
        result["evidence"].append("bottom_plus_four_side_faces")


def _validate_rectangular_boss(
    result: dict,
    face_indices: list[int],
    nodes: dict[int, dict],
    internal_edges: list[dict],
) -> None:
    if len(face_indices) < 5:
        _invalid(result, "min_5_faces_required")
    counts = _primitive_counts(face_indices, nodes)
    if counts.get("plane", 0) < 5:
        _invalid(result, "requires_planar_top_and_side_faces")
    convex_edges = [
        edge for edge in internal_edges
        if edge.get("convexity") == "convex" or int(edge.get("convexity_id", 0)) == 1
    ]
    if len(convex_edges) < 4:
        _invalid(result, "insufficient_convex_internal_edges")
    if result["valid"]:
        result["evidence"].append("top_plus_four_side_faces")


def _validate_cylinder_boss(
    result: dict,
    face_indices: list[int],
    nodes: dict[int, dict],
    internal_edges: list[dict],
) -> None:
    counts = _primitive_counts(face_indices, nodes)
    if counts.get("cylinder", 0) < 1:
        _invalid(result, "missing_cylindrical_side_face")
    if counts.get("plane", 0) < 1:
        _invalid(result, "missing_boss_top_face")
    if not _has_internal_circular_edge(internal_edges):
        _invalid(result, "missing_side_top_circular_edge")
    if result["valid"]:
        result["evidence"].append("cylindrical_side_plus_top_face")


def _validate_rectangular_slot(
    result: dict,
    face_indices: list[int],
    nodes: dict[int, dict],
    internal_edges: list[dict],
) -> None:
    counts = _primitive_counts(face_indices, nodes)
    if len(face_indices) < 3 or counts.get("plane", 0) < 3:
        _invalid(result, "rect_slot_requires_bottom_plus_two_sides")
    if counts.get("plane", 0) != len(face_indices):
        _invalid(result, "rect_slot_requires_planar_faces")
    right_angle_concave_edges = _count_unique_edges(
        internal_edges,
        lambda edge: _is_line_edge(edge)
        and _is_concave_edge(edge)
        and _edge_dihedral_is_near(edge, math.pi / 2.0, math.radians(12.0)),
    )
    if right_angle_concave_edges < 2:
        _invalid(result, "rect_slot_requires_two_right_angle_bottom_side_edges")
    if result["valid"]:
        result["evidence"].append("rectangular_slot_bottom_and_side_planes")


def _validate_v_slot(
    result: dict,
    face_indices: list[int],
    nodes: dict[int, dict],
    internal_edges: list[dict],
) -> None:
    counts = _primitive_counts(face_indices, nodes)
    if len(face_indices) < 2:
        _invalid(result, "v_slot_requires_two_side_planes")
    unsupported_surfaces = [
        idx for idx in face_indices
        if not _is_v_slot_surface(nodes.get(idx, {}).get("primitive_surface"))
    ]
    if unsupported_surfaces:
        _invalid(result, "v_slot_contains_unsupported_surfaces")

    long_side_faces = []
    swept_side_faces = []
    for idx in face_indices:
        node = nodes.get(idx, {})
        primitive = node.get("primitive_surface")
        if primitive == "plane" and _node_descriptor(node, "projected_aspect_ratio", 0.0) >= 3.0:
            long_side_faces.append(idx)
        elif primitive in {"surface_of_revolution", "surface_of_extrusion", "bspline_surface"}:
            swept_side_faces.append(idx)
    if len(long_side_faces) < 2:
        if len(long_side_faces) + len(swept_side_faces) < 2:
            _invalid(result, "slot_v_requires_two_long_inclined_side_faces")

    v_valley_edges = _count_unique_edges(
        internal_edges,
        lambda edge: _is_line_edge(edge)
        and _is_concave_edge(edge)
        and math.radians(20.0) <= float(edge.get("dihedral_angle", math.pi / 2.0) or 0.0) <= math.radians(80.0),
    )
    has_swept_side_surfaces = bool(swept_side_faces)
    has_inclined_plane_pair = _has_inclined_plane_pair([nodes[idx] for idx in long_side_faces])
    if v_valley_edges < 1 and not has_swept_side_surfaces and not has_inclined_plane_pair:
        _invalid(result, "missing_v_slot_concave_valley_edge")
    if result["valid"]:
        if v_valley_edges >= 1:
            result["evidence"].append("two_inclined_planes_with_long_concave_valley")
        elif has_swept_side_surfaces:
            result["evidence"].append("swept_v_slot_surfaces")
        else:
            result["evidence"].append("two_inclined_long_planes")


def _validate_rib_web(result: dict, face_indices: list[int], nodes: dict[int, dict]) -> None:
    plane_nodes = [
        nodes[idx]
        for idx in face_indices
        if nodes.get(idx, {}).get("primitive_surface") == "plane"
    ]
    if len(plane_nodes) < 2:
        _invalid(result, "thin_parallel_face_pair_required")
        return

    for left, right in combinations(plane_nodes, 2):
        left_normal = _node_specific(left, "normal", [0.0, 0.0, 0.0])
        right_normal = _node_specific(right, "normal", [0.0, 0.0, 0.0])
        if _angle_between(left_normal, right_normal) > math.radians(10.0):
            continue
        thickness = _distance_between_centers(left, right)
        long_extent = max(_node_descriptor(left, "projected_long_extent"), _node_descriptor(right, "projected_long_extent"))
        short_extent = max(_node_descriptor(left, "projected_short_extent"), _node_descriptor(right, "projected_short_extent"))
        reference_extent = max(min(long_extent, short_extent), 1e-9)
        if thickness / reference_extent <= 0.45 and long_extent / max(thickness, 1e-9) >= 6.0:
            result["evidence"].append("thin_parallel_face_pair")
            return

    _invalid(result, "thin_parallel_face_pair_required")


def _distance_between_centers(left: dict, right: dict) -> float:
    left_center = _node_center(left)
    right_center = _node_center(right)
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left_center, right_center)))


def _has_internal_circular_edge(edge_attrs: list[dict]) -> bool:
    return any(
        _is_circular_edge(edge)
        for edge in edge_attrs
    )


def _count_unique_edges(edge_attrs: list[dict], predicate) -> int:
    seen = set()
    for edge in edge_attrs:
        if not predicate(edge):
            continue
        src = int(edge.get("source_face", -1))
        dst = int(edge.get("target_face", -1))
        key = (
            min(src, dst),
            max(src, dst),
            edge.get("curve_type", "unknown"),
            round(float(edge.get("length", 0.0) or 0.0), 6),
        )
        seen.add(key)
    return len(seen)


def _is_circular_edge(edge: dict) -> bool:
    return (
        edge.get("curve_type") == "circle"
        or int(edge.get("curve_type_id", 0) or 0) == 2
        or float(edge.get("is_circle", 0.0) or 0.0) > 0.5
    )


def _is_line_edge(edge: dict) -> bool:
    return (
        edge.get("curve_type") == "line"
        or int(edge.get("curve_type_id", 0) or 0) == 1
        or float(edge.get("is_line", 0.0) or 0.0) > 0.5
    )


def _is_concave_edge(edge: dict) -> bool:
    return edge.get("convexity") == "concave" or int(edge.get("convexity_id", 0) or 0) == 2


def _edge_dihedral_is_near(edge: dict, target: float, tolerance: float) -> bool:
    value = float(edge.get("dihedral_angle", 0.0) or 0.0)
    return abs(value - target) <= tolerance


def _is_v_slot_surface(primitive: str | None) -> bool:
    return primitive in {"plane", "surface_of_revolution", "surface_of_extrusion", "bspline_surface"}


def _has_inclined_plane_pair(nodes: list[dict]) -> bool:
    for left, right in combinations(nodes, 2):
        left_normal = _node_specific(left, "normal", [0.0, 0.0, 0.0])
        right_normal = _node_specific(right, "normal", [0.0, 0.0, 0.0])
        angle = _angle_between(left_normal, right_normal)
        if math.radians(8.0) <= angle <= math.radians(80.0):
            return True
    return False
