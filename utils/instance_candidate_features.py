from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from utils.topology_rules import validate_feature_topology


PRIMITIVE_BUCKETS = ("plane", "cylinder", "cone", "torus", "other")
CURVE_BUCKETS = ("line", "circle", "ellipse", "spline", "other")
CONVEXITY_BUCKETS = ("concave", "convex", "smooth", "unknown")
LOOP_ROLE_BUCKETS = ("inner", "outer", "unknown")
CYLINDER_ROLE_BUCKETS = ("inner", "outer", "ambiguous", "unknown")

TOPOLOGY_REASON_BUCKETS = (
    "single_cylindrical_wall_required",
    "through_hole_should_not_include_bottom_or_countersink_faces",
    "requires_two_circular_opening_edges",
    "missing_cylindrical_wall",
    "missing_bottom_face",
    "missing_wall_bottom_circular_edge",
    "missing_conical_countersink_face",
    "missing_small_cylindrical_wall",
    "missing_cone_cylinder_circular_edge",
    "min_2_coaxial_cylinders_required",
    "missing_shoulder_plane",
    "counterbore_should_not_include_bottom_face",
    "shoulder_plane_must_connect_large_and_small_cylinders",
    "multi_radius_difference_missing",
    "missing_circular_step_edges",
    "min_5_faces_required",
    "requires_planar_bottom_and_side_faces",
    "requires_planar_top_and_side_faces",
    "insufficient_concave_internal_edges",
    "insufficient_convex_internal_edges",
    "thin_parallel_face_pair_required",
)

TOPOLOGY_EVIDENCE_BUCKETS = (
    "single_cylindrical_wall_with_two_openings",
    "cylinder_wall_plus_bottom_face",
    "large_cylinder_shoulder_small_through_cylinder",
    "cone_plus_small_cylindrical_wall",
    "bottom_plus_four_side_faces",
    "top_plus_four_side_faces",
    "cylindrical_side_plus_top_face",
    "thin_parallel_face_pair",
    "transition_face_candidate",
)

UNKNOWN_MACROS = {"", "unknown", "unknown_feature", "stock", "stock_face", "background", None}
PROPOSAL_SOURCE_BUCKETS = (
    "same_instance",
    "raw_macro_component",
    "topology_rule",
    "fallback_rule",
    "hybrid_merged",
    "truth",
    "assembled_instance_candidate",
    "other",
)

ASSEMBLY_REASON_BUCKETS = (
    "topology_review_targeted_rib_residual",
    "topology_review_targeted_fillet_residual",
    "topology_review_targeted_chamfer_residual",
    "topology_rib_stock",
    "topology_mixed_rib",
    "topology_existing_rib_cap_completion",
    "topology_transition_cap_rib",
    "topology_pocket_embedded_rib",
    "topology_existing_rib_transition_absorption",
    "topology_fillet_stock_transition",
    "topology_chamfer_stock_transition",
    "topology_through_hole_opening",
    "topology_counterbore",
    "topology_countersink",
    "topology_cylinder_boss",
    "topology_rectangular_slot",
    "topology_u_slot",
    "topology_o_slot",
    "topology_v_slot",
    "source_macro_positive_clone",
    "assembly_reason_positive_clone",
    "other",
)

PREDICTION_MACRO_BUCKETS = (
    "stock",
    "rib",
    "web",
    "pocket_rect",
    "pocket_circular",
    "slot_rect",
    "slot_u",
    "slot_o",
    "slot_v",
    "fillet",
    "chamfer",
    "through_hole_round",
    "blind_hole_round",
    "counterbore_hole",
    "countersink_hole",
    "obround_hole",
    "cylinder_boss",
    "rectangular_boss",
    "other",
)


def extract_instance_candidate_features(
    graph_data: dict[str, Any],
    candidate: Mapping[str, Any],
    prediction_faces: list[dict[str, Any]] | None = None,
    candidate_macro: str | None = None,
) -> dict[str, float]:
    """Extract JSON-friendly object features for an assembled candidate.

    The extractor intentionally does not read graph truth labels such as
    feature_sets, instance_id, or node macro_feature fields.  Candidate macro is
    treated as proposal metadata, and face vote features come only from
    prediction_faces.
    """

    face_indices = _candidate_face_indices(candidate)
    face_set = set(face_indices)
    nodes = _nodes_by_idx(graph_data)
    candidate_nodes = [nodes.get(idx, {}) for idx in face_indices]
    areas = [_node_area(node) for node in candidate_nodes]
    total_area = sum(_node_area(node) for node in nodes.values())
    candidate_area = sum(areas)
    macro = candidate_macro if candidate_macro is not None else _candidate_macro(candidate)

    features: dict[str, float] = {
        "face_count": float(len(face_indices)),
        "candidate_area": float(candidate_area),
        "model_area": float(total_area),
        "area_ratio": _safe_div(candidate_area, total_area),
        "area_min": float(min(areas)) if areas else 0.0,
        "area_max": float(max(areas)) if areas else 0.0,
        "area_mean": _safe_div(candidate_area, len(areas)),
        "area_max_min_ratio": _safe_div(max(areas) if areas else 0.0, min(areas) if areas else 0.0),
        "area_bbox_ratio": _area_bbox_ratio(candidate_nodes, candidate_area),
        "same_instance_probability_mean": _candidate_probability(candidate, "mean"),
        "same_instance_probability_min": _candidate_probability(candidate, "min"),
        "same_instance_probability_max": _candidate_probability(candidate, "max"),
        "boundary_same_instance_probability_mean": _candidate_probability(candidate, "boundary_mean"),
        "predicted_macro_vote_fraction": 0.0,
        "topology_valid": 0.0,
    }

    _add_primitive_features(features, candidate_nodes, len(face_indices))
    internal_edges, boundary_edges = _split_candidate_edges(graph_data, face_set)
    _add_edge_features(features, internal_edges, "internal")
    _add_edge_features(features, boundary_edges, "boundary")
    _add_unique_edge_role_features(features, internal_edges, "internal")
    _add_unique_edge_role_features(features, boundary_edges, "boundary")
    _add_candidate_wall_role_features(features, boundary_edges, face_set)
    topology_result = _topology_result(graph_data, macro, face_indices)
    _add_topology_result_features(features, topology_result)
    _add_prediction_vote_features(features, prediction_faces, face_indices, macro)
    _add_proposal_source_features(features, candidate)
    _add_assembly_reason_features(features, candidate)

    return {key: _finite_float(value) for key, value in sorted(features.items())}


def _candidate_face_indices(candidate: Mapping[str, Any]) -> list[int]:
    raw_indices = candidate.get("face_indices", [])
    if raw_indices is None:
        return []
    return sorted({int(idx) for idx in raw_indices if _is_int_like(idx)})


def _candidate_macro(candidate: Mapping[str, Any]) -> str | None:
    for key in ("proposal_macro", "candidate_macro"):
        value = candidate.get(key)
        if value is not None:
            return str(value)
    return None


def _candidate_sources(candidate: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    """Return (proposal_sources_set, all_source_tags)."""
    proposal_srcs: set[str] = set()
    raw_sources = candidate.get("proposal_sources")
    if isinstance(raw_sources, Iterable) and not isinstance(raw_sources, (str, bytes, Mapping)):
        for source in raw_sources:
            token = str(source).strip()
            if token:
                proposal_srcs.add(token)
    all_tags = set(proposal_srcs)
    source = candidate.get("source")
    if source is not None:
        token = str(source).strip()
        if token:
            all_tags.add(token)
    if not all_tags:
        all_tags.add("other")
    return proposal_srcs, all_tags


def _add_proposal_source_features(features: dict[str, float], candidate: Mapping[str, Any]) -> None:
    proposal_srcs, all_tags = _candidate_sources(candidate)
    features["proposal_source_count"] = float(len(proposal_srcs))
    for bucket in PROPOSAL_SOURCE_BUCKETS:
        if bucket == "other":
            value = 1.0 if not any(source in PROPOSAL_SOURCE_BUCKETS[:-1] for source in all_tags) else 0.0
        else:
            value = 1.0 if bucket in all_tags else 0.0
        features[f"proposal_source_{bucket}"] = value


def _candidate_assembly_reasons(candidate: Mapping[str, Any]) -> set[str]:
    reasons: set[str] = set()
    raw_reasons = candidate.get("assembly_reasons")
    if isinstance(raw_reasons, Iterable) and not isinstance(raw_reasons, (str, bytes, Mapping)):
        for reason in raw_reasons:
            token = str(reason).strip()
            if token:
                reasons.add(token)
    augmentation = candidate.get("augmentation")
    if augmentation is not None:
        token = str(augmentation).strip()
        if token:
            reasons.add(token)
    return reasons


def _add_assembly_reason_features(features: dict[str, float], candidate: Mapping[str, Any]) -> None:
    reasons = _candidate_assembly_reasons(candidate)
    features["assembly_reason_count"] = float(len(reasons))
    known = set(ASSEMBLY_REASON_BUCKETS[:-1])
    for bucket in ASSEMBLY_REASON_BUCKETS:
        if bucket == "other":
            value = 1.0 if reasons and not any(reason in known for reason in reasons) else 0.0
        else:
            value = 1.0 if bucket in reasons else 0.0
        features[f"assembly_reason_{bucket}"] = value


def _candidate_probability(candidate: Mapping[str, Any], kind: str) -> float:
    aliases = {
        "mean": (
            "same_instance_probability_mean",
            "mean_same_instance_probability",
            "internal_same_instance_probability_mean",
        ),
        "min": (
            "same_instance_probability_min",
            "min_same_instance_probability",
            "internal_same_instance_probability_min",
        ),
        "max": (
            "same_instance_probability_max",
            "max_same_instance_probability",
            "internal_same_instance_probability_max",
        ),
        "boundary_mean": (
            "boundary_same_instance_probability_mean",
            "mean_boundary_same_instance_probability",
        ),
    }
    for key in aliases[kind]:
        value = candidate.get(key)
        if _is_number(value):
            return float(value)
    return 0.0


def _nodes_by_idx(graph_data: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    nodes: dict[int, dict[str, Any]] = {}
    for fallback_idx, raw_record in enumerate(_raw_node_records(graph_data)):
        record = dict(raw_record)
        idx = _record_idx(record, fallback_idx)
        nodes[idx] = record
    return nodes


def _raw_node_records(graph_data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = _to_builtin(graph_data.get("nodes", graph_data.get("faces", [])))
    if isinstance(raw, Mapping):
        records = []
        for key, value in raw.items():
            if isinstance(value, Mapping):
                record = dict(value)
                record.setdefault("idx", key)
                records.append(record)
        return records
    if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
        return [record for record in raw if isinstance(record, Mapping)]
    return []


def _record_idx(record: Mapping[str, Any], fallback_idx: int) -> int:
    for key in ("idx", "face_index", "index", "id"):
        value = record.get(key)
        if _is_int_like(value):
            return int(value)
    return fallback_idx


def _node_area(node: Mapping[str, Any]) -> float:
    for key in ("area", "surface_area", "face_area", "area_mm2"):
        value = node.get(key)
        if _is_number(value):
            return float(value)
    attributes = node.get("attributes")
    if isinstance(attributes, Mapping):
        physical = attributes.get("physical")
        if isinstance(physical, Mapping) and _is_number(physical.get("area")):
            return float(physical["area"])
        if _is_number(attributes.get("area")):
            return float(attributes["area"])
    return 0.0


def _primitive(node: Mapping[str, Any]) -> str:
    value = None
    for key in ("primitive_surface", "primitive", "surface_type", "primitive_type"):
        value = node.get(key)
        if value is not None:
            break
    if value is None:
        attributes = node.get("attributes")
        if isinstance(attributes, Mapping):
            for key in ("primitive_surface", "primitive", "surface_type", "type"):
                value = attributes.get(key)
                if value is not None:
                    break
    token = str(value or "").strip().lower()
    if token in {"toroid", "torus", "toroidal"}:
        return "torus"
    if token in {"plane", "cylinder", "cone"}:
        return token
    return "other"


def _add_primitive_features(features: dict[str, float], nodes: list[Mapping[str, Any]], face_count: int) -> None:
    counts = Counter(_primitive(node) for node in nodes)
    missing_count = max(0, face_count - len(nodes))
    if missing_count:
        counts["other"] += missing_count
    for bucket in PRIMITIVE_BUCKETS:
        count = float(counts.get(bucket, 0))
        features[f"primitive_{bucket}_count"] = count
        features[f"primitive_{bucket}_ratio"] = _safe_div(count, face_count)


def _split_candidate_edges(
    graph_data: Mapping[str, Any],
    face_set: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    internal: list[dict[str, Any]] = []
    boundary: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for src, dst, attr in _edge_records(graph_data):
        src_in = src in face_set
        dst_in = dst in face_set
        if not src_in and not dst_in:
            continue
        key = (
            min(src, dst),
            max(src, dst),
            _curve_type(attr),
            _convexity(attr),
            round(float(attr.get("length", 0.0) or 0.0), 6),
        )
        if key in seen:
            continue
        seen.add(key)
        enriched = dict(attr)
        enriched.setdefault("source_face", src)
        enriched.setdefault("target_face", dst)
        if src_in and dst_in:
            internal.append(enriched)
        elif src_in != dst_in:
            boundary.append(enriched)
    return internal, boundary


def _edge_records(graph_data: Mapping[str, Any]) -> list[tuple[int, int, dict[str, Any]]]:
    raw_edges = _extract_graph_edges(graph_data)
    raw_attrs = _to_builtin(graph_data.get("edge_attributes", []))
    attrs = raw_attrs if isinstance(raw_attrs, list) else []
    records: list[tuple[int, int, dict[str, Any]]] = []
    for index, (src, dst) in enumerate(raw_edges):
        attr = dict(attrs[index]) if index < len(attrs) and isinstance(attrs[index], Mapping) else {}
        attr_src_dst = _pair_from_record(attr)
        if attr_src_dst is not None:
            src, dst = attr_src_dst
        records.append((int(src), int(dst), attr))
    return records


def _extract_graph_edges(graph_data: Mapping[str, Any]) -> list[tuple[int, int]]:
    if "edges" in graph_data:
        raw_edges = _to_builtin(graph_data.get("edges", []))
        if isinstance(raw_edges, Iterable) and not isinstance(raw_edges, (str, bytes, Mapping)):
            return [pair for record in raw_edges if (pair := _pair_from_record(record)) is not None]
    raw_edge_index = _to_builtin(graph_data.get("edge_index", []))
    if _looks_like_edge_index(raw_edge_index):
        return [(int(src), int(dst)) for src, dst in zip(raw_edge_index[0], raw_edge_index[1])]
    if isinstance(raw_edge_index, Iterable) and not isinstance(raw_edge_index, (str, bytes, Mapping)):
        return [pair for record in raw_edge_index if (pair := _pair_from_record(record)) is not None]
    return []


def _pair_from_record(record: Any) -> tuple[int, int] | None:
    record = _to_builtin(record)
    if isinstance(record, Mapping):
        for left_key, right_key in (("src", "dst"), ("source", "target"), ("source_face", "target_face")):
            if _is_int_like(record.get(left_key)) and _is_int_like(record.get(right_key)):
                return int(record[left_key]), int(record[right_key])
        for key in ("face_pair", "faces", "pair"):
            if key in record:
                return _pair_from_record(record[key])
        return None
    if (
        isinstance(record, (list, tuple))
        and len(record) >= 2
        and _is_int_like(record[0])
        and _is_int_like(record[1])
    ):
        return int(record[0]), int(record[1])
    return None


def _add_edge_features(features: dict[str, float], edges: list[dict[str, Any]], prefix: str) -> None:
    edge_count = len(edges)
    features[f"{prefix}_edge_count"] = float(edge_count)
    curve_counts = Counter(_curve_type(edge) for edge in edges)
    convexity_counts = Counter(_convexity(edge) for edge in edges)
    for bucket in CURVE_BUCKETS:
        count = float(curve_counts.get(bucket, 0))
        features[f"{prefix}_{bucket}_count"] = count
        features[f"{prefix}_{bucket}_ratio"] = _safe_div(count, edge_count)
    for bucket in CONVEXITY_BUCKETS:
        count = float(convexity_counts.get(bucket, 0))
        features[f"{prefix}_{bucket}_count"] = count
        features[f"{prefix}_{bucket}_ratio"] = _safe_div(count, edge_count)


def _add_unique_edge_role_features(features: dict[str, float], edges: list[dict[str, Any]], prefix: str) -> None:
    unique_edges = _unique_edge_records(edges)
    edge_count = len(unique_edges)
    features[f"{prefix}_unique_edge_count"] = float(edge_count)

    circle_edges = [edge for edge in unique_edges if _curve_type(edge) == "circle"]
    line_edges = [edge for edge in unique_edges if _curve_type(edge) == "line"]
    features[f"{prefix}_unique_circle_count"] = float(len(circle_edges))
    features[f"{prefix}_unique_line_count"] = float(len(line_edges))
    features[f"{prefix}_unique_circle_ratio"] = _safe_div(len(circle_edges), edge_count)
    features[f"{prefix}_unique_line_ratio"] = _safe_div(len(line_edges), edge_count)
    features[f"{prefix}_unique_circle_length_sum"] = float(sum(_edge_length(edge) for edge in circle_edges))
    features[f"{prefix}_unique_line_length_sum"] = float(sum(_edge_length(edge) for edge in line_edges))
    features[f"{prefix}_unique_circle_length_mean"] = _safe_div(
        sum(_edge_length(edge) for edge in circle_edges),
        len(circle_edges),
    )

    loop_counts = Counter(_loop_role(edge) for edge in unique_edges)
    cylinder_counts = Counter(_cylinder_wall_role(edge) for edge in unique_edges)
    for bucket in LOOP_ROLE_BUCKETS:
        count = float(loop_counts.get(bucket, 0))
        features[f"{prefix}_loop_role_{bucket}_count"] = count
        features[f"{prefix}_loop_role_{bucket}_ratio"] = _safe_div(count, edge_count)
        circle_count = float(sum(1 for edge in circle_edges if _loop_role(edge) == bucket))
        features[f"{prefix}_loop_role_{bucket}_circle_count"] = circle_count
        features[f"{prefix}_loop_role_{bucket}_circle_ratio"] = _safe_div(circle_count, len(circle_edges))
    for bucket in CYLINDER_ROLE_BUCKETS:
        count = float(cylinder_counts.get(bucket, 0))
        features[f"{prefix}_cylinder_role_{bucket}_count"] = count
        features[f"{prefix}_cylinder_role_{bucket}_ratio"] = _safe_div(count, edge_count)
        circle_count = float(sum(1 for edge in circle_edges if _cylinder_wall_role(edge) == bucket))
        features[f"{prefix}_cylinder_role_{bucket}_circle_count"] = circle_count
        features[f"{prefix}_cylinder_role_{bucket}_circle_ratio"] = _safe_div(circle_count, len(circle_edges))


def _unique_edge_records(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    for edge in edges:
        src = int(edge.get("source_face", -1)) if _is_int_like(edge.get("source_face", -1)) else -1
        dst = int(edge.get("target_face", -1)) if _is_int_like(edge.get("target_face", -1)) else -1
        key = (
            min(src, dst),
            max(src, dst),
            _curve_type(edge),
            _loop_role(edge),
            _cylinder_wall_role(edge),
            round(_edge_length(edge), 6),
            round(float(edge.get("circle_radius", 0.0) or 0.0), 6),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(edge)
    return unique


def _edge_length(edge: Mapping[str, Any]) -> float:
    return _finite_float(edge.get("length", 0.0))


def _loop_role(edge: Mapping[str, Any]) -> str:
    role = str(edge.get("adjacent_plane_loop_role", "") or "").strip().lower()
    if role in {"inner_loop", "inner"}:
        return "inner"
    if role in {"outer_loop", "outer"}:
        return "outer"
    if _finite_float(edge.get("adjacent_plane_loop_role_inner", 0.0)) > 0.5:
        return "inner"
    if _finite_float(edge.get("adjacent_plane_loop_role_outer", 0.0)) > 0.5:
        return "outer"
    role_id = edge.get("adjacent_plane_loop_role_id")
    if _is_int_like(role_id):
        return {1: "inner", 2: "outer"}.get(int(role_id), "unknown")
    return "unknown"


def _cylinder_wall_role(edge: Mapping[str, Any]) -> str:
    role = str(edge.get("cylindrical_wall_role_hint", "") or "").strip().lower()
    if "inner" in role:
        return "inner"
    if "outer" in role:
        return "outer"
    if "ambiguous" in role:
        return "ambiguous"
    if _finite_float(edge.get("cylindrical_wall_role_inner_hint", 0.0)) > 0.5:
        return "inner"
    if _finite_float(edge.get("cylindrical_wall_role_outer_hint", 0.0)) > 0.5:
        return "outer"
    if _finite_float(edge.get("cylindrical_wall_role_ambiguous_hint", 0.0)) > 0.5:
        return "ambiguous"
    role_id = edge.get("cylindrical_wall_role_hint_id")
    if _is_int_like(role_id):
        return {1: "inner", 2: "outer", 3: "ambiguous"}.get(int(role_id), "unknown")
    return "unknown"


def _add_candidate_wall_role_features(
    features: dict[str, float],
    boundary_edges: list[dict[str, Any]],
    face_set: set[int],
) -> None:
    role_faces: dict[str, set[int]] = {bucket: set() for bucket in CYLINDER_ROLE_BUCKETS}
    loop_faces: dict[str, set[int]] = {bucket: set() for bucket in LOOP_ROLE_BUCKETS}
    for edge in _unique_edge_records(boundary_edges):
        src = int(edge.get("source_face", -1)) if _is_int_like(edge.get("source_face", -1)) else -1
        dst = int(edge.get("target_face", -1)) if _is_int_like(edge.get("target_face", -1)) else -1
        candidate_face = src if src in face_set else dst if dst in face_set else None
        if candidate_face is None:
            continue
        role_faces[_cylinder_wall_role(edge)].add(candidate_face)
        loop_faces[_loop_role(edge)].add(candidate_face)

    candidate_count = len(face_set)
    for bucket in CYLINDER_ROLE_BUCKETS:
        features[f"candidate_boundary_{bucket}_wall_face_count"] = float(len(role_faces[bucket]))
        features[f"candidate_boundary_{bucket}_wall_face_fraction"] = _safe_div(len(role_faces[bucket]), candidate_count)
    for bucket in LOOP_ROLE_BUCKETS:
        features[f"candidate_boundary_{bucket}_loop_face_count"] = float(len(loop_faces[bucket]))
        features[f"candidate_boundary_{bucket}_loop_face_fraction"] = _safe_div(len(loop_faces[bucket]), candidate_count)


def _topology_result(graph_data: dict[str, Any], candidate_macro: str | None, face_indices: list[int]) -> dict[str, Any]:
    if candidate_macro in UNKNOWN_MACROS:
        return {"valid": False, "reasons": ["unknown_macro"], "evidence": [], "may_attach_to_parent_feature": False}
    try:
        return validate_feature_topology(str(candidate_macro), face_indices, graph_data)
    except Exception:
        return {"valid": False, "reasons": ["topology_validation_exception"], "evidence": [], "may_attach_to_parent_feature": False}


def _add_topology_result_features(features: dict[str, float], result: Mapping[str, Any]) -> None:
    reasons = [str(item) for item in result.get("reasons", []) if item is not None]
    evidence = [str(item) for item in result.get("evidence", []) if item is not None]
    features["topology_valid"] = 1.0 if result.get("valid") else 0.0
    features["topology_reason_count"] = float(len(reasons))
    features["topology_evidence_count"] = float(len(evidence))
    features["topology_may_attach_to_parent_feature"] = 1.0 if result.get("may_attach_to_parent_feature") else 0.0
    reason_set = {_sanitize_feature_token(item) for item in reasons}
    evidence_set = {_sanitize_feature_token(item) for item in evidence}
    for reason in TOPOLOGY_REASON_BUCKETS:
        features[f"topology_reason_{reason}"] = 1.0 if reason in reason_set else 0.0
    for item in evidence_set:
        if item in TOPOLOGY_EVIDENCE_BUCKETS:
            features[f"topology_evidence_{item}"] = 1.0
    for item in TOPOLOGY_EVIDENCE_BUCKETS:
        features.setdefault(f"topology_evidence_{item}", 0.0)


def _sanitize_feature_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() else "_" for ch in value.strip().lower())
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_")


def _curve_type(edge: Mapping[str, Any]) -> str:
    value = edge.get("curve_type")
    if value is not None:
        token = str(value).strip().lower()
        if token in {"line", "circle", "ellipse", "spline"}:
            return token
        if token in {"bspline", "b_spline", "b-spline", "spline_curve"}:
            return "spline"
    curve_type_id = edge.get("curve_type_id")
    if _is_int_like(curve_type_id):
        return {1: "line", 2: "circle", 3: "ellipse", 4: "spline"}.get(int(curve_type_id), "other")
    if float(edge.get("is_line", 0.0) or 0.0) > 0.5:
        return "line"
    if float(edge.get("is_circle", 0.0) or 0.0) > 0.5:
        return "circle"
    return "other"


def _convexity(edge: Mapping[str, Any]) -> str:
    value = edge.get("convexity")
    if value is not None:
        token = str(value).strip().lower()
        if token in {"concave", "convex", "smooth"}:
            return token
    convexity_id = edge.get("convexity_id")
    if _is_int_like(convexity_id):
        return {1: "convex", 2: "concave", 3: "smooth"}.get(int(convexity_id), "unknown")
    return "unknown"


def _add_prediction_vote_features(
    features: dict[str, float],
    prediction_faces: list[dict[str, Any]] | None,
    face_indices: list[int],
    candidate_macro: str | None,
) -> None:
    for bucket in PREDICTION_MACRO_BUCKETS:
        features[f"prediction_macro_{bucket}_fraction"] = 0.0
    if not face_indices or not prediction_faces:
        return

    prediction_by_idx = {}
    for fallback_idx, face in enumerate(prediction_faces):
        if isinstance(face, Mapping):
            prediction_by_idx[_record_idx(face, fallback_idx)] = face

    macros = [_prediction_macro(prediction_by_idx.get(idx, {})) for idx in face_indices]
    counts = Counter(macro if macro in PREDICTION_MACRO_BUCKETS[:-1] else "other" for macro in macros)
    for bucket in PREDICTION_MACRO_BUCKETS:
        features[f"prediction_macro_{bucket}_fraction"] = _safe_div(counts.get(bucket, 0), len(face_indices))
    if candidate_macro not in UNKNOWN_MACROS:
        features["predicted_macro_vote_fraction"] = _safe_div(
            sum(1 for macro in macros if macro == str(candidate_macro)),
            len(face_indices),
        )


def _prediction_macro(face: Mapping[str, Any]) -> str:
    for key in ("macro_feature", "predicted_macro_feature", "predicted_macro", "macro", "raw_macro_feature"):
        value = face.get(key)
        if value is not None:
            return str(value)
    return "unknown_feature"


def _area_bbox_ratio(nodes: list[Mapping[str, Any]], candidate_area: float) -> float:
    descriptor_area = 0.0
    for node in nodes:
        descriptors = node.get("attributes", {}).get("shape_descriptors", {})
        if not isinstance(descriptors, Mapping):
            continue
        long_extent = float(descriptors.get("projected_long_extent", 0.0) or 0.0)
        short_extent = float(descriptors.get("projected_short_extent", 0.0) or 0.0)
        if long_extent > 0.0 and short_extent > 0.0:
            descriptor_area += long_extent * short_extent
    if descriptor_area > 1e-12:
        return _safe_div(candidate_area, descriptor_area)

    centers = [_node_center(node) for node in nodes]
    if len(centers) < 2:
        return 0.0
    spans = []
    for axis in range(3):
        values = [center[axis] for center in centers]
        spans.append(max(values) - min(values))
    spans = sorted((span for span in spans if span > 1e-12), reverse=True)
    if len(spans) < 2:
        return 0.0
    return _safe_div(candidate_area, spans[0] * spans[1])


def _node_center(node: Mapping[str, Any]) -> list[float]:
    physical = node.get("attributes", {}).get("physical", {})
    center = physical.get("center_of_mass", [0.0, 0.0, 0.0]) if isinstance(physical, Mapping) else [0.0, 0.0, 0.0]
    values = list(center[:3]) if isinstance(center, (list, tuple)) else [0.0, 0.0, 0.0]
    while len(values) < 3:
        values.append(0.0)
    return [float(value) for value in values[:3]]

def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0

def _safe_div(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if abs(denominator) <= 1e-12:
        return 0.0
    return float(numerator) / denominator


def _looks_like_edge_index(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], (list, tuple))
        and isinstance(value[1], (list, tuple))
        and len(value[0]) == len(value[1])
        and all(_is_int_like(item) for row in value for item in row)
    )


def _to_builtin(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return value.tolist()
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int_like(value: Any) -> bool:
    return _is_number(value) or (isinstance(value, str) and value.lstrip("-").isdigit())


__all__ = ["extract_instance_candidate_features"]

