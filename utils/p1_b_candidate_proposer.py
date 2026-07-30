"""General topology candidate sources for the P1-B recognition pipeline."""

from __future__ import annotations

from collections import defaultdict, deque
import math
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "topocae_p1_b_candidate_proposals_1"
HOLE_HINTS = [
    "through_hole_round",
    "blind_hole_round",
    "stepped_hole_round",
    "counterbore_hole",
    "countersink_hole",
    "obround_hole",
]
SLOT_HINTS = ["slot_u", "slot_rect", "slot_v", "slot_o"]


def generate_candidate_proposals(
    graph_data: Mapping[str, Any],
    prediction_data: Mapping[str, Any],
    assembly_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Generate candidates from topology only, without sample names or truth ids."""

    proposals: list[dict[str, Any]] = []
    for candidate in assembly_result.get("candidates", []):
        proposals.append(_proposal(
            source="same_instance_strong_component",
            faces=candidate.get("face_indices", []),
            macro_hints=candidate.get("provisional_macros", []),
            evidence={"assembly_candidate_id": candidate.get("candidate_id"), "assembly_trace": candidate.get("trace", {})},
        ))
    proposals.extend(_coaxial_chain_proposals(graph_data))
    proposals.extend(_loop_guided_proposals(graph_data))
    proposals.extend(_concave_region_proposals(graph_data))
    proposals.extend(_elongated_channel_proposals(graph_data))
    proposals.extend(_thin_wall_proposals(graph_data, prediction_data))
    proposals.extend(_parent_transition_proposals(graph_data, prediction_data))
    deduplicated = _deduplicate(proposals)
    for index, proposal in enumerate(deduplicated):
        proposal["candidate_id"] = f"proposal_{index:05d}"
    source_counts: dict[str, int] = defaultdict(int)
    for proposal in deduplicated:
        for source in proposal["proposal_sources"]:
            source_counts[source] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_count": len(deduplicated),
        "source_counts": dict(sorted(source_counts.items())),
        "proposals": deduplicated,
        "contract": {
            "uses_truth_face_ids": False,
            "uses_sample_names": False,
            "uses_absolute_area_constants": False,
            "all_scale_thresholds_dimensionless": True,
        },
    }


def evaluate_proposal_oracle_recall(
    proposals: Iterable[Mapping[str, Any]],
    truth_instances: Iterable[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.95,
) -> dict[str, Any]:
    proposal_sets = [set(int(index) for index in proposal.get("face_indices", [])) for proposal in proposals]
    records = []
    hits = 0
    for position, truth in enumerate(truth_instances):
        truth_set = set(int(index) for index in truth.get("face_indices", []))
        scores = [_iou(truth_set, proposal_set) for proposal_set in proposal_sets]
        best_iou = max(scores, default=0.0)
        hit = bool(truth_set and best_iou >= iou_threshold)
        hits += int(hit)
        records.append({
            "truth_instance_id": truth.get("instance_id", position + 1),
            "macro_feature": truth.get("macro_feature"),
            "face_indices": sorted(truth_set),
            "best_proposal_iou": best_iou,
            "hit": hit,
        })
    count = len(records)
    return {
        "schema_version": "topocae_p1_b_proposal_oracle_recall_1",
        "iou_threshold": float(iou_threshold),
        "truth_instance_count": count,
        "hit_count": hits,
        "oracle_recall": hits / count if count else 0.0,
        "records": records,
    }


def _coaxial_chain_proposals(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    nodes = _nodes(graph)
    scale = _part_scale(graph)
    curved = [index for index, node in nodes.items() if _primitive(node) in {"cylinder", "cone"}]
    parent = {index: index for index in curved}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for offset, left in enumerate(curved):
        for right in curved[offset + 1:]:
            if _coaxial(nodes[left], nodes[right], scale):
                union(left, right)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in curved:
        groups[find(index)].append(index)
    return [
        _proposal(
            source="coaxial_cylinder_cone_chain",
            faces=group,
            macro_hints=HOLE_HINTS + ["cylinder_boss"],
            evidence={"coaxial": True, "chain_length": len(group)},
        )
        for group in groups.values()
        if group
    ]


def _loop_guided_proposals(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    nodes = _nodes(graph)
    attrs = _unique_edge_attributes(graph)
    proposals = []
    for index, node in nodes.items():
        if _primitive(node) not in {"cylinder", "cone"}:
            continue
        adjacent_planes = []
        inner = outer = ambiguous = 0
        for pair, attribute_list in attrs.items():
            if index not in pair:
                continue
            neighbor = pair[1] if pair[0] == index else pair[0]
            if _primitive(nodes.get(neighbor, {})) == "plane":
                adjacent_planes.append(neighbor)
            for attribute in attribute_list:
                role = str(attribute.get("adjacent_plane_loop_role", "unknown"))
                wall = str(attribute.get("cylindrical_wall_role_hint", "unknown"))
                inner += int(role == "inner" or wall == "inner_hole_wall")
                outer += int(role == "outer" or wall == "outer_cylinder_wall")
                ambiguous += int(role in {"unknown", "ambiguous"} or wall in {"unknown", "ambiguous"})
        hints = HOLE_HINTS if inner >= outer else ["cylinder_boss"]
        proposals.append(_proposal(
            source="loop_material_side_guided",
            faces=[index, *adjacent_planes],
            macro_hints=hints,
            evidence={
                "wall_face_index": index,
                "adjacent_plane_face_indices": sorted(set(adjacent_planes)),
                "inner_evidence_count": inner,
                "outer_evidence_count": outer,
                "ambiguous_evidence_count": ambiguous,
            },
        ))
    return proposals


def _concave_region_proposals(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    nodes = _nodes(graph)
    adjacency: dict[int, set[int]] = defaultdict(set)
    for pair, attribute_list in _unique_edge_attributes(graph).items():
        if any(str(attribute.get("convexity", "")) == "concave" for attribute in attribute_list):
            adjacency[pair[0]].add(pair[1])
            adjacency[pair[1]].add(pair[0])
    components = _components(adjacency, set(adjacency))
    proposals = []
    for component in components:
        primitives = {_primitive(nodes.get(index, {})) for index in component}
        hints = ["pocket_rect", "pocket_circular", *SLOT_HINTS]
        if "cylinder" in primitives:
            hints = ["pocket_circular", *SLOT_HINTS, "pocket_rect"]
        proposals.append(_proposal(
            source="concave_boundary_region",
            faces=component,
            macro_hints=hints,
            evidence={"concave_connected": True, "primitive_surfaces": sorted(primitives)},
        ))
    return proposals


def _elongated_channel_proposals(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    nodes = _nodes(graph)
    adjacency = _adjacency(graph)
    proposals = []
    for index, node in nodes.items():
        recognition = _recognition(node)
        aspect = max(
            float(recognition.get("local_extent_long_middle_ratio", 0.0) or 0.0),
            float(recognition.get("projected_aspect_ratio", 0.0) or 0.0),
            float(recognition.get("uv_aspect_ratio", 0.0) or 0.0),
        )
        if aspect < 2.5:
            continue
        neighborhood = {index}
        for neighbor in adjacency.get(index, set()):
            if _primitive(nodes.get(neighbor, {})) in {"plane", "cylinder", "cone"}:
                neighborhood.add(neighbor)
        proposals.append(_proposal(
            source="elongated_swept_channel",
            faces=neighborhood,
            macro_hints=SLOT_HINTS,
            evidence={"dimensionless_aspect_ratio": aspect, "aspect_threshold": 2.5},
        ))
    return proposals


def _thin_wall_proposals(graph: Mapping[str, Any], prediction: Mapping[str, Any]) -> list[dict[str, Any]]:
    nodes = _nodes(graph)
    predicted = _prediction_macros(prediction)
    adjacency = _adjacency(graph)
    seeds = {
        index
        for index, node in nodes.items()
        if predicted.get(index) in {"rib", "web"}
        or (
            _primitive(node) == "plane"
            and float(_recognition(node).get("face_area_over_part_total_area", 1.0) or 1.0) <= 0.12
            and float(_recognition(node).get("local_extent_long_middle_ratio", 0.0) or 0.0) >= 2.0
        )
    }
    restricted = {index: {neighbor for neighbor in adjacency.get(index, set()) if neighbor in seeds} for index in seeds}
    components = _components(restricted, seeds)
    return [
        _proposal(
            source="thin_wall_rib_web",
            faces=component,
            macro_hints=["rib", "web"],
            evidence={"dimensionless_thin_wall_filter": True},
        )
        for component in components
        if component
    ]


def _parent_transition_proposals(graph: Mapping[str, Any], prediction: Mapping[str, Any]) -> list[dict[str, Any]]:
    nodes = _nodes(graph)
    predicted = _prediction_macros(prediction)
    adjacency = _adjacency(graph)
    proposals = []
    for index, node in nodes.items():
        primitive = _primitive(node)
        macro = predicted.get(index)
        if macro not in {"fillet", "chamfer"} and primitive not in {"toroid", "cone"}:
            continue
        parents = sorted(adjacency.get(index, set()))
        hints = [macro] if macro in {"fillet", "chamfer"} else (["fillet"] if primitive == "toroid" else ["chamfer"])
        proposals.append(_proposal(
            source="parent_surface_transition",
            faces=[index],
            macro_hints=hints,
            evidence={"parent_face_indices": parents, "transition_primitive": primitive},
        ))
    return proposals


def _proposal(
    *, source: str, faces: Iterable[int], macro_hints: Iterable[str], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "proposal_sources": [source],
        "face_indices": sorted(set(int(index) for index in faces)),
        "macro_hints": sorted(set(str(value) for value in macro_hints if value)),
        "topology_evidence": dict(evidence),
    }


def _deduplicate(proposals: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_faces: dict[tuple[int, ...], dict[str, Any]] = {}
    for raw in proposals:
        faces = tuple(sorted(set(int(index) for index in raw.get("face_indices", []))))
        if not faces:
            continue
        current = by_faces.get(faces)
        if current is None:
            by_faces[faces] = {
                "proposal_sources": list(raw.get("proposal_sources", [])),
                "face_indices": list(faces),
                "macro_hints": list(raw.get("macro_hints", [])),
                "topology_evidence": [dict(raw.get("topology_evidence", {}))],
            }
            continue
        current["proposal_sources"] = sorted(set(current["proposal_sources"]) | set(raw.get("proposal_sources", [])))
        current["macro_hints"] = sorted(set(current["macro_hints"]) | set(raw.get("macro_hints", [])))
        current["topology_evidence"].append(dict(raw.get("topology_evidence", {})))
    return [by_faces[key] for key in sorted(by_faces, key=lambda item: (item[0], len(item), item))]


def _coaxial(left: Mapping[str, Any], right: Mapping[str, Any], scale: float) -> bool:
    left_axis, left_location = _axis(left)
    right_axis, right_location = _axis(right)
    if left_axis is None or right_axis is None or left_location is None or right_location is None:
        return False
    dot = abs(sum(a * b for a, b in zip(left_axis, right_axis)))
    if dot < math.cos(math.radians(2.0)):
        return False
    delta = [b - a for a, b in zip(left_location, right_location)]
    cross = _cross(delta, left_axis)
    line_distance = math.sqrt(sum(value * value for value in cross))
    return line_distance / max(scale, 1e-12) <= 1e-5


def _axis(node: Mapping[str, Any]) -> tuple[list[float] | None, list[float] | None]:
    geometry = node.get("attributes", {}).get("geometry_v2", {})
    direction = geometry.get("axis_direction")
    location = geometry.get("axis_location")
    specific = node.get("attributes", {}).get("specific", {})
    direction = direction or specific.get("axis_direction") or specific.get("axis")
    location = location or specific.get("axis_location") or specific.get("location")
    if not isinstance(direction, (list, tuple)) or len(direction) < 3:
        return None, None
    if not isinstance(location, (list, tuple)) or len(location) < 3:
        return None, None
    direction_values = [float(value) for value in direction[:3]]
    norm = math.sqrt(sum(value * value for value in direction_values))
    if norm <= 1e-12:
        return None, None
    return [value / norm for value in direction_values], [float(value) for value in location[:3]]


def _part_scale(graph: Mapping[str, Any]) -> float:
    return float(
        graph.get("part_metadata_v2", {}).get("normalization", {}).get("part_characteristic_length", 1.0) or 1.0
    )


def _nodes(graph: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {int(node.get("idx", position)): node for position, node in enumerate(graph.get("nodes", []))}


def _primitive(node: Mapping[str, Any]) -> str:
    return str(node.get("primitive_surface", "unknown_surface"))


def _recognition(node: Mapping[str, Any]) -> Mapping[str, Any]:
    return node.get("attributes", {}).get("recognition_features_v2", {})


def _prediction_macros(prediction: Mapping[str, Any]) -> dict[int, str]:
    return {
        int(face.get("idx", position)): str(face.get("macro_feature", face.get("raw_macro_feature", "stock")))
        for position, face in enumerate(prediction.get("faces", []))
    }


def _adjacency(graph: Mapping[str, Any]) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for edge in graph.get("edges", []):
        if not isinstance(edge, (list, tuple)) or len(edge) < 2:
            continue
        left, right = int(edge[0]), int(edge[1])
        if left != right:
            adjacency[left].add(right)
            adjacency[right].add(left)
    return adjacency


def _unique_edge_attributes(graph: Mapping[str, Any]) -> dict[tuple[int, int], list[Mapping[str, Any]]]:
    result: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for edge, attribute in zip(graph.get("edges", []), graph.get("edge_attributes", [])):
        if not isinstance(edge, (list, tuple)) or len(edge) < 2 or not isinstance(attribute, Mapping):
            continue
        result[tuple(sorted((int(edge[0]), int(edge[1]))))].append(attribute)
    return result


def _components(adjacency: Mapping[int, set[int]], nodes: set[int]) -> list[list[int]]:
    remaining = set(nodes)
    groups = []
    while remaining:
        start = min(remaining)
        queue = deque([start])
        remaining.remove(start)
        group = []
        while queue:
            current = queue.popleft()
            group.append(current)
            for neighbor in sorted(adjacency.get(current, set())):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        groups.append(sorted(group))
    return groups


def _cross(left: list[float], right: list[float]) -> list[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def _iou(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


__all__ = ["SCHEMA_VERSION", "evaluate_proposal_oracle_recall", "generate_candidate_proposals"]
