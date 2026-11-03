"""P1-B verifier v2: recognize legal one-face and parent-bridged slot topology."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import utils.p1_b_topology_verifier as base


_BASE_CLASS_FEASIBILITY = base._class_feasibility


def _class_feasibility_v2(
    graph: Mapping[str, Any],
    faces: list[int],
    macro: str,
) -> dict[str, Any]:
    result = dict(_BASE_CLASS_FEASIBILITY(graph, faces, macro))
    reasons = list(result.get("reasons", []))
    nodes = {int(node.get("idx", position)): node for position, node in enumerate(graph.get("nodes", []))}
    if macro in {"slot_u", "slot_rect", "slot_v", "slot_o"} and "candidate_not_brep_adjacent_connected" in reasons:
        if _components_share_boundary_parent(graph, set(faces)):
            reasons.remove("candidate_not_brep_adjacent_connected")
            result["connected_via_shared_boundary_parent"] = True
    if macro == "slot_o" and "region_feature_has_fewer_than_two_faces" in reasons and len(faces) == 1:
        node = nodes.get(int(faces[0]), {})
        primitive = str(node.get("primitive_surface", "unknown_surface"))
        topology = node.get("attributes", {}).get("topology_v2", {})
        outer_edge_count = int(topology.get("outer_loop_edge_count", 0) or 0)
        if primitive in {"cylinder", "torus", "toroid", "surface_of_revolution"} and outer_edge_count >= 3:
            reasons.remove("region_feature_has_fewer_than_two_faces")
            result["closed_single_surface_slot_evidence"] = {
                "primitive_surface": primitive,
                "outer_loop_edge_count": outer_edge_count,
            }
    if macro == "fillet" and "fillet_without_curved_transition_surface" in reasons:
        if any(str(nodes.get(index, {}).get("primitive_surface", "")) == "torus" for index in faces):
            reasons.remove("fillet_without_curved_transition_surface")
            result["torus_alias_normalized"] = True
    result["reasons"] = reasons
    result["feasible"] = not reasons
    result["verifier_revision"] = "p1_b_topology_verifier_v2"
    return result


def _components_share_boundary_parent(graph: Mapping[str, Any], faces: set[int]) -> bool:
    if len(faces) < 2:
        return False
    adjacency: dict[int, set[int]] = defaultdict(set)
    for edge in graph.get("edges", []):
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            left, right = int(edge[0]), int(edge[1])
            adjacency[left].add(right)
            adjacency[right].add(left)
    common_neighbors = None
    for face in sorted(faces):
        outside = adjacency.get(face, set()) - faces
        common_neighbors = outside if common_neighbors is None else common_neighbors & outside
    return bool(common_neighbors)


base._class_feasibility = _class_feasibility_v2

SCHEMA_VERSION = "topocae_p1_b_topology_verifier_2"
select_global_candidate_thresholds = base.select_global_candidate_thresholds
verify_candidate_prediction = base.verify_candidate_prediction


__all__ = ["SCHEMA_VERSION", "select_global_candidate_thresholds", "verify_candidate_prediction"]
