"""Hybrid object proposal generation for v5.12 diagnostic.

Combines three proposal sources:
  1. same_instance       — existing assemble_instance_candidates
  2. raw_macro_component — connected components of same normalized non-stock macro
  3. topology_rule       — existing B-Rep topology finders

Does NOT modify GNN architecture, training, generation, or default inference.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from typing import Any

from utils.area_truth_normalization import normalize_truth_macro
from utils.topology_postprocess import (
    find_blind_hole_candidates,
    find_chamfer_stock_candidates,
    find_counterbore_candidates,
    find_countersink_candidates,
    find_review_targeted_chamfer_candidates,
    find_review_targeted_fillet_candidates,
    find_review_targeted_rib_candidates,
    find_cylinder_boss_candidates,
    find_existing_rib_transition_absorption_candidates,
    find_existing_rib_web_cap_completion_candidates,
    find_fillet_stock_candidates,
    find_mixed_rib_candidates,
    find_o_slot_candidates,
    find_obround_hole_candidates,
    find_pocket_embedded_rib_candidates,
    find_rectangular_prism_candidates,
    find_rectangular_slot_candidates,
    find_rib_stock_candidates,
    find_supported_rectangular_boss_candidates,
    find_through_hole_opening_candidates,
    find_transition_cap_rib_candidates,
    find_u_slot_candidates,
    find_v_slot_stock_candidates,
)


TARGET_PROPOSAL_MACROS = {
    "through_hole_round",
    "blind_hole_round",
    "stepped_hole_round",
    "counterbore_hole",
    "countersink_hole",
    "obround_hole",
    "slot_rect",
    "slot_u",
    "slot_o",
    "slot_v",
    "pocket_rect",
    "rib",
    "web",
    "fillet",
    "chamfer",
    "cylinder_boss",
    "rectangular_boss",
}

STOCK_OR_BACKGROUND_MACROS = {
    "",
    "stock",
    "stock_face",
    "background",
    "unknown",
    "unknown_feature",
    "pocket_circular",
}


def normalize_macro(value: object) -> str:
    text = str(value).strip()
    if not text:
        return "unknown"
    return normalize_truth_macro(text)


def node_idx(record: dict[str, Any]) -> int:
    for key in ("idx", "face_index", "index", "id"):
        value = record.get(key)
        if value is not None:
            return int(value)
    raise ValueError(f"face record has no index field: {record}")


def node_area(record: dict[str, Any]) -> float:
    attributes = record.get("attributes")
    if isinstance(attributes, dict):
        physical = attributes.get("physical")
        if isinstance(physical, dict) and physical.get("area") is not None:
            return float(physical["area"])
    if record.get("area") is not None:
        return float(record["area"])
    return 0.0


def extract_graph_edges(graph: dict[str, Any]) -> list[tuple[int, int]]:
    edges = []
    for edge in graph.get("edges", []):
        if isinstance(edge, dict):
            if "src" in edge and "dst" in edge:
                edges.append((int(edge["src"]), int(edge["dst"])))
            elif "source" in edge and "target" in edge:
                edges.append((int(edge["source"]), int(edge["target"])))
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            edges.append((int(edge[0]), int(edge[1])))
    return edges


def graph_nodes_by_idx(graph: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {node_idx(node): node for node in graph.get("nodes", graph.get("faces", []))}


def prediction_faces_by_idx(prediction: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {node_idx(face): face for face in prediction.get("faces", prediction.get("nodes", []))}


def merge_face_records(graph: dict[str, Any], prediction: dict[str, Any]) -> list[dict[str, Any]]:
    graph_nodes = graph_nodes_by_idx(graph)
    predicted_faces = prediction_faces_by_idx(prediction)
    merged = []
    for idx in sorted(graph_nodes):
        node = graph_nodes[idx]
        pred = predicted_faces.get(idx, {})
        raw_macro = pred.get("macro_feature") or pred.get("predicted_macro_feature") or pred.get("macro") or ""
        raw_semantic = pred.get("semantic_subtype") or pred.get("predicted_semantic_subtype") or ""
        primitive = pred.get("primitive_surface") or node.get("primitive_surface") or node.get("primitive") or "unknown"
        record = {
            "idx": idx,
            "primitive_surface": primitive,
            "macro_feature": normalize_macro(raw_macro),
            "raw_macro_feature": str(raw_macro),
            "semantic_subtype": str(raw_semantic),
            "raw_semantic_subtype": str(raw_semantic),
            "area": node_area(node),
            "attributes": node.get("attributes", {}),
        }
        for key in ("macro_confidence", "semantic_confidence", "derived_macro_features", "derived_macro_feature_scores"):
            if key in pred:
                record[key] = pred[key]
        merged.append(record)
    return merged


def candidate_record(source: str, macro: str, face_indices: Iterable[int], reason: str) -> dict[str, Any]:
    faces = sorted(int(idx) for idx in face_indices)
    return {
        "source": source,
        "proposal_sources": [source],
        "proposal_macro": normalize_macro(macro),
        "face_indices": faces,
        "seed_face_indices": faces,
        "assembly_reasons": [reason],
        "mean_same_instance_probability": None,
        "min_same_instance_probability": None,
    }


def raw_macro_component_proposals(graph: dict[str, Any], faces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    face_by_idx = {int(face["idx"]): face for face in faces}
    target_faces = {
        idx
        for idx, face in face_by_idx.items()
        if face.get("macro_feature") in TARGET_PROPOSAL_MACROS
        and face.get("macro_feature") not in STOCK_OR_BACKGROUND_MACROS
    }
    adjacency: dict[int, set[int]] = {idx: set() for idx in target_faces}
    for src, dst in extract_graph_edges(graph):
        if src not in target_faces or dst not in target_faces:
            continue
        if face_by_idx[src].get("macro_feature") != face_by_idx[dst].get("macro_feature"):
            continue
        adjacency[src].add(dst)
        adjacency[dst].add(src)

    visited: set[int] = set()
    proposals: list[dict[str, Any]] = []
    for seed in sorted(target_faces):
        if seed in visited:
            continue
        macro = face_by_idx[seed]["macro_feature"]
        queue = deque([seed])
        visited.add(seed)
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(adjacency.get(current, set())):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        proposals.append(candidate_record("raw_macro_component", macro, component, "raw_macro_connected_component"))
    return proposals


def edge_key(src: int, dst: int) -> tuple[int, int]:
    left, right = sorted((int(src), int(dst)))
    return (left, right)


def edge_attributes_by_key(graph: dict[str, Any]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    edges = extract_graph_edges(graph)
    attributes = graph.get("edge_attributes", [])
    result: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for position, edge in enumerate(edges):
        attr = attributes[position] if position < len(attributes) and isinstance(attributes[position], dict) else {}
        result[edge_key(*edge)].append(attr)
    return result


def is_circle_edge(attr: dict[str, Any]) -> bool:
    curve_type = str(attr.get("curve_type", "")).lower()
    return curve_type == "circle" or attr.get("is_circle") == 1.0 or attr.get("curve_type_id") == 2


def is_line_edge(attr: dict[str, Any]) -> bool:
    curve_type = str(attr.get("curve_type", "")).lower()
    return curve_type == "line" or attr.get("is_line") == 1.0 or attr.get("curve_type_id") == 1


def opening_cylinder_fallback_proposals(
    graph: dict[str, Any],
    faces: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    nodes = graph_nodes_by_idx(graph)
    face_by_idx = {int(face["idx"]): face for face in faces}
    attrs_by_key = edge_attributes_by_key(graph)
    circular_plane_neighbors: dict[int, set[int]] = defaultdict(set)
    line_edge_count: dict[int, int] = defaultdict(int)

    for src, dst in extract_graph_edges(graph):
        attrs = attrs_by_key.get(edge_key(src, dst), [{}])
        src_node = nodes.get(src, {})
        dst_node = nodes.get(dst, {})
        for attr in attrs:
            if is_circle_edge(attr):
                if src_node.get("primitive_surface") == "cylinder" and dst_node.get("primitive_surface") == "plane":
                    circular_plane_neighbors[src].add(dst)
                if dst_node.get("primitive_surface") == "cylinder" and src_node.get("primitive_surface") == "plane":
                    circular_plane_neighbors[dst].add(src)
            if is_line_edge(attr):
                line_edge_count[src] += 1
                line_edge_count[dst] += 1

    proposals: list[dict[str, Any]] = []
    for idx, node in sorted(nodes.items()):
        if node.get("primitive_surface") != "cylinder":
            continue
        face = face_by_idx.get(idx, {})
        macro = normalize_macro(face.get("macro_feature", ""))
        if macro in {"fillet", "slot_o", "slot_u", "cylinder_boss"}:
            continue
        if node_area(node) <= 1e-9:
            continue
        if len(circular_plane_neighbors.get(idx, set())) < 2:
            continue
        if line_edge_count.get(idx, 0) > 2:
            continue
        proposals.append(candidate_record("fallback_rule", "through_hole_round", [idx], "fallback_two_circular_plane_boundaries"))
    return proposals


def topology_rule_proposals(
    graph: dict[str, Any],
    faces: list[dict[str, Any]],
    repaired_faces: set[int] | None = None,
) -> list[dict[str, Any]]:
    repaired = set(repaired_faces or set())
    proposal_specs: list[tuple[str, str, list[list[int]]]] = [
        ("through_hole_round", "topology_through_hole_opening", find_through_hole_opening_candidates(faces, graph, repaired)),
        ("blind_hole_round", "topology_blind_hole_wall_bottom", find_blind_hole_candidates(faces, graph, repaired)),
        ("counterbore_hole", "topology_counterbore", find_counterbore_candidates(faces, graph, repaired)),
        ("countersink_hole", "topology_countersink", find_countersink_candidates(faces, graph, repaired)),
        ("cylinder_boss", "topology_cylinder_boss", find_cylinder_boss_candidates(faces, graph, repaired)),
        ("obround_hole", "topology_obround_hole", find_obround_hole_candidates(faces, graph, repaired)),
        ("pocket_rect", "topology_rectangular_pocket", find_rectangular_prism_candidates(faces, graph, repaired, "pocket_rect", "concave")),
        ("rectangular_boss", "topology_rectangular_boss", find_rectangular_prism_candidates(faces, graph, repaired, "rectangular_boss", "convex")),
        ("rectangular_boss", "topology_supported_rectangular_boss", find_supported_rectangular_boss_candidates(faces, graph, repaired)),
        ("slot_u", "topology_u_slot", find_u_slot_candidates(faces, graph, repaired)),
        ("slot_rect", "topology_rectangular_slot", find_rectangular_slot_candidates(faces, graph, repaired)),
        ("slot_o", "topology_o_slot", find_o_slot_candidates(faces, graph, repaired)),
        ("slot_v", "topology_v_slot", find_v_slot_stock_candidates(faces, graph, repaired)),
        ("rib", "topology_mixed_rib", find_mixed_rib_candidates(faces, graph, repaired)),
        ("rib", "topology_rib_stock", find_rib_stock_candidates(faces, graph, repaired)),
        ("rib", "topology_existing_rib_cap_completion", find_existing_rib_web_cap_completion_candidates(faces, graph, repaired)),
        ("rib", "topology_transition_cap_rib", find_transition_cap_rib_candidates(faces, graph, repaired)),
        ("rib", "topology_pocket_embedded_rib", find_pocket_embedded_rib_candidates(faces, graph, repaired)),
        ("rib", "topology_existing_rib_transition_absorption", find_existing_rib_transition_absorption_candidates(faces, graph, repaired)),
        ("fillet", "topology_fillet_stock_transition", find_fillet_stock_candidates(faces, graph, repaired)),
        ("chamfer", "topology_chamfer_stock_transition", find_chamfer_stock_candidates(faces, graph, repaired)),
        ("rib", "topology_review_targeted_rib_residual", find_review_targeted_rib_candidates(faces, graph, repaired)),
        ("fillet", "topology_review_targeted_fillet_residual", find_review_targeted_fillet_candidates(faces, graph, repaired)),
        ("chamfer", "topology_review_targeted_chamfer_residual", find_review_targeted_chamfer_candidates(faces, graph, repaired)),
    ]
    proposals: list[dict[str, Any]] = []
    for macro, reason, candidates in proposal_specs:
        for face_indices in candidates:
            proposals.append(candidate_record("topology_rule", macro, face_indices, reason))
    return proposals


def proposal_key(proposal: dict[str, Any]) -> tuple[str, tuple[int, ...]]:
    return (
        normalize_macro(proposal.get("proposal_macro", "")),
        tuple(sorted(int(idx) for idx in proposal.get("face_indices", []))),
    )


def deduplicate_proposals(proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged_by_key: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    source_order = ["same_instance", "raw_macro_component", "topology_rule"]
    for proposal in proposals:
        key = proposal_key(proposal)
        if not key[1]:
            continue
        current = merged_by_key.get(key)
        normalized = dict(proposal)
        normalized["proposal_macro"] = key[0]
        normalized["face_indices"] = list(key[1])
        normalized["seed_face_indices"] = sorted(int(idx) for idx in normalized.get("seed_face_indices", key[1]))
        normalized["proposal_sources"] = list(normalized.get("proposal_sources", [normalized.get("source", "unknown")]))
        normalized["assembly_reasons"] = list(normalized.get("assembly_reasons", []))
        if current is None:
            merged_by_key[key] = normalized
            continue
        source_set = set(current.get("proposal_sources", [])) | set(normalized.get("proposal_sources", []))
        reason_list = list(current.get("assembly_reasons", []))
        for reason in normalized.get("assembly_reasons", []):
            if reason not in reason_list:
                reason_list.append(reason)
        current["source"] = "hybrid_merged"
        current["proposal_sources"] = sorted(
            source_set,
            key=lambda item: source_order.index(item) if item in source_order else 99,
        )
        current["assembly_reasons"] = reason_list
        for key_name in ("mean_same_instance_probability", "min_same_instance_probability"):
            if current.get(key_name) is None and normalized.get(key_name) is not None:
                current[key_name] = normalized[key_name]
    return [
        merged_by_key[key]
        for key in sorted(merged_by_key, key=lambda item: (item[0], item[1][0], len(item[1]), item[1]))
    ]


def generate_hybrid_object_proposals(
    graph: dict[str, Any],
    prediction: dict[str, Any],
    same_instance_proposals: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    faces = merge_face_records(graph, prediction)
    proposals: list[dict[str, Any]] = []
    for proposal in same_instance_proposals or []:
        item = dict(proposal)
        item["source"] = "same_instance"
        item["proposal_sources"] = ["same_instance"]
        item["proposal_macro"] = normalize_macro(item.get("proposal_macro", ""))
        proposals.append(item)
    proposals.extend(raw_macro_component_proposals(graph, faces))
    proposals.extend(topology_rule_proposals(graph, faces))
    proposals.extend(opening_cylinder_fallback_proposals(graph, faces))
    return deduplicate_proposals(proposals)
