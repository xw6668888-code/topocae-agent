from __future__ import annotations

from collections import deque
from copy import deepcopy
from itertools import combinations

from utils.topology_rules import validate_feature_topology


MAX_RECTANGULAR_POCKET_BOTTOM_ASPECT_RATIO = 8.0


DEFAULT_MACRO_FEATURE_IDS = {
    "stock": 0,
    "through_hole_round": 1,
    "blind_hole_round": 2,
    "stepped_hole_round": 3,
    "counterbore_hole": 4,
    "countersink_hole": 5,
    "obround_hole": 6,
    "slot_u": 9,
    "slot_rect": 10,
    "pocket_rect": 11,
    "pocket_circular": 12,
    "cylinder_boss": 14,
    "rib": 20,
    "web": 21,
    "rectangular_boss": 24,
    "fillet": 25,
    "chamfer": 26,
    "slot_v": 33,
    "slot_o": 35,
}

DEFAULT_SEMANTIC_SUBTYPE_IDS = {
    "stock_face": 0,
    "through_hole_cylinder": 5,
    "blind_hole_cylinder": 6,
    "blind_hole_bottom": 7,
    "cylinder_boss_side": 12,
    "cylinder_boss_top": 13,
    "rect_pocket_side_plane": 24,
    "rect_pocket_bottom_plane": 25,
    "slot_rect_side_plane": 26,
    "slot_rect_bottom_plane": 27,
    "slot_u_side_plane": 30,
    "slot_u_end_cylinder": 31,
    "slot_u_bottom_plane": 32,
    "fillet_blend_surface": 33,
    "chamfer_bevel_plane": 34,
    "counterbore_large_cylinder": 35,
    "counterbore_step_plane": 36,
    "counterbore_small_cylinder": 37,
    "countersink_cone": 38,
    "countersink_small_cylinder": 39,
    "circular_pocket_side_cylinder": 40,
    "circular_pocket_bottom_plane": 41,
    "pad_side_plane": 50,
    "pad_top_plane": 51,
    "rib_side_plane": 52,
    "rib_top_plane": 53,
    "slot_v_side_plane": 57,
    "slot_o_side_plane": 61,
    "slot_o_bottom_plane": 62,
}


def apply_topology_postprocess(
    prediction_data: dict,
    graph_data: dict,
    *,
    same_instance_threshold: float = 0.5,
) -> dict:
    corrected = deepcopy(prediction_data)
    faces = corrected.get("faces", [])
    components = connected_components_for_prediction(faces, graph_data, same_instance_threshold=same_instance_threshold)
    invalid_components = []
    repaired_components = []
    demoted_face_count = 0
    semantic_repair_face_count = 0
    candidate_repair_records, candidate_repaired_faces = repair_topology_candidates_v4_2(faces, graph_data)
    repaired_components.extend(candidate_repair_records)
    split_repair_records, pre_repaired_faces = repair_split_slot_v_components(faces, graph_data)
    repaired_components.extend(split_repair_records)
    pre_repaired_faces.update(candidate_repaired_faces)

    for component in components:
        component_face_set = set(component["face_indices"])
        if component_face_set.issubset(pre_repaired_faces):
            continue
        if component_face_set & pre_repaired_faces:
            component = dict(component)
            component["face_indices"] = sorted(component_face_set - pre_repaired_faces)
        macro_feature = component["macro_feature"]
        if macro_feature in {"rib", "web"}:
            split_through_holes = _rib_component_through_hole_faces(component["face_indices"], graph_data)
            if split_through_holes:
                remap_faces_to_macro(
                    faces,
                    split_through_holes,
                    "through_hole_round",
                    graph_data,
                    ["rib_component_full_cylinder_split_to_through_hole"],
                )
                repaired_components.append({
                    "from_macro_feature": macro_feature,
                    "to_macro_feature": "through_hole_round",
                    "face_indices": split_through_holes,
                    "face_count": len(split_through_holes),
                    "source_reasons": ["rib_component_contains_full_through_hole_cylinder"],
                    "repair_evidence": ["single_full_cylinder_with_two_planar_circular_openings"],
                    "action": "split_remapped",
                })
                component = dict(component)
                component["face_indices"] = [idx for idx in component["face_indices"] if idx not in set(split_through_holes)]
                pre_repaired_faces.update(split_through_holes)
                if not component["face_indices"]:
                    continue
            if macro_feature == "web":
                remap_faces_to_macro(
                    faces,
                    component["face_indices"],
                    "rib",
                    graph_data,
                    ["web_merged_to_rib_target_for_v5_4_b"],
                )
                repair_rib_semantics(faces, component["face_indices"], graph_data)
                repaired_components.append({
                    "from_macro_feature": "web",
                    "to_macro_feature": "rib",
                    "face_indices": component["face_indices"],
                    "face_count": len(component["face_indices"]),
                    "source_reasons": ["rib_web_shared_thin_wall_target"],
                    "repair_evidence": ["valid_thin_wall_component"],
                    "action": "remapped",
                })
                component = dict(component)
                component["macro_feature"] = "rib"
                macro_feature = "rib"
        validation = validate_feature_topology(macro_feature, component["face_indices"], graph_data)
        if validation["valid"]:
            if macro_feature == "slot_rect":
                refined_slot = _refine_overmerged_rectangular_slot_component(component["face_indices"], graph_data)
                if refined_slot is not None and set(refined_slot) != set(component["face_indices"]):
                    extra_faces = sorted(set(component["face_indices"]) - set(refined_slot))
                    remap_faces_to_macro(
                        faces,
                        refined_slot,
                        "slot_rect",
                        graph_data,
                        ["overmerged_slot_rect_refined_to_consistent_long_faces"],
                    )
                    repair_rectangular_slot_semantics(faces, refined_slot, graph_data)
                    demoted_face_count += demote_faces_to_stock(
                        faces,
                        extra_faces,
                        ["overmerged_slot_rect_extra_faces_demoted"],
                    )
                    repaired_components.append({
                        "from_macro_feature": "slot_rect",
                        "to_macro_feature": "slot_rect",
                        "face_indices": refined_slot,
                        "face_count": len(refined_slot),
                        "source_reasons": ["overmerged_slot_rect_component"],
                        "repair_evidence": ["three_consistent_long_faces"],
                        "demoted_extra_faces": extra_faces,
                        "action": "split_refined",
                    })
                    continue
            if macro_feature == "chamfer":
                chamfer_shape_reason = _chamfer_shape_invalid_reason(component["face_indices"], graph_data)
                if chamfer_shape_reason is not None:
                    invalid_record = {
                        "macro_feature": macro_feature,
                        "normalized_family": validation["normalized_family"],
                        "face_indices": component["face_indices"],
                        "face_count": len(component["face_indices"]),
                        "reasons": [chamfer_shape_reason],
                        "evidence": validation["evidence"],
                        "action": "demoted_to_stock",
                    }
                    invalid_components.append(invalid_record)
                    demoted_face_count += demote_faces_to_stock(
                        faces,
                        component["face_indices"],
                        [chamfer_shape_reason],
                    )
                    continue
            if macro_feature == "pocket_rect":
                pocket_shape_reason = _rectangular_pocket_shape_invalid_reason(component["face_indices"], graph_data)
                if pocket_shape_reason is not None:
                    invalid_record = {
                        "macro_feature": macro_feature,
                        "normalized_family": validation["normalized_family"],
                        "face_indices": component["face_indices"],
                        "face_count": len(component["face_indices"]),
                        "reasons": [pocket_shape_reason],
                        "evidence": validation["evidence"],
                        "action": "demoted_to_stock",
                    }
                    invalid_components.append(invalid_record)
                    demoted_face_count += demote_faces_to_stock(
                        faces,
                        component["face_indices"],
                        [pocket_shape_reason],
                    )
                    continue
            if macro_feature == "rib":
                rib_shape_reason = _rib_component_shape_invalid_reason(component["face_indices"], graph_data)
                if rib_shape_reason is not None:
                    invalid_record = {
                        "macro_feature": macro_feature,
                        "normalized_family": validation["normalized_family"],
                        "face_indices": component["face_indices"],
                        "face_count": len(component["face_indices"]),
                        "reasons": [rib_shape_reason],
                        "evidence": validation["evidence"],
                        "action": "demoted_to_stock",
                    }
                    invalid_components.append(invalid_record)
                    demoted_face_count += demote_faces_to_stock(
                        faces,
                        component["face_indices"],
                        [rib_shape_reason],
                    )
                    continue
            if macro_feature == "counterbore_hole":
                counterbore_role_reason = _counterbore_outer_role_invalid_reason(component["face_indices"], graph_data)
                if counterbore_role_reason is not None:
                    invalid_record = {
                        "macro_feature": macro_feature,
                        "normalized_family": validation["normalized_family"],
                        "face_indices": component["face_indices"],
                        "face_count": len(component["face_indices"]),
                        "reasons": [counterbore_role_reason],
                        "evidence": validation["evidence"],
                        "action": "demoted_to_stock",
                    }
                    invalid_components.append(invalid_record)
                    demoted_face_count += demote_faces_to_stock(
                        faces,
                        component["face_indices"],
                        [counterbore_role_reason],
                    )
                    continue
            if macro_feature == "through_hole_round" and not _through_hole_has_functional_opening_role(
                component["face_indices"], graph_data
            ):
                invalid_record = {
                    "macro_feature": macro_feature,
                    "normalized_family": validation["normalized_family"],
                    "face_indices": component["face_indices"],
                    "face_count": len(component["face_indices"]),
                    "reasons": ["through_hole_functional_opening_role_required"],
                    "evidence": validation["evidence"],
                    "action": "demoted_to_stock",
                }
                invalid_components.append(invalid_record)
                demoted_face_count += demote_faces_to_stock(
                    faces,
                    component["face_indices"],
                    ["through_hole_functional_opening_role_required"],
                )
                continue
            semantic_repair_face_count += repair_valid_component_semantics(
                faces,
                component["face_indices"],
                macro_feature,
                graph_data,
            )
            continue

        repair = propose_component_repair(macro_feature, component["face_indices"], graph_data)
        if repair is not None:
            repaired_components.append({
                "from_macro_feature": macro_feature,
                "to_macro_feature": repair["target_macro_feature"],
                "face_indices": component["face_indices"],
                "face_count": len(component["face_indices"]),
                "source_reasons": validation["reasons"],
                "repair_evidence": repair["evidence"],
                "action": "remapped",
            })
            remap_faces_to_macro(
                faces,
                component["face_indices"],
                repair["target_macro_feature"],
                graph_data,
                validation["reasons"],
            )
            continue

        if macro_feature == "obround_hole":
            split_record, split_demoted = repair_invalid_obround_as_transition_or_stock(
                faces,
                component["face_indices"],
                graph_data,
                validation["reasons"],
            )
            if split_record is not None:
                repaired_components.append(split_record)
                demoted_face_count += split_demoted
                continue

        invalid_record = {
            "macro_feature": macro_feature,
            "normalized_family": validation["normalized_family"],
            "face_indices": component["face_indices"],
            "face_count": len(component["face_indices"]),
            "reasons": validation["reasons"],
            "evidence": validation["evidence"],
            "action": "demoted_to_stock",
        }
        invalid_components.append(invalid_record)
        demoted_face_count += demote_faces_to_stock(faces, component["face_indices"], validation["reasons"])

    corrected["topology_postprocess"] = {
        "enabled": True,
        "same_instance_threshold": same_instance_threshold,
        "component_count": len(components),
        "invalid_component_count": len(invalid_components),
        "repaired_component_count": len(repaired_components),
        "demoted_face_count": demoted_face_count,
        "semantic_repair_face_count": semantic_repair_face_count,
        "invalid_components": invalid_components,
        "repaired_components": repaired_components,
    }
    return corrected


def repair_topology_candidates_v4_2(faces: list[dict], graph_data: dict) -> tuple[list[dict], set[int]]:
    records = []
    repaired_faces: set[int] = set()

    for candidate in find_countersink_candidates(faces, graph_data, repaired_faces):
        remap_faces_to_macro(faces, candidate, "countersink_hole", graph_data, ["candidate_cone_cylinder_countersink"])
        records.append(_repair_record("candidate", "countersink_hole", candidate, ["cone_plus_cylinder_candidate"]))
        repaired_faces.update(candidate)

    for candidate in find_counterbore_candidates(faces, graph_data, repaired_faces):
        remap_faces_to_macro(faces, candidate, "counterbore_hole", graph_data, ["candidate_counterbore_large_step_small"])
        repair_counterbore_semantics(faces, candidate, graph_data)
        records.append(_repair_record("candidate", "counterbore_hole", candidate, ["large_cylinder_shoulder_small_through_cylinder"]))
        repaired_faces.update(candidate)

    for candidate in find_cylinder_boss_candidates(faces, graph_data, repaired_faces):
        remap_faces_to_macro(faces, candidate, "cylinder_boss", graph_data, ["candidate_exterior_cylinder_boss"])
        records.append(_repair_record("candidate", "cylinder_boss", candidate, ["exterior_cylinder_side_plus_top_candidate"]))
        repaired_faces.update(candidate)

    for candidate in find_blind_hole_candidates(faces, graph_data, repaired_faces):
        remap_faces_to_macro(faces, candidate, "blind_hole_round", graph_data, ["candidate_cylinder_bottom_blind_hole"])
        records.append(_repair_record("candidate", "blind_hole_round", candidate, ["cylinder_wall_plus_bottom_candidate"]))
        repaired_faces.update(candidate)

    for candidate in find_obround_hole_candidates(faces, graph_data, repaired_faces):
        remap_faces_to_macro(faces, candidate, "obround_hole", graph_data, ["candidate_obround_topology_recovery"])
        records.append(_repair_record("candidate", "obround_hole", candidate, ["two_planes_plus_two_cylindrical_ends"]))
        repaired_faces.update(candidate)

    for candidate in find_through_hole_opening_candidates(faces, graph_data, repaired_faces):
        remap_faces_to_macro(faces, candidate, "through_hole_round", graph_data, ["candidate_through_hole_opening_role"])
        records.append(_repair_record("candidate", "through_hole_round", candidate, ["two_circular_opening_edges"]))
        repaired_faces.update(candidate)

    for candidate in find_mixed_rib_candidates(faces, graph_data, repaired_faces):
        candidate = _rib_candidate_with_absorbed_transition_faces(candidate, faces, graph_data, repaired_faces, min_candidate_neighbors=2)
        remap_faces_to_macro(faces, candidate, "rib", graph_data, ["candidate_mixed_prediction_thin_wall_rib"])
        repair_rib_semantics(faces, candidate, graph_data)
        records.append(_repair_record("candidate", "rib", candidate, ["rib_seed_plus_two_convex_connected_planes"]))
        repaired_faces.update(candidate)

    for candidate in find_existing_rib_web_cap_completion_candidates(faces, graph_data, repaired_faces):
        remap_faces_to_macro(faces, candidate, "rib", graph_data, ["candidate_existing_rib_web_small_cap_completion"])
        repair_rib_semantics(faces, candidate, graph_data)
        records.append(_repair_record("candidate", "rib", candidate, ["existing_rib_or_web_plus_small_stock_end_caps"]))
        repaired_faces.update(candidate)

    for candidate in find_transition_cap_rib_candidates(faces, graph_data, repaired_faces):
        remap_faces_to_macro(faces, candidate, "rib", graph_data, ["candidate_transition_cap_thin_wall_rib"])
        repair_rib_semantics(faces, candidate, graph_data)
        records.append(_repair_record("candidate", "rib", candidate, ["two_parallel_thin_planes_plus_convex_transition_cap"]))
        repaired_faces.update(candidate)

    for candidate in find_pocket_embedded_rib_candidates(faces, graph_data, repaired_faces):
        remap_faces_to_macro(faces, candidate, "rib", graph_data, ["candidate_pocket_embedded_thin_wall_rib"])
        repair_rib_semantics(faces, candidate, graph_data)
        records.append(_repair_record("candidate", "rib", candidate, ["pocket_embedded_high_aspect_thin_faces_plus_support_face"]))
        repaired_faces.update(candidate)

    for candidate in find_rectangular_prism_candidates(faces, graph_data, repaired_faces, "pocket_rect", "concave"):
        if _rectangular_pocket_shape_invalid_reason(candidate, graph_data) is not None:
            continue
        remap_faces_to_macro(faces, candidate, "pocket_rect", graph_data, ["candidate_rectangular_pocket_completion"])
        repair_rectangular_prism_semantics(faces, candidate, "pocket_rect", graph_data, "concave")
        records.append(_repair_record("candidate", "pocket_rect", candidate, ["five_concave_planar_faces"]))
        repaired_faces.update(candidate)

    for candidate in find_rectangular_prism_candidates(faces, graph_data, repaired_faces, "rectangular_boss", "convex"):
        remap_faces_to_macro(faces, candidate, "rectangular_boss", graph_data, ["candidate_rectangular_boss_completion"])
        repair_rectangular_prism_semantics(faces, candidate, "rectangular_boss", graph_data, "convex")
        records.append(_repair_record("candidate", "rectangular_boss", candidate, ["five_convex_planar_faces"]))
        repaired_faces.update(candidate)

    for candidate in find_supported_rectangular_boss_candidates(faces, graph_data, repaired_faces):
        remap_faces_to_macro(faces, candidate, "rectangular_boss", graph_data, ["candidate_supported_rectangular_boss"])
        repair_rectangular_prism_semantics(faces, candidate, "rectangular_boss", graph_data, "convex")
        records.append(_repair_record("candidate", "rectangular_boss", candidate, ["five_convex_planar_faces_plus_large_support"]))
        repaired_faces.update(candidate)

    for candidate in find_existing_rib_transition_absorption_candidates(faces, graph_data, repaired_faces):
        candidate, through_holes = _rib_candidate_without_functional_through_holes(candidate, graph_data)
        if through_holes:
            remap_faces_to_macro(faces, through_holes, "through_hole_round", graph_data, ["candidate_rib_full_cylinder_split_to_through_hole"])
            records.append(_repair_record("candidate", "through_hole_round", through_holes, ["rib_candidate_contains_full_through_hole_cylinder"]))
            repaired_faces.update(through_holes)
        if not candidate or not validate_feature_topology("rib", candidate, graph_data)["valid"]:
            continue
        remap_faces_to_macro(faces, candidate, "rib", graph_data, ["existing_rib_absorbed_adjacent_transition"])
        repair_rib_semantics(faces, candidate, graph_data)
        records.append(_repair_record("candidate", "rib", candidate, ["valid_rib_plus_adjacent_transition_faces"]))
        repaired_faces.update(candidate)

    for candidate in find_u_slot_candidates(faces, graph_data, repaired_faces):
        remap_faces_to_macro(faces, candidate, "slot_u", graph_data, ["candidate_u_slot_cross_section"])
        records.append(_repair_record("candidate", "slot_u", candidate, ["cylindrical_bottom_plus_two_side_planes"]))
        repaired_faces.update(candidate)

    for candidate in find_rectangular_slot_candidates(faces, graph_data, repaired_faces):
        remap_faces_to_macro(faces, candidate, "slot_rect", graph_data, ["candidate_rectangular_slot_cross_section"])
        repair_rectangular_slot_semantics(faces, candidate, graph_data)
        records.append(_repair_record("candidate", "slot_rect", candidate, ["planar_bottom_plus_two_concave_side_planes"]))
        repaired_faces.update(candidate)

    for candidate in find_o_slot_candidates(faces, graph_data, repaired_faces):
        remap_faces_to_macro(faces, candidate, "slot_o", graph_data, ["candidate_o_slot_large_cylindrical_span"])
        records.append(_repair_record("candidate", "slot_o", candidate, ["large_swept_or_cylindrical_slot_surface"]))
        repaired_faces.update(candidate)

    for candidate in find_v_slot_stock_candidates(faces, graph_data, repaired_faces):
        remap_faces_to_macro(faces, candidate, "slot_v", graph_data, ["candidate_stock_v_slot_pair"])
        records.append(_repair_record("candidate", "slot_v", candidate, ["two_stock_long_inclined_side_planes"]))
        repaired_faces.update(candidate)

    for candidate in find_rib_stock_candidates(faces, graph_data, repaired_faces):
        candidate = _rib_candidate_with_absorbed_transition_faces(candidate, faces, graph_data, repaired_faces)
        remap_faces_to_macro(faces, candidate, "rib", graph_data, ["candidate_stock_thin_wall_rib"])
        repair_rib_semantics(faces, candidate, graph_data)
        records.append(_repair_record("candidate", "rib", candidate, ["thin_parallel_side_faces_plus_top_face"]))
        repaired_faces.update(candidate)

    for candidate in find_fillet_stock_candidates(faces, graph_data, repaired_faces):
        remap_faces_to_macro(faces, candidate, "fillet", graph_data, ["candidate_stock_curved_transition_face"])
        records.append(_repair_record("candidate", "fillet", candidate, ["single_curved_transition_stock_face"]))
        repaired_faces.update(candidate)

    for candidate in find_chamfer_stock_candidates(faces, graph_data, repaired_faces):
        remap_faces_to_macro(faces, candidate, "chamfer", graph_data, ["candidate_stock_narrow_bevel_face"])
        records.append(_repair_record("candidate", "chamfer", candidate, ["single_narrow_high_aspect_bevel_plane"]))
        repaired_faces.update(candidate)

    return records, repaired_faces


def connected_components_for_prediction(
    pred_faces: list[dict],
    graph_data: dict,
    *,
    same_instance_threshold: float = 0.5,
) -> list[dict]:
    node_count = len(graph_data.get("nodes", []))
    face_by_idx = {int(face["idx"]): face for face in pred_faces}
    adjacency = build_postprocess_adjacency(
        graph_data=graph_data,
        pred_faces=face_by_idx,
        node_count=node_count,
        same_instance_threshold=same_instance_threshold,
    )
    visited = set()
    components = []

    for face in pred_faces:
        start_idx = int(face["idx"])
        if start_idx in visited:
            continue
        macro_feature_id = int(face.get("macro_feature_id", 0))
        if macro_feature_id <= 0:
            visited.add(start_idx)
            continue

        macro_feature = face.get("macro_feature", "unknown_feature")
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

        components.append({
            "macro_feature_id": macro_feature_id,
            "macro_feature": macro_feature,
            "face_indices": sorted(component),
        })

    return components


def build_postprocess_adjacency(
    *,
    graph_data: dict,
    pred_faces: dict[int, dict],
    node_count: int,
    same_instance_threshold: float,
) -> dict[int, set[int]]:
    adjacency = {idx: set() for idx in range(node_count)}
    edge_scores = {
        (int(edge["source_face"]), int(edge["target_face"])): float(edge.get("same_instance_probability", 1.0))
        for edge in graph_data.get("prediction_edge_same_instance", [])
    }
    has_edge_scores = bool(edge_scores)

    for edge in graph_data.get("edges", []):
        if not isinstance(edge, list) or len(edge) != 2:
            continue
        src, dst = int(edge[0]), int(edge[1])
        if src not in pred_faces or dst not in pred_faces:
            continue
        if has_edge_scores:
            score = max(edge_scores.get((src, dst), 0.0), edge_scores.get((dst, src), 0.0))
            if score < same_instance_threshold:
                continue
        adjacency.setdefault(src, set()).add(dst)
        adjacency.setdefault(dst, set()).add(src)

    return adjacency


def find_countersink_candidates(faces: list[dict], graph_data: dict, repaired_faces: set[int]) -> list[list[int]]:
    face_by_idx = {int(face["idx"]): face for face in faces}
    nodes = _nodes_by_idx(graph_data)
    candidates = []
    for src, dst, attr in _edge_records(graph_data):
        if src in repaired_faces or dst in repaired_faces or not _is_circle_edge(attr):
            continue
        left = nodes.get(src, {})
        right = nodes.get(dst, {})
        primitives = {left.get("primitive_surface"), right.get("primitive_surface")}
        if primitives != {"cone", "cylinder"}:
            continue
        cone_idx = src if left.get("primitive_surface") == "cone" else dst
        cylinder_idx = dst if cone_idx == src else src
        cone_face = face_by_idx.get(cone_idx, {})
        cylinder_face = face_by_idx.get(cylinder_idx, {})
        if not _face_allowed_for_countersink_cone(cone_face, cone_idx, cylinder_idx, graph_data):
            continue
        if not _face_allowed_for_countersink_cylinder(cylinder_face):
            continue
        if not _countersink_cylinder_has_exit_opening(cylinder_idx, cone_idx, graph_data):
            continue
        candidate = sorted([cone_idx, cylinder_idx])
        if validate_feature_topology("countersink_hole", candidate, graph_data)["valid"] and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def find_counterbore_candidates(faces: list[dict], graph_data: dict, repaired_faces: set[int]) -> list[list[int]]:
    face_by_idx = {int(face["idx"]): face for face in faces}
    nodes = _nodes_by_idx(graph_data)
    candidates = []
    for plane_idx, node in sorted(nodes.items()):
        if plane_idx in repaired_faces or node.get("primitive_surface") != "plane":
            continue
        plane_face = face_by_idx.get(plane_idx, {})
        if not _face_allowed_for_counterbore_part(plane_face):
            continue
        cylinder_neighbors = sorted(_circular_cylinder_neighbors(plane_idx, graph_data) - repaired_faces)
        if len(cylinder_neighbors) < 2:
            continue
        for left_idx, right_idx in combinations(cylinder_neighbors, 2):
            left = nodes.get(left_idx, {})
            right = nodes.get(right_idx, {})
            if not _cylinders_are_coaxial(left, right):
                continue
            left_radius = _node_radius(left)
            right_radius = _node_radius(right)
            if min(left_radius, right_radius) <= 1e-9:
                continue
            if max(left_radius, right_radius) / min(left_radius, right_radius) < 1.15:
                continue
            large_idx, small_idx = (left_idx, right_idx) if left_radius >= right_radius else (right_idx, left_idx)
            large_face = face_by_idx.get(large_idx, {})
            small_face = face_by_idx.get(small_idx, {})
            candidate_faces = [large_face, plane_face, small_face]
            if not (
                any(_face_has_counterbore_seed(face) for face in candidate_faces)
                or _face_has_stepped_counterbore_shoulder_seed(plane_face)
            ):
                continue
            if not _counterbore_shoulder_plane_is_local(plane_idx, large_idx, small_idx, graph_data):
                continue
            if not _face_allowed_for_counterbore_part(large_face):
                continue
            if not _face_allowed_for_counterbore_part(small_face):
                continue
            if _small_cylinder_has_bottom_plane(small_idx, plane_idx, graph_data):
                continue
            candidate = sorted([large_idx, plane_idx, small_idx])
            if _counterbore_outer_role_invalid_reason(candidate, graph_data) is not None:
                continue
            if validate_feature_topology("counterbore_hole", candidate, graph_data)["valid"] and candidate not in candidates:
                candidates.append(candidate)
    return candidates


def find_blind_hole_candidates(faces: list[dict], graph_data: dict, repaired_faces: set[int]) -> list[list[int]]:
    face_by_idx = {int(face["idx"]): face for face in faces}
    nodes = _nodes_by_idx(graph_data)
    candidates = []
    for src, dst, attr in _edge_records(graph_data):
        if src in repaired_faces or dst in repaired_faces or not _is_circle_edge(attr):
            continue
        left = nodes.get(src, {})
        right = nodes.get(dst, {})
        primitives = {left.get("primitive_surface"), right.get("primitive_surface")}
        if primitives != {"cylinder", "plane"}:
            continue
        cylinder_idx = src if left.get("primitive_surface") == "cylinder" else dst
        plane_idx = dst if cylinder_idx == src else src
        cylinder_face = face_by_idx.get(cylinder_idx, {})
        plane_face = face_by_idx.get(plane_idx, {})
        if not (
            _face_looks_like_blind_cylinder(cylinder_face)
            or (
                _face_looks_like_round_boss_or_hole_candidate(cylinder_face, plane_face)
                and _cylinder_has_inner_hole_role(cylinder_idx, graph_data)
            )
        ):
            continue
        if plane_face.get("macro_feature") not in {"cylinder_boss", "blind_hole_round"}:
            continue
        candidate = sorted([cylinder_idx, plane_idx])
        if validate_feature_topology("blind_hole_round", candidate, graph_data)["valid"] and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def find_cylinder_boss_candidates(faces: list[dict], graph_data: dict, repaired_faces: set[int]) -> list[list[int]]:
    face_by_idx = {int(face["idx"]): face for face in faces}
    nodes = _nodes_by_idx(graph_data)
    candidates = []
    for src, dst, attr in _edge_records(graph_data):
        if src in repaired_faces or dst in repaired_faces or not _is_circle_edge(attr):
            continue
        left = nodes.get(src, {})
        right = nodes.get(dst, {})
        primitives = {left.get("primitive_surface"), right.get("primitive_surface")}
        if primitives != {"cylinder", "plane"}:
            continue
        cylinder_idx = src if left.get("primitive_surface") == "cylinder" else dst
        plane_idx = dst if cylinder_idx == src else src
        cylinder_face = face_by_idx.get(cylinder_idx, {})
        plane_face = face_by_idx.get(plane_idx, {})
        if not _face_looks_like_round_boss_or_hole_candidate(cylinder_face, plane_face):
            continue
        if _cylinder_has_inner_hole_role(cylinder_idx, graph_data):
            continue
        if not _cap_plane_is_exterior_to_main_slab(cylinder_idx, plane_idx, graph_data):
            continue
        candidate = sorted([cylinder_idx, plane_idx])
        if validate_feature_topology("cylinder_boss", candidate, graph_data)["valid"] and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def find_obround_hole_candidates(faces: list[dict], graph_data: dict, repaired_faces: set[int]) -> list[list[int]]:
    face_by_idx = {int(face["idx"]): face for face in faces}
    nodes = _nodes_by_idx(graph_data)
    allowed = {
        idx
        for idx, node in nodes.items()
        if idx not in repaired_faces
        and node.get("primitive_surface") in {"plane", "cylinder"}
        and _face_allowed_for_obround_candidate(face_by_idx.get(idx, {}))
    }
    cylinders = sorted(idx for idx in allowed if nodes.get(idx, {}).get("primitive_surface") == "cylinder")
    candidates: list[list[int]] = []
    occupied: set[int] = set()
    for left_idx, right_idx in combinations(cylinders, 2):
        if left_idx in occupied or right_idx in occupied:
            continue
        left = nodes.get(left_idx, {})
        right = nodes.get(right_idx, {})
        if _angle_between_abs(_node_axis(left), _node_axis(right)) > 0.14:
            continue
        left_radius = _node_radius(left)
        right_radius = _node_radius(right)
        if min(left_radius, right_radius) <= 1e-9:
            continue
        if max(left_radius, right_radius) / min(left_radius, right_radius) > 1.2:
            continue
        common_planes = sorted(
            _concave_line_plane_neighbors(left_idx, graph_data, allowed)
            & _concave_line_plane_neighbors(right_idx, graph_data, allowed)
        )
        for plane_pair in combinations(common_planes, 2):
            candidate = sorted([left_idx, right_idx, *plane_pair])
            if set(candidate) & occupied:
                continue
            if candidate in candidates:
                continue
            if validate_feature_topology("obround_hole", candidate, graph_data)["valid"]:
                candidates.append(candidate)
                occupied.update(candidate)
                break
    return candidates


def find_through_hole_opening_candidates(faces: list[dict], graph_data: dict, repaired_faces: set[int]) -> list[list[int]]:
    face_by_idx = {int(face["idx"]): face for face in faces}
    nodes = _nodes_by_idx(graph_data)
    candidates = []
    for idx, node in nodes.items():
        if idx in repaired_faces or node.get("primitive_surface") != "cylinder":
            continue
        face = face_by_idx.get(idx, {})
        if not _face_allowed_for_through_hole_opening_candidate(face):
            continue
        if _looks_like_transition_cylinder(idx, graph_data):
            continue
        candidate = [idx]
        if not validate_feature_topology("through_hole_round", candidate, graph_data)["valid"]:
            continue
        if not _through_hole_has_functional_opening_role(candidate, graph_data):
            continue
        candidates.append(candidate)
    return candidates


def find_mixed_rib_candidates(faces: list[dict], graph_data: dict, repaired_faces: set[int]) -> list[list[int]]:
    face_by_idx = {int(face["idx"]): face for face in faces}
    nodes = _nodes_by_idx(graph_data)
    rib_seed_faces = sorted(
        int(face["idx"])
        for face in faces
        if int(face["idx"]) not in repaired_faces
        and _face_has_macro_evidence(face, "rib")
        and nodes.get(int(face["idx"]), {}).get("primitive_surface") == "plane"
    )
    allowed = {
        idx
        for idx, node in nodes.items()
        if idx not in repaired_faces
        and node.get("primitive_surface") == "plane"
        and _face_allowed_for_mixed_rib_candidate(face_by_idx.get(idx, {}))
    }
    candidates: list[list[int]] = []
    occupied: set[int] = set()
    for seed_idx in rib_seed_faces:
        if seed_idx in occupied:
            continue
        neighbors = sorted(_convex_line_neighbors(seed_idx, graph_data) & allowed)
        for left_idx, right_idx in combinations(neighbors, 2):
            candidate = sorted([seed_idx, left_idx, right_idx])
            if set(candidate) & occupied:
                continue
            if candidate in candidates:
                continue
            if not _mixed_rib_candidate_has_non_rib_support(candidate, face_by_idx):
                continue
            if _rib_candidate_is_too_large_for_model_context(candidate, graph_data):
                continue
            if _rib_component_shape_invalid_reason(candidate, graph_data) is not None:
                continue
            if validate_feature_topology("rib", candidate, graph_data)["valid"]:
                candidates.append(candidate)
                occupied.update(candidate)
                break
    return candidates

def find_rectangular_prism_candidates(
    faces: list[dict],
    graph_data: dict,
    repaired_faces: set[int],
    target_macro: str,
    required_convexity: str,
) -> list[list[int]]:
    face_by_idx = {int(face["idx"]): face for face in faces}
    nodes = _nodes_by_idx(graph_data)
    seed_faces = {
        int(face["idx"])
        for face in faces
        if int(face["idx"]) not in repaired_faces and _face_has_macro_evidence(face, target_macro)
    }
    if not seed_faces:
        return []

    candidates = [
        candidate
        for candidate in _valid_rectangular_seed_components(
            seed_faces,
            target_macro,
            required_convexity,
            graph_data,
        )
        if _rectangular_prism_candidate_supported(target_macro, candidate, graph_data)
    ]
    protected_seed_faces = {idx for candidate in candidates for idx in candidate}
    for seed_idx in sorted(seed_faces):
        if seed_idx in protected_seed_faces:
            continue
        seed_node = nodes.get(seed_idx, {})
        seed_area = _node_area(seed_node)
        if seed_area <= 1e-9:
            continue
        max_stock_area = seed_area * 6.0
        allowed = {
            idx
            for idx, node in nodes.items()
            if idx not in repaired_faces
            and node.get("primitive_surface") == "plane"
            and _face_allowed_for_rectangular_completion(face_by_idx.get(idx, {}), target_macro)
            and (
                idx in seed_faces
                or _face_has_macro_evidence(face_by_idx.get(idx, {}), target_macro)
                or _node_area(node) <= max_stock_area
            )
        }
        component = _plane_component_from_seed(seed_idx, allowed, graph_data, required_convexity)
        if len(component) < 5:
            continue
        for candidate in _best_rectangular_subsets(
            component,
            seed_faces,
            target_macro,
            graph_data,
            anchor_idx=seed_idx,
            required_convexity=required_convexity,
        ):
            if candidate not in candidates and _rectangular_prism_candidate_supported(target_macro, candidate, graph_data):
                candidates.append(candidate)
            break
    return candidates


def _valid_rectangular_seed_components(
    seed_faces: set[int],
    target_macro: str,
    required_convexity: str,
    graph_data: dict,
) -> list[list[int]]:
    adjacency = {idx: set() for idx in seed_faces}
    for src, dst, attr in _edge_records(graph_data):
        if src not in seed_faces or dst not in seed_faces:
            continue
        if not _is_line_edge(attr) or not _edge_has_convexity(attr, required_convexity):
            continue
        adjacency.setdefault(src, set()).add(dst)
        adjacency.setdefault(dst, set()).add(src)

    candidates: list[list[int]] = []
    visited: set[int] = set()
    for seed_idx in sorted(seed_faces):
        if seed_idx in visited:
            continue
        component = []
        queue = deque([seed_idx])
        visited.add(seed_idx)
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(adjacency.get(current, set())):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        candidate = sorted(component)
        if validate_feature_topology(target_macro, candidate, graph_data)["valid"]:
            candidates.append(candidate)
    return candidates


def find_supported_rectangular_boss_candidates(faces: list[dict], graph_data: dict, repaired_faces: set[int]) -> list[list[int]]:
    face_by_idx = {int(face["idx"]): face for face in faces}
    nodes = _nodes_by_idx(graph_data)
    allowed = {
        idx
        for idx, node in nodes.items()
        if idx not in repaired_faces
        and node.get("primitive_surface") == "plane"
        and _face_allowed_for_rectangular_completion(face_by_idx.get(idx, {}), "rectangular_boss")
    }
    candidates = []
    occupied: set[int] = set()
    for center_idx in sorted(allowed):
        if center_idx in occupied:
            continue
        convex_neighbors = sorted(_convex_line_neighbors(center_idx, graph_data) & allowed)
        if len(convex_neighbors) < 4:
            continue
        for side_indices in combinations(convex_neighbors, 4):
            candidate = sorted([center_idx, *side_indices])
            if set(candidate) & occupied:
                continue
            if candidate in candidates:
                continue
            if not validate_feature_topology("rectangular_boss", candidate, graph_data)["valid"]:
                continue
            if not _rectangular_boss_has_large_support(candidate, graph_data):
                continue
            candidates.append(candidate)
            occupied.update(candidate)
            break
    candidates.sort(key=lambda item: (sum(_node_area(nodes.get(idx, {})) for idx in item), item))
    return candidates


def find_u_slot_candidates(faces: list[dict], graph_data: dict, repaired_faces: set[int]) -> list[list[int]]:
    face_by_idx = {int(face["idx"]): face for face in faces}
    nodes = _nodes_by_idx(graph_data)
    edge_records = _edge_records(graph_data)
    candidates = []
    cylinder_indices = [
        idx for idx, node in nodes.items()
        if idx not in repaired_faces and node.get("primitive_surface") in {"cylinder", "surface_of_revolution", "surface_of_extrusion"}
    ]
    for cylinder_idx in cylinder_indices:
        cylinder_face = face_by_idx.get(cylinder_idx)
        if cylinder_face is None:
            continue
        if not _face_allowed_for_u_slot_bottom_candidate(cylinder_face):
            continue
        if _node_descriptor(nodes.get(cylinder_idx, {}), "uv_u_span") > 5.5:
            continue
        if _belongs_to_multi_cylinder_obround_prediction(cylinder_idx, face_by_idx, graph_data):
            continue
        side_planes = []
        for src, dst, attr in edge_records:
            if cylinder_idx not in {src, dst} or not _is_line_edge(attr) or not _is_concave_edge(attr):
                continue
            neighbor = dst if src == cylinder_idx else src
            if neighbor in repaired_faces:
                continue
            node = nodes.get(neighbor, {})
            face = face_by_idx.get(neighbor, {})
            if node.get("primitive_surface") != "plane":
                continue
            if face.get("macro_feature") not in {"obround_hole", "slot_u", "slot_rect", "stock", "web"}:
                continue
            if _node_descriptor(node, "projected_aspect_ratio") < 5.0:
                continue
            side_planes.append(neighbor)
        side_planes = sorted(set(side_planes))
        if len(side_planes) < 2:
            continue
        candidate = sorted([cylinder_idx, *side_planes[:2]])
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _face_allowed_for_u_slot_bottom_candidate(face: dict) -> bool:
    macros = {
        "obround_hole",
        "slot_u",
        "fillet",
        "stock",
        "web",
        "through_hole_round",
        "countersink_hole",
    }
    semantics = {
        "obround_hole_end_cylinder",
        "slot_u_bottom_plane",
        "fillet_blend_surface",
        "stock_face",
        "web_side_plane",
        "through_hole_cylinder",
        "countersink_small_cylinder",
    }
    return (
        face.get("macro_feature", "stock") in macros
        or face.get("raw_macro_feature") in macros
        or face.get("semantic_subtype") in semantics
        or face.get("raw_semantic_subtype") in semantics
    )


def find_rectangular_slot_candidates(faces: list[dict], graph_data: dict, repaired_faces: set[int]) -> list[list[int]]:
    face_by_idx = {int(face["idx"]): face for face in faces}
    nodes = _nodes_by_idx(graph_data)
    candidates = []
    allowed_macros = {"stock", "web", "slot_rect", "slot_u"}
    for bottom_idx, bottom in nodes.items():
        if bottom_idx in repaired_faces or bottom.get("primitive_surface") != "plane":
            continue
        if not _face_allowed_for_stock_candidate(face_by_idx.get(bottom_idx, {}), allowed_macros):
            continue
        if _node_descriptor(bottom, "projected_aspect_ratio") < 5.0:
            continue
        side_planes = []
        for src, dst, attr in _edge_records(graph_data):
            if bottom_idx not in {src, dst} or not _is_line_edge(attr) or not _is_concave_edge(attr):
                continue
            neighbor = dst if src == bottom_idx else src
            if neighbor in repaired_faces:
                continue
            node = nodes.get(neighbor, {})
            face = face_by_idx.get(neighbor, {})
            if node.get("primitive_surface") != "plane":
                continue
            if not _face_allowed_for_stock_candidate(face, allowed_macros):
                continue
            if _node_descriptor(node, "projected_aspect_ratio") < 5.0:
                continue
            side_planes.append(neighbor)
        for side_pair in combinations(sorted(set(side_planes)), 2):
            candidate = sorted([bottom_idx, *side_pair])
            if candidate in candidates:
                continue
            if not _rectangular_slot_candidate_has_consistent_long_faces(candidate, graph_data):
                continue
            if validate_feature_topology("slot_rect", candidate, graph_data)["valid"]:
                candidates.append(candidate)
                break
    return candidates


def find_o_slot_candidates(faces: list[dict], graph_data: dict, repaired_faces: set[int]) -> list[list[int]]:
    face_by_idx = {int(face["idx"]): face for face in faces}
    nodes = _nodes_by_idx(graph_data)
    candidates = []
    for idx, node in nodes.items():
        if idx in repaired_faces:
            continue
        primitive = node.get("primitive_surface")
        if primitive not in {"cylinder", "surface_of_revolution", "surface_of_extrusion", "bspline_surface"}:
            continue
        face = face_by_idx.get(idx)
        if face is None or face.get("macro_feature") not in {"fillet", "slot_o", "stock", "obround_hole"}:
            continue
        if _belongs_to_multi_cylinder_obround_prediction(idx, face_by_idx, graph_data):
            continue
        if primitive == "cylinder":
            if _looks_like_exterior_cylinder_not_slot_bottom(idx, node, graph_data):
                continue
            uv_u_span = _node_descriptor(node, "uv_u_span")
            if uv_u_span > 1e-9 and uv_u_span < 2.8:
                continue
            if _has_u_slot_like_concave_side_planes(idx, graph_data, repaired_faces):
                continue
            angle = _cylindrical_face_angle(idx, node, graph_data)
            if angle < 2.35:
                continue
            if _long_line_edge_count(idx, graph_data) < 2:
                continue
        elif not _looks_like_large_swept_slot_surface(idx, node, graph_data):
            continue
        candidates.append([idx])
    return candidates


def find_v_slot_stock_candidates(faces: list[dict], graph_data: dict, repaired_faces: set[int]) -> list[list[int]]:
    face_by_idx = {int(face["idx"]): face for face in faces}
    nodes = _nodes_by_idx(graph_data)
    candidates = []
    for src, dst, attr in _edge_records(graph_data):
        if src >= dst or src in repaired_faces or dst in repaired_faces:
            continue
        if not _is_line_edge(attr) or not _is_concave_edge(attr):
            continue
        left = nodes.get(src, {})
        right = nodes.get(dst, {})
        if left.get("primitive_surface") != "plane" or right.get("primitive_surface") != "plane":
            continue
        if not _face_allowed_for_stock_candidate(face_by_idx.get(src, {}), {"stock", "slot_v", "slot_rect", "chamfer"}):
            continue
        if not _face_allowed_for_stock_candidate(face_by_idx.get(dst, {}), {"stock", "slot_v", "slot_rect", "chamfer"}):
            continue
        if _node_descriptor(left, "projected_aspect_ratio") < 5.0:
            continue
        if _node_descriptor(right, "projected_aspect_ratio") < 5.0:
            continue
        candidate = sorted([src, dst])
        if validate_feature_topology("slot_v", candidate, graph_data)["valid"] and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def find_existing_rib_web_cap_completion_candidates(
    faces: list[dict],
    graph_data: dict,
    repaired_faces: set[int],
) -> list[list[int]]:
    face_by_idx = {int(face["idx"]): face for face in faces}
    nodes = _nodes_by_idx(graph_data)
    rib_faces = {
        int(face["idx"])
        for face in faces
        if int(face["idx"]) not in repaired_faces
        and (
            _face_has_macro_evidence(face, "rib")
            or _face_has_macro_evidence(face, "web")
        )
    }
    candidates: list[list[int]] = []
    for component in _connected_subsets(rib_faces, graph_data):
        if not validate_feature_topology("rib", component, graph_data)["valid"]:
            continue
        expanded = _rib_candidate_with_small_stock_cap_faces(component, face_by_idx, graph_data, repaired_faces)
        if set(expanded) == set(component):
            continue
        if _rib_candidate_is_too_large_for_model_context(expanded, graph_data):
            continue
        if validate_feature_topology("rib", expanded, graph_data)["valid"] and expanded not in candidates:
            candidates.append(expanded)
    return candidates


def find_transition_cap_rib_candidates(
    faces: list[dict],
    graph_data: dict,
    repaired_faces: set[int],
) -> list[list[int]]:
    face_by_idx = {int(face["idx"]): face for face in faces}
    nodes = _nodes_by_idx(graph_data)
    candidates: list[list[int]] = []
    for cap_idx, cap_node in sorted(nodes.items()):
        if cap_idx in repaired_faces:
            continue
        if cap_node.get("primitive_surface") != "cone":
            continue
        cap_face = face_by_idx.get(cap_idx, {})
        if not _face_allowed_for_transition_cap_rib_candidate(cap_face):
            continue
        plane_neighbors = sorted(
            neighbor
            for neighbor in _convex_line_neighbors(cap_idx, graph_data)
            if neighbor not in repaired_faces
            and nodes.get(neighbor, {}).get("primitive_surface") == "plane"
            and _face_allowed_for_stock_candidate(face_by_idx.get(neighbor, {}), {"stock", "rib", "web", "fillet", "chamfer"})
        )
        for left_idx, right_idx in combinations(plane_neighbors, 2):
            left = nodes.get(left_idx, {})
            right = nodes.get(right_idx, {})
            if _angle_between_abs(_node_normal(left), _node_normal(right)) > 0.18:
                continue
            if _node_descriptor(left, "projected_aspect_ratio") < 2.0:
                continue
            if _node_descriptor(right, "projected_aspect_ratio") < 2.0:
                continue
            max_plane_area = max(_node_area(left), _node_area(right))
            if _node_area(cap_node) > max(max_plane_area * 0.35, 1e-6):
                continue
            candidate = sorted([cap_idx, left_idx, right_idx])
            if _rib_candidate_is_too_large_for_model_context(candidate, graph_data):
                continue
            if validate_feature_topology("rib", candidate, graph_data)["valid"] and candidate not in candidates:
                candidates.append(candidate)
                break
    return candidates


def find_pocket_embedded_rib_candidates(
    faces: list[dict],
    graph_data: dict,
    repaired_faces: set[int],
) -> list[list[int]]:
    face_by_idx = {int(face["idx"]): face for face in faces}
    nodes = _nodes_by_idx(graph_data)
    raw_candidates: list[list[int]] = []
    for support_idx, support_node in sorted(nodes.items()):
        if support_idx in repaired_faces or support_node.get("primitive_surface") != "plane":
            continue
        if not _face_allowed_for_stock_candidate(face_by_idx.get(support_idx, {}), {"stock", "pocket_rect", "rib", "web"}):
            continue
        support_area = _node_area(support_node)
        if support_area <= 1e-9:
            continue
        thin_neighbors = sorted(
            neighbor
            for neighbor in _line_neighbors(support_idx, graph_data)
            if _face_is_embedded_rib_thin_plane(neighbor, support_area, face_by_idx, nodes, repaired_faces)
        )
        if len(thin_neighbors) < 5:
            continue
        candidate = sorted([support_idx, *thin_neighbors])
        if _rib_candidate_is_too_large_for_model_context(candidate, graph_data):
            continue
        if validate_feature_topology("rib", candidate, graph_data)["valid"] and candidate not in raw_candidates:
            raw_candidates.append(candidate)

    raw_candidates.sort(key=lambda item: (len(item), _candidate_area(item, nodes)), reverse=True)
    candidates: list[list[int]] = []
    occupied: set[int] = set()
    for candidate in raw_candidates:
        if set(candidate) & occupied:
            continue
        candidates.append(candidate)
        occupied.update(candidate)
    return candidates


def _rib_component_through_hole_faces(face_indices: list[int], graph_data: dict) -> list[int]:
    return sorted(
        int(idx)
        for idx in face_indices
        if _through_hole_has_functional_opening_role([int(idx)], graph_data)
        and _long_line_edge_count(int(idx), graph_data) == 0
    )


def _rib_candidate_without_functional_through_holes(
    candidate: list[int],
    graph_data: dict,
) -> tuple[list[int], list[int]]:
    through_holes = _rib_component_through_hole_faces(candidate, graph_data)
    if not through_holes:
        return sorted(int(idx) for idx in candidate), []
    through_set = set(through_holes)
    return sorted(int(idx) for idx in candidate if int(idx) not in through_set), through_holes


def _line_neighbors(face_idx: int, graph_data: dict) -> set[int]:
    neighbors = set()
    for src, dst, attr in _edge_records(graph_data):
        if face_idx not in {src, dst} or not _is_line_edge(attr):
            continue
        neighbors.add(dst if src == face_idx else src)
    return neighbors


def _candidate_area(face_indices: list[int], nodes: dict[int, dict]) -> float:
    return sum(_node_area(nodes.get(int(idx), {})) for idx in face_indices)


def _connected_subsets(face_indices: set[int], graph_data: dict) -> list[list[int]]:
    if not face_indices:
        return []
    adjacency = {idx: set() for idx in face_indices}
    for src, dst, _attr in _edge_records(graph_data):
        if src in face_indices and dst in face_indices:
            adjacency.setdefault(src, set()).add(dst)
            adjacency.setdefault(dst, set()).add(src)
    components: list[list[int]] = []
    visited: set[int] = set()
    for seed_idx in sorted(face_indices):
        if seed_idx in visited:
            continue
        queue = deque([seed_idx])
        visited.add(seed_idx)
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(adjacency.get(current, set())):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        components.append(sorted(component))
    return components


def _rib_candidate_with_small_stock_cap_faces(
    candidate: list[int],
    face_by_idx: dict[int, dict],
    graph_data: dict,
    repaired_faces: set[int],
) -> list[int]:
    nodes = _nodes_by_idx(graph_data)
    candidate_set = {int(idx) for idx in candidate}
    if not candidate_set:
        return candidate
    max_candidate_area = max(_node_area(nodes.get(idx, {})) for idx in candidate_set)
    absorbed = set(candidate_set)
    for neighbor, node in sorted(nodes.items()):
        if neighbor in repaired_faces or neighbor in absorbed:
            continue
        if node.get("primitive_surface") != "plane":
            continue
        if not _face_allowed_for_stock_candidate(face_by_idx.get(neighbor, {}), {"stock", "rib", "web", "pocket_rect"}):
            continue
        area = _node_area(node)
        if area <= 1e-9 or area > max_candidate_area * 0.45:
            continue
        if _node_descriptor(node, "projected_aspect_ratio") < 3.0:
            continue
        if not _stock_cap_has_rib_boundary_edges(neighbor, candidate_set, graph_data):
            continue
        absorbed.add(neighbor)
    return sorted(absorbed)


def _stock_cap_has_rib_boundary_edges(face_idx: int, candidate_set: set[int], graph_data: dict) -> bool:
    candidate_neighbors: set[int] = set()
    has_convex = False
    has_concave = False
    for src, dst, attr in _edge_records(graph_data):
        if face_idx not in {src, dst}:
            continue
        neighbor = dst if src == face_idx else src
        if neighbor not in candidate_set or not _is_line_edge(attr):
            continue
        candidate_neighbors.add(neighbor)
        if _edge_has_convexity(attr, "convex"):
            has_convex = True
        if _is_concave_edge(attr):
            has_concave = True
    return len(candidate_neighbors) >= 2 and has_convex and has_concave


def _face_allowed_for_transition_cap_rib_candidate(face: dict) -> bool:
    return (
        face.get("macro_feature") in {"stock", "fillet", "chamfer", "rib", "web"}
        or face.get("raw_macro_feature") in {"stock", "fillet", "chamfer", "rib", "web"}
        or face.get("semantic_subtype") in {"stock_face", "fillet_blend_surface", "chamfer_bevel_plane", "rib_top_plane", "rib_side_plane"}
        or face.get("raw_semantic_subtype") in {"stock_face", "fillet_blend_surface", "chamfer_bevel_plane", "rib_top_plane", "rib_side_plane"}
    )


def _face_is_embedded_rib_thin_plane(
    face_idx: int,
    support_area: float,
    face_by_idx: dict[int, dict],
    nodes: dict[int, dict],
    repaired_faces: set[int],
) -> bool:
    if face_idx in repaired_faces:
        return False
    node = nodes.get(face_idx, {})
    if node.get("primitive_surface") != "plane":
        return False
    if not _face_allowed_for_stock_candidate(face_by_idx.get(face_idx, {}), {"pocket_rect", "stock", "rib", "web"}):
        return False
    area = _node_area(node)
    if area <= 1e-9 or area > support_area * 0.8:
        return False
    if _node_descriptor(node, "projected_aspect_ratio") < 8.0:
        return False
    short_extent = _node_descriptor(node, "projected_short_extent")
    if short_extent > 1e-9 and short_extent > 3.5:
        return False
    return True


def find_existing_rib_transition_absorption_candidates(
    faces: list[dict],
    graph_data: dict,
    repaired_faces: set[int],
) -> list[list[int]]:
    rib_faces = {
        int(face["idx"])
        for face in faces
        if int(face["idx"]) not in repaired_faces
        and (
            _face_has_macro_evidence(face, "rib")
            or _face_has_macro_evidence(face, "web")
        )
    }
    if not rib_faces:
        return []

    adjacency = {idx: set() for idx in rib_faces}
    for src, dst, _attr in _edge_records(graph_data):
        if src not in rib_faces or dst not in rib_faces:
            continue
        adjacency.setdefault(src, set()).add(dst)
        adjacency.setdefault(dst, set()).add(src)

    candidates: list[list[int]] = []
    visited: set[int] = set()
    for seed_idx in sorted(rib_faces):
        if seed_idx in visited:
            continue
        component = []
        queue = deque([seed_idx])
        visited.add(seed_idx)
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(adjacency.get(current, set())):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        component = sorted(component)
        if not validate_feature_topology("rib", component, graph_data)["valid"]:
            continue
        expanded = _rib_candidate_with_absorbed_transition_faces(component, faces, graph_data, repaired_faces)
        if set(expanded) == set(component):
            continue
        if validate_feature_topology("rib", expanded, graph_data)["valid"] and expanded not in candidates:
            candidates.append(expanded)
    return candidates

def find_rib_stock_candidates(faces: list[dict], graph_data: dict, repaired_faces: set[int]) -> list[list[int]]:
    face_by_idx = {int(face["idx"]): face for face in faces}
    nodes = _nodes_by_idx(graph_data)
    stock_planes = [
        idx
        for idx, node in nodes.items()
        if idx not in repaired_faces
        and node.get("primitive_surface") == "plane"
        and _face_allowed_for_stock_candidate(face_by_idx.get(idx, {}), {"stock", "rib", "web"})
    ]
    candidates = []
    for i, left_idx in enumerate(stock_planes):
        for right_idx in stock_planes[i + 1:]:
            left = nodes[left_idx]
            right = nodes[right_idx]
            if _angle_between_abs(_node_normal(left), _node_normal(right)) > 0.18:
                continue
            if _node_descriptor(left, "projected_aspect_ratio") < 2.0:
                continue
            if _node_descriptor(right, "projected_aspect_ratio") < 2.0:
                continue
            thickness = _distance_between_nodes(left, right)
            if thickness <= 1e-9:
                continue
            long_extent = max(_node_descriptor(left, "projected_long_extent"), _node_descriptor(right, "projected_long_extent"))
            short_extent = max(_node_descriptor(left, "projected_short_extent"), _node_descriptor(right, "projected_short_extent"))
            if long_extent / thickness < 12.0:
                continue
            if thickness / max(short_extent, 1e-9) > 0.25:
                continue
            cap_idx = _rib_cap_between_side_faces(left_idx, right_idx, stock_planes, graph_data)
            if cap_idx is None or cap_idx in repaired_faces:
                continue
            cap = nodes.get(cap_idx, {})
            if _node_area(cap) > max(_node_area(left), _node_area(right)) * 0.4:
                continue
            cap_long_extent = _node_descriptor(cap, "projected_long_extent")
            if cap_long_extent > 1e-9 and cap_long_extent < long_extent * 0.55:
                continue
            candidate = sorted([cap_idx, left_idx, right_idx])
            if _rib_candidate_is_too_large_for_model_context(candidate, graph_data):
                continue
            if validate_feature_topology("rib", candidate, graph_data)["valid"] and candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _rib_candidate_with_absorbed_transition_faces(
    candidate: list[int],
    faces: list[dict],
    graph_data: dict,
    repaired_faces: set[int],
    *,
    min_candidate_neighbors: int = 1,
) -> list[int]:
    face_by_idx = {int(face["idx"]): face for face in faces}
    nodes = _nodes_by_idx(graph_data)
    candidate_set = {int(idx) for idx in candidate}
    if not candidate_set:
        return candidate
    max_rib_face_area = max(_node_area(nodes.get(idx, {})) for idx in candidate_set)
    absorbed = set(candidate_set)
    for src, dst, _attr in _edge_records(graph_data):
        src_in = src in candidate_set
        dst_in = dst in candidate_set
        if src_in == dst_in:
            continue
        neighbor = dst if src_in else src
        if neighbor in repaired_faces or neighbor in absorbed:
            continue
        face = face_by_idx.get(neighbor, {})
        if not _face_is_transition_for_rib_absorption(face):
            continue
        if _candidate_neighbor_count(neighbor, candidate_set, graph_data) < min_candidate_neighbors:
            continue
        node = nodes.get(neighbor, {})
        area = _node_area(node)
        if area <= 1e-9 or area > max(max_rib_face_area * 0.25, 1e-6):
            continue
        primitive = node.get("primitive_surface")
        if primitive == "plane" and _node_descriptor(node, "projected_aspect_ratio") < 8.0:
            continue
        if primitive not in {"plane", "cylinder", "toroid", "surface_of_revolution", "surface_of_extrusion", "bspline_surface"}:
            continue
        absorbed.add(neighbor)
    return sorted(absorbed)




def _candidate_neighbor_count(face_idx: int, candidate_set: set[int], graph_data: dict) -> int:
    neighbors = set()
    for src, dst, attr in _edge_records(graph_data):
        if face_idx not in {src, dst}:
            continue
        if not _is_line_edge(attr):
            continue
        other = dst if src == face_idx else src
        if other in candidate_set:
            neighbors.add(other)
    return len(neighbors)
def _face_is_transition_for_rib_absorption(face: dict) -> bool:
    transition_macros = {"fillet", "chamfer", "pocket_circular"}
    transition_semantics = {
        "fillet_blend_surface",
        "chamfer_bevel_plane",
        "circular_pocket_side_cylinder",
    }
    return (
        face.get("macro_feature") in transition_macros
        or face.get("raw_macro_feature") in transition_macros
        or face.get("semantic_subtype") in transition_semantics
        or face.get("raw_semantic_subtype") in transition_semantics
    )

def find_fillet_stock_candidates(faces: list[dict], graph_data: dict, repaired_faces: set[int]) -> list[list[int]]:
    face_by_idx = {int(face["idx"]): face for face in faces}
    nodes = _nodes_by_idx(graph_data)
    candidates = []
    for idx, node in nodes.items():
        if idx in repaired_faces:
            continue
        primitive = node.get("primitive_surface")
        if primitive not in {"cylinder", "toroid", "surface_of_revolution", "surface_of_extrusion", "bspline_surface"}:
            continue
        if not _face_allowed_for_stock_candidate(face_by_idx.get(idx, {}), {"stock", "fillet"}):
            if not _face_allowed_for_low_confidence_fillet_repair(face_by_idx.get(idx, {})):
                continue
        if _face_allowed_for_low_confidence_fillet_repair(face_by_idx.get(idx, {})) and not _looks_like_transition_cylinder(idx, graph_data):
            continue
        if _node_area(node) <= 1e-6:
            continue
        if primitive == "cylinder" and _looks_like_exterior_cylinder_not_slot_bottom(idx, node, graph_data):
            continue
        if (
            primitive == "cylinder"
            and not _face_allowed_for_low_confidence_fillet_repair(face_by_idx.get(idx, {}))
            and _looks_like_convex_partial_cylinder_opening(idx, graph_data)
        ):
            continue
        if primitive == "cylinder" and _long_line_edge_count(idx, graph_data) < 1:
            continue
        if _boundary_neighbor_count(idx, graph_data) < 2:
            continue
        candidates.append([idx])
    return candidates


def find_chamfer_stock_candidates(faces: list[dict], graph_data: dict, repaired_faces: set[int]) -> list[list[int]]:
    face_by_idx = {int(face["idx"]): face for face in faces}
    nodes = _nodes_by_idx(graph_data)
    candidates = []
    for idx, node in nodes.items():
        if idx in repaired_faces or node.get("primitive_surface") != "plane":
            continue
        if not _face_allowed_for_chamfer_candidate(face_by_idx.get(idx, {})):
            continue
        if _node_descriptor(node, "projected_aspect_ratio") < 15.0:
            continue
        if _boundary_neighbor_count(idx, graph_data) > 6:
            continue
        if not _has_chamfer_angle_edges(idx, graph_data):
            continue
        if _long_line_edge_count(idx, graph_data) < 2:
            continue
        neighbor_areas = _neighbor_areas(idx, graph_data)
        if neighbor_areas and max(neighbor_areas) < _node_area(node) * 5.0:
            continue
        candidates.append([idx])
    return candidates


# ---------------------------------------------------------------------------
# v5.23 review-targeted residual finder helpers
# ---------------------------------------------------------------------------


def _review_targeted_face_allowed(face: dict, allowed_macros: set[str], allowed_semantics: set[str]) -> bool:
    return (
        face.get("macro_feature") in allowed_macros
        or face.get("raw_macro_feature") in allowed_macros
        or face.get("semantic_subtype") in allowed_semantics
        or face.get("raw_semantic_subtype") in allowed_semantics
        or float(face.get("macro_confidence", 1.0) or 1.0) < 0.50
    )


def _review_targeted_neighbor_large_support_count(face_idx: int, graph_data: dict, *, min_area_ratio: float = 4.0) -> int:
    nodes = _nodes_by_idx(graph_data)
    node = nodes.get(face_idx, {})
    area = _node_area(node)
    if area <= 1e-9:
        return 0
    count = 0
    for src, dst, attr in _edge_records(graph_data):
        if face_idx not in {src, dst}:
            continue
        other = dst if src == face_idx else src
        if _node_area(nodes.get(other, {})) >= area * min_area_ratio:
            count += 1
    return count


def _review_targeted_has_two_long_edges(face_idx: int, graph_data: dict) -> bool:
    lengths = []
    for src, dst, attr in _edge_records(graph_data):
        if face_idx not in {src, dst}:
            continue
        if _is_line_edge(attr):
            lengths.append(float(attr.get("length", 0.0) or 0.0))
    positive = [length for length in lengths if length > 1e-9]
    if len(positive) < 2:
        return False
    positive.sort(reverse=True)
    return positive[1] >= positive[0] * 0.35


# ---------------------------------------------------------------------------
# v5.23 review-targeted residual finders
# ---------------------------------------------------------------------------


def find_review_targeted_rib_candidates(faces: list[dict], graph_data: dict, repaired_faces: set[int]) -> list[list[int]]:
    face_by_idx = {int(face["idx"]): face for face in faces}
    nodes = _nodes_by_idx(graph_data)
    candidate_faces = [
        idx
        for idx, node in nodes.items()
        if idx not in repaired_faces
        and node.get("primitive_surface") == "plane"
        and _review_targeted_face_allowed(
            face_by_idx.get(idx, {}),
            {"stock", "rib", "web", "fillet", "chamfer"},
            {"stock_face", "rib_side_plane", "rib_top_plane", "fillet_blend_surface", "chamfer_bevel_plane"},
        )
    ]
    candidates: list[list[int]] = []
    for i, left_idx in enumerate(candidate_faces):
        for right_idx in candidate_faces[i + 1:]:
            left = nodes[left_idx]
            right = nodes[right_idx]
            if _angle_between_abs(_node_normal(left), _node_normal(right)) > 0.22:
                continue
            left_aspect = _node_descriptor(left, "projected_aspect_ratio")
            right_aspect = _node_descriptor(right, "projected_aspect_ratio")
            if max(left_aspect, right_aspect) < 3.0:
                continue
            thickness = _distance_between_nodes(left, right)
            if thickness <= 1e-9:
                continue
            long_extent = max(_node_descriptor(left, "projected_long_extent"), _node_descriptor(right, "projected_long_extent"))
            short_extent = max(_node_descriptor(left, "projected_short_extent"), _node_descriptor(right, "projected_short_extent"))
            if long_extent <= 1e-9 or short_extent <= 1e-9:
                continue
            if long_extent / thickness < 6.0:
                continue
            if thickness / max(short_extent, 1e-9) > 0.45:
                continue
            cap_idx = _rib_cap_between_side_faces(left_idx, right_idx, candidate_faces, graph_data)
            if cap_idx is None or cap_idx in repaired_faces:
                continue
            cap = nodes.get(cap_idx, {})
            if cap.get("primitive_surface") != "plane":
                continue
            if _node_area(cap) > max(_node_area(left), _node_area(right)) * 0.6:
                continue
            candidate = sorted([left_idx, right_idx, cap_idx])
            if _rib_candidate_is_too_large_for_model_context(candidate, graph_data):
                continue
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def find_review_targeted_fillet_candidates(faces: list[dict], graph_data: dict, repaired_faces: set[int]) -> list[list[int]]:
    face_by_idx = {int(face["idx"]): face for face in faces}
    nodes = _nodes_by_idx(graph_data)
    candidates: list[list[int]] = []
    for idx, node in nodes.items():
        if idx in repaired_faces:
            continue
        primitive = node.get("primitive_surface")
        if primitive not in {"cylinder", "toroid", "surface_of_revolution", "surface_of_extrusion", "bspline_surface"}:
            continue
        face = face_by_idx.get(idx, {})
        if not _review_targeted_face_allowed(face, {"stock", "fillet", "blind_hole_round", "pocket_circular"}, {"stock_face", "fillet_blend_surface"}):
            continue
        if _node_area(node) <= 1e-6:
            continue
        if primitive == "cylinder" and _looks_like_exterior_cylinder_not_slot_bottom(idx, node, graph_data):
            continue
        if primitive == "cylinder" and _looks_like_convex_partial_cylinder_opening(idx, graph_data):
            continue
        if _boundary_neighbor_count(idx, graph_data) < 2:
            continue
        if _review_targeted_neighbor_large_support_count(idx, graph_data, min_area_ratio=3.0) < 1:
            continue
        if not _review_targeted_has_two_long_edges(idx, graph_data):
            continue
        candidate = [idx]
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def find_review_targeted_chamfer_candidates(faces: list[dict], graph_data: dict, repaired_faces: set[int]) -> list[list[int]]:
    face_by_idx = {int(face["idx"]): face for face in faces}
    nodes = _nodes_by_idx(graph_data)
    candidates: list[list[int]] = []
    for idx, node in nodes.items():
        if idx in repaired_faces or node.get("primitive_surface") != "plane":
            continue
        face = face_by_idx.get(idx, {})
        if not _review_targeted_face_allowed(face, {"stock", "chamfer"}, {"stock_face", "chamfer_bevel_plane"}):
            continue
        area = _node_area(node)
        if area <= 1e-6:
            continue
        aspect = _node_descriptor(node, "projected_aspect_ratio")
        if aspect < 7.0:
            continue
        if _boundary_neighbor_count(idx, graph_data) > 8:
            continue
        if _review_targeted_neighbor_large_support_count(idx, graph_data, min_area_ratio=5.0) < 1:
            continue
        if not _review_targeted_has_two_long_edges(idx, graph_data):
            continue
        candidate = [idx]
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def repair_invalid_obround_as_transition_or_stock(
    faces: list[dict],
    face_indices: list[int],
    graph_data: dict,
    reasons: list[str],
) -> tuple[dict | None, int]:
    nodes = _nodes_by_idx(graph_data)
    demote_indices = []
    fillet_indices = []
    chamfer_indices = []
    for idx in face_indices:
        primitive = nodes.get(idx, {}).get("primitive_surface")
        if primitive == "plane":
            demote_indices.append(idx)
        elif primitive in {"cylinder", "surface_of_revolution", "surface_of_extrusion"}:
            fillet_indices.append(idx)
        else:
            demote_indices.append(idx)

    if not fillet_indices and not chamfer_indices:
        return None, 0

    demoted = 0
    if demote_indices:
        demoted += demote_faces_to_stock(faces, demote_indices, reasons + ["split_invalid_obround_non_transition_faces"])
    if fillet_indices:
        remap_faces_to_macro(faces, fillet_indices, "fillet", graph_data, reasons + ["split_invalid_obround_cylinders_as_fillet"])
    if chamfer_indices:
        remap_faces_to_macro(faces, chamfer_indices, "chamfer", graph_data, reasons + ["split_invalid_obround_planes_as_chamfer"])

    return {
        "from_macro_feature": "obround_hole",
        "to_macro_feature": "mixed_stock_transition",
        "face_indices": sorted(face_indices),
        "face_count": len(face_indices),
        "source_reasons": list(reasons),
        "repair_evidence": ["invalid_obround_split_by_primitive_surface"],
        "action": "split_to_stock_and_transition",
    }, demoted


def demote_faces_to_stock(faces: list[dict], face_indices: list[int], reasons: list[str]) -> int:
    face_set = set(int(idx) for idx in face_indices)
    demoted = 0
    for face in faces:
        if int(face["idx"]) not in face_set:
            continue
        if "raw_macro_feature_id" not in face:
            face["raw_macro_feature_id"] = face.get("macro_feature_id")
            face["raw_macro_feature"] = face.get("macro_feature")
            face["raw_semantic_subtype_id"] = face.get("semantic_subtype_id")
            face["raw_semantic_subtype"] = face.get("semantic_subtype")
        face["macro_feature_id"] = 0
        face["macro_feature"] = "stock"
        face["semantic_subtype_id"] = 0
        face["semantic_subtype"] = "stock_face"
        face["topology_postprocess_action"] = "demoted_to_stock"
        face["topology_invalid_reasons"] = list(reasons)
        demoted += 1
    return demoted


def propose_component_repair(macro_feature: str, face_indices: list[int], graph_data: dict) -> dict | None:
    if macro_feature in {"countersink_hole", "stepped_hole_round"}:
        if _single_small_partial_cylinder_transition(face_indices, graph_data):
            return {
                "target_macro_feature": "fillet",
                "evidence": ["single_small_partial_cylinder_with_line_boundaries"],
            }
        through_validation = validate_feature_topology("through_hole_round", face_indices, graph_data)
        if through_validation["valid"] and _through_hole_has_functional_opening_role(face_indices, graph_data):
            return {
                "target_macro_feature": "through_hole_round",
                "evidence": through_validation["evidence"],
            }

    if macro_feature == "slot_rect":
        v_slot_validation = validate_feature_topology("slot_v", face_indices, graph_data)
        if v_slot_validation["valid"]:
            return {
                "target_macro_feature": "slot_v",
                "evidence": v_slot_validation["evidence"],
            }

    return None


def _single_small_partial_cylinder_transition(face_indices: list[int], graph_data: dict) -> bool:
    if len(face_indices) != 1:
        return False
    face_idx = int(face_indices[0])
    nodes = _nodes_by_idx(graph_data)
    node = nodes.get(face_idx, {})
    if node.get("primitive_surface") != "cylinder":
        return False
    radius = _node_radius(node)
    area = _node_area(node)
    if radius <= 1e-9 or area <= 1e-9:
        return False
    uv_u_span = _node_descriptor(node, "uv_u_span")
    if uv_u_span > 1e-9 and uv_u_span >= 2.8:
        return False
    line_edges = []
    circular_plane_neighbors = set()
    for src, dst, attr in _edge_records(graph_data):
        if face_idx not in {src, dst}:
            continue
        neighbor = dst if src == face_idx else src
        if _is_line_edge(attr):
            line_edges.append(attr)
        if _is_circle_edge(attr) and nodes.get(neighbor, {}).get("primitive_surface") == "plane":
            circular_plane_neighbors.add(neighbor)
    if len(line_edges) < 2:
        return False
    if len(circular_plane_neighbors) < 1:
        return False
    if _through_hole_has_functional_opening_role([face_idx], graph_data) and _long_line_edge_count(face_idx, graph_data) == 0:
        return False
    return True


def repair_split_slot_v_components(faces: list[dict], graph_data: dict) -> tuple[list[dict], set[int]]:
    slot_rect_faces = {
        int(face["idx"])
        for face in faces
        if face.get("macro_feature") == "slot_rect" and int(face.get("macro_feature_id", 0)) > 0
    }
    if len(slot_rect_faces) < 2:
        return [], set()

    adjacency = {idx: set() for idx in slot_rect_faces}
    for edge in graph_data.get("edges", []):
        if not isinstance(edge, list) or len(edge) != 2:
            continue
        src, dst = int(edge[0]), int(edge[1])
        if src not in slot_rect_faces or dst not in slot_rect_faces:
            continue
        adjacency[src].add(dst)
        adjacency[dst].add(src)

    records = []
    repaired_faces = set()
    visited = set()
    for start_idx in sorted(slot_rect_faces):
        if start_idx in visited:
            continue
        queue = deque([start_idx])
        visited.add(start_idx)
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)

        component = sorted(component)
        if len(component) < 2:
            continue
        v_validation = validate_feature_topology("slot_v", component, graph_data)
        if not v_validation["valid"]:
            continue
        remap_faces_to_macro(
            faces,
            component,
            "slot_v",
            graph_data,
            ["split_slot_rect_component_satisfies_slot_v_topology"],
        )
        records.append({
            "from_macro_feature": "slot_rect",
            "to_macro_feature": "slot_v",
            "face_indices": component,
            "face_count": len(component),
            "source_reasons": ["split_slot_rect_component_satisfies_slot_v_topology"],
            "repair_evidence": v_validation["evidence"],
            "action": "remapped_split_component",
        })
        repaired_faces.update(component)

    return records, repaired_faces


def remap_faces_to_macro(
    faces: list[dict],
    face_indices: list[int],
    target_macro_feature: str,
    graph_data: dict,
    reasons: list[str],
) -> int:
    face_set = set(int(idx) for idx in face_indices)
    graph_nodes = {int(node["idx"]): node for node in graph_data.get("nodes", [])}
    target_macro_id = resolve_macro_feature_id(faces, target_macro_feature)
    remapped = 0
    for face in faces:
        face_idx = int(face["idx"])
        if face_idx not in face_set:
            continue
        preserve_raw_prediction(face)
        face["macro_feature_id"] = target_macro_id
        face["macro_feature"] = target_macro_feature
        semantic = semantic_for_macro_and_node(target_macro_feature, graph_nodes.get(face_idx, {}))
        if semantic is not None:
            semantic_key, semantic_id = semantic
            face["semantic_subtype_id"] = semantic_id
            face["semantic_subtype"] = semantic_key
        face["topology_postprocess_action"] = f"remapped_to_{target_macro_feature}"
        face["topology_invalid_reasons"] = list(reasons)
        remapped += 1
    return remapped


def repair_valid_component_semantics(
    faces: list[dict],
    face_indices: list[int],
    macro_feature: str,
    graph_data: dict,
) -> int:
    if macro_feature == "counterbore_hole":
        return repair_counterbore_semantics(faces, face_indices, graph_data)
    if macro_feature in {"rib", "web"}:
        return repair_rib_semantics(faces, face_indices, graph_data)
    if macro_feature not in {"through_hole_round", "slot_v", "countersink_hole"}:
        return 0
    face_set = set(int(idx) for idx in face_indices)
    graph_nodes = {int(node["idx"]): node for node in graph_data.get("nodes", [])}
    repaired = 0
    for face in faces:
        face_idx = int(face["idx"])
        if face_idx not in face_set:
            continue
        semantic = semantic_for_macro_and_node(macro_feature, graph_nodes.get(face_idx, {}))
        if semantic is None:
            continue
        semantic_key, semantic_id = semantic
        if face.get("semantic_subtype") == semantic_key and int(face.get("semantic_subtype_id", semantic_id)) == semantic_id:
            continue
        preserve_raw_prediction(face)
        face["semantic_subtype_id"] = semantic_id
        face["semantic_subtype"] = semantic_key
        face["topology_semantic_repair_action"] = f"semantic_repaired_for_{macro_feature}"
        repaired += 1
    return repaired


def preserve_raw_prediction(face: dict) -> None:
    if "raw_macro_feature_id" not in face:
        face["raw_macro_feature_id"] = face.get("macro_feature_id")
        face["raw_macro_feature"] = face.get("macro_feature")
        face["raw_semantic_subtype_id"] = face.get("semantic_subtype_id")
        face["raw_semantic_subtype"] = face.get("semantic_subtype")


def resolve_macro_feature_id(faces: list[dict], target_macro_feature: str) -> int:
    for face in faces:
        if face.get("macro_feature") == target_macro_feature:
            return int(face.get("macro_feature_id", DEFAULT_MACRO_FEATURE_IDS[target_macro_feature]))
    return DEFAULT_MACRO_FEATURE_IDS[target_macro_feature]


def semantic_for_macro_and_node(target_macro_feature: str, node: dict) -> tuple[str, int] | None:
    primitive = node.get("primitive_surface")
    if target_macro_feature == "through_hole_round" and primitive == "cylinder":
        return "through_hole_cylinder", DEFAULT_SEMANTIC_SUBTYPE_IDS["through_hole_cylinder"]
    if target_macro_feature == "blind_hole_round":
        if primitive == "cylinder":
            return "blind_hole_cylinder", DEFAULT_SEMANTIC_SUBTYPE_IDS["blind_hole_cylinder"]
        if primitive == "plane":
            return "blind_hole_bottom", DEFAULT_SEMANTIC_SUBTYPE_IDS["blind_hole_bottom"]
    if target_macro_feature == "countersink_hole":
        if primitive == "cone":
            return "countersink_cone", DEFAULT_SEMANTIC_SUBTYPE_IDS["countersink_cone"]
        if primitive == "cylinder":
            return "countersink_small_cylinder", DEFAULT_SEMANTIC_SUBTYPE_IDS["countersink_small_cylinder"]
    if target_macro_feature == "counterbore_hole":
        if primitive == "plane":
            return "counterbore_step_plane", DEFAULT_SEMANTIC_SUBTYPE_IDS["counterbore_step_plane"]
        if primitive == "cylinder":
            return "counterbore_small_cylinder", DEFAULT_SEMANTIC_SUBTYPE_IDS["counterbore_small_cylinder"]
    if target_macro_feature == "pocket_circular":
        if primitive == "cylinder":
            return "circular_pocket_side_cylinder", DEFAULT_SEMANTIC_SUBTYPE_IDS["circular_pocket_side_cylinder"]
        if primitive == "plane":
            return "circular_pocket_bottom_plane", DEFAULT_SEMANTIC_SUBTYPE_IDS["circular_pocket_bottom_plane"]
    if target_macro_feature == "cylinder_boss":
        if primitive == "cylinder":
            return "cylinder_boss_side", DEFAULT_SEMANTIC_SUBTYPE_IDS["cylinder_boss_side"]
        if primitive == "plane":
            return "cylinder_boss_top", DEFAULT_SEMANTIC_SUBTYPE_IDS["cylinder_boss_top"]
    if target_macro_feature == "slot_rect" and primitive == "plane":
        return "slot_rect_side_plane", DEFAULT_SEMANTIC_SUBTYPE_IDS["slot_rect_side_plane"]
    if target_macro_feature == "slot_u":
        if primitive == "plane":
            return "slot_u_side_plane", DEFAULT_SEMANTIC_SUBTYPE_IDS["slot_u_side_plane"]
        if primitive in {"cylinder", "surface_of_revolution", "surface_of_extrusion"}:
            return "slot_u_bottom_plane", DEFAULT_SEMANTIC_SUBTYPE_IDS["slot_u_bottom_plane"]
    if target_macro_feature == "slot_v" and primitive == "plane":
        return "slot_v_side_plane", DEFAULT_SEMANTIC_SUBTYPE_IDS["slot_v_side_plane"]
    if target_macro_feature == "slot_o" and primitive in {"cylinder", "surface_of_revolution", "surface_of_extrusion", "bspline_surface"}:
        return "slot_o_bottom_plane", DEFAULT_SEMANTIC_SUBTYPE_IDS["slot_o_bottom_plane"]
    if target_macro_feature == "rib" and primitive == "plane":
        return "rib_side_plane", DEFAULT_SEMANTIC_SUBTYPE_IDS["rib_side_plane"]
    if target_macro_feature == "fillet" and primitive in {"cylinder", "toroid", "surface_of_revolution", "surface_of_extrusion", "bspline_surface"}:
        return "fillet_blend_surface", DEFAULT_SEMANTIC_SUBTYPE_IDS["fillet_blend_surface"]
    if target_macro_feature == "chamfer" and primitive == "plane":
        return "chamfer_bevel_plane", DEFAULT_SEMANTIC_SUBTYPE_IDS["chamfer_bevel_plane"]
    return None


def repair_counterbore_semantics(faces: list[dict], face_indices: list[int], graph_data: dict) -> int:
    face_set = set(int(idx) for idx in face_indices)
    nodes = _nodes_by_idx(graph_data)
    cylinders = [
        idx
        for idx in face_set
        if nodes.get(idx, {}).get("primitive_surface") == "cylinder"
    ]
    planes = [
        idx
        for idx in face_set
        if nodes.get(idx, {}).get("primitive_surface") == "plane"
    ]
    if len(cylinders) < 2 or not planes:
        return 0
    cylinders.sort(key=lambda idx: _node_radius(nodes.get(idx, {})), reverse=True)
    large_idx = cylinders[0]
    small_indices = set(cylinders[1:])
    semantics = {large_idx: "counterbore_large_cylinder"}
    for idx in small_indices:
        semantics[idx] = "counterbore_small_cylinder"
    for idx in planes:
        semantics[idx] = "counterbore_step_plane"

    repaired = 0
    for face in faces:
        face_idx = int(face["idx"])
        semantic_key = semantics.get(face_idx)
        if semantic_key is None:
            continue
        semantic_id = DEFAULT_SEMANTIC_SUBTYPE_IDS[semantic_key]
        if face.get("semantic_subtype") == semantic_key and int(face.get("semantic_subtype_id", semantic_id)) == semantic_id:
            continue
        preserve_raw_prediction(face)
        face["semantic_subtype"] = semantic_key
        face["semantic_subtype_id"] = semantic_id
        face["topology_semantic_repair_action"] = "semantic_repaired_for_counterbore_hole"
        repaired += 1
    return repaired


def repair_rib_semantics(faces: list[dict], face_indices: list[int], graph_data: dict) -> int:
    top_idx = _rib_top_face(face_indices, graph_data)
    if top_idx is None:
        return 0
    face_set = {int(idx) for idx in face_indices}
    repaired = 0
    for face in faces:
        face_idx = int(face["idx"])
        if face_idx not in face_set:
            continue
        semantic_key = "rib_top_plane" if face_idx == top_idx else "rib_side_plane"
        semantic_id = DEFAULT_SEMANTIC_SUBTYPE_IDS[semantic_key]
        if face.get("semantic_subtype") == semantic_key and int(face.get("semantic_subtype_id", semantic_id)) == semantic_id:
            continue
        preserve_raw_prediction(face)
        face["semantic_subtype"] = semantic_key
        face["semantic_subtype_id"] = semantic_id
        face["topology_semantic_repair_action"] = "semantic_repaired_for_rib"
        repaired += 1
    return repaired


def _refine_overmerged_rectangular_slot_component(face_indices: list[int], graph_data: dict) -> list[int] | None:
    face_indices = sorted(int(idx) for idx in face_indices)
    if len(face_indices) <= 3:
        return None

    candidates = []
    for candidate in combinations(face_indices, 3):
        candidate = list(candidate)
        if not validate_feature_topology("slot_rect", candidate, graph_data)["valid"]:
            continue
        if not _rectangular_slot_candidate_has_consistent_long_faces(candidate, graph_data):
            continue
        total_area = sum(_node_area(_nodes_by_idx(graph_data).get(idx, {})) for idx in candidate)
        candidates.append((total_area, candidate))
    if not candidates:
        return None
    return sorted(min(candidates, key=lambda item: item[0])[1])


def _rectangular_slot_candidate_has_consistent_long_faces(face_indices: list[int], graph_data: dict) -> bool:
    nodes = _nodes_by_idx(graph_data)
    long_extents = [
        _node_descriptor(nodes.get(int(idx), {}), "projected_long_extent")
        for idx in face_indices
    ]
    positive_extents = [extent for extent in long_extents if extent > 1e-9]
    if len(positive_extents) != len(face_indices):
        return True
    if min(positive_extents) / max(positive_extents) < 0.7:
        return False
    aspects = [
        _node_descriptor(nodes.get(int(idx), {}), "projected_aspect_ratio")
        for idx in face_indices
    ]
    positive_aspects = [aspect for aspect in aspects if aspect > 1e-9]
    if len(positive_aspects) != len(face_indices):
        return True
    return all(aspect >= 5.0 for aspect in positive_aspects)


def repair_rectangular_slot_semantics(faces: list[dict], face_indices: list[int], graph_data: dict) -> int:
    bottom_idx = _central_rectangular_face(face_indices, graph_data, "concave")
    if bottom_idx is None:
        return 0
    face_set = {int(idx) for idx in face_indices}
    repaired = 0
    for face in faces:
        face_idx = int(face["idx"])
        if face_idx not in face_set:
            continue
        if face_idx == bottom_idx:
            semantic_key = "slot_rect_bottom_plane"
        else:
            semantic_key = "slot_rect_side_plane"
        semantic_id = DEFAULT_SEMANTIC_SUBTYPE_IDS[semantic_key]
        if face.get("semantic_subtype") == semantic_key and int(face.get("semantic_subtype_id", semantic_id)) == semantic_id:
            continue
        preserve_raw_prediction(face)
        face["semantic_subtype"] = semantic_key
        face["semantic_subtype_id"] = semantic_id
        face["topology_semantic_repair_action"] = "semantic_repaired_for_slot_rect"
        repaired += 1
    return repaired


def _chamfer_shape_invalid_reason(face_indices: list[int], graph_data: dict) -> str | None:
    if len(face_indices) != 1:
        return None
    face_idx = int(face_indices[0])
    node = _nodes_by_idx(graph_data).get(face_idx, {})
    if node.get("primitive_surface") != "plane":
        return None
    neighbor_areas = _neighbor_areas(face_idx, graph_data)
    if not neighbor_areas:
        return None
    if not _has_chamfer_angle_edges(face_idx, graph_data):
        return "chamfer_requires_bevel_angle_edges"
    if max(neighbor_areas) < _node_area(node) * 5.0:
        return "chamfer_requires_large_adjacent_support_face"
    return None


def _rectangular_prism_candidate_supported(target_macro: str, face_indices: list[int], graph_data: dict) -> bool:
    if target_macro != "pocket_rect":
        return True
    return _rectangular_pocket_shape_invalid_reason(face_indices, graph_data) is None


def _rectangular_pocket_shape_invalid_reason(face_indices: list[int], graph_data: dict) -> str | None:
    bottom_idx = _central_rectangular_face(face_indices, graph_data, "concave")
    if bottom_idx is None:
        return None
    bottom = _nodes_by_idx(graph_data).get(bottom_idx, {})
    aspect_ratio = _node_descriptor(bottom, "projected_aspect_ratio")
    if aspect_ratio > MAX_RECTANGULAR_POCKET_BOTTOM_ASPECT_RATIO:
        return "rectangular_pocket_bottom_aspect_ratio_too_high"
    return None




def _rib_component_shape_invalid_reason(face_indices: list[int], graph_data: dict) -> str | None:
    face_set = sorted({int(idx) for idx in face_indices})
    if len(face_set) != 3:
        return None
    nodes = _nodes_by_idx(graph_data)
    plane_indices = [idx for idx in face_set if nodes.get(idx, {}).get("primitive_surface") == "plane"]
    if len(plane_indices) != 3:
        return None

    total_area = sum(_node_area(node) for node in nodes.values())
    candidate_area = sum(_node_area(nodes.get(idx, {})) for idx in face_set)
    if total_area <= 1e-9 or candidate_area / total_area < 0.30:
        return None

    for left_idx, right_idx in combinations(plane_indices, 2):
        left = nodes.get(left_idx, {})
        right = nodes.get(right_idx, {})
        if _angle_between_abs(_node_normal(left), _node_normal(right)) > 0.18:
            continue
        thickness = _distance_between_nodes(left, right)
        if thickness <= 1e-9:
            continue
        long_extent = max(_node_descriptor(left, "projected_long_extent"), _node_descriptor(right, "projected_long_extent"))
        short_extent = max(_node_descriptor(left, "projected_short_extent"), _node_descriptor(right, "projected_short_extent"))
        reference_extent = max(min(long_extent, short_extent), 1e-9)
        if thickness / reference_extent > 0.45 or long_extent / max(thickness, 1e-9) < 6.0:
            continue
        left_area_fraction = _node_area(left) / total_area
        right_area_fraction = _node_area(right) / total_area
        if min(left_area_fraction, right_area_fraction) < 0.08:
            continue
        left_aspect = _node_descriptor(left, "projected_aspect_ratio")
        right_aspect = _node_descriptor(right, "projected_aspect_ratio")
        if max(left_aspect, right_aspect) > 3.2:
            continue
        left_short = _node_descriptor(left, "projected_short_extent")
        right_short = _node_descriptor(right, "projected_short_extent")
        if min(left_short, right_short) < 20.0:
            continue
        return "rib_component_contains_stock_scale_plate_faces"
    return None
def repair_rectangular_prism_semantics(
    faces: list[dict],
    face_indices: list[int],
    target_macro_feature: str,
    graph_data: dict,
    required_convexity: str,
) -> int:
    if target_macro_feature == "pocket_rect":
        central_semantic = "rect_pocket_bottom_plane"
        side_semantic = "rect_pocket_side_plane"
    elif target_macro_feature == "rectangular_boss":
        central_semantic = "pad_top_plane"
        side_semantic = "pad_side_plane"
    else:
        return 0

    central_idx = _central_rectangular_face(face_indices, graph_data, required_convexity)
    if central_idx is None:
        return 0

    central_id = DEFAULT_SEMANTIC_SUBTYPE_IDS[central_semantic]
    side_id = DEFAULT_SEMANTIC_SUBTYPE_IDS[side_semantic]
    face_set = {int(idx) for idx in face_indices}
    repaired = 0
    for face in faces:
        face_idx = int(face["idx"])
        if face_idx not in face_set:
            continue
        semantic_key, semantic_id = (central_semantic, central_id) if face_idx == central_idx else (side_semantic, side_id)
        if face.get("semantic_subtype") == semantic_key and int(face.get("semantic_subtype_id", semantic_id)) == semantic_id:
            continue
        preserve_raw_prediction(face)
        face["semantic_subtype"] = semantic_key
        face["semantic_subtype_id"] = semantic_id
        face["topology_semantic_repair_action"] = f"semantic_repaired_for_{target_macro_feature}"
        repaired += 1
    return repaired


def _repair_record(from_macro: str, to_macro: str, face_indices: list[int], evidence: list[str]) -> dict:
    return {
        "from_macro_feature": from_macro,
        "to_macro_feature": to_macro,
        "face_indices": sorted(int(idx) for idx in face_indices),
        "face_count": len(face_indices),
        "source_reasons": ["candidate_level_topology_repair"],
        "repair_evidence": list(evidence),
        "action": "candidate_remapped",
    }


def _nodes_by_idx(graph_data: dict) -> dict[int, dict]:
    return {int(node["idx"]): node for node in graph_data.get("nodes", [])}


def _edge_records(graph_data: dict) -> list[tuple[int, int, dict]]:
    records = []
    edge_attrs = graph_data.get("edge_attributes", [])
    for index, edge in enumerate(graph_data.get("edges", [])):
        if not isinstance(edge, list) or len(edge) != 2:
            continue
        src, dst = int(edge[0]), int(edge[1])
        attr = dict(edge_attrs[index]) if index < len(edge_attrs) else {}
        attr.setdefault("source_face", src)
        attr.setdefault("target_face", dst)
        records.append((src, dst, attr))
    return records


def _face_has_macro_evidence(face: dict, macro_feature: str) -> bool:
    return face.get("macro_feature") == macro_feature or face.get("raw_macro_feature") == macro_feature


def _face_allowed_for_chamfer_candidate(face: dict) -> bool:
    if _face_allowed_for_stock_candidate(face, {"stock", "chamfer"}):
        return True
    return (
        face.get("macro_feature") == "pocket_rect"
        or face.get("raw_macro_feature") == "pocket_rect"
        or face.get("semantic_subtype") in {"rect_pocket_side_plane", "rect_pocket_bottom_plane"}
        or face.get("raw_semantic_subtype") in {"rect_pocket_side_plane", "rect_pocket_bottom_plane"}
    )

def _face_allowed_for_obround_candidate(face: dict) -> bool:
    macros = {
        "obround_hole",
        "stepped_hole_round",
        "slot_u",
        "slot_rect",
        "stock",
    }
    semantics = {
        "obround_hole_side_plane",
        "obround_hole_end_cylinder",
        "stepped_hole_large_hole_cylinder",
        "stepped_hole_small_hole_cylinder",
        "stepped_hole_step_plane",
        "stepped_hole_bottom",
        "stepped_hole_small_hole_bottom",
        "fillet_blend_surface",
        "stock_face",
    }
    return (
        face.get("macro_feature", "stock") in macros
        or face.get("raw_macro_feature") in macros
        or face.get("semantic_subtype") in semantics
        or face.get("raw_semantic_subtype") in semantics
    )


def _face_allowed_for_through_hole_opening_candidate(face: dict) -> bool:
    hole_recovery_macros = {
        "counterbore_hole",
        "countersink_hole",
        "stepped_hole_round",
        "through_hole_round",
    }
    hole_recovery_semantics = {
        "counterbore_large_cylinder",
        "counterbore_small_cylinder",
        "countersink_small_cylinder",
        "stepped_hole_large_hole_cylinder",
        "stepped_hole_small_hole_cylinder",
        "through_hole_cylinder",
    }
    return (
        face.get("macro_feature") == "stock"
        or face.get("raw_macro_feature") == "stock"
        or face.get("semantic_subtype") == "stock_face"
        or face.get("raw_semantic_subtype") == "stock_face"
        or face.get("macro_feature") == "cylinder_boss"
        or face.get("raw_macro_feature") == "cylinder_boss"
        or face.get("semantic_subtype") == "cylinder_boss_side"
        or face.get("raw_semantic_subtype") == "cylinder_boss_side"
        or face.get("macro_feature") in hole_recovery_macros
        or face.get("raw_macro_feature") in hole_recovery_macros
        or face.get("semantic_subtype") in hole_recovery_semantics
        or face.get("raw_semantic_subtype") in hole_recovery_semantics
    )
def _face_allowed_for_mixed_rib_candidate(face: dict) -> bool:
    macros = {
        "stock",
        "rib",
        "web",
        "pocket_rect",
        "rectangular_boss",
    }
    semantics = {
        "stock_face",
        "rib_side_plane",
        "rib_top_plane",
        "rect_pocket_bottom_plane",
        "rect_pocket_side_plane",
        "pad_top_plane",
        "pad_side_plane",
    }
    return (
        face.get("macro_feature", "stock") in macros
        or face.get("raw_macro_feature") in macros
        or face.get("semantic_subtype") in semantics
        or face.get("raw_semantic_subtype") in semantics
    )


def _mixed_rib_candidate_has_non_rib_support(candidate: list[int], face_by_idx: dict[int, dict]) -> bool:
    macros = {
        face_by_idx.get(idx, {}).get("macro_feature", "stock")
        for idx in candidate
    }
    raw_macros = {
        face_by_idx.get(idx, {}).get("raw_macro_feature")
        for idx in candidate
    }
    return bool((macros | raw_macros) & {"pocket_rect", "rectangular_boss", "stock", "web"})

def _face_allowed_for_rectangular_completion(face: dict, target_macro: str) -> bool:
    macro = face.get("macro_feature")
    raw_macro = face.get("raw_macro_feature")
    if target_macro == "rectangular_boss":
        return macro in {"stock", target_macro, "web"} or raw_macro in {target_macro, "web"}
    return macro in {"stock", target_macro} or raw_macro == target_macro


def _face_allowed_for_stock_candidate(face: dict, allowed_macros: set[str]) -> bool:
    macro = face.get("macro_feature", "stock")
    raw_macro = face.get("raw_macro_feature")
    return macro in allowed_macros or raw_macro in allowed_macros


def _face_allowed_for_counterbore_part(face: dict) -> bool:
    macros = {
        "stock",
        "counterbore_hole",
        "stepped_hole_round",
        "through_hole_round",
    }
    semantics = {
        "stock_face",
        "counterbore_large_cylinder",
        "counterbore_step_plane",
        "counterbore_small_cylinder",
        "stepped_hole_large_hole_cylinder",
        "stepped_hole_step_plane",
        "stepped_hole_small_hole_cylinder",
        "through_hole_cylinder",
    }
    return (
        face.get("macro_feature", "stock") in macros
        or face.get("raw_macro_feature") in macros
        or face.get("semantic_subtype") in semantics
        or face.get("raw_semantic_subtype") in semantics
    )


def _face_has_counterbore_seed(face: dict) -> bool:
    return (
        face.get("macro_feature") == "counterbore_hole"
        or face.get("raw_macro_feature") == "counterbore_hole"
        or face.get("semantic_subtype") in {
            "counterbore_large_cylinder",
            "counterbore_step_plane",
            "counterbore_small_cylinder",
        }
        or face.get("raw_semantic_subtype") in {
            "counterbore_large_cylinder",
            "counterbore_step_plane",
            "counterbore_small_cylinder",
        }
    )


def _face_has_stepped_counterbore_shoulder_seed(face: dict) -> bool:
    return (
        face.get("macro_feature") == "stepped_hole_round"
        or face.get("raw_macro_feature") == "stepped_hole_round"
        or face.get("semantic_subtype") == "stepped_hole_step_plane"
        or face.get("raw_semantic_subtype") == "stepped_hole_step_plane"
    )


def _counterbore_shoulder_plane_is_local(plane_idx: int, large_idx: int, small_idx: int, graph_data: dict) -> bool:
    nodes = _nodes_by_idx(graph_data)
    plane_area = _node_area(nodes.get(plane_idx, {}))
    cylinder_area = max(_node_area(nodes.get(large_idx, {})), _node_area(nodes.get(small_idx, {})))
    if plane_area <= 1e-9 or cylinder_area <= 1e-9:
        return False
    return plane_area <= cylinder_area * 2.5


def _face_allowed_for_countersink_cone(
    face: dict,
    cone_idx: int | None = None,
    cylinder_idx: int | None = None,
    graph_data: dict | None = None,
) -> bool:
    if (
        face.get("macro_feature") == "countersink_hole"
        or face.get("raw_macro_feature") == "countersink_hole"
        or face.get("semantic_subtype") == "countersink_cone"
        or face.get("raw_semantic_subtype") == "countersink_cone"
    ):
        return True
    if face.get("macro_feature") != "chamfer" and face.get("raw_macro_feature") != "chamfer":
        return False
    if cone_idx is None or cylinder_idx is None or graph_data is None:
        return False
    return _chamfer_cone_has_countersink_context(cone_idx, cylinder_idx, graph_data)


def _chamfer_cone_has_countersink_context(cone_idx: int, cylinder_idx: int, graph_data: dict) -> bool:
    nodes = _nodes_by_idx(graph_data)
    cone = nodes.get(cone_idx, {})
    cylinder = nodes.get(cylinder_idx, {})
    if cone.get("primitive_surface") != "cone" or cylinder.get("primitive_surface") != "cylinder":
        return False
    if _angle_between_abs(_node_axis(cone), _node_axis(cylinder)) > 0.18:
        return False
    cone_area = _node_area(cone)
    cylinder_area = _node_area(cylinder)
    if cone_area <= 1e-9 or cylinder_area <= 1e-9:
        return False
    if cone_area / max(cylinder_area, 1e-9) > 3.0:
        return False
    return len(_unique_circular_neighbors(cylinder_idx, graph_data)) >= 2


def _unique_circular_neighbors(face_idx: int, graph_data: dict) -> set[int]:
    neighbors = set()
    for src, dst, attr in _edge_records(graph_data):
        if face_idx not in {src, dst} or not _is_circle_edge(attr):
            continue
        neighbors.add(dst if src == face_idx else src)
    return neighbors


def _face_allowed_for_countersink_cylinder(face: dict) -> bool:
    return (
        face.get("macro_feature") in {"through_hole_round", "countersink_hole", "stock"}
        or face.get("raw_macro_feature") in {"through_hole_round", "countersink_hole", "stock"}
        or face.get("semantic_subtype") in {"through_hole_cylinder", "countersink_small_cylinder", "stock_face"}
        or face.get("raw_semantic_subtype") in {"through_hole_cylinder", "countersink_small_cylinder", "stock_face"}
    )


def _countersink_cylinder_has_exit_opening(cylinder_idx: int, cone_idx: int, graph_data: dict) -> bool:
    nodes = _nodes_by_idx(graph_data)
    non_cone_neighbors = [
        dst if src == cylinder_idx else src
        for src, dst, attr in _edge_records(graph_data)
        if cylinder_idx in {src, dst} and (dst if src == cylinder_idx else src) != cone_idx and _is_circle_edge(attr)
    ]
    unique_neighbors = sorted(set(non_cone_neighbors))
    if not unique_neighbors:
        return True
    return any(nodes.get(neighbor, {}).get("primitive_surface") == "plane" for neighbor in unique_neighbors)


def _through_hole_has_functional_opening_role(face_indices: list[int], graph_data: dict) -> bool:
    if len(face_indices) != 1:
        return False
    face_idx = int(face_indices[0])
    nodes = _nodes_by_idx(graph_data)
    node = nodes.get(face_idx, {})
    if node.get("primitive_surface") != "cylinder":
        return False
    if _looks_like_exterior_cylinder_not_slot_bottom(face_idx, node, graph_data):
        return False
    uv_u_span = _node_descriptor(node, "uv_u_span")
    if uv_u_span > 1e-9 and uv_u_span < 5.5:
        return False
    circular_neighbors = set()
    for src, dst, attr in _edge_records(graph_data):
        if face_idx not in {src, dst} or not _is_circle_edge(attr):
            continue
        circular_neighbors.add(dst if src == face_idx else src)
    if len(circular_neighbors) != 2:
        return False
    if not all(nodes.get(neighbor, {}).get("primitive_surface") == "plane" for neighbor in circular_neighbors):
        return False

    loop_summary = _circular_plane_loop_role_summary(face_idx, graph_data)
    if loop_summary["known_neighbor_count"] > 0:
        if loop_summary["outer_neighbor_count"] > 0:
            return False
        return loop_summary["inner_neighbor_count"] >= 2
    return True


def _circular_plane_loop_role_summary(face_idx: int, graph_data: dict) -> dict[str, int]:
    nodes = _nodes_by_idx(graph_data)
    roles_by_neighbor: dict[int, set[str]] = {}
    for src, dst, attr in _edge_records(graph_data):
        if face_idx not in {src, dst} or not _is_circle_edge(attr):
            continue
        neighbor = dst if src == face_idx else src
        if nodes.get(neighbor, {}).get("primitive_surface") != "plane":
            continue
        role = _edge_loop_role(attr)
        if role == "unknown":
            continue
        roles_by_neighbor.setdefault(neighbor, set()).add(role)

    inner_neighbors = {neighbor for neighbor, roles in roles_by_neighbor.items() if "inner_loop" in roles and "outer_loop" not in roles}
    outer_neighbors = {neighbor for neighbor, roles in roles_by_neighbor.items() if "outer_loop" in roles}
    return {
        "known_neighbor_count": len(roles_by_neighbor),
        "inner_neighbor_count": len(inner_neighbors),
        "outer_neighbor_count": len(outer_neighbors),
    }


def _edge_loop_role(edge_attr: dict) -> str:
    role = edge_attr.get("adjacent_plane_loop_role")
    if role in {"inner_loop", "outer_loop"}:
        return role
    role_id = int(edge_attr.get("adjacent_plane_loop_role_id", 0) or 0)
    if role_id == 1 or float(edge_attr.get("adjacent_plane_loop_role_inner", 0.0) or 0.0) > 0.5:
        return "inner_loop"
    if role_id == 2 or float(edge_attr.get("adjacent_plane_loop_role_outer", 0.0) or 0.0) > 0.5:
        return "outer_loop"
    return "unknown"


def _face_allowed_for_low_confidence_fillet_repair(face: dict) -> bool:
    macro = face.get("macro_feature")
    semantic = face.get("semantic_subtype")
    if macro != "through_hole_round" and semantic != "through_hole_cylinder":
        return False
    return float(face.get("macro_confidence", 1.0) or 1.0) < 0.7


def _belongs_to_multi_cylinder_obround_prediction(face_idx: int, face_by_idx: dict[int, dict], graph_data: dict) -> bool:
    face = face_by_idx.get(face_idx, {})
    if face.get("macro_feature") != "obround_hole" and face.get("raw_macro_feature") != "obround_hole":
        return False
    nodes = _nodes_by_idx(graph_data)
    allowed = {
        idx
        for idx, candidate_face in face_by_idx.items()
        if candidate_face.get("macro_feature") == "obround_hole"
        or candidate_face.get("raw_macro_feature") == "obround_hole"
    }
    if face_idx not in allowed:
        return False
    adjacency = {idx: set() for idx in allowed}
    for src, dst, _attr in _edge_records(graph_data):
        if src in allowed and dst in allowed:
            adjacency.setdefault(src, set()).add(dst)
            adjacency.setdefault(dst, set()).add(src)
    visited = {face_idx}
    queue = deque([face_idx])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency.get(current, set()):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    if len(visited) < 4:
        return False
    cylinder_count = sum(1 for idx in visited if nodes.get(idx, {}).get("primitive_surface") == "cylinder")
    plane_count = sum(1 for idx in visited if nodes.get(idx, {}).get("primitive_surface") == "plane")
    return cylinder_count >= 2 and plane_count >= 2


def _rectangular_boss_has_large_support(face_indices: list[int], graph_data: dict) -> bool:
    nodes = _nodes_by_idx(graph_data)
    face_set = {int(idx) for idx in face_indices}
    candidate_area = sum(_node_area(nodes.get(idx, {})) for idx in face_set)
    if candidate_area <= 1e-9:
        return False
    for src, dst, attr in _edge_records(graph_data):
        if not _is_line_edge(attr):
            continue
        if (src in face_set) == (dst in face_set):
            continue
        neighbor_idx = dst if src in face_set else src
        neighbor = nodes.get(neighbor_idx, {})
        if neighbor.get("primitive_surface") != "plane":
            continue
        if _node_area(neighbor) >= candidate_area * 3.0:
            return True
    return False


def _has_u_slot_like_concave_side_planes(face_idx: int, graph_data: dict, repaired_faces: set[int]) -> bool:
    nodes = _nodes_by_idx(graph_data)
    side_planes = set()
    for src, dst, attr in _edge_records(graph_data):
        if face_idx not in {src, dst} or not _is_line_edge(attr) or not _is_concave_edge(attr):
            continue
        neighbor = dst if src == face_idx else src
        if neighbor in repaired_faces:
            continue
        node = nodes.get(neighbor, {})
        if node.get("primitive_surface") != "plane":
            continue
        if _node_descriptor(node, "projected_aspect_ratio") < 5.0:
            continue
        side_planes.add(neighbor)
    return len(side_planes) >= 2


def _looks_like_transition_cylinder(face_idx: int, graph_data: dict) -> bool:
    if _has_u_slot_like_concave_side_planes(face_idx, graph_data, set()):
        return False
    node = _nodes_by_idx(graph_data).get(face_idx, {})
    if node.get("primitive_surface") == "cylinder":
        angle = _cylindrical_face_angle(face_idx, node, graph_data)
        if angle >= 2.35 and _long_line_edge_count(face_idx, graph_data) >= 2:
            return False
    line_edges = [
        attr for src, dst, attr in _edge_records(graph_data)
        if face_idx in {src, dst} and _is_line_edge(attr)
    ]
    if len(line_edges) < 2:
        return False
    return any(_edge_has_convexity(edge, "convex") for edge in line_edges)


def _rib_candidate_is_too_large_for_model_context(face_indices: list[int], graph_data: dict) -> bool:
    nodes = _nodes_by_idx(graph_data)
    face_set = {int(idx) for idx in face_indices}
    candidate_area = sum(_node_area(nodes.get(idx, {})) for idx in face_set)
    total_area = sum(_node_area(node) for node in nodes.values())
    if candidate_area <= 1e-9 or total_area <= candidate_area + 1e-9:
        return False
    return candidate_area / total_area > 0.5


def _has_chamfer_angle_edges(face_idx: int, graph_data: dict) -> bool:
    seen = set()
    for src, dst, attr in _edge_records(graph_data):
        if face_idx not in {src, dst} or not _is_line_edge(attr):
            continue
        angle = float(attr.get("dihedral_angle", 0.0) or 0.0)
        if not 0.35 <= angle <= 1.25:
            continue
        seen.add((min(src, dst), max(src, dst)))
    return len(seen) >= 2


def _plane_component_from_seed(
    seed_idx: int,
    allowed_faces: set[int],
    graph_data: dict,
    required_convexity: str,
) -> list[int]:
    if seed_idx not in allowed_faces:
        return []
    adjacency = {idx: set() for idx in allowed_faces}
    for src, dst, attr in _edge_records(graph_data):
        if src not in allowed_faces or dst not in allowed_faces:
            continue
        if not _is_line_edge(attr) or not _edge_has_convexity(attr, required_convexity):
            continue
        adjacency.setdefault(src, set()).add(dst)
        adjacency.setdefault(dst, set()).add(src)

    visited = {seed_idx}
    queue = deque([seed_idx])
    while queue:
        current = queue.popleft()
        for neighbor in sorted(adjacency.get(current, set())):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    return sorted(visited)


def _best_rectangular_subsets(
    component: list[int],
    seed_faces: set[int],
    target_macro: str,
    graph_data: dict,
    *,
    anchor_idx: int | None = None,
    required_convexity: str | None = None,
) -> list[list[int]]:
    from itertools import combinations

    if len(component) < 5:
        return []
    search_component = list(component)
    if anchor_idx is not None and required_convexity is not None and len(search_component) > 14:
        search_component = _local_rectangular_search_pool(
            anchor_idx,
            search_component,
            graph_data,
            required_convexity,
        )
        if len(search_component) < 5:
            return []
    scored = []
    local_seed_faces = set(seed_faces) & set(search_component)
    max_subset_size = min(len(search_component), 7)
    for size in range(5, max_subset_size + 1):
        for subset in combinations(search_component, size):
            if local_seed_faces and not (set(subset) & local_seed_faces):
                continue
            candidate = sorted(int(idx) for idx in subset)
            validation = validate_feature_topology(target_macro, candidate, graph_data)
            if not validation["valid"]:
                continue
            seed_count = len(set(candidate) & seed_faces)
            internal_edge_count = _desired_internal_line_edge_count(candidate, graph_data)
            total_area = sum(_node_area(node) for idx, node in _nodes_by_idx(graph_data).items() if idx in set(candidate))
            scored.append((-seed_count, -internal_edge_count, size, total_area, candidate))
    scored.sort()
    return [item[-1] for item in scored]


def _local_rectangular_search_pool(
    anchor_idx: int,
    component: list[int],
    graph_data: dict,
    required_convexity: str,
    *,
    max_depth: int = 2,
    max_faces: int = 14,
) -> list[int]:
    component_set = set(int(idx) for idx in component)
    if anchor_idx not in component_set:
        return component

    adjacency = {idx: set() for idx in component_set}
    for src, dst, attr in _edge_records(graph_data):
        if src not in component_set or dst not in component_set:
            continue
        if not _is_line_edge(attr) or not _edge_has_convexity(attr, required_convexity):
            continue
        adjacency.setdefault(src, set()).add(dst)
        adjacency.setdefault(dst, set()).add(src)

    distances = {anchor_idx: 0}
    queue = deque([anchor_idx])
    while queue:
        current = queue.popleft()
        current_depth = distances[current]
        if current_depth >= max_depth:
            continue
        for neighbor in sorted(adjacency.get(current, set())):
            if neighbor in distances:
                continue
            distances[neighbor] = current_depth + 1
            queue.append(neighbor)

    if len(distances) < 5:
        return component

    nodes = _nodes_by_idx(graph_data)
    ordered = sorted(
        distances,
        key=lambda idx: (
            distances[idx],
            -_desired_internal_line_edge_count([idx, *adjacency.get(idx, set())], graph_data),
            _node_area(nodes.get(idx, {})),
            idx,
        ),
    )
    return sorted(ordered[:max_faces])


def _desired_internal_line_edge_count(face_indices: list[int], graph_data: dict) -> int:
    face_set = set(face_indices)
    seen = set()
    for src, dst, attr in _edge_records(graph_data):
        if src not in face_set or dst not in face_set or not _is_line_edge(attr):
            continue
        seen.add((min(src, dst), max(src, dst), attr.get("convexity"), round(float(attr.get("length", 0.0) or 0.0), 6)))
    return len(seen)


def _central_rectangular_face(face_indices: list[int], graph_data: dict, required_convexity: str) -> int | None:
    face_set = set(int(idx) for idx in face_indices)
    degree = {idx: 0 for idx in face_set}
    for src, dst, attr in _edge_records(graph_data):
        if src not in face_set or dst not in face_set:
            continue
        if not _is_line_edge(attr) or not _edge_has_convexity(attr, required_convexity):
            continue
        degree[src] += 1
    if not degree:
        return None
    return max(sorted(degree), key=lambda idx: (degree[idx], -_node_area(_nodes_by_idx(graph_data).get(idx, {}))))


def _edge_has_convexity(edge: dict, required_convexity: str) -> bool:
    if required_convexity == "concave":
        return _is_concave_edge(edge)
    if required_convexity == "convex":
        return edge.get("convexity") == "convex" or int(edge.get("convexity_id", 0) or 0) == 1
    return False


def _circular_cylinder_neighbors(face_idx: int, graph_data: dict) -> set[int]:
    nodes = _nodes_by_idx(graph_data)
    neighbors = set()
    for src, dst, attr in _edge_records(graph_data):
        if face_idx not in {src, dst} or not _is_circle_edge(attr):
            continue
        neighbor_idx = dst if src == face_idx else src
        if nodes.get(neighbor_idx, {}).get("primitive_surface") == "cylinder":
            neighbors.add(neighbor_idx)
    return neighbors


def _cylinders_are_coaxial(left: dict, right: dict) -> bool:
    import math

    if left.get("primitive_surface") != "cylinder" or right.get("primitive_surface") != "cylinder":
        return False
    return _angle_between_abs(_node_axis(left), _node_axis(right)) <= math.radians(8.0)


def _small_cylinder_has_bottom_plane(small_idx: int, shoulder_idx: int, graph_data: dict) -> bool:
    nodes = _nodes_by_idx(graph_data)
    small_node = nodes.get(small_idx, {})
    radius = _node_radius(small_node)
    if radius <= 1e-9:
        return False
    approximate_disk_area = 3.141592653589793 * radius * radius
    for src, dst, attr in _edge_records(graph_data):
        if small_idx not in {src, dst} or not _is_circle_edge(attr):
            continue
        neighbor_idx = dst if src == small_idx else src
        if neighbor_idx == shoulder_idx:
            continue
        neighbor = nodes.get(neighbor_idx, {})
        if neighbor.get("primitive_surface") != "plane":
            continue
        if _node_area(neighbor) <= approximate_disk_area * 4.0:
            return True
    return False


def _face_looks_like_round_boss_or_hole_candidate(cylinder_face: dict, plane_face: dict) -> bool:
    candidate_macros = {
        "blind_hole_round",
        "counterbore_hole",
        "cylinder_boss",
        "stepped_hole_round",
        "stock",
    }
    candidate_semantics = {
        "blind_hole_cylinder",
        "blind_hole_bottom",
        "counterbore_small_cylinder",
        "counterbore_step_plane",
        "counterbore_large_cylinder",
        "cylinder_boss_side",
        "cylinder_boss_top",
        "stepped_hole_large_hole_cylinder",
        "stepped_hole_small_hole_cylinder",
        "stepped_hole_step_plane",
        "stepped_hole_bottom",
        "stock_face",
    }
    return (
        cylinder_face.get("macro_feature") in candidate_macros
        or cylinder_face.get("raw_macro_feature") in candidate_macros
        or cylinder_face.get("semantic_subtype") in candidate_semantics
        or cylinder_face.get("raw_semantic_subtype") in candidate_semantics
        or plane_face.get("macro_feature") in candidate_macros
        or plane_face.get("raw_macro_feature") in candidate_macros
        or plane_face.get("semantic_subtype") in candidate_semantics
        or plane_face.get("raw_semantic_subtype") in candidate_semantics
    )


def _cap_plane_is_exterior_to_main_slab(cylinder_idx: int, plane_idx: int, graph_data: dict) -> bool:
    nodes = _nodes_by_idx(graph_data)
    cylinder = nodes.get(cylinder_idx, {})
    cap = nodes.get(plane_idx, {})
    axis = _node_axis(cylinder)
    if _vector_norm(axis) <= 1e-9:
        return False

    cap_area = max(_node_area(cap), 1e-9)
    cap_coord = _dot3(_node_center(cap), axis)
    slab_planes = []
    for idx, node in nodes.items():
        if idx in {cylinder_idx, plane_idx} or node.get("primitive_surface") != "plane":
            continue
        if _node_area(node) < cap_area * 20.0:
            continue
        normal = _node_normal(node)
        if abs(_dot3(normal, axis)) < 0.85:
            continue
        slab_planes.append((_node_area(node), _dot3(_node_center(node), axis)))

    if len(slab_planes) < 2:
        return False
    slab_coords = [coord for _, coord in sorted(slab_planes, reverse=True)[:2]]
    low, high = min(slab_coords), max(slab_coords)
    radius = max(_node_radius(cylinder), 1e-9)
    tolerance = max(radius * 0.2, 1e-4)
    return cap_coord < low - tolerance or cap_coord > high + tolerance


def _is_circle_edge(edge: dict) -> bool:
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


def _face_looks_like_blind_cylinder(face: dict) -> bool:
    return (
        face.get("macro_feature") == "blind_hole_round"
        or face.get("semantic_subtype") == "blind_hole_cylinder"
        or face.get("raw_macro_feature") == "blind_hole_round"
        or face.get("raw_semantic_subtype") == "blind_hole_cylinder"
    )


def _cylinder_has_inner_hole_role(face_idx: int, graph_data: dict) -> bool:
    for src, dst, attr in _edge_records(graph_data):
        if face_idx not in {src, dst} or not _is_circle_edge(attr):
            continue
        if attr.get("cylindrical_wall_role_hint") == "inner_hole_wall":
            return True
        if int(attr.get("cylindrical_wall_role_hint_id", 0) or 0) == 1:
            return True
        if float(attr.get("cylindrical_wall_role_inner_hint", 0.0) or 0.0) > 0.5:
            return True
        if _edge_loop_role(attr) == "inner_loop":
            return True
    return False


def _cylinder_wall_role_summary(face_idx: int, graph_data: dict) -> dict[str, int]:
    summary = {"known": 0, "inner": 0, "outer": 0}
    for src, dst, attr in _edge_records(graph_data):
        if face_idx not in {src, dst} or not _is_circle_edge(attr):
            continue
        role = "unknown"
        if attr.get("cylindrical_wall_role_hint") == "inner_hole_wall":
            role = "inner"
        elif attr.get("cylindrical_wall_role_hint") == "outer_cylinder_wall":
            role = "outer"
        elif int(attr.get("cylindrical_wall_role_hint_id", 0) or 0) == 1:
            role = "inner"
        elif int(attr.get("cylindrical_wall_role_hint_id", 0) or 0) == 2:
            role = "outer"
        elif float(attr.get("cylindrical_wall_role_inner_hint", 0.0) or 0.0) > 0.5:
            role = "inner"
        elif float(attr.get("cylindrical_wall_role_outer_hint", 0.0) or 0.0) > 0.5:
            role = "outer"
        else:
            loop_role = _edge_loop_role(attr)
            if loop_role == "inner_loop":
                role = "inner"
            elif loop_role == "outer_loop":
                role = "outer"
        if role == "unknown":
            continue
        summary["known"] += 1
        summary[role] += 1
    return summary


def _counterbore_outer_role_invalid_reason(face_indices: list[int], graph_data: dict) -> str | None:
    nodes = _nodes_by_idx(graph_data)
    cylinder_indices = [
        int(idx)
        for idx in face_indices
        if nodes.get(int(idx), {}).get("primitive_surface") == "cylinder"
    ]
    if len(cylinder_indices) < 2:
        return None
    summaries = [_cylinder_wall_role_summary(idx, graph_data) for idx in cylinder_indices]
    known_summaries = [summary for summary in summaries if summary["known"] > 0]
    if len(known_summaries) < len(cylinder_indices):
        return None
    if any(summary["inner"] > 0 for summary in known_summaries):
        return None
    if all(summary["outer"] > 0 for summary in known_summaries):
        return "counterbore_requires_inner_hole_wall_role"
    return None

def _node_descriptor(node: dict, key: str, default: float = 0.0) -> float:
    return float(node.get("attributes", {}).get("shape_descriptors", {}).get(key, default) or default)


def _node_area(node: dict) -> float:
    return float(node.get("attributes", {}).get("physical", {}).get("area", 0.0) or 0.0)


def _node_radius(node: dict) -> float:
    return float(node.get("attributes", {}).get("specific", {}).get("radius", 0.0) or 0.0)


def _node_center(node: dict) -> list[float]:
    center = node.get("attributes", {}).get("physical", {}).get("center_of_mass", [0.0, 0.0, 0.0])
    return [float(value) for value in center[:3]]


def _node_axis(node: dict) -> list[float]:
    return _normalize_vector(node.get("attributes", {}).get("specific", {}).get("axis", [0.0, 0.0, 0.0]))


def _node_normal(node: dict) -> list[float]:
    return _normalize_vector(node.get("attributes", {}).get("specific", {}).get("normal", [0.0, 0.0, 0.0]))


def _distance_between_nodes(left: dict, right: dict) -> float:
    left_center = _node_center(left)
    right_center = _node_center(right)
    return sum((a - b) ** 2 for a, b in zip(left_center, right_center)) ** 0.5


def _angle_between_abs(left: list[float], right: list[float]) -> float:
    left = _normalize_vector(left)
    right = _normalize_vector(right)
    if _vector_norm(left) <= 1e-12 or _vector_norm(right) <= 1e-12:
        return 3.141592653589793
    value = max(-1.0, min(1.0, abs(_dot3(left, right))))
    import math

    return math.acos(value)


def _boundary_neighbor_count(face_idx: int, graph_data: dict) -> int:
    neighbors = set()
    for src, dst, _attr in _edge_records(graph_data):
        if src == face_idx:
            neighbors.add(dst)
        elif dst == face_idx:
            neighbors.add(src)
    return len(neighbors)


def _neighbor_areas(face_idx: int, graph_data: dict) -> list[float]:
    nodes = _nodes_by_idx(graph_data)
    neighbors = set()
    for src, dst, _attr in _edge_records(graph_data):
        if src == face_idx:
            neighbors.add(dst)
        elif dst == face_idx:
            neighbors.add(src)
    return [_node_area(nodes[idx]) for idx in neighbors if idx in nodes]


def _looks_like_large_swept_slot_surface(face_idx: int, node: dict, graph_data: dict) -> bool:
    if _node_area(node) < 5000.0:
        return False
    edge_lengths = [
        float(attr.get("length", 0.0) or 0.0)
        for src, dst, attr in _edge_records(graph_data)
        if face_idx in {src, dst}
    ]
    positive = [length for length in edge_lengths if length > 1e-9]
    if len(positive) < 2:
        return False
    longest = max(positive)
    return sum(1 for length in positive if length >= 0.75 * longest) >= 2


def _looks_like_convex_partial_cylinder_opening(face_idx: int, graph_data: dict) -> bool:
    nodes = _nodes_by_idx(graph_data)
    node = nodes.get(face_idx, {})
    if _node_radius(node) < 15.0 and _node_area(node) < 400.0:
        return False
    planar_circular_neighbors = set()
    has_convex_line = False
    has_concave_line = False
    for src, dst, attr in _edge_records(graph_data):
        if face_idx not in {src, dst}:
            continue
        neighbor = dst if src == face_idx else src
        if _is_circle_edge(attr) and nodes.get(neighbor, {}).get("primitive_surface") == "plane":
            planar_circular_neighbors.add(neighbor)
        if _is_line_edge(attr):
            if _edge_has_convexity(attr, "convex"):
                has_convex_line = True
            if _is_concave_edge(attr):
                has_concave_line = True
    return len(planar_circular_neighbors) == 2 and has_convex_line and not has_concave_line


def _looks_like_exterior_cylinder_not_slot_bottom(face_idx: int, node: dict, graph_data: dict) -> bool:
    radius = _node_radius(node)
    area = _node_area(node)
    if radius < 50.0 or area < 50000.0:
        return False
    if _boundary_neighbor_count(face_idx, graph_data) >= 6:
        return True
    nodes = _nodes_by_idx(graph_data)
    for src, dst, _attr in _edge_records(graph_data):
        if face_idx not in {src, dst}:
            continue
        neighbor = dst if src == face_idx else src
        if nodes.get(neighbor, {}).get("primitive_surface") == "sphere":
            return True
    return False


def _rib_cap_between_side_faces(
    left_idx: int,
    right_idx: int,
    candidate_planes: list[int],
    graph_data: dict,
) -> int | None:
    candidate_set = set(candidate_planes) - {left_idx, right_idx}
    left_neighbors = _convex_line_neighbors(left_idx, graph_data)
    right_neighbors = _convex_line_neighbors(right_idx, graph_data)
    common = sorted(candidate_set & left_neighbors & right_neighbors)
    if not common:
        return None
    nodes = _nodes_by_idx(graph_data)
    return min(common, key=lambda idx: _node_area(nodes.get(idx, {})))


def _concave_line_plane_neighbors(face_idx: int, graph_data: dict, allowed_faces: set[int]) -> set[int]:
    nodes = _nodes_by_idx(graph_data)
    neighbors = set()
    for src, dst, attr in _edge_records(graph_data):
        if face_idx not in {src, dst}:
            continue
        if not _is_line_edge(attr) or not _is_concave_edge(attr):
            continue
        neighbor = dst if src == face_idx else src
        if neighbor not in allowed_faces:
            continue
        if nodes.get(neighbor, {}).get("primitive_surface") != "plane":
            continue
        neighbors.add(neighbor)
    return neighbors

def _convex_line_neighbors(face_idx: int, graph_data: dict) -> set[int]:
    neighbors = set()
    for src, dst, attr in _edge_records(graph_data):
        if face_idx not in {src, dst}:
            continue
        if not _is_line_edge(attr) or not _edge_has_convexity(attr, "convex"):
            continue
        neighbors.add(dst if src == face_idx else src)
    return neighbors


def _rib_top_face(face_indices: list[int], graph_data: dict) -> int | None:
    nodes = _nodes_by_idx(graph_data)
    face_set = {int(idx) for idx in face_indices}
    if not face_set:
        return None
    convex_degree = {idx: 0 for idx in face_set}
    for src, dst, attr in _edge_records(graph_data):
        if src not in face_set or dst not in face_set:
            continue
        if _is_line_edge(attr) and _edge_has_convexity(attr, "convex"):
            convex_degree[src] += 1
    return min(
        face_set,
        key=lambda idx: (
            -convex_degree.get(idx, 0),
            _node_area(nodes.get(idx, {})),
        ),
    )


def _dot3(left: list[float], right: list[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _vector_norm(vector: list[float]) -> float:
    return _dot3(vector, vector) ** 0.5


def _normalize_vector(vector: list[float]) -> list[float]:
    values = [float(value) for value in vector[:3]]
    norm = _vector_norm(values)
    if norm <= 1e-12:
        return [0.0, 0.0, 0.0]
    return [value / norm for value in values]


def _long_line_edge_count(face_idx: int, graph_data: dict) -> int:
    lengths = []
    for src, dst, attr in _edge_records(graph_data):
        if face_idx not in {src, dst} or not _is_line_edge(attr):
            continue
        length = float(attr.get("length", 0.0) or 0.0)
        if length > 1e-9:
            lengths.append(length)
    if not lengths:
        return 0
    longest = max(lengths)
    return sum(1 for length in lengths if length >= 0.75 * longest)


def _cylindrical_face_angle(face_idx: int, node: dict, graph_data: dict) -> float:
    radius = _node_radius(node)
    area = _node_area(node)
    if radius <= 1e-9 or area <= 1e-9:
        return 0.0
    line_lengths = [
        float(attr.get("length", 0.0) or 0.0)
        for src, dst, attr in _edge_records(graph_data)
        if face_idx in {src, dst} and _is_line_edge(attr)
    ]
    if not line_lengths:
        return 0.0
    length = max(line_lengths)
    if length <= 1e-9:
        return 0.0
    return area / (radius * length)





















