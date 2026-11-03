"""Unified score-based global decoding for P1-B candidate objects."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "topocae_p1_b_global_decode_1"
TRANSITIONS = {"fillet", "chamfer"}
POCKETS = {"pocket_rect", "pocket_circular"}
THIN_WALLS = {"rib", "web"}


def decode_feature_objects(
    graph_data: Mapping[str, Any],
    verified_candidates: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select traceable non-conflicting objects using one global score order."""

    candidates = [dict(record) for record in verified_candidates]
    accepted = [record for record in candidates if record.get("status") == "accepted"]
    abstained = [record for record in candidates if record.get("status") != "accepted"]
    ordered = sorted(
        accepted,
        key=lambda record: (
            -float(record.get("final_confidence", 0.0) or 0.0),
            -len(record.get("refined_face_indices", [])),
            str(record.get("candidate_id", "")),
        ),
    )
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    adjacency = _adjacency(graph_data)

    for candidate in ordered:
        conflicts = []
        attachments = []
        for existing in selected:
            relation = _candidate_relation(candidate, existing, adjacency)
            if relation["kind"] == "conflict":
                conflicts.append(relation)
            elif relation["kind"] == "attachment":
                attachments.append(relation)
        if conflicts:
            rejected.append({
                "candidate_id": candidate.get("candidate_id"),
                "action": "global_conflict_reject",
                "predicted_macro": candidate.get("predicted_macro"),
                "face_indices": list(candidate.get("refined_face_indices", [])),
                "final_confidence": candidate.get("final_confidence"),
                "conflicts": conflicts,
            })
            continue
        selected_record = dict(candidate)
        selected_record["global_score"] = float(candidate.get("final_confidence", 0.0) or 0.0)
        selected_record["attachments"] = attachments
        selected.append(selected_record)

    objects = []
    for index, record in enumerate(selected):
        objects.append({
            "feature_object_id": f"feature_object_{index:04d}",
            "macro_feature": record["predicted_macro"],
            "face_indices": sorted(int(face) for face in record.get("refined_face_indices", [])),
            "boundary_face_indices": sorted(int(face) for face in record.get("boundary_face_indices", [])),
            "confidence": float(record.get("final_confidence", 0.0) or 0.0),
            "candidate_id": record.get("candidate_id"),
            "attachments": record.get("attachments", []),
            "trace": {
                "candidate_score": record.get("macro_confidence_before_correction"),
                "validity_score": record.get("candidate_validity_score"),
                "topology_ambiguity_penalty": record.get("topology_ambiguity_penalty"),
                "global_score": record.get("global_score"),
                "topology_evidence": record.get("topology_evidence", {}),
                "original_face_indices": record.get("original_face_indices", []),
                "refined_face_indices": record.get("refined_face_indices", []),
                "class_mask": record.get("class_mask", {}),
                "threshold_selection_sha256": record.get("threshold_selection_sha256"),
            },
        })
    face_owners: dict[int, list[str]] = defaultdict(list)
    for object_record in objects:
        for face in object_record["face_indices"]:
            face_owners[face].append(object_record["feature_object_id"])
    unexplained_overlap = {
        str(face): owners
        for face, owners in face_owners.items()
        if len(owners) > 1 and not _overlap_explained(face, owners, objects)
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "grouping_method": "same_instance_plus_candidate_topology_global_decode",
        "feature_objects": objects,
        "accepted_candidate_count": len(objects),
        "global_conflict_reject_count": len(rejected),
        "abstention_count": len(abstained),
        "global_conflict_rejections": rejected,
        "abstained_candidates": abstained,
        "unexplained_face_overlap": unexplained_overlap,
        "traceability": {
            "original_face_ids_preserved": True,
            "score_order_global": True,
            "main_feature_overlap_default": "reject_lower_score",
            "transition_attachment_allowed": True,
            "pocket_thin_wall_coexistence": "allowed_only_without_face_set_swallowing",
        },
    }


def _candidate_relation(
    candidate: Mapping[str, Any],
    existing: Mapping[str, Any],
    adjacency: Mapping[int, set[int]],
) -> dict[str, Any]:
    candidate_faces = set(int(face) for face in candidate.get("refined_face_indices", []))
    existing_faces = set(int(face) for face in existing.get("refined_face_indices", []))
    overlap = candidate_faces & existing_faces
    candidate_macro = str(candidate.get("predicted_macro", "unknown"))
    existing_macro = str(existing.get("predicted_macro", "unknown"))
    adjacent = any(neighbor in existing_faces for face in candidate_faces for neighbor in adjacency.get(face, set()))
    relation = {
        "candidate_id": candidate.get("candidate_id"),
        "existing_candidate_id": existing.get("candidate_id"),
        "candidate_macro": candidate_macro,
        "existing_macro": existing_macro,
        "overlap_face_indices": sorted(overlap),
        "brep_adjacent": adjacent,
    }
    if candidate_macro in TRANSITIONS or existing_macro in TRANSITIONS:
        transition = candidate if candidate_macro in TRANSITIONS else existing
        parent = existing if candidate_macro in TRANSITIONS else candidate
        if adjacent or overlap:
            return {
                **relation,
                "kind": "attachment",
                "transition_candidate_id": transition.get("candidate_id"),
                "parent_candidate_id": parent.get("candidate_id"),
                "explanation": "fillet_or_chamfer_parent_attachment",
            }
    if ({candidate_macro, existing_macro} & POCKETS) and ({candidate_macro, existing_macro} & THIN_WALLS):
        if not overlap and adjacent:
            return {**relation, "kind": "coexist", "explanation": "pocket_embedded_thin_wall_disjoint_faces"}
        return {**relation, "kind": "conflict", "explanation": "pocket_thin_wall_face_set_swallowing"}
    if overlap:
        return {**relation, "kind": "conflict", "explanation": "unexplained_main_feature_overlap"}
    return {**relation, "kind": "independent", "explanation": "disjoint_candidates"}


def _overlap_explained(face: int, owners: list[str], objects: list[Mapping[str, Any]]) -> bool:
    owner_records = [record for record in objects if record.get("feature_object_id") in owners]
    macros = {str(record.get("macro_feature")) for record in owner_records}
    return bool(macros & TRANSITIONS)


def _adjacency(graph: Mapping[str, Any]) -> dict[int, set[int]]:
    result: dict[int, set[int]] = defaultdict(set)
    for edge in graph.get("edges", []):
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            left, right = int(edge[0]), int(edge[1])
            result[left].add(right)
            result[right].add(left)
    return result


__all__ = ["SCHEMA_VERSION", "decode_feature_objects"]
