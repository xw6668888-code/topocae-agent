"""Synthetic-truth hard negatives defined only by generic topology relations."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from utils.p1_b_candidate_proposer import _coaxial, _part_scale


SCHEMA_VERSION = "topocae_p1_b_candidate_hard_negatives_1"
CONFUSION_MAP = {
    "slot_o": ["fillet", "obround_hole"],
    "fillet": ["slot_o"],
    "slot_u": ["obround_hole"],
    "obround_hole": ["slot_u", "slot_o"],
    "cylinder_boss": ["blind_hole_round"],
    "blind_hole_round": ["cylinder_boss"],
    "rib": ["pocket_rect", "stock"],
    "web": ["pocket_rect", "stock"],
    "pocket_rect": ["rib", "web"],
}


def generate_candidate_hard_negatives(
    graph_data: Mapping[str, Any],
    truth_instances: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create missing-face, superset, merge, coaxial-role, and confusion negatives."""

    truths = [_truth_record(record, position) for position, record in enumerate(truth_instances)]
    adjacency = _adjacency(graph_data)
    nodes = _nodes(graph_data)
    negatives: list[dict[str, Any]] = []

    for truth in truths:
        faces = truth["face_indices"]
        if len(faces) > 1:
            removal_order = sorted(faces, key=lambda index: (-len(adjacency.get(index, set())), index))[:2]
            for removed in removal_order:
                negatives.append(_negative(
                    "missing_truth_face_subset",
                    [index for index in faces if index != removed],
                    truth["macro_feature"],
                    [truth["instance_id"]],
                    {"removed_face_index": removed},
                ))
        outside_neighbors = sorted({
            neighbor
            for index in faces
            for neighbor in adjacency.get(index, set())
            if neighbor not in set(faces)
        })
        for added in outside_neighbors[:2]:
            negatives.append(_negative(
                "adjacent_stock_or_transition_superset",
                [*faces, added],
                truth["macro_feature"],
                [truth["instance_id"]],
                {"added_face_index": added},
            ))
        for confused_macro in CONFUSION_MAP.get(truth["macro_feature"], []):
            negatives.append(_negative(
                "wrong_macro_hard_confusion",
                faces,
                confused_macro,
                [truth["instance_id"]],
                {"truth_macro_feature": truth["macro_feature"]},
            ))

    for left_position, left in enumerate(truths):
        left_set = set(left["face_indices"])
        for right in truths[left_position + 1:]:
            right_set = set(right["face_indices"])
            if any(neighbor in right_set for index in left_set for neighbor in adjacency.get(index, set())):
                negatives.append(_negative(
                    "adjacent_instance_merge",
                    sorted(left_set | right_set),
                    left["macro_feature"],
                    [left["instance_id"], right["instance_id"]],
                    {"second_macro_feature": right["macro_feature"]},
                ))
            coaxial_pair = _first_coaxial_pair(left_set, right_set, nodes, _part_scale(graph_data))
            if coaxial_pair is not None and left["macro_feature"] != right["macro_feature"]:
                negatives.append(_negative(
                    "coaxial_different_function_merge",
                    sorted(left_set | right_set),
                    left["macro_feature"],
                    [left["instance_id"], right["instance_id"]],
                    {"coaxial_face_pair": list(coaxial_pair), "second_macro_feature": right["macro_feature"]},
                ))

    unique = {}
    for record in negatives:
        key = (record["negative_kind"], record["proposed_macro"], tuple(record["face_indices"]))
        unique.setdefault(key, record)
    records = [unique[key] for key in sorted(unique)]
    for index, record in enumerate(records):
        record["hard_negative_id"] = f"hard_negative_{index:06d}"
    return {
        "schema_version": SCHEMA_VERSION,
        "record_count": len(records),
        "records": records,
        "contract": {
            "source": "synthetic_truth_and_generic_topology_only",
            "uses_sample_names": False,
            "uses_real_part_face_ids": False,
            "validity_target": 0,
        },
    }


def _negative(
    kind: str,
    faces: Iterable[int],
    proposed_macro: str,
    source_instance_ids: Iterable[Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "negative_kind": kind,
        "face_indices": sorted(set(int(index) for index in faces)),
        "proposed_macro": str(proposed_macro),
        "validity_target": 0,
        "source_instance_ids": list(source_instance_ids),
        "topology_evidence": dict(evidence),
    }


def _truth_record(record: Mapping[str, Any], position: int) -> dict[str, Any]:
    return {
        "instance_id": record.get("instance_id", position + 1),
        "macro_feature": str(record.get("macro_feature", "unknown_feature")),
        "face_indices": sorted(set(int(index) for index in record.get("face_indices", []))),
    }


def _first_coaxial_pair(
    left: set[int], right: set[int], nodes: Mapping[int, Mapping[str, Any]], scale: float
) -> tuple[int, int] | None:
    for left_index in sorted(left):
        if str(nodes.get(left_index, {}).get("primitive_surface", "")) not in {"cylinder", "cone"}:
            continue
        for right_index in sorted(right):
            if str(nodes.get(right_index, {}).get("primitive_surface", "")) not in {"cylinder", "cone"}:
                continue
            if _coaxial(nodes[left_index], nodes[right_index], scale):
                return left_index, right_index
    return None


def _adjacency(graph: Mapping[str, Any]) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    for edge in graph.get("edges", []):
        if not isinstance(edge, (list, tuple)) or len(edge) < 2:
            continue
        left, right = int(edge[0]), int(edge[1])
        result.setdefault(left, set()).add(right)
        result.setdefault(right, set()).add(left)
    return result


def _nodes(graph: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {int(node.get("idx", position)): node for position, node in enumerate(graph.get("nodes", []))}


__all__ = ["SCHEMA_VERSION", "generate_candidate_hard_negatives"]
