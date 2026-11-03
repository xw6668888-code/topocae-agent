"""P1-B same-instance assembly with an explicit, globally frozen protocol.

This module intentionally does not read truth labels while assembling candidates.
Truth instance ids are used only by the separate evaluation helper.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "topocae_p1_b_same_instance_assembly_1"
SYMMETRIZATION_METHOD = "arithmetic_mean_of_available_directed_probabilities"
STOCK_MACROS = {"", "stock", "stock_face", "background", "unknown", "unknown_feature", None}

MACRO_FAMILY = {
    "through_hole_round": "hole",
    "blind_hole_round": "hole",
    "stepped_hole_round": "hole",
    "counterbore_hole": "hole",
    "countersink_hole": "hole",
    "obround_hole": "hole",
    "pocket_rect": "subtractive_open",
    "pocket_circular": "subtractive_open",
    "slot_u": "slot",
    "slot_rect": "slot",
    "slot_v": "slot",
    "slot_o": "slot",
    "cylinder_boss": "boss",
    "rectangular_boss": "boss",
    "rib": "thin_wall",
    "web": "thin_wall",
    "fillet": "transition",
    "chamfer": "transition",
}


def select_global_same_instance_threshold(
    records: Iterable[Mapping[str, Any]],
    *,
    allowed_thresholds: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Select one threshold using Synthetic-Val only.

    Each record must contain ``split``, ``probability`` and binary ``label``.
    The selector rejects mixed or differently named splits instead of silently
    accepting development-test, real-part, or holdout observations.
    """

    normalized: list[tuple[float, int]] = []
    for record in records:
        split = str(record.get("split", ""))
        if split != "synthetic_val":
            raise ValueError("same-instance threshold selection accepts split='synthetic_val' only")
        probability = _probability(record.get("probability"))
        label = int(record.get("label", -1))
        if probability is None or label not in {0, 1}:
            raise ValueError("each threshold record needs a finite probability and binary label")
        normalized.append((probability, label))
    if not normalized:
        raise ValueError("at least one Synthetic-Val edge record is required")

    thresholds = sorted({float(value) for value in (allowed_thresholds or [i / 100.0 for i in range(5, 96)])})
    if not thresholds or any(value < 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("threshold candidates must be inside [0, 1]")
    scans = []
    for threshold in thresholds:
        metrics = _binary_metrics(normalized, threshold)
        scans.append({"threshold": threshold, **metrics})
    best = max(scans, key=lambda item: (item["f1"], item["precision"], item["recall"], item["threshold"]))
    selection_payload = {
        "schema_version": "topocae_p1_b_same_instance_threshold_selection_1",
        "selection_split": "synthetic_val",
        "global_threshold": best["threshold"],
        "objective": "edge_f1_then_precision_then_recall_then_higher_threshold",
        "edge_count": len(normalized),
        "best_metrics": {key: best[key] for key in ("precision", "recall", "f1", "tp", "fp", "fn", "tn")},
        "symmetrization_method": SYMMETRIZATION_METHOD,
        "scan": scans,
    }
    canonical = json.dumps(selection_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    selection_payload["selection_sha256"] = sha256(canonical.encode("utf-8")).hexdigest().upper()
    return selection_payload


def symmetrize_directed_probabilities(
    graph_data: Mapping[str, Any],
    prediction_data: Mapping[str, Any],
) -> dict[tuple[int, int], dict[str, Any]]:
    """Return one record per undirected B-Rep adjacency.

    Forward and reverse values are averaged. Repeated observations in one
    direction are averaged first. A missing direction remains traceable and is
    never replaced by a category/part-specific constant.
    """

    graph_edges = _graph_edges(graph_data)
    observations = _directed_probability_observations(graph_edges, prediction_data)
    unique_pairs = sorted({tuple(sorted(edge)) for edge in graph_edges if edge[0] != edge[1]})
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for left, right in unique_pairs:
        forward = observations.get((left, right), [])
        reverse = observations.get((right, left), [])
        direction_means = []
        if forward:
            direction_means.append(sum(forward) / len(forward))
        if reverse:
            direction_means.append(sum(reverse) / len(reverse))
        probability = sum(direction_means) / len(direction_means) if direction_means else None
        result[(left, right)] = {
            "source": left,
            "target": right,
            "probability": probability,
            "forward_probability": sum(forward) / len(forward) if forward else None,
            "reverse_probability": sum(reverse) / len(reverse) if reverse else None,
            "direction_count": len(direction_means),
            "symmetrization_method": SYMMETRIZATION_METHOD,
        }
    return result


def assemble_same_instance_candidates(
    graph_data: Mapping[str, Any],
    prediction_data: Mapping[str, Any],
    *,
    global_threshold: float,
    threshold_selection_sha256: str,
) -> dict[str, Any]:
    """Assemble adjacent non-stock faces across provisional macro boundaries."""

    threshold = float(global_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("global_threshold must be inside [0, 1]")
    if not threshold_selection_sha256:
        raise ValueError("a Synthetic-Val threshold selection hash is required")

    nodes = _nodes_by_idx(graph_data)
    face_predictions = _prediction_faces_by_idx(prediction_data)
    macros = {idx: _face_macro(face_predictions.get(idx, nodes.get(idx, {}))) for idx in nodes}
    seed_indices = sorted(idx for idx, macro in macros.items() if macro not in STOCK_MACROS)
    parent = {idx: idx for idx in seed_indices}
    edge_records = symmetrize_directed_probabilities(graph_data, prediction_data)
    edge_attrs = _undirected_edge_attributes(graph_data)
    accepted_edges: list[dict[str, Any]] = []
    rejected_edges: list[dict[str, Any]] = []

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for pair, probability_record in edge_records.items():
        left, right = pair
        probability = probability_record["probability"]
        reason = "below_global_threshold"
        compatible = False
        if probability is not None and probability >= threshold:
            if left not in parent or right not in parent:
                reason = "stock_or_unknown_face_not_used_as_same_instance_bridge"
            else:
                compatible, reason = _topology_compatible(
                    macros[left], macros[right], edge_attrs.get(pair, []), nodes.get(left, {}), nodes.get(right, {})
                )
        record = {
            **probability_record,
            "left_macro": macros.get(left),
            "right_macro": macros.get(right),
            "topology_compatible": compatible,
            "decision_reason": reason,
        }
        if probability is not None and probability >= threshold and compatible:
            union(left, right)
            accepted_edges.append(record)
        else:
            rejected_edges.append(record)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in seed_indices:
        groups[find(index)].append(index)
    candidates = []
    for candidate_id, faces in enumerate(sorted((sorted(group) for group in groups.values()), key=lambda item: item[0])):
        face_set = set(faces)
        internal = [record for pair, record in edge_records.items() if pair[0] in face_set and pair[1] in face_set]
        probabilities = [float(record["probability"]) for record in internal if record["probability"] is not None]
        macro_counts = Counter(macros[index] for index in faces)
        candidates.append({
            "candidate_id": f"same_instance_{candidate_id:04d}",
            "source": "same_instance_strong_component",
            "face_indices": faces,
            "provisional_macros": sorted(macro_counts),
            "provisional_macro_histogram": dict(sorted(macro_counts.items())),
            "cross_provisional_macro": len(macro_counts) > 1,
            "internal_same_instance_probability_mean": sum(probabilities) / len(probabilities) if probabilities else None,
            "internal_same_instance_probability_min": min(probabilities) if probabilities else None,
            "internal_same_instance_probability_max": max(probabilities) if probabilities else None,
            "trace": {
                "assembly_threshold": threshold,
                "threshold_selection_sha256": threshold_selection_sha256,
                "symmetrization_method": SYMMETRIZATION_METHOD,
                "adjacency_constrained": True,
                "topology_compatibility_constrained": True,
            },
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "global_threshold": threshold,
        "threshold_selection_sha256": threshold_selection_sha256,
        "symmetrization_method": SYMMETRIZATION_METHOD,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "accepted_edges": accepted_edges,
        "rejected_edges": rejected_edges,
        "missing_bidirectional_pair_count": sum(1 for record in edge_records.values() if record["direction_count"] < 2),
    }


def evaluate_same_instance_edges(
    graph_data: Mapping[str, Any],
    prediction_data: Mapping[str, Any],
    *,
    global_threshold: float,
) -> dict[str, Any]:
    """Evaluate undirected edge P/R/F1 from synthetic truth instance ids."""

    nodes = _nodes_by_idx(graph_data)
    pairs = symmetrize_directed_probabilities(graph_data, prediction_data)
    values: list[tuple[float, int]] = []
    missing = 0
    for (left, right), record in pairs.items():
        probability = record["probability"]
        if probability is None:
            missing += 1
            probability = 0.0
        left_instance = int(nodes.get(left, {}).get("instance_id", 0) or 0)
        right_instance = int(nodes.get(right, {}).get("instance_id", 0) or 0)
        label = int(left_instance > 0 and left_instance == right_instance)
        values.append((float(probability), label))
    metrics = _binary_metrics(values, float(global_threshold))
    return {
        "schema_version": "topocae_p1_b_same_instance_edge_metrics_1",
        "threshold": float(global_threshold),
        "symmetrization_method": SYMMETRIZATION_METHOD,
        "edge_count": len(values),
        "missing_probability_count": missing,
        **metrics,
    }


def _binary_metrics(values: Iterable[tuple[float, int]], threshold: float) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for probability, label in values:
        predicted = int(probability >= threshold)
        if predicted and label:
            tp += 1
        elif predicted and not label:
            fp += 1
        elif not predicted and label:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _topology_compatible(
    left_macro: str | None,
    right_macro: str | None,
    attributes: list[Mapping[str, Any]],
    left_node: Mapping[str, Any],
    right_node: Mapping[str, Any],
) -> tuple[bool, str]:
    for attribute in attributes:
        for key in ("same_instance_compatible", "assembly_compatible"):
            if key in attribute and attribute[key] is False:
                return False, "explicit_topology_incompatibility"
    left_family = MACRO_FAMILY.get(str(left_macro), str(left_macro))
    right_family = MACRO_FAMILY.get(str(right_macro), str(right_macro))
    if left_family == right_family:
        return True, "same_macro_family"
    if "transition" in {left_family, right_family}:
        return True, "parent_transition_compatibility"
    if {left_family, right_family} <= {"hole", "subtractive_open", "slot"}:
        return True, "subtractive_family_compatibility"
    if {left_family, right_family} <= {"boss", "thin_wall"}:
        return True, "additive_family_compatibility"
    left_primitive = str(left_node.get("primitive_surface", ""))
    right_primitive = str(right_node.get("primitive_surface", ""))
    if {left_primitive, right_primitive} <= {"cylinder", "cone", "plane"} and {
        left_family,
        right_family,
    } <= {"hole", "subtractive_open", "slot"}:
        return True, "hole_chain_surface_compatibility"
    return False, "incompatible_additive_subtractive_families"


def _directed_probability_observations(
    graph_edges: list[tuple[int, int]], prediction_data: Mapping[str, Any]
) -> dict[tuple[int, int], list[float]]:
    raw = None
    for key in (
        "edge_same_instance_probabilities",
        "edge_same_instance_probability",
        "edge_same_instance",
        "prediction_edge_same_instance",
    ):
        if key in prediction_data:
            raw = prediction_data[key]
            break
    observations: dict[tuple[int, int], list[float]] = defaultdict(list)
    if raw is None:
        return observations
    if isinstance(raw, Mapping) and "probabilities" in raw:
        raw = raw["probabilities"]
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            pair = _pair_from_key(key)
            probability = _probability(value)
            if pair is not None and probability is not None:
                observations[pair].append(probability)
        return observations
    if not isinstance(raw, (list, tuple)):
        return observations
    for position, item in enumerate(raw):
        pair = None
        if isinstance(item, Mapping):
            pair = _pair_from_record(item)
            if pair is None and "edge_index" in item:
                edge_index = int(item["edge_index"])
                if 0 <= edge_index < len(graph_edges):
                    pair = graph_edges[edge_index]
        elif position < len(graph_edges):
            pair = graph_edges[position]
        probability = _probability(item)
        if pair is not None and probability is not None:
            observations[pair].append(probability)
    return observations


def _graph_edges(graph_data: Mapping[str, Any]) -> list[tuple[int, int]]:
    result = []
    for item in graph_data.get("edges", []):
        pair = _pair_from_record(item)
        if pair is not None:
            result.append(pair)
    return result


def _undirected_edge_attributes(graph_data: Mapping[str, Any]) -> dict[tuple[int, int], list[Mapping[str, Any]]]:
    result: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for edge, attribute in zip(_graph_edges(graph_data), graph_data.get("edge_attributes", [])):
        if isinstance(attribute, Mapping):
            result[tuple(sorted(edge))].append(attribute)
    return result


def _nodes_by_idx(graph_data: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {int(node.get("idx", position)): node for position, node in enumerate(graph_data.get("nodes", []))}


def _prediction_faces_by_idx(prediction_data: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {int(face.get("idx", position)): face for position, face in enumerate(prediction_data.get("faces", []))}


def _face_macro(record: Mapping[str, Any]) -> str | None:
    for key in ("macro_feature", "raw_macro_feature", "predicted_macro_feature"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return "stock"


def _pair_from_record(record: Any) -> tuple[int, int] | None:
    if isinstance(record, Mapping):
        for left_key, right_key in (("src", "dst"), ("source", "target"), ("source_face", "target_face")):
            if left_key in record and right_key in record:
                return int(record[left_key]), int(record[right_key])
        if "face_pair" in record:
            record = record["face_pair"]
    if isinstance(record, (list, tuple)) and len(record) >= 2:
        return int(record[0]), int(record[1])
    return None


def _pair_from_key(value: Any) -> tuple[int, int] | None:
    pair = _pair_from_record(value)
    if pair is not None:
        return pair
    if isinstance(value, str):
        cleaned = value.strip().replace("(", "").replace(")", "").replace("[", "").replace("]", "")
        for delimiter in (",", "-", ":", "|"):
            parts = [part.strip() for part in cleaned.split(delimiter)]
            if len(parts) == 2 and all(part.lstrip("-").isdigit() for part in parts):
                return int(parts[0]), int(parts[1])
    return None


def _probability(value: Any) -> float | None:
    if isinstance(value, Mapping):
        for key in ("probability", "same_instance_probability", "p_same_instance", "score"):
            if key in value:
                return _probability(value[key])
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return min(1.0, max(0.0, number))


__all__ = [
    "SCHEMA_VERSION",
    "SYMMETRIZATION_METHOD",
    "assemble_same_instance_candidates",
    "evaluate_same_instance_edges",
    "select_global_same_instance_threshold",
    "symmetrize_directed_probabilities",
]
