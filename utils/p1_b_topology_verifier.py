"""P1-B topology verifier: mask, veto, confidence correction, abstention only."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from hashlib import sha256
import json
import math
from typing import Any, Iterable, Mapping

from utils.p1_b_candidate_features import MACRO_CLASSES


SCHEMA_VERSION = "topocae_p1_b_topology_verifier_1"


def select_global_candidate_thresholds(
    records: Iterable[Mapping[str, Any]],
    *,
    threshold_grid: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Choose all candidate operating thresholds from Synthetic-Val only."""

    normalized = []
    membership_values: list[tuple[float, int]] = []
    boundary_values: list[tuple[float, int]] = []
    for record in records:
        if str(record.get("split", "")) != "synthetic_val":
            raise ValueError("candidate threshold selection accepts split='synthetic_val' only")
        macro_confidence = _score(record.get("macro_confidence"))
        validity_score = _score(record.get("validity_score"))
        label = int(record.get("accept_label", -1))
        if macro_confidence is None or validity_score is None or label not in {0, 1}:
            raise ValueError("candidate records require macro_confidence, validity_score, and binary accept_label")
        normalized.append((macro_confidence, validity_score, label))
        membership_values.extend(_score_labels(record.get("membership_probabilities", []), record.get("membership_labels", [])))
        boundary_values.extend(_score_labels(record.get("boundary_probabilities", []), record.get("boundary_labels", [])))
    if not normalized:
        raise ValueError("at least one Synthetic-Val candidate record is required")
    grid = sorted(set(float(value) for value in (threshold_grid or [index / 100.0 for index in range(5, 96, 5)])))
    if not grid or any(value < 0.0 or value > 1.0 for value in grid):
        raise ValueError("threshold grid must be inside [0, 1]")
    scans = []
    for confidence in grid:
        for validity in grid:
            metrics = _acceptance_metrics(normalized, confidence, validity)
            scans.append({"min_confidence": confidence, "min_validity": validity, **metrics})
    best = max(
        scans,
        key=lambda item: (
            item["f1"],
            item["precision"],
            item["recall"],
            item["min_confidence"],
            item["min_validity"],
        ),
    )
    membership = _best_binary_threshold(membership_values, grid, default=0.5)
    boundary = _best_binary_threshold(boundary_values, grid, default=0.5)
    result = {
        "schema_version": "topocae_p1_b_candidate_threshold_selection_1",
        "selection_split": "synthetic_val",
        "min_candidate_confidence": best["min_confidence"],
        "min_candidate_validity": best["min_validity"],
        "membership_threshold": membership["threshold"],
        "boundary_threshold": boundary["threshold"],
        "candidate_acceptance_metrics": {key: best[key] for key in ("precision", "recall", "f1", "tp", "fp", "fn", "tn")},
        "membership_metrics": membership,
        "boundary_metrics": boundary,
        "objective": "global_f1_then_precision_then_recall_then_higher_threshold",
        "candidate_record_count": len(normalized),
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    result["selection_sha256"] = sha256(canonical.encode("utf-8")).hexdigest().upper()
    return result


def verify_candidate_prediction(
    graph_data: Mapping[str, Any],
    proposal: Mapping[str, Any],
    model_prediction: Mapping[str, Any],
    *,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply only allowed topology actions; never hard-remap class A to class B."""

    if str(thresholds.get("selection_split", "")) != "synthetic_val":
        raise ValueError("verifier thresholds must be selected on Synthetic-Val")
    if not thresholds.get("selection_sha256"):
        raise ValueError("verifier threshold selection hash is required")
    classes = list(model_prediction.get("macro_classes", MACRO_CLASSES))
    probabilities = [float(value) for value in model_prediction.get("macro_probabilities", [])]
    logits = [float(value) for value in model_prediction.get("macro_logits", [])]
    if len(classes) != len(probabilities):
        raise ValueError("macro class and probability dimensions do not match")
    if logits and len(logits) != len(classes):
        raise ValueError("macro logit dimension does not match class dimension")

    face_indices = [int(index) for index in proposal.get("face_indices", [])]
    structural = {macro: _class_feasibility(graph_data, face_indices, macro) for macro in classes}
    masked = [probability if structural[macro]["feasible"] else 0.0 for macro, probability in zip(classes, probabilities)]
    masked_total = sum(masked)
    masked_probabilities = [value / masked_total for value in masked] if masked_total > 0.0 else [0.0] * len(masked)
    unmasked_index = max(range(len(probabilities)), key=probabilities.__getitem__) if probabilities else None
    masked_index = max(range(len(masked_probabilities)), key=masked_probabilities.__getitem__) if masked_total > 0.0 else None
    selected_macro = classes[masked_index] if masked_index is not None else "unknown"
    macro_confidence = masked_probabilities[masked_index] if masked_index is not None else 0.0
    validity_score = float(model_prediction.get("validity_score", 0.0) or 0.0)

    membership_probabilities = [float(value) for value in model_prediction.get("membership_probabilities", [])]
    if len(membership_probabilities) != len(face_indices):
        raise ValueError("membership probability count must match proposal face count")
    membership_threshold = float(thresholds["membership_threshold"])
    refined_faces = [face for face, probability in zip(face_indices, membership_probabilities) if probability >= membership_threshold]
    refined_faces = _best_connected_membership_component(graph_data, refined_faces, face_indices, membership_probabilities)
    selected_topology = _class_feasibility(graph_data, refined_faces, selected_macro) if selected_macro != "unknown" else {
        "feasible": False,
        "reasons": ["no_feasible_class_after_mask"],
    }
    ambiguity_penalty = _ambiguity_penalty(graph_data, refined_faces)
    corrected_confidence = macro_confidence * validity_score * ambiguity_penalty
    abstention_reasons = []
    if not refined_faces:
        abstention_reasons.append("empty_membership_after_refinement")
    if not selected_topology["feasible"]:
        abstention_reasons.extend(selected_topology["reasons"])
    if validity_score < float(thresholds["min_candidate_validity"]):
        abstention_reasons.append("validity_below_global_threshold")
    if corrected_confidence < float(thresholds["min_candidate_confidence"]):
        abstention_reasons.append("confidence_below_global_threshold")
    status = "abstain" if abstention_reasons else "accepted"
    boundary_probabilities = [float(value) for value in model_prediction.get("boundary_probabilities", [])]
    boundary_faces = [
        face
        for face, probability in zip(face_indices, boundary_probabilities)
        if probability >= float(thresholds["boundary_threshold"])
    ] if len(boundary_probabilities) == len(face_indices) else []
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": proposal.get("candidate_id"),
        "status": status,
        "predicted_macro": selected_macro if status == "accepted" else "unknown",
        "abstain": status == "abstain",
        "abstention_reasons": sorted(set(abstention_reasons)),
        "original_face_indices": face_indices,
        "refined_face_indices": refined_faces,
        "boundary_face_indices": sorted(boundary_faces),
        "macro_confidence_before_correction": macro_confidence,
        "candidate_validity_score": validity_score,
        "topology_ambiguity_penalty": ambiguity_penalty,
        "final_confidence": corrected_confidence,
        "model_unmasked_top1": classes[unmasked_index] if unmasked_index is not None else "unknown",
        "masked_top1": selected_macro,
        "class_mask": {macro: record["feasible"] for macro, record in structural.items()},
        "topology_evidence": selected_topology,
        "threshold_selection_sha256": thresholds["selection_sha256"],
        "verifier_contract": {
            "allowed_actions": ["class_mask", "legality_veto", "confidence_correction", "abstention"],
            "hard_class_rewrite": False,
        },
    }


def _class_feasibility(graph: Mapping[str, Any], faces: list[int], macro: str) -> dict[str, Any]:
    nodes = _nodes(graph)
    face_set = set(faces)
    if not face_set:
        return {"feasible": False, "reasons": ["empty_candidate"], "primitive_histogram": {}}
    primitives = Counter(str(nodes.get(index, {}).get("primitive_surface", "unknown_surface")) for index in face_set)
    connected = _is_connected(graph, face_set)
    reasons = []
    if not connected:
        reasons.append("candidate_not_brep_adjacent_connected")
    curved = primitives["cylinder"] + primitives["cone"] + primitives["toroid"] + primitives["bspline_surface"]
    if macro in {
        "through_hole_round", "blind_hole_round", "stepped_hole_round", "counterbore_hole", "countersink_hole", "cylinder_boss"
    } and primitives["cylinder"] + primitives["cone"] == 0:
        reasons.append("round_feature_without_cylinder_or_cone")
    if macro == "obround_hole" and primitives["cylinder"] + primitives["plane"] < 2:
        reasons.append("obround_without_curved_and_planar_support")
    if macro in {"slot_u", "slot_rect", "slot_v", "slot_o", "pocket_rect", "pocket_circular", "rectangular_boss"} and len(face_set) < 2:
        reasons.append("region_feature_has_fewer_than_two_faces")
    if macro in {"rib", "web"} and (len(face_set) < 2 or primitives["plane"] < 1):
        reasons.append("thin_wall_without_planar_multi_face_support")
    if macro == "fillet" and curved == 0:
        reasons.append("fillet_without_curved_transition_surface")
    if macro == "chamfer" and primitives["plane"] + primitives["cone"] == 0:
        reasons.append("chamfer_without_plane_or_cone")
    return {
        "feasible": not reasons,
        "reasons": reasons,
        "connected": connected,
        "primitive_histogram": dict(sorted(primitives.items())),
        "face_count": len(face_set),
    }


def _best_connected_membership_component(
    graph: Mapping[str, Any],
    refined_faces: list[int],
    original_faces: list[int],
    probabilities: list[float],
) -> list[int]:
    if not refined_faces:
        return []
    adjacency = _adjacency(graph)
    remaining = set(refined_faces)
    components = []
    while remaining:
        start = min(remaining)
        queue = deque([start])
        remaining.remove(start)
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    probability_by_face = dict(zip(original_faces, probabilities))
    return max(
        components,
        key=lambda component: (sum(probability_by_face.get(index, 0.0) for index in component), len(component), -min(component)),
    )


def _ambiguity_penalty(graph: Mapping[str, Any], faces: list[int]) -> float:
    face_set = set(faces)
    values = []
    for edge, attribute in zip(graph.get("edges", []), graph.get("edge_attributes", [])):
        if not isinstance(edge, (list, tuple)) or len(edge) < 2:
            continue
        left, right = int(edge[0]), int(edge[1])
        if (left in face_set) == (right in face_set):
            continue
        values.append(float(attribute.get("recognition_features_v2", {}).get("material_side_ambiguous_evidence", 0.0) or 0.0))
    ambiguity = sum(values) / len(values) if values else 0.0
    return max(0.75, 1.0 - 0.25 * min(1.0, ambiguity))


def _acceptance_metrics(values: list[tuple[float, float, int]], confidence: float, validity: float) -> dict[str, Any]:
    labels = [((macro >= confidence and valid >= validity), label) for macro, valid, label in values]
    return _binary_metrics(labels)


def _best_binary_threshold(values: list[tuple[float, int]], grid: list[float], default: float) -> dict[str, Any]:
    if not values:
        return {"threshold": default, "precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "record_count": 0}
    scans = []
    for threshold in grid:
        metrics = _binary_metrics([(score >= threshold, label) for score, label in values])
        scans.append({"threshold": threshold, **metrics})
    best = max(scans, key=lambda item: (item["f1"], item["precision"], item["recall"], item["threshold"]))
    return {**best, "record_count": len(values)}


def _binary_metrics(values: Iterable[tuple[bool, int]]) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for prediction, label in values:
        if prediction and label:
            tp += 1
        elif prediction and not label:
            fp += 1
        elif not prediction and label:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _score_labels(scores: Any, labels: Any) -> list[tuple[float, int]]:
    if not isinstance(scores, (list, tuple)) or not isinstance(labels, (list, tuple)) or len(scores) != len(labels):
        return []
    result = []
    for score, label in zip(scores, labels):
        value = _score(score)
        binary = int(label)
        if value is not None and binary in {0, 1}:
            result.append((value, binary))
    return result


def _score(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return min(1.0, max(0.0, number))


def _is_connected(graph: Mapping[str, Any], faces: set[int]) -> bool:
    if not faces:
        return False
    adjacency = _adjacency(graph)
    seen = set()
    queue = deque([min(faces)])
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        queue.extend(neighbor for neighbor in adjacency.get(current, set()) if neighbor in faces and neighbor not in seen)
    return seen == faces


def _adjacency(graph: Mapping[str, Any]) -> dict[int, set[int]]:
    result: dict[int, set[int]] = defaultdict(set)
    for edge in graph.get("edges", []):
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            left, right = int(edge[0]), int(edge[1])
            result[left].add(right)
            result[right].add(left)
    return result


def _nodes(graph: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {int(node.get("idx", position)): node for position, node in enumerate(graph.get("nodes", []))}


__all__ = ["SCHEMA_VERSION", "select_global_candidate_thresholds", "verify_candidate_prediction"]
