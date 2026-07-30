from __future__ import annotations

import math
from typing import Any

from core.graph_v2_1 import canonical_graph_signature
from utils.training_export_v2_1 import export_graph_v2_training_record


def compare_graph_v2_invariance(
    reference_graph: dict[str, Any],
    transformed_graph: dict[str, Any],
    *,
    absolute_tolerance: float = 1e-6,
) -> dict[str, Any]:
    reference = export_graph_v2_training_record(reference_graph, ndigits=10)
    transformed = export_graph_v2_training_record(transformed_graph, ndigits=10)
    node_agreement = _feature_agreement(reference["node_features"], transformed["node_features"], absolute_tolerance)
    edge_agreement = _feature_agreement(reference["edge_features"], transformed["edge_features"], absolute_tolerance)
    macro_agreement = _prediction_agreement(
        _probe_predictions(reference["node_features"], class_count=18, phase=0.17),
        _probe_predictions(transformed["node_features"], class_count=18, phase=0.17),
    )
    role_agreement = _prediction_agreement(
        _probe_predictions(reference["node_features"], class_count=8, phase=0.31),
        _probe_predictions(transformed["node_features"], class_count=8, phase=0.31),
    )
    instance_node_agreement = _prediction_agreement(
        _probe_predictions(reference["node_features"], class_count=4, phase=0.47),
        _probe_predictions(transformed["node_features"], class_count=4, phase=0.47),
    )
    same_instance_edge_agreement = _prediction_agreement(
        _probe_predictions(reference["edge_features"], class_count=2, phase=0.59),
        _probe_predictions(transformed["edge_features"], class_count=2, phase=0.59),
    )
    canonical_equal = canonical_graph_signature(reference_graph) == canonical_graph_signature(transformed_graph)
    components = [
        node_agreement,
        edge_agreement,
        macro_agreement,
        role_agreement,
        instance_node_agreement,
        same_instance_edge_agreement,
        1.0 if canonical_equal else 0.0,
    ]
    return {
        "score": min(components),
        "node_feature_agreement": node_agreement,
        "edge_feature_agreement": edge_agreement,
        "macro_probe_agreement": macro_agreement,
        "role_probe_agreement": role_agreement,
        "instance_node_probe_agreement": instance_node_agreement,
        "same_instance_edge_probe_agreement": same_instance_edge_agreement,
        "canonical_signature_equal": canonical_equal,
        "probe_kind": "fixed_deterministic_linear_heads_for_representation_invariance_only",
    }


def compare_partition_invariance(reference_graph: dict[str, Any], partitioned_graph: dict[str, Any]) -> dict[str, Any]:
    reference_signature = canonical_graph_signature(reference_graph)
    partitioned_signature = canonical_graph_signature(partitioned_graph)
    equal = reference_signature == partitioned_signature
    return {
        "score": 1.0 if equal else 0.0,
        "canonical_signature_equal": equal,
        "reference_face_count": len(reference_graph.get("nodes", [])),
        "partitioned_face_count": len(partitioned_graph.get("nodes", [])),
        "reference_patch_count": len(reference_graph.get("canonical_patches", [])),
        "partitioned_patch_count": len(partitioned_graph.get("canonical_patches", [])),
    }


def _feature_agreement(first: list[list[float]], second: list[list[float]], tolerance: float) -> float:
    if len(first) != len(second):
        return 0.0
    total = 0
    matched = 0
    for first_row, second_row in zip(first, second):
        if len(first_row) != len(second_row):
            return 0.0
        for first_value, second_value in zip(first_row, second_row):
            total += 1
            if abs(float(first_value) - float(second_value)) <= tolerance:
                matched += 1
    return matched / total if total else 1.0


def _probe_predictions(rows: list[list[float]], *, class_count: int, phase: float) -> list[int]:
    predictions = []
    for row in rows:
        logits = []
        for class_index in range(class_count):
            score = 0.0
            for feature_index, value in enumerate(row):
                weight = math.sin((feature_index + 1) * (class_index + 1) * phase)
                score += float(value) * weight
            logits.append(score)
        predictions.append(max(range(class_count), key=lambda index: logits[index]) if logits else 0)
    return predictions


def _prediction_agreement(first: list[int], second: list[int]) -> float:
    if len(first) != len(second):
        return 0.0
    return sum(a == b for a, b in zip(first, second)) / len(first) if first else 1.0


__all__ = ["compare_graph_v2_invariance", "compare_partition_invariance"]
