from __future__ import annotations

"""Class-agnostic polarity-aware confidence fusion on top of verifier-v4."""

from typing import Any, Mapping

from p1_c.oriented_cylinder_polarity_v1 import FEATURE_NAME
import utils.p1_b_topology_verifier_v4 as verifier_v4


SCHEMA_VERSION = "topocae_p1_c_topology_verifier_polarity_fusion_1"


def verify_candidate_prediction(
    graph_data: Mapping[str, Any],
    proposal: Mapping[str, Any],
    model_prediction: Mapping[str, Any],
    *,
    thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(verifier_v4.verify_candidate_prediction(
        graph_data, proposal, model_prediction, thresholds=thresholds
    ))
    refined = [int(index) for index in result.get("refined_face_indices", [])]
    strength = polarity_strength(graph_data, refined)
    agreement = float(result.get("face_candidate_macro_agreement", 1.0) or 0.0)
    effective_agreement = agreement + strength * (1.0 - agreement)
    correction = 0.05 + 0.95 * effective_agreement
    before = float(result.get("final_confidence_before_face_consensus", result.get("final_confidence", 0.0)) or 0.0)
    corrected = before * correction
    reasons = set(str(reason) for reason in result.get("abstention_reasons", []))
    consensus_reason = "face_candidate_consensus_corrected_confidence_below_global_threshold"
    if corrected >= float(thresholds["min_candidate_confidence"]):
        reasons.discard(consensus_reason)
    else:
        reasons.add(consensus_reason)
    result["polarity_strength"] = strength
    result["polarity_effective_face_candidate_agreement"] = effective_agreement
    result["face_candidate_consensus_correction"] = correction
    result["final_confidence"] = corrected
    result["abstention_reasons"] = sorted(reasons)
    result["status"] = "abstain" if reasons else "accepted"
    result["abstain"] = result["status"] == "abstain"
    selected_macro = str(result.get("masked_top1", "unknown"))
    result["predicted_macro"] = selected_macro if result["status"] == "accepted" else "unknown"
    result["schema_version"] = SCHEMA_VERSION
    result["verifier_contract"] = {
        **dict(result.get("verifier_contract", {})),
        "polarity_fusion": "class_agnostic_absolute_mean_cylinder_polarity",
        "uses_class_names": False,
        "uses_truth": False,
        "uses_sample_name_or_area": False,
        "per_class_thresholds": False
    }
    return result


def polarity_strength(graph_data: Mapping[str, Any], face_indices: list[int]) -> float:
    nodes = {
        int(node.get("idx", position)): node
        for position, node in enumerate(graph_data.get("nodes", []))
    }
    values = []
    for index in face_indices:
        node = nodes.get(index, {})
        if node.get("primitive_surface") != "cylinder":
            continue
        raw = node.get("attributes", {}).get("recognition_features_v2", {}).get(FEATURE_NAME)
        if raw is None:
            raise RuntimeError(f"cylinder node lacks {FEATURE_NAME}")
        values.append(float(raw))
    if not values:
        return 0.0
    return max(0.0, min(1.0, abs(sum(values) / len(values))))


__all__ = ["SCHEMA_VERSION", "polarity_strength", "verify_candidate_prediction"]

