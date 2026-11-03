"""P1-B verifier v4: universal face/candidate macro-consensus confidence.

The correction is category-agnostic.  It never changes a predicted class and
never uses truth, sample names, areas, or real-part identifiers.  It only
reduces the confidence of a candidate whose selected object macro disagrees
with the frozen face-GNN predictions on its refined member faces.
"""

from __future__ import annotations

from typing import Any, Mapping

import utils.p1_b_topology_verifier_v3 as revision_3


SCHEMA_VERSION = "topocae_p1_b_topology_verifier_4"
select_global_candidate_thresholds = revision_3.select_global_candidate_thresholds


def verify_candidate_prediction(
    graph_data: Mapping[str, Any],
    proposal: Mapping[str, Any],
    model_prediction: Mapping[str, Any],
    *,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(revision_3.verify_candidate_prediction(
        graph_data, proposal, model_prediction, thresholds=thresholds,
    ))
    raw = model_prediction.get("face_macro_predictions", {})
    if isinstance(raw, (list, tuple)):
        raw = {index: value for index, value in enumerate(raw)}
    selected_macro = str(result.get("masked_top1", "unknown"))
    refined = [int(index) for index in result.get("refined_face_indices", [])]
    observations = []
    if isinstance(raw, Mapping):
        for index in refined:
            value = raw.get(index, raw.get(str(index)))
            if value is not None:
                observations.append(str(value))
    agreement = (
        sum(value == selected_macro for value in observations) / len(observations)
        if observations else 1.0
    )
    correction = 0.05 + 0.95 * agreement
    corrected = float(result.get("final_confidence", 0.0) or 0.0) * correction
    result["face_candidate_macro_agreement"] = agreement
    result["face_candidate_consensus_correction"] = correction
    result["final_confidence_before_face_consensus"] = float(result.get("final_confidence", 0.0) or 0.0)
    result["final_confidence"] = corrected
    reasons = set(str(reason) for reason in result.get("abstention_reasons", []))
    if corrected < float(thresholds["min_candidate_confidence"]):
        reasons.add("face_candidate_consensus_corrected_confidence_below_global_threshold")
    result["abstention_reasons"] = sorted(reasons)
    result["status"] = "abstain" if reasons else "accepted"
    result["abstain"] = result["status"] == "abstain"
    result["predicted_macro"] = selected_macro if result["status"] == "accepted" else "unknown"
    result["schema_version"] = SCHEMA_VERSION
    result["verifier_contract"] = {
        **dict(result.get("verifier_contract", {})),
        "face_candidate_consensus": "category_agnostic_confidence_correction_only",
        "face_candidate_consensus_floor": 0.05,
        "hard_class_rewrite": False,
        "uses_truth": False,
        "uses_sample_name_or_area": False,
    }
    return result


__all__ = ["SCHEMA_VERSION", "select_global_candidate_thresholds", "verify_candidate_prediction"]
