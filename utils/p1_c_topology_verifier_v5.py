from __future__ import annotations

from typing import Any, Mapping

import utils.p1_b_topology_verifier_v3 as revision_3


SCHEMA_VERSION = "topocae_p1_c_topology_verifier_boundary_context_5"
select_global_candidate_thresholds = revision_3.select_global_candidate_thresholds


def verify_candidate_prediction(
    graph_data: Mapping[str, Any],
    proposal: Mapping[str, Any],
    model_prediction: Mapping[str, Any],
    *,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(
        revision_3.verify_candidate_prediction(
            graph_data, proposal, model_prediction, thresholds=thresholds
        )
    )
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
        if observations
        else 1.0
    )
    result["face_candidate_macro_agreement"] = agreement
    result["face_candidate_consensus_mode"] = "audit_only"
    result["final_confidence_before_face_consensus"] = float(
        result.get("final_confidence", 0.0) or 0.0
    )
    result["schema_version"] = SCHEMA_VERSION
    result["verifier_contract"] = {
        **dict(result.get("verifier_contract", {})),
        "candidate_macro_evidence": "boundary_context_candidate_classifier",
        "face_candidate_consensus": "audit_only_no_confidence_mutation",
        "hard_class_rewrite": False,
        "uses_truth": False,
        "uses_sample_name_or_area": False,
    }
    return result


__all__ = ["SCHEMA_VERSION", "select_global_candidate_thresholds", "verify_candidate_prediction"]
