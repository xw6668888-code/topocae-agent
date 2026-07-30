"""Single P1-B Graph v2 -> instance -> candidate -> object inference flow."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Callable, Mapping

from models.p1_b_candidate_topology_classifier import CandidateTopologyClassifier, predict_candidate_record
from utils.p1_b_candidate_features import build_candidate_tensor_record
from utils.p1_b_candidate_proposer import generate_candidate_proposals
from utils.p1_b_global_decode import decode_feature_objects
from utils.p1_b_instance_assembly import assemble_same_instance_candidates
from utils.p1_b_topology_verifier import verify_candidate_prediction


PIPELINE_SCHEMA_VERSION = "topocae_p1_b_general_recognition_pipeline_1"


def run_p1_b_general_recognition(
    graph_v2: Mapping[str, Any],
    face_gnn_prediction: Mapping[str, Any],
    candidate_model: CandidateTopologyClassifier | Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
    *,
    same_instance_thresholds: Mapping[str, Any],
    candidate_thresholds: Mapping[str, Any],
    device: str = "cpu",
) -> dict[str, Any]:
    """Run the only supported final object construction path for P1-B."""

    if graph_v2.get("schema_version") != "topocae_graph_v2_relative_1_1":
        raise ValueError("P1-B pipeline requires Graph v2.1")
    if str(same_instance_thresholds.get("selection_split", "")) != "synthetic_val":
        raise ValueError("same-instance threshold must be selected on Synthetic-Val")
    if not same_instance_thresholds.get("selection_sha256"):
        raise ValueError("same-instance threshold selection hash is required")
    if str(candidate_thresholds.get("selection_split", "")) != "synthetic_val":
        raise ValueError("candidate thresholds must be selected on Synthetic-Val")
    if not candidate_thresholds.get("selection_sha256"):
        raise ValueError("candidate threshold selection hash is required")

    assembly = assemble_same_instance_candidates(
        graph_v2,
        face_gnn_prediction,
        global_threshold=float(same_instance_thresholds["global_threshold"]),
        threshold_selection_sha256=str(same_instance_thresholds["selection_sha256"]),
    )
    proposal_result = generate_candidate_proposals(graph_v2, face_gnn_prediction, assembly)
    tensor_records = []
    model_predictions = []
    verified = []
    for proposal in proposal_result["proposals"]:
        tensor_record = build_candidate_tensor_record(graph_v2, face_gnn_prediction, proposal)
        if isinstance(candidate_model, CandidateTopologyClassifier):
            prediction = predict_candidate_record(candidate_model, tensor_record, device=device)
        elif callable(candidate_model):
            prediction = dict(candidate_model(tensor_record, proposal))
        else:
            raise TypeError("candidate_model must be CandidateTopologyClassifier or a candidate predictor callable")
        prediction.setdefault("candidate_id", proposal.get("candidate_id"))
        prediction.setdefault("face_indices", proposal.get("face_indices", []))
        verification = verify_candidate_prediction(
            graph_v2,
            proposal,
            prediction,
            thresholds=candidate_thresholds,
        )
        tensor_records.append(tensor_record)
        model_predictions.append(prediction)
        verified.append(verification)
    decoded = decode_feature_objects(graph_v2, verified)
    manifest_payload = {
        "graph_schema": graph_v2.get("schema_version"),
        "same_instance_threshold_selection_sha256": same_instance_thresholds["selection_sha256"],
        "candidate_threshold_selection_sha256": candidate_thresholds["selection_sha256"],
        "candidate_model_schema": (
            candidate_model.metadata().get("schema_version")
            if isinstance(candidate_model, CandidateTopologyClassifier)
            else "external_candidate_predictor_callable"
        ),
        "grouping_method": decoded["grouping_method"],
    }
    canonical = json.dumps(manifest_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "sample_id": face_gnn_prediction.get("sample_id", graph_v2.get("sample_id")),
        "graph_schema": graph_v2.get("schema_version"),
        "assembly": assembly,
        "candidate_proposals": proposal_result,
        "candidate_tensor_records": tensor_records,
        "candidate_model_predictions": model_predictions,
        "verified_candidates": verified,
        "decoded_feature_objects": decoded,
        "pipeline_manifest": {
            **manifest_payload,
            "pipeline_sha256": sha256(canonical.encode("utf-8")).hexdigest().upper(),
            "legacy_same_macro_component_fallback": False,
            "truth_used_during_inference": False,
            "rules_hard_relabel_classes": False,
        },
    }


__all__ = ["PIPELINE_SCHEMA_VERSION", "run_p1_b_general_recognition"]
