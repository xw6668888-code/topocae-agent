"""Frozen P1-B Graph-v2 -> object inference flow used after training."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Callable, Mapping

from models.p1_b_candidate_topology_classifier import CandidateTopologyClassifier, predict_candidate_record
from utils.p1_b_candidate_features import build_candidate_tensor_record
from utils.p1_b_candidate_proposer import generate_candidate_proposals
from utils.p1_b_global_decode_v2 import decode_feature_objects
from utils.p1_b_instance_assembly import assemble_same_instance_candidates
from utils.p1_b_topology_verifier_v4 import verify_candidate_prediction


PIPELINE_SCHEMA_VERSION = "topocae_p1_b_general_recognition_pipeline_4"


def run_p1_b_general_recognition(
    graph_v2: Mapping[str, Any],
    face_gnn_prediction: Mapping[str, Any],
    candidate_model: CandidateTopologyClassifier | Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
    *,
    same_instance_thresholds: Mapping[str, Any],
    candidate_thresholds: Mapping[str, Any],
    global_decode_selection: Mapping[str, Any],
    device: str = "cpu",
) -> dict[str, Any]:
    if graph_v2.get("schema_version") != "topocae_graph_v2_relative_1_1":
        raise ValueError("P1-B final pipeline requires Graph v2.1")
    _require_synthetic_val_selection(same_instance_thresholds, "same-instance")
    _require_synthetic_val_selection(candidate_thresholds, "candidate")
    _require_synthetic_val_selection(global_decode_selection, "global-decode")
    raw_face_macros = _raw_face_macros(face_gnn_prediction)
    assembly = assemble_same_instance_candidates(
        graph_v2, face_gnn_prediction,
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
            raise TypeError("candidate_model must be CandidateTopologyClassifier or a predictor callable")
        prediction.setdefault("candidate_id", proposal.get("candidate_id"))
        prediction.setdefault("face_indices", proposal.get("face_indices", []))
        prediction["face_macro_predictions"] = raw_face_macros
        verification = verify_candidate_prediction(
            graph_v2, proposal, prediction, thresholds=candidate_thresholds,
        )
        tensor_records.append(tensor_record)
        model_predictions.append(prediction)
        verified.append(verification)
    decoded = decode_feature_objects(
        graph_v2, verified,
        subset_confidence_tolerance=float(global_decode_selection["global_subset_confidence_tolerance"]),
        selection_sha256=str(global_decode_selection["selection_sha256"]),
    )
    manifest_payload = {
        "graph_schema": graph_v2.get("schema_version"),
        "same_instance_threshold_selection_sha256": same_instance_thresholds["selection_sha256"],
        "candidate_threshold_selection_sha256": candidate_thresholds["selection_sha256"],
        "global_decode_selection_sha256": global_decode_selection["selection_sha256"],
        "candidate_model_schema": (
            candidate_model.metadata().get("schema_version")
            if isinstance(candidate_model, CandidateTopologyClassifier)
            else "external_candidate_predictor_callable"
        ),
        "topology_verifier_schema": "topocae_p1_b_topology_verifier_4",
        "global_decode_schema": "topocae_p1_b_global_decode_2",
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
            "real_part_specific_thresholds": False,
            "llm_step_recognition": False,
        },
    }


def _require_synthetic_val_selection(selection: Mapping[str, Any], name: str) -> None:
    if str(selection.get("selection_split", "")) != "synthetic_val":
        raise ValueError(f"{name} selection must come from Synthetic-Val")
    if not selection.get("selection_sha256"):
        raise ValueError(f"{name} selection hash is required")


def _raw_face_macros(prediction: Mapping[str, Any]) -> dict[int, str]:
    result = {}
    for position, face in enumerate(prediction.get("faces", [])):
        if isinstance(face, Mapping):
            result[int(face.get("idx", position))] = str(face.get("macro_feature", face.get("raw_macro_feature", "stock")))
    return result


__all__ = ["PIPELINE_SCHEMA_VERSION", "run_p1_b_general_recognition"]
