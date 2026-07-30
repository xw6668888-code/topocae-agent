"""Frozen P1-C clean-context Graph-v2 -> feature-object inference flow.

The module is intentionally independent of dataset truth and sample metadata.  It
uses the P1-B architecture, the frozen P1-C candidate representation, the generic
oriented-cylinder polarity verifier, and the iteration-2 Synthetic-Val global
selection.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from typing import Any, Callable, Mapping

from models.p1_b_candidate_topology_classifier import (
    CandidateTopologyClassifier,
    predict_candidate_record,
)
from utils.p1_b_candidate_proposer import generate_candidate_proposals
from utils.p1_b_global_decode_v2 import decode_feature_objects
from utils.p1_b_instance_assembly import assemble_same_instance_candidates
from utils.p1_c_candidate_features_polarity_v1 import build_candidate_tensor_record
from utils.p1_c_topology_verifier_polarity_v1 import verify_candidate_prediction


PIPELINE_SCHEMA_VERSION = "topocae_p1_c_general_recognition_pipeline_1"
REQUIRED_GRAPH_SCHEMA = "topocae_graph_v2_relative_1_1"
REQUIRED_POLARITY_REVISION = "p1_c_oriented_cylinder_polarity_v1"


def run_p1_c_general_recognition(
    graph_v2: Mapping[str, Any],
    face_gnn_prediction: Mapping[str, Any],
    candidate_model: CandidateTopologyClassifier
    | Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
    *,
    same_instance_thresholds: Mapping[str, Any],
    candidate_thresholds: Mapping[str, Any],
    global_candidate_decode_selection: Mapping[str, Any],
    device: str = "cpu",
) -> dict[str, Any]:
    """Run the frozen P1-C pipeline without exposing truth to inference code."""

    if graph_v2.get("schema_version") != REQUIRED_GRAPH_SCHEMA:
        raise ValueError("P1-C frozen pipeline requires Graph v2.1")
    protocol = graph_v2.get("graph_v2_protocol", {})
    if protocol.get("oriented_cylinder_polarity_revision") != REQUIRED_POLARITY_REVISION:
        raise ValueError("P1-C frozen pipeline requires generic oriented-cylinder polarity")
    _require_synthetic_val_selection(same_instance_thresholds, "same-instance")
    _require_synthetic_val_selection(candidate_thresholds, "candidate")
    _require_iteration2_selection(global_candidate_decode_selection)

    graph = strip_inference_truth(graph_v2)
    prediction = strip_prediction_truth(face_gnn_prediction)
    selected = global_candidate_decode_selection["selected"]
    thresholds = dict(candidate_thresholds)
    thresholds["min_candidate_confidence"] = float(selected["min_candidate_confidence"])

    raw_face_macros = _raw_face_macros(prediction)
    assembly = assemble_same_instance_candidates(
        graph,
        prediction,
        global_threshold=float(same_instance_thresholds["global_threshold"]),
        threshold_selection_sha256=str(same_instance_thresholds["selection_sha256"]),
    )
    proposal_result = generate_candidate_proposals(graph, prediction, assembly)
    tensor_records: list[dict[str, Any]] = []
    model_predictions: list[dict[str, Any]] = []
    verified: list[dict[str, Any]] = []
    for proposal in proposal_result["proposals"]:
        tensor_record = build_candidate_tensor_record(graph, prediction, proposal)
        if isinstance(candidate_model, CandidateTopologyClassifier):
            model_prediction = predict_candidate_record(candidate_model, tensor_record, device=device)
        elif callable(candidate_model):
            model_prediction = dict(candidate_model(tensor_record, proposal))
        else:
            raise TypeError("candidate_model must be CandidateTopologyClassifier or predictor callable")
        model_prediction.setdefault("candidate_id", proposal.get("candidate_id"))
        model_prediction.setdefault("face_indices", proposal.get("face_indices", []))
        model_prediction["face_macro_predictions"] = raw_face_macros
        verification = verify_candidate_prediction(
            graph,
            proposal,
            model_prediction,
            thresholds=thresholds,
        )
        tensor_records.append(tensor_record)
        model_predictions.append(model_prediction)
        verified.append(verification)

    decoded = decode_feature_objects(
        graph,
        verified,
        subset_confidence_tolerance=float(selected["subset_confidence_tolerance"]),
        selection_sha256=str(global_candidate_decode_selection["selection_sha256"]),
    )
    manifest_payload = {
        "graph_schema": graph.get("schema_version"),
        "oriented_cylinder_polarity_revision": REQUIRED_POLARITY_REVISION,
        "same_instance_threshold_selection_sha256": same_instance_thresholds["selection_sha256"],
        "candidate_threshold_selection_sha256": candidate_thresholds["selection_sha256"],
        "global_candidate_decode_selection_sha256": global_candidate_decode_selection["selection_sha256"],
        "min_candidate_confidence": thresholds["min_candidate_confidence"],
        "global_subset_confidence_tolerance": selected["subset_confidence_tolerance"],
        "topology_verifier_schema": "topocae_p1_c_topology_verifier_polarity_fusion_1",
        "global_decode_schema": "topocae_p1_b_global_decode_2",
        "grouping_method": decoded["grouping_method"],
    }
    canonical = json.dumps(manifest_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "sample_id": face_gnn_prediction.get("sample_id", graph_v2.get("sample_id")),
        "graph_schema": graph.get("schema_version"),
        "assembly": assembly,
        "candidate_proposals": proposal_result,
        "candidate_tensor_records": tensor_records,
        "candidate_model_predictions": model_predictions,
        "verified_candidates": verified,
        "decoded_feature_objects": decoded,
        "pipeline_manifest": {
            **manifest_payload,
            "pipeline_sha256": sha256(canonical.encode("utf-8")).hexdigest().lower(),
            "truth_used_during_inference": False,
            "sample_metadata_used_as_model_input": False,
            "rules_hard_relabel_classes": False,
            "real_part_specific_thresholds": False,
            "per_class_thresholds": False,
            "llm_step_recognition": False,
        },
    }


def strip_inference_truth(graph_v2: Mapping[str, Any]) -> dict[str, Any]:
    """Remove labels, instance truth and generation metadata before inference."""

    graph = deepcopy(dict(graph_v2))
    for key in (
        "feature_sets",
        "object_truth",
        "labels",
        "face_labels",
        "primary_target",
        "context",
        "context_type",
        "host_family",
        "generator_revision",
        "seed",
        "source_path",
    ):
        graph.pop(key, None)
    for node in graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        for key in ("macro_feature", "semantic_role", "instance_id", "primary_secondary_role"):
            node.pop(key, None)
        attributes = node.get("attributes")
        if isinstance(attributes, dict):
            for key in ("macro_feature", "semantic_role", "instance_id", "truth"):
                attributes.pop(key, None)
    return graph


def strip_prediction_truth(prediction: Mapping[str, Any]) -> dict[str, Any]:
    """Retain predictions/embeddings while removing optional evaluation truth."""

    clean = deepcopy(dict(prediction))
    for key in ("truth", "feature_sets", "object_truth", "labels"):
        clean.pop(key, None)
    for face in clean.get("faces", []):
        if isinstance(face, dict):
            for key in ("truth_macro_feature", "truth_semantic_role", "truth_instance_id"):
                face.pop(key, None)
    return clean


def _require_synthetic_val_selection(selection: Mapping[str, Any], name: str) -> None:
    if str(selection.get("selection_split", "")) != "synthetic_val":
        raise ValueError(f"{name} selection must come from Synthetic-Val")
    if not selection.get("selection_sha256"):
        raise ValueError(f"{name} selection hash is required")


def _require_iteration2_selection(selection: Mapping[str, Any]) -> None:
    _require_synthetic_val_selection(selection, "global candidate/decode")
    if selection.get("schema_version") != "topocae_p1_c_stage6_iteration2_global_candidate_decode_selection_1":
        raise ValueError("unexpected P1-C global candidate/decode selection schema")
    selected = selection.get("selected", {})
    if not selected.get("absolute_object_gates_pass"):
        raise ValueError("frozen global candidate/decode selection did not pass absolute gates")
    if float(selected.get("worst_active_class_f1", 0.0)) < 0.8:
        raise ValueError("frozen global candidate/decode selection failed worst-class gate")


def _raw_face_macros(prediction: Mapping[str, Any]) -> dict[int, str]:
    return {
        int(face.get("idx", position)): str(face.get("macro_feature", face.get("raw_macro_feature", "stock")))
        for position, face in enumerate(prediction.get("faces", []))
        if isinstance(face, Mapping)
    }


__all__ = [
    "PIPELINE_SCHEMA_VERSION",
    "run_p1_c_general_recognition",
    "strip_inference_truth",
    "strip_prediction_truth",
]
