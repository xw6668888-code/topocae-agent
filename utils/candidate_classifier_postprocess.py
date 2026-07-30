from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import joblib

from build_candidate_classifier_dataset import (
    build_prediction_component_proposals,
    build_rule_proposals,
    proposal_macro_features,
)
from utils.candidate_topology_features import extract_candidate_topology_features
from utils.topology_postprocess import demote_faces_to_stock, remap_faces_to_macro


REJECT_LABEL = "reject"


def load_candidate_classifier_bundle(path: Path) -> dict:
    return joblib.load(path)


def apply_candidate_classifier_postprocess(
    prediction_data: dict,
    graph_data: dict,
    classifier_bundle: dict,
    *,
    same_instance_threshold: float = 0.5,
    min_confidence: float = 0.5,
) -> dict:
    corrected = deepcopy(prediction_data)
    metadata = classifier_bundle["metadata"]
    estimator = classifier_bundle["estimator"]
    feature_schema = list(metadata["feature_schema"])
    id_to_label = {int(key): value for key, value in metadata["id_to_label"].items()}
    target_macros = target_macros_from_feature_schema(feature_schema)
    proposals = build_inference_candidate_proposals(
        graph_data,
        corrected,
        target_macros,
        same_instance_threshold=same_instance_threshold,
    )
    classified = classify_candidate_proposals(
        graph_data,
        corrected,
        proposals,
        estimator=estimator,
        feature_schema=feature_schema,
        id_to_label=id_to_label,
    )
    repair_records = apply_classified_candidate_repairs(
        corrected["faces"],
        graph_data,
        classified,
        min_confidence=min_confidence,
    )
    corrected["candidate_classifier_postprocess"] = {
        "enabled": True,
        "candidate_count": len(classified),
        "accepted_count": sum(1 for record in repair_records if record["action"] == "accepted_remap"),
        "rejected_prediction_component_count": sum(
            1 for record in repair_records if record["action"] == "rejected_prediction_component_to_stock"
        ),
        "min_confidence": min_confidence,
        "records": repair_records,
    }
    return corrected


def target_macros_from_feature_schema(feature_schema: list[str]) -> set[str]:
    prefix = "proposal_macro_"
    return {
        item[len(prefix):]
        for item in feature_schema
        if item.startswith(prefix)
    }


def build_inference_candidate_proposals(
    graph_data: dict,
    prediction_data: dict,
    target_macros: set[str],
    *,
    same_instance_threshold: float,
) -> list[dict]:
    proposals = []
    proposals.extend(
        build_prediction_component_proposals(
            graph_data,
            prediction_data,
            target_macros,
            same_instance_threshold=same_instance_threshold,
        )
    )
    proposals.extend(build_rule_proposals(graph_data, prediction_data, target_macros))
    seen = set()
    unique = []
    for proposal in proposals:
        face_indices = tuple(sorted(int(idx) for idx in proposal.get("face_indices", [])))
        key = (proposal.get("source"), proposal.get("proposal_macro"), face_indices)
        if not face_indices or key in seen:
            continue
        seen.add(key)
        unique.append({
            "source": proposal.get("source"),
            "proposal_macro": proposal.get("proposal_macro"),
            "face_indices": list(face_indices),
        })
    return unique


def classify_candidate_proposals(
    graph_data: dict,
    prediction_data: dict,
    proposals: list[dict],
    *,
    estimator,
    feature_schema: list[str],
    id_to_label: dict[int, str],
) -> list[dict]:
    records = []
    target_macros = target_macros_from_feature_schema(feature_schema)
    for proposal in proposals:
        features = extract_candidate_topology_features(
            graph_data,
            proposal["face_indices"],
            prediction_faces=prediction_data.get("faces", []),
            candidate_macro=proposal["proposal_macro"],
        )
        features.update(proposal_macro_features(proposal["proposal_macro"], target_macros))
        feature_vector = [float(features.get(feature_name, 0.0)) for feature_name in feature_schema]
        predicted_id = int(estimator.predict([feature_vector])[0])
        label = id_to_label.get(predicted_id, REJECT_LABEL)
        confidence = prediction_confidence(estimator, feature_vector, predicted_id)
        records.append({
            **proposal,
            "predicted_label": label,
            "predicted_label_id": predicted_id,
            "confidence": confidence,
            "features": features,
        })
    return records


def prediction_confidence(estimator, feature_vector: list[float], predicted_id: int) -> float:
    if not hasattr(estimator, "predict_proba"):
        return 1.0
    probabilities = estimator.predict_proba([feature_vector])[0]
    classes = [int(item) for item in getattr(estimator, "classes_", range(len(probabilities)))]
    for index, class_id in enumerate(classes):
        if class_id == predicted_id:
            return float(probabilities[index])
    return float(max(probabilities)) if len(probabilities) else 0.0


def apply_classified_candidate_repairs(
    faces: list[dict],
    graph_data: dict,
    classified: list[dict],
    *,
    min_confidence: float,
) -> list[dict]:
    records = []
    occupied_accept_faces: set[int] = set()
    rejected_prediction_components = [
        record
        for record in classified
        if record["source"] == "prediction_component"
        and record["predicted_label"] == REJECT_LABEL
        and record["confidence"] >= min_confidence
    ]
    for record in sorted(rejected_prediction_components, key=lambda item: item["confidence"], reverse=True):
        demote_faces_to_stock(
            faces,
            record["face_indices"],
            ["candidate_classifier_rejected_prediction_component"],
        )
        records.append(_repair_record(record, "rejected_prediction_component_to_stock"))

    accepted = [
        record
        for record in classified
        if record["predicted_label"] != REJECT_LABEL
        and record["confidence"] >= min_confidence
    ]
    for record in sorted(accepted, key=lambda item: item["confidence"], reverse=True):
        face_set = set(record["face_indices"])
        if face_set & occupied_accept_faces:
            continue
        remap_faces_to_macro(
            faces,
            record["face_indices"],
            record["predicted_label"],
            graph_data,
            ["candidate_classifier_remap"],
        )
        occupied_accept_faces.update(face_set)
        records.append(_repair_record(record, "accepted_remap"))
    return records


def _repair_record(record: dict, action: str) -> dict:
    return {
        "action": action,
        "source": record["source"],
        "proposal_macro": record["proposal_macro"],
        "predicted_label": record["predicted_label"],
        "confidence": record["confidence"],
        "face_indices": record["face_indices"],
        "face_count": len(record["face_indices"]),
    }
