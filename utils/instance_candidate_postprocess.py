from __future__ import annotations

from copy import deepcopy
from numbers import Number
from pathlib import Path
from typing import Any

import joblib

from utils.instance_candidate_assembly import assemble_instance_candidates
from utils.instance_candidate_features import extract_instance_candidate_features
from utils.topology_postprocess import remap_faces_to_macro


REJECT_LABEL = "reject"


def load_instance_candidate_classifier_bundle(path: Path) -> dict:
    return joblib.load(path)


def classify_instance_candidate_proposals(
    graph_data: dict,
    prediction_data: dict,
    proposals: list[dict],
    classifier_bundle: dict,
) -> list[dict]:
    metadata = classifier_bundle["metadata"]
    estimator = classifier_bundle["estimator"]
    feature_schema = list(metadata["feature_schema"])
    id_to_label = _normalized_id_to_label(metadata["id_to_label"])

    records = []
    for proposal in proposals:
        face_indices = _face_indices(proposal)
        proposal_macro = proposal.get("proposal_macro")
        features = extract_instance_candidate_features(
            graph_data,
            proposal,
            prediction_faces=prediction_data.get("faces", []),
            candidate_macro=str(proposal_macro) if proposal_macro is not None else None,
        )
        feature_vector = [float(features.get(feature_name, 0.0)) for feature_name in feature_schema]
        diagnostic_features = _json_safe(features)
        predicted_label_id = int(estimator.predict([feature_vector])[0])
        predicted_label = str(id_to_label.get(predicted_label_id, REJECT_LABEL))
        confidence = _prediction_confidence(estimator, feature_vector, predicted_label_id)
        records.append(
            {
                "source": proposal.get("source"),
                "proposal_macro": proposal_macro,
                "predicted_label": predicted_label,
                "predicted_label_id": predicted_label_id,
                "confidence": confidence,
                "face_indices": face_indices,
                "features": diagnostic_features,
            }
        )
    return records


def apply_instance_candidate_postprocess(
    prediction_data: dict,
    graph_data: dict,
    classifier_bundle: dict,
    *,
    min_confidence: float = 0.5,
    strong_threshold: float = 0.55,
    weak_threshold: float = 0.35,
) -> dict:
    corrected = deepcopy(prediction_data)
    proposals = assemble_instance_candidates(
        graph_data,
        corrected,
        strong_threshold=strong_threshold,
        weak_threshold=weak_threshold,
    )
    records = [
        _diagnostic_record(record)
        for record in classify_instance_candidate_proposals(
            graph_data,
            corrected,
            proposals,
            classifier_bundle,
        )
    ]

    occupied_faces: set[int] = set()
    for record in sorted(records, key=lambda item: float(item["confidence"]), reverse=True):
        face_set = set(record["face_indices"])
        if record["predicted_label"] == REJECT_LABEL:
            record["action"] = "rejected_label"
            continue
        if float(record["confidence"]) < float(min_confidence):
            record["action"] = "skipped_low_confidence"
            continue
        overlap = sorted(face_set & occupied_faces)
        if overlap:
            record["action"] = "skipped_overlap"
            record["overlap_face_indices"] = overlap
            continue

        remapped_count = remap_faces_to_macro(
            corrected.get("faces", []),
            record["face_indices"],
            record["predicted_label"],
            graph_data,
            ["instance_candidate_classifier_remap"],
        )
        occupied_faces.update(face_set)
        record["action"] = "accepted_remap"
        record["remapped_face_count"] = remapped_count

    accepted_count = sum(1 for record in records if record["action"] == "accepted_remap")
    corrected["instance_candidate_postprocess"] = {
        "enabled": True,
        "candidate_count": len(records),
        "accepted_count": accepted_count,
        "rejected_count": len(records) - accepted_count,
        "min_confidence": min_confidence,
        "strong_threshold": strong_threshold,
        "weak_threshold": weak_threshold,
        "records": records,
    }
    return corrected


def _diagnostic_record(record: dict) -> dict:
    diagnostic = dict(record)
    diagnostic["classification"] = {
        "label": record["predicted_label"],
        "label_id": record["predicted_label_id"],
        "confidence": record["confidence"],
    }
    diagnostic["action"] = "unprocessed"
    return diagnostic


def _prediction_confidence(estimator: Any, feature_vector: list[float], predicted_label_id: int) -> float:
    if not hasattr(estimator, "predict_proba"):
        return 1.0
    probabilities = estimator.predict_proba([feature_vector])[0]
    classes = _estimator_classes(estimator, len(probabilities))
    for index, class_id in enumerate(classes):
        if _label_ids_equal(class_id, predicted_label_id):
            return float(probabilities[index])
    return float(max(probabilities)) if len(probabilities) else 0.0


def _estimator_classes(estimator: Any, probability_count: int) -> list[Any]:
    classes = getattr(estimator, "classes_", None)
    if classes is None:
        final_estimator = _final_estimator(estimator)
        classes = getattr(final_estimator, "classes_", None) if final_estimator is not None else None
    if classes is None:
        return list(range(probability_count))
    if hasattr(classes, "tolist"):
        classes = classes.tolist()
    return list(classes)


def _final_estimator(estimator: Any) -> Any | None:
    steps = getattr(estimator, "steps", None)
    if steps:
        return steps[-1][1]
    named_steps = getattr(estimator, "named_steps", None)
    if named_steps:
        values = list(named_steps.values())
        if values:
            return values[-1]
    return None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Number):
        return float(value)
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)

def _normalized_id_to_label(raw_id_to_label: dict) -> dict[int, str]:
    return {int(key): str(value) for key, value in raw_id_to_label.items()}


def _label_ids_equal(left: Any, right: int) -> bool:
    try:
        return int(left) == int(right)
    except (TypeError, ValueError):
        return left == right


def _face_indices(proposal: dict) -> list[int]:
    return sorted({int(idx) for idx in proposal.get("face_indices", [])})


__all__ = [
    "REJECT_LABEL",
    "apply_instance_candidate_postprocess",
    "classify_instance_candidate_proposals",
    "load_instance_candidate_classifier_bundle",
]




