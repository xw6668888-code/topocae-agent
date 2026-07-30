from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import joblib

from build_rib_boundary_classifier_dataset import (
    NOT_RIB_MEMBER_LABEL,
    RIB_MEMBER_LABEL,
    adjacent_faces,
    extract_rib_boundary_features,
)
from utils.topology_postprocess import (
    demote_faces_to_stock,
    remap_faces_to_macro,
    repair_rib_semantics,
)


EXPANDABLE_MACROS = {"stock", "pocket_rect", "pocket_circular", "web"}


def load_rib_boundary_classifier_bundle(path: Path) -> dict:
    return joblib.load(path)


def apply_rib_boundary_classifier_postprocess(
    prediction_data: dict,
    graph_data: dict,
    classifier_bundle: dict,
    *,
    min_confidence: float = 0.8,
) -> dict:
    corrected = deepcopy(prediction_data)
    faces = corrected.get("faces", [])
    metadata = classifier_bundle["metadata"]
    estimator = classifier_bundle["estimator"]
    feature_schema = list(metadata["feature_schema"])
    id_to_label = {int(key): value for key, value in metadata["id_to_label"].items()}

    records = []
    decision_records = []
    pruned_faces: set[int] = set()
    expanded_faces: set[int] = set()

    for component in rib_components(faces, graph_data):
        component_set = set(component)
        if len(component) >= 3:
            for candidate_face_idx in component:
                seed_faces = sorted(component_set - {candidate_face_idx})
                classification = classify_boundary_candidate(
                    graph_data,
                    seed_faces,
                    candidate_face_idx,
                    estimator=estimator,
                    feature_schema=feature_schema,
                    id_to_label=id_to_label,
                )
                decision = make_decision_record(
                    classification,
                    stage="prune",
                    action=decision_action(
                        expected_label=NOT_RIB_MEMBER_LABEL,
                        classification=classification,
                        min_confidence=min_confidence,
                    ),
                    seed_face_indices=seed_faces,
                    candidate_face_idx=candidate_face_idx,
                )
                decision_records.append(decision)
                if classification["label"] != NOT_RIB_MEMBER_LABEL or classification["confidence"] < min_confidence:
                    continue
                demote_faces_to_stock(
                    faces,
                    [candidate_face_idx],
                    ["rib_boundary_classifier_pruned_non_member"],
                )
                pruned_faces.add(candidate_face_idx)
                component_set.discard(candidate_face_idx)
                decision["action"] = "pruned_to_stock"
                records.append(decision)

        seed_faces = sorted(component_set)
        if len(seed_faces) < 2:
            continue
        for candidate_face_idx in sorted(adjacent_faces(graph_data, component_set) - component_set):
            if candidate_face_idx in pruned_faces or candidate_face_idx in expanded_faces:
                continue
            candidate_face = face_by_idx(faces).get(candidate_face_idx)
            if candidate_face is None:
                continue
            if str(candidate_face.get("macro_feature")) not in EXPANDABLE_MACROS:
                decision_records.append(
                    {
                        "stage": "expand",
                        "action": "skipped_non_expandable_macro",
                        "seed_face_indices": seed_faces,
                        "candidate_face_idx": candidate_face_idx,
                        "candidate_macro_feature": candidate_face.get("macro_feature"),
                    }
                )
                continue
            classification = classify_boundary_candidate(
                graph_data,
                seed_faces,
                candidate_face_idx,
                estimator=estimator,
                feature_schema=feature_schema,
                id_to_label=id_to_label,
            )
            decision = make_decision_record(
                classification,
                stage="expand",
                action=decision_action(
                    expected_label=RIB_MEMBER_LABEL,
                    classification=classification,
                    min_confidence=min_confidence,
                ),
                seed_face_indices=seed_faces,
                candidate_face_idx=candidate_face_idx,
            )
            decision_records.append(decision)
            if classification["label"] != RIB_MEMBER_LABEL or classification["confidence"] < min_confidence:
                continue
            remap_faces_to_macro(
                faces,
                [candidate_face_idx],
                "rib",
                graph_data,
                ["rib_boundary_classifier_expanded_member"],
            )
            expanded_faces.add(candidate_face_idx)
            component_set.add(candidate_face_idx)
            decision["action"] = "expanded_to_rib"
            records.append(decision)

        if component_set:
            repair_rib_semantics(faces, sorted(component_set), graph_data)

    corrected["rib_boundary_classifier_postprocess"] = {
        "enabled": True,
        "candidate_count": len(records),
        "pruned_count": len(pruned_faces),
        "expanded_count": len(expanded_faces),
        "min_confidence": min_confidence,
        "records": records,
        "decision_count": len(decision_records),
        "decision_records": decision_records,
    }
    return corrected


def decision_action(
    *,
    expected_label: str,
    classification: dict,
    min_confidence: float,
) -> str:
    if classification["label"] != expected_label:
        return "skipped_label_mismatch"
    if classification["confidence"] < min_confidence:
        return "skipped_low_confidence"
    return "accepted"


def make_decision_record(
    classification: dict,
    *,
    stage: str,
    action: str,
    seed_face_indices: list[int],
    candidate_face_idx: int,
) -> dict:
    return {
        **classification,
        "stage": stage,
        "action": action,
        "seed_face_indices": seed_face_indices,
        "candidate_face_idx": candidate_face_idx,
    }


def classify_boundary_candidate(
    graph_data: dict,
    seed_face_indices: list[int],
    candidate_face_idx: int,
    *,
    estimator,
    feature_schema: list[str],
    id_to_label: dict[int, str],
) -> dict:
    features = extract_rib_boundary_features(graph_data, seed_face_indices, int(candidate_face_idx))
    feature_vector = [float(features.get(feature_name, 0.0)) for feature_name in feature_schema]
    predicted_id = int(estimator.predict([feature_vector])[0])
    return {
        "label": id_to_label.get(predicted_id, NOT_RIB_MEMBER_LABEL),
        "label_id": predicted_id,
        "confidence": prediction_confidence(estimator, feature_vector, predicted_id),
        "features": features,
    }


def prediction_confidence(estimator, feature_vector: list[float], predicted_id: int) -> float:
    if not hasattr(estimator, "predict_proba"):
        return 1.0
    probabilities = estimator.predict_proba([feature_vector])[0]
    classes = [int(item) for item in getattr(estimator, "classes_", range(len(probabilities)))]
    for index, class_id in enumerate(classes):
        if class_id == predicted_id:
            return float(probabilities[index])
    return float(max(probabilities)) if len(probabilities) else 0.0


def rib_components(faces: list[dict], graph_data: dict) -> list[list[int]]:
    rib_faces = {
        int(face["idx"])
        for face in faces
        if str(face.get("macro_feature")) == "rib"
    }
    if not rib_faces:
        return []
    adjacency = {face_idx: set() for face_idx in rib_faces}
    for edge in graph_data.get("edges", []) or []:
        if not isinstance(edge, list) or len(edge) != 2:
            continue
        src, dst = int(edge[0]), int(edge[1])
        if src in rib_faces and dst in rib_faces:
            adjacency.setdefault(src, set()).add(dst)
            adjacency.setdefault(dst, set()).add(src)

    visited = set()
    components = []
    for start in sorted(rib_faces):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency.get(current, set())):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))
    return components


def face_by_idx(faces: list[dict]) -> dict[int, dict]:
    return {int(face["idx"]): face for face in faces}
