from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import joblib

from utils.rib_candidate_features import extract_rib_candidate_features
from utils.topology_postprocess import remap_faces_to_macro, repair_rib_semantics


RIB_CANDIDATE_LABEL = "rib_candidate"
NOT_RIB_CANDIDATE_LABEL = "not_rib_candidate"
PROMOTABLE_MACROS = {"stock", "web", "pocket_rect", "pocket_circular", "rectangular_boss"}
CONDITIONAL_PROMOTABLE_MACROS = {"fillet", "chamfer", "cylinder_boss"}
PROTECTED_MACROS = {
    "through_hole_round",
    "blind_hole_round",
    "stepped_hole_round",
    "counterbore_hole",
    "countersink_hole",
    "obround_hole",
    "slot_rect",
    "slot_u",
    "slot_v",
    "slot_o",
}


def load_rib_candidate_proposer_bundle(path: Path) -> dict:
    return joblib.load(path)


def apply_rib_candidate_proposer_postprocess(
    prediction_data: dict,
    graph_data: dict,
    classifier_bundle: dict,
    *,
    min_confidence: float = 0.75,
) -> dict:
    corrected = deepcopy(prediction_data)
    faces = corrected.get("faces", [])
    estimator = classifier_bundle["estimator"]
    metadata = classifier_bundle["metadata"]
    feature_schema = list(metadata["feature_schema"])
    id_to_label = {int(key): value for key, value in metadata["id_to_label"].items()}
    face_lookup = face_by_idx(faces)

    records = []
    promoted_faces = []
    skipped_protected_count = 0
    for face in faces:
        face_idx = int(face["idx"])
        macro = str(face.get("macro_feature"))
        classification = classify_rib_candidate(
            graph_data,
            face_idx,
            face,
            estimator=estimator,
            feature_schema=feature_schema,
            id_to_label=id_to_label,
        )
        decision = {
            **classification,
            "candidate_face_idx": face_idx,
            "candidate_macro_feature": macro,
            "action": "skipped_label_mismatch",
        }
        if macro in PROTECTED_MACROS:
            skipped_protected_count += 1
            decision["action"] = "skipped_protected_macro"
            records.append(decision)
            continue
        is_conditional_promotable = macro in CONDITIONAL_PROMOTABLE_MACROS
        if macro not in PROMOTABLE_MACROS and macro != "rib" and not is_conditional_promotable:
            decision["action"] = "skipped_non_promotable_macro"
            records.append(decision)
            continue
        if classification["label"] != RIB_CANDIDATE_LABEL:
            records.append(decision)
            continue
        if classification["confidence"] < min_confidence:
            decision["action"] = "skipped_low_confidence"
            records.append(decision)
            continue
        if is_conditional_promotable and not has_adjacent_rib_face(faces, graph_data, face_idx):
            decision["action"] = "skipped_conditional_without_rib_neighbor"
            records.append(decision)
            continue
        if macro == "rib":
            decision["action"] = "already_rib"
            records.append(decision)
            continue
        remap_faces_to_macro(
            faces,
            [face_idx],
            "rib",
            graph_data,
            ["rib_candidate_proposer_promoted_face"],
        )
        promoted_faces.append(face_idx)
        decision["action"] = "promoted_to_rib"
        records.append(decision)

    for component in rib_components(faces, graph_data):
        repair_rib_semantics(faces, component, graph_data)

    corrected["rib_candidate_proposer_postprocess"] = {
        "enabled": True,
        "min_confidence": min_confidence,
        "candidate_count": len(records),
        "promoted_count": len(promoted_faces),
        "promoted_faces": promoted_faces,
        "skipped_protected_count": skipped_protected_count,
        "records": records,
    }
    return corrected


def classify_rib_candidate(
    graph_data: dict,
    candidate_face_idx: int,
    prediction_face: dict,
    *,
    estimator,
    feature_schema: list[str],
    id_to_label: dict[int, str],
) -> dict:
    features = extract_rib_candidate_features(
        graph_data,
        candidate_face_idx,
        prediction_face=prediction_face,
    )
    feature_vector = [float(features.get(name, 0.0)) for name in feature_schema]
    predicted_id = int(estimator.predict([feature_vector])[0])
    return {
        "label": id_to_label.get(predicted_id, NOT_RIB_CANDIDATE_LABEL),
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
            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def has_adjacent_rib_face(faces: list[dict], graph_data: dict, face_idx: int) -> bool:
    rib_faces = {
        int(face["idx"])
        for face in faces
        if str(face.get("macro_feature")) == "rib"
    }
    if not rib_faces:
        return False
    face_idx = int(face_idx)
    for edge in graph_data.get("edges", []) or []:
        if not isinstance(edge, list) or len(edge) != 2:
            continue
        src, dst = int(edge[0]), int(edge[1])
        if src == face_idx and dst in rib_faces:
            return True
        if dst == face_idx and src in rib_faces:
            return True
    return False


def face_by_idx(faces: list[dict]) -> dict[int, dict]:
    return {int(face["idx"]): face for face in faces}


