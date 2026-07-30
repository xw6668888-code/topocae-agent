from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from build_rib_boundary_classifier_dataset import adjacent_faces, iter_edge_attributes
from utils.rib_candidate_features import extract_rib_candidate_features


RIB_CANDIDATE_LABEL = "rib_candidate"
NOT_RIB_CANDIDATE_LABEL = "not_rib_candidate"
HARDCASE_POSITIVE_SOURCE = "colored_rib_hardcase_positive"
HARDCASE_NEGATIVE_SOURCE = "colored_rib_hardcase_negative"
HOLE_MACROS = {
    "through_hole_round",
    "blind_hole_round",
    "stepped_hole_round",
    "counterbore_hole",
    "countersink_hole",
    "obround_hole",
}


def build_colored_rib_hardcase_records(
    *,
    review_template: dict[str, Any],
    colored_diagnostics: dict[str, Any],
    root_dir: Path,
    prediction_dir: Path,
    positive_weight: float = 5.0,
    negative_weight: float = 3.0,
    max_negatives_per_sample: int = 120,
) -> list[dict[str, Any]]:
    samples_by_id = {
        str(sample.get("sample_id")): sample
        for sample in review_template.get("samples", []) or []
        if sample.get("sample_id")
    }
    diagnostics_by_id = {
        str(sample.get("sample_id")): sample
        for sample in colored_diagnostics.get("samples", []) or []
        if sample.get("sample_id")
    }
    records: list[dict[str, Any]] = []
    for sample_id, diagnostic in sorted(diagnostics_by_id.items()):
        sample = samples_by_id.get(sample_id)
        if sample is None:
            continue
        graph_path = resolve_path(root_dir, sample.get("graph_path"))
        prediction_path = prediction_dir / f"{sample_id}.json"
        if graph_path is None or not graph_path.exists() or not prediction_path.exists():
            continue
        graph_data = load_json(graph_path)
        prediction_data = load_json(prediction_path)
        prediction_by_idx = prediction_face_by_idx(prediction_data)
        rib_faces = {int(face_idx) for face_idx in diagnostic.get("colored_rib_faces", []) or []}
        background_faces = {int(face_idx) for face_idx in diagnostic.get("colored_background_faces", []) or []}
        if not rib_faces and not background_faces:
            continue
        instance_by_face = reviewed_rib_instance_by_face(sample)
        instance_sizes = instance_face_counts(instance_by_face)

        for face_idx in sorted(rib_faces):
            records.append(
                make_hardcase_record(
                    sample_id=sample_id,
                    graph_path=graph_path,
                    graph_data=graph_data,
                    prediction_by_idx=prediction_by_idx,
                    face_idx=face_idx,
                    label=RIB_CANDIDATE_LABEL,
                    source=HARDCASE_POSITIVE_SOURCE,
                    sample_weight=positive_weight,
                    hard_case_types=positive_hard_case_types(
                        graph_data=graph_data,
                        prediction_by_idx=prediction_by_idx,
                        face_idx=face_idx,
                        instance_size=instance_sizes.get(instance_by_face.get(face_idx, ""), len(rib_faces)),
                        rib_faces=rib_faces,
                    ),
                    rib_instance_id=instance_by_face.get(face_idx, ""),
                )
            )

        for face_idx in select_negative_faces(
            graph_data=graph_data,
            prediction_by_idx=prediction_by_idx,
            rib_faces=rib_faces,
            background_faces=background_faces,
            max_negatives=max_negatives_per_sample,
        ):
            records.append(
                make_hardcase_record(
                    sample_id=sample_id,
                    graph_path=graph_path,
                    graph_data=graph_data,
                    prediction_by_idx=prediction_by_idx,
                    face_idx=face_idx,
                    label=NOT_RIB_CANDIDATE_LABEL,
                    source=HARDCASE_NEGATIVE_SOURCE,
                    sample_weight=negative_weight,
                    hard_case_types=negative_hard_case_types(
                        graph_data=graph_data,
                        prediction_by_idx=prediction_by_idx,
                        face_idx=face_idx,
                        rib_faces=rib_faces,
                    ),
                    rib_instance_id="",
                )
            )
    return records


def make_hardcase_record(
    *,
    sample_id: str,
    graph_path: Path,
    graph_data: dict[str, Any],
    prediction_by_idx: dict[int, dict[str, Any]],
    face_idx: int,
    label: str,
    source: str,
    sample_weight: float,
    hard_case_types: list[str],
    rib_instance_id: str,
) -> dict[str, Any]:
    prediction_face = prediction_by_idx.get(int(face_idx), {})
    return {
        "sample_id": sample_id,
        "graph_path": str(graph_path),
        "source": source,
        "label": label,
        "candidate_face_idx": int(face_idx),
        "rib_instance_id": rib_instance_id,
        "sample_weight": float(sample_weight),
        "hard_case_types": sorted(set(hard_case_types)),
        "prediction_macro_feature": prediction_face.get("macro_feature", ""),
        "features": extract_rib_candidate_features(
            graph_data,
            int(face_idx),
            prediction_face=prediction_face,
        ),
    }


def select_negative_faces(
    *,
    graph_data: dict[str, Any],
    prediction_by_idx: dict[int, dict[str, Any]],
    rib_faces: set[int],
    background_faces: set[int],
    max_negatives: int,
) -> list[int]:
    if not background_faces:
        return []
    rib_neighbors = adjacent_faces(graph_data, rib_faces) if rib_faces else set()
    ranked = []
    for face_idx in sorted(background_faces - rib_faces):
        prediction_macro = str(prediction_by_idx.get(face_idx, {}).get("macro_feature", ""))
        priority = 9
        if face_idx in rib_neighbors:
            priority = min(priority, 0)
        if prediction_macro == "rib":
            priority = min(priority, 1)
        if prediction_macro in {"stock", "web", "pocket_rect", "pocket_circular", "rectangular_boss"}:
            priority = min(priority, 3)
        if priority < 9:
            ranked.append((priority, face_idx))
    ranked.sort()
    return [face_idx for _priority, face_idx in ranked[:max_negatives]]


def positive_hard_case_types(
    *,
    graph_data: dict[str, Any],
    prediction_by_idx: dict[int, dict[str, Any]],
    face_idx: int,
    instance_size: int,
    rib_faces: set[int],
) -> list[str]:
    prediction_macro = str(prediction_by_idx.get(face_idx, {}).get("macro_feature", ""))
    types = ["colored_rib_positive"]
    if instance_size >= 4:
        types.append("complex_multi_face_rib")
    if prediction_macro != "rib":
        types.append("missed_or_non_rib_prediction")
    if has_circle_edge(graph_data, face_idx) or adjacent_to_macro(prediction_by_idx, graph_data, face_idx, HOLE_MACROS):
        types.append("hole_cut_rib")
    if adjacent_to_macro(prediction_by_idx, graph_data, face_idx, {"pocket_rect", "pocket_circular"}):
        types.append("pocket_inner_rib")
    if len(rib_faces) >= 8:
        types.append("multi_instance_or_dense_rib_context")
    return types


def negative_hard_case_types(
    *,
    graph_data: dict[str, Any],
    prediction_by_idx: dict[int, dict[str, Any]],
    face_idx: int,
    rib_faces: set[int],
) -> list[str]:
    prediction_macro = str(prediction_by_idx.get(face_idx, {}).get("macro_feature", ""))
    types = ["colored_background_negative"]
    if face_idx in adjacent_faces(graph_data, rib_faces) if rib_faces else False:
        types.append("adjacent_non_rib_negative")
    if prediction_macro == "rib":
        types.append("false_rib_negative")
    if prediction_macro in {"pocket_rect", "pocket_circular"}:
        types.append("pocket_boundary_negative")
    if prediction_macro in {"rectangular_boss", "web"}:
        types.append("thin_or_boss_like_negative")
    return types


def merge_rib_candidate_records(base_records: list[dict[str, Any]], hard_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    for record in list(base_records) + list(hard_records):
        key = (str(record.get("sample_id")), int(record.get("candidate_face_idx")))
        if key not in merged:
            merged[key] = normalize_record(record)
            continue
        existing = merged[key]
        if is_hardcase(record) and existing.get("label") != record.get("label"):
            merged[key] = normalize_record(record)
            continue
        if existing.get("label") == record.get("label"):
            existing["sample_weight"] = max(float(existing.get("sample_weight", 1.0)), float(record.get("sample_weight", 1.0)))
            existing["hard_case_types"] = sorted(set(existing.get("hard_case_types", [])) | set(record.get("hard_case_types", [])))
            existing["source"] = merge_source(existing.get("source", ""), record.get("source", ""))
    return sorted(merged.values(), key=lambda item: (item["sample_id"], int(item["candidate_face_idx"]), item["label"]))


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    normalized.setdefault("sample_weight", 1.0)
    normalized.setdefault("hard_case_types", [])
    return normalized


def is_hardcase(record: dict[str, Any]) -> bool:
    return str(record.get("source", "")).startswith("colored_rib_hardcase")


def merge_source(left: str, right: str) -> str:
    parts = []
    for value in [left, right]:
        for part in str(value).split("+"):
            if part and part not in parts:
                parts.append(part)
    return "+".join(parts)


def reviewed_rib_instance_by_face(sample: dict[str, Any]) -> dict[int, str]:
    mapping = {}
    for candidate in sample.get("candidate_faces", []) or []:
        review = candidate.get("review", {}) or {}
        if review.get("is_rib_face") is True:
            mapping[int(candidate["face_idx"])] = str(review.get("rib_instance_id") or "rib_1")
    return mapping


def instance_face_counts(instance_by_face: dict[int, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for instance_id in instance_by_face.values():
        counts[instance_id] = counts.get(instance_id, 0) + 1
    return counts


def adjacent_to_macro(
    prediction_by_idx: dict[int, dict[str, Any]],
    graph_data: dict[str, Any],
    face_idx: int,
    macros: set[str],
) -> bool:
    for neighbor in adjacent_faces(graph_data, {int(face_idx)}):
        if str(prediction_by_idx.get(neighbor, {}).get("macro_feature", "")) in macros:
            return True
    return False


def has_circle_edge(graph_data: dict[str, Any], face_idx: int) -> bool:
    face_idx = int(face_idx)
    for edge in iter_edge_attributes(graph_data):
        if int(edge.get("source_face", -1)) != face_idx and int(edge.get("target_face", -1)) != face_idx:
            continue
        if str(edge.get("curve_type", "")).lower() == "circle":
            return True
    return False


def prediction_face_by_idx(prediction_data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(face.get("idx", idx)): face for idx, face in enumerate(prediction_data.get("faces", []) or [])}


def resolve_path(root_dir: Path, value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (root_dir / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))
