from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Any


MEDIUM_STEEL_RGB = (0.596063172350652, 0.666666666666667, 0.686259250782025)


def is_background_color(color_name: str | None, rgb: list[float] | tuple[float, float, float] | None) -> bool:
    if color_name and color_name.strip().lower().replace("_", " ") == "medium steel":
        return True
    if rgb is None or len(rgb) < 3:
        return False
    return color_distance(rgb, MEDIUM_STEEL_RGB) <= 0.03


def color_distance(
    rgb_a: list[float] | tuple[float, float, float],
    rgb_b: list[float] | tuple[float, float, float],
) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(rgb_a[:3], rgb_b[:3])))


def color_key(color_name: str | None, rgb: list[float] | tuple[float, float, float] | None) -> str:
    if rgb is not None and len(rgb) >= 3:
        return ",".join(f"{float(channel):.4f}" for channel in rgb[:3])
    return (color_name or "unknown").strip().lower() or "unknown"


def apply_colored_rib_truth_to_template(
    template: dict[str, Any],
    colored_truth_by_sample: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    reviewed = copy.deepcopy(template)
    diagnostics = {
        "schema": "colored_rib_truth_diagnostics_v1",
        "samples": [],
    }

    for sample in reviewed.get("samples", []):
        sample_id = str(sample.get("sample_id", ""))
        colored_truth = colored_truth_by_sample.get(sample_id, {})
        rib_faces = {int(idx): info for idx, info in colored_truth.get("rib_faces", {}).items()}
        background_faces = {int(idx): info for idx, info in colored_truth.get("background_faces", {}).items()}
        instance_ids = _rib_instance_ids(rib_faces)
        candidate_faces = sample.setdefault("candidate_faces", [])
        original_candidate_indices = {int(candidate.get("face_idx")) for candidate in candidate_faces}
        appended_colored_rib_candidates = 0
        for face_idx in sorted(set(rib_faces) - original_candidate_indices):
            candidate_faces.append(
                {
                    "face_idx": face_idx,
                    "candidate_reasons": ["colored_step_rib_face_outside_template"],
                    "review": {"is_rib_face": None, "rib_instance_id": "", "notes": ""},
                }
            )
            appended_colored_rib_candidates += 1
        candidate_indices = {int(candidate.get("face_idx")) for candidate in candidate_faces}
        filled_true = 0
        filled_false = 0
        left_null = 0

        for candidate in candidate_faces:
            face_idx = int(candidate.get("face_idx"))
            review = candidate.setdefault("review", {})
            review.setdefault("notes", "")
            if face_idx in rib_faces:
                review["is_rib_face"] = True
                review["rib_instance_id"] = instance_ids[_face_color_key(rib_faces[face_idx])]
                review["notes"] = _append_note(review.get("notes", ""), "filled_from_colored_step")
                filled_true += 1
            elif face_idx in background_faces:
                review["is_rib_face"] = False
                review["rib_instance_id"] = ""
                review["notes"] = _append_note(review.get("notes", ""), "filled_from_colored_step_background")
                filled_false += 1
            else:
                review.setdefault("is_rib_face", None)
                review.setdefault("rib_instance_id", "")
                left_null += 1

        diagnostics["samples"].append(
            {
                "sample_id": sample_id,
                "candidate_count": len(candidate_indices),
                "original_candidate_count": len(original_candidate_indices),
                "appended_colored_rib_candidates": appended_colored_rib_candidates,
                "colored_rib_faces": sorted(rib_faces),
                "colored_background_faces": sorted(background_faces),
                "colored_rib_faces_outside_original_template": sorted(set(rib_faces) - original_candidate_indices),
                "colored_rib_faces_outside_template": sorted(set(rib_faces) - candidate_indices),
                "colored_background_faces_outside_template": sorted(set(background_faces) - candidate_indices),
                "filled_true": filled_true,
                "filled_false": filled_false,
                "left_null": left_null,
                "unmatched_colored_faces": colored_truth.get("unmatched_colored_faces", []),
                "ambiguous_colored_faces": colored_truth.get("ambiguous_colored_faces", []),
                "step_face_count": colored_truth.get("step_face_count"),
                "graph_face_count": colored_truth.get("graph_face_count"),
                "face_count_matches_graph": colored_truth.get("face_count_matches_graph"),
                "colored_override_count": colored_truth.get("colored_override_count"),
            }
        )

    reviewed["review_source"] = {
        "method": "colored_step_face_truth",
        "background_color": "Medium Steel",
        "rib_color_rule": "any non-background face color is treated as a rib/web instance",
    }
    return reviewed, diagnostics


def match_colored_step_faces_to_graph(
    colored_faces: list[dict[str, Any]],
    graph_data: dict[str, Any],
    *,
    max_score: float = 0.025,
    ambiguity_ratio: float = 1.25,
) -> dict[str, Any]:
    nodes = graph_data.get("nodes", [])
    rib_faces: dict[int, dict[str, Any]] = {}
    background_faces: dict[int, dict[str, Any]] = {}
    unmatched: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []

    for colored_face in colored_faces:
        ranked = sorted(
            (
                (_match_score(colored_face, node), node)
                for node in nodes
                if _primitive(colored_face) == _primitive(node)
            ),
            key=lambda item: item[0],
        )
        if not ranked or ranked[0][0] > max_score:
            unmatched.append(_compact_colored_face(colored_face, "no_graph_match"))
            continue
        if len(ranked) > 1 and ranked[1][0] <= max(ranked[0][0] * ambiguity_ratio, ranked[0][0] + 1e-6):
            ambiguous.append(
                {
                    **_compact_colored_face(colored_face, "ambiguous_graph_match"),
                    "candidate_face_indices": [int(ranked[0][1]["idx"]), int(ranked[1][1]["idx"])],
                    "scores": [round(ranked[0][0], 6), round(ranked[1][0], 6)],
                }
            )
            continue

        face_idx = int(ranked[0][1]["idx"])
        info = {
            "source_face_idx": colored_face.get("source_face_idx"),
            "color_name": colored_face.get("color_name", ""),
            "rgb": colored_face.get("rgb"),
            "color_key": color_key(colored_face.get("color_name"), colored_face.get("rgb")),
            "match_score": round(ranked[0][0], 8),
        }
        if is_background_color(colored_face.get("color_name"), colored_face.get("rgb")):
            background_faces[face_idx] = info
        else:
            rib_faces[face_idx] = info

    return {
        "rib_faces": rib_faces,
        "background_faces": background_faces,
        "unmatched_colored_faces": unmatched,
        "ambiguous_colored_faces": ambiguous,
    }


def extract_step_face_color_overrides(step_text: str) -> dict[str, Any]:
    entities, ordered_entity_ids = _parse_step_entities(step_text)
    advanced_face_ids = [
        entity_id
        for entity_id in ordered_entity_ids
        if _entity_type(entities.get(entity_id, "")) == "ADVANCED_FACE"
    ]
    face_index_by_entity = {entity_id: idx for idx, entity_id in enumerate(advanced_face_ids)}
    face_colors: dict[int, dict[str, Any]] = {}

    for entity_id in ordered_entity_ids:
        record = entities.get(entity_id, "")
        if _entity_type(record) != "OVER_RIDING_STYLED_ITEM":
            continue
        parsed = _parse_overriding_styled_item(record)
        if parsed is None:
            continue
        style_ids, target_id = parsed
        if target_id not in face_index_by_entity:
            continue
        color = None
        style_id_used = None
        for style_id in style_ids:
            color = _resolve_color_from_entity(style_id, entities, set())
            if color:
                style_id_used = style_id
                break
        if not color:
            continue
        face_idx = face_index_by_entity[target_id]
        face_colors[face_idx] = {
            "color_name": color["color_name"],
            "rgb": color["rgb"],
            "color_key": color_key(color["color_name"], color["rgb"]),
            "advanced_face_entity": target_id,
            "style_entity": style_id_used,
            "overriding_style_entity": entity_id,
        }

    return {
        "face_count": len(advanced_face_ids),
        "face_entity_ids": advanced_face_ids,
        "face_colors": face_colors,
    }


def build_colored_truth_from_step(step_path: Path, graph_data: dict[str, Any]) -> dict[str, Any]:
    extracted = extract_step_face_color_overrides(step_path.read_text(encoding="utf-8", errors="ignore"))
    nodes = graph_data.get("nodes", [])
    rib_faces: dict[int, dict[str, Any]] = {}
    background_faces: dict[int, dict[str, Any]] = {}

    for node in nodes:
        face_idx = int(node.get("idx"))
        info = extracted["face_colors"].get(face_idx)
        if info and not is_background_color(info.get("color_name"), info.get("rgb")):
            rib_faces[face_idx] = {
                **info,
                "source_face_idx": face_idx,
                "match_method": "advanced_face_order",
            }
        else:
            background_faces[face_idx] = {
                **(info or {}),
                "source_face_idx": face_idx,
                "color_name": (info or {}).get("color_name", "Medium Steel"),
                "color_key": (info or {}).get("color_key", color_key("Medium Steel", MEDIUM_STEEL_RGB)),
                "rgb": (info or {}).get("rgb", list(MEDIUM_STEEL_RGB)),
                "match_method": "advanced_face_order_default_background",
            }

    return {
        "rib_faces": rib_faces,
        "background_faces": background_faces,
        "unmatched_colored_faces": [],
        "ambiguous_colored_faces": [],
        "step_face_count": extracted["face_count"],
        "graph_face_count": len(nodes),
        "face_count_matches_graph": extracted["face_count"] == len(nodes),
        "colored_override_count": len(extracted["face_colors"]),
    }


def _parse_step_entities(step_text: str) -> tuple[dict[int, str], list[int]]:
    entities: dict[int, str] = {}
    ordered_ids: list[int] = []
    for match in re.finditer(r"#(\d+)\s*=\s*(.*?);", step_text, flags=re.DOTALL):
        entity_id = int(match.group(1))
        entities[entity_id] = " ".join(match.group(2).strip().split())
        ordered_ids.append(entity_id)
    return entities, ordered_ids


def _entity_type(record: str) -> str:
    match = re.match(r"([A-Z0-9_]+)\s*\(", record)
    return match.group(1) if match else ""


def _parse_overriding_styled_item(record: str) -> tuple[list[int], int] | None:
    match = re.match(r"OVER_RIDING_STYLED_ITEM\s*\([^,]*,\s*\((.*?)\)\s*,\s*#(\d+)\s*,", record)
    if not match:
        return None
    style_ids = [int(value) for value in re.findall(r"#(\d+)", match.group(1))]
    target_id = int(match.group(2))
    return style_ids, target_id


def _resolve_color_from_entity(entity_id: int, entities: dict[int, str], visited: set[int]) -> dict[str, Any] | None:
    if entity_id in visited:
        return None
    visited.add(entity_id)
    record = entities.get(entity_id, "")
    if not record:
        return None
    color = _parse_colour_rgb(record)
    if color:
        return color
    color = _parse_draughting_pre_defined_colour(record)
    if color:
        return color
    for ref_id in _entity_refs(record):
        color = _resolve_color_from_entity(ref_id, entities, visited)
        if color:
            return color
    return None


def _entity_refs(record: str) -> list[int]:
    return [int(value) for value in re.findall(r"#(\d+)", record)]


def _parse_colour_rgb(record: str) -> dict[str, Any] | None:
    number = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?"
    match = re.search(
        rf"COLOUR_RGB\s*\(\s*'([^']*)'\s*,\s*({number})\s*,\s*({number})\s*,\s*({number})\s*\)",
        record,
    )
    if not match:
        return None
    return {
        "color_name": match.group(1),
        "rgb": [float(match.group(2)), float(match.group(3)), float(match.group(4))],
    }


def _parse_draughting_pre_defined_colour(record: str) -> dict[str, Any] | None:
    match = re.search(r"DRAUGHTING_PRE_DEFINED_COLOUR\s*\(\s*'([^']*)'\s*\)", record)
    if not match:
        return None
    name = match.group(1).strip()
    rgb_by_name = {
        "red": [1.0, 0.0, 0.0],
        "green": [0.0, 1.0, 0.0],
        "blue": [0.0, 0.0, 1.0],
        "yellow": [1.0, 1.0, 0.0],
        "cyan": [0.0, 1.0, 1.0],
        "magenta": [1.0, 0.0, 1.0],
        "black": [0.0, 0.0, 0.0],
        "white": [1.0, 1.0, 1.0],
    }
    return {"color_name": name, "rgb": rgb_by_name.get(name.lower())}


def _rib_instance_ids(rib_faces: dict[int, dict[str, Any]]) -> dict[str, str]:
    color_keys = sorted({_face_color_key(info) for info in rib_faces.values()})
    return {key: f"rib_color_{idx + 1:02d}" for idx, key in enumerate(color_keys)}


def _face_color_key(info: dict[str, Any]) -> str:
    return str(info.get("color_key") or color_key(info.get("color_name"), info.get("rgb")))


def _append_note(existing: str, note: str) -> str:
    notes = [item for item in str(existing or "").split(";") if item]
    if note not in notes:
        notes.append(note)
    return ";".join(notes)


def _primitive(item: dict[str, Any]) -> str:
    return str(item.get("primitive_surface") or item.get("attributes", {}).get("surface_type_str", "")).lower()


def _match_score(colored_face: dict[str, Any], node: dict[str, Any]) -> float:
    colored_area = float(colored_face.get("area", 0.0) or 0.0)
    node_area = _node_area(node)
    area_score = abs(colored_area - node_area) / max(abs(colored_area), abs(node_area), 1.0)
    center_score = _center_distance(colored_face.get("center", []), _node_center(node)) / max(_model_scale(node), 1.0)
    specific_score = _specific_score(colored_face.get("specific", {}), node.get("attributes", {}).get("specific", {}))
    return area_score * 0.65 + center_score * 0.25 + specific_score * 0.10


def _node_area(node: dict[str, Any]) -> float:
    return float(node.get("attributes", {}).get("physical", {}).get("area", 0.0) or 0.0)


def _node_center(node: dict[str, Any]) -> list[float]:
    center = node.get("attributes", {}).get("physical", {}).get("center_of_mass", [])
    return [float(value) for value in center[:3]] if len(center) >= 3 else [0.0, 0.0, 0.0]


def _model_scale(node: dict[str, Any]) -> float:
    descriptors = node.get("attributes", {}).get("shape_descriptors", {})
    return float(descriptors.get("bbox_diagonal", 0.0) or descriptors.get("bbox_longest_extent", 0.0) or 100.0)


def _center_distance(a: list[float], b: list[float]) -> float:
    if len(a) < 3 or len(b) < 3:
        return 0.0
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a[:3], b[:3])))


def _specific_score(colored_specific: dict[str, Any], node_specific: dict[str, Any]) -> float:
    if "radius" in colored_specific and "radius" in node_specific:
        radius_a = float(colored_specific.get("radius") or 0.0)
        radius_b = float(node_specific.get("radius") or 0.0)
        return abs(radius_a - radius_b) / max(abs(radius_a), abs(radius_b), 1.0)
    for key in ("normal", "axis"):
        if key in colored_specific and key in node_specific:
            return _vector_angle_score(colored_specific.get(key), node_specific.get(key))
    return 0.0


def _vector_angle_score(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) < 3 or len(b) < 3:
        return 0.0
    na = _normalize(a)
    nb = _normalize(b)
    dot = abs(sum(x * y for x, y in zip(na, nb)))
    return 1.0 - max(-1.0, min(1.0, dot))


def _normalize(vector: list[float]) -> list[float]:
    length = math.sqrt(sum(float(value) ** 2 for value in vector[:3]))
    if length <= 1e-12:
        return [0.0, 0.0, 0.0]
    return [float(value) / length for value in vector[:3]]


def _compact_colored_face(colored_face: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "source_face_idx": colored_face.get("source_face_idx"),
        "color_name": colored_face.get("color_name", ""),
        "rgb": colored_face.get("rgb"),
        "primitive_surface": colored_face.get("primitive_surface"),
        "area": colored_face.get("area"),
        "center": colored_face.get("center"),
        "reason": reason,
    }


