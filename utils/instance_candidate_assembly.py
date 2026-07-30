from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


STOCK_MACROS = {"", "stock", "stock_face", "background", "unknown", "unknown_feature", None}
RIB_WEB_MACROS = {"rib", "web"}
SLOT_LIKE_MACROS = {"slot_u", "slot_o", "slot_rect", "slot_v"}
WEAK_EXPANSION_MACROS = RIB_WEB_MACROS | SLOT_LIKE_MACROS


class EdgeProbabilityLookup(dict):
    """Dictionary with normalized undirected edge keys."""

    def __getitem__(self, key: Any) -> float:
        return super().__getitem__(_edge_key(key))

    def __setitem__(self, key: Any, value: Any) -> None:
        normalized = _edge_key(key)
        probability = float(value)
        current = super().get(normalized)
        if current is None or probability > current:
            super().__setitem__(normalized, probability)

    def __contains__(self, key: object) -> bool:
        try:
            normalized = _edge_key(key)
        except (TypeError, ValueError):
            return False
        return super().__contains__(normalized)

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            normalized = _edge_key(key)
        except (TypeError, ValueError):
            return default
        return super().get(normalized, default)


def build_edge_probability_lookup(graph_data: dict[str, Any], prediction_data: dict[str, Any]) -> EdgeProbabilityLookup:
    """Build an undirected edge -> same-instance probability lookup.

    Prediction entries may be plain probabilities aligned to graph_data["edges"],
    or records that identify an edge by edge_index, src/dst, or face_pair.
    Missing edge probabilities are intentionally absent from the lookup.
    """

    graph_edges = _extract_graph_edges(graph_data)
    lookup = EdgeProbabilityLookup()
    raw_predictions = _first_present(
        prediction_data,
        (
            "edge_same_instance",
            "edge_same_instance_probabilities",
            "edge_same_instance_probability",
            "prediction_edge_same_instance",
        ),
    )
    if raw_predictions is None:
        return lookup

    raw_predictions = _to_builtin(raw_predictions)
    if isinstance(raw_predictions, Mapping) and "probabilities" in raw_predictions:
        raw_predictions = raw_predictions["probabilities"]

    if isinstance(raw_predictions, Mapping):
        for key, value in raw_predictions.items():
            edge = _edge_from_mapping_key(key, graph_edges)
            probability = _probability_from_record(value)
            if edge is not None and probability is not None:
                lookup[edge] = probability
        return lookup

    if _is_iterable(raw_predictions):
        for position, record in enumerate(raw_predictions):
            record = _to_builtin(record)
            edge = _edge_from_prediction_record(record, graph_edges, position)
            probability = _probability_from_record(record)
            if edge is not None and probability is not None:
                lookup[edge] = probability

    return lookup


def assemble_seed_components(
    graph_data: dict[str, Any],
    prediction_data: dict[str, Any],
    strong_threshold: float = 0.55,
    weak_threshold: float = 0.35,
    target_macros: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Assemble initial non-stock seed components using strong same-instance edges."""

    del weak_threshold  # Reserved for API symmetry with assemble_instance_candidates.
    graph_edges = _extract_graph_edges(graph_data)
    edge_lookup = build_edge_probability_lookup(graph_data, prediction_data)
    faces = _face_records(graph_data, prediction_data)
    target_set = set(target_macros) if target_macros is not None else None
    seed_indices = [
        idx
        for idx, record in faces.items()
        if _is_seed_macro(_macro(record)) and _macro_in_targets(_macro(record), target_set)
    ]
    seed_set = set(seed_indices)
    parent = {idx: idx for idx in seed_indices}

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for src, dst in graph_edges:
        probability = edge_lookup.get((src, dst))
        if probability is None or probability < strong_threshold:
            continue
        if src not in seed_set or dst not in seed_set:
            continue
        if _strong_macro_compatible(_macro(faces[src]), _macro(faces[dst])):
            union(src, dst)

    groups: dict[int, list[int]] = defaultdict(list)
    for idx in seed_indices:
        groups[find(idx)].append(idx)

    candidates = []
    for group in sorted(groups.values(), key=lambda values: (min(values), len(values))):
        face_indices = sorted(group)
        reasons = ["strong_same_instance_edge"] if len(face_indices) > 1 else ["single_face_seed"]
        candidates.append(
            _candidate_record(
                source="same_instance_seed_component",
                proposal_macro=_proposal_macro(face_indices, faces),
                face_indices=face_indices,
                seed_face_indices=face_indices,
                graph_edges=graph_edges,
                edge_lookup=edge_lookup,
                assembly_reasons=reasons,
            )
        )
    return candidates


def expand_weak_candidate_faces(
    candidates: list[dict[str, Any]],
    graph_data: dict[str, Any],
    prediction_data: dict[str, Any],
    edge_lookup: EdgeProbabilityLookup | None = None,
    weak_threshold: float = 0.35,
    target_macros: Iterable[str] | None = None,
    stock_area_guard_ratio: float = 0.75,
) -> list[dict[str, Any]]:
    """Add one-hop weakly compatible faces to rib/web and slot-like candidates."""

    graph_edges = _extract_graph_edges(graph_data)
    edge_attributes_by_key = _edge_attributes_by_key(graph_data)
    edge_lookup = edge_lookup if edge_lookup is not None else build_edge_probability_lookup(graph_data, prediction_data)
    faces = _face_records(graph_data, prediction_data)
    target_set = set(target_macros) if target_macros is not None else None
    expanded_candidates = []

    for candidate in candidates:
        proposal_macro = candidate.get("proposal_macro")
        face_set = set(int(idx) for idx in candidate.get("face_indices", []))
        seed_set = set(int(idx) for idx in candidate.get("seed_face_indices", candidate.get("face_indices", [])))
        reasons = list(candidate.get("assembly_reasons", []))
        changed = False

        if (
            proposal_macro in WEAK_EXPANSION_MACROS
            and _macro_in_targets(str(proposal_macro), target_set)
            and seed_set
        ):
            for src, dst in graph_edges:
                probability = edge_lookup.get((src, dst))
                if probability is None or probability < weak_threshold:
                    continue

                if src in seed_set and dst not in face_set:
                    neighbor = dst
                elif dst in seed_set and src not in face_set:
                    neighbor = src
                else:
                    continue

                if neighbor not in faces:
                    continue
                allowed, reason = _weak_expansion_reason(
                    str(proposal_macro),
                    faces[neighbor],
                    seed_set,
                    faces,
                    edge_attributes_by_key.get(_edge_key((src, dst)), []),
                    stock_area_guard_ratio,
                )
                if allowed:
                    face_set.add(neighbor)
                    changed = True
                    _append_reason(reasons, reason)
                elif reason == "large_stock_area_guard":
                    _append_reason(reasons, reason)

        source = "same_instance_weak_expanded" if changed else candidate.get("source", "same_instance_seed_component")
        expanded_candidates.append(
            _candidate_record(
                source=source,
                proposal_macro=str(proposal_macro),
                face_indices=sorted(face_set),
                seed_face_indices=sorted(seed_set),
                graph_edges=graph_edges,
                edge_lookup=edge_lookup,
                assembly_reasons=reasons,
            )
        )

    return expanded_candidates


def deduplicate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate candidates by face set, keeping the most informative record."""

    best_by_faces: dict[tuple[int, ...], dict[str, Any]] = {}
    for candidate in candidates:
        key = tuple(sorted(int(idx) for idx in candidate.get("face_indices", [])))
        if not key:
            continue
        current = best_by_faces.get(key)
        if current is None or _candidate_score(candidate) > _candidate_score(current):
            best_by_faces[key] = _normalized_candidate(candidate)
    return [best_by_faces[key] for key in sorted(best_by_faces)]


def assemble_instance_candidates(
    graph_data: dict[str, Any],
    prediction_data: dict[str, Any],
    strong_threshold: float = 0.55,
    weak_threshold: float = 0.35,
    target_macros: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Build same-instance candidate records without using labels or feature_sets."""

    seeds = assemble_seed_components(
        graph_data,
        prediction_data,
        strong_threshold=strong_threshold,
        weak_threshold=weak_threshold,
        target_macros=target_macros,
    )
    edge_lookup = build_edge_probability_lookup(graph_data, prediction_data)
    expanded = expand_weak_candidate_faces(
        seeds,
        graph_data,
        prediction_data,
        edge_lookup=edge_lookup,
        weak_threshold=weak_threshold,
        target_macros=target_macros,
    )
    return deduplicate_candidates(expanded)


def _candidate_record(
    *,
    source: str,
    proposal_macro: str,
    face_indices: list[int],
    seed_face_indices: list[int],
    graph_edges: list[tuple[int, int]],
    edge_lookup: EdgeProbabilityLookup,
    assembly_reasons: list[str],
) -> dict[str, Any]:
    mean_probability, min_probability = _same_instance_probability_stats(face_indices, graph_edges, edge_lookup)
    return {
        "source": source,
        "proposal_macro": proposal_macro,
        "face_indices": sorted(face_indices),
        "seed_face_indices": sorted(seed_face_indices),
        "mean_same_instance_probability": mean_probability,
        "min_same_instance_probability": min_probability,
        "assembly_reasons": list(assembly_reasons),
    }


def _same_instance_probability_stats(
    face_indices: Iterable[int],
    graph_edges: list[tuple[int, int]],
    edge_lookup: EdgeProbabilityLookup,
) -> tuple[float | None, float | None]:
    face_set = set(face_indices)
    probabilities = [
        float(edge_lookup[(src, dst)])
        for src, dst in graph_edges
        if src in face_set and dst in face_set and (src, dst) in edge_lookup
    ]
    if not probabilities:
        return None, None
    return sum(probabilities) / len(probabilities), min(probabilities)


def _proposal_macro(face_indices: list[int], faces: dict[int, dict[str, Any]]) -> str:
    ordered_macros = [_macro(faces[idx]) for idx in face_indices if idx in faces and _is_seed_macro(_macro(faces[idx]))]
    if not ordered_macros:
        return "unknown_feature"
    counts = Counter(ordered_macros)
    return max(ordered_macros, key=lambda macro: (counts[macro], -ordered_macros.index(macro)))


def _weak_expansion_reason(
    proposal_macro: str,
    neighbor_face: dict[str, Any],
    seed_set: set[int],
    faces: dict[int, dict[str, Any]],
    edge_attributes: list[dict[str, Any]],
    stock_area_guard_ratio: float,
) -> tuple[bool, str]:
    neighbor_macro = _macro(neighbor_face)
    if proposal_macro in SLOT_LIKE_MACROS and neighbor_macro in SLOT_LIKE_MACROS:
        if _slot_edge_compatible(edge_attributes):
            return True, "weak_slot_like_expansion"
        return False, "weak_topology_guard"
    if proposal_macro in RIB_WEB_MACROS and neighbor_macro in RIB_WEB_MACROS:
        if _rib_stock_edge_compatible(edge_attributes):
            return True, "weak_macro_compatible_expansion"
        return False, "weak_topology_guard"
    if _strong_macro_compatible(proposal_macro, neighbor_macro):
        return True, "weak_macro_compatible_expansion"
    if proposal_macro in RIB_WEB_MACROS and _is_stock_like_face(neighbor_face):
        if not _primitive_is_plane(neighbor_face):
            return False, "protected_nonplanar_stock_guard"
        if not _rib_stock_edge_compatible(edge_attributes):
            return False, "weak_topology_guard"
        if _passes_stock_area_guard(neighbor_face, seed_set, faces, stock_area_guard_ratio):
            return True, "weak_stock_like_expansion"
        return False, "large_stock_area_guard"
    return False, "weak_macro_incompatible"


def _passes_stock_area_guard(
    neighbor_face: dict[str, Any],
    seed_set: set[int],
    faces: dict[int, dict[str, Any]],
    stock_area_guard_ratio: float,
) -> bool:
    seed_areas = [_area(faces[idx]) for idx in seed_set if idx in faces]
    max_seed_area = max(seed_areas) if seed_areas else 0.0
    if max_seed_area <= 0.0:
        return False
    return _area(neighbor_face) <= max_seed_area * stock_area_guard_ratio


def _candidate_score(candidate: dict[str, Any]) -> tuple[int, int, int, int, int]:
    source = str(candidate.get("source", ""))
    source_score = 1 if "weak_expanded" in source or source == "expanded" else 0
    stats_score = sum(
        1 for key in ("mean_same_instance_probability", "min_same_instance_probability") if candidate.get(key) is not None
    )
    return (
        stats_score,
        len(candidate.get("assembly_reasons", [])),
        len(candidate.get("seed_face_indices", [])),
        len(candidate),
        source_score,
    )


def _normalized_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(candidate)
    normalized["face_indices"] = sorted(int(idx) for idx in normalized.get("face_indices", []))
    normalized["seed_face_indices"] = sorted(int(idx) for idx in normalized.get("seed_face_indices", []))
    normalized["assembly_reasons"] = list(normalized.get("assembly_reasons", []))
    return normalized


def _extract_graph_edges(graph_data: dict[str, Any]) -> list[tuple[int, int]]:
    if "edges" in graph_data:
        raw_edges = _to_builtin(graph_data.get("edges", []))
        return [_edge for record in raw_edges if (_edge := _pair_from_record(record)) is not None]
    raw_edge_index = _to_builtin(graph_data.get("edge_index", []))
    if _looks_like_edge_index(raw_edge_index):
        return [(int(src), int(dst)) for src, dst in zip(raw_edge_index[0], raw_edge_index[1])]
    return [_edge for record in raw_edge_index if (_edge := _pair_from_record(record)) is not None]


def _edge_attributes_by_key(graph_data: dict[str, Any]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    graph_edges = _extract_graph_edges(graph_data)
    raw_attributes = _to_builtin(graph_data.get("edge_attributes", []))
    by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    if not _is_iterable(raw_attributes):
        return by_key
    for position, raw_attribute in enumerate(raw_attributes):
        if not isinstance(raw_attribute, Mapping):
            continue
        attribute = dict(raw_attribute)
        edge = _pair_from_record(attribute)
        if edge is None and position < len(graph_edges):
            edge = graph_edges[position]
        if edge is not None:
            by_key[_edge_key(edge)].append(attribute)
    return by_key


def _primitive(face_record: Mapping[str, Any]) -> str:
    for key in ("primitive_surface", "primitive", "surface_type", "primitive_type"):
        value = face_record.get(key)
        if value is not None:
            return str(value).lower()
    attributes = face_record.get("attributes")
    if isinstance(attributes, Mapping):
        for key in ("primitive_surface", "primitive", "surface_type", "type"):
            value = attributes.get(key)
            if value is not None:
                return str(value).lower()
    return "unknown"


def _primitive_is_plane(face_record: Mapping[str, Any]) -> bool:
    return _primitive(face_record) == "plane"


def _rib_stock_edge_compatible(edge_attributes: list[dict[str, Any]]) -> bool:
    if any(_explicit_edge_protected(attribute) for attribute in edge_attributes):
        return False
    for attribute in edge_attributes:
        if _explicit_edge_compatible(attribute):
            return True
        if _is_line_edge(attribute) and _edge_convexity(attribute) in {"concave", "smooth"}:
            return True
    return False


def _slot_edge_compatible(edge_attributes: list[dict[str, Any]]) -> bool:
    if any(_explicit_edge_protected(attribute) for attribute in edge_attributes):
        return False
    for attribute in edge_attributes:
        if _explicit_edge_compatible(attribute):
            return True
        if _edge_convexity(attribute) in {"concave", "smooth"}:
            return True
    return False


def _explicit_edge_protected(attribute: Mapping[str, Any]) -> bool:
    for key in ("same_instance_compatible", "assembly_compatible", "weak_assembly_compatible"):
        value = attribute.get(key)
        if value is False or value == 0:
            return True
    for key in ("relation", "topology_relation", "edge_role", "role", "wall_role"):
        value = attribute.get(key)
        if value is None:
            continue
        token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        if token in {"protected_boundary", "protected", "incompatible"}:
            return True
    return False

def _explicit_edge_compatible(attribute: Mapping[str, Any]) -> bool:
    for key in ("same_instance_compatible", "assembly_compatible", "weak_assembly_compatible"):
        if attribute.get(key) is True or attribute.get(key) == 1:
            return True
    tokens = []
    for key in ("relation", "topology_relation", "edge_role", "role", "wall_role"):
        value = attribute.get(key)
        if value is not None:
            tokens.append(str(value).lower())
    return any(token in text for text in tokens for token in ("thin_wall", "attached", "slot_internal", "slot_wall"))


def _is_line_edge(attribute: Mapping[str, Any]) -> bool:
    curve_type = str(attribute.get("curve_type", "")).lower()
    return curve_type == "line" or attribute.get("is_line") == 1.0 or attribute.get("curve_type_id") == 1


def _edge_convexity(attribute: Mapping[str, Any]) -> str:
    value = attribute.get("convexity")
    if value is not None:
        return str(value).lower()
    convexity_id = attribute.get("convexity_id")
    return {1: "convex", 2: "concave", 3: "smooth"}.get(convexity_id, "unknown")
def _edge_from_prediction_record(
    record: Any,
    graph_edges: list[tuple[int, int]],
    position: int,
) -> tuple[int, int] | None:
    if _is_number(record):
        return graph_edges[position] if position < len(graph_edges) else None
    if isinstance(record, Mapping):
        if "edge_index" in record:
            edge_index = _to_builtin(record["edge_index"])
            if _is_number(edge_index):
                index = int(edge_index)
                return graph_edges[index] if 0 <= index < len(graph_edges) else None
            pair = _pair_from_record(edge_index)
            if pair is not None:
                return pair
        pair = _pair_from_record(record)
        if pair is not None:
            return pair
    pair = _pair_from_record(record)
    if pair is not None:
        return pair
    return graph_edges[position] if position < len(graph_edges) else None


def _edge_from_mapping_key(key: Any, graph_edges: list[tuple[int, int]]) -> tuple[int, int] | None:
    key = _to_builtin(key)
    if _is_number(key):
        index = int(key)
        return graph_edges[index] if 0 <= index < len(graph_edges) else None
    if isinstance(key, str):
        stripped = key.strip("()[] ")
        for separator in (",", "-", ":"):
            if separator in stripped:
                parts = [part.strip() for part in stripped.split(separator)]
                if len(parts) == 2 and all(part.lstrip("-").isdigit() for part in parts):
                    return int(parts[0]), int(parts[1])
    return _pair_from_record(key)


def _probability_from_record(record: Any) -> float | None:
    record = _to_builtin(record)
    if _is_number(record):
        return float(record)
    if isinstance(record, Mapping):
        for key in (
            "probability",
            "same_instance_probability",
            "edge_same_instance_probability",
            "p_same_instance",
            "score",
            "value",
        ):
            if key in record and _is_number(record[key]):
                return float(record[key])
    if isinstance(record, (list, tuple)) and len(record) >= 3 and _is_number(record[2]):
        return float(record[2])
    return None


def _pair_from_record(record: Any) -> tuple[int, int] | None:
    record = _to_builtin(record)
    if isinstance(record, Mapping):
        for left_key, right_key in (("src", "dst"), ("source", "target"), ("source_face", "target_face")):
            if left_key in record and right_key in record:
                return int(record[left_key]), int(record[right_key])
        for key in ("face_pair", "faces", "pair"):
            if key in record:
                return _pair_from_record(record[key])
        return None
    if isinstance(record, (list, tuple)) and len(record) >= 2 and _is_number(record[0]) and _is_number(record[1]):
        return int(record[0]), int(record[1])
    return None


def _face_records(graph_data: dict[str, Any], prediction_data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for fallback_idx, raw_record in enumerate(_raw_face_records(graph_data)):
        record = _geometry_only_face_record(dict(raw_record))
        idx = _face_idx(record, fallback_idx)
        records.setdefault(idx, {}).update(record)

    for fallback_idx, raw_record in enumerate(_raw_face_records(prediction_data)):
        record = dict(raw_record)
        idx = _face_idx(record, fallback_idx)
        records.setdefault(idx, {"idx": idx}).update(record)

    node_count = _node_count(graph_data, records)
    for idx in range(node_count):
        records.setdefault(idx, {"idx": idx})
    return dict(sorted(records.items()))


def _geometry_only_face_record(record: dict[str, Any]) -> dict[str, Any]:
    for key in (
        "macro_feature",
        "predicted_macro_feature",
        "predicted_macro",
        "macro",
        "raw_macro_feature",
        "semantic_subtype",
        "semantic_label",
    ):
        record.pop(key, None)
    return record


def _raw_face_records(data: dict[str, Any]) -> list[Mapping[str, Any]]:
    raw = _to_builtin(data.get("faces", data.get("nodes", [])))
    if isinstance(raw, Mapping):
        items = []
        for key, value in raw.items():
            if isinstance(value, Mapping):
                record = dict(value)
                record.setdefault("idx", key)
                items.append(record)
        return items
    if _is_iterable(raw):
        return [record for record in raw if isinstance(record, Mapping)]
    return []


def _node_count(graph_data: dict[str, Any], records: dict[int, dict[str, Any]]) -> int:
    max_index = max(records.keys(), default=-1)
    for src, dst in _extract_graph_edges(graph_data):
        max_index = max(max_index, src, dst)
    return max_index + 1


def _face_idx(record: Mapping[str, Any], fallback_idx: int) -> int:
    for key in ("idx", "face_index", "index", "id"):
        if key in record and _is_number(record[key]):
            return int(record[key])
    return fallback_idx


def _macro(face_record: Mapping[str, Any]) -> str:
    for key in ("macro_feature", "predicted_macro_feature", "predicted_macro", "macro"):
        value = face_record.get(key)
        if value is not None:
            return str(value)
    return "unknown_feature"


def _is_seed_macro(macro: str | None) -> bool:
    return macro not in STOCK_MACROS


def _is_stock_like_face(face_record: Mapping[str, Any]) -> bool:
    return _macro(face_record) in STOCK_MACROS


def _strong_macro_compatible(left_macro: str | None, right_macro: str | None) -> bool:
    if not _is_seed_macro(left_macro) or not _is_seed_macro(right_macro):
        return False
    return left_macro == right_macro or {left_macro, right_macro} <= RIB_WEB_MACROS


def _macro_in_targets(macro: str, target_set: set[str] | None) -> bool:
    return target_set is None or macro in target_set


def _area(face_record: Mapping[str, Any]) -> float:
    for key in ("area", "surface_area", "face_area", "area_mm2"):
        value = face_record.get(key)
        if _is_number(value):
            return float(value)
    attributes = face_record.get("attributes")
    if isinstance(attributes, Mapping):
        physical = attributes.get("physical")
        if isinstance(physical, Mapping) and _is_number(physical.get("area")):
            return float(physical["area"])
        if _is_number(attributes.get("area")):
            return float(attributes["area"])
    return 1.0


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _edge_key(value: Any) -> tuple[int, int]:
    pair = _pair_from_record(value)
    if pair is None:
        raise TypeError(f"expected an edge pair, got {value!r}")
    src, dst = pair
    return (src, dst) if src <= dst else (dst, src)


def _looks_like_edge_index(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], (list, tuple))
        and isinstance(value[1], (list, tuple))
        and len(value[0]) == len(value[1])
        and all(_is_number(item) for row in value for item in row)
    )


def _first_present(data: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _to_builtin(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return value.tolist()
    return value


def _is_iterable(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


__all__ = [
    "assemble_instance_candidates",
    "assemble_seed_components",
    "build_edge_probability_lookup",
    "deduplicate_candidates",
    "expand_weak_candidate_faces",
]
