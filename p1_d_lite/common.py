from __future__ import annotations

from copy import deepcopy
import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "reports" / "p1_d_lite_v1"

ABSOLUTE_AREA_TOLERANCE = 5.0
RELATIVE_AREA_TOLERANCE = 0.01

GRAPH_TRUTH_KEYS = {
    "feature_sets", "object_truth", "labels", "face_labels", "primary_target",
    "context", "context_type", "host_family", "generator_revision", "seed",
    "source_path", "generation", "p1_c_automatic_audit", "dataset_provenance",
}
NODE_TRUTH_KEYS = {
    "macro_feature", "macro_feature_id", "semantic_label", "semantic_role",
    "instance_id", "primary_secondary_role", "truth",
}
PREDICTION_TRUTH_KEYS = {"truth", "feature_sets", "object_truth", "labels"}
FACE_PREDICTION_TRUTH_KEYS = {
    "truth_macro_feature", "truth_semantic_role", "truth_instance_id",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    _require_new_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    _require_new_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fieldnames or (list(rows[0]) if rows else []))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def strip_graph_truth(graph: Mapping[str, Any]) -> dict[str, Any]:
    clean = deepcopy(dict(graph))
    for key in GRAPH_TRUTH_KEYS:
        clean.pop(key, None)
    protocol = clean.get("graph_v2_protocol")
    if isinstance(protocol, dict):
        protocol["labels_provided"] = False
    for node in clean.get("nodes", []):
        if not isinstance(node, dict):
            continue
        for key in NODE_TRUTH_KEYS:
            node.pop(key, None)
        attributes = node.get("attributes")
        if isinstance(attributes, dict):
            for key in NODE_TRUTH_KEYS:
                attributes.pop(key, None)
    return clean


def strip_prediction_truth(prediction: Mapping[str, Any]) -> dict[str, Any]:
    clean = deepcopy(dict(prediction))
    for key in PREDICTION_TRUTH_KEYS:
        clean.pop(key, None)
    for face in clean.get("faces", []):
        if isinstance(face, dict):
            for key in FACE_PREDICTION_TRUTH_KEYS:
                face.pop(key, None)
    return clean


def validate_graph(graph: Mapping[str, Any]) -> None:
    if graph.get("schema_version") != "topocae_graph_v2_relative_1_1":
        raise ValueError("expected Graph v2.1")
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("graph nodes/edges must be lists")
    count = len(nodes)
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError("each edge must be a two-index list")
        if any(not isinstance(index, int) or index < 0 or index >= count for index in edge):
            raise ValueError("edge endpoint outside node range")


def face_area(graph: Mapping[str, Any], face_index: int) -> float:
    nodes = graph.get("nodes", [])
    if face_index < 0 or face_index >= len(nodes):
        raise IndexError(face_index)
    node = nodes[face_index]
    attributes = node.get("attributes", {}) if isinstance(node, Mapping) else {}
    physical = attributes.get("physical", {}) if isinstance(attributes, Mapping) else {}
    value = physical.get("area", node.get("area", 0.0) if isinstance(node, Mapping) else 0.0)
    return float(value or 0.0)


def candidate_area(graph: Mapping[str, Any], face_indices: Iterable[int]) -> float:
    return sum(face_area(graph, int(index)) for index in sorted(set(face_indices)))


def undirected_adjacency(graph: Mapping[str, Any]) -> dict[int, set[int]]:
    adjacency = {index: set() for index in range(len(graph.get("nodes", [])))}
    for source, target in graph.get("edges", []):
        source_i, target_i = int(source), int(target)
        adjacency[source_i].add(target_i)
        adjacency[target_i].add(source_i)
    return adjacency


def is_connected(graph: Mapping[str, Any], face_indices: Iterable[int]) -> bool:
    members = set(int(value) for value in face_indices)
    if not members:
        return False
    adjacency = undirected_adjacency(graph)
    pending = [next(iter(members))]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend((adjacency.get(current, set()) & members) - seen)
    return seen == members


def connected_macro_objects(
    graph: Mapping[str, Any],
    face_macros: Mapping[int, str],
    *,
    stock_labels: set[str] | None = None,
) -> list[dict[str, Any]]:
    ignored = stock_labels or {"stock", "unknown", "candidate", ""}
    adjacency = undirected_adjacency(graph)
    remaining = {int(index) for index, macro in face_macros.items() if str(macro) not in ignored}
    objects: list[dict[str, Any]] = []
    while remaining:
        seed = min(remaining)
        macro = str(face_macros[seed])
        pending = [seed]
        component: set[int] = set()
        while pending:
            current = pending.pop()
            if current in component or current not in remaining or str(face_macros[current]) != macro:
                continue
            component.add(current)
            pending.extend(adjacency.get(current, set()))
        remaining -= component
        faces = sorted(component)
        objects.append({
            "macro_feature": macro,
            "face_indices": faces,
            "face_count": len(faces),
            "area": candidate_area(graph, faces),
        })
    return objects


def normalize_objects(graph: Mapping[str, Any], records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for position, record in enumerate(records):
        faces = sorted({int(index) for index in record.get("face_indices", [])})
        macro = str(record.get("macro_feature", record.get("predicted_macro", "unknown")))
        result.append({
            "object_id": str(record.get("feature_object_id", record.get("candidate_id", f"object_{position:05d}"))),
            "macro_feature": macro,
            "face_indices": faces,
            "face_count": len(faces),
            "area": float(record.get("area", candidate_area(graph, faces))),
            "confidence": object_confidence(record),
        })
    return result


def object_confidence(record: Mapping[str, Any]) -> float | None:
    for key in ("final_confidence", "confidence", "macro_confidence"):
        value = record.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, Mapping):
            for nested in ("macro_confidence_mean", "score", "value"):
                if isinstance(value.get(nested), (int, float)):
                    return float(value[nested])
    return None


def area_compatible(truth_area: float, prediction_area: float) -> bool:
    delta = abs(float(truth_area) - float(prediction_area))
    return delta <= ABSOLUTE_AREA_TOLERANCE or delta <= RELATIVE_AREA_TOLERANCE * max(abs(float(truth_area)), 1.0)


def one_to_one_area_match(
    truths: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidates: list[tuple[float, int, int]] = []
    for truth_index, truth in enumerate(truths):
        for prediction_index, prediction in enumerate(predictions):
            if str(truth.get("macro_feature")) != str(prediction.get("macro_feature")):
                continue
            truth_area = float(truth.get("area", 0.0))
            predicted_area = float(prediction.get("area", 0.0))
            if area_compatible(truth_area, predicted_area):
                candidates.append((abs(truth_area - predicted_area), truth_index, prediction_index))
    used_truth: set[int] = set()
    used_prediction: set[int] = set()
    matches = []
    for error, truth_index, prediction_index in sorted(candidates):
        if truth_index in used_truth or prediction_index in used_prediction:
            continue
        used_truth.add(truth_index)
        used_prediction.add(prediction_index)
        matches.append({
            "truth_index": truth_index,
            "prediction_index": prediction_index,
            "absolute_area_error": error,
        })
    truth_count = len(truths)
    prediction_count = len(predictions)
    matched = len(matches)
    precision = matched / prediction_count if prediction_count else 0.0
    recall = matched / truth_count if truth_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "truth_count": truth_count,
        "prediction_count": prediction_count,
        "matched_count": matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
        "unmatched_truth_indices": sorted(set(range(truth_count)) - used_truth),
        "unmatched_prediction_indices": sorted(set(range(prediction_count)) - used_prediction),
    }


def canonical_object_signature(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        ({
            "macro_feature": str(record.get("macro_feature", record.get("predicted_macro", "unknown"))),
            "face_indices": sorted(int(index) for index in record.get("face_indices", [])),
        } for record in records),
        key=lambda row: (row["macro_feature"], row["face_indices"]),
    )


def _require_new_output(path: Path) -> None:
    resolved = path.resolve()
    root = OUTPUT_ROOT.resolve()
    if root not in resolved.parents and resolved != root and resolved != (ROOT / "CODEX_CONTEXT_P1_D_LITE.md").resolve():
        raise ValueError(f"P1-D Lite output escaped the authorized new-output scope: {resolved}")

