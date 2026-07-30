"""Deterministic Oracle feature-object memory for P4-A1."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from utils.cae_intent import load_feature_role_policy


ROOT = Path(__file__).resolve().parents[2]
MEMORY_SCHEMA_PATH = ROOT / "schemas" / "p4_a1_feature_object_memory.schema.json"
MEMORY_SCHEMA_VERSION = "topocae-feature-object-memory/1.0.0"
FORBIDDEN_MEMORY_KEYS = frozenset(
    {
        "query",
        "ground_truth",
        "target_instance_ids",
        "target_face_indices",
        "semantic_evidence",
        "uniqueness_contract",
        "canonical_record_id",
        "rendering_id",
    }
)
COORDINATE_PARAMETER_KEYS = frozenset(
    {
        "cx",
        "cy",
        "cz",
        "center",
        "centroid",
        "origin",
        "location",
        "axis",
        "direction",
        "normal",
    }
)


class FeatureObjectMemoryError(ValueError):
    """Raised when a source graph cannot produce a legal deterministic memory."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _as_vector(value: Any, *, field: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
        or any(not isinstance(item, (int, float)) for item in value)
    ):
        raise FeatureObjectMemoryError(f"{field} must be a numeric length-3 vector")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) for item in result):
        raise FeatureObjectMemoryError(f"{field} contains a non-finite coordinate")
    return result


def _bbox_center(graph: Mapping[str, Any]) -> list[float]:
    raw = graph.get("part_metadata_v2", {}).get("raw_physical", {})
    minimum = _as_vector(raw.get("axis_aligned_bbox_min"), field="bbox min")
    maximum = _as_vector(raw.get("axis_aligned_bbox_max"), field="bbox max")
    return [(left + right) / 2.0 for left, right in zip(minimum, maximum)]


def _part_contract(graph: Mapping[str, Any]) -> dict[str, Any]:
    part_reference = graph.get("part_reference")
    if isinstance(part_reference, Mapping) and "centroid" in part_reference:
        point = _as_vector(part_reference["centroid"], field="part_reference.centroid")
        kind = str(part_reference.get("kind", "solid_centroid"))
        source = "p3_oracle_graph.part_reference"
    else:
        point = _bbox_center(graph)
        kind = "part_bbox_center"
        source = "derived_from_part_metadata_v2.axis_aligned_bbox"
    return {
        "reference": {
            "kind": kind,
            "point": point,
            "source": source,
            "coordinate_dependent": True,
        },
        "local_reference_frame": {
            "kind": "graph_cartesian",
            "origin": point,
            "axes": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "source": "graph_coordinate_system",
            "rigid_rotation_covariant": True,
        },
    }


def _nodes_by_index(graph: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise FeatureObjectMemoryError("Source graph has no nodes list")
    for node in nodes:
        if not isinstance(node, Mapping) or not isinstance(node.get("idx"), int):
            raise FeatureObjectMemoryError("Every graph node must have an integer idx")
        index = int(node["idx"])
        if index in result:
            raise FeatureObjectMemoryError(f"Duplicate graph node index: {index}")
        result[index] = node
    expected = set(range(len(nodes)))
    if set(result) != expected:
        raise FeatureObjectMemoryError(
            "Graph face indices must be contiguous zero-based node indices"
        )
    return result


def _weighted_centroid_and_area(
    face_indices: Sequence[int],
    nodes: Mapping[int, Mapping[str, Any]],
) -> tuple[list[float], float]:
    weighted = [0.0, 0.0, 0.0]
    total_area = 0.0
    fallback_points: list[list[float]] = []
    for index in face_indices:
        node = nodes[index]
        physical = node.get("attributes", {}).get("physical", {})
        point = _as_vector(
            physical.get("center_of_mass"),
            field=f"nodes[{index}].attributes.physical.center_of_mass",
        )
        fallback_points.append(point)
        area_value = physical.get("area")
        area = (
            float(area_value)
            if isinstance(area_value, (int, float)) and float(area_value) >= 0
            else 0.0
        )
        total_area += area
        for axis in range(3):
            weighted[axis] += point[axis] * area
    if total_area > 0:
        return [value / total_area for value in weighted], total_area
    if not fallback_points:
        raise FeatureObjectMemoryError("Feature object cannot have an empty face set")
    count = float(len(fallback_points))
    return [
        sum(point[axis] for point in fallback_points) / count for axis in range(3)
    ], 0.0


def _metric_row(
    graph: Mapping[str, Any],
    instance_id: int,
) -> Mapping[str, Any] | None:
    rows = graph.get("metric_table")
    if not isinstance(rows, list):
        return None
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("instance_id") == instance_id
    ]
    if len(matches) > 1:
        raise FeatureObjectMemoryError(
            f"Duplicate metric_table row for instance {instance_id}"
        )
    return matches[0] if matches else None


def _legacy_descriptor_contract(
    geometry_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    complete_keys = sorted(
        key
        for key, value in geometry_parameters.items()
        if key not in COORDINATE_PARAMETER_KEYS
        and isinstance(value, (bool, int, float, str))
    )
    return {
        "availability": "derived_legacy_singleton",
        "comparison_metric": None,
        "comparison_semantics": None,
        "scope": "singleton_no_ranking",
        "is_complete_feature_description": False,
        "complete_dimension_keys": complete_keys,
    }


def _descriptor_contract(feature_set: Mapping[str, Any]) -> dict[str, Any]:
    existing = feature_set.get("descriptor_contract")
    if isinstance(existing, Mapping):
        result = copy.deepcopy(dict(existing))
        result["availability"] = "p3_oracle"
        return result
    parameters = feature_set.get("geometry_parameters")
    if not isinstance(parameters, Mapping):
        parameters = {}
    return _legacy_descriptor_contract(parameters)


def _role_mapping(
    *,
    feature_type: str,
    face_indices: Sequence[int],
    nodes: Mapping[int, Mapping[str, Any]],
    feature_policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    mappings: list[dict[str, Any]] = []
    for role in sorted(
        feature_policy["selectable_roles"],
        key=lambda item: str(item["role"]),
    ):
        role_name = str(role["role"])
        selection_status = str(role["selection_status"])
        unsupported = selection_status.startswith("unsupported")
        candidates = {
            str(item) for item in role.get("semantic_subtype_candidates", [])
        }
        if role_name == "feature_region":
            selected = list(face_indices)
        else:
            selected = sorted(
                index
                for index in face_indices
                if str(nodes[index].get("semantic_role", "")) in candidates
            )
        if unsupported:
            status = "UNSUPPORTED"
            selected = []
            reason = (
                f"{feature_type}.{role_name} is explicitly unsupported by "
                "the frozen P2 role policy."
            )
        elif not selected:
            status = "UNAVAILABLE"
            reason = (
                f"No source-graph face realizes the semantic subtypes for "
                f"{feature_type}.{role_name}."
            )
        else:
            status = "AVAILABLE"
            reason = ""
        mappings.append(
            {
                "face_role": role_name,
                "status": status,
                "selection_status": selection_status,
                "semantic_subtype_candidates": sorted(candidates),
                "face_indices": selected,
                "region_kind": str(role["region_kind"]),
                "requires_complete_instance": bool(
                    role["requires_complete_instance"]
                ),
                "allowed_operations": sorted(
                    str(item) for item in role["allowed_operations"]
                ),
                "abstention_required": status != "AVAILABLE",
                "reason": reason,
            }
        )
    return mappings


def _canonical_feature_type(
    policy: Mapping[str, Any],
    feature_type: str,
) -> str:
    for canonical, feature in policy["features"].items():
        if feature_type == canonical or feature_type in feature.get("aliases", []):
            return str(canonical)
    return feature_type


def _object_record(
    *,
    ordinal: int,
    feature_set: Mapping[str, Any],
    graph: Mapping[str, Any],
    nodes: Mapping[int, Mapping[str, Any]],
    part_point: Sequence[float],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    source_feature_type = str(feature_set.get("macro_feature", ""))
    feature_type = _canonical_feature_type(policy, source_feature_type)
    feature_policy = policy["features"].get(feature_type)
    if not isinstance(feature_policy, Mapping):
        raise FeatureObjectMemoryError(
            f"Feature {feature_type!r} is outside the frozen 17-class P2 policy"
        )
    instance_id = feature_set.get("instance_id")
    if not isinstance(instance_id, int) or instance_id <= 0:
        raise FeatureObjectMemoryError("Feature instance_id must be a positive integer")
    raw_indices = feature_set.get("node_indices")
    if (
        not isinstance(raw_indices, list)
        or not raw_indices
        or any(not isinstance(index, int) for index in raw_indices)
    ):
        raise FeatureObjectMemoryError(
            f"Feature instance {instance_id} has no valid node_indices"
        )
    face_indices = sorted(set(int(index) for index in raw_indices))
    missing = [index for index in face_indices if index not in nodes]
    if missing:
        raise FeatureObjectMemoryError(
            f"Feature instance {instance_id} has out-of-range faces: {missing}"
        )
    for index in face_indices:
        node_instance = nodes[index].get("instance_id")
        node_feature = _canonical_feature_type(
            policy, str(nodes[index].get("macro_feature", ""))
        )
        if node_instance != instance_id or node_feature != feature_type:
            raise FeatureObjectMemoryError(
                "feature_sets and node instance/macro labels disagree"
            )

    centroid, area = _weighted_centroid_and_area(face_indices, nodes)
    metric = _metric_row(graph, instance_id)
    if metric is not None and "centroid" in metric:
        metric_centroid = _as_vector(
            metric["centroid"], field=f"metric_table[{instance_id}].centroid"
        )
        centroid = metric_centroid
    if isinstance(feature_set.get("metric_evidence"), Mapping):
        evidence = feature_set["metric_evidence"]
        if isinstance(evidence.get("area"), (int, float)):
            area = float(evidence["area"])
        if "centroid" in evidence:
            centroid = _as_vector(
                evidence["centroid"],
                field=f"feature_sets[{instance_id}].metric_evidence.centroid",
            )
    distance = math.dist(centroid, [float(value) for value in part_point])
    if metric is not None and isinstance(metric.get("nearest_distance"), (int, float)):
        distance = float(metric["nearest_distance"])

    comparison_metric = (
        str(metric["comparison_metric"])
        if metric is not None and metric.get("comparison_metric") is not None
        else None
    )
    comparison_value = (
        float(metric["comparison_value"])
        if metric is not None
        and isinstance(metric.get("comparison_value"), (int, float))
        else None
    )
    role_map = _role_mapping(
        feature_type=feature_type,
        face_indices=face_indices,
        nodes=nodes,
        feature_policy=feature_policy,
    )
    unavailable = [
        item["face_role"] for item in role_map if item["status"] == "UNAVAILABLE"
    ]
    geometry_parameters = feature_set.get("geometry_parameters")
    if not isinstance(geometry_parameters, Mapping):
        geometry_parameters = {}
    return {
        "object_id": f"obj-{ordinal:04d}",
        "canonical_feature_type": feature_type,
        "feature_family": str(feature_policy["family"]),
        "semantic_face_roles": sorted(
            {
                str(nodes[index].get("semantic_role"))
                for index in face_indices
                if nodes[index].get("semantic_role")
            }
        ),
        "face_role_map": role_map,
        "geometry_parameters": copy.deepcopy(dict(geometry_parameters)),
        "descriptor_contract": _descriptor_contract(feature_set),
        "descriptors": {
            "centroid": centroid,
            "area": area,
            "comparison_metric": comparison_metric,
            "comparison_value": comparison_value,
            "comparison_semantics": (
                str(metric["comparison_semantics"])
                if metric is not None
                and metric.get("comparison_semantics") is not None
                else None
            ),
            "nearest_reference": (
                str(metric.get("nearest_reference", "part_reference"))
                if metric is not None
                else "part_reference"
            ),
            "distance_to_part_reference": distance,
        },
        "topology": {
            "source_face_count": len(nodes),
            "object_face_count": len(face_indices),
            "complete_instance_face_indices": face_indices,
            "indices_are_zero_based": True,
            "node_labels_consistent": True,
        },
        "legality": {
            "status": "VALID" if not unavailable else "NEEDS_CLARIFICATION",
            "issues": [
                f"supported_role_without_source_faces:{role}" for role in unavailable
            ],
        },
        "abstention": {
            "required": bool(unavailable),
            "reasons": [
                f"role_unavailable:{role}" for role in unavailable
            ],
            "explicit_role_restrictions": [
                item["face_role"]
                for item in role_map
                if item["status"] == "UNSUPPORTED"
            ],
        },
    }


@lru_cache(maxsize=1)
def _memory_validator() -> Draft202012Validator:
    schema = json.loads(MEMORY_SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def _forbidden_key_paths(value: Any, path: tuple[Any, ...] = ()) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = path + (key,)
            if key in FORBIDDEN_MEMORY_KEYS:
                found.append("/" + "/".join(str(item) for item in child_path))
            found.extend(_forbidden_key_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_key_paths(child, path + (index,)))
    return found


def validate_feature_object_memory(value: Any) -> dict[str, Any]:
    errors = [
        {
            "path": "/" + "/".join(str(item) for item in failure.absolute_path),
            "message": failure.message,
        }
        for failure in sorted(
            _memory_validator().iter_errors(value),
            key=lambda item: list(item.absolute_path),
        )
    ]
    for path in _forbidden_key_paths(value):
        errors.append(
            {
                "path": path,
                "message": "Inference-forbidden query/ground-truth key in memory",
            }
        )
    return {"valid": not errors, "errors": errors}


def build_feature_object_memory(
    graph: Mapping[str, Any],
    *,
    source_graph_sha256: str | None = None,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a byte-deterministic, query-independent candidate memory."""

    if not isinstance(graph, Mapping):
        raise FeatureObjectMemoryError("Source graph must be an object mapping")
    if graph.get("schema_version") != "topocae_graph_v2_relative_1_1":
        raise FeatureObjectMemoryError(
            f"Unsupported graph schema: {graph.get('schema_version')!r}"
        )
    forbidden = _forbidden_key_paths(graph)
    if forbidden:
        raise FeatureObjectMemoryError(
            "Source graph unexpectedly contains query/ground-truth fields: "
            + ", ".join(forbidden[:5])
        )
    actual_policy = policy if policy is not None else load_feature_role_policy()
    nodes = _nodes_by_index(graph)
    part = _part_contract(graph)
    feature_sets = graph.get("feature_sets")
    if not isinstance(feature_sets, list):
        raise FeatureObjectMemoryError("Source graph has no feature_sets list")
    ordered_sets = sorted(
        (item for item in feature_sets if isinstance(item, Mapping)),
        key=lambda item: (
            str(item.get("macro_feature", "")),
            int(item.get("instance_id", -1)),
            tuple(item.get("node_indices", [])),
        ),
    )
    if not ordered_sets:
        raise FeatureObjectMemoryError("Source graph has no Oracle feature instances")
    objects = [
        _object_record(
            ordinal=index,
            feature_set=feature_set,
            graph=graph,
            nodes=nodes,
            part_point=part["reference"]["point"],
            policy=actual_policy,
        )
        for index, feature_set in enumerate(ordered_sets, start=1)
    ]
    all_object_faces = [
        set(item["topology"]["complete_instance_face_indices"]) for item in objects
    ]
    overlaps = [
        [left + 1, right + 1]
        for left in range(len(all_object_faces))
        for right in range(left + 1, len(all_object_faces))
        if all_object_faces[left] & all_object_faces[right]
    ]
    graph_hash = source_graph_sha256 or _sha256_json(graph)
    if not isinstance(graph_hash, str) or len(graph_hash) != 64:
        raise FeatureObjectMemoryError("source_graph_sha256 must be a SHA-256 hex digest")
    memory = {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "memory_mode": "oracle_geometry",
        "source_graph": {
            "schema_version": str(graph["schema_version"]),
            "sha256": graph_hash.lower(),
        },
        "part": {
            "part_ref": f"part-{graph_hash[:24].lower()}",
            **part,
            "face_index_contract": {
                "kind": "source_graph_node_index",
                "base": 0,
                "minimum": 0,
                "maximum": len(nodes) - 1,
                "raw_step_face_ids_exposed": False,
            },
        },
        "objects": objects,
        "memory_legality": {
            "status": "VALID" if not overlaps else "INVALID",
            "object_face_overlap_pairs": overlaps,
            "source_face_count": len(nodes),
        },
        "field_invariance": {
            "rotation_invariant": [
                "/objects/*/descriptors/area",
                "/objects/*/descriptors/comparison_value",
                "/objects/*/descriptors/distance_to_part_reference",
                "/objects/*/geometry_parameters/scalar_dimensions",
                "/objects/*/topology",
            ],
            "rigid_rotation_covariant": [
                "/part/reference/point",
                "/part/local_reference_frame/axes",
                "/objects/*/descriptors/centroid",
            ],
            "coordinate_frame_dependent": [
                "/part/local_reference_frame",
                "/objects/*/geometry_parameters/coordinate_fields",
            ],
        },
    }
    validation = validate_feature_object_memory(memory)
    if not validation["valid"]:
        raise FeatureObjectMemoryError(
            "Constructed memory violates its schema: "
            + json.dumps(validation["errors"][:10], ensure_ascii=False)
        )
    return memory


def canonical_memory_bytes(memory: Mapping[str, Any]) -> bytes:
    """Serialize memory deterministically for hashing and byte-repeatability tests."""

    validation = validate_feature_object_memory(memory)
    if not validation["valid"]:
        raise FeatureObjectMemoryError(
            "Cannot serialize invalid memory: "
            + json.dumps(validation["errors"][:10], ensure_ascii=False)
        )
    return _canonical_bytes(memory)


__all__ = [
    "FeatureObjectMemoryError",
    "build_feature_object_memory",
    "canonical_memory_bytes",
    "validate_feature_object_memory",
]
