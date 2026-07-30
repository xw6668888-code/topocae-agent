from __future__ import annotations

import copy
import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .constants import (
    CANONICAL_FEATURES,
    DATASET_SPLIT_MAP,
    MODEL_AREA_UNIT,
    MODEL_LENGTH_UNIT,
    RECORD_VERSION,
    SEED,
    SOURCE_MACRO_TO_CANONICAL,
    SOURCE_MANIFEST,
    assert_source_path,
)
from .io import canonical_sha256, read_json, read_jsonl


UNIT_SYSTEM = {
    "length": "mm",
    "area": "mm2",
    "force": "N",
    "pressure": "MPa",
    "displacement": "mm",
    "angle": "deg",
}

METRIC_KEYS = {
    "diameter": ("diameter",),
    "radius": ("radius",),
    "depth": ("depth", "profile_depth"),
    "length": ("length", "l", "path_length"),
    "width": ("width", "w", "profile_width"),
    "thickness": ("thickness",),
}


def _stable_score(*parts: Any) -> str:
    return hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _round(value: float) -> float:
    result = round(float(value), 6)
    return 0.0 if result == -0.0 else result


def _tolerance(value: float) -> float:
    return max(1e-6, abs(float(value)) * 1e-6)


def _canonical_features_in_manifest(record: dict[str, Any]) -> list[str]:
    return sorted(
        {
            SOURCE_MACRO_TO_CANONICAL[source_macro]
            for source_macro in record.get("macro_instance_counts", {})
            if source_macro in SOURCE_MACRO_TO_CANONICAL
        }
    )


def select_pilot_sources(
    manifest_records: list[dict[str, Any]] | None = None,
    *,
    seed: int = SEED,
    split_quotas: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    records = manifest_records if manifest_records is not None else read_jsonl(SOURCE_MANIFEST)
    quotas = split_quotas or {"train": 120, "val": 20, "development_test": 20}
    selected: list[dict[str, Any]] = []
    for source_split, quota in quotas.items():
        candidates = [record for record in records if record["split"] == source_split]
        candidates.sort(key=lambda record: _stable_score(seed, source_split, record["sample_id"]))
        used: set[str] = set()

        # First guarantee all 17 canonical features in every split where available.
        for feature in CANONICAL_FEATURES:
            record = next(
                (
                    item
                    for item in candidates
                    if item["sample_id"] not in used and feature in _canonical_features_in_manifest(item)
                ),
                None,
            )
            if record is None:
                raise ValueError(f"No {feature} source available in {source_split}")
            selected.append({"record": record, "focus_feature": feature})
            used.add(str(record["sample_id"]))

        # Preserve both rib and web provenance in the train pilot.
        if source_split == "train":
            for source_macro in ("rib", "web"):
                record = next(
                    (
                        item
                        for item in candidates
                        if item["sample_id"] not in used and source_macro in item.get("macro_instance_counts", {})
                    ),
                    None,
                )
                if record is not None:
                    selected.append({"record": record, "focus_feature": "rib"})
                    used.add(str(record["sample_id"]))

        current = sum(1 for item in selected if item["record"]["split"] == source_split)
        for index, record in enumerate(item for item in candidates if item["sample_id"] not in used):
            if current >= quota:
                break
            features = _canonical_features_in_manifest(record)
            if not features:
                continue
            selected.append({"record": record, "focus_feature": features[index % len(features)]})
            used.add(str(record["sample_id"]))
            current += 1
        if current != quota:
            raise ValueError(f"Unable to satisfy pilot source quota for {source_split}: {current}/{quota}")
    return selected


def _metric_values(feature_set: dict[str, Any], members: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    parameters = feature_set.get("geometry_parameters") or {}
    metrics: dict[str, dict[str, Any]] = {}
    for metric, keys in METRIC_KEYS.items():
        for key in keys:
            value = parameters.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0:
                metrics[metric] = {
                    "value": float(value),
                    "unit": MODEL_LENGTH_UNIT,
                    "source": "feature_set.geometry_parameters",
                    "source_key": key,
                }
                break
    areas = [float(node["attributes"]["physical"]["area"]) for node in members]
    if areas and all(math.isfinite(value) and value > 0 for value in areas):
        metrics["area"] = {
            "value": sum(areas),
            "unit": MODEL_AREA_UNIT,
            "source": "node.attributes.physical",
            "source_key": "area_sum_over_instance",
        }
    return metrics


def _weighted_centroid(members: list[dict[str, Any]]) -> list[float]:
    weighted = [0.0, 0.0, 0.0]
    total = 0.0
    for node in members:
        physical = node["attributes"]["physical"]
        area = float(physical["area"])
        center = [float(value) for value in physical["center_of_mass"]]
        for index in range(3):
            weighted[index] += area * center[index]
        total += area
    if total <= 0:
        raise ValueError("Feature object has no positive face area")
    return [value / total for value in weighted]


def load_feature_objects(record: dict[str, Any], crosswalk: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    graph = read_json(assert_source_path(record["graph_path"]))
    node_by_idx = {int(node["idx"]): node for node in graph["nodes"]}
    objects: list[dict[str, Any]] = []
    for feature_set in graph["feature_sets"]:
        source_macro = str(feature_set["macro_feature"])
        canonical_feature = SOURCE_MACRO_TO_CANONICAL[source_macro]
        indices = sorted({int(index) for index in feature_set["node_indices"]})
        members = [node_by_idx[index] for index in indices]
        observed_roles = sorted({str(node["semantic_role"]) for node in members})
        available_roles: dict[str, Any] = {}
        for role in crosswalk["features"][canonical_feature]["face_roles"]:
            status = str(role["selection_status"])
            if status.startswith("unsupported"):
                continue
            candidates = set(role["semantic_subtype_candidates"])
            if role["requires_complete_instance"]:
                selected = indices
            else:
                selected = sorted(
                    int(node["idx"]) for node in members if str(node["semantic_role"]) in candidates
                )
                present = set(observed_roles) & candidates
                if not selected or (status == "composite" and present != candidates):
                    continue
            available_roles[str(role["face_role"])] = {
                **copy.deepcopy(role),
                "face_indices": selected,
            }
        objects.append(
            {
                "instance_id": int(feature_set["instance_id"]),
                "source_macro": source_macro,
                "feature_type": canonical_feature,
                "instance_faces": indices,
                "observed_semantic_roles": observed_roles,
                "roles": available_roles,
                "metrics": _metric_values(feature_set, members),
                "centroid": _weighted_centroid(members),
            }
        )
    return graph, objects


def _groups(objects: Iterable[dict[str, Any]], focus_feature: str) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in objects:
        grouped[item["feature_type"]].append(item)
    ordered_features = [focus_feature] + [feature for feature in CANONICAL_FEATURES if feature != focus_feature]
    return [sorted(grouped[feature], key=lambda item: item["instance_id"]) for feature in ordered_features if grouped[feature]]


def _common_role_for_operation(
    group: list[dict[str, Any]],
    operation_preferences: Iterable[str],
    *,
    role_offset: int = 0,
) -> tuple[str, str] | None:
    candidates: list[tuple[str, str]] = []
    for operation in operation_preferences:
        common_roles = set(group[0]["roles"])
        for item in group[1:]:
            common_roles &= set(item["roles"])
        for role_name in sorted(common_roles, key=lambda name: (name == "feature_region", name)):
            if all(operation in item["roles"][role_name]["allowed_operations"] for item in group):
                candidates.append((role_name, operation))
    if not candidates:
        return None
    return candidates[role_offset % len(candidates)]


def _operation(operation_id: str, operation_type: str, target: dict[str, Any] | None = None) -> dict[str, Any]:
    if operation_type == "fixed_support":
        parameters = {"constrained_dofs": ["U1", "U2", "U3"]}
    elif operation_type == "displacement":
        parameters = {
            "components": [
                {"dof": "U1", "value": {"dimension": "displacement", "value": 0.1, "unit": "mm"}}
            ],
            "reference_frame": {"kind": "global_cartesian"},
        }
    elif operation_type == "pressure":
        parameters = {
            "magnitude": {"dimension": "pressure", "value": 1.0, "unit": "MPa"},
            "direction": {"kind": "surface_normal"},
        }
    elif operation_type == "surface_traction":
        parameters = {
            "magnitude": {"dimension": "pressure", "value": 0.75, "unit": "MPa"},
            "direction": {"kind": "global_cartesian", "vector": [1.0, 0.0, 0.0]},
        }
    elif operation_type == "local_mesh_size":
        parameters = {"size": {"dimension": "length", "value": 0.5, "unit": "mm"}}
    elif operation_type == "material_assignment":
        target = {"scope": "whole_part"}
        parameters = {
            "material_name": "CQG-LinearElastic",
            "elastic": {
                "youngs_modulus": {"dimension": "pressure", "value": 210.0, "unit": "GPa"},
                "poisson_ratio": 0.3,
            },
        }
    elif operation_type == "global_mesh_size":
        target = {"scope": "whole_part"}
        parameters = {"size": {"dimension": "length", "value": 4.0, "unit": "mm"}}
    else:
        raise ValueError(f"Unknown operation: {operation_type}")
    return {
        "operation_id": operation_id,
        "operation_type": operation_type,
        "target": copy.deepcopy(target),
        "parameters": parameters,
    }


def _valid_result(operations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "cae-intent-result/1.0.0",
        "status": "VALID",
        "intent": {
            "schema_version": "cae-intent/1.0.0",
            "analysis_type": "static_structural",
            "operations": operations,
            "source_language": "en-US",
            "assumptions": [],
            "unit_system": copy.deepcopy(UNIT_SYSTEM),
            "normalization_audit": [],
        },
    }


def _clarification_result(
    *,
    code: str,
    field_path: str,
    description: str,
    ambiguity: bool = False,
) -> dict[str, Any]:
    if ambiguity:
        missing: list[dict[str, Any]] = []
        ambiguities = [
            {
                "code": "A_TARGET_INSTANCE",
                "field_path": field_path,
                "description": description,
                "options": ["candidate A", "candidate B"],
            }
        ]
        error_code = "E_AMBIGUOUS_REQUIREMENT"
    else:
        missing = [{"code": code, "field_path": field_path, "description": description}]
        ambiguities = []
        error_code = "E_MISSING_INFORMATION"
    return {
        "schema_version": "cae-intent-result/1.0.0",
        "status": "NEEDS_CLARIFICATION",
        "error_code": error_code,
        "field_path": field_path,
        "missing_information": missing,
        "ambiguities": ambiguities,
        "clarification_questions": [
            {"question_id": "q1", "field_path": field_path, "question": f"Please clarify: {description}"}
        ],
    }


def _invalid_result(description: str) -> dict[str, Any]:
    return {
        "schema_version": "cae-intent-result/1.0.0",
        "status": "INVALID",
        "error_code": "E_SCHEMA_VALIDATION",
        "field_path": "/operations/0/target",
        "missing_information": [],
        "ambiguities": [],
        "clarification_questions": [],
    }


def _metric_evidence(metric: str, item: dict[str, Any], value: float) -> dict[str, Any]:
    source = item["metrics"][metric]
    return {
        "metric": metric,
        "value": _round(value),
        "unit": source["unit"],
        "source": source["source"],
        "stable_tolerance": _round(_tolerance(value)),
    }


def _feature_target(
    group: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    role_name: str,
    operation_id: str,
    operation_type: str,
    quantifier: str,
    filters: dict[str, Any] | None = None,
    count: int | None = None,
    comparison_metric: str | None = None,
    reference_frame: str | None = None,
    uniqueness_kind: str,
    rationale: str,
    metric_evidence: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    target: dict[str, Any] = {
        "feature_type": group[0]["feature_type"],
        "quantifier": quantifier,
        "face_role": role_name,
        "filters": copy.deepcopy(filters or {}),
    }
    if count is not None:
        target["count"] = count
    if comparison_metric is not None:
        target["comparison_metric"] = comparison_metric
    if reference_frame is not None:
        target["reference_frame"] = {"kind": reference_frame}

    evidence = []
    target_faces: set[int] = set()
    for item in selected:
        selected_faces = sorted(item["roles"][role_name]["face_indices"])
        target_faces.update(selected_faces)
        evidence.append(
            {
                "instance_id": item["instance_id"],
                "source_macro_feature": item["source_macro"],
                "instance_face_indices": sorted(item["instance_faces"]),
                "selected_face_indices": selected_faces,
                "observed_semantic_roles": sorted(item["observed_semantic_roles"]),
                "role_candidate_subtypes": sorted(item["roles"][role_name]["semantic_subtype_candidates"]),
            }
        )
    uniqueness: dict[str, Any] = {
        "kind": uniqueness_kind,
        "candidate_count": len(group),
        "selected_count": len(selected),
        "rationale": rationale,
    }
    if metric_evidence:
        uniqueness["metric_evidence"] = metric_evidence
    if reference_frame:
        uniqueness["reference_frame"] = reference_frame
    ground = {
        "operation_id": operation_id,
        "operation_type": operation_type,
        "target_contract": "feature_face_set",
        "grounding_required": True,
        "canonical_feature_type": group[0]["feature_type"],
        "source_macro_features": sorted({item["source_macro"] for item in selected}),
        "face_role": role_name,
        "quantifier": quantifier,
        "target_instance_ids": sorted(item["instance_id"] for item in selected),
        "target_face_indices": sorted(target_faces),
        "semantic_evidence": evidence,
        "uniqueness_contract": uniqueness,
    }
    verbal = {
        "feature_type": group[0]["feature_type"],
        "face_role": role_name,
        "quantifier": quantifier,
        "count": count,
        "filters": copy.deepcopy(filters or {}),
        "comparison_metric": comparison_metric,
        "reference_frame": reference_frame,
    }
    return target, ground, verbal


def _whole_ground(operation_id: str, operation_type: str) -> dict[str, Any]:
    return {
        "operation_id": operation_id,
        "operation_type": operation_type,
        "target_contract": "whole_part",
        "grounding_required": False,
        "scope": "whole_part",
    }


def _executable_ground(targets: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "contract_status": "executable",
        "intent_status": "VALID",
        "geometry_source": "oracle",
        "targets": targets,
    }


def _non_executable_ground(status: str, reason: str) -> dict[str, Any]:
    return {
        "contract_status": "non_executable",
        "intent_status": status,
        "geometry_source": "oracle",
        "targets": [],
        "non_executable_reason": reason,
    }


def _single_valid_plan(
    level: str,
    group: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    role_name: str,
    operation_type: str,
    quantifier: str,
    *,
    filters: dict[str, Any] | None = None,
    count: int | None = None,
    comparison_metric: str | None = None,
    reference_frame: str | None = None,
    uniqueness_kind: str,
    rationale: str,
    metric_evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    target, ground, verbal = _feature_target(
        group,
        selected,
        role_name=role_name,
        operation_id="op1",
        operation_type=operation_type,
        quantifier=quantifier,
        filters=filters,
        count=count,
        comparison_metric=comparison_metric,
        reference_frame=reference_frame,
        uniqueness_kind=uniqueness_kind,
        rationale=rationale,
        metric_evidence=metric_evidence,
    )
    operation = _operation("op1", operation_type, target)
    return {
        "level": level,
        "status": "VALID",
        "intent_result": _valid_result([operation]),
        "ground_truth": _executable_ground([ground]),
        "verbalization": {**verbal, "utterance_kind": "executable", "operations": [operation]},
    }


def _plan_l1(groups: list[list[dict[str, Any]]], offset: int) -> dict[str, Any]:
    single_group = next((group for group in groups if len(group) == 1), groups[0][:1])
    pair = _common_role_for_operation(
        single_group,
        ("fixed_support", "pressure", "surface_traction", "displacement", "local_mesh_size"),
        role_offset=offset,
    )
    if pair is None:
        raise ValueError("No L1-compatible target")
    role_name, operation_type = pair
    return _single_valid_plan(
        "L1",
        single_group,
        single_group,
        role_name,
        operation_type,
        "exactly_one",
        uniqueness_kind="exactly_one",
        rationale="Exactly one source feature instance of this canonical type exists in the graph.",
    )


def _plan_l2(groups: list[list[dict[str, Any]]], offset: int) -> dict[str, Any]:
    group = max(groups, key=lambda items: (len(items), items[0]["feature_type"] == groups[0][0]["feature_type"]))
    pair = _common_role_for_operation(
        group,
        ("fixed_support", "pressure", "surface_traction", "local_mesh_size", "displacement"),
        role_offset=offset,
    )
    if pair is None:
        raise ValueError("No L2-compatible target")
    role_name, operation_type = pair
    if len(group) > 1 and offset % 2:
        quantifier = "count"
        count = len(group)
        kind = "count"
        rationale = "The complete matching set is unique and its cardinality equals the requested count."
    else:
        quantifier = "all"
        count = None
        kind = "all"
        rationale = "All source instances matching the canonical feature and role are selected."
    return _single_valid_plan(
        "L2",
        group,
        group,
        role_name,
        operation_type,
        quantifier,
        count=count,
        uniqueness_kind=kind,
        rationale=rationale,
    )


def _plan_l3(groups: list[list[dict[str, Any]]], offset: int) -> dict[str, Any]:
    metric_order = ["diameter", "radius", "depth", "area", "length", "width", "thickness"]
    metric_order = metric_order[offset % len(metric_order):] + metric_order[: offset % len(metric_order)]
    for group in groups:
        if not all("feature_region" in item["roles"] for item in group):
            continue
        for metric in metric_order:
            available = [item for item in group if metric in item["metrics"]]
            for focus in available:
                value = float(focus["metrics"][metric]["value"])
                tolerance = _tolerance(value)
                matches = [
                    item
                    for item in available
                    if abs(float(item["metrics"][metric]["value"]) - value) <= tolerance
                ]
                if len(matches) != 1:
                    continue
                unit = focus["metrics"][metric]["unit"]
                dimension = "area" if unit == "mm2" else "length"
                filters = {
                    metric: {
                        "operator": "eq",
                        "value": {
                            "dimension": dimension,
                            "value": _round(value),
                            "unit": unit,
                            "tolerance": _round(tolerance),
                        },
                    }
                }
                return _single_valid_plan(
                    "L3",
                    group,
                    [focus],
                    "feature_region",
                    "local_mesh_size",
                    "exactly_one",
                    filters=filters,
                    uniqueness_kind="attribute_filter",
                    rationale=f"The graph-derived {metric} filter matches exactly one eligible feature instance.",
                    metric_evidence=[_metric_evidence(metric, focus, value)],
                )
    focus = groups[0][0]
    result = _clarification_result(
        code="M_REFERENCE",
        field_path="/operations/0/target/filters",
        description="No unit-traceable attribute in the request uniquely identifies one target instance.",
    )
    return {
        "level": "L3",
        "status": "NEEDS_CLARIFICATION",
        "intent_result": result,
        "ground_truth": _non_executable_ground("NEEDS_CLARIFICATION", "The stated attribute does not yield a unique exact target."),
        "verbalization": {
            "utterance_kind": "missing_attribute",
            "feature_type": focus["feature_type"],
            "face_role": "feature_region",
            "quantifier": "exactly_one",
            "operations": [{"operation_type": "local_mesh_size"}],
        },
    }


def _distance_to_bbox_min(item: dict[str, Any], bbox_min: list[float]) -> float:
    return math.sqrt(sum((float(item["centroid"][index]) - float(bbox_min[index])) ** 2 for index in range(3)))


def _plan_l4(
    groups: list[list[dict[str, Any]]],
    graph: dict[str, Any],
    offset: int,
) -> dict[str, Any]:
    multi_groups = [group for group in groups if len(group) >= 2 and all("feature_region" in item["roles"] for item in group)]
    bbox_min = [float(value) for value in graph["part_metadata_v2"]["raw_physical"]["axis_aligned_bbox_min"]]
    for group in multi_groups:
        if offset % 3 == 0:
            distances = [(item, _distance_to_bbox_min(item, bbox_min)) for item in group]
            distances.sort(key=lambda pair: pair[1])
            tolerance = _tolerance(distances[0][1])
            if distances[1][1] - distances[0][1] > tolerance:
                selected, value = distances[0]
                filters = {"nearest_to": {"reference_type": "bounding_box_min"}}
                evidence = {
                    "metric": "distance",
                    "value": _round(value),
                    "unit": "mm",
                    "source": "graph_centroid_computation",
                    "stable_tolerance": _round(tolerance),
                }
                return _single_valid_plan(
                    "L4",
                    group,
                    [selected],
                    "feature_region",
                    "local_mesh_size",
                    "nearest",
                    filters=filters,
                    reference_frame="global_cartesian",
                    uniqueness_kind="nearest",
                    rationale="Nearest-instance distance has a stable gap from the runner-up in the global frame.",
                    metric_evidence=[evidence],
                )
        metrics = sorted(set.intersection(*(set(item["metrics"]) for item in group)))
        if not metrics:
            continue
        metric = metrics[offset % len(metrics)]
        ranked = sorted(group, key=lambda item: float(item["metrics"][metric]["value"]), reverse=not bool(offset % 2))
        best = float(ranked[0]["metrics"][metric]["value"])
        second = float(ranked[1]["metrics"][metric]["value"])
        tolerance = _tolerance(best)
        if abs(best - second) <= tolerance:
            continue
        quantifier = "smallest" if offset % 2 else "largest"
        return _single_valid_plan(
            "L4",
            group,
            [ranked[0]],
            "feature_region",
            "local_mesh_size",
            quantifier,
            comparison_metric=metric,
            uniqueness_kind="comparison",
            rationale=f"The {quantifier} {metric} has a stable gap from the runner-up.",
            metric_evidence=[_metric_evidence(metric, ranked[0], best)],
        )

    focus = groups[0][0]
    result = _clarification_result(
        code="M_REFERENCE",
        field_path="/operations/0/target/filters/nearest_to",
        description="A nearest-target request requires an explicit spatial reference and at least two candidates.",
    )
    return {
        "level": "L4",
        "status": "NEEDS_CLARIFICATION",
        "intent_result": result,
        "ground_truth": _non_executable_ground("NEEDS_CLARIFICATION", "Nearest is not uniquely defined by the request."),
        "verbalization": {
            "utterance_kind": "missing_reference",
            "feature_type": focus["feature_type"],
            "face_role": "feature_region",
            "quantifier": "nearest",
            "operations": [{"operation_type": "local_mesh_size"}],
        },
    }


def _plan_l5_multi(groups: list[list[dict[str, Any]]], offset: int) -> dict[str, Any]:
    all_objects = [item for group in groups for item in group]
    displacement_target: tuple[list[dict[str, Any]], str, str] | None = None
    for group in groups:
        pair = _common_role_for_operation(group, ("displacement",))
        if pair is not None:
            displacement_target = (group, pair[0], pair[1])
            break
    if displacement_target is None:
        group = groups[0]
        pair = _common_role_for_operation(
            group,
            ("fixed_support", "pressure", "surface_traction", "local_mesh_size"),
            role_offset=offset,
        )
        if pair is None:
            raise ValueError("No feature operation for multi-operation plan")
        feature_group, role_name, feature_operation = group, pair[0], pair[1]
    else:
        feature_group, role_name, feature_operation = displacement_target
    quantifier = "all"
    target, feature_ground, verbal_target = _feature_target(
        feature_group,
        feature_group,
        role_name=role_name,
        operation_id="op3",
        operation_type=feature_operation,
        quantifier=quantifier,
        uniqueness_kind="all",
        rationale="All eligible source instances matching the feature-role contract are selected.",
    )
    operations = [
        _operation("op1", "material_assignment"),
        _operation("op2", "global_mesh_size"),
        _operation("op3", feature_operation, target),
    ]
    grounds = [_whole_ground("op1", "material_assignment"), _whole_ground("op2", "global_mesh_size"), feature_ground]
    return {
        "level": "L5",
        "status": "VALID",
        "intent_result": _valid_result(operations),
        "ground_truth": _executable_ground(grounds),
        "verbalization": {
            **verbal_target,
            "utterance_kind": "multi_operation",
            "operations": operations,
        },
    }


def _plan_l5_clarification(groups: list[list[dict[str, Any]]]) -> dict[str, Any]:
    for group in groups:
        pair = _common_role_for_operation(group, ("pressure",))
        if pair is not None:
            role_name, operation_type = pair
            result = _clarification_result(
                code="M_DIRECTION",
                field_path="/operations/0/parameters/direction",
                description="Pressure orientation must be surface_normal or opposite_surface_normal.",
            )
            verbal = {
                "utterance_kind": "missing_direction",
                "feature_type": group[0]["feature_type"],
                "face_role": role_name,
                "quantifier": "all",
                "operations": [{"operation_type": operation_type, "magnitude": 1.0, "unit": "MPa"}],
            }
            reason = "Pressure direction is missing; no executable intent or target face set is retained."
            break
    else:
        group = groups[0]
        role_name = "feature_region"
        operation_type = "local_mesh_size"
        result = _clarification_result(
            code="M_MESH_SIZE",
            field_path="/operations/0/parameters/size",
            description="A positive local mesh size and unit are required.",
        )
        verbal = {
            "utterance_kind": "missing_mesh_size",
            "feature_type": group[0]["feature_type"],
            "face_role": role_name,
            "quantifier": "all",
            "operations": [{"operation_type": operation_type}],
        }
        reason = "Local mesh size is missing; no executable intent or target face set is retained."
    return {
        "level": "L5",
        "status": "NEEDS_CLARIFICATION",
        "intent_result": result,
        "ground_truth": _non_executable_ground("NEEDS_CLARIFICATION", reason),
        "verbalization": verbal,
    }


def _plan_l5_invalid(groups: list[list[dict[str, Any]]]) -> dict[str, Any]:
    focus = groups[0][0]
    description = "Complex set difference/negation is outside frozen CAE-IR v1."
    return {
        "level": "L5",
        "status": "INVALID",
        "intent_result": _invalid_result(description),
        "ground_truth": _non_executable_ground("INVALID", description),
        "verbalization": {
            "utterance_kind": "unsupported_negation",
            "feature_type": focus["feature_type"],
            "face_role": "feature_region",
            "quantifier": "all",
            "comparison_metric": "area",
            "operations": [{"operation_type": "local_mesh_size", "size": 0.5, "unit": "mm"}],
        },
    }


def _semantic_fingerprint(plan: dict[str, Any]) -> str:
    intent_result = copy.deepcopy(plan["intent_result"])
    if intent_result.get("status") == "VALID":
        intent_result["intent"]["source_language"] = "__LANGUAGE__"
    return canonical_sha256(
        {
            "level": plan["level"],
            "intent_result": intent_result,
            "ground_truth": plan["ground_truth"],
            "verbalization": plan["verbalization"],
        }
    )


def generate_plans_for_source(
    selected_source: dict[str, Any],
    crosswalk: dict[str, Any],
    *,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    record = selected_source["record"]
    focus_feature = selected_source["focus_feature"]
    graph, objects = load_feature_objects(record, crosswalk)
    groups = _groups(objects, focus_feature)
    offset = int(_stable_score(seed, record["sample_id"])[:8], 16)
    plans = [
        _plan_l1(groups, offset),
        _plan_l2(groups, offset + 1),
        _plan_l3(groups, offset + 2),
        _plan_l4(groups, graph, offset + 3),
        _plan_l5_multi(groups, offset + 4),
        _plan_l5_clarification(groups),
        _plan_l5_invalid(groups),
    ]
    if not 4 <= len(plans) <= 8:
        raise AssertionError("Planner must emit 4-8 plans per selected graph")
    for ordinal, plan in enumerate(plans, start=1):
        plan["source_record"] = copy.deepcopy(record)
        plan["focus_feature"] = focus_feature
        plan["plan_ordinal"] = ordinal
        fingerprint = _semantic_fingerprint(plan)
        plan["canonical_semantics_sha256"] = fingerprint
        plan["canonical_record_id"] = f"CQG.{fingerprint[:24]}"
    return plans


def generate_pilot_plans(crosswalk: dict[str, Any], *, seed: int = SEED) -> list[dict[str, Any]]:
    selected = select_pilot_sources(seed=seed)
    plans: list[dict[str, Any]] = []
    for source in selected:
        plans.extend(generate_plans_for_source(source, crosswalk, seed=seed))
    ids = [plan["canonical_record_id"] for plan in plans]
    if len(ids) != len(set(ids)):
        raise ValueError("Canonical record IDs are not unique")
    return plans


def source_ref_for_plan(plan: dict[str, Any]) -> dict[str, Any]:
    record = plan["source_record"]
    return {
        "canonical_record_id": plan["canonical_record_id"],
        "canonical_semantics_sha256": plan["canonical_semantics_sha256"],
        "sample_id": record["sample_id"],
        "graph_path": str(assert_source_path(record["graph_path"])),
        "graph_sha256": str(record["graph_sha256"]).lower(),
        "geometry_path": str(assert_source_path(record["geometry_path"])),
        "geometry_format": record["geometry_format"],
        "base_geometry_family_id": record["base_geometry_family_id"],
        "source_split": record["split"],
        "dataset_split": DATASET_SPLIT_MAP[record["split"]],
        "derivation_kind": record["derivation_kind"],
        "exact_graph_signature": str(record["exact_graph_signature"]).lower(),
        "near_graph_signature": str(record["near_graph_signature"]).lower(),
    }


def make_record(
    plan: dict[str, Any],
    *,
    record_kind: str,
    query: dict[str, Any],
) -> dict[str, Any]:
    intent_result = copy.deepcopy(plan["intent_result"])
    if intent_result.get("status") == "VALID":
        intent_result["intent"]["source_language"] = query["language"]
    return {
        "record_schema_version": RECORD_VERSION,
        "record_kind": record_kind,
        "source_ref": source_ref_for_plan(plan),
        "query": copy.deepcopy(query),
        "intent_result": intent_result,
        "ground_truth": copy.deepcopy(plan["ground_truth"]),
    }
