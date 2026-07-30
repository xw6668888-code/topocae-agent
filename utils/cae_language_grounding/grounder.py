"""Deterministic CAE-IR to object/face-set grounding for P4-A1."""

from __future__ import annotations

import copy
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from utils.cae_intent import (
    CAEIntentValidationError,
    check_role_operation_compatibility,
    load_feature_role_policy,
    normalize_intent_units,
    validate_intent_result,
)
import utils.cae_intent as _p2_cae_intent

# The frozen P2 loader reparses a 90 KiB policy on every validation.  P4 treats
# that policy as immutable and memoizes the existing loader in-process without
# changing the frozen source file or its semantics.
_cached_policy_loader = lru_cache(maxsize=8)(_p2_cae_intent.load_feature_role_policy)
_p2_cae_intent.load_feature_role_policy = _cached_policy_loader
load_feature_role_policy = _cached_policy_loader

from .memory import validate_feature_object_memory


ROOT = Path(__file__).resolve().parents[2]
GROUNDING_SCHEMA_PATH = ROOT / "schemas" / "p4_a1_grounding_result.schema.json"
GROUNDING_SCHEMA_VERSION = "topocae-grounding-result/1.0.0"
INTERNAL_UNITS = {
    "length": "mm",
    "area": "mm2",
    "force": "N",
    "pressure": "MPa",
    "displacement": "mm",
    "angle": "deg",
}
WHOLE_PART_OPERATIONS = frozenset({"material_assignment", "global_mesh_size"})
QUANTITY_FILTERS = frozenset(
    {"diameter", "radius", "depth", "area", "length", "width", "thickness"}
)
STATUS_PRIORITY = {"VALID": 0, "NEEDS_CLARIFICATION": 1, "INVALID": 2}


class GroundingError(ValueError):
    """Raised for malformed inputs, never for normal verifier abstention."""


def _operation_result(
    operation: Mapping[str, Any],
    *,
    status: str,
    error_code: str | None = None,
    message: str = "",
    target_contract: str = "feature_face_set",
    quantifier: str | None = None,
    face_role: str | None = None,
    selected_object_ids: Sequence[str] = (),
    selected_face_indices: Sequence[int] = (),
    candidate_count_before_filters: int = 0,
    candidate_count_after_filters: int = 0,
    trace: Sequence[Mapping[str, Any]] = (),
    verifier_checks: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    checks = dict(verifier_checks or {})
    verifier_errors = [] if status == "VALID" else [message or str(error_code)]
    return {
        "operation_id": str(operation.get("operation_id", "")),
        "operation_type": str(operation.get("operation_type", "")),
        "status": status,
        "error_code": error_code,
        "message": message,
        "target_contract": target_contract,
        "quantifier": quantifier,
        "face_role": face_role,
        "selected_object_ids": sorted(set(str(item) for item in selected_object_ids)),
        "selected_face_indices": sorted(set(int(item) for item in selected_face_indices)),
        "candidate_count_before_filters": int(candidate_count_before_filters),
        "candidate_count_after_filters": int(candidate_count_after_filters),
        "trace": [copy.deepcopy(dict(item)) for item in trace],
        "verifier": {
            "status": status,
            "checks": checks,
            "errors": verifier_errors,
            "silent_fallback_used": False,
        },
    }


def _quantity_value(obj: Mapping[str, Any], name: str) -> float | None:
    parameters = obj.get("geometry_parameters", {})
    aliases = {
        "diameter": ("diameter",),
        "radius": ("radius",),
        "depth": ("depth", "profile_depth"),
        "length": ("length", "l", "path_length"),
        "width": ("width", "w", "profile_width"),
        "thickness": ("thickness",),
    }
    if isinstance(parameters, Mapping):
        for key in aliases.get(name, (name,)):
            value = parameters.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    descriptors = obj.get("descriptors", {})
    if name == "area" and isinstance(descriptors.get("area"), (int, float)):
        return float(descriptors["area"])
    if (
        descriptors.get("comparison_metric") == name
        and isinstance(descriptors.get("comparison_value"), (int, float))
    ):
        return float(descriptors["comparison_value"])
    return None


def _default_tolerance(expected: float) -> float:
    return max(1e-6, abs(expected) * 1e-6)


def _quantity_filter_matches(
    actual: float,
    filter_spec: Mapping[str, Any],
) -> bool:
    operator = str(filter_spec["operator"])
    quantity = filter_spec["value"]
    expected = float(quantity["value"])
    tolerance = float(quantity.get("tolerance", _default_tolerance(expected)))
    if operator == "eq":
        return math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)
    if operator == "lt":
        return actual < expected
    if operator == "lte":
        return actual <= expected
    if operator == "gt":
        return actual > expected
    if operator == "gte":
        return actual >= expected
    if operator == "between":
        upper = float(filter_spec["upper_value"]["value"])
        return expected <= actual <= upper
    raise GroundingError(f"Unsupported quantity filter operator: {operator!r}")


def _apply_filters(
    candidates: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]], str | None]:
    remaining = list(candidates)
    trace: list[dict[str, Any]] = []
    filters = target.get("filters", {})
    if not isinstance(filters, Mapping):
        raise GroundingError("Validated target filters must be an object")
    for name in sorted(filters):
        if name == "nearest_to":
            trace.append(
                {
                    "stage": "candidate_filter",
                    "filter": "nearest_to",
                    "action": "deferred_to_nearest_selector",
                    "remaining": len(remaining),
                }
            )
            continue
        if name == "spatial_relation":
            return (
                [],
                trace,
                "E_SPATIAL_RELATION_UNSUPPORTED",
            )
        if name not in QUANTITY_FILTERS:
            return [], trace, "E_FILTER_UNSUPPORTED"
        spec = filters[name]
        filtered: list[Mapping[str, Any]] = []
        unavailable = 0
        for obj in remaining:
            value = _quantity_value(obj, name)
            if value is None:
                unavailable += 1
                continue
            if _quantity_filter_matches(value, spec):
                filtered.append(obj)
        remaining = filtered
        trace.append(
            {
                "stage": "candidate_filter",
                "filter": name,
                "remaining": len(remaining),
                "descriptor_unavailable": unavailable,
            }
        )
    return remaining, trace, None


def _select_extreme(
    candidates: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    largest: bool,
) -> tuple[list[Mapping[str, Any]], str | None]:
    values: list[tuple[float, Mapping[str, Any]]] = []
    for obj in candidates:
        descriptors = obj["descriptors"]
        if (
            descriptors.get("comparison_metric") != metric
            or not isinstance(descriptors.get("comparison_value"), (int, float))
        ):
            return [], "E_COMPARISON_DESCRIPTOR_UNAVAILABLE"
        values.append((float(descriptors["comparison_value"]), obj))
    if not values:
        return [], "E_NO_CANDIDATE"
    ordered = sorted(values, key=lambda item: (item[0], item[1]["object_id"]))
    chosen_value = ordered[-1][0] if largest else ordered[0][0]
    scale = max(max(abs(value) for value, _ in ordered), 1e-9)
    tolerance = max(1e-6, scale * 0.015)
    tied = [
        obj
        for value, obj in ordered
        if math.isclose(value, chosen_value, rel_tol=0.0, abs_tol=tolerance)
    ]
    if len(tied) != 1:
        return [], "E_DESCRIPTOR_TIE"
    return tied, None


def _select_nearest(
    candidates: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], str | None]:
    reference = target.get("filters", {}).get("nearest_to", {})
    if reference.get("reference_type") != "part_centroid":
        return [], "E_NEAREST_REFERENCE_UNSUPPORTED"
    values = sorted(
        [
            (
                float(obj["descriptors"]["distance_to_part_reference"]),
                obj,
            )
            for obj in candidates
        ],
        key=lambda item: (item[0], item[1]["object_id"]),
    )
    if not values:
        return [], "E_NO_CANDIDATE"
    chosen = values[0][0]
    tied = [
        obj
        for value, obj in values
        if math.isclose(value, chosen, rel_tol=0.0, abs_tol=1e-6)
    ]
    if len(tied) != 1:
        return [], "E_SPATIAL_TIE"
    return tied, None


def _select_candidates(
    candidates: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], str | None]:
    quantifier = str(target["quantifier"])
    if not candidates:
        return [], "E_NO_CANDIDATE"
    if quantifier == "all":
        return list(candidates), None
    if quantifier == "count":
        expected = int(target["count"])
        if len(candidates) != expected:
            return [], "E_COUNT_MISMATCH"
        return list(candidates), None
    if quantifier in {"any_one", "exactly_one"}:
        if len(candidates) != 1:
            return [], "E_TARGET_NOT_UNIQUE"
        return [candidates[0]], None
    if quantifier in {"largest", "smallest"}:
        return _select_extreme(
            candidates,
            metric=str(target["comparison_metric"]),
            largest=quantifier == "largest",
        )
    if quantifier == "nearest":
        return _select_nearest(candidates, target)
    return [], "E_QUANTIFIER_UNSUPPORTED"


def _error_status(error_code: str) -> str:
    if error_code in {
        "E_COUNT_MISMATCH",
        "E_TARGET_NOT_UNIQUE",
        "E_DESCRIPTOR_TIE",
        "E_SPATIAL_TIE",
        "E_ROLE_UNAVAILABLE",
    }:
        return "NEEDS_CLARIFICATION"
    return "INVALID"


def _expand_role(
    selected: Sequence[Mapping[str, Any]],
    face_role: str,
) -> tuple[list[int], str | None]:
    faces: set[int] = set()
    for obj in selected:
        mappings = {
            str(item["face_role"]): item for item in obj["face_role_map"]
        }
        role = mappings.get(face_role)
        if role is None:
            return [], "E_ROLE_UNKNOWN"
        if role["status"] == "UNSUPPORTED":
            return [], "E_ROLE_UNSUPPORTED"
        if role["status"] != "AVAILABLE":
            return [], "E_ROLE_UNAVAILABLE"
        faces.update(int(index) for index in role["face_indices"])
    if not faces:
        return [], "E_EMPTY_FACE_SET"
    return sorted(faces), None


def _verify_operation(
    *,
    operation: Mapping[str, Any],
    memory: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    faces: Sequence[int],
    policy: Mapping[str, Any],
) -> tuple[dict[str, bool], list[str]]:
    target = operation["target"]
    maximum = int(memory["part"]["face_index_contract"]["maximum"])
    checks = {
        "memory_legal": memory["memory_legality"]["status"] == "VALID",
        "selected_objects_exist": all(
            obj in memory["objects"] for obj in selected
        ),
        "face_set_nonempty": bool(faces),
        "face_indices_in_range": all(0 <= int(index) <= maximum for index in faces),
        "face_indices_unique": len(faces) == len(set(faces)),
        "role_operation_compatible": False,
        "selected_faces_within_objects": False,
        "silent_fallback_absent": True,
    }
    compatibility = check_role_operation_compatibility(
        policy,
        str(target["feature_type"]),
        str(target["face_role"]),
        str(operation["operation_type"]),
    )
    checks["role_operation_compatible"] = bool(compatibility["compatible"])
    object_faces = {
        int(index)
        for obj in selected
        for index in obj["topology"]["complete_instance_face_indices"]
    }
    checks["selected_faces_within_objects"] = set(faces).issubset(object_faces)
    errors = [name for name, passed in checks.items() if not passed]
    return checks, errors


def _ground_feature_operation(
    operation: Mapping[str, Any],
    memory: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    target = operation["target"]
    feature_type = str(target["feature_type"])
    quantifier = str(target["quantifier"])
    face_role = str(target["face_role"])
    candidates = [
        obj
        for obj in memory["objects"]
        if obj["canonical_feature_type"] == feature_type
        and obj["legality"]["status"] != "INVALID"
    ]
    before = len(candidates)
    filtered, filter_trace, filter_error = _apply_filters(candidates, target)
    if filter_error:
        status = _error_status(filter_error)
        return _operation_result(
            operation,
            status=status,
            error_code=filter_error,
            message=f"Candidate filtering failed: {filter_error}",
            quantifier=quantifier,
            face_role=face_role,
            candidate_count_before_filters=before,
            candidate_count_after_filters=0,
            trace=filter_trace,
        )
    selected, selection_error = _select_candidates(filtered, target)
    selection_trace = {
        "stage": "quantifier_selection",
        "quantifier": quantifier,
        "candidate_count": len(filtered),
        "selected_object_ids": sorted(
            str(obj["object_id"]) for obj in selected
        ),
    }
    if selection_error:
        status = _error_status(selection_error)
        return _operation_result(
            operation,
            status=status,
            error_code=selection_error,
            message=f"Deterministic selection failed: {selection_error}",
            quantifier=quantifier,
            face_role=face_role,
            candidate_count_before_filters=before,
            candidate_count_after_filters=len(filtered),
            trace=[*filter_trace, selection_trace],
        )
    faces, role_error = _expand_role(selected, face_role)
    role_trace = {
        "stage": "face_role_expansion",
        "face_role": face_role,
        "face_count": len(faces),
    }
    if role_error:
        status = _error_status(role_error)
        return _operation_result(
            operation,
            status=status,
            error_code=role_error,
            message=f"Semantic face-role expansion failed: {role_error}",
            quantifier=quantifier,
            face_role=face_role,
            selected_object_ids=[obj["object_id"] for obj in selected],
            candidate_count_before_filters=before,
            candidate_count_after_filters=len(filtered),
            trace=[*filter_trace, selection_trace, role_trace],
        )
    checks, verifier_errors = _verify_operation(
        operation=operation,
        memory=memory,
        selected=selected,
        faces=faces,
        policy=policy,
    )
    if verifier_errors:
        return _operation_result(
            operation,
            status="INVALID",
            error_code="E_VERIFIER_REJECTION",
            message="Verifier rejected: " + ", ".join(verifier_errors),
            quantifier=quantifier,
            face_role=face_role,
            selected_object_ids=[obj["object_id"] for obj in selected],
            selected_face_indices=faces,
            candidate_count_before_filters=before,
            candidate_count_after_filters=len(filtered),
            trace=[*filter_trace, selection_trace, role_trace],
            verifier_checks=checks,
        )
    return _operation_result(
        operation,
        status="VALID",
        quantifier=quantifier,
        face_role=face_role,
        selected_object_ids=[obj["object_id"] for obj in selected],
        selected_face_indices=faces,
        candidate_count_before_filters=before,
        candidate_count_after_filters=len(filtered),
        trace=[*filter_trace, selection_trace, role_trace],
        verifier_checks=checks,
    )


def _ground_whole_part_operation(
    operation: Mapping[str, Any],
) -> dict[str, Any]:
    operation_type = str(operation["operation_type"])
    valid = (
        operation_type in WHOLE_PART_OPERATIONS
        and operation.get("target", {}).get("scope") == "whole_part"
    )
    return _operation_result(
        operation,
        status="VALID" if valid else "INVALID",
        error_code=None if valid else "E_WHOLE_PART_CONTRACT",
        message="" if valid else "Whole-part target/operation contract mismatch",
        target_contract="whole_part",
        trace=[
            {
                "stage": "whole_part_selection",
                "scope": "whole_part",
                "grounding_required": False,
            }
        ],
        verifier_checks={
            "whole_part_operation_allowed": valid,
            "silent_fallback_absent": True,
        },
    )


@lru_cache(maxsize=1)
def _grounding_validator() -> Draft202012Validator:
    schema = json.loads(GROUNDING_SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def validate_grounding_result(value: Any) -> dict[str, Any]:
    errors = [
        {
            "path": "/" + "/".join(str(item) for item in failure.absolute_path),
            "message": failure.message,
        }
        for failure in sorted(
            _grounding_validator().iter_errors(value),
            key=lambda item: list(item.absolute_path),
        )
    ]
    return {"valid": not errors, "errors": errors}


def ground_intent_result(
    intent_result: Mapping[str, Any],
    memory: Mapping[str, Any],
    *,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Ground validated P2 output without accepting or inspecting query text."""

    ir_validation = validate_intent_result(intent_result)
    if not ir_validation["valid"]:
        raise GroundingError(
            "Input is not a valid P2 cae-intent-result: "
            + json.dumps(ir_validation["errors"][:10], ensure_ascii=False)
        )
    memory_validation = validate_feature_object_memory(memory)
    if not memory_validation["valid"]:
        raise GroundingError(
            "Input feature-object memory is invalid: "
            + json.dumps(memory_validation["errors"][:10], ensure_ascii=False)
        )
    input_status = str(intent_result["status"])
    if input_status != "VALID":
        result = {
            "schema_version": GROUNDING_SCHEMA_VERSION,
            "status": input_status,
            "input_intent_status": input_status,
            "part_ref": str(memory["part"]["part_ref"]),
            "error_code": str(intent_result.get("error_code", "")) or None,
            "message": "P2 returned a non-executable intent result; grounding not run.",
            "operations": [],
            "verifier": {
                "status": "NOT_RUN",
                "all_operations_valid": False,
                "illegal_face_id_count": 0,
                "silent_fallback_count": 0,
            },
        }
        validation = validate_grounding_result(result)
        if not validation["valid"]:
            raise GroundingError(json.dumps(validation["errors"], ensure_ascii=False))
        return result

    actual_policy = policy if policy is not None else load_feature_role_policy()
    try:
        intent = normalize_intent_units(
            intent_result["intent"],
            target_units=INTERNAL_UNITS,
        )
    except CAEIntentValidationError as exc:
        raise GroundingError(f"Validated P2 intent failed normalization: {exc}") from exc
    operation_results = []
    for operation in intent["operations"]:
        if operation["operation_type"] in WHOLE_PART_OPERATIONS:
            operation_results.append(_ground_whole_part_operation(operation))
        else:
            operation_results.append(
                _ground_feature_operation(operation, memory, actual_policy)
            )
    status = max(
        (item["status"] for item in operation_results),
        key=lambda item: STATUS_PRIORITY[item],
    )
    first_error = next(
        (item for item in operation_results if item["status"] != "VALID"),
        None,
    )
    illegal_count = sum(
        not item["verifier"]["checks"].get("face_indices_in_range", True)
        for item in operation_results
    )
    silent_count = sum(
        bool(item["verifier"]["silent_fallback_used"])
        for item in operation_results
    )
    result = {
        "schema_version": GROUNDING_SCHEMA_VERSION,
        "status": status,
        "input_intent_status": input_status,
        "part_ref": str(memory["part"]["part_ref"]),
        "error_code": first_error["error_code"] if first_error else None,
        "message": first_error["message"] if first_error else "",
        "operations": operation_results,
        "verifier": {
            "status": status,
            "all_operations_valid": status == "VALID",
            "illegal_face_id_count": illegal_count,
            "silent_fallback_count": silent_count,
        },
    }
    validation = validate_grounding_result(result)
    if not validation["valid"]:
        raise GroundingError(
            "Grounding result violates its schema: "
            + json.dumps(validation["errors"][:10], ensure_ascii=False)
        )
    return result


__all__ = [
    "GroundingError",
    "ground_intent_result",
    "validate_grounding_result",
]
