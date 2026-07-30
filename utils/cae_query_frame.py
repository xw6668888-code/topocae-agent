"""Deterministic CAEQueryFrame v1 compiler for the frozen P2 CAE-IR."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator

from utils.cae_intent import validate_intent_result


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = (
    ROOT / "reports" / "p4_api_r2a_compact_query_frame_v1"
    / "query_frame.schema.json"
)
FRAME_VERSION = "cae-query-frame/1.0.0"
RESULT_VERSION = "cae-intent-result/1.0.0"
INTENT_VERSION = "cae-intent/1.0.0"
DEFAULT_UNIT_SYSTEM = {
    "length": "mm",
    "area": "mm2",
    "force": "N",
    "pressure": "MPa",
    "displacement": "mm",
    "angle": "deg",
}
FEATURE_ACTIONS = frozenset({
    "fixed_support", "displacement", "pressure", "surface_traction",
    "local_mesh_size",
})
WHOLE_PART_ACTIONS = frozenset({"material_assignment", "global_mesh_size"})
ACTIONS = FEATURE_ACTIONS | WHOLE_PART_ACTIONS
SELECTORS = frozenset({
    "all", "any_one", "exactly_one", "count", "largest", "smallest", "nearest",
})
DESCRIPTORS = (
    "diameter", "radius", "depth", "area", "length", "width", "thickness",
)
ROOT_VALID_FIELDS = frozenset({"frame_version", "status", "language", "operations"})
ROOT_ISSUE_FIELDS = frozenset({
    "frame_version", "status", "error_code", "error_path",
    "missing_information", "ambiguities", "clarification_questions",
})
TARGET_FIELDS = frozenset({
    "feature", "face_role", "selector", "instance_count", "compare_by",
    "geometry_filters", "selection_frame",
})
PARAMETER_FIELDS = frozenset({
    "constrained_dofs", "displacements", "load_frame", "magnitude",
    "direction", "material", "mesh_size",
})
OPERATION_COMMON_FIELDS = frozenset({"operation_key", "action"})
FORBIDDEN_IDENTITY_KEYS = frozenset({
    "face_id", "face_ids", "face_index", "face_indices", "feature_instance_id",
    "feature_instance_ids", "instance_id", "instance_ids", "ground_truth",
    "source_identity", "source_ref", "target_face_indices", "target_instance_ids",
})


class QueryFrameCompileError(ValueError):
    """Stable fail-closed error returned by the deterministic compiler."""

    def __init__(self, code: str, field_path: str, message: str):
        super().__init__(f"{code} at {field_path}: {message}")
        self.code = code
        self.field_path = field_path if field_path.startswith("/") else "/"
        self.message = message

    def as_dict(self) -> dict[str, str]:
        return {
            "error_code": self.code,
            "field_path": self.field_path,
            "message": self.message,
        }


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _pointer(parts: Iterable[Any]) -> str:
    encoded = [str(part).replace("~", "~0").replace("/", "~1") for part in parts]
    return "/" + "/".join(encoded) if encoded else "/"


def _raise(code: str, path: str, message: str) -> None:
    raise QueryFrameCompileError(code, path, message)


def _find_forbidden_identity(value: Any, path: tuple[Any, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = path + (key,)
            if str(key).lower() in FORBIDDEN_IDENTITY_KEYS:
                _raise("QF_E_IDENTITY_FIELD_FORBIDDEN", _pointer(child_path), "identity field is forbidden")
            _find_forbidden_identity(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _find_forbidden_identity(child, path + (index,))


@lru_cache(maxsize=1)
def _schema_validator() -> Draft202012Validator:
    return Draft202012Validator(load_schema())

def _schema_error(value: Mapping[str, Any]) -> QueryFrameCompileError | None:
    validator = _schema_validator()
    errors = sorted(
        validator.iter_errors(value),
        key=lambda error: (
            tuple(str(part) for part in error.absolute_path),
            tuple(str(part) for part in error.absolute_schema_path),
            error.message,
        ),
    )
    if not errors:
        return None
    error = errors[0]
    path = list(error.absolute_path)
    if error.validator == "required":
        match = re.search(r"'([^']+)' is a required property", error.message)
        if match:
            path.append(match.group(1))
        code = "QF_E_REQUIRED_FIELD"
    elif error.validator == "additionalProperties":
        match = re.search(r"\('([^']+)' was unexpected\)", error.message)
        if match:
            path.append(match.group(1))
        code = "QF_E_UNEXPECTED_FIELD"
    elif error.validator in {"enum", "const"}:
        code = "QF_E_INVALID_ENUM"
    else:
        code = "QF_E_SCHEMA_VALIDATION"
    return QueryFrameCompileError(code, _pointer(path), error.message)


def _require_fields(value: Mapping[str, Any], required: Iterable[str], path: str) -> None:
    for field in required:
        if field not in value:
            _raise("QF_E_REQUIRED_FIELD", f"{path}/{field}".replace("//", "/"), "required field is missing")


def _reject_fields(value: Mapping[str, Any], allowed: frozenset[str], path: str) -> None:
    extras = sorted(set(value) - allowed)
    if extras:
        _raise("QF_E_CONFLICT", f"{path}/{extras[0]}".replace("//", "/"), "field conflicts with selected branch")


def _validate_quantity_filter(item: Mapping[str, Any], path: str) -> None:
    operator = item["operator"]
    has_upper = "upper_value" in item or "upper_unit" in item or "upper_tolerance" in item
    if operator == "between":
        _require_fields(item, ("upper_value", "upper_unit"), path)
    elif has_upper:
        _raise("QF_E_CONFLICT", f"{path}/upper_value", "upper bound is only valid for between")


def _validate_target(operation: Mapping[str, Any], path: str) -> None:
    selector = operation["selector"]
    if selector == "count":
        _require_fields(operation, ("instance_count",), path)
    elif "instance_count" in operation:
        _raise("QF_E_CONFLICT", f"{path}/instance_count", "instance_count requires selector=count")
    if selector in {"largest", "smallest"}:
        _require_fields(operation, ("compare_by",), path)
    elif "compare_by" in operation:
        _raise("QF_E_CONFLICT", f"{path}/compare_by", "compare_by requires largest or smallest")

    filters = operation.get("geometry_filters", {})
    if not isinstance(filters, Mapping):
        return
    for descriptor in DESCRIPTORS:
        if descriptor in filters:
            _validate_quantity_filter(filters[descriptor], f"{path}/geometry_filters/{descriptor}")
    needs_frame = "spatial_relation" in filters or "nearest_to" in filters
    if selector == "nearest":
        if "nearest_to" not in filters:
            _raise("QF_E_REQUIRED_FIELD", f"{path}/geometry_filters/nearest_to", "nearest selector requires nearest_to")
        needs_frame = True
    if needs_frame and "selection_frame" not in operation:
        _raise("QF_E_REQUIRED_FIELD", f"{path}/selection_frame", "spatial selection requires selection_frame")
    if not needs_frame and "selection_frame" in operation:
        _raise("QF_E_CONFLICT", f"{path}/selection_frame", "selection_frame has no spatial selector")


def _validate_operation(operation: Mapping[str, Any], index: int) -> None:
    path = f"/operations/{index}"
    action = operation["action"]
    if action in FEATURE_ACTIONS:
        _require_fields(operation, ("feature", "face_role", "selector"), path)
        allowed = OPERATION_COMMON_FIELDS | TARGET_FIELDS
        _validate_target(operation, path)
    else:
        allowed = OPERATION_COMMON_FIELDS

    action_parameters = {
        "fixed_support": frozenset({"constrained_dofs"}),
        "displacement": frozenset({"displacements", "load_frame"}),
        "pressure": frozenset({"magnitude", "direction"}),
        "surface_traction": frozenset({"magnitude", "direction"}),
        "material_assignment": frozenset({"material"}),
        "global_mesh_size": frozenset({"mesh_size"}),
        "local_mesh_size": frozenset({"mesh_size"}),
    }[action]
    _require_fields(operation, action_parameters, path)
    allowed |= action_parameters
    _reject_fields(operation, allowed, path)

    if action == "fixed_support":
        dofs = operation["constrained_dofs"]
        if len(dofs) != 3 or len(set(dofs)) != 3:
            _raise("QF_E_CONFLICT", f"{path}/constrained_dofs", "fixed support requires U1, U2 and U3 exactly once")
    if action == "displacement":
        dofs = [item["dof"] for item in operation["displacements"]]
        if len(dofs) != len(set(dofs)):
            _raise("QF_E_CONFLICT", f"{path}/displacements", "displacement DOFs must be unique")


def _semantic_validate(frame: Mapping[str, Any]) -> None:
    status = frame["status"]
    if status == "valid":
        _require_fields(frame, ("language", "operations"), "/")
        _reject_fields(frame, ROOT_VALID_FIELDS, "/")
        for index, operation in enumerate(frame["operations"]):
            _validate_operation(operation, index)
        ids = [operation.get("operation_key", f"op{index + 1}") for index, operation in enumerate(frame["operations"])]
        if len(ids) != len(set(ids)):
            _raise("QF_E_OPERATION_ID_DUPLICATE", "/operations", "operation keys must be unique")
    else:
        _require_fields(
            frame,
            ("error_code", "error_path", "missing_information", "ambiguities", "clarification_questions"),
            "/",
        )
        _reject_fields(frame, ROOT_ISSUE_FIELDS, "/")
        if status == "needs_clarification":
            if frame["error_code"] not in {"E_MISSING_INFORMATION", "E_AMBIGUOUS_REQUIREMENT"}:
                _raise("QF_E_INVALID_ENUM", "/error_code", "clarification status has incompatible error code")
            if not frame["missing_information"] and not frame["ambiguities"]:
                _raise("QF_E_REQUIRED_FIELD", "/missing_information", "clarification requires missing information or ambiguity")
            if not frame["clarification_questions"]:
                _raise("QF_E_REQUIRED_FIELD", "/clarification_questions", "clarification requires a question")


def validate_query_frame(frame: Any) -> dict[str, Any]:
    try:
        if not isinstance(frame, Mapping):
            _raise("QF_E_JSON_TYPE", "/", "query frame root must be an object")
        _find_forbidden_identity(frame)
        error = _schema_error(frame)
        if error is not None:
            raise error
        _semantic_validate(frame)
    except QueryFrameCompileError as error:
        return {"valid": False, "errors": [error.as_dict()]}
    return {"valid": True, "errors": []}


def _quantity(value: Mapping[str, Any], dimension: str) -> dict[str, Any]:
    result = {"dimension": dimension, "value": value["value"], "unit": value["unit"]}
    if "tolerance" in value:
        result["tolerance"] = value["tolerance"]
    return result


def _reference(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {"reference_type": value["kind"]}
    if "axis" in value:
        result["axis"] = value["axis"]
    if "feature" in value:
        result["feature_type"] = value["feature"]
    if "face_role" in value:
        result["face_role"] = value["face_role"]
    return result


def _quantity_filter(value: Mapping[str, Any], dimension: str) -> dict[str, Any]:
    core = {"value": value["value"], "unit": value["unit"]}
    if "tolerance" in value:
        core["tolerance"] = value["tolerance"]
    result = {"operator": value["operator"], "value": _quantity(core, dimension)}
    if "upper_value" in value:
        upper = {"value": value["upper_value"], "unit": value["upper_unit"]}
        if "upper_tolerance" in value:
            upper["tolerance"] = value["upper_tolerance"]
        result["upper_value"] = _quantity(upper, dimension)
    return result


def _filters(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for descriptor in DESCRIPTORS:
        if descriptor in value:
            dimension = "area" if descriptor == "area" else "length"
            result[descriptor] = _quantity_filter(value[descriptor], dimension)
    if "nearest_to" in value:
        result["nearest_to"] = _reference(value["nearest_to"])
    if "spatial_relation" in value:
        spatial = value["spatial_relation"]
        compiled = {
            "relation": spatial["relation"],
            "reference": _reference(spatial["reference"]),
        }
        if "distance" in spatial:
            compiled["distance"] = _quantity(spatial["distance"], "length")
        result["spatial_relation"] = compiled
    return result


def _direction(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {"kind": value["kind"]}
    if "vector" in value:
        result["vector"] = list(value["vector"])
    return result


def _target(operation: Mapping[str, Any]) -> dict[str, Any]:
    if operation["action"] in WHOLE_PART_ACTIONS:
        return {"scope": "whole_part"}
    result: dict[str, Any] = {
        "feature_type": operation["feature"],
        "quantifier": operation["selector"],
        "face_role": operation["face_role"],
        "filters": _filters(operation.get("geometry_filters", {})),
    }
    if "instance_count" in operation:
        result["count"] = operation["instance_count"]
    if "compare_by" in operation:
        result["comparison_metric"] = operation["compare_by"]
    if "selection_frame" in operation:
        result["reference_frame"] = {"kind": operation["selection_frame"]}
    return result


def _parameters(operation: Mapping[str, Any]) -> dict[str, Any]:
    action = operation["action"]
    if action == "fixed_support":
        return {"constrained_dofs": list(operation["constrained_dofs"])}
    if action == "displacement":
        components = []
        for item in operation["displacements"]:
            quantity = {key: item[key] for key in ("value", "unit", "tolerance") if key in item}
            components.append({"dof": item["dof"], "value": _quantity(quantity, "displacement")})
        return {"components": components, "reference_frame": {"kind": operation["load_frame"]}}
    if action in {"pressure", "surface_traction"}:
        return {
            "magnitude": _quantity(operation["magnitude"], "pressure"),
            "direction": _direction(operation["direction"]),
        }
    if action == "material_assignment":
        material = operation["material"]
        return {
            "material_name": material["name"],
            "elastic": {
                "youngs_modulus": _quantity(material["youngs_modulus"], "pressure"),
                "poisson_ratio": material["poisson_ratio"],
            },
        }
    return {"size": _quantity(operation["mesh_size"], "length")}


def _compile_valid(frame: Mapping[str, Any]) -> dict[str, Any]:
    operations = []
    for index, operation in enumerate(frame["operations"]):
        operations.append({
            "operation_id": operation.get("operation_key", f"op{index + 1}"),
            "operation_type": operation["action"],
            "target": _target(operation),
            "parameters": _parameters(operation),
        })
    return {
        "schema_version": RESULT_VERSION,
        "status": "VALID",
        "intent": {
            "schema_version": INTENT_VERSION,
            "analysis_type": "static_structural",
            "operations": operations,
            "source_language": frame["language"],
            "assumptions": [],
            "unit_system": dict(DEFAULT_UNIT_SYSTEM),
            "normalization_audit": [],
        },
    }


def _compile_nonvalid(frame: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": RESULT_VERSION,
        "status": "NEEDS_CLARIFICATION" if frame["status"] == "needs_clarification" else "INVALID",
        "error_code": frame["error_code"],
        "field_path": frame["error_path"],
        "missing_information": [{
            "code": item["code"],
            "field_path": item["path"],
            "description": item["description"],
        } for item in frame["missing_information"]],
        "ambiguities": [{
            "code": item["code"],
            "field_path": item["path"],
            "description": item["description"],
            "options": list(item["options"]),
        } for item in frame["ambiguities"]],
        "clarification_questions": [{
            "question_id": item["key"],
            "field_path": item["path"],
            "question": item["question"],
        } for item in frame["clarification_questions"]],
    }


def compile_query_frame(frame: Any) -> dict[str, Any]:
    validation = validate_query_frame(frame)
    if not validation["valid"]:
        error = validation["errors"][0]
        raise QueryFrameCompileError(error["error_code"], error["field_path"], error["message"])
    compiled = _compile_valid(frame) if frame["status"] == "valid" else _compile_nonvalid(frame)
    canonical_validation = validate_intent_result(compiled)
    if not canonical_validation["valid"]:
        first = canonical_validation["errors"][0]
        _raise(
            "QF_E_CANONICAL_REJECTED",
            str(first.get("field_path") or "/"),
            str(first.get("message") or first.get("error_code") or "frozen validator rejected output"),
        )
    return compiled


def compile_query_frame_json(frame: Any) -> bytes:
    return json.dumps(
        compile_query_frame(frame),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _frame_quantity(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {"value": value["value"], "unit": value["unit"]}
    if "tolerance" in value:
        result["tolerance"] = value["tolerance"]
    return result


def _frame_reference(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {"kind": value["reference_type"]}
    if "axis" in value:
        result["axis"] = value["axis"]
    if "feature_type" in value:
        result["feature"] = value["feature_type"]
    if "face_role" in value:
        result["face_role"] = value["face_role"]
    return result


def _frame_quantity_filter(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {"operator": value["operator"], **_frame_quantity(value["value"])}
    if "upper_value" in value:
        upper = _frame_quantity(value["upper_value"])
        result["upper_value"] = upper["value"]
        result["upper_unit"] = upper["unit"]
        if "tolerance" in upper:
            result["upper_tolerance"] = upper["tolerance"]
    return result


def _frame_filters(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for descriptor in DESCRIPTORS:
        if descriptor in value:
            result[descriptor] = _frame_quantity_filter(value[descriptor])
    if "nearest_to" in value:
        result["nearest_to"] = _frame_reference(value["nearest_to"])
    if "spatial_relation" in value:
        spatial = value["spatial_relation"]
        compiled = {
            "relation": spatial["relation"],
            "reference": _frame_reference(spatial["reference"]),
        }
        if "distance" in spatial:
            compiled["distance"] = _frame_quantity(spatial["distance"])
        result["spatial_relation"] = compiled
    return result


def _frame_operation(operation: Mapping[str, Any], index: int) -> dict[str, Any]:
    action = operation["operation_type"]
    frame: dict[str, Any] = {"action": action}
    default_id = f"op{index + 1}"
    if operation["operation_id"] != default_id:
        frame["operation_key"] = operation["operation_id"]
    target = operation["target"]
    if action in FEATURE_ACTIONS:
        frame.update({
            "feature": target["feature_type"],
            "face_role": target["face_role"],
            "selector": target["quantifier"],
        })
        if "count" in target:
            frame["instance_count"] = target["count"]
        if "comparison_metric" in target:
            frame["compare_by"] = target["comparison_metric"]
        if target["filters"]:
            frame["geometry_filters"] = _frame_filters(target["filters"])
        if "reference_frame" in target:
            frame["selection_frame"] = target["reference_frame"]["kind"]
    parameters = operation["parameters"]
    if action == "fixed_support":
        frame["constrained_dofs"] = list(parameters["constrained_dofs"])
    elif action == "displacement":
        frame["displacements"] = [
            {"dof": item["dof"], **_frame_quantity(item["value"])}
            for item in parameters["components"]
        ]
        frame["load_frame"] = parameters["reference_frame"]["kind"]
    elif action in {"pressure", "surface_traction"}:
        frame["magnitude"] = _frame_quantity(parameters["magnitude"])
        frame["direction"] = deepcopy(parameters["direction"])
    elif action == "material_assignment":
        frame["material"] = {
            "name": parameters["material_name"],
            "youngs_modulus": _frame_quantity(parameters["elastic"]["youngs_modulus"]),
            "poisson_ratio": parameters["elastic"]["poisson_ratio"],
        }
    else:
        frame["mesh_size"] = _frame_quantity(parameters["size"])
    return frame


def canonical_to_query_frame(result: Mapping[str, Any]) -> dict[str, Any]:
    canonical_validation = validate_intent_result(result)
    if not canonical_validation["valid"]:
        first = canonical_validation["errors"][0]
        _raise("QF_E_CANONICAL_INPUT_INVALID", str(first.get("field_path") or "/"), "canonical input is invalid")
    status = result["status"]
    if status == "VALID":
        intent = result["intent"]
        fixed = (
            intent["schema_version"] == INTENT_VERSION
            and intent["analysis_type"] == "static_structural"
            and intent["assumptions"] == []
            and intent["unit_system"] == DEFAULT_UNIT_SYSTEM
            and intent["normalization_audit"] == []
        )
        if not fixed:
            _raise("QF_E_CANONICAL_UNREPRESENTABLE", "/intent", "canonical intent uses context outside QueryFrame v1")
        frame = {
            "frame_version": FRAME_VERSION,
            "status": "valid",
            "language": intent["source_language"],
            "operations": [
                _frame_operation(operation, index)
                for index, operation in enumerate(intent["operations"])
            ],
        }
    else:
        frame = {
            "frame_version": FRAME_VERSION,
            "status": "needs_clarification" if status == "NEEDS_CLARIFICATION" else "invalid",
            "error_code": result["error_code"],
            "error_path": result["field_path"],
            "missing_information": [{
                "code": item["code"],
                "path": item["field_path"],
                "description": item["description"],
            } for item in result["missing_information"]],
            "ambiguities": [{
                "code": item["code"],
                "path": item["field_path"],
                "description": item["description"],
                "options": list(item["options"]),
            } for item in result["ambiguities"]],
            "clarification_questions": [{
                "key": item["question_id"],
                "path": item["field_path"],
                "question": item["question"],
            } for item in result["clarification_questions"]],
        }
    validation = validate_query_frame(frame)
    if not validation["valid"]:
        first = validation["errors"][0]
        raise QueryFrameCompileError(first["error_code"], first["field_path"], first["message"])
    return frame


__all__ = [
    "FRAME_VERSION", "QueryFrameCompileError", "canonical_to_query_frame",
    "compile_query_frame", "compile_query_frame_json", "load_schema",
    "validate_query_frame",
]
