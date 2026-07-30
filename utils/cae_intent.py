"""Strict CAE-IR v1 parsing, validation, units, and role-policy helpers.

This module deliberately does not parse natural language, call a model, select geometry
sources, resolve face IDs, or execute a solver.
"""

from __future__ import annotations

import copy
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from jsonschema import Draft202012Validator
from referencing import Registry, Resource


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schemas"
POLICY_PATH = ROOT / "configs" / "cae_feature_role_policy.yaml"

FORBIDDEN_IDENTITY_KEYS = {
    "raw_face_id", "face_id", "face_ids", "faces",
    "sample_id", "part_id", "model_file", "model_filename",
    "file_name", "filename", "area_truth", "truth_area",
}
GROUNDING_ONLY_KEYS = {
    "geometry_source", "recognizer_capability_level", "coverage_status",
    "abstention_reasons", "feature_object_schema_version", "recognizer_name",
}
WHOLE_PART_OPERATIONS = {"material_assignment", "global_mesh_size"}

# Factor converts a value into the dimension's internal reference unit.
# References are mm, mm2, N, MPa, mm, and deg respectively.
UNIT_FACTORS: dict[str, dict[str, float]] = {
    "length": {"um": 0.001, "mm": 1.0, "cm": 10.0, "m": 1000.0},
    "area": {"mm2": 1.0, "cm2": 100.0, "m2": 1_000_000.0},
    "force": {"N": 1.0, "kN": 1000.0, "MN": 1_000_000.0},
    "pressure": {"Pa": 1e-6, "kPa": 1e-3, "MPa": 1.0, "GPa": 1000.0, "N/mm2": 1.0},
    "displacement": {"um": 0.001, "mm": 1.0, "cm": 10.0, "m": 1000.0},
    "angle": {"deg": 1.0, "rad": 180.0 / math.pi},
}
FILTER_DIMENSIONS = {
    "diameter": "length", "radius": "length", "depth": "length",
    "area": "area", "length": "length", "width": "length", "thickness": "length",
}


class CAEIntentValidationError(ValueError):
    """Raised by strict parsing/normalization with stable structured errors."""

    def __init__(self, errors: Sequence[Mapping[str, Any]]):
        self.errors = [dict(error) for error in errors]
        super().__init__(json.dumps(self.errors, ensure_ascii=False, sort_keys=True))


def _pointer(parts: Sequence[Any]) -> str:
    if not parts:
        return "/"
    encoded = [str(part).replace("~", "~0").replace("/", "~1") for part in parts]
    return "/" + "/".join(encoded)


def _error(code: str, path: Sequence[Any] | str, message: str) -> dict[str, str]:
    return {
        "error_code": code,
        "field_path": path if isinstance(path, str) else _pointer(path),
        "message": message,
    }


def _walk(value: Any, path: tuple[Any, ...] = ()):
    yield path, value
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk(item, path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, path + (index,))


def _load_schema(name: str) -> dict[str, Any]:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


@lru_cache(maxsize=3)
def _validator(name: str) -> Draft202012Validator:
    schemas = {
        schema_name: _load_schema(schema_name)
        for schema_name in (
            "cae_intent.schema.json",
            "cae_intent_result.schema.json",
            "cae_grounding_context.schema.json",
        )
    }
    registry = Registry().with_resources(
        [(schema["$id"], Resource.from_contents(schema)) for schema in schemas.values()]
    )
    return Draft202012Validator(schemas[name], registry=registry)


def _schema_errors(value: Any, schema_name: str) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for failure in sorted(_validator(schema_name).iter_errors(value), key=lambda item: list(item.absolute_path)):
        leaves = [failure]
        while any(item.context for item in leaves):
            expanded = []
            for item in leaves:
                expanded.extend(item.context or [item])
            leaves = expanded
        for leaf in leaves:
            errors.append(_error("E_SCHEMA_VALIDATION", tuple(leaf.absolute_path), leaf.message))
    return errors


def _deduplicate_errors(errors: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for error in errors:
        key = (str(error["error_code"]), str(error["field_path"]), str(error.get("message", "")))
        unique[key] = dict(error)
    return sorted(unique.values(), key=lambda item: (item["field_path"], item["error_code"], item.get("message", "")))


def _canonical_alias_map(policy: Mapping[str, Any]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for canonical, feature in policy["features"].items():
        aliases[canonical] = canonical
        for alias in feature.get("aliases", []):
            aliases[str(alias)] = canonical
    return aliases


def parse_intent_dict(value: Mapping[str, Any], *, validate: bool = True) -> dict[str, Any]:
    """Copy and canonicalize an intent mapping; never repairs malformed JSON or guesses fields."""

    if not isinstance(value, Mapping):
        raise CAEIntentValidationError([_error("E_JSON_TYPE", "/", "CAE intent must be an object mapping.")])
    intent = copy.deepcopy(dict(value))
    aliases = _canonical_alias_map(load_feature_role_policy())
    operations = intent.get("operations")
    if isinstance(operations, list):
        for operation in operations:
            if not isinstance(operation, dict):
                continue
            target = operation.get("target")
            if isinstance(target, dict) and isinstance(target.get("feature_type"), str):
                target["feature_type"] = aliases.get(target["feature_type"], target["feature_type"])
    if validate:
        result = validate_intent(intent)
        if not result["valid"]:
            raise CAEIntentValidationError(result["errors"])
    return intent


def validate_intent(value: Any) -> dict[str, Any]:
    """Validate schema plus stable semantic invariants and return structured errors."""

    if not isinstance(value, Mapping):
        errors = [_error("E_JSON_TYPE", "/", "CAE intent must be an object mapping.")]
        return {"valid": False, "errors": errors}

    errors: list[dict[str, Any]] = []
    for path, item in _walk(value):
        if not isinstance(item, Mapping):
            continue
        for key in item:
            if key in FORBIDDEN_IDENTITY_KEYS:
                errors.append(_error("E_IDENTITY_FIELD_FORBIDDEN", path + (key,), "Identity/truth fields are forbidden in CAE intent."))
            if key in GROUNDING_ONLY_KEYS:
                errors.append(_error("E_GEOMETRY_SOURCE_IN_INTENT", path + (key,), "Grounding-envelope fields are forbidden in pure CAE intent."))

    errors.extend(_schema_errors(value, "cae_intent.schema.json"))

    for path, item in _walk(value):
        if not isinstance(item, Mapping) or not {"dimension", "value", "unit"}.issubset(item):
            continue
        dimension = item.get("dimension")
        unit = item.get("unit")
        if dimension not in UNIT_FACTORS or unit not in UNIT_FACTORS.get(str(dimension), {}):
            code = "E_UNIT_UNSUPPORTED" if not any(unit in units for units in UNIT_FACTORS.values()) else "E_UNIT_DIMENSION_MISMATCH"
            errors.append(_error(code, path + ("unit",), f"Unit {unit!r} is not valid for dimension {dimension!r}."))

    operations = value.get("operations")
    if isinstance(operations, list):
        seen_ids: set[str] = set()
        policy = load_feature_role_policy()
        for index, operation in enumerate(operations):
            if not isinstance(operation, Mapping):
                continue
            op_path = ("operations", index)
            operation_id = operation.get("operation_id")
            if isinstance(operation_id, str):
                if operation_id in seen_ids:
                    errors.append(_error("E_OPERATION_ID_DUPLICATE", op_path + ("operation_id",), "operation_id values must be unique."))
                seen_ids.add(operation_id)

            operation_type = operation.get("operation_type")
            target = operation.get("target")
            if isinstance(target, Mapping) and operation_type not in WHOLE_PART_OPERATIONS:
                quantifier = target.get("quantifier")
                has_count = "count" in target
                if (quantifier == "count") != has_count:
                    errors.append(_error("E_QUANTIFIER_COUNT_CONFLICT", op_path + ("target", "count"), "count is required only when quantifier is count."))
                filters = target.get("filters")
                has_spatial = isinstance(filters, Mapping) and any(key in filters for key in ("spatial_relation", "nearest_to"))
                has_frame = "reference_frame" in target
                if has_spatial and not has_frame:
                    errors.append(_error("E_REFERENCE_FRAME_REQUIRED", op_path + ("target",), "Spatial target filters require reference_frame."))
                if not has_spatial and has_frame:
                    errors.append(_error("E_REFERENCE_FRAME_UNEXPECTED", op_path + ("target", "reference_frame"), "reference_frame is allowed only for spatial target filters."))
                if isinstance(filters, Mapping):
                    for filter_name, expected_dimension in FILTER_DIMENSIONS.items():
                        filter_value = filters.get(filter_name)
                        if not isinstance(filter_value, Mapping):
                            continue
                        for quantity_key in ("value", "upper_value"):
                            quantity = filter_value.get(quantity_key)
                            if isinstance(quantity, Mapping) and quantity.get("dimension") != expected_dimension:
                                errors.append(_error("E_UNIT_DIMENSION_MISMATCH", op_path + ("target", "filters", filter_name, quantity_key, "dimension"), f"{filter_name} requires {expected_dimension}."))
                        upper_present = "upper_value" in filter_value
                        if (filter_value.get("operator") == "between") != upper_present:
                            errors.append(_error("E_FILTER_RANGE_INVALID", op_path + ("target", "filters", filter_name), "between requires upper_value; other operators forbid it."))
                compatibility = check_role_operation_compatibility(
                    policy,
                    str(target.get("feature_type", "")),
                    str(target.get("face_role", "")),
                    str(operation_type or ""),
                )
                if not compatibility["compatible"]:
                    errors.append(_error(compatibility["error_code"], op_path + ("target", "face_role"), compatibility["reason"]))

            parameters = operation.get("parameters")
            if operation_type == "fixed_support" and isinstance(parameters, Mapping):
                if set(parameters.get("constrained_dofs", [])) != {"U1", "U2", "U3"}:
                    errors.append(_error("E_SCHEMA_VALIDATION", op_path + ("parameters", "constrained_dofs"), "fixed_support must constrain U1, U2, and U3."))
            if operation_type == "displacement" and isinstance(parameters, Mapping):
                dofs = [component.get("dof") for component in parameters.get("components", []) if isinstance(component, Mapping)]
                if len(dofs) != len(set(dofs)):
                    errors.append(_error("E_DISPLACEMENT_DOF_DUPLICATE", op_path + ("parameters", "components"), "Displacement DOFs must be unique."))
            if operation_type == "pressure" and isinstance(parameters, Mapping):
                direction = parameters.get("direction")
                if not isinstance(direction, Mapping) or direction.get("kind") not in {"surface_normal", "opposite_surface_normal"}:
                    errors.append(_error("E_PRESSURE_DIRECTION_INVALID", op_path + ("parameters", "direction"), "Pressure requires explicit surface_normal or opposite_surface_normal."))
            if operation_type in {"pressure", "surface_traction"} and isinstance(parameters, Mapping):
                magnitude = parameters.get("magnitude")
                if isinstance(magnitude, Mapping) and isinstance(magnitude.get("value"), (int, float)) and magnitude["value"] < 0:
                    errors.append(_error("E_SCHEMA_VALIDATION", op_path + ("parameters", "magnitude", "value"), "Magnitude must be non-negative; direction carries orientation."))
            if operation_type in {"global_mesh_size", "local_mesh_size"} and isinstance(parameters, Mapping):
                size = parameters.get("size")
                if isinstance(size, Mapping) and isinstance(size.get("value"), (int, float)) and size["value"] <= 0:
                    errors.append(_error("E_SCHEMA_VALIDATION", op_path + ("parameters", "size", "value"), "Mesh size must be positive."))

    result_errors = _deduplicate_errors(errors)
    return {"valid": not result_errors, "errors": result_errors}


def _clean_number(value: float) -> float:
    rounded = round(float(value), 12)
    return 0.0 if rounded == -0.0 else rounded


def normalize_intent_units(
    value: Mapping[str, Any],
    target_units: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Normalize every typed quantity and append deterministic conversion audit records.

    Abaqus is not involved: target units are an explicit project convention, not a
    solver-provided unit system.
    """

    intent = parse_intent_dict(value, validate=True)
    targets = dict(intent["unit_system"])
    if target_units is not None:
        targets.update(dict(target_units))
    for dimension, units in UNIT_FACTORS.items():
        target = targets.get(dimension)
        if target not in units:
            raise CAEIntentValidationError([_error("E_UNIT_UNSUPPORTED", f"/unit_system/{dimension}", f"Unsupported target unit {target!r} for {dimension}.")])

    existing_audit = copy.deepcopy(intent.get("normalization_audit", []))
    new_audit: list[dict[str, Any]] = []

    def convert(item: Any, path: tuple[Any, ...]) -> None:
        if isinstance(item, dict):
            if {"dimension", "value", "unit"}.issubset(item):
                dimension = str(item["dimension"])
                source_unit = str(item["unit"])
                target_unit = targets[dimension]
                if source_unit != target_unit:
                    factor = UNIT_FACTORS[dimension][source_unit] / UNIT_FACTORS[dimension][target_unit]
                    original_value = float(item["value"])
                    original_tolerance = item.get("tolerance")
                    item["value"] = _clean_number(original_value * factor)
                    item["unit"] = target_unit
                    record: dict[str, Any] = {
                        "field_path": _pointer(path), "dimension": dimension,
                        "from_unit": source_unit, "to_unit": target_unit,
                        "factor": _clean_number(factor),
                        "original_value": original_value,
                        "normalized_value": item["value"],
                    }
                    if original_tolerance is not None:
                        item["tolerance"] = _clean_number(float(original_tolerance) * factor)
                        record["original_tolerance"] = float(original_tolerance)
                        record["normalized_tolerance"] = item["tolerance"]
                    new_audit.append(record)
                return
            for key, child in list(item.items()):
                if key != "normalization_audit":
                    convert(child, path + (key,))
        elif isinstance(item, list):
            for index, child in enumerate(item):
                convert(child, path + (index,))

    convert(intent, ())
    intent["unit_system"] = targets
    by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for record in existing_audit + new_audit:
        key = (record["field_path"], record["dimension"], record["from_unit"], record["to_unit"])
        by_key[key] = record
    intent["normalization_audit"] = [by_key[key] for key in sorted(by_key)]
    result = validate_intent(intent)
    if not result["valid"]:
        raise CAEIntentValidationError(result["errors"])
    return intent


def validate_intent_result(value: Any) -> dict[str, Any]:
    """Validate the VALID/NEEDS_CLARIFICATION/INVALID discriminated union."""

    if not isinstance(value, Mapping):
        return {"valid": False, "errors": [_error("E_JSON_TYPE", "/", "Intent result must be an object mapping.")]}
    errors: list[dict[str, Any]] = _schema_errors(value, "cae_intent_result.schema.json")
    if value.get("status") != "VALID":
        for path, item in _walk(value):
            if isinstance(item, Mapping):
                for key in item:
                    if key in {"intent", "operations"}:
                        errors.append(_error("E_NONVALID_RESULT_HAS_INTENT", path + (key,), "Non-VALID results cannot carry executable intent or operations."))
    elif isinstance(value.get("intent"), Mapping):
        errors.extend(validate_intent(value["intent"])["errors"])
    result_errors = _deduplicate_errors(errors)
    return {"valid": not result_errors, "errors": result_errors}


def load_feature_role_policy(path: str | Path | None = None) -> dict[str, Any]:
    """Load and minimally self-audit the policy; unsafe YAML constructors are disabled."""

    policy_path = Path(path) if path is not None else POLICY_PATH
    with policy_path.open("r", encoding="utf-8") as handle:
        policy = yaml.safe_load(handle)
    if not isinstance(policy, dict) or policy.get("schema_version") != "topocae_cae_feature_role_policy_1":
        raise ValueError("Unsupported or malformed feature-role policy")
    features = policy.get("features")
    if not isinstance(features, dict) or len(features) != 17 or policy.get("canonical_feature_count") != 17:
        raise ValueError("Feature-role policy must contain exactly 17 canonical features")
    required = {
        "family", "canonical_name", "aliases", "selectable_roles",
        "semantic_subtype_candidates", "region_kind", "allowed_operations",
        "requires_complete_instance", "may_include_transition", "selection_notes",
        "unsupported_or_ambiguous_cases",
    }
    for canonical, feature in features.items():
        if not isinstance(feature, dict) or required - set(feature):
            raise ValueError(f"Malformed feature policy entry: {canonical}")
        if feature.get("canonical_name") != canonical:
            raise ValueError(f"Canonical feature mismatch: {canonical}")
    return policy


def check_role_operation_compatibility(
    policy: Mapping[str, Any] | None,
    feature_type: str,
    face_role: str,
    operation_type: str,
) -> dict[str, Any]:
    """Check geometry-target compatibility without deciding engineering intent."""

    actual_policy = policy if policy is not None else load_feature_role_policy()
    aliases = _canonical_alias_map(actual_policy)
    canonical = aliases.get(feature_type, feature_type)
    feature = actual_policy.get("features", {}).get(canonical)
    if not isinstance(feature, Mapping):
        return {"compatible": False, "error_code": "E_ROLE_UNKNOWN", "canonical_feature": canonical,
                "selection_status": "unknown", "reason": f"Unknown feature type: {feature_type}"}
    role = next((item for item in feature.get("selectable_roles", []) if item.get("role") == face_role), None)
    if role is None:
        return {"compatible": False, "error_code": "E_ROLE_UNKNOWN", "canonical_feature": canonical,
                "selection_status": "unknown", "reason": f"Unknown face role {face_role!r} for {canonical}."}
    allowed = operation_type in role.get("allowed_operations", [])
    if role.get("selection_status", "").startswith("unsupported"):
        allowed = False
    return {
        "compatible": allowed,
        "error_code": "" if allowed else "E_ROLE_OPERATION_INCOMPATIBLE",
        "canonical_feature": canonical,
        "selection_status": role.get("selection_status"),
        "reason": (
            "Role and operation are compatible under the geometry policy."
            if allowed else f"Operation {operation_type!r} is not allowed for {canonical}.{face_role}."
        ),
    }


def build_clarification_result(
    *,
    missing_information: Sequence[Mapping[str, Any]] = (),
    ambiguities: Sequence[Mapping[str, Any]] = (),
    clarification_questions: Sequence[Mapping[str, Any]] | None = None,
    field_path: str | None = None,
) -> dict[str, Any]:
    """Build a non-executable NEEDS_CLARIFICATION result from structured issues."""

    missing = [copy.deepcopy(dict(item)) for item in missing_information]
    ambiguous = [copy.deepcopy(dict(item)) for item in ambiguities]
    if not missing and not ambiguous:
        raise ValueError("At least one missing-information or ambiguity item is required")
    if clarification_questions is None:
        issues = missing + ambiguous
        questions = [
            {
                "question_id": f"q{index + 1}",
                "field_path": str(issue["field_path"]),
                "question": f"Please clarify: {issue['description']}",
            }
            for index, issue in enumerate(issues)
        ]
    else:
        questions = [copy.deepcopy(dict(item)) for item in clarification_questions]
    primary_path = field_path or str((missing + ambiguous)[0]["field_path"])
    result = {
        "schema_version": "cae-intent-result/1.0.0",
        "status": "NEEDS_CLARIFICATION",
        "error_code": "E_MISSING_INFORMATION" if missing else "E_AMBIGUOUS_REQUIREMENT",
        "field_path": primary_path,
        "missing_information": missing,
        "ambiguities": ambiguous,
        "clarification_questions": questions,
    }
    validation = validate_intent_result(result)
    if not validation["valid"]:
        raise CAEIntentValidationError(validation["errors"])
    return result


__all__ = [
    "CAEIntentValidationError",
    "parse_intent_dict",
    "validate_intent",
    "normalize_intent_units",
    "validate_intent_result",
    "load_feature_role_policy",
    "check_role_operation_compatibility",
    "build_clarification_result",
]
