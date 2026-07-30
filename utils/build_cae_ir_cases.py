"""Generate and self-check non-placeholder CAE-IR v1 test fixtures."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from utils.cae_intent import (
    build_clarification_result,
    load_feature_role_policy,
    normalize_intent_units,
    validate_intent,
    validate_intent_result,
)


UNIT_SYSTEM = {
    "length": "mm", "area": "mm2", "force": "N",
    "pressure": "MPa", "displacement": "mm", "angle": "deg",
}


def quantity(dimension: str, value: float, unit: str, tolerance: float | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"dimension": dimension, "value": value, "unit": unit}
    if tolerance is not None:
        result["tolerance"] = tolerance
    return result


def target(
    feature: str,
    role: str,
    *,
    quantifier: str = "all",
    count: int | None = None,
    comparison_metric: str | None = None,
    filters: dict[str, Any] | None = None,
    reference_frame: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "feature_type": feature,
        "quantifier": quantifier,
        "face_role": role,
        "filters": copy.deepcopy(filters or {}),
    }
    if count is not None:
        result["count"] = count
    if comparison_metric is not None:
        result["comparison_metric"] = comparison_metric
    if reference_frame:
        result["reference_frame"] = {"kind": "global_cartesian"}
    return result


def operation(operation_id: str, operation_type: str, target_value: dict[str, Any] | None = None) -> dict[str, Any]:
    if operation_type == "material_assignment":
        return {
            "operation_id": operation_id,
            "operation_type": operation_type,
            "target": {"scope": "whole_part"},
            "parameters": {
                "material_name": "reviewed_linear_elastic_material",
                "elastic": {"youngs_modulus": quantity("pressure", 210, "GPa"), "poisson_ratio": 0.3},
            },
        }
    if operation_type == "global_mesh_size":
        return {
            "operation_id": operation_id,
            "operation_type": operation_type,
            "target": {"scope": "whole_part"},
            "parameters": {"size": quantity("length", 0.5, "cm")},
        }
    if target_value is None:
        raise ValueError(f"Feature target required for {operation_type}")
    parameters: dict[str, Any]
    if operation_type == "fixed_support":
        parameters = {"constrained_dofs": ["U1", "U2", "U3"]}
    elif operation_type == "displacement":
        parameters = {
            "components": [{"dof": "U1", "value": quantity("displacement", 0.2, "mm")}],
            "reference_frame": {"kind": "global_cartesian"},
        }
    elif operation_type == "pressure":
        parameters = {"magnitude": quantity("pressure", 2.5, "MPa"), "direction": {"kind": "surface_normal"}}
    elif operation_type == "surface_traction":
        parameters = {
            "magnitude": quantity("pressure", 1.25, "MPa"),
            "direction": {"kind": "global_cartesian", "vector": [1.0, 0.0, 0.0]},
        }
    elif operation_type == "local_mesh_size":
        parameters = {"size": quantity("length", 0.8, "mm")}
    else:
        raise ValueError(operation_type)
    return {"operation_id": operation_id, "operation_type": operation_type, "target": target_value, "parameters": parameters}


def intent(operations: list[dict[str, Any]], language: str = "zh-CN") -> dict[str, Any]:
    return {
        "schema_version": "cae-intent/1.0.0",
        "analysis_type": "static_structural",
        "operations": operations,
        "source_language": language,
        "assumptions": [],
        "unit_system": copy.deepcopy(UNIT_SYSTEM),
        "normalization_audit": [],
    }


def _supported_role_map() -> tuple[dict[str, str], list[tuple[str, str]]]:
    policy = load_feature_role_policy()
    primary: dict[str, str] = {}
    unsupported: list[tuple[str, str]] = []
    for feature, entry in policy["features"].items():
        for role in entry["selectable_roles"]:
            if role["selection_status"] in {"direct", "composite"} and feature not in primary:
                primary[feature] = role["role"]
            if role["selection_status"].startswith("unsupported"):
                unsupported.append((feature, role["role"]))
    return primary, unsupported


def _case(case_id: str, status: str, description: str, payload: dict[str, Any], tags: list[str]) -> dict[str, Any]:
    return {"case_id": case_id, "expected_status": status, "description": description, "payload": payload, "tags": tags}


def build_cases() -> list[dict[str, Any]]:
    primary, unsupported_roles = _supported_role_map()
    features = list(primary)
    surface_ops = ["fixed_support", "displacement", "pressure", "surface_traction", "local_mesh_size"]
    cases: list[dict[str, Any]] = []
    serial = 1

    def add(status: str, description: str, payload: dict[str, Any], tags: list[str]) -> None:
        nonlocal serial
        prefix = {"VALID": "V", "NEEDS_CLARIFICATION": "C", "INVALID": "I"}[status]
        cases.append(_case(f"{prefix}{serial:03d}", status, description, payload, tags))
        serial += 1

    # Every canonical feature appears in a semantically valid, independently named case.
    for index, feature in enumerate(features):
        op_type = surface_ops[index % len(surface_ops)]
        op = operation("feature_op", op_type, target(feature, primary[feature]))
        add("VALID", f"Apply {op_type} to the reviewed {primary[feature]} role of {feature}.", intent([op]), ["feature_coverage", f"feature:{feature}", f"operation:{op_type}"])

    # Every operation type has a direct coverage case, including whole-part operations.
    for index, op_type in enumerate(surface_ops + ["material_assignment", "global_mesh_size"]):
        feature = features[index % len(features)]
        op = operation("operation_coverage", op_type, None if op_type in {"material_assignment", "global_mesh_size"} else target(feature, primary[feature], quantifier="exactly_one"))
        add("VALID", f"Exercise the complete typed contract for {op_type}.", intent([op], "en-US"), ["operation_coverage", f"operation:{op_type}"])

    # Twenty-two real conversions spanning filters, pressure/traction, displacement, mesh, and material modulus.
    conversion_specs = [
        ("pressure", "kPa", 2500.0), ("pressure", "Pa", 1_500_000.0), ("pressure", "GPa", 0.004),
        ("traction", "kPa", 875.0), ("traction", "Pa", 900_000.0), ("traction", "GPa", 0.002),
        ("displacement", "cm", 0.12), ("displacement", "m", 0.0015), ("displacement", "um", 250.0),
        ("local_mesh", "cm", 0.15), ("local_mesh", "m", 0.002), ("local_mesh", "um", 700.0),
        ("global_mesh", "cm", 0.35), ("global_mesh", "m", 0.004), ("global_mesh", "um", 1200.0),
        ("diameter_filter", "cm", 1.0), ("diameter_filter", "m", 0.012), ("diameter_filter", "um", 8000.0),
        ("area_filter", "cm2", 2.5), ("area_filter", "m2", 0.0002),
        ("material_modulus", "GPa", 70.0), ("material_modulus", "kPa", 210_000_000.0),
    ]
    for index, (kind, unit, value) in enumerate(conversion_specs):
        feature = features[index % len(features)]
        feature_target = target(feature, primary[feature])
        if kind == "pressure":
            op = operation("convert_pressure", "pressure", feature_target)
            op["parameters"]["magnitude"] = quantity("pressure", value, unit, abs(value) * 0.01)
        elif kind == "traction":
            op = operation("convert_traction", "surface_traction", feature_target)
            op["parameters"]["magnitude"] = quantity("pressure", value, unit)
        elif kind == "displacement":
            op = operation("convert_displacement", "displacement", feature_target)
            op["parameters"]["components"][0]["value"] = quantity("displacement", value, unit)
        elif kind == "local_mesh":
            op = operation("convert_local_mesh", "local_mesh_size", feature_target)
            op["parameters"]["size"] = quantity("length", value, unit)
        elif kind == "global_mesh":
            op = operation("convert_global_mesh", "global_mesh_size")
            op["parameters"]["size"] = quantity("length", value, unit)
        elif kind == "diameter_filter":
            feature_target["filters"] = {"diameter": {"operator": "eq", "value": quantity("length", value, unit)}}
            op = operation("convert_diameter", "fixed_support", feature_target)
        elif kind == "area_filter":
            feature_target["filters"] = {"area": {"operator": "gte", "value": quantity("area", value, unit)}}
            op = operation("convert_area", "pressure", feature_target)
        else:
            op = operation("convert_modulus", "material_assignment")
            op["parameters"]["elastic"]["youngs_modulus"] = quantity("pressure", value, unit)
        payload = intent([op])
        normalized = normalize_intent_units(payload)
        if not normalized["normalization_audit"]:
            raise AssertionError(f"Conversion fixture produced no audit: {kind} {unit}")
        add("VALID", f"Normalize {kind} from {unit} into the explicit project unit context.", payload, ["unit_conversion", f"dimension_case:{kind}"])

    # Ten multi-operation intents exercise unique IDs and mixed whole-part/feature scopes.
    for index in range(10):
        feature = features[index % len(features)]
        surface_type = surface_ops[index % len(surface_ops)]
        ops = [
            operation(f"material_{index}", "material_assignment"),
            operation(f"mesh_{index}", "global_mesh_size"),
            operation(f"surface_{index}", surface_type, target(feature, primary[feature], quantifier="exactly_one")),
        ]
        add("VALID", f"Combine material, global mesh, and {surface_type} for one static-structural intent.", intent(ops), ["multi_operation", f"operation:{surface_type}"])

    # Ten quantity/geometric-selector cases cover count, comparisons, nearest, and spatial filters.
    selector_builders = [
        lambda f, r: target(f, r, quantifier="count", count=2),
        lambda f, r: target(f, r, quantifier="largest", comparison_metric="area"),
        lambda f, r: target(f, r, quantifier="smallest", comparison_metric="diameter"),
        lambda f, r: target(f, r, quantifier="nearest", filters={"nearest_to": {"reference_type": "part_centroid"}}, reference_frame=True),
        lambda f, r: target(f, r, quantifier="all", filters={"diameter": {"operator": "between", "value": quantity("length", 8, "mm"), "upper_value": quantity("length", 12, "mm")}}),
        lambda f, r: target(f, r, quantifier="exactly_one", filters={"radius": {"operator": "lte", "value": quantity("length", 5, "mm")}}),
        lambda f, r: target(f, r, quantifier="any_one", filters={"spatial_relation": {"relation": "above", "reference": {"reference_type": "part_centroid"}}}, reference_frame=True),
        lambda f, r: target(f, r, quantifier="all", filters={"depth": {"operator": "gte", "value": quantity("length", 4, "mm")}}),
        lambda f, r: target(f, r, quantifier="all", filters={"width": {"operator": "eq", "value": quantity("length", 6, "mm", 0.1)}}),
        lambda f, r: target(f, r, quantifier="all", filters={"thickness": {"operator": "lt", "value": quantity("length", 3, "mm")}}),
    ]
    for index, builder in enumerate(selector_builders):
        feature = features[index % len(features)]
        op = operation("selector", surface_ops[index % len(surface_ops)], builder(feature, primary[feature]))
        add("VALID", f"Select {feature} using selector pattern {index + 1} without identity fields.", intent([op]), ["quantifier_or_filter"])

    missing_codes = [
        "M_TARGET", "M_FEATURE_TYPE", "M_FACE_ROLE", "M_QUANTIFIER", "M_COUNT",
        "M_MAGNITUDE", "M_UNIT", "M_DIRECTION", "M_DISPLACEMENT_COMPONENT", "M_MATERIAL_NAME",
        "M_MATERIAL_PROPERTY", "M_MESH_SIZE", "M_REFERENCE",
    ]
    for index in range(20):
        code = missing_codes[index % len(missing_codes)]
        path = f"/operations/{index % 3}/required_field"
        result = build_clarification_result(missing_information=[{
            "code": code, "field_path": path,
            "description": f"Required engineering input {code} was not provided for case {index + 1}.",
        }])
        add("NEEDS_CLARIFICATION", f"Request the missing {code} value without constructing a partial operation.", result, ["structured_clarification", "missing_information"])

    ambiguity_codes = [
        "A_FEATURE_TYPE", "A_FACE_ROLE", "A_TARGET_INSTANCE", "A_DIRECTION_ORIENTATION",
        "A_UNIT", "A_MATERIAL", "A_SPATIAL_REFERENCE",
    ]
    for index in range(20):
        code = ambiguity_codes[index % len(ambiguity_codes)]
        path = f"/operations/{index % 2}/ambiguous_field"
        result = build_clarification_result(ambiguities=[{
            "code": code, "field_path": path,
            "description": f"Two engineering interpretations remain for {code} in case {index + 1}.",
            "options": [f"{code}_option_a", f"{code}_option_b"],
        }])
        add("NEEDS_CLARIFICATION", f"Expose {code} as a structured ambiguity and ask one deterministic question.", result, ["structured_clarification", "ambiguity"])

    # Fifteen explicit role-operation negatives: three unsupported roles x five surface operations.
    for feature, role in unsupported_roles:
        for op_type in surface_ops:
            payload = intent([operation("unsupported_role", op_type, target(feature, role))])
            add("INVALID", f"Reject {op_type} on unsupported precise role {feature}.{role}.", payload, ["role_operation_negative"])

    # Twenty-five independent malformed/unsafe inputs.
    base = intent([operation("base", "pressure", target("through_hole_round", "hole_wall"))])
    mutations: list[tuple[str, Any, list[str]]] = []
    for key, value in [
        ("face_id", 7), ("face_ids", [1, 2]), ("sample_id", "sample-7"),
        ("part_id", "part-4"), ("model_filename", "secret.step"),
    ]:
        def identity_mutation(key=key, value=value):
            item = copy.deepcopy(base); item["operations"][0]["target"][key] = value; return item
        mutations.append((f"Reject forbidden identity field {key} in a feature target.", identity_mutation, ["identity_rejection"]))
    for key, value in [("geometry_source", "oracle"), ("coverage_status", "ACCEPTED"), ("abstention_reasons", [])]:
        def envelope_mutation(key=key, value=value):
            item = copy.deepcopy(base); item[key] = value; return item
        mutations.append((f"Reject grounding-envelope field {key} in pure intent.", envelope_mutation, ["envelope_isolation"]))

    def mutate(path_name: str):
        item = copy.deepcopy(base)
        op = item["operations"][0]
        if path_name == "pressure_unit_force": op["parameters"]["magnitude"] = quantity("pressure", 10, "N")
        elif path_name == "pressure_unit_unknown": op["parameters"]["magnitude"] = quantity("pressure", 10, "psi")
        elif path_name == "area_wrong_dimension": op["target"]["filters"] = {"area": {"operator": "gte", "value": quantity("length", 10, "mm")}}
        elif path_name == "between_missing_upper": op["target"]["filters"] = {"diameter": {"operator": "between", "value": quantity("length", 5, "mm")}}
        elif path_name == "eq_with_upper": op["target"]["filters"] = {"diameter": {"operator": "eq", "value": quantity("length", 5, "mm"), "upper_value": quantity("length", 6, "mm")}}
        elif path_name == "count_missing": op["target"]["quantifier"] = "count"
        elif path_name == "count_unexpected": op["target"]["count"] = 2
        elif path_name == "spatial_frame_missing": op["target"]["filters"] = {"nearest_to": {"reference_type": "part_centroid"}}
        elif path_name == "frame_unexpected": op["target"]["reference_frame"] = {"kind": "global_cartesian"}
        elif path_name == "pressure_cartesian": op["parameters"]["direction"] = {"kind": "global_cartesian", "vector": [0, 0, 1]}
        elif path_name == "duplicate_id": item["operations"].append(copy.deepcopy(op))
        elif path_name == "unconfirmed_assumption": item["assumptions"] = [{"assumption_id": "a1", "text": "guessed", "confirmed_by": "agent"}]
        elif path_name == "negative_pressure": op["parameters"]["magnitude"]["value"] = -1
        elif path_name == "zero_mesh": item = intent([operation("mesh", "global_mesh_size")]); item["operations"][0]["parameters"]["size"]["value"] = 0
        elif path_name == "unknown_property": op["parameters"]["unexpected"] = True
        elif path_name == "duplicate_dof": item = intent([operation("disp", "displacement", target("blind_hole_round", "hole_wall"))]); item["operations"][0]["parameters"]["components"].append(copy.deepcopy(item["operations"][0]["parameters"]["components"][0]))
        elif path_name == "web_without_parse": op["target"]["feature_type"] = "web"
        else: raise AssertionError(path_name)
        return item

    named_mutations = [
        "pressure_unit_force", "pressure_unit_unknown", "area_wrong_dimension",
        "between_missing_upper", "eq_with_upper", "count_missing", "count_unexpected",
        "spatial_frame_missing", "frame_unexpected", "pressure_cartesian", "duplicate_id",
        "unconfirmed_assumption", "negative_pressure", "zero_mesh", "unknown_property",
        "duplicate_dof", "web_without_parse",
    ]
    for name in named_mutations:
        mutations.append((f"Reject malformed semantic contract variant {name}.", lambda name=name: mutate(name), ["strict_invalid"]))

    for description, builder, tags in mutations[:25]:
        add("INVALID", description, builder(), tags)

    return cases


def verify_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(case["expected_status"] for case in cases)
    tag_counts = Counter(tag for case in cases for tag in case["tags"])
    ids = [case["case_id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise AssertionError("Duplicate case IDs")
    for case in cases:
        if "placeholder" in case["description"].lower() or len(case["description"]) < 20:
            raise AssertionError(f"Non-substantive case: {case['case_id']}")
        if case["expected_status"] == "VALID":
            result = validate_intent(case["payload"])
            if not result["valid"]:
                raise AssertionError(f"Expected VALID {case['case_id']}: {result['errors'][:3]}")
        elif case["expected_status"] == "NEEDS_CLARIFICATION":
            result = validate_intent_result(case["payload"])
            if not result["valid"] or "operations" in case["payload"] or "intent" in case["payload"]:
                raise AssertionError(f"Invalid clarification case: {case['case_id']}")
        else:
            result = validate_intent(case["payload"])
            if result["valid"]:
                raise AssertionError(f"Expected INVALID: {case['case_id']}")
    if status_counts["VALID"] < 50 or status_counts["NEEDS_CLARIFICATION"] < 35 or status_counts["INVALID"] < 35:
        raise AssertionError(status_counts)
    for tag, minimum in [("unit_conversion", 20), ("role_operation_negative", 15), ("multi_operation", 10), ("quantifier_or_filter", 10)]:
        if tag_counts[tag] < minimum:
            raise AssertionError(f"{tag}: {tag_counts[tag]} < {minimum}")
    return {"total": len(cases), "status_counts": dict(status_counts), "tag_counts": dict(tag_counts)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    cases = build_cases()
    summary = verify_cases(cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n" for case in cases), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
