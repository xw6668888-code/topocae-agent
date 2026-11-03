"""Frozen G10 M6 case registry and E3-to-engineering binding contract.

This module is intentionally free of OCC and Abaqus imports.  It can verify and
freeze the 13-case execution denominator before any CAD edit or CAE solve.  It
never opens the engineering truth sidecar; the only historical per-part fields
consumed are source STEP, truth-free E3 output, and source-only parity records.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from g10_common import (
    ARTIFACT, REPORT, ROOT, checked, descriptor, digest, now, read_json,
    write_json,
)


DEPLOYMENT_SEED = 20260719
E3_SYSTEM = "frozen_D0_R_Q_seed_20260719_E3_bayesian_joint"
E3_SCHEMA = "g09_e3_bayesian_step_service_v1"
OCC_VERSION = "7.9.0"
ABAQUS_RELEASE = "Abaqus 2020"
M6_ROOT = ARTIFACT / "m6_engineering"
STAGED_PREDICTIONS = M6_ROOT / "frozen_e3_outputs"
MANIFEST_PATH = REPORT / "M6_ENGINEERING_MANIFEST.json"
CAUSALITY_PATH = REPORT / "M6_CAUSALITY_REGRESSION.json"
G09_REPORT = ROOT / "reports/aei_paper_goals_v1/g09"
G09_ACCEPTANCE = G09_REPORT / "E3_ENGINEERING_FRESH_STEP_ACCEPTANCE_V2.json"
G09_FREEZE = G09_REPORT / "FINAL_FREEZE.json"
OLD_TASKS = ROOT / "reports/aei_case_suite_v2/registered_tasks.json"
OLD_MATERIALS = ROOT / "reports/aei_case_suite_v2/materials/selected_materials.json"
RADIUS_TOLERANCE_MM = 1e-5


class BindingError(ValueError):
    """A fail-closed engineering binding failure with a stable status code."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def target(classes: str | list[str], radius: float | None = None,
           scope: str = "all") -> dict[str, Any]:
    return {
        "feature_classes": [classes] if isinstance(classes, str) else classes,
        "radius_mm": radius,
        "scope": scope,
    }


def ready(domain: str, operations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "2",
        "status": "ready",
        "domain": domain,
        "operations": operations,
        "clarifications": [],
        "reason": "G10 frozen user-confirmed reference plan; not an LLM result",
    }


def force(feature: str, magnitude: float, direction: list[float],
          role: str = "top_face", radius: float | None = None) -> dict[str, Any]:
    return {
        "action": "apply_force",
        "target": target(feature, radius),
        "role": role,
        "magnitude_N": magnitude,
        "direction": direction,
        "basis": "per_instance",
    }


def cae(material: str, mesh: float, fixed: dict[str, Any],
        load: dict[str, Any], local: float | None = None) -> dict[str, Any]:
    operations: list[dict[str, Any]] = [
        {"action": "set_material", "material_id": material},
        {"action": "set_global_mesh", "size_mm": mesh},
    ]
    if local is not None:
        operations.append({
            "action": "set_local_mesh",
            "target": target("chamfer"),
            "size_mm": local,
        })
    operations.extend([
        {"action": "fix_features", "target": fixed, "role": "whole_feature"},
        load,
        {"action": "set_analysis", "type": "linear_static"},
    ])
    return ready("cae", operations)


def case_plans() -> dict[str, dict[str, Any]]:
    """Return the 13 plans fixed by G10 and prior confirmed amendments."""
    plans = {
        "angle_bracket": cae(
            "Steel45", 2.0, target("counterbore_hole"),
            force("obround_hole", 100.0, [0.0, -1.0, 0.0], "inner_walls"),
        ),
        "clevis": cae(
            "Steel45", 1.0, target("through_hole_round", 3.765),
            {
                "action": "apply_traction",
                "target": target("cylinder_boss"),
                "role": "top_face",
                "magnitude_MPa": 50.0,
                "direction": [0.0, 0.0, 1.0],
            },
            local=0.3,
        ),
        "flanged_support": cae(
            "Steel45", 1.0, target("countersink_hole"),
            force("through_hole_round", 500.0, [0.0, 0.0, 1.0],
                  "inner_walls", radius=4.0),
        ),
        # These two mesh values carry the explicit 2026-09-09 user amendment.
        "sealing_cover": cae(
            "Brass", 5.0, target("counterbore_hole"),
            {
                "action": "apply_pressure",
                "target": target("pocket_circular"),
                "role": "cavity_interior",
                "magnitude_MPa": 10.0,
            },
            local=2.0,
        ),
        "stiffened_panel": cae(
            "Steel45", 10.0, target("counterbore_hole"),
            force("rectangular_boss", 1000.0, [0.0, -1.0, 0.0]),
        ),
        "turbine_blade": cae(
            "PlainCarbonSteel", 1.0, target(["slot_rect", "slot_o"]),
            {
                "action": "apply_centrifugal",
                "angular_velocity_rad_s": 100.0,
                "axis_point_mm": [0.0, 0.0, 0.0],
                "axis_direction": [0.0, 0.0, 1.0],
            },
        ),
        "u_shaped_bracket": cae(
            "PlainCarbonSteel", 1.0, target("through_hole_round"),
            force("cylinder_boss", 500.0, [0.0, -1.0, 0.0]),
        ),
        "bearing_housing": ready("cad", [{
            "action": "set_hole_radius",
            "target": target("blind_hole_round", 4.0),
            "new_radius_mm": 3.0,
        }]),
        "flanged_sleeve": ready("cad", [{
            "action": "set_hole_radius",
            "target": target("through_hole_round", 20.0),
            "new_radius_mm": 15.0,
        }]),
        "guide_rail": ready("cad", [{
            "action": "set_hole_spacing",
            "target": target("counterbore_hole"),
            "old_spacing_mm": 200.0,
            "new_spacing_mm": 150.0,
            "anchor": "central_hole",
            "spacing_definition": "adjacent_centers",
        }]),
        "mounting_plate": ready("cad", [{
            "action": "set_hole_radius",
            "target": target("through_hole_round", 22.5),
            "new_radius_mm": 24.0,
        }]),
        "support_bracket": ready("cad", [{
            "action": "set_thickness",
            "target": target("rib"),
            "old_thickness_mm": 2.0,
            "new_thickness_mm": 4.0,
            "anchor": "midplane",
        }]),
        "support_frame": ready("cad", [{
            "action": "set_thickness",
            "target": target("web"),
            "old_thickness_mm": 4.6985,
            "new_thickness_mm": 3.0,
            "anchor": "midplane",
        }]),
    }
    if len(plans) != 13:
        raise AssertionError("The engineering denominator must remain 13")
    if sum(plan["domain"] == "cad" for plan in plans.values()) != 6:
        raise AssertionError("The CAD denominator must remain 6")
    if sum(plan["domain"] == "cae" for plan in plans.values()) != 7:
        raise AssertionError("The CAE denominator must remain 7")
    return plans


MESH_SENSITIVITY = {
    "angle_bracket": {
        "reason": "directional per-instance surface force with multiple loaded objects",
        "levels": [
            {"id": "coarse", "global_mesh_mm": 3.0, "local_mesh_mm": None},
            {"id": "registered", "global_mesh_mm": 2.0, "local_mesh_mm": None},
            {"id": "fine", "global_mesh_mm": 1.5, "local_mesh_mm": None},
        ],
        "comparison_quantities": ["max_displacement_mm", "reaction_force_N"],
    },
    "sealing_cover": {
        "reason": "normal pressure with both planar-bottom and cylindrical-sidewall regions",
        "levels": [
            {"id": "coarse", "global_mesh_mm": 7.5, "local_mesh_mm": 3.0},
            {"id": "registered", "global_mesh_mm": 5.0, "local_mesh_mm": 2.0},
            {"id": "fine", "global_mesh_mm": 3.75, "local_mesh_mm": 1.5},
        ],
        "comparison_quantities": ["max_displacement_mm", "reaction_force_N"],
    },
    "interpretation_limit": (
        "Do not use singular peak stress as convergence or strength-safety evidence; "
        "this study checks bounded workflow sensitivity of global quantities."
    ),
}


MATERIALS = {
    "Steel45": {
        "name": "Steel45", "youngs_modulus_mpa": 210000.0,
        "poisson_ratio": 0.30, "density_tonne_mm3": 7.85e-9,
        "source": "previously confirmed user parameters",
    },
    "Brass": {
        "name": "Brass", "youngs_modulus_mpa": 100000.0,
        "poisson_ratio": 0.33, "density_tonne_mm3": 8.5e-9,
        "source": "SI_mm111111_Brass",
        "library_sha256": "0eb83be6e7a848b471c00f1c3885693bce7d8a43835bf0b9a0977f5b1de35cd7",
    },
    "PlainCarbonSteel": {
        "name": "PlainCarbonSteel", "youngs_modulus_mpa": 210000.0,
        "poisson_ratio": 0.28, "density_tonne_mm3": 7.8e-9,
        "source": "SI_mm111111_PlainCarbonSteel",
        "library_sha256": "0eb83be6e7a848b471c00f1c3885693bce7d8a43835bf0b9a0977f5b1de35cd7",
    },
}


def slug(part: str) -> str:
    return Path(part).stem.lower().replace(" ", "_").replace("-", "_")


def _descriptors(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= set(value):
            yield {key: value[key] for key in ("path", "bytes", "sha256")}
        for child in value.values():
            yield from _descriptors(child)
    elif isinstance(value, list):
        for child in value:
            yield from _descriptors(child)


def object_id(prediction: dict[str, Any], obj: dict[str, Any]) -> str:
    """Return a geometry-version-scoped object ID, not a permanent face number."""
    identity = {
        "geometry_version": prediction["input_step"]["sha256"],
        "class": obj["class"],
        "graph_members": sorted(obj["graph_node_indices"]),
        "step_members": sorted(obj["original_face_numbers"]),
    }
    return f"e3:{identity['geometry_version'][:12]}:{obj['class']}:{digest(identity)[:16]}"


def validate_prediction(prediction: dict[str, Any], expected_step: dict[str, Any]) -> None:
    if prediction.get("schema") != E3_SCHEMA:
        raise BindingError("wrong_prediction_schema")
    if prediction.get("system") != E3_SYSTEM:
        raise BindingError("wrong_frozen_system")
    if prediction.get("truth_used") is not False:
        raise BindingError("truth_access_in_inference")
    if prediction.get("input_step") != expected_step:
        raise BindingError("stale_geometry_version")
    source = prediction.get("source_identity", {})
    if source.get("R", {}).get("seed") != DEPLOYMENT_SEED \
            or source.get("Q", {}).get("seed") != DEPLOYMENT_SEED:
        raise BindingError("wrong_deployment_seed")
    for item in _descriptors(source):
        checked(item)

    face_classes = prediction.get("face_macro_classes")
    objects = prediction.get("feature_objects")
    if not isinstance(face_classes, list) or not face_classes:
        raise BindingError("missing_face_predictions")
    if not isinstance(objects, list) or not objects:
        raise BindingError("missing_feature_objects")
    indices: set[int] = set()
    keys: set[tuple[str, tuple[int, ...]]] = set()
    for obj in objects:
        index = obj.get("object_index")
        if not isinstance(index, int) or index in indices:
            raise BindingError("duplicate_object_index")
        indices.add(index)
        graph = obj.get("graph_node_indices")
        faces = obj.get("original_face_numbers")
        if not isinstance(graph, list) or not graph or len(graph) != len(set(graph)):
            raise BindingError("invalid_object_graph_members")
        if not isinstance(faces, list) or not faces or len(faces) != len(set(faces)):
            raise BindingError("invalid_object_step_members")
        if sorted(faces) != sorted(value + 1 for value in graph):
            raise BindingError("graph_step_face_mapping_changed")
        if min(graph) < 0 or max(graph) >= len(face_classes):
            raise BindingError("object_face_out_of_range")
        key = (obj.get("class", ""), tuple(sorted(graph)))
        if not key[0] or key in keys:
            raise BindingError("duplicate_or_empty_object")
        keys.add(key)
        roles = obj.get("roles")
        if not isinstance(roles, dict) or "whole_feature" not in roles:
            raise BindingError("missing_whole_feature_role")
        for name, group in roles.items():
            if not isinstance(group, dict):
                raise BindingError("ambiguous_role", name)
            role_faces = group.get("face_numbers")
            if not isinstance(role_faces, list) or not role_faces \
                    or len(role_faces) != len(set(role_faces)):
                raise BindingError("invalid_role_faces", name)
            if min(role_faces) < 1 or max(role_faces) > len(face_classes):
                raise BindingError("role_face_out_of_range", name)
            if name == "whole_feature" and sorted(role_faces) != sorted(faces):
                raise BindingError("whole_feature_role_changed")
            if group.get("object_index") != index or group.get("class") != obj["class"]:
                raise BindingError("role_object_identity_changed", name)


def _matches_selector(obj: dict[str, Any], selector: dict[str, Any]) -> bool:
    if obj["class"] not in selector["feature_classes"]:
        return False
    radius = selector.get("radius_mm")
    if radius is None:
        return True
    radii = list(obj.get("parameters", {}).get("radii_mm", []))
    return any(abs(float(value) - float(radius)) <= RADIUS_TOLERANCE_MM
               for value in radii)


def resolve_role(obj: dict[str, Any], role: str) -> dict[str, Any]:
    value = obj.get("roles", {}).get(role)
    if isinstance(value, list):
        raise BindingError("ambiguous_role", role)
    if not isinstance(value, dict):
        raise BindingError("missing_role", f"{obj.get('class')}:{role}")
    faces = value.get("face_numbers")
    if not isinstance(faces, list) or not faces:
        raise BindingError("empty_role", role)
    return value


def resolve_plan(plan: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    if plan.get("status") != "ready" or plan.get("domain") not in {"cad", "cae"}:
        raise BindingError("nonready_plan")
    selections = []
    for operation_index, operation in enumerate(plan.get("operations", [])):
        selector = operation.get("target")
        if selector is None:
            continue
        matched = [obj for obj in prediction["feature_objects"]
                   if _matches_selector(obj, selector)]
        if not matched:
            raise BindingError("no_matching_object", operation["action"])
        if selector.get("scope") == "single" and len(matched) != 1:
            raise BindingError("ambiguous_object_selection", operation["action"])
        if operation["action"] == "set_hole_spacing" and len(matched) != 3:
            raise BindingError("spacing_requires_three_objects")
        role = operation.get("role", "whole_feature")
        groups = [resolve_role(obj, role) for obj in matched]
        selections.append({
            "operation_index": operation_index,
            "action": operation["action"],
            "selector": selector,
            "role": role,
            "object_ids": [object_id(prediction, obj) for obj in matched],
            "object_indices": [obj["object_index"] for obj in matched],
            "role_face_sets": [sorted(group["face_numbers"]) for group in groups],
            "selection_count": len(matched),
        })
    if not selections:
        raise BindingError("plan_has_no_object_binding")
    payload = {
        "geometry_version": prediction["input_step"]["sha256"],
        "prediction_system": prediction["system"],
        "prediction_source_identity_digest": digest(prediction["source_identity"]),
        "plan_digest": digest(plan),
        "selections": selections,
    }
    payload["binding_digest"] = digest(payload)
    return payload


def assert_frozen_binding(expected: dict[str, Any], plan: dict[str, Any],
                          prediction: dict[str, Any]) -> dict[str, Any]:
    actual = resolve_plan(plan, prediction)
    if actual != expected:
        raise BindingError("frozen_binding_changed")
    return actual


def _mutate_execution_parameter(plan: dict[str, Any]) -> tuple[dict[str, Any], str]:
    mutated = deepcopy(plan)
    priority = (
        "new_radius_mm", "new_spacing_mm", "new_thickness_mm", "magnitude_N",
        "magnitude_MPa", "angular_velocity_rad_s", "size_mm",
    )
    for key in priority:
        for operation in mutated["operations"]:
            if key in operation:
                operation[key] = float(operation[key]) * 1.001
                return mutated, key
    raise BindingError("no_mutable_execution_parameter")


def causality_regression(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    role_ambiguity_checked = False
    for case in case_rows:
        prediction = read_json(checked(case["prediction"]))
        plan = case["plan"]
        baseline = resolve_plan(plan, prediction)
        first_selection = baseline["selections"][0]
        removed_index = first_selection["object_indices"][0]
        removed = deepcopy(prediction)
        removed["feature_objects"] = [obj for obj in removed["feature_objects"]
                                      if obj["object_index"] != removed_index]
        try:
            changed = resolve_plan(plan, removed)
            removal_status = "binding_changed"
            removal_passed = changed["binding_digest"] != baseline["binding_digest"] \
                and removed_index not in changed["selections"][0]["object_indices"]
        except BindingError as error:
            removal_status = f"rejected:{error.code}"
            removal_passed = True

        changed_plan, parameter = _mutate_execution_parameter(plan)
        parameter_binding = resolve_plan(changed_plan, prediction)
        parameter_passed = parameter_binding["binding_digest"] != baseline["binding_digest"]

        absent_plan = deepcopy(plan)
        first_target = next(op["target"] for op in absent_plan["operations"] if "target" in op)
        first_target["feature_classes"] = ["__g10_absent_feature__"]
        try:
            resolve_plan(absent_plan, prediction)
            absent_status, absent_passed = "unexpected_accept", False
        except BindingError as error:
            absent_status = f"rejected:{error.code}"
            absent_passed = error.code == "no_matching_object"

        stale = deepcopy(prediction)
        stale["input_step"]["sha256"] = "0" * 64
        try:
            validate_prediction(stale, case["source_step"])
            stale_status, stale_passed = "unexpected_accept", False
        except BindingError as error:
            stale_status = f"rejected:{error.code}"
            stale_passed = error.code == "stale_geometry_version"

        role_status = "not_applicable"
        role_passed = True
        specialized = next((selection for selection in baseline["selections"]
                            if selection["role"] != "whole_feature"), None)
        if specialized is not None and not role_ambiguity_checked:
            ambiguous = deepcopy(prediction)
            obj_index = specialized["object_indices"][0]
            obj = next(item for item in ambiguous["feature_objects"]
                       if item["object_index"] == obj_index)
            original = obj["roles"][specialized["role"]]
            obj["roles"][specialized["role"]] = [original, deepcopy(original)]
            try:
                resolve_plan(plan, ambiguous)
                role_status, role_passed = "unexpected_accept", False
            except BindingError as error:
                role_status = f"rejected:{error.code}"
                role_passed = error.code == "ambiguous_role"
            role_ambiguity_checked = True

        passed = all((removal_passed, parameter_passed, absent_passed,
                      stale_passed, role_passed))
        rows.append({
            "case_id": case["case_id"],
            "removed_object_index": removed_index,
            "removal_result": removal_status,
            "removal_passed": removal_passed,
            "mutated_parameter": parameter,
            "parameter_binding_changed": parameter_passed,
            "absent_target_result": absent_status,
            "absent_target_passed": absent_passed,
            "stale_geometry_result": stale_status,
            "stale_geometry_passed": stale_passed,
            "ambiguous_role_result": role_status,
            "ambiguous_role_passed": role_passed,
            "passed": passed,
        })
    if not role_ambiguity_checked:
        raise BindingError("role_ambiguity_regression_not_exercised")
    return {
        "schema": "g10_m6_causality_regression_v1",
        "created_utc": now(),
        "cases": len(rows),
        "all_passed": all(row["passed"] for row in rows),
        "tests": {
            "prediction_object_removal": "binding changes or execution rejects",
            "plan_parameter_mutation": "bound execution digest changes",
            "absent_target": "explicit no_matching_object rejection",
            "stale_geometry": "explicit stale_geometry_version rejection",
            "role_ambiguity": "explicit ambiguous_role rejection",
        },
        "rows": rows,
        "truth_used": False,
    }


def _historical_rows() -> list[dict[str, Any]]:
    freeze = read_json(G09_FREEZE)
    if checked(freeze["engineering_evidence"]) != G09_ACCEPTANCE.resolve():
        raise BindingError("g09_engineering_freeze_changed")
    acceptance = read_json(G09_ACCEPTANCE)
    if acceptance.get("schema") != "g09_e3_fresh_step_engineering_acceptance_v1" \
            or acceptance.get("system") != "one E3 Bayesian R+Q service on every STEP, no sample-specific switch" \
            or acceptance.get("truth_used_in_inference") is not False \
            or acceptance.get("engineering_development_only") is not True:
        raise BindingError("invalid_g09_engineering_acceptance")
    batch = read_json(checked(acceptance["batch_seal"]))
    if batch.get("schema") != "g09_e3_bayesian_step_batch_v1" \
            or batch.get("system") != E3_SYSTEM or batch.get("parts") != 13 \
            or batch.get("truth_used") is not False:
        raise BindingError("invalid_g09_engineering_batch_seal")
    rows = acceptance.get("per_part")
    if not isinstance(rows, list) or len(rows) != 13:
        raise BindingError("engineering_denominator_changed")
    return rows


def _stage_prediction(origin: dict[str, Any], case_id: str) -> dict[str, Any]:
    source = checked(origin)
    prediction = read_json(source)
    destination = STAGED_PREDICTIONS / f"{case_id}.json"
    if destination.exists():
        existing = read_json(destination)
        if existing != prediction:
            raise BindingError("staged_prediction_changed", case_id)
    else:
        write_json(destination, prediction, overwrite=False)
    return descriptor(destination)


def _legacy_plan_audit(plans: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Check old confirmed plans without importing their recognition or truth data."""
    old = read_json(OLD_TASKS)
    indexed = {item["id"]: item for item in old}
    if set(indexed) != set(plans):
        raise BindingError("historical_case_registry_changed")
    differences = {}
    for case_id, plan in plans.items():
        historical = indexed[case_id]["reference_plan"]
        if case_id in {"sealing_cover", "stiffened_panel"}:
            # The only expected difference is the later authorized coarse mesh.
            comparable = deepcopy(plan)
            historical_by_action = {op["action"]: op for op in historical["operations"]}
            for index, operation in enumerate(comparable["operations"]):
                if operation["action"] in {"set_global_mesh", "set_local_mesh"}:
                    comparable["operations"][index] = deepcopy(
                        historical_by_action[operation["action"]]
                    )
            comparable["reason"] = historical["reason"]
            if comparable != historical:
                raise BindingError("mesh_amendment_changed_non_mesh_semantics", case_id)
            differences[case_id] = {
                "historical_plan_digest": digest(historical),
                "g10_plan_digest": digest(plan),
                "reason": "2026-09-09 user-authorized mesh amendment",
            }
            continue
        comparable = deepcopy(plan)
        comparable["reason"] = historical["reason"]
        if comparable != historical:
            raise BindingError("historical_plan_semantics_changed", case_id)
    if read_json(OLD_MATERIALS) != MATERIALS:
        raise BindingError("confirmed_material_registry_changed")
    return differences


def build_manifest() -> dict[str, Any]:
    """Stage verified G09 outputs and create the immutable M6 manifest once."""
    if MANIFEST_PATH.exists():
        return verify_manifest()
    plans = case_plans()
    mesh_amendments = _legacy_plan_audit(plans)
    source_identity_digest: str | None = None
    case_rows = []
    for historical in _historical_rows():
        part = historical["part"]
        case_id = slug(part)
        if case_id not in plans:
            raise BindingError("unexpected_engineering_part", part)
        source_step = historical["input_step"]
        checked(source_step)
        origin = historical["fresh_output"]
        checked(origin)
        checked(historical["parity"])
        staged = _stage_prediction(origin, case_id)
        prediction = read_json(checked(staged))
        validate_prediction(prediction, source_step)
        identity = digest(prediction["source_identity"])
        if source_identity_digest is None:
            source_identity_digest = identity
        elif source_identity_digest != identity:
            raise BindingError("mixed_e3_source_identity")
        plan = plans[case_id]
        binding = resolve_plan(plan, prediction)
        case_rows.append({
            "case_id": case_id,
            "part": part,
            "domain": plan["domain"],
            "source_step": source_step,
            "prediction_origin": origin,
            "prediction": staged,
            "source_only_parity": historical["parity"],
            "plan": plan,
            "plan_digest": digest(plan),
            "frozen_binding": binding,
            "engineering_development_only": True,
        })
    case_rows.sort(key=lambda row: row["case_id"])
    if {row["case_id"] for row in case_rows} != set(plans):
        raise BindingError("missing_engineering_case")

    causality = causality_regression(case_rows)
    if not causality["all_passed"]:
        raise BindingError("causality_regression_failed")
    write_json(CAUSALITY_PATH, causality, overwrite=False)

    sources = {
        "g09_final_freeze": descriptor(G09_FREEZE),
        "g09_engineering_acceptance": descriptor(G09_ACCEPTANCE),
        "historical_confirmed_tasks": descriptor(OLD_TASKS),
        "historical_confirmed_materials": descriptor(OLD_MATERIALS),
        "contract_source": descriptor(Path(__file__)),
        "runner_source": descriptor(Path(__file__).with_name("m6_runner.py")),
        "legacy_cad_kernel": descriptor(ROOT / "tools/aei_case_suite/cad_executor.py"),
        "legacy_cae_compiler": descriptor(ROOT / "tools/aei_case_suite/cae_executor.py"),
        "legacy_cae_template": descriptor(ROOT / "tools/aei_case_suite/abaqus_generic.template.py"),
        "legacy_odb_verifier": descriptor(ROOT / "tools/aei_case_suite/abaqus_verification.py"),
        "legacy_geometry_adapter": descriptor(ROOT / "tools/aei_case_suite/geometry.py"),
        "legacy_preflight_measurement": descriptor(ROOT / "tools/aei_case_suite/preflight.py"),
        "legacy_case_geometry": descriptor(ROOT / "tools/aei_two_case_pilot/build_cases.py"),
        "legacy_plan_schema": descriptor(ROOT / "tools/aei_case_suite/contracts.py"),
        "legacy_plan_error": descriptor(ROOT / "tools/aei_two_case_pilot/plan_contract.py"),
        "legacy_parameter_policy": descriptor(ROOT / "tools/aei_case_suite/parameter_policy.py"),
        "environment_inventory": descriptor(REPORT / "ENVIRONMENT_INITIAL.json"),
        "abaqus_launcher": descriptor(Path("F:/ABAQUS2020/commands/abaqus.bat")),
    }
    manifest = {
        "schema": "g10_m6_engineering_manifest_v1",
        "created_utc": now(),
        "status": "frozen_before_g10_engineering_execution",
        "system": E3_SYSTEM,
        "deployment_seed": DEPLOYMENT_SEED,
        "runtime_versions": {"pythonocc": OCC_VERSION, "abaqus": ABAQUS_RELEASE},
        "source_identity_digest": source_identity_digest,
        "denominator": {"parts": 13, "cad": 6, "cae": 7},
        "cases": case_rows,
        "materials": MATERIALS,
        "mesh_sensitivity": MESH_SENSITIVITY,
        "mesh_amendments_from_historical_reference": mesh_amendments,
        "causality_regression": descriptor(CAUSALITY_PATH),
        "source_assets": sources,
        "execution_policy": {
            "isolated_directory_per_run": True,
            "one_receipt_per_case_run": True,
            "no_automatic_retry": True,
            "cad_requires_roundtrip_reimport_audit": True,
            "cae_requires_completed_odb_and_force_balance_audit": True,
            "compile_or_prepare_only_is_not_execution": True,
            "prediction_object_id_scope": "source geometry version only",
            "post_edit_mapping": "independent geometric witnesses, never permanent source face IDs",
        },
        "truth_used": False,
        "historical_truth_sidecar_read": False,
    }
    write_json(MANIFEST_PATH, manifest, overwrite=False)
    return verify_manifest()


def verify_manifest() -> dict[str, Any]:
    manifest = read_json(MANIFEST_PATH)
    if manifest.get("schema") != "g10_m6_engineering_manifest_v1" \
            or manifest.get("status") != "frozen_before_g10_engineering_execution" \
            or manifest.get("truth_used") is not False \
            or manifest.get("historical_truth_sidecar_read") is not False:
        raise BindingError("invalid_m6_manifest")
    if manifest.get("denominator") != {"parts": 13, "cad": 6, "cae": 7}:
        raise BindingError("engineering_denominator_changed")
    for item in manifest["source_assets"].values():
        checked(item)
    causality = read_json(checked(manifest["causality_regression"]))
    if causality.get("all_passed") is not True or causality.get("cases") != 13:
        raise BindingError("causality_evidence_changed")
    identities = set()
    for case in manifest["cases"]:
        checked(case["source_step"])
        checked(case["prediction_origin"])
        checked(case["source_only_parity"])
        prediction = read_json(checked(case["prediction"]))
        validate_prediction(prediction, case["source_step"])
        if case["plan_digest"] != digest(case["plan"]):
            raise BindingError("plan_changed", case["case_id"])
        assert_frozen_binding(case["frozen_binding"], case["plan"], prediction)
        identities.add(digest(prediction["source_identity"]))
    if identities != {manifest["source_identity_digest"]}:
        raise BindingError("mixed_e3_source_identity")
    return manifest


def case_from_manifest(manifest: dict[str, Any], case_id: str) -> dict[str, Any]:
    matches = [case for case in manifest["cases"] if case["case_id"] == case_id]
    if len(matches) != 1:
        raise BindingError("unknown_case", case_id)
    return matches[0]


if __name__ == "__main__":
    manifest = build_manifest()
    print(json.dumps({
        "manifest": descriptor(MANIFEST_PATH),
        "parts": manifest["denominator"]["parts"],
        "cad": manifest["denominator"]["cad"],
        "cae": manifest["denominator"]["cae"],
        "causality_all_passed": True,
        "truth_used": False,
    }, ensure_ascii=False, indent=2))
