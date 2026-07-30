"""Frozen-policy hybrid resolution for P4-R3A."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from utils.cae_intent import validate_intent_result
from utils.cae_query_frame import FRAME_VERSION, compile_query_frame, validate_query_frame
from utils.cae_query_frame_runtime_r2b_1 import P2_ROLE_POLICY_SHA256
from utils.cae_text_evidence import CAETextEvidenceExtractor


ROOT = Path(__file__).resolve().parents[1]
P2_POLICY_PATH = ROOT / "reports" / "p2_a1_cae_ir_v1" / "freeze" / "configs" / "cae_feature_role_policy.yaml"
RESOLUTION_VERSION = "cae-hybrid-resolution/1.0.0"
_IDENTITY_KEYS = frozenset({
    "face_id", "face_ids", "face_index", "face_indices", "feature_instance_id",
    "feature_instance_ids", "instance_id", "instance_ids", "ground_truth",
    "source_identity", "source_ref", "target_face_indices", "target_instance_ids",
})
_MISSING_CODES = {
    "feature": "M_FEATURE_TYPE", "face_role": "M_FACE_ROLE", "selector": "M_QUANTIFIER",
    "magnitude": "M_MAGNITUDE", "direction": "M_DIRECTION",
    "displacements": "M_DISPLACEMENT_COMPONENT", "load_frame": "M_REFERENCE",
    "material.name": "M_MATERIAL_NAME", "material.youngs_modulus": "M_MATERIAL_PROPERTY",
    "material.poisson_ratio": "M_MATERIAL_PROPERTY", "mesh_size": "M_MESH_SIZE",
    "nearest_to": "M_REFERENCE",
}


class HybridResolutionError(RuntimeError):
    """The resolver could not produce a contract-valid fail-closed result."""


@lru_cache(maxsize=1)
def _policy() -> dict[str, Any]:
    digest = hashlib.sha256(P2_POLICY_PATH.read_bytes()).hexdigest()
    if digest != P2_ROLE_POLICY_SHA256:
        raise HybridResolutionError("P4R3A_E_P2_POLICY_HASH")
    value = yaml.safe_load(P2_POLICY_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("features"), dict):
        raise HybridResolutionError("P4R3A_E_P2_POLICY_TYPE")
    return value


def _identity_present(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(str(key).lower() in _IDENTITY_KEYS or _identity_present(child) for key, child in value.items())
    if isinstance(value, list):
        return any(_identity_present(child) for child in value)
    return False


def _policy_decision(feature: str, role: str, action: str) -> tuple[bool, str | None]:
    feature_value = _policy()["features"].get(feature)
    if not isinstance(feature_value, Mapping):
        return False, None
    for role_value in feature_value.get("selectable_roles", []):
        if role_value.get("role") != role:
            continue
        review = role_value.get("operation_review", {}).get(action, {})
        return review.get("compatible") is True, review.get("review_id")
    return False, None


def _item_value(item: Mapping[str, Any]) -> Any:
    return deepcopy(item["normalized"])


def _trace(
    field_path: str,
    original: Any,
    final: Any,
    decision: str,
    evidence: Mapping[str, Any] | None = None,
    policy_reference: str | None = None,
) -> dict[str, Any]:
    return {
        "field_path": field_path,
        "original_llm_value": deepcopy(original),
        "final_value": deepcopy(final),
        "evidence_span": deepcopy(evidence["span"]) if evidence else None,
        "rule_id": str(evidence["rule_id"]) if evidence else None,
        "p2_policy_reference": policy_reference,
        "decision": decision,
    }


def _invalid_frame(code: str, path: str) -> dict[str, Any]:
    return {
        "frame_version": FRAME_VERSION,
        "status": "invalid",
        "error_code": code,
        "error_path": path,
        "missing_information": [],
        "ambiguities": [],
        "clarification_questions": [],
    }


def _missing_path(index: int, parameter: str) -> str:
    if parameter.startswith("material."):
        return f"/operations/{index}/material/{parameter.split('.', 1)[1]}"
    if parameter == "nearest_to":
        return f"/operations/{index}/geometry_filters/nearest_to"
    return f"/operations/{index}/{parameter}"


def _clarification_frame(missing: list[tuple[int, str]], ambiguity_path: str | None = None) -> dict[str, Any]:
    if ambiguity_path is not None:
        return {
            "frame_version": FRAME_VERSION,
            "status": "needs_clarification",
            "error_code": "E_AMBIGUOUS_REQUIREMENT",
            "error_path": ambiguity_path,
            "missing_information": [],
            "ambiguities": [{
                "code": "A_FACE_ROLE",
                "path": ambiguity_path,
                "description": "Explicit text and frozen feature-role policy do not uniquely resolve the field.",
                "options": ["provide_explicit_role", "choose_policy_compatible_role"],
            }],
            "clarification_questions": [{
                "key": "q1", "path": ambiguity_path,
                "question": "Which policy-compatible engineering face role is intended?",
            }],
        }
    unique = []
    seen = set()
    for index, parameter in missing:
        key = (index, parameter)
        if key not in seen:
            seen.add(key)
            unique.append(key)
    information = []
    questions = []
    for number, (index, parameter) in enumerate(unique, 1):
        path = _missing_path(index, parameter)
        information.append({
            "code": _MISSING_CODES.get(parameter, "M_TARGET"),
            "path": path,
            "description": f"Explicit text does not uniquely provide {parameter}.",
        })
        questions.append({"key": f"q{number}", "path": path, "question": f"What {parameter} should operation {index + 1} use?"})
    if not information:
        information.append({"code": "M_TARGET", "path": "/operations", "description": "Operations cannot be uniquely aligned to text clauses."})
        questions.append({"key": "q1", "path": "/operations", "question": "Please separate and specify each requested operation."})
    return {
        "frame_version": FRAME_VERSION,
        "status": "needs_clarification",
        "error_code": "E_MISSING_INFORMATION",
        "error_path": information[0]["path"],
        "missing_information": information,
        "ambiguities": [],
        "clarification_questions": questions,
    }


class HybridCAEQueryFrameResolver:
    """Combine a QueryFrame proposal with explicit text evidence."""

    def __init__(self, extractor: CAETextEvidenceExtractor | None = None):
        self.extractor = extractor or CAETextEvidenceExtractor()

    def resolve(self, inference_input: Mapping[str, Any], proposal: Mapping[str, Any]) -> dict[str, Any]:
        evidence = self.extractor.extract(inference_input)
        trace: list[dict[str, Any]] = []

        if _identity_present(proposal):
            frame = _invalid_frame("E_IDENTITY_FIELD_FORBIDDEN", "/proposal")
            trace.append(_trace("/proposal", "<redacted>", None, "reject"))
            return self._finish(frame, evidence, trace)

        exclusions = [item for item in evidence["evidence"] if item["category"] == "exclusion"]
        if exclusions:
            frame = _invalid_frame("E_UNSUPPORTED_OPERATION", "/operations/target/selector")
            trace.append(_trace("/operations/target/selector", None, None, "reject", exclusions[0]))
            return self._finish(frame, evidence, trace)

        initial_validation = validate_query_frame(proposal)
        if not initial_validation["valid"]:
            frame = _invalid_frame("E_SCHEMA_VALIDATION", "/proposal")
            trace.append(_trace("/proposal", "<schema-invalid>", None, "reject"))
            return self._finish(frame, evidence, trace)

        if proposal["status"] != "valid":
            trace.append(_trace("/status", proposal["status"], proposal["status"], "accept"))
            return self._finish(deepcopy(proposal), evidence, trace)

        frame = deepcopy(proposal)
        operations = frame["operations"]
        clause_actions = self._clause_actions(evidence)
        if len(clause_actions) != len(operations):
            trace.append(_trace("/operations", len(operations), None, "clarify"))
            return self._finish(_clarification_frame([]), evidence, trace)

        missing = self._missing_by_operation(evidence, len(operations))
        if missing:
            for index, parameter, item in missing:
                trace.append(_trace(_missing_path(index, parameter), operations[index].get(parameter), None, "clarify", item))
            return self._finish(_clarification_frame([(index, parameter) for index, parameter, _ in missing]), evidence, trace)

        for index, operation in enumerate(operations):
            clause_index = clause_actions[index][0]
            issue = self._resolve_operation(operation, index, clause_index, evidence, trace)
            if issue is not None:
                trace.append(_trace(issue, None, None, "clarify"))
                return self._finish(_clarification_frame([], issue), evidence, trace)

        return self._finish(frame, evidence, trace)

    @staticmethod
    def _items(evidence: Mapping[str, Any], clause_index: int, category: str) -> list[dict[str, Any]]:
        return [item for item in evidence["evidence"] if item["clause_index"] == clause_index and item["category"] == category]

    def _clause_actions(self, evidence: Mapping[str, Any]) -> list[tuple[int, dict[str, Any]]]:
        rows = []
        for clause in evidence["clauses"]:
            items = self._items(evidence, clause["index"], "operation_cue")
            values = {str(item["normalized"]) for item in items}
            if len(values) == 1:
                rows.append((clause["index"], items[0]))
        return rows

    def _missing_by_operation(self, evidence: Mapping[str, Any], operation_count: int) -> list[tuple[int, str, dict[str, Any]]]:
        action_rows = self._clause_actions(evidence)
        clause_to_operation = {clause_index: index for index, (clause_index, _) in enumerate(action_rows[:operation_count])}
        rows = []
        for item in evidence["evidence"]:
            if item["category"] != "missing_required_parameter" or item["clause_index"] not in clause_to_operation:
                continue
            rows.append((clause_to_operation[item["clause_index"]], str(item["normalized"]["parameter"]), item))
        return rows

    def _resolve_operation(
        self,
        operation: dict[str, Any],
        index: int,
        clause_index: int,
        evidence: Mapping[str, Any],
        trace: list[dict[str, Any]],
    ) -> str | None:
        path = f"/operations/{index}"
        action_item = self._unique(evidence, clause_index, "operation_cue")
        feature_item = self._unique(evidence, clause_index, "feature_mention")
        role_item = self._unique(evidence, clause_index, "face_role_mention")

        for field, item in (("action", action_item), ("feature", feature_item), ("face_role", role_item)):
            original = operation.get(field)
            if item is not None and item["normalized"] != original:
                operation[field] = deepcopy(item["normalized"])
                decision = "override_by_explicit_evidence"
            else:
                decision = "accept"
            trace.append(_trace(f"{path}/{field}", original, operation.get(field), decision, item))
        for field in ("action", "feature", "face_role"):
            if not operation.get(field):
                trace.append(_trace(f"{path}/{field}", None, None, "clarify"))
                return f"{path}/{field}"

        compatible, review_id = _policy_decision(operation["feature"], operation["face_role"], operation["action"])
        for row in trace[-3:]:
            row["p2_policy_reference"] = review_id
        if not compatible:
            return f"{path}/face_role"

        selector_items = self._items(evidence, clause_index, "selector")
        selector_values = {item["normalized"]["selector"] for item in selector_items}
        if len(selector_values) > 1:
            return f"{path}/selector"
        if selector_items:
            item = selector_items[0]
            original = operation.get("selector")
            normalized = item["normalized"]
            operation["selector"] = normalized["selector"]
            operation.pop("instance_count", None)
            operation.pop("compare_by", None)
            if normalized["selector"] == "count":
                operation["instance_count"] = normalized["count"]
            if normalized["selector"] in {"largest", "smallest"}:
                operation["compare_by"] = normalized["compare_by"]
            if normalized["selector"] != "nearest":
                filters = operation.get("geometry_filters")
                if isinstance(filters, dict):
                    filters.pop("nearest_to", None)
                    if not filters:
                        operation.pop("geometry_filters", None)
                if "geometry_filters" not in operation:
                    operation.pop("selection_frame", None)
            decision = "accept" if original == operation["selector"] else "override_by_explicit_evidence"
            trace.append(_trace(f"{path}/selector", original, operation["selector"], decision, item, review_id))
            if normalized["selector"] == "nearest":
                reference = self._unique(evidence, clause_index, "spatial_reference")
                if reference is None:
                    return f"{path}/geometry_filters/nearest_to"
                operation.setdefault("geometry_filters", {})["nearest_to"] = self._reference(_item_value(reference))
                operation["selection_frame"] = "global_cartesian"

        action = operation["action"]
        if action in {"pressure", "surface_traction"}:
            self._override(operation, "magnitude", self._unique(evidence, clause_index, "load_magnitude"), path, trace, review_id)
            direction = self._unique(evidence, clause_index, "direction")
            if direction is not None:
                normalized = self._direction(_item_value(direction))
                self._override_value(operation, "direction", normalized, direction, path, trace, review_id)
        elif action == "displacement":
            components = self._items(evidence, clause_index, "displacement_component")
            if components:
                values = [deepcopy(item["normalized"]) for item in components]
                self._override_value(operation, "displacements", values, components[0], path, trace, review_id)
            frame_item = self._unique(evidence, clause_index, "coordinate_frame")
            if frame_item is not None:
                frame_map = {"global": "global_cartesian", "local": "part_local_cartesian"}
                if frame_item["normalized"] not in frame_map:
                    return f"{path}/load_frame"
                self._override_value(operation, "load_frame", frame_map[frame_item["normalized"]], frame_item, path, trace, review_id)
        elif action in {"global_mesh_size", "local_mesh_size"}:
            self._override(operation, "mesh_size", self._unique(evidence, clause_index, "mesh_size"), path, trace, review_id)
        elif action == "material_assignment":
            name = self._unique(evidence, clause_index, "material_name")
            youngs = self._unique(evidence, clause_index, "youngs_modulus")
            poisson = self._unique(evidence, clause_index, "poisson_ratio")
            if name and youngs and poisson:
                material = {"name": name["normalized"], "youngs_modulus": deepcopy(youngs["normalized"]), "poisson_ratio": poisson["normalized"]}
                self._override_value(operation, "material", material, name, path, trace, review_id)
        elif action == "fixed_support":
            dofs = [item["normalized"] for item in self._items(evidence, clause_index, "degree_of_freedom")]
            if dofs:
                if set(dofs) != {"U1", "U2", "U3"}:
                    return f"{path}/constrained_dofs"
                self._override_value(operation, "constrained_dofs", dofs, self._items(evidence, clause_index, "degree_of_freedom")[0], path, trace, review_id)
        return None

    def _unique(self, evidence: Mapping[str, Any], clause_index: int, category: str) -> dict[str, Any] | None:
        items = self._items(evidence, clause_index, category)
        unique = {}
        for item in items:
            unique[repr(item["normalized"])] = item
        return next(iter(unique.values())) if len(unique) == 1 else None

    @staticmethod
    def _direction(value: Mapping[str, Any]) -> dict[str, Any]:
        if value["kind"] == "normal":
            return {"kind": "surface_normal"}
        if value["kind"] == "opposite_normal":
            return {"kind": "opposite_surface_normal"}
        return {"kind": "global_cartesian", "vector": list(value["vector"])}

    @staticmethod
    def _reference(value: Mapping[str, Any]) -> dict[str, Any]:
        if value["kind"] == "axis":
            return {"kind": "datum_axis", "axis": value["axis"]}
        return {"kind": "part_centroid"}

    def _override(self, operation: dict[str, Any], field: str, item: Mapping[str, Any] | None, path: str, trace: list[dict[str, Any]], review_id: str | None) -> None:
        if item is not None:
            self._override_value(operation, field, {key: item["normalized"][key] for key in ("value", "unit")}, item, path, trace, review_id)

    @staticmethod
    def _override_value(operation: dict[str, Any], field: str, value: Any, item: Mapping[str, Any], path: str, trace: list[dict[str, Any]], review_id: str | None) -> None:
        original = deepcopy(operation.get(field))
        operation[field] = deepcopy(value)
        decision = "accept" if original == value else "override_by_explicit_evidence"
        trace.append(_trace(f"{path}/{field}", original, value, decision, item, review_id))

    @staticmethod
    def _finish(frame: Mapping[str, Any], evidence: Mapping[str, Any], trace: list[dict[str, Any]]) -> dict[str, Any]:
        validation = validate_query_frame(frame)
        if not validation["valid"]:
            raise HybridResolutionError(f"P4R3A_E_FRAME_OUTPUT:{validation['errors'][0]}")
        compiled = compile_query_frame(frame)
        p2 = validate_intent_result(compiled)
        if not p2["valid"]:
            raise HybridResolutionError(f"P4R3A_E_P2_OUTPUT:{p2['errors'][0]}")
        return {
            "resolution_version": RESOLUTION_VERSION,
            "frame": deepcopy(frame),
            "compiled_result": compiled,
            "frame_valid": True,
            "p2_valid": True,
            "evidence": deepcopy(evidence),
            "decision_trace": deepcopy(trace),
        }


__all__ = ["HybridCAEQueryFrameResolver", "HybridResolutionError", "RESOLUTION_VERSION"]
