"""One deterministic general prompt revision for P4-API-R2B.1."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml

from utils import cae_query_frame_runtime as base
from utils.cae_intent import validate_intent_result
from utils.cae_query_frame import compile_query_frame, validate_query_frame


ROOT = Path(__file__).resolve().parents[1]
P2_FREEZE = ROOT / "reports" / "p2_a1_cae_ir_v1" / "freeze"
P2_ROLE_POLICY_PATH = P2_FREEZE / "configs" / "cae_feature_role_policy.yaml"
P2_RESULT_SCHEMA_PATH = P2_FREEZE / "schemas" / "cae_intent_result.schema.json"
P2_ROLE_POLICY_SHA256 = "4022e7e0d6d09203bb456a81f5900f5570940f1e94294cb457fde249c2093943"
P2_RESULT_SCHEMA_SHA256 = "00d5da9125031158786bd853c172f0aae1fb63c0424fd19b3ff82f6918e37727"

DEFAULT_MAX_OUTPUT_TOKENS = base.DEFAULT_MAX_OUTPUT_TOKENS
KNOWN_FRAME_ERROR_CODES = base.KNOWN_FRAME_ERROR_CODES
NORMAL_ENDPOINT = base.NORMAL_ENDPOINT
PROFILE_NAME = base.PROFILE_NAME
QUERY_FRAME_SCHEMA_SHA256 = base.QUERY_FRAME_SCHEMA_SHA256
RuntimeContractError = base.RuntimeContractError
build_query_frame_repair_prompt = base.build_query_frame_repair_prompt


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _walk(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


@lru_cache(maxsize=1)
def _frozen_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    if _sha256(P2_ROLE_POLICY_PATH) != P2_ROLE_POLICY_SHA256:
        raise RuntimeContractError("R2B1_E_P2_ROLE_POLICY_HASH")
    if _sha256(P2_RESULT_SCHEMA_PATH) != P2_RESULT_SCHEMA_SHA256:
        raise RuntimeContractError("R2B1_E_P2_RESULT_SCHEMA_HASH")
    policy = yaml.safe_load(P2_ROLE_POLICY_PATH.read_text(encoding="utf-8"))
    result_schema = json.loads(P2_RESULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or not isinstance(result_schema, dict):
        raise RuntimeContractError("R2B1_E_FROZEN_SOURCE_TYPE")
    return policy, result_schema


@lru_cache(maxsize=1)
def _compatibility_rows() -> tuple[dict[str, Any], ...]:
    policy, _ = _frozen_sources()
    rows = []
    for feature, feature_value in sorted(policy["features"].items()):
        for role in sorted(feature_value["selectable_roles"], key=lambda item: item["role"]):
            rows.append({
                "feature": feature,
                "face_role": role["role"],
                "compatible_operations": sorted(
                    operation
                    for operation, review in role["operation_review"].items()
                    if review["compatible"] is True
                ),
            })
    return tuple(rows)


def feature_role_compatibility_table() -> list[dict[str, Any]]:
    return [dict(row) for row in _compatibility_rows()]


def _p2_error_codes() -> set[str]:
    _, schema = _frozen_sources()
    return {
        str(item)
        for node in _walk(schema)
        for item in node.get("enum", [])
        if isinstance(item, str) and item.startswith("E_")
    }


@lru_cache(maxsize=1)
def _status_rules() -> tuple[dict[str, str], ...]:
    rows = (
        {
            "rule_id": "unsupported_set_difference",
            "trigger": "unsupported set difference, exclusion, or complex negative selection",
            "outcome": "invalid",
            "source_error_code": "E_UNSUPPORTED_OPERATION",
        },
        {
            "rule_id": "nearest_without_reference",
            "trigger": "nearest selector without an explicit spatial reference",
            "outcome": "needs_clarification",
            "source_error_code": "E_MISSING_INFORMATION",
        },
        {
            "rule_id": "required_operation_information_missing",
            "trigger": "required unit, direction, count, or material parameter is absent",
            "outcome": "needs_clarification",
            "source_error_code": "E_MISSING_INFORMATION",
        },
        {
            "rule_id": "no_unconfirmed_assumptions",
            "trigger": "completion would require assuming information not stated by the user",
            "outcome": "needs_clarification",
            "source_error_code": "E_ASSUMPTION_UNCONFIRMED",
        },
    )
    codes = _p2_error_codes()
    if any(row["source_error_code"] not in codes for row in rows):
        raise RuntimeContractError("R2B1_E_STATUS_RULE_ERROR_CODE")
    return rows


def status_decision_rules() -> list[dict[str, str]]:
    return [dict(row) for row in _status_rules()]


def quantifier_rules() -> dict[str, str]:
    return {
        "all/every/所有": "all",
        "explicit integer N": "count+instance_count=N",
        "largest/smallest/nearest": "preserve selector; never map to all",
        "unique/the only/唯一一个": "exactly_one",
        "candidate cardinality": "never changes the text quantifier",
    }


def multi_operation_checklist() -> list[str]:
    return [
        "action", "target", "selector", "value", "unit",
        "direction_or_dof", "material_parameters", "mesh_size",
    ]


@lru_cache(maxsize=1)
def _revision_examples() -> tuple[dict[str, Any], ...]:
    examples = (
        {
            "name": "unsupported_set_difference",
            "provenance": "handcrafted_r2b_1",
            "input": {
                "text": "Refine every annular recess except those beside a datum notch.",
                "language": "en-US",
            },
            "frame": {
                "frame_version": "cae-query-frame/1.0.0",
                "status": "invalid",
                "error_code": "E_UNSUPPORTED_OPERATION",
                "error_path": "/intent/operations/0/target",
                "missing_information": [],
                "ambiguities": [],
                "clarification_questions": [],
            },
        },
        {
            "name": "nearest_without_reference",
            "provenance": "handcrafted_r2b_1",
            "input": {
                "text": "Use a 0.7 mm local mesh on the nearest rectangular boss.",
                "language": "en-US",
            },
            "frame": {
                "frame_version": "cae-query-frame/1.0.0",
                "status": "needs_clarification",
                "error_code": "E_MISSING_INFORMATION",
                "error_path": "/intent/operations/0/target/filters/nearest_to",
                "missing_information": [{
                    "code": "M_REFERENCE",
                    "path": "/intent/operations/0/target/filters/nearest_to",
                    "description": "The spatial reference for nearest is missing.",
                }],
                "ambiguities": [],
                "clarification_questions": [{
                    "key": "nearest-reference",
                    "path": "/intent/operations/0/target/filters/nearest_to",
                    "question": "What spatial reference defines nearest?",
                }],
            },
        },
        {
            "name": "explicit_local_face_role",
            "provenance": "handcrafted_r2b_1",
            "input": {
                "text": "Apply 0.4 MPa outward traction to the only rounded end wall of an obround opening, then mesh every chamfer bevel face at 0.6 mm.",
                "language": "en-US",
            },
            "frame": {
                "frame_version": "cae-query-frame/1.0.0",
                "status": "valid",
                "language": "en-US",
                "operations": [
                    {
                        "action": "surface_traction",
                        "feature": "obround_hole",
                        "face_role": "rounded_end_wall",
                        "selector": "exactly_one",
                        "magnitude": {"value": 0.4, "unit": "MPa"},
                        "direction": {"kind": "surface_normal"},
                    },
                    {
                        "action": "local_mesh_size",
                        "feature": "chamfer",
                        "face_role": "bevel_face",
                        "selector": "all",
                        "mesh_size": {"value": 0.6, "unit": "mm"},
                    },
                ],
            },
        },
        {
            "name": "count_and_exactly_one",
            "provenance": "handcrafted_r2b_1",
            "input": {
                "text": "Mesh three closed-slot side walls at 0.8 mm and the only circular boss top at 1.1 mm.",
                "language": "en-US",
            },
            "frame": {
                "frame_version": "cae-query-frame/1.0.0",
                "status": "valid",
                "language": "en-US",
                "operations": [
                    {
                        "action": "local_mesh_size",
                        "feature": "slot_o",
                        "face_role": "side_wall",
                        "selector": "count",
                        "instance_count": 3,
                        "mesh_size": {"value": 0.8, "unit": "mm"},
                    },
                    {
                        "action": "local_mesh_size",
                        "feature": "cylinder_boss",
                        "face_role": "top_face",
                        "selector": "exactly_one",
                        "mesh_size": {"value": 1.1, "unit": "mm"},
                    },
                ],
            },
        },
        {
            "name": "multi_operation_parameter_preservation",
            "provenance": "handcrafted_r2b_1",
            "input": {
                "text": "Assign Alloy Q with E 73 GPa and ratio 0.31, set a 3 mm global mesh, press every round through-opening wall at 1.2 MPa inward, and move its only wall U2 by 0.05 mm in the part frame.",
                "language": "en-US",
            },
            "frame": {
                "frame_version": "cae-query-frame/1.0.0",
                "status": "valid",
                "language": "en-US",
                "operations": [
                    {
                        "action": "material_assignment",
                        "material": {
                            "name": "Alloy Q",
                            "youngs_modulus": {"value": 73, "unit": "GPa"},
                            "poisson_ratio": 0.31,
                        },
                    },
                    {"action": "global_mesh_size", "mesh_size": {"value": 3, "unit": "mm"}},
                    {
                        "action": "pressure",
                        "feature": "through_hole_round",
                        "face_role": "hole_wall",
                        "selector": "all",
                        "magnitude": {"value": 1.2, "unit": "MPa"},
                        "direction": {"kind": "opposite_surface_normal"},
                    },
                    {
                        "action": "displacement",
                        "feature": "through_hole_round",
                        "face_role": "hole_wall",
                        "selector": "exactly_one",
                        "displacements": [{"dof": "U2", "value": 0.05, "unit": "mm"}],
                        "load_frame": "part_local_cartesian",
                    },
                ],
            },
        },
    )
    for item in examples:
        frame_check = validate_query_frame(item["frame"])
        if not frame_check["valid"]:
            raise RuntimeContractError("R2B1_E_EXAMPLE_FRAME", item["name"])
        compiled = compile_query_frame(item["frame"])
        if not validate_intent_result(compiled)["valid"]:
            raise RuntimeContractError("R2B1_E_EXAMPLE_P2", item["name"])
    return examples


def runtime_examples() -> tuple[dict[str, Any], ...]:
    return base.runtime_examples() + _revision_examples()


@lru_cache(maxsize=1)
def build_query_frame_runtime_prompt() -> str:
    sections = [
        base.build_query_frame_runtime_prompt().rstrip("\n"),
        "R2B.1 one-time general prompt revision. Apply these rules before producing a valid frame.",
        "FROZEN_P2_FEATURE_OPERATION_FACE_ROLE_COMPATIBILITY_JSON=" + _canonical_json(feature_role_compatibility_table()),
        "FAIL_CLOSED_STATUS_DECISION_RULES_JSON=" + _canonical_json(status_decision_rules()),
        "QUANTIFIER_PRESERVATION_RULES_JSON=" + _canonical_json(quantifier_rules()),
        "MULTI_OPERATION_OUTPUT_CHECKLIST_JSON=" + _canonical_json(multi_operation_checklist()),
        "First count every requested operation. Preserve text order and verify each checklist field before output. Never omit or rewrite parameters because targets may ground to the same region.",
        "Preserve an explicitly named local face role from the frozen policy; never replace it with feature_region.",
        "Do not invent missing information. Selector meaning comes from text and never from the number of geometry candidates.",
        "R2B1_HANDCRAFTED_GENERAL_EXAMPLES_JSON=" + _canonical_json(_revision_examples()),
    ]
    return "\n".join(sections) + "\n"


def runtime_prompt_audit() -> dict[str, Any]:
    prompt = build_query_frame_runtime_prompt()
    return {
        **base.runtime_prompt_audit(),
        "schema_version": "topocae-p4-api-r2b-1-runtime-prompt-audit/1",
        "runtime_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "runtime_prompt_utf8_bytes": len(prompt.encode("utf-8")),
        "p2_role_policy_sha256": P2_ROLE_POLICY_SHA256,
        "p2_result_schema_sha256": P2_RESULT_SCHEMA_SHA256,
        "compatibility_row_count": len(feature_role_compatibility_table()),
        "revision_example_count": len(_revision_examples()),
        "deterministic_reproduction": prompt.encode("utf-8") == build_query_frame_runtime_prompt().encode("utf-8"),
    }


def _validate_input(value: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"text", "language"}:
        raise RuntimeContractError("RUNTIME_E_INPUT_FIELDS")
    if not isinstance(value["text"], str) or not value["text"].strip():
        raise RuntimeContractError("RUNTIME_E_INPUT_TYPE")
    if not isinstance(value["language"], str) or not value["language"]:
        raise RuntimeContractError("RUNTIME_E_INPUT_TYPE")
    return {"text": value["text"], "language": value["language"]}


def build_deepseek_json_object_request(
    inference_input: Mapping[str, Any],
    *,
    model: str = base.DEFAULT_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> dict[str, Any]:
    clean = _validate_input(inference_input)
    return {
        "endpoint": NORMAL_ENDPOINT,
        "profile": PROFILE_NAME,
        "payload": {
            "model": model,
            "messages": [
                {"role": "system", "content": build_query_frame_runtime_prompt()},
                {"role": "user", "content": _canonical_json(clean)},
            ],
            "temperature": 0.0,
            "max_tokens": int(max_output_tokens),
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
        },
    }


MockTransport = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]


def execute_mock_query_frame_request(
    inference_input: Mapping[str, Any], mock_transport: MockTransport
) -> dict[str, Any]:
    request = build_deepseek_json_object_request(inference_input)
    response = mock_transport(request["endpoint"], request["payload"])
    raw = response["choices"][0]["message"]["content"]
    frame = json.loads(raw)
    validation = validate_query_frame(frame)
    if not validation["valid"]:
        first = validation["errors"][0]
        raise RuntimeContractError(str(first["error_code"]), str(first["field_path"]))
    compiled = compile_query_frame(frame)
    if not validate_intent_result(compiled)["valid"]:
        raise RuntimeContractError("R2B1_E_MOCK_P2")
    return {
        "frame": frame,
        "compiled_result": compiled,
        "frame_valid": True,
        "compiler_valid": True,
        "p2_valid": True,
        "mock_request_count": 1,
        "network_request_count": 0,
        "endpoint": request["endpoint"],
        "profile": request["profile"],
    }


__all__ = [
    "DEFAULT_MAX_OUTPUT_TOKENS", "KNOWN_FRAME_ERROR_CODES", "NORMAL_ENDPOINT",
    "PROFILE_NAME", "QUERY_FRAME_SCHEMA_SHA256", "RuntimeContractError",
    "build_deepseek_json_object_request", "build_query_frame_repair_prompt",
    "build_query_frame_runtime_prompt", "execute_mock_query_frame_request",
    "feature_role_compatibility_table", "multi_operation_checklist",
    "quantifier_rules", "runtime_examples", "runtime_prompt_audit",
    "status_decision_rules",
]
