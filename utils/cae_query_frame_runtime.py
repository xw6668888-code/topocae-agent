"""Offline runtime prompt and mock request contract for CAEQueryFrame v1."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from utils.cae_intent import validate_intent_result
from utils.cae_query_frame import (
    canonical_to_query_frame,
    compile_query_frame,
    load_schema,
    validate_query_frame,
)


ROOT = Path(__file__).resolve().parents[1]
R2A_REPORT = ROOT / "reports" / "p4_api_r2a_compact_query_frame_v1"
QUERY_FRAME_SCHEMA_PATH = R2A_REPORT / "query_frame.schema.json"
R2A_BASE_PROMPT_PATH = R2A_REPORT / "query_frame_prompt.zh-CN.md"
QUERY_FRAME_SCHEMA_SHA256 = "5f1913121e750493520d818b1d377d90add70a72e90c6612f6ec47cd7179da43"
R2A_BASE_PROMPT_SHA256 = "aae89eb07898edcf98962e9eabce052bd878fd5b96f2185b9104b19b3f5c7142"
PROFILE_NAME = "deepseek_json_object"
NORMAL_ENDPOINT = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_MAX_OUTPUT_TOKENS = 2048
MAX_SCHEMA_SUMMARY_BYTES = 2048

KNOWN_FRAME_ERROR_CODES = frozenset({
    "QF_E_JSON_TYPE",
    "QF_E_IDENTITY_FIELD_FORBIDDEN",
    "QF_E_INVALID_ENUM",
    "QF_E_REQUIRED_FIELD",
    "QF_E_UNEXPECTED_FIELD",
    "QF_E_SCHEMA_VALIDATION",
    "QF_E_CONFLICT",
    "QF_E_OPERATION_ID_DUPLICATE",
    "QF_E_CANONICAL_REJECTED",
    "QF_E_CANONICAL_INPUT_INVALID",
    "QF_E_CANONICAL_UNREPRESENTABLE",
})

MUTUAL_EXCLUSION_RULES = (
    {
        "when": "status=valid",
        "required": ["language", "operations"],
        "forbidden": [
            "error_code", "error_path", "missing_information", "ambiguities",
            "clarification_questions",
        ],
    },
    {
        "when": "status in needs_clarification,invalid",
        "required": [
            "error_code", "error_path", "missing_information", "ambiguities",
            "clarification_questions",
        ],
        "forbidden": ["language", "operations"],
    },
    {
        "when": "action in fixed_support,displacement,pressure,surface_traction,local_mesh_size",
        "required": ["feature", "face_role", "selector"],
        "forbidden": [],
    },
    {
        "when": "action in material_assignment,global_mesh_size",
        "required": [],
        "forbidden": [
            "feature", "face_role", "selector", "instance_count", "compare_by",
            "geometry_filters", "selection_frame",
        ],
    },
    {
        "when": "action parameters",
        "required_by_action": {
            "fixed_support": ["constrained_dofs"],
            "displacement": ["displacements", "load_frame"],
            "pressure": ["magnitude", "direction"],
            "surface_traction": ["magnitude", "direction"],
            "material_assignment": ["material"],
            "global_mesh_size": ["mesh_size"],
            "local_mesh_size": ["mesh_size"],
        },
        "exact_parameter_branch": True,
    },
    {
        "when": "selector=count",
        "required": ["instance_count"],
        "forbidden_unless": "instance_count",
    },
    {
        "when": "selector in largest,smallest",
        "required": ["compare_by"],
        "forbidden_unless": "compare_by",
    },
    {
        "when": "selector=nearest",
        "required": ["geometry_filters.nearest_to", "selection_frame"],
        "forbidden": [],
    },
)

_BANNED_PROMPT_TERMS = (
    "face id", "instance id", "source identity", "ground truth", "graph", "answer",
)
_BANNED_SUMMARY_TERMS = (
    "ground_truth", "canonical_target", "face_id", "instance_id",
    "source_identity", "graph", "answer",
)


class RuntimeContractError(ValueError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}:{detail}" if detail else code)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _verify_frozen_sources() -> None:
    if _sha256(QUERY_FRAME_SCHEMA_PATH) != QUERY_FRAME_SCHEMA_SHA256:
        raise RuntimeContractError("RUNTIME_E_R2A_SCHEMA_HASH")
    if _sha256(R2A_BASE_PROMPT_PATH) != R2A_BASE_PROMPT_SHA256:
        raise RuntimeContractError("RUNTIME_E_R2A_PROMPT_HASH")


def _walk_schema(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Mapping[str, Any]]]:
    if isinstance(value, Mapping):
        yield path, value
        for key, child in value.items():
            yield from _walk_schema(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_schema(child, path + (str(index),))


def _schema_inventory(schema: Mapping[str, Any]) -> dict[str, Any]:
    objects = []
    enums = []
    for path, node in _walk_schema(schema):
        if node.get("type") == "object":
            properties = list(node.get("properties", {}))
            required = list(node.get("required", []))
            objects.append({
                "path": "/" + "/".join(path) if path else "/",
                "required": required,
                "optional": sorted(set(properties) - set(required)),
                "closed": node.get("additionalProperties") is False,
            })
        if "enum" in node:
            enums.append({
                "path": "/" + "/".join(path),
                "values": list(node["enum"]),
            })
    return {
        "objects": sorted(objects, key=lambda item: item["path"]),
        "enums": sorted(enums, key=lambda item: item["path"]),
    }


def _rules_schema_consistent(schema: Mapping[str, Any]) -> bool:
    root_fields = set(schema["properties"])
    operation = schema["$defs"]["operation"]["properties"]
    operation_fields = set(operation)
    actions = set(operation["action"]["enum"])
    selectors = set(operation["selector"]["enum"])
    dotted_allowed = {"geometry_filters.nearest_to"}
    for rule in MUTUAL_EXCLUSION_RULES:
        for field in rule.get("required", []):
            if field not in root_fields | operation_fields | dotted_allowed:
                return False
        for field in rule.get("forbidden", []):
            if field not in root_fields | operation_fields:
                return False
        mapping = rule.get("required_by_action", {})
        if mapping and set(mapping) != actions:
            return False
        if any(field not in operation_fields for fields in mapping.values() for field in fields):
            return False
    return {"count", "largest", "smallest", "nearest"}.issubset(selectors)


@lru_cache(maxsize=1)
def runtime_examples() -> tuple[dict[str, Any], ...]:
    examples: tuple[dict[str, Any], ...] = (
        {
            "name": "valid_single_pressure",
            "provenance": "handcrafted_r2a_1",
            "input": {
                "text": "Apply a 2 MPa normal pressure to every complete circular pocket region.",
                "language": "en-US",
            },
            "frame": {
                "frame_version": "cae-query-frame/1.0.0",
                "status": "valid",
                "language": "en-US",
                "operations": [{
                    "action": "pressure",
                    "feature": "pocket_circular",
                    "face_role": "feature_region",
                    "selector": "all",
                    "magnitude": {"value": 2, "unit": "MPa"},
                    "direction": {"kind": "surface_normal"},
                }],
            },
        },
        {
            "name": "valid_multi_setup",
            "provenance": "handcrafted_r2a_1",
            "input": {
                "text": "为整件指定线弹性材料和 4 mm 全局网格，并将所有加强筋局部细化到 1 mm。",
                "language": "zh-CN",
            },
            "frame": {
                "frame_version": "cae-query-frame/1.0.0",
                "status": "valid",
                "language": "zh-CN",
                "operations": [
                    {
                        "action": "material_assignment",
                        "material": {
                            "name": "generic_linear_elastic",
                            "youngs_modulus": {"value": 70, "unit": "GPa"},
                            "poisson_ratio": 0.33,
                        },
                    },
                    {"action": "global_mesh_size", "mesh_size": {"value": 4, "unit": "mm"}},
                    {
                        "action": "local_mesh_size",
                        "feature": "rib",
                        "face_role": "feature_region",
                        "selector": "all",
                        "mesh_size": {"value": 1, "unit": "mm"},
                    },
                ],
            },
        },
        {
            "name": "needs_direction",
            "provenance": "handcrafted_r2a_1",
            "input": {
                "text": "Apply 3 MPa pressure to the rectangular pocket floor.",
                "language": "en-US",
            },
            "frame": {
                "frame_version": "cae-query-frame/1.0.0",
                "status": "needs_clarification",
                "error_code": "E_MISSING_INFORMATION",
                "error_path": "/operations/0/parameters/direction",
                "missing_information": [{
                    "code": "M_DIRECTION",
                    "path": "/operations/0/parameters/direction",
                    "description": "The pressure orientation is not specified.",
                }],
                "ambiguities": [],
                "clarification_questions": [{
                    "key": "q1",
                    "path": "/operations/0/parameters/direction",
                    "question": "Which pressure orientation should be used?",
                }],
            },
        },
        {
            "name": "invalid_analysis",
            "provenance": "handcrafted_r2a_1",
            "input": {
                "text": "Run an unsupported transient thermal study.",
                "language": "en-US",
            },
            "frame": {
                "frame_version": "cae-query-frame/1.0.0",
                "status": "invalid",
                "error_code": "E_UNSUPPORTED_ANALYSIS",
                "error_path": "/analysis_type",
                "missing_information": [],
                "ambiguities": [],
                "clarification_questions": [],
            },
        },
    )
    for item in examples:
        validation = validate_query_frame(item["frame"])
        if not validation["valid"]:
            raise RuntimeContractError("RUNTIME_E_EXAMPLE_FRAME", item["name"])
        compiled = compile_query_frame(item["frame"])
        if not validate_intent_result(compiled)["valid"]:
            raise RuntimeContractError("RUNTIME_E_EXAMPLE_COMPILE", item["name"])
    return examples


@lru_cache(maxsize=1)
def build_query_frame_runtime_prompt() -> str:
    _verify_frozen_sources()
    schema = load_schema()
    inventory = _schema_inventory(schema)
    sections = [
        "CAEQueryFrame v1 runtime contract for DeepSeek JSON-object mode.",
        f"R2A_BASE_PROMPT_SHA256={R2A_BASE_PROMPT_SHA256}",
        f"QUERY_FRAME_SCHEMA_SHA256={QUERY_FRAME_SCHEMA_SHA256}",
        "Return exactly one complete JSON object and no surrounding prose or markup.",
        "Extract only explicit CAE operation, feature category, engineering surface role, quantity scope, geometric descriptors, loads, constraints, material, mesh parameters, and status.",
        "Do not infer missing engineering values. Use needs_clarification for resolvable missing or ambiguous input, and invalid for unsupported input. Non-valid frames must not contain operations.",
        "Use enum strings exactly as supplied by the embedded Schema. The deterministic compiler supplies the fixed analysis context and validates the result.",
        "EMBEDDED_QUERY_FRAME_SCHEMA_JSON=" + _canonical_json(schema),
        "SCHEMA_DERIVED_FIELD_INVENTORY_JSON=" + _canonical_json(inventory),
        "COMPILER_MUTUAL_EXCLUSION_RULES_JSON=" + _canonical_json(MUTUAL_EXCLUSION_RULES),
        "HANDCRAFTED_SCHEMA_VALIDATED_EXAMPLES_JSON=" + _canonical_json(runtime_examples()),
    ]
    prompt = "\n".join(sections) + "\n"
    lowered = prompt.lower()
    for term in _BANNED_PROMPT_TERMS:
        if term in lowered:
            raise RuntimeContractError("RUNTIME_E_FORBIDDEN_PROMPT_TERM", term)
    return prompt


def runtime_prompt_audit() -> dict[str, Any]:
    prompt = build_query_frame_runtime_prompt()
    schema = load_schema()
    inventory = _schema_inventory(schema)
    operation = schema["$defs"]["operation"]["properties"]
    return {
        "schema_version": "topocae-p4-api-r2a-1-runtime-prompt-audit/1",
        "runtime_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "runtime_prompt_utf8_bytes": len(prompt.encode("utf-8")),
        "query_frame_schema_sha256": QUERY_FRAME_SCHEMA_SHA256,
        "base_prompt_sha256": R2A_BASE_PROMPT_SHA256,
        "base_instruction_source_verified": True,
        "json_literal_present": "JSON" in prompt,
        "embedded_schema_exact": _canonical_json(schema) in prompt,
        "root_fields": list(schema["properties"]),
        "operation_fields": list(operation),
        "action_enums": list(operation["action"]["enum"]),
        "object_inventory_count": len(inventory["objects"]),
        "enum_inventory_count": len(inventory["enums"]),
        "required_optional_inventory_complete": all(
            set(item["required"]) | set(item["optional"])
            for item in inventory["objects"]
        ),
        "mutual_exclusion_rules_schema_consistent": _rules_schema_consistent(schema),
        "example_count": len(runtime_examples()),
        "example_statuses": sorted({item["frame"]["status"] for item in runtime_examples()}),
        "multi_operation_example_count": sum(
            len(item["frame"].get("operations", [])) > 1 for item in runtime_examples()
        ),
        "forbidden_term_count": sum(term in prompt.lower() for term in _BANNED_PROMPT_TERMS),
        "deterministic_reproduction": prompt.encode("utf-8") == build_query_frame_runtime_prompt().encode("utf-8"),
    }


def _validate_inference_input(inference_input: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(inference_input, Mapping) or set(inference_input) != {"text", "language"}:
        raise RuntimeContractError("RUNTIME_E_INPUT_FIELDS")
    text = inference_input["text"]
    language = inference_input["language"]
    if not isinstance(text, str) or not text.strip() or not isinstance(language, str) or not language:
        raise RuntimeContractError("RUNTIME_E_INPUT_TYPE")
    return {"text": text, "language": language}


def build_query_frame_repair_prompt(
    inference_input: Mapping[str, Any],
    error_code: str,
    schema_summary: Mapping[str, Any],
    *,
    attempt: int = 1,
) -> str:
    clean_input = _validate_inference_input(inference_input)
    if attempt != 1:
        raise RuntimeContractError("RUNTIME_E_REPAIR_LIMIT")
    if error_code not in KNOWN_FRAME_ERROR_CODES:
        raise RuntimeContractError("RUNTIME_E_ERROR_CODE")
    required_summary = {"field_path", "required", "allowed_enum", "rule"}
    if not isinstance(schema_summary, Mapping) or set(schema_summary) != required_summary:
        raise RuntimeContractError("RUNTIME_E_SCHEMA_SUMMARY_FIELDS")
    summary_text = _canonical_json(schema_summary)
    if len(summary_text.encode("utf-8")) > MAX_SCHEMA_SUMMARY_BYTES:
        raise RuntimeContractError("RUNTIME_E_SCHEMA_SUMMARY_SIZE")
    lowered = summary_text.lower()
    if any(term in lowered for term in _BANNED_SUMMARY_TERMS):
        raise RuntimeContractError("RUNTIME_E_SCHEMA_SUMMARY_EVIDENCE")
    value = {
        "input": clean_input,
        "error_code": error_code,
        "schema_summary": dict(schema_summary),
        "instruction": "Return one complete replacement JSON object matching only this schema summary.",
        "repair_attempt": 1,
    }
    return _canonical_json(value)


def build_deepseek_json_object_request(
    inference_input: Mapping[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> dict[str, Any]:
    clean_input = _validate_inference_input(inference_input)
    if not isinstance(model, str) or not model or max_output_tokens < 1:
        raise RuntimeContractError("RUNTIME_E_REQUEST_CONFIG")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": build_query_frame_runtime_prompt()},
            {"role": "user", "content": _canonical_json(clean_input)},
        ],
        "temperature": 0.0,
        "max_tokens": int(max_output_tokens),
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
    }
    return {"endpoint": NORMAL_ENDPOINT, "profile": PROFILE_NAME, "payload": payload}


def _strict_json_object(raw: str) -> dict[str, Any]:
    duplicates: list[str] = []

    def pairs_hook(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                duplicates.append(str(key))
            value[key] = item
        return value

    try:
        value = json.loads(raw, object_pairs_hook=pairs_hook)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeContractError("RUNTIME_E_OUTPUT_JSON") from exc
    if duplicates or not isinstance(value, dict):
        raise RuntimeContractError("RUNTIME_E_OUTPUT_OBJECT")
    return value


MockTransport = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]


def execute_mock_query_frame_request(
    inference_input: Mapping[str, Any],
    mock_transport: MockTransport,
) -> dict[str, Any]:
    request = build_deepseek_json_object_request(inference_input)
    response = mock_transport(request["endpoint"], request["payload"])
    try:
        raw = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeContractError("RUNTIME_E_MOCK_RESPONSE") from exc
    if not isinstance(raw, str):
        raise RuntimeContractError("RUNTIME_E_MOCK_CONTENT")
    frame = _strict_json_object(raw)
    frame_validation = validate_query_frame(frame)
    if not frame_validation["valid"]:
        first = frame_validation["errors"][0]
        raise RuntimeContractError(str(first["error_code"]), str(first["field_path"]))
    compiled = compile_query_frame(frame)
    p2_validation = validate_intent_result(compiled)
    if not p2_validation["valid"]:
        raise RuntimeContractError("RUNTIME_E_P2_VALIDATION")
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
    "runtime_examples", "runtime_prompt_audit",
]
