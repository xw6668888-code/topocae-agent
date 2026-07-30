"""Offline DeepSeek strict-tool transport schema and lossless converters.

This module deliberately contains no HTTP client and never reads environment
variables. The canonical P2 schemas remain the authority after decoding.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from jsonschema import Draft202012Validator

from utils.cae_intent import validate_intent_result


ROOT = Path(__file__).resolve().parents[2]
RESULT_SCHEMA_PATH = ROOT / "schemas" / "cae_intent_result.schema.json"
INTENT_SCHEMA_PATH = ROOT / "schemas" / "cae_intent.schema.json"

PROFILE_NAME = "deepseek_strict_tool_beta"
TRANSPORT_SCHEMA_VERSION = "topocae-deepseek-strict-transport/1.0.0"
STRICT_BASE_URL = "https://api.deepseek.com/beta"
TOOL_NAME = "emit_topocae_cae_intent_result"

PROFILE_CONTRACT = {
    "name": PROFILE_NAME,
    "base_url": STRICT_BASE_URL,
    "request_mode": "tools/function/strict",
    "function_strict": True,
    "tool_choice": "forced_named_function",
    "endpoint_fallback": False,
    "provider_fallback": False,
    "model_fallback": False,
    "protocol_fallback": False,
    "real_client_implemented": False,
}

_FORBIDDEN_KEYWORDS = frozenset({
    "oneOf", "allOf", "if", "then", "else", "not",
    "minLength", "maxLength", "minItems", "maxItems", "uniqueItems",
    "const",
})
_ALLOWED_SCHEMA_KEYS = frozenset({
    "type", "properties", "required", "additionalProperties", "items",
    "enum", "anyOf", "$ref", "$def", "pattern", "format",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "multipleOf", "description",
})
_ALLOWED_TYPES = frozenset({
    "object", "string", "number", "integer", "boolean", "array",
})


class TransportContractError(ValueError):
    """A stable coded failure at the canonical/transport boundary."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


@lru_cache(maxsize=1)
def _source_schemas() -> dict[str, dict[str, Any]]:
    return {
        "result": json.loads(RESULT_SCHEMA_PATH.read_text(encoding="utf-8")),
        "intent": json.loads(INTENT_SCHEMA_PATH.read_text(encoding="utf-8")),
    }


def _resolve_source_ref(ref: str, origin: str) -> tuple[dict[str, Any], str]:
    schemas = _source_schemas()
    if ref == "cae_intent.schema.json":
        return schemas["intent"], "intent"
    prefix = "#/$defs/"
    if ref.startswith(prefix):
        name = ref[len(prefix):]
        try:
            return schemas[origin]["$defs"][name], origin
        except KeyError as exc:
            raise TransportContractError("E_SOURCE_REF", ref) from exc
    raise TransportContractError("E_EXTERNAL_REF", ref)


def _rewrite_ref(ref: str, origin: str) -> str:
    if ref == "cae_intent.schema.json":
        return "#/$def/canonical_intent"
    prefix = "#/$defs/"
    if ref.startswith(prefix):
        return f"#/$def/{origin}_{ref[len(prefix):]}"
    raise TransportContractError("E_EXTERNAL_REF", ref)


def _merge_object_schema(base: dict[str, Any], extra: Mapping[str, Any]) -> None:
    if extra.get("type") is not None:
        base["type"] = extra["type"]
    if extra.get("additionalProperties") is False:
        base["additionalProperties"] = False
    if isinstance(extra.get("properties"), Mapping):
        properties = base.setdefault("properties", {})
        for name, schema in extra["properties"].items():
            properties[name] = deepcopy(schema)
    if isinstance(extra.get("required"), list):
        required = list(base.get("required", []))
        for name in extra["required"]:
            if name not in required:
                required.append(name)
        base["required"] = required


def _materialize(schema: Mapping[str, Any], origin: str) -> tuple[dict[str, Any], str]:
    if "$ref" in schema:
        resolved, resolved_origin = _resolve_source_ref(str(schema["$ref"]), origin)
        return _materialize(resolved, resolved_origin)
    result = {
        key: deepcopy(value)
        for key, value in schema.items()
        if key != "allOf"
    }
    for component in schema.get("allOf", []):
        if any(key in component for key in ("if", "then", "else", "not")):
            continue
        materialized, component_origin = _materialize(component, origin)
        if component_origin != origin:
            origin = component_origin
        _merge_object_schema(result, materialized)
    return result, origin


def _presence_schema(value_schema: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "anyOf": [
            {
                "type": "object",
                "properties": {"presence": {"enum": ["ABSENT"]}},
                "required": ["presence"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "presence": {"enum": ["PRESENT"]},
                    "value": deepcopy(value_schema),
                },
                "required": ["presence", "value"],
                "additionalProperties": False,
            },
        ]
    }


def _transform_schema(schema: Mapping[str, Any], origin: str) -> dict[str, Any]:
    if "$ref" in schema:
        return {"$ref": _rewrite_ref(str(schema["$ref"]), origin)}
    if "allOf" in schema:
        schema, origin = _materialize(schema, origin)
    if "oneOf" in schema:
        return {"anyOf": [_transform_schema(item, origin) for item in schema["oneOf"]]}
    if "anyOf" in schema and "type" not in schema and "properties" not in schema:
        return {"anyOf": [_transform_schema(item, origin) for item in schema["anyOf"]]}

    if schema.get("type") == "object" or "properties" in schema:
        source_properties = schema.get("properties", {})
        source_required = set(schema.get("required", []))
        properties: dict[str, Any] = {}
        for name, child in source_properties.items():
            transformed = _transform_schema(child, origin)
            properties[name] = (
                transformed if name in source_required else _presence_schema(transformed)
            )
        return {
            "type": "object",
            "properties": properties,
            "required": list(source_properties),
            "additionalProperties": False,
        }

    result: dict[str, Any] = {}
    if "type" in schema:
        result["type"] = schema["type"]
    if "const" in schema:
        result["enum"] = [deepcopy(schema["const"])]
    elif "enum" in schema:
        result["enum"] = deepcopy(schema["enum"])
    for keyword in (
        "pattern", "format", "minimum", "maximum", "exclusiveMinimum",
        "exclusiveMaximum", "multipleOf",
    ):
        if keyword in schema:
            result[keyword] = deepcopy(schema[keyword])
    if schema.get("type") == "array":
        result["items"] = _transform_schema(schema.get("items", {}), origin)
    return result


@lru_cache(maxsize=1)
def build_transport_schema() -> dict[str, Any]:
    schemas = _source_schemas()
    definitions: dict[str, Any] = {
        "canonical_intent": _transform_schema(schemas["intent"], "intent")
    }
    for origin in ("result", "intent"):
        for name, schema in schemas[origin].get("$defs", {}).items():
            definitions[f"{origin}_{name}"] = _transform_schema(schema, origin)
    result_branches = [
        {"$ref": "#/$def/result_validResult"},
        {"$ref": "#/$def/result_clarificationResult"},
        {"$ref": "#/$def/result_invalidResult"},
    ]
    return {
        "type": "object",
        "properties": {
            "transport_schema_version": {"enum": [TRANSPORT_SCHEMA_VERSION]},
            "result": {"anyOf": result_branches},
        },
        "required": ["transport_schema_version", "result"],
        "additionalProperties": False,
        "$def": definitions,
    }


def _json_pointer(parts: Sequence[Any]) -> str:
    if not parts:
        return "/"
    return "/" + "/".join(
        str(part).replace("~", "~0").replace("/", "~1") for part in parts
    )


def validate_deepseek_strict_subset_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the documented DeepSeek strict subset independently."""

    errors: list[dict[str, str]] = []
    definitions = schema.get("$def", {})

    def error(code: str, path: tuple[Any, ...], detail: str) -> None:
        errors.append({"code": code, "path": _json_pointer(path), "detail": detail})

    def visit(node: Any, path: tuple[Any, ...]) -> None:
        if not isinstance(node, Mapping):
            error("E_SCHEMA_NODE_TYPE", path, "schema node must be an object")
            return
        unknown = sorted(set(node) - _ALLOWED_SCHEMA_KEYS)
        for keyword in unknown:
            code = "E_FORBIDDEN_KEYWORD" if keyword in _FORBIDDEN_KEYWORDS else "E_UNSUPPORTED_KEYWORD"
            error(code, path + (keyword,), keyword)
        schema_type = node.get("type")
        if schema_type is not None and schema_type not in _ALLOWED_TYPES:
            error("E_UNSUPPORTED_TYPE", path + ("type",), str(schema_type))
        if schema_type == "object":
            properties = node.get("properties")
            required = node.get("required")
            if not isinstance(properties, Mapping):
                error("E_OBJECT_PROPERTIES", path, "properties must be an object")
                properties = {}
            if not isinstance(required, list) or set(required) != set(properties):
                error("E_OBJECT_REQUIRED", path, "all and only properties must be required")
            if node.get("additionalProperties") is not False:
                error("E_OBJECT_OPEN", path, "additionalProperties must be false")
            for name, child in properties.items():
                visit(child, path + ("properties", name))
        elif "properties" in node or "required" in node or "additionalProperties" in node:
            error("E_OBJECT_KEYWORD_WITHOUT_OBJECT", path, "object keywords require type=object")
        if "items" in node:
            if schema_type != "array":
                error("E_ITEMS_WITHOUT_ARRAY", path + ("items",), "items requires type=array")
            visit(node["items"], path + ("items",))
        if "anyOf" in node:
            branches = node["anyOf"]
            if not isinstance(branches, list) or not branches:
                error("E_ANYOF_EMPTY", path + ("anyOf",), "anyOf must be non-empty")
            else:
                for index, child in enumerate(branches):
                    visit(child, path + ("anyOf", index))
        if "$ref" in node:
            ref = node["$ref"]
            prefix = "#/$def/"
            if not isinstance(ref, str) or not ref.startswith(prefix):
                error("E_REF_EXTERNAL", path + ("$ref",), str(ref))
            elif ref[len(prefix):] not in definitions:
                error("E_REF_MISSING", path + ("$ref",), ref)
        if "$def" in node:
            if path:
                error("E_DEF_NONROOT", path + ("$def",), "$def is root-only")
            if not isinstance(node["$def"], Mapping):
                error("E_DEF_TYPE", path + ("$def",), "$def must be an object")
            else:
                for name, child in node["$def"].items():
                    visit(child, path + ("$def", name))
        if "enum" in node and (not isinstance(node["enum"], list) or not node["enum"]):
            error("E_ENUM_EMPTY", path + ("enum",), "enum must be non-empty")

    visit(schema, ())
    if schema.get("type") != "object":
        error("E_ROOT_OBJECT", (), "transport root must be object")
    return {
        "valid": not errors,
        "errors": errors,
        "forbidden_keywords": sorted(_FORBIDDEN_KEYWORDS),
        "profile": PROFILE_NAME,
    }


def _singleton_values(schema: Mapping[str, Any]) -> set[Any] | None:
    if "const" in schema:
        return {schema["const"]}
    enum = schema.get("enum")
    if isinstance(enum, list):
        return set(enum)
    return None


def _select_source_branch(value: Any, branches: Sequence[Mapping[str, Any]], origin: str) -> Mapping[str, Any]:
    matches: list[Mapping[str, Any]] = []
    for branch in branches:
        materialized, branch_origin = _materialize(branch, origin)
        if not isinstance(value, Mapping):
            if materialized.get("type") != "object":
                matches.append(branch)
            continue
        discriminated = False
        mismatch = False
        for name, child in materialized.get("properties", {}).items():
            allowed = _singleton_values(child)
            if allowed is not None and name in value:
                discriminated = True
                if value[name] not in allowed:
                    mismatch = True
                    break
        if discriminated and not mismatch:
            matches.append(branch)
    if len(matches) == 1:
        return matches[0]
    raise TransportContractError("E_UNION_BRANCH", f"origin={origin}, matches={len(matches)}")


def _encode(value: Any, schema: Mapping[str, Any], origin: str) -> Any:
    if "$ref" in schema:
        resolved, resolved_origin = _resolve_source_ref(str(schema["$ref"]), origin)
        return _encode(value, resolved, resolved_origin)
    if "allOf" in schema:
        schema, origin = _materialize(schema, origin)
    if "oneOf" in schema:
        branch = _select_source_branch(value, schema["oneOf"], origin)
        return _encode(value, branch, origin)
    if "anyOf" in schema and "type" not in schema and "properties" not in schema:
        branch = _select_source_branch(value, schema["anyOf"], origin)
        return _encode(value, branch, origin)
    if schema.get("type") == "object" or "properties" in schema:
        if not isinstance(value, Mapping):
            raise TransportContractError("E_CANONICAL_OBJECT")
        properties = schema.get("properties", {})
        extras = sorted(set(value) - set(properties))
        if extras:
            raise TransportContractError("E_CANONICAL_EXTRA", ",".join(extras))
        required = set(schema.get("required", []))
        encoded: dict[str, Any] = {}
        for name, child in properties.items():
            if name in required:
                if name not in value:
                    raise TransportContractError("E_CANONICAL_MISSING", name)
                encoded[name] = _encode(value[name], child, origin)
            elif name in value:
                encoded[name] = {
                    "presence": "PRESENT",
                    "value": _encode(value[name], child, origin),
                }
            else:
                encoded[name] = {"presence": "ABSENT"}
        return encoded
    if schema.get("type") == "array":
        if not isinstance(value, list):
            raise TransportContractError("E_CANONICAL_ARRAY")
        return [_encode(item, schema.get("items", {}), origin) for item in value]
    return deepcopy(value)


def _decode(value: Any, schema: Mapping[str, Any], origin: str) -> Any:
    if "$ref" in schema:
        resolved, resolved_origin = _resolve_source_ref(str(schema["$ref"]), origin)
        return _decode(value, resolved, resolved_origin)
    if "allOf" in schema:
        schema, origin = _materialize(schema, origin)
    if "oneOf" in schema:
        branch = _select_source_branch(value, schema["oneOf"], origin)
        return _decode(value, branch, origin)
    if "anyOf" in schema and "type" not in schema and "properties" not in schema:
        branch = _select_source_branch(value, schema["anyOf"], origin)
        return _decode(value, branch, origin)
    if schema.get("type") == "object" or "properties" in schema:
        if not isinstance(value, Mapping):
            raise TransportContractError("E_TRANSPORT_OBJECT")
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        decoded: dict[str, Any] = {}
        for name, child in properties.items():
            if name not in value:
                raise TransportContractError("E_TRANSPORT_MISSING", name)
            if name in required:
                decoded[name] = _decode(value[name], child, origin)
                continue
            wrapper = value[name]
            if not isinstance(wrapper, Mapping):
                raise TransportContractError("E_PRESENCE_OBJECT", name)
            presence = wrapper.get("presence")
            if presence == "ABSENT" and set(wrapper) == {"presence"}:
                continue
            if presence == "PRESENT" and set(wrapper) == {"presence", "value"}:
                decoded[name] = _decode(wrapper["value"], child, origin)
                continue
            raise TransportContractError("E_PRESENCE_FLAG", name)
        return decoded
    if schema.get("type") == "array":
        if not isinstance(value, list):
            raise TransportContractError("E_TRANSPORT_ARRAY")
        return [_decode(item, schema.get("items", {}), origin) for item in value]
    return deepcopy(value)


def _result_source_branch(status: Any) -> Mapping[str, Any]:
    name = {
        "VALID": "validResult",
        "NEEDS_CLARIFICATION": "clarificationResult",
        "INVALID": "invalidResult",
    }.get(status)
    if name is None:
        raise TransportContractError("E_RESULT_STATUS", str(status))
    return _source_schemas()["result"]["$defs"][name]


@lru_cache(maxsize=1)
def _transport_instance_validator() -> Draft202012Validator:
    schema = deepcopy(build_transport_schema())
    definitions = schema.pop("$def")
    schema["$defs"] = definitions

    def rewrite_refs(node: Any) -> None:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$def/"):
                node["$ref"] = "#/$defs/" + ref[len("#/$def/"):]
            for child in node.values():
                rewrite_refs(child)
        elif isinstance(node, list):
            for child in node:
                rewrite_refs(child)

    rewrite_refs(schema)
    return Draft202012Validator(schema)


def validate_transport_instance(value: Any) -> dict[str, Any]:
    errors = sorted(_transport_instance_validator().iter_errors(value), key=lambda item: list(item.path))
    return {
        "valid": not errors,
        "errors": [
            {"path": _json_pointer(tuple(error.absolute_path)), "message": error.message}
            for error in errors
        ],
    }


def canonical_to_transport(value: Mapping[str, Any], *, validate_canonical: bool = True) -> dict[str, Any]:
    if validate_canonical:
        validation = validate_intent_result(value)
        if not validation["valid"]:
            raise TransportContractError("E_CANONICAL_VALIDATION")
    branch = _result_source_branch(value.get("status"))
    return {
        "transport_schema_version": TRANSPORT_SCHEMA_VERSION,
        "result": _encode(value, branch, "result"),
    }


def transport_to_canonical(
    value: Mapping[str, Any],
    *,
    validate_transport: bool = True,
    validate_canonical: bool = True,
) -> dict[str, Any]:
    if validate_transport:
        validation = validate_transport_instance(value)
        if not validation["valid"]:
            raise TransportContractError("E_TRANSPORT_VALIDATION")
    if value.get("transport_schema_version") != TRANSPORT_SCHEMA_VERSION:
        raise TransportContractError("E_TRANSPORT_VERSION")
    result = value.get("result")
    if not isinstance(result, Mapping):
        raise TransportContractError("E_TRANSPORT_RESULT")
    branch = _result_source_branch(result.get("status"))
    decoded = _decode(result, branch, "result")
    if validate_canonical:
        validation = validate_intent_result(decoded)
        if not validation["valid"]:
            raise TransportContractError("E_CANONICAL_VALIDATION_AFTER_DECODE")
    return decoded


def _normalized_base_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", ""))


def build_mock_tool_payload(
    *,
    base_url: str,
    model: str,
    messages: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Build a payload only; this function cannot perform network I/O."""

    if _normalized_base_url(base_url) != STRICT_BASE_URL:
        raise TransportContractError("E_STRICT_BETA_ENDPOINT", base_url)
    if not model.strip():
        raise TransportContractError("E_MODEL_REQUIRED")
    return {
        "model": model,
        "messages": [dict(message) for message in messages],
        "tools": [{
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "description": "Emit a TopoCAE CAE intent result transport object.",
                "strict": True,
                "parameters": build_transport_schema(),
            },
        }],
        "tool_choice": {
            "type": "function",
            "function": {"name": TOOL_NAME},
        },
    }


def _strict_json_object(text: str) -> Mapping[str, Any]:
    duplicates: list[str] = []

    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                duplicates.append(str(key))
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=pairs_hook)
    except json.JSONDecodeError as exc:
        raise TransportContractError("E_TOOL_ARGUMENTS_JSON", exc.msg) from exc
    if duplicates:
        raise TransportContractError("E_TOOL_ARGUMENTS_DUPLICATE_KEY", ",".join(sorted(set(duplicates))))
    if not isinstance(value, Mapping):
        raise TransportContractError("E_TOOL_ARGUMENTS_TYPE")
    return value


def parse_mock_tool_call_response(response: Mapping[str, Any]) -> dict[str, Any]:
    """Parse only a forced named tool call; content fallback is forbidden."""

    try:
        choices = response["choices"]
        message = choices[0]["message"]
        calls = message["tool_calls"]
    except (KeyError, IndexError, TypeError) as exc:
        raise TransportContractError("E_TOOL_RESPONSE_SHAPE") from exc
    if not isinstance(calls, list) or len(calls) != 1:
        raise TransportContractError("E_TOOL_CALL_COUNT")
    call = calls[0]
    try:
        if call["type"] != "function" or call["function"]["name"] != TOOL_NAME:
            raise TransportContractError("E_TOOL_NAME")
        arguments = call["function"]["arguments"]
    except (KeyError, TypeError) as exc:
        raise TransportContractError("E_TOOL_RESPONSE_SHAPE") from exc
    if not isinstance(arguments, str):
        raise TransportContractError("E_TOOL_ARGUMENTS_TYPE")
    transport = _strict_json_object(arguments)
    return transport_to_canonical(transport)


def recompute_r1c_combined_schema_metrics(
    train_metrics: Mapping[str, Any], val_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    """Recompute combined schema metrics with record-count weighting."""

    rows = [train_metrics, val_metrics]
    total = sum(int(row["record_count"]) for row in rows)
    if total <= 0:
        raise TransportContractError("E_R1C_METRIC_COUNT")

    def weighted(section: str, key: str) -> float:
        return sum(
            int(row["record_count"]) * float(row[section][key]) for row in rows
        ) / total

    retry_count = sum(int(row["language_cae_ir"]["retry_count"]) for row in rows)
    return {
        "record_count": total,
        "schema_valid_rate_first_round": weighted(
            "language_cae_ir", "schema_valid_rate_first_round"
        ),
        "schema_valid_rate_after_retry": weighted(
            "language_cae_ir", "schema_valid_rate_after_retry"
        ),
        "retry_count": retry_count,
        "retry_rate": retry_count / total,
    }


__all__ = [
    "PROFILE_CONTRACT", "PROFILE_NAME", "STRICT_BASE_URL", "TOOL_NAME",
    "TRANSPORT_SCHEMA_VERSION", "TransportContractError",
    "build_mock_tool_payload", "build_transport_schema",
    "canonical_to_transport", "parse_mock_tool_call_response",
    "recompute_r1c_combined_schema_metrics", "transport_to_canonical",
    "validate_deepseek_strict_subset_schema", "validate_transport_instance",
]
