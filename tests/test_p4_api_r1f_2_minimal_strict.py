"""Offline tests for the R1F.2 minimal strict capability control."""

from __future__ import annotations

import json
import unittest

from run_p4_api_r1f_2_minimal_strict_control import (
    ControlProtocolError,
    ENDPOINT,
    MINIMAL_SCHEMA,
    NoRedirectHandler,
    TOOL_NAME,
    build_payload,
    classify_http_failure,
    parse_success_response,
    sanitize_provider_error,
)


EXPECTED_SCHEMA = {
    "type": "object",
    "properties": {
        "ok": {
            "type": "boolean",
            "description": "Return true.",
        }
    },
    "required": ["ok"],
    "additionalProperties": False,
}


def test_minimal_schema_is_exact_and_has_no_complex_constructs() -> None:
    assert MINIMAL_SCHEMA == EXPECTED_SCHEMA
    text = json.dumps(MINIMAL_SCHEMA, sort_keys=True)
    for forbidden in ("$def", "$ref", "anyOf", "array"):
        assert forbidden not in text


def test_payload_uses_forced_strict_named_function_and_only_control_message() -> None:
    payload = build_payload("deepseek-v4-pro")

    assert ENDPOINT == "https://api.deepseek.com/beta/chat/completions"
    assert payload["model"] == "deepseek-v4-pro"
    assert payload["messages"] == [{
        "role": "user",
        "content": "Call the function and return ok=true.",
    }]
    assert payload["tools"] == [{
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Return the minimal strict capability control result.",
            "strict": True,
            "parameters": EXPECTED_SCHEMA,
        },
    }]
    assert payload["tool_choice"] == {
        "type": "function",
        "function": {"name": TOOL_NAME},
    }


def test_http_redirects_are_not_followed() -> None:
    handler = NoRedirectHandler()
    assert handler.redirect_request(None, None, 302, "Found", {}, "https://other.invalid") is None


def test_generic_http_400_is_unresolved_not_automatic_schema_rejection() -> None:
    raw = json.dumps({
        "error": {
            "message": "Bad request.",
            "code": "bad_request",
            "type": "invalid_request_error",
            "param": None,
        }
    }).encode()

    evidence = sanitize_provider_error(400, raw)

    assert evidence["provider_error_code"] == "bad_request"
    assert evidence["provider_error_type"] == "invalid_request_error"
    assert evidence["provider_error_param"] is None
    assert evidence["mentions"] == {
        "schema": False,
        "strict": False,
        "tool": False,
        "model": False,
    }
    assert "message" not in evidence
    assert classify_http_failure(400, evidence) == (
        "MINIMAL_STRICT_REJECTED_UNRESOLVED",
        "E_MINIMAL_STRICT_REJECTED_UNRESOLVED",
    )


def test_explicit_unsupported_strict_tool_profile_is_classified_as_unavailable() -> None:
    raw = json.dumps({
        "error": {
            "message": "This model does not support strict tool calls.",
            "code": "unsupported_feature",
            "type": "invalid_request_error",
            "param": "tools[0].function.strict",
        }
    }).encode()

    evidence = sanitize_provider_error(400, raw)

    assert evidence["mentions"] == {
        "schema": False,
        "strict": True,
        "tool": True,
        "model": True,
    }
    assert evidence["capability_unavailable_language"] is True
    assert "does not support" not in json.dumps(evidence)
    assert classify_http_failure(400, evidence) == (
        "DEEPSEEK_STRICT_PROFILE_UNAVAILABLE_WITH_EVIDENCE",
        "E_DEEPSEEK_STRICT_PROFILE_UNAVAILABLE",
    )


def test_schema_only_http_400_remains_unresolved() -> None:
    raw = json.dumps({"error": {"message": "Schema is invalid."}}).encode()
    evidence = sanitize_provider_error(400, raw)
    assert evidence["mentions"]["schema"] is True
    assert evidence["capability_unavailable_language"] is False
    assert classify_http_failure(400, evidence)[0] == "MINIMAL_STRICT_REJECTED_UNRESOLVED"


def test_explicit_missing_model_is_profile_unavailable() -> None:
    raw = json.dumps({
        "error": {
            "message": "Model Not Exist",
            "code": "invalid_request_error",
            "type": "invalid_request_error",
            "param": "model",
        }
    }).encode()
    evidence = sanitize_provider_error(400, raw)
    assert evidence["mentions"]["model"] is True
    assert evidence["capability_unavailable_language"] is True
    assert classify_http_failure(400, evidence)[0] == (
        "DEEPSEEK_STRICT_PROFILE_UNAVAILABLE_WITH_EVIDENCE"
    )


def test_success_response_requires_exact_true_argument_and_returns_safe_metadata() -> None:
    response = {
        "model": "deepseek-v4-pro",
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {"tool_calls": [{
                "type": "function",
                "function": {
                    "name": TOOL_NAME,
                    "arguments": '{"ok":true}',
                },
            }]},
        }],
        "usage": {"prompt_tokens": 50, "completion_tokens": 5},
    }

    result = parse_success_response(200, response)

    assert result == {
        "http_status": 200,
        "actual_model": "deepseek-v4-pro",
        "finish_reason": "tool_calls",
        "tool_call_count": 1,
        "tool_name": TOOL_NAME,
        "arguments_exact": True,
        "local_schema_valid": True,
        "prompt_tokens": 50,
        "completion_tokens": 5,
    }


def test_success_response_rejects_non_exact_arguments(arguments: str) -> None:
    response = {
        "model": "deepseek-v4-pro",
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {"tool_calls": [{
                "type": "function",
                "function": {"name": TOOL_NAME, "arguments": arguments},
            }]},
        }],
    }

    with unittest.TestCase().assertRaises(ControlProtocolError):
        parse_success_response(200, response)
_NO_ARGUMENT_TESTS = [
    test_minimal_schema_is_exact_and_has_no_complex_constructs,
    test_payload_uses_forced_strict_named_function_and_only_control_message,
    test_http_redirects_are_not_followed,
    test_generic_http_400_is_unresolved_not_automatic_schema_rejection,
    test_explicit_unsupported_strict_tool_profile_is_classified_as_unavailable,
    test_schema_only_http_400_remains_unresolved,
    test_explicit_missing_model_is_profile_unavailable,
    test_success_response_requires_exact_true_argument_and_returns_safe_metadata,
]


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite(
        unittest.FunctionTestCase(function) for function in _NO_ARGUMENT_TESTS
    )
    for arguments in ('{"ok":false}', '{"ok":true,"extra":1}', '[]'):
        suite.addTest(unittest.FunctionTestCase(
            lambda value=arguments: test_success_response_rejects_non_exact_arguments(value),
            description=f"reject non-exact arguments: {arguments}",
        ))
    return suite
