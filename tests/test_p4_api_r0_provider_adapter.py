from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from utils.cae_language_grounding.llm_adapter import (
    LLMProtocolError,
    LLMRuntimeConfig,
    LLMTransportError,
    OpenAICompatibleJSONClient,
)
from utils.cae_language_grounding.reliable_runner import (
    FormalTestLockedError,
    assert_split_authorized,
    deterministic_window,
    redact_secrets,
)


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
}


def _validator(value):
    valid = isinstance(value, dict) and isinstance(value.get("answer"), str)
    return {"valid": valid, "errors": [] if valid else [{"code": "bad"}]}


def _response(content='{"answer":"ok"}', **overrides):
    value = {
        "id": "mock-id",
        "model": "served-model-revision",
        "system_fingerprint": "fp_mock",
        "choices": [
            {"finish_reason": "stop", "message": {"content": content}}
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }
    value.update(overrides)
    return value


def _config(**overrides):
    base = LLMRuntimeConfig(
        base_url="https://provider.invalid",
        model="requested-model",
        api_key="unit-test-secret",
        seed=17,
        timeout_seconds=2,
        revision="rev-1",
        dtype="bf16",
        quantization="none",
        backend="mock",
        gpu="none",
        context_length=4096,
        constrained_decoding=True,
    )
    return replace(base, **overrides)


def _infer(client, *, schema_name="unit_structured"):
    return client.infer_json(
        inference_input={"text": "offline", "language": "en-US"},
        system_prompt="Return JSON.",
        schema_name=schema_name,
        schema=SCHEMA,
        validator=_validator,
    )


class ProviderContractTests(unittest.TestCase):
    def test_openai_profile_uses_json_schema_and_complete_endpoint(self):
        calls = []

        def transport(url, headers, payload, timeout):
            calls.append((url, headers, payload, timeout))
            return _response()

        config = _config(
            base_url="https://provider.invalid/v1/chat/completions",
            provider_profile="openai_json_schema",
            max_tokens=123,
        )
        outcome = _infer(OpenAICompatibleJSONClient(config, transport=transport))
        self.assertTrue(outcome.schema_valid_first_round)
        self.assertEqual(calls[0][0], config.base_url)
        self.assertEqual(calls[0][2]["response_format"]["type"], "json_schema")
        self.assertEqual(calls[0][2]["seed"], 17)
        self.assertEqual(calls[0][2]["temperature"], 0)
        self.assertEqual(calls[0][2]["max_tokens"], 123)
        self.assertNotIn("thinking", calls[0][2])

    def test_deepseek_profile_uses_json_object_and_disables_thinking(self):
        calls = []

        def transport(url, headers, payload, timeout):
            calls.append((url, headers, payload, timeout))
            return _response()

        config = _config(
            base_url="https://provider.invalid/v1",
            provider_profile="deepseek_json_object",
        )
        _infer(OpenAICompatibleJSONClient(config, transport=transport))
        self.assertEqual(
            calls[0][0], "https://provider.invalid/v1/chat/completions"
        )
        self.assertEqual(calls[0][2]["response_format"], {"type": "json_object"})
        self.assertEqual(calls[0][2]["thinking"], {"type": "disabled"})
        self.assertNotIn("seed", calls[0][2])

    def test_success_records_sanitized_runtime_metadata(self):
        outcome = _infer(
            OpenAICompatibleJSONClient(_config(), transport=lambda *_: _response())
        )
        self.assertEqual(outcome.actual_model, "served-model-revision")
        self.assertEqual(outcome.system_fingerprint, "fp_mock")
        self.assertEqual(outcome.input_tokens, 3)
        self.assertEqual(outcome.output_tokens, 2)
        self.assertGreaterEqual(outcome.latency_ms, 0)
        self.assertNotIn("unit-test-secret", json.dumps(outcome.public_metadata()))

    def test_invalid_json_gets_exactly_one_schema_repair(self):
        calls = []

        def transport(*_):
            calls.append(None)
            return _response("not-json" if len(calls) == 1 else '{"answer":"ok"}')

        outcome = _infer(OpenAICompatibleJSONClient(_config(), transport=transport))
        self.assertEqual(len(calls), 2)
        self.assertEqual(outcome.retry_count, 1)
        self.assertTrue(outcome.schema_valid_after_retry)

    def test_response_failures_have_stable_codes(self):
        cases = {
            "E_RUNTIME_EMPTY_CONTENT": _response(""),
            "E_RUNTIME_TRUNCATED": _response("{}", choices=[{
                "finish_reason": "length", "message": {"content": "{}"}
            }]),
            "E_RUNTIME_USAGE_MISSING": _response(usage=None),
            "E_RUNTIME_RESPONSE_SHAPE": {"choices": []},
        }
        for code, response in cases.items():
            with self.subTest(code=code):
                outcome = _infer(
                    OpenAICompatibleJSONClient(
                        _config(), transport=lambda *_, response=response: response
                    )
                )
                self.assertEqual(outcome.failure_code, code)
                self.assertEqual(outcome.backend_request_count, 2)

    def test_retries_429_500_and_timeout_but_not_other_4xx(self):
        for error in (
            LLMTransportError("E_HTTP_429", status_code=429, retryable=True),
            LLMTransportError("E_HTTP_500", status_code=500, retryable=True),
            LLMTransportError("E_HTTP_TIMEOUT", retryable=True),
        ):
            calls = []

            def recover(*_):
                calls.append(None)
                if len(calls) == 1:
                    raise error
                return _response()

            outcome = _infer(
                OpenAICompatibleJSONClient(
                    _config(max_http_retries=1), transport=recover, sleep=lambda _: None
                )
            )
            self.assertTrue(outcome.schema_valid_after_retry)
            self.assertEqual(outcome.http_retry_count, 1)
            self.assertEqual(len(calls), 2)

        calls = []

        def reject(*_):
            calls.append(None)
            raise LLMTransportError("E_HTTP_400", status_code=400, retryable=False)

        with self.assertRaises(LLMTransportError):
            _infer(
                OpenAICompatibleJSONClient(
                    _config(max_http_retries=3), transport=reject, sleep=lambda _: None
                )
            )
        self.assertEqual(len(calls), 1)


class ReliableRunnerTests(unittest.TestCase):
    def test_cache_is_atomic_idempotent_isolated_and_recovers_from_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []

            def transport(*_):
                calls.append(None)
                return _response()

            client = OpenAICompatibleJSONClient(
                _config(cache_dir=directory), transport=transport
            )
            first = _infer(client, schema_name="structured")
            second = _infer(client, schema_name="structured")
            direct = _infer(client, schema_name="direct")
            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertFalse(direct.cache_hit)
            self.assertEqual(len(calls), 2)
            self.assertNotEqual(first.request_hash, direct.request_hash)

            cache_path = Path(directory) / "structured" / f"{first.request_hash}.json"
            cache_path.write_text("corrupt", encoding="utf-8")
            repaired = _infer(client, schema_name="structured")
            self.assertFalse(repaired.cache_hit)
            self.assertEqual(len(calls), 3)
            json.loads(cache_path.read_text(encoding="utf-8"))

    def test_deterministic_window_and_formal_lock(self):
        values = list(range(10))
        self.assertEqual(
            deterministic_window(
                values, offset=1, limit=2, shard_index=1, shard_count=3
            ),
            [4, 7],
        )
        assert_split_authorized("val")
        with self.assertRaises(FormalTestLockedError):
            assert_split_authorized("test")

    def test_recursive_secret_redaction(self):
        value = {
            "Authorization": "Bearer abc",
            "api_key": "abc",
            "nested": {"endpoint": "https://user:pass@example.invalid/v1"},
        }
        sanitized = redact_secrets(value)
        encoded = json.dumps(sanitized)
        self.assertNotIn("abc", encoded)
        self.assertNotIn("user:pass", encoded)
        self.assertIn("REDACTED", encoded)


if __name__ == "__main__":
    unittest.main()
