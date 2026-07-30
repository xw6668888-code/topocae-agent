from __future__ import annotations

import io
import json
from pathlib import Path
import unittest
import urllib.error

from utils.cae_language_grounding.deepseek_strict_real_adapter import (
    DeepSeekStrictToolBetaAdapter,
    PROFILE_NAME,
    STRICT_CHAT_ENDPOINT,
    SingleEntryHandshakeCache,
    StrictHandshakeError,
)
from utils.cae_language_grounding.deepseek_strict_transport import (
    TOOL_NAME,
    canonical_to_transport,
)


ROOT = Path(__file__).resolve().parents[1]


def canonical_fixture() -> dict:
    schema = json.loads(
        (ROOT / "schemas" / "cae_intent_result.schema.json").read_text(encoding="utf-8")
    )
    return schema["examples"][0]


def successful_response() -> dict:
    arguments = json.dumps(canonical_to_transport(canonical_fixture()), ensure_ascii=False)
    return {
        "model": "deepseek-v4-pro",
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "tool_calls": [{
                    "type": "function",
                    "function": {"name": TOOL_NAME, "arguments": arguments},
                }]
            },
        }],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 300,
            "prompt_cache_hit_tokens": 0,
        },
    }


def adapter_with(transport):
    return DeepSeekStrictToolBetaAdapter(
        api_key="x", model="deepseek-v4-pro", transport=transport
    )


class R1FStrictHandshakeTests(unittest.TestCase):
    def test_exact_beta_endpoint_strict_tool_and_forced_choice(self):
        calls = []

        def transport(endpoint, headers, payload, timeout):
            calls.append((endpoint, headers, payload, timeout))
            return 200, successful_response()

        adapter = adapter_with(transport)
        outcome = adapter.handshake([{"role": "user", "content": "mock"}])
        self.assertEqual(1, len(calls))
        endpoint, headers, payload, _ = calls[0]
        self.assertEqual(STRICT_CHAT_ENDPOINT, endpoint)
        self.assertIn("Authorization", headers)
        function = payload["tools"][0]["function"]
        self.assertTrue(function["strict"])
        self.assertEqual(TOOL_NAME, function["name"])
        self.assertEqual(TOOL_NAME, payload["tool_choice"]["function"]["name"])
        self.assertNotIn("response_format", payload)
        self.assertEqual(PROFILE_NAME, adapter.profile)
        self.assertTrue(outcome.canonical_valid)

    def test_success_replays_from_memory_without_second_backend_request(self):
        calls = 0

        def transport(endpoint, headers, payload, timeout):
            nonlocal calls
            calls += 1
            return 200, successful_response()

        adapter = adapter_with(transport)
        cache = SingleEntryHandshakeCache()
        first = cache.execute("fixed-key", adapter, [{"role": "user", "content": "mock"}])
        replay = cache.execute("fixed-key", adapter, [{"role": "user", "content": "mock"}])
        self.assertEqual(1, calls)
        self.assertFalse(first.cache_hit)
        self.assertTrue(replay.cache_hit)
        self.assertEqual(0, replay.backend_request_count)
        self.assertEqual(first.canonical, replay.canonical)

    def test_http_schema_rejection_is_sanitized_and_not_retried(self):
        calls = 0
        body = json.dumps({
            "error": {"message": "Invalid schema at $defs/result_validResult"}
        }).encode("utf-8")

        def transport(endpoint, headers, payload, timeout):
            nonlocal calls
            calls += 1
            raise urllib.error.HTTPError(
                endpoint, 400, "Bad Request", {}, io.BytesIO(body)
            )

        adapter = adapter_with(transport)
        with self.assertRaises(StrictHandshakeError) as raised:
            adapter.handshake([{"role": "user", "content": "mock"}])
        self.assertEqual(1, calls)
        self.assertEqual("STRICT_SCHEMA_REJECTED", raised.exception.category)
        self.assertEqual("$defs/result_validResult", raised.exception.schema_parameter_path)
        metadata = raised.exception.public_metadata()
        self.assertNotIn("message", metadata)
        self.assertFalse(metadata["raw_error_persisted"])

    def test_identity_mismatch_fails_before_transport(self):
        with self.assertRaisesRegex(ValueError, "E_STRICT_PROFILE_IDENTITY"):
            DeepSeekStrictToolBetaAdapter(
                api_key="x", model="deepseek-v4-pro", profile="other"
            )
        with self.assertRaisesRegex(ValueError, "E_STRICT_BASE_URL_IDENTITY"):
            DeepSeekStrictToolBetaAdapter(
                api_key="x", model="deepseek-v4-pro",
                base_url="https://api.deepseek.com",
            )
        with self.assertRaisesRegex(ValueError, "E_STRICT_MODEL_IDENTITY"):
            DeepSeekStrictToolBetaAdapter(
                api_key="x", model="deepseek-v4-flash"
            )

    def test_public_metadata_excludes_canonical_and_raw_arguments(self):
        metadata = adapter_with(
            lambda *args: (200, successful_response())
        ).handshake([]).public_metadata()
        self.assertNotIn("canonical", metadata)
        self.assertNotIn("arguments", metadata)
        self.assertFalse(metadata["raw_response_persisted"])
        self.assertFalse(metadata["tool_arguments_persisted"])


if __name__ == "__main__":
    unittest.main()
