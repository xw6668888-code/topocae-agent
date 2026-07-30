from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import run_p4_api_r1a_smoke as smoke
from utils.cae_language_grounding.llm_adapter import LLMRuntimeConfig


class R1AAuditPersistenceTests(unittest.TestCase):
    def test_counter_and_outcome_are_persisted_without_secrets(self):
        original_output = smoke.OUTPUT
        original_transport = smoke._default_transport
        try:
            with tempfile.TemporaryDirectory() as directory:
                smoke.OUTPUT = Path(directory)
                smoke._default_transport = lambda *_: {
                    "model": "deepseek-v4-pro",
                    "choices": [{
                        "finish_reason": "stop",
                        "message": {"content": '{"answer":"ok"}'},
                    }],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                }
                transport = smoke.CappedTransport(
                    initial_backend_requests=1
                )
                transport.stage = "structured"
                config = replace(
                    LLMRuntimeConfig.from_env(),
                    api_key=None,
                    cache_dir=None,
                )
                client = smoke.AuditedClient(config, transport)
                client.stage = "structured"
                client.example_id = "offline-fixture"
                schema = {
                    "type": "object",
                    "required": ["answer"],
                    "properties": {"answer": {"type": "string"}},
                }
                outcome = client.infer_json(
                    inference_input={"text": "offline", "language": "en-US"},
                    system_prompt="Return JSON.",
                    schema_name="audit_fixture",
                    schema=schema,
                    validator=lambda value: {
                        "valid": value == {"answer": "ok"},
                        "errors": [],
                    },
                )
                counter = json.loads(
                    (smoke.OUTPUT / "backend_request_counter.json")
                    .read_text(encoding="utf-8")
                )
                rows = (
                    (smoke.OUTPUT / "outcome_audit.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
                self.assertTrue(outcome.schema_valid_after_retry)
                self.assertEqual(counter["backend_request_count"], 2)
                self.assertEqual(len(rows), 1)
                self.assertNotIn("api_key", rows[0].lower())
                self.assertNotIn("authorization", rows[0].lower())
        finally:
            smoke.OUTPUT = original_output
            smoke._default_transport = original_transport


if __name__ == "__main__":
    unittest.main()
