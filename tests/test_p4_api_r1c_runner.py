from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import run_p4_api_r1c_structured_validation as r1c


class R1CRunnerTests(unittest.TestCase):
    def test_budgeted_transport_enforces_backend_cap(self):
        original_output = r1c.OUTPUT
        original_cap = r1c.BACKEND_REQUEST_CAP
        try:
            with tempfile.TemporaryDirectory() as directory:
                r1c.OUTPUT = Path(directory)
                r1c.BACKEND_REQUEST_CAP = 1
                response = {
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                }
                transport = r1c.BudgetedCappedTransport(transport=lambda *_: response)
                payload = {"max_tokens": 4, "messages": []}
                self.assertIs(transport(None, None, payload, None), response)
                with self.assertRaisesRegex(RuntimeError, "E_BACKEND_REQUEST_CAP"):
                    transport(None, None, payload, None)
                audit = json.loads((r1c.OUTPUT / "request_budget_audit.json").read_text(encoding="utf-8"))
                self.assertEqual(audit["backend_request_count"], 1)
        finally:
            r1c.OUTPUT = original_output
            r1c.BACKEND_REQUEST_CAP = original_cap

    def test_error_sanitization_excludes_messages(self):
        result = r1c.sanitize_errors([{
            "error_code": "E_SCHEMA_VALIDATION",
            "field_path": "/intent",
            "message": "private response detail",
        }])
        self.assertEqual(result, [{
            "error_code": "E_SCHEMA_VALIDATION",
            "field_path": "/intent",
        }])

    def test_outcome_schema_metrics_override_placeholder_validity(self):
        metrics = {
            "record_count": 2,
            "language_cae_ir": {
                "schema_valid_rate_first_round": 1.0,
                "schema_valid_rate_after_retry": 1.0,
            },
            "runtime": {"retry_rate": 0.0},
        }
        rows = [
            {
                "schema_valid_first_round": False,
                "schema_valid_after_retry": False,
                "retry_count": 1,
            },
            {
                "schema_valid_first_round": True,
                "schema_valid_after_retry": True,
                "retry_count": 0,
            },
        ]
        r1c.apply_outcome_schema_metrics(metrics, rows)
        self.assertEqual(metrics["language_cae_ir"]["schema_valid_rate_first_round"], 0.5)
        self.assertEqual(metrics["language_cae_ir"]["schema_valid_rate_after_retry"], 0.5)
        self.assertEqual(metrics["runtime"]["retry_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
