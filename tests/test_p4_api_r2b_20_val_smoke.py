from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import run_p4_api_r2b_20_val_smoke as r2b


EXACT_ENV = {
    "TOPOCAE_LLM_PROVIDER_PROFILE": "deepseek_json_object",
    "TOPOCAE_LLM_BASE_URL": "https://api.deepseek.com",
    "TOPOCAE_LLM_MODEL": "deepseek-v4-pro",
}


class R2BSmokeTests(unittest.TestCase):
    def test_preflight_accepts_exact_identity_and_frozen_twenty(self) -> None:
        audit, examples = r2b.build_preflight(EXACT_ENV, api_key_configured=True)
        self.assertTrue(audit["gate_pass"])
        self.assertEqual(
            audit["endpoint"], "https://api.deepseek.com/chat/completions"
        )
        self.assertEqual(audit["sample_count"], 20)
        self.assertTrue(audit["sample_order_preserved"])
        self.assertTrue(audit["all_query_sha256_match"])
        self.assertTrue(audit["all_ground_truth_sha256_match"])
        self.assertEqual(len(examples), 20)

    def test_preflight_rejects_beta_before_request(self) -> None:
        audit, _ = r2b.build_preflight(
            {**EXACT_ENV, "TOPOCAE_LLM_BASE_URL": "https://api.deepseek.com/beta"},
            api_key_configured=True,
        )
        self.assertFalse(audit["gate_pass"])
        self.assertEqual(audit["stable_error_code"], "R2B_PREFLIGHT_E_BASE_URL")

    def test_repair_payload_excludes_previous_output(self) -> None:
        inference_input = {"text": "Apply pressure.", "language": "en-US"}
        summary = {
            "field_path": "/operations/0/magnitude",
            "required": ["value", "unit"],
            "allowed_enum": [],
            "rule": "QF_E_REQUIRED_FIELD",
        }
        request = r2b.build_request(
            inference_input,
            repair=("QF_E_REQUIRED_FIELD", summary),
        )
        self.assertEqual(
            request["endpoint"], "https://api.deepseek.com/chat/completions"
        )
        self.assertEqual(
            request["payload"]["response_format"], {"type": "json_object"}
        )
        self.assertEqual(len(request["payload"]["messages"]), 2)
        repair_value = json.loads(request["payload"]["messages"][1]["content"])
        self.assertEqual(repair_value["input"], inference_input)
        self.assertNotIn(
            "previous", request["payload"]["messages"][1]["content"].lower()
        )
        self.assertNotIn(
            "assistant", {row["role"] for row in request["payload"]["messages"]}
        )

    def test_repair_is_limited_to_structural_phases(self) -> None:
        cases = [
            ("json_parse", "QF_E_JSON_TYPE", True),
            ("frame_validation", "QF_E_REQUIRED_FIELD", True),
            ("frame_compile", "QF_E_CANONICAL_REJECTED", True),
            ("transport", "E_HTTP_500", False),
            ("semantic", "STATUS_MISMATCH", False),
            ("grounding", "E_NO_CANDIDATE", False),
        ]
        for phase, error_code, expected in cases:
            with self.subTest(phase=phase, error_code=error_code):
                self.assertEqual(r2b.repair_allowed(phase, error_code), expected)

    def test_request_ledger_reserves_atomically_and_enforces_caps(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "request_ledger.json"
            ledger = r2b.RequestLedger(path)
            ledger.reserve(
                sample_id="s1", stage="first_round", request_hash="a" * 64
            )
            ledger.reserve(sample_id="s1", stage="repair", request_hash="b" * 64)
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                value["counters"],
                {
                    "first_round_requests": 1,
                    "repair_requests": 1,
                    "backend_requests": 2,
                    "http_retries": 0,
                },
            )
            with self.assertRaisesRegex(
                r2b.R2BError, "R2B_E_FIRST_ROUND_SAMPLE_DUPLICATE"
            ):
                ledger.reserve(
                    sample_id="s1", stage="first_round", request_hash="c" * 64
                )

    def test_semantic_metrics_are_exact_for_oracle_predictions(self) -> None:
        _, examples = r2b.build_preflight(EXACT_ENV, api_key_configured=True)
        predictions = {
            item.example_id: item.oracle_intent_result for item in examples
        }
        metrics = r2b.compute_semantic_metrics(examples, predictions)
        self.assertEqual(metrics["status_accuracy"], 1.0)
        self.assertEqual(metrics["action_macro_f1"], 1.0)
        self.assertEqual(metrics["feature_class_macro_f1"], 1.0)
        self.assertEqual(metrics["selector_accuracy"], 1.0)
        self.assertEqual(metrics["canonical_semantic_exact"], 1.0)
        self.assertEqual(metrics["multi_operation_exact"], 1.0)


if __name__ == "__main__":
    unittest.main()
