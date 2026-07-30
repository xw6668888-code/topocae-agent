from __future__ import annotations

import json
import unittest

from utils.cae_intent import validate_intent_result
from utils.cae_query_frame import compile_query_frame, load_schema, validate_query_frame


INPUT = {"text": "Apply a 2 MPa pressure to all circular pocket floors.", "language": "en-US"}


class RuntimePromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from utils import cae_query_frame_runtime as runtime

        cls.runtime = runtime

    def test_prompt_is_byte_deterministic_and_embeds_frozen_schema(self):
        first = self.runtime.build_query_frame_runtime_prompt()
        second = self.runtime.build_query_frame_runtime_prompt()
        self.assertIsInstance(first, str)
        self.assertEqual(first.encode("utf-8"), second.encode("utf-8"))
        self.assertIn("JSON", first)
        self.assertIn(self.runtime.QUERY_FRAME_SCHEMA_SHA256, first)
        compact_schema = json.dumps(load_schema(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self.assertIn(compact_schema, first)

    def test_prompt_inventory_is_schema_derived_and_complete(self):
        audit = self.runtime.runtime_prompt_audit()
        schema = load_schema()
        self.assertEqual(set(schema["properties"]), set(audit["root_fields"]))
        self.assertEqual(
            set(schema["$defs"]["operation"]["properties"]),
            set(audit["operation_fields"]),
        )
        self.assertEqual(
            set(schema["$defs"]["operation"]["properties"]["action"]["enum"]),
            set(audit["action_enums"]),
        )
        self.assertTrue(audit["required_optional_inventory_complete"])
        self.assertTrue(audit["mutual_exclusion_rules_schema_consistent"])

    def test_four_handcrafted_examples_cover_three_states_and_multi_operation(self):
        examples = self.runtime.runtime_examples()
        self.assertEqual(4, len(examples))
        self.assertEqual(
            {"valid", "needs_clarification", "invalid"},
            {item["frame"]["status"] for item in examples},
        )
        self.assertTrue(any(len(item["frame"].get("operations", [])) > 1 for item in examples))
        for item in examples:
            with self.subTest(example=item["name"]):
                self.assertEqual("handcrafted_r2a_1", item["provenance"])
                self.assertEqual({"text", "language"}, set(item["input"]))
                self.assertTrue(validate_query_frame(item["frame"])["valid"])
                compiled = compile_query_frame(item["frame"])
                self.assertTrue(validate_intent_result(compiled)["valid"])

    def test_runtime_prompt_contains_no_forbidden_evidence_terms(self):
        prompt = self.runtime.build_query_frame_runtime_prompt().lower()
        for forbidden in (
            "face id", "instance id", "source identity", "ground truth", "graph", "answer",
        ):
            self.assertNotIn(forbidden, prompt)

    def test_schema_hash_matches_frozen_r2a_delivery(self):
        self.assertEqual(
            "5f1913121e750493520d818b1d377d90add70a72e90c6612f6ec47cd7179da43",
            self.runtime.QUERY_FRAME_SCHEMA_SHA256,
        )


class RepairPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from utils import cae_query_frame_runtime as runtime

        cls.runtime = runtime

    def test_repair_prompt_contains_only_bounded_contract_fields(self):
        summary = {
            "field_path": "/operations/0/mesh_size",
            "required": ["mesh_size"],
            "allowed_enum": [],
            "rule": "local_mesh_size requires mesh_size",
        }
        prompt = self.runtime.build_query_frame_repair_prompt(
            INPUT, "QF_E_REQUIRED_FIELD", summary, attempt=1
        )
        value = json.loads(prompt)
        self.assertEqual(
            {"input", "error_code", "schema_summary", "instruction", "repair_attempt"},
            set(value),
        )
        self.assertEqual(INPUT, value["input"])
        self.assertEqual(1, value["repair_attempt"])
        self.assertNotIn("canonical", prompt.lower())
        self.assertNotIn("ground_truth", prompt.lower())
        self.assertNotIn("correct", prompt.lower())

    def test_repair_rejects_second_attempt_extra_input_and_unknown_code(self):
        summary = {"field_path": "/status", "required": [], "allowed_enum": ["valid"], "rule": "enum"}
        cases = [
            (INPUT, "QF_E_INVALID_ENUM", summary, 2, "RUNTIME_E_REPAIR_LIMIT"),
            ({**INPUT, "extra": "x"}, "QF_E_INVALID_ENUM", summary, 1, "RUNTIME_E_INPUT_FIELDS"),
            (INPUT, "SOME_ERROR", summary, 1, "RUNTIME_E_ERROR_CODE"),
        ]
        for inference_input, code, item, attempt, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaises(self.runtime.RuntimeContractError) as caught:
                    self.runtime.build_query_frame_repair_prompt(
                        inference_input, code, item, attempt=attempt
                    )
                self.assertEqual(expected, caught.exception.code)

    def test_repair_is_deterministic_and_summary_is_size_bounded(self):
        summary = {"field_path": "/x", "required": [], "allowed_enum": [], "rule": "x"}
        first = self.runtime.build_query_frame_repair_prompt(INPUT, "QF_E_CONFLICT", summary)
        second = self.runtime.build_query_frame_repair_prompt(dict(INPUT), "QF_E_CONFLICT", dict(summary))
        self.assertEqual(first.encode("utf-8"), second.encode("utf-8"))
        with self.assertRaises(self.runtime.RuntimeContractError) as caught:
            self.runtime.build_query_frame_repair_prompt(
                INPUT, "QF_E_CONFLICT", {**summary, "rule": "x" * 3000}
            )
        self.assertEqual("RUNTIME_E_SCHEMA_SUMMARY_SIZE", caught.exception.code)


class MockRequestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from utils import cae_query_frame_runtime as runtime

        cls.runtime = runtime

    def test_request_uses_normal_json_object_contract(self):
        request = self.runtime.build_deepseek_json_object_request(INPUT)
        self.assertEqual("https://api.deepseek.com/chat/completions", request["endpoint"])
        self.assertNotIn("/beta", request["endpoint"])
        self.assertEqual("deepseek_json_object", request["profile"])
        payload = request["payload"]
        self.assertEqual({"type": "json_object"}, payload["response_format"])
        self.assertEqual(self.runtime.build_query_frame_runtime_prompt(), payload["messages"][0]["content"])
        self.assertEqual(INPUT, json.loads(payload["messages"][1]["content"]))
        self.assertNotIn("strict", json.dumps(payload).lower())
        self.assertNotIn("json_schema", json.dumps(payload).lower())
        self.assertEqual({"text", "language"}, set(json.loads(payload["messages"][1]["content"])))

    def test_mock_output_runs_frame_compiler_and_frozen_validator(self):
        frame = self.runtime.runtime_examples()[0]["frame"]
        calls = []

        def mock_transport(endpoint, payload):
            calls.append((endpoint, payload))
            return {
                "model": "deepseek-v4-pro",
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(frame)}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

        outcome = self.runtime.execute_mock_query_frame_request(INPUT, mock_transport)
        self.assertEqual(1, len(calls))
        self.assertEqual(0, outcome["network_request_count"])
        self.assertEqual(1, outcome["mock_request_count"])
        self.assertTrue(outcome["frame_valid"])
        self.assertTrue(outcome["compiler_valid"])
        self.assertTrue(outcome["p2_valid"])
        self.assertEqual(frame, outcome["frame"])


if __name__ == "__main__":
    unittest.main()
