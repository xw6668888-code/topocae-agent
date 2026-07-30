from __future__ import annotations

import json
import unittest


DESIRED_ACTIONS = {
    "fixed_support", "displacement", "pressure", "surface_traction",
    "material_assignment", "global_mesh_size", "local_mesh_size",
}
DESIRED_QUANTIFIERS = {"all", "exactly_one", "count", "largest", "smallest", "nearest"}
ALLOWED_SAMPLE_FIELDS = {
    "sample_id", "language", "difficulty", "status", "coverage_tags",
    "query_sha256", "ground_truth_sha256",
}


class SmokeManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import run_p4_api_r2a_1_runtime_prompt as runner

        cls.runner = runner
        cls.manifest = runner.build_smoke_manifest()
        cls.coverage = runner.build_smoke_coverage_audit(cls.manifest)

    def test_selection_has_exact_count_language_quotas_and_unique_ids(self):
        samples = self.manifest["samples"]
        self.assertEqual(20, len(samples))
        languages = {language: sum(row["language"] == language for row in samples) for language in ("en-US", "zh-CN")}
        self.assertEqual({"en-US": 10, "zh-CN": 10}, languages)
        self.assertEqual(20, len({row["sample_id"] for row in samples}))
        self.assertEqual(0, self.coverage["duplicate_canonical_id_count"])

    def test_set_cover_meets_required_semantic_coverage(self):
        self.assertEqual({"L1", "L2", "L3", "L4", "L5"}, set(self.coverage["difficulty_counts"]))
        self.assertEqual({"VALID", "NEEDS_CLARIFICATION", "INVALID"}, set(self.coverage["status_counts"]))
        self.assertGreater(self.coverage["operation_scope_counts"].get("single", 0), 0)
        self.assertGreater(self.coverage["operation_scope_counts"].get("multi", 0), 0)
        self.assertEqual(DESIRED_ACTIONS, set(self.coverage["action_counts"]))
        self.assertTrue(DESIRED_QUANTIFIERS.issubset(self.coverage["quantifier_counts"]))
        self.assertTrue(self.coverage["coverage_gate_pass"])

    def test_manifest_is_redacted_and_hash_only_for_payloads(self):
        for row in self.manifest["samples"]:
            with self.subTest(sample_id=row["sample_id"]):
                self.assertEqual(ALLOWED_SAMPLE_FIELDS, set(row))
                self.assertRegex(row["query_sha256"], r"^[0-9a-f]{64}$")
                self.assertRegex(row["ground_truth_sha256"], r"^[0-9a-f]{64}$")
                encoded = json.dumps(row, ensure_ascii=False).lower()
                for forbidden in ("query_text", "intent_result", "canonical_payload", "ground_truth\""):
                    self.assertNotIn(forbidden, encoded)

    def test_selection_is_deterministic_and_not_first_twenty(self):
        first = self.runner.canonical_json_bytes(self.manifest)
        second = self.runner.canonical_json_bytes(self.runner.build_smoke_manifest())
        self.assertEqual(first, second)
        self.assertTrue(self.coverage["selection_reproduction_byte_exact"])
        self.assertFalse(self.coverage["equals_first_20_val_renderings"])
        self.assertEqual("deterministic_weighted_greedy_set_cover_v1", self.manifest["selection_algorithm"])


class OfflineAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import run_p4_api_r2a_1_runtime_prompt as runner

        cls.runner = runner
        cls.audits = runner.build_all_audits()

    def test_runtime_repair_and_mock_audits_pass(self):
        runtime = self.audits["runtime_prompt"]
        repair = self.audits["repair_contract"]
        mock = self.audits["mock_request"]
        self.assertTrue(runtime["gate_pass"])
        self.assertTrue(repair["gate_pass"])
        self.assertEqual(1, repair["max_repair_attempts"])
        self.assertTrue(mock["gate_pass"])
        self.assertEqual("https://api.deepseek.com/chat/completions", mock["endpoint"])
        self.assertEqual({"type": "json_object"}, mock["response_format"])
        self.assertEqual(0, mock["network_request_count"])
        self.assertFalse(mock["provider_schema_field_present"])

    def test_cost_budget_is_conservative_and_hard_capped(self):
        cost = self.audits["cost_budget"]
        self.assertGreater(cost["system_prompt_utf8_bytes"], 0)
        self.assertGreater(cost["system_prompt_token_estimate"], 0)
        self.assertGreaterEqual(cost["system_prompt_token_upper_bound"], cost["system_prompt_token_estimate"])
        self.assertGreater(cost["single_request_max_payload_bytes"], cost["system_prompt_utf8_bytes"])
        self.assertGreater(cost["max_output_tokens"], 0)
        self.assertEqual(20, cost["first_round_request_cap"])
        self.assertEqual(20, cost["repair_request_cap"])
        self.assertEqual(40, cost["total_request_hard_cap"])
        self.assertGreater(cost["first_round_cost_upper_cny"], 0)
        self.assertGreater(cost["repair_cost_upper_cny"], 0)
        self.assertEqual(0, cost["actual_requests"])

    def test_leakage_formal_and_freeze_gates_pass(self):
        leakage = self.audits["leakage"]
        freeze = self.audits["freeze"]
        self.assertEqual(0, leakage["formal_test_access_count"])
        self.assertEqual(0, leakage["decoded_test_lines"])
        self.assertEqual(0, leakage["example_train_val_text_overlap_count"])
        self.assertFalse(leakage["query_payload_persisted"])
        self.assertFalse(leakage["ground_truth_payload_persisted"])
        self.assertEqual(leakage["formal_test_ledger_before"], leakage["formal_test_ledger_after"])
        self.assertEqual(0, freeze["change_count"])
        self.assertTrue(freeze["r2a_unchanged"])
        self.assertTrue(freeze["p1_unchanged"])
        self.assertTrue(freeze["p2_unchanged"])
        self.assertTrue(freeze["p3_unchanged"])
        self.assertTrue(freeze["historical_p4_unchanged"])


if __name__ == "__main__":
    unittest.main()
