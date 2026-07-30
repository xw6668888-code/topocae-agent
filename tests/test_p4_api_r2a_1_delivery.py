from __future__ import annotations

import json
import unittest


REQUIRED_FILES = {
    "final_acceptance.zh-CN.md",
    "acceptance.json",
    "runtime_prompt_contract.zh-CN.md",
    "runtime_prompt_audit.json",
    "mock_request_audit.json",
    "repair_contract_audit.json",
    "smoke_sample_manifest.json",
    "smoke_coverage_audit.json",
    "cost_budget_audit.json",
    "leakage_audit.json",
    "regression_results.json",
    "freeze_audit.json",
    "delivery_manifest.json",
    "CODEX_CONTEXT_P4_API_R2A_1.md",
}


class DeliveryContractTests(unittest.TestCase):
    def test_delivery_payloads_are_complete_and_ready(self):
        import run_p4_api_r2a_1_runtime_prompt as runner

        regression = {
            "status": "PASS", "tests_run": 1, "return_code": 0,
            "interpreter_path": "W:\\python\\envs\\feagpt\\python.exe",
            "python_version": "3.11.14", "test_pattern": "test_p4*.py",
            "stdout_or_stderr_persisted": False,
        }
        payloads = runner.build_delivery_payloads(regression)
        self.assertEqual(REQUIRED_FILES - {"delivery_manifest.json"}, set(payloads))
        acceptance = payloads["acceptance.json"]
        self.assertEqual("READY_FOR_P4_API_R2B_20_VAL_SMOKE_REVIEW", acceptance["status"])
        self.assertTrue(all(acceptance["checks"].values()))

    def test_persisted_json_payloads_contain_no_raw_sample_text(self):
        import run_p4_api_r2a_1_runtime_prompt as runner

        payloads = runner.build_delivery_payloads({"status": "PASS", "tests_run": 1, "return_code": 0})
        encoded = json.dumps(payloads, ensure_ascii=False).lower()
        self.assertNotIn('"query_text"', encoded)
        self.assertNotIn('"ground_truth":', encoded)
        self.assertNotIn('"canonical_id"', encoded)


if __name__ == "__main__":
    unittest.main()
