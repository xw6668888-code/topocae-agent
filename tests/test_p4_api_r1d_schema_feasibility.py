from __future__ import annotations

import unittest

import run_p4_api_r1d_strict_schema_feasibility as r1d


class R1DStrictSchemaFeasibilityTests(unittest.TestCase):
    def test_frozen_schema_is_rejected_by_both_strict_subsets(self):
        rows = [r1d.inspect_schema(path) for path in r1d.SCHEMA_PATHS]
        inventory = r1d.effective_inventory(rows)
        result = r1d.provider_compatibility(inventory)
        self.assertEqual(result["compatible_candidate_count"], 0)
        self.assertFalse(result["real_api_smoke_authorized_by_gate"])
        self.assertIn(
            "E_ROOT_MUST_BE_OBJECT",
            result["evaluated_candidates"]["openai_json_schema"]["blockers"],
        )
        self.assertIn(
            "E_MIN_ITEMS_UNSUPPORTED",
            result["evaluated_candidates"]["deepseek_strict_tool_beta"]["blockers"],
        )

    def test_schema_hashes_remain_frozen(self):
        actual = {
            str(path.relative_to(r1d.ROOT)).replace("\\", "/"): r1d.sha256(path)
            for path in r1d.SCHEMA_PATHS
        }
        self.assertEqual(actual, r1d.EXPECTED_SCHEMA_HASHES)


if __name__ == "__main__":
    unittest.main()
