from __future__ import annotations

import unittest

import verify_p4_api_r2b_1_delivery as verifier


class R2B1OfflineGateTests(unittest.TestCase):
    def test_non_regression_checks_are_all_required(self) -> None:
        audit = verifier.build_offline_gate(run_regression=False)
        self.assertTrue(audit["prompt_byte_deterministic"])
        self.assertTrue(audit["feature_role_policy_exact"])
        self.assertTrue(audit["all_mock_frame_compiler_p2_valid"])
        self.assertEqual(0, audit["train_val_exact_overlap_count"])
        self.assertEqual(0, audit["decoded_formal_test_lines"])
        self.assertEqual(0, audit["frozen_asset_change_count"])
        self.assertIsNone(audit["regression_return_code"])
        self.assertFalse(audit["gate_pass"])


if __name__ == "__main__":
    unittest.main()
