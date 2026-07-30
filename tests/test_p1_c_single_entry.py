from __future__ import annotations

import unittest

import run_paper_v1_p1_c_pipeline as entry


class SingleEntryTests(unittest.TestCase):
    def test_status_reports_completed_holdout_once(self) -> None:
        status = entry.status()
        self.assertTrue(status["preholdout_frozen"])
        self.assertTrue(status["real_dev_complete"])
        self.assertEqual(status["holdout_v3_inference_pass_count"], 1)
        self.assertFalse(status["p2_entered"])

    def test_single_entry_refuses_second_holdout(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "second run forbidden"):
            entry.holdout_once()


if __name__ == "__main__":
    unittest.main()
