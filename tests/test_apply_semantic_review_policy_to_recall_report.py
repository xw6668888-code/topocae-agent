from __future__ import annotations

import unittest

from apply_semantic_review_policy_to_recall_report import (
    accepted_policy_keys,
    apply_semantic_policy,
    normalize_key_macro,
)


class ApplySemanticReviewPolicyToRecallReportTest(unittest.TestCase):
    def test_normalize_key_macro_unifies_hole_aliases(self):
        self.assertEqual(normalize_key_macro("through_hole"), "through_hole_round")
        self.assertEqual(normalize_key_macro("blind_hole"), "blind_hole_round")
        self.assertEqual(normalize_key_macro("cylindrical_boss"), "cylinder_boss")

    def test_accepted_policy_keys_uses_only_accept_family_match(self):
        rows = [
            {"sample": "part10", "truth_macro": "through_hole_round", "truth_area": "1912.3562", "user_decision": "accept_family_match"},
            {"sample": "part12", "truth_macro": "through_hole_round", "truth_area": "125.6637", "user_decision": "truth_label_should_change"},
            {"sample": "part18", "truth_macro": "slot_o", "truth_area": "3553.06", "user_decision": "keep_strict"},
        ]

        keys = accepted_policy_keys(rows)

        self.assertEqual(keys, {("part10", "through_hole_round", 1912.3562)})

    def test_apply_semantic_policy_recovers_only_matching_strict_failures(self):
        report = {
            "best_by_final_survival_recall": {
                "truth_count": 4,
                "final_survival_stage": {"matched_truth_count": 1, "recall": 0.25},
                "samples": [
                    {
                        "sample": "part10",
                        "truth_failures": [
                            {"macro": "through_hole_round", "area": 1912.3562, "face_count": 1},
                            {"macro": "fillet", "area": 88.0, "face_count": 1},
                        ],
                    },
                    {
                        "sample": "part11",
                        "truth_failures": [
                            {"macro": "through_hole_round", "area": 558.5587, "face_count": 1},
                        ],
                    },
                ],
            }
        }
        review_rows = [
            {"sample": "part10", "truth_macro": "through_hole_round", "truth_area": "1912.3562", "user_decision": "accept_family_match"},
            {"sample": "part11", "truth_macro": "through_hole_round", "truth_area": "558.5587", "user_decision": "accept_family_match"},
            {"sample": "part12", "truth_macro": "through_hole_round", "truth_area": "125.6637", "user_decision": "truth_label_should_change"},
        ]

        adjusted = apply_semantic_policy(report, review_rows, area_tolerance=0.02)

        self.assertEqual(adjusted["strict_matched_truth_count"], 1)
        self.assertEqual(adjusted["policy_recovered_count"], 2)
        self.assertEqual(adjusted["semantic_policy_matched_truth_count"], 3)
        self.assertEqual(adjusted["truth_count"], 4)
        self.assertAlmostEqual(adjusted["semantic_policy_recall"], 0.75)


if __name__ == "__main__":
    unittest.main()
