from __future__ import annotations

import unittest

from review_aware_residual_failure_atlas import (
    classify_miss_with_review,
    choose_next_action,
    normalize_macro,
    summarize_review_aware_misses,
)


class ReviewAwareResidualFailureAtlasTest(unittest.TestCase):
    def test_normalize_macro_unifies_known_aliases(self):
        self.assertEqual(normalize_macro("through_hole"), "through_hole_round")
        self.assertEqual(normalize_macro("blind_hole"), "blind_hole_round")
        self.assertEqual(normalize_macro("stepped_hole"), "stepped_hole_round")
        self.assertEqual(normalize_macro("cylindrical_boss"), "cylinder_boss")
        self.assertEqual(normalize_macro("rectangle_boss"), "rectangular_boss")
        self.assertEqual(normalize_macro("fillet"), "fillet")

    def test_accept_family_match_becomes_auxiliary_not_new_review(self):
        miss = {"sample": "part10", "macro": "through_hole_round", "area": 1912.3562, "miss_cause": "cross_macro_area_compatible"}
        review_rows = [
            {"sample": "part10", "truth_macro": "through_hole_round", "truth_area": "1912.3562", "user_decision": "accept_family_match"}
        ]

        row = classify_miss_with_review(miss, review_rows)

        self.assertEqual(row["review_status"], "semantic_policy_accepted_auxiliary")
        self.assertEqual(row["user_decision"], "accept_family_match")

    def test_keep_strict_becomes_confirmed_strict_failure(self):
        miss = {"sample": "part18", "macro": "slot_o", "area": 3553.0576, "miss_cause": "cross_macro_area_compatible"}
        review_rows = [
            {"sample": "part18", "truth_macro": "slot_o", "truth_area": "3553.0576", "user_decision": "keep_strict"}
        ]

        row = classify_miss_with_review(miss, review_rows)

        self.assertEqual(row["review_status"], "confirmed_strict_cross_macro")

    def test_truth_label_change_matches_by_area_after_truth_refresh(self):
        miss = {"sample": "part12", "macro": "fillet", "area": 125.6637, "miss_cause": "cross_macro_area_compatible"}
        review_rows = [
            {"sample": "part12", "truth_macro": "through_hole_round", "truth_area": "125.6637", "user_decision": "truth_label_should_change"}
        ]

        row = classify_miss_with_review(miss, review_rows)

        self.assertEqual(row["review_status"], "truth_corrected_strict_failure")

    def test_cross_macro_without_review_is_unreviewed(self):
        miss = {"sample": "part99", "macro": "rib", "area": 100.0, "miss_cause": "cross_macro_area_compatible"}

        row = classify_miss_with_review(miss, [])

        self.assertEqual(row["review_status"], "unreviewed_cross_macro")

    def test_non_cross_macro_is_engineering_failure(self):
        miss = {"sample": "part4", "macro": "rib", "area": 458.4, "miss_cause": "no_related_candidate"}

        row = classify_miss_with_review(miss, [])

        self.assertEqual(row["review_status"], "non_cross_macro_engineering_failure")

    def test_summarize_review_aware_misses_counts_status_and_macro(self):
        rows = [
            {"review_status": "confirmed_strict_cross_macro", "macro": "rib", "miss_cause": "cross_macro_area_compatible"},
            {"review_status": "confirmed_strict_cross_macro", "macro": "rib", "miss_cause": "cross_macro_area_compatible"},
            {"review_status": "non_cross_macro_engineering_failure", "macro": "fillet", "miss_cause": "no_same_macro_candidates"},
        ]

        summary = summarize_review_aware_misses(rows)

        self.assertEqual(summary["miss_count"], 3)
        self.assertEqual(summary["by_review_status"]["confirmed_strict_cross_macro"], 2)
        self.assertEqual(summary["by_macro"]["rib"], 2)
        self.assertEqual(summary["by_miss_cause"]["cross_macro_area_compatible"], 2)

    def test_choose_next_action_stops_for_unreviewed_cross_macro(self):
        summary = {"by_review_status": {"unreviewed_cross_macro": 1, "confirmed_strict_cross_macro": 10}}

        recommendation, reason = choose_next_action(summary)

        self.assertEqual(recommendation, "user_review_new_cross_macro_rows")
        self.assertIn("unreviewed_cross_macro", reason)

    def test_choose_next_action_recommends_targeted_repair_after_review(self):
        summary = {
            "by_review_status": {
                "confirmed_strict_cross_macro": 18,
                "truth_corrected_strict_failure": 4,
                "non_cross_macro_engineering_failure": 39,
                "semantic_policy_accepted_auxiliary": 2,
            }
        }

        recommendation, reason = choose_next_action(summary)

        self.assertEqual(recommendation, "write_targeted_proposer_repair_plan")
        self.assertIn("strict residual failures", reason)


if __name__ == "__main__":
    unittest.main()
