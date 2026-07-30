from __future__ import annotations

import unittest

from write_v5_21_decision import choose_next_recommendation


class WriteV521DecisionTest(unittest.TestCase):
    def test_recommends_proposer_repair_for_no_same_macro_dominance(self):
        atlas_summary = {
            "by_cause": {
                "no_same_macro_candidates": 20,
                "cross_macro_area_compatible": 5,
            },
            "by_macro": {
                "rib": 12,
                "fillet": 8,
            },
        }
        recommendation, reason = choose_next_recommendation(atlas_summary)
        self.assertEqual(recommendation, "targeted_proposer_repair")
        self.assertIn("no_same_macro_candidates", reason)

    def test_recommends_semantic_review_if_cross_macro_still_dominates(self):
        atlas_summary = {
            "by_cause": {
                "cross_macro_area_compatible": 20,
                "no_same_macro_candidates": 5,
            },
            "by_macro": {},
        }
        recommendation, reason = choose_next_recommendation(atlas_summary)
        self.assertEqual(recommendation, "inspect_remaining_cross_macro_semantics")
        self.assertIn("cross_macro_area_compatible", reason)


if __name__ == "__main__":
    unittest.main()
