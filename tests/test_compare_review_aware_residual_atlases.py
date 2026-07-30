from __future__ import annotations

import unittest

from compare_review_aware_residual_atlases import compare_summaries, delta_counts


class CompareReviewAwareResidualAtlasesTest(unittest.TestCase):
    def test_delta_counts_subtracts_new_from_old_by_key(self):
        old = {"rib": 23, "fillet": 18, "slot_u": 1}
        new = {"rib": 20, "fillet": 18, "chamfer": 3}

        delta = delta_counts(old, new)

        self.assertEqual(delta["rib"], -3)
        self.assertEqual(delta["fillet"], 0)
        self.assertEqual(delta["slot_u"], -1)
        self.assertEqual(delta["chamfer"], 3)

    def test_compare_summaries_reports_macro_and_status_deltas(self):
        old = {"summary": {"by_macro": {"rib": 23}, "by_review_status": {"non_cross_macro_engineering_failure": 41}}}
        new = {"summary": {"by_macro": {"rib": 20}, "by_review_status": {"non_cross_macro_engineering_failure": 38}}}

        comparison = compare_summaries(old, new)

        self.assertEqual(comparison["macro_delta"]["rib"], -3)
        self.assertEqual(comparison["review_status_delta"]["non_cross_macro_engineering_failure"], -3)


if __name__ == "__main__":
    unittest.main()
