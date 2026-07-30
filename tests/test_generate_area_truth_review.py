import unittest

from generate_area_truth_review import evaluate_sample


class GenerateAreaTruthReviewTest(unittest.TestCase):
    def test_connected_slot_o_prediction_can_match_multiple_truth_slots(self):
        feature_doc = {
            "feature_objects": [
                {
                    "macro_feature": "slot_o",
                    "face_count": 2,
                    "face_indices": [33, 34],
                    "fingerprint": {"area_total": 139.1217},
                }
            ]
        }
        truth_entries = [
            {"macro": "slot_o", "face_count": 1, "area": 69.56085},
            {"macro": "slot_o", "face_count": 1, "area": 69.56085},
        ]

        report = evaluate_sample("part6", feature_doc, truth_entries, abs_tol=5.0, rel_tol=0.01)

        self.assertEqual(report["truth_count"], 2)
        self.assertEqual(report["prediction_count"], 1)
        self.assertEqual(report["matched_truth_count"], 2)
        self.assertEqual(report["matched_prediction_count"], 1)
        self.assertEqual(report["matched_count"], 2)
        self.assertEqual(report["missed_truth"], [])
        self.assertEqual(report["false_positives"], [])
        self.assertEqual(report["matches"][0]["match_type"], "merged_truth")
        self.assertEqual(report["matches"][0]["truth"]["merged_truth_count"], 2)
        self.assertAlmostEqual(report["recall"], 1.0)
        self.assertAlmostEqual(report["precision"], 1.0)

    def test_non_mergeable_macro_still_requires_one_to_one_match(self):
        feature_doc = {
            "feature_objects": [
                {
                    "macro_feature": "fillet",
                    "face_count": 2,
                    "face_indices": [10, 11],
                    "fingerprint": {"area_total": 100.0},
                }
            ]
        }
        truth_entries = [
            {"macro": "fillet", "face_count": 1, "area": 50.0},
            {"macro": "fillet", "face_count": 1, "area": 50.0},
        ]

        report = evaluate_sample("part_x", feature_doc, truth_entries, abs_tol=5.0, rel_tol=0.01)

        self.assertEqual(report["matched_truth_count"], 0)
        self.assertEqual(report["matched_prediction_count"], 0)
        self.assertEqual(len(report["missed_truth"]), 2)
        self.assertEqual(len(report["false_positives"]), 1)


if __name__ == "__main__":
    unittest.main()
