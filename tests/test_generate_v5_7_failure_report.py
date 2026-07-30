import json
import tempfile
import unittest
from pathlib import Path

from generate_v5_7_failure_report import build_v5_7_report


class GenerateV57FailureReportTest(unittest.TestCase):
    def test_report_summarizes_area_errors_and_rib_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction_dir = root / "predictions"
            prediction_dir.mkdir()
            area_review_path = root / "area_review.json"
            area_review_path.write_text(
                json.dumps(
                    {
                        "sample_count": 1,
                        "truth_count": 2,
                        "prediction_count": 2,
                        "matched_truth_count": 1,
                        "matched_prediction_count": 1,
                        "recall": 0.5,
                        "precision": 0.5,
                        "samples": [
                            {
                                "sample_id": "part1",
                                "missed_truth": [
                                    {"macro": "rib", "face_count": 3, "area": 100.0}
                                ],
                                "false_positives": [
                                    {"macro": "fillet", "face_count": 1, "area": 50.0, "face_indices": [7]}
                                ],
                                "matches": [],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (prediction_dir / "part1.json").write_text(
                json.dumps(
                    {
                        "sample_id": "part1",
                        "rib_boundary_classifier_postprocess": {
                            "decision_count": 2,
                            "records": [{"action": "pruned_to_stock"}],
                            "decision_records": [
                                {"action": "pruned_to_stock", "stage": "prune"},
                                {"action": "skipped_low_confidence", "stage": "expand"},
                            ],
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            report = build_v5_7_report(
                area_review_json=area_review_path,
                prediction_path=prediction_dir,
                feature_objects_path=None,
                recursive=False,
                dataset_name="toy",
            )

        self.assertEqual(report["overall"]["recall"], 0.5)
        self.assertEqual(report["residual_error_summary"]["missed_macro_counts"]["rib"], 1)
        self.assertEqual(report["residual_error_summary"]["false_positive_macro_counts"]["fillet"], 1)
        self.assertEqual(report["rib_boundary_decision_summary"]["action_counts"]["pruned_to_stock"], 1)
        self.assertEqual(report["rib_boundary_decision_summary"]["action_counts"]["skipped_low_confidence"], 1)
        self.assertTrue(report["fix_recommendations"])


if __name__ == "__main__":
    unittest.main()
