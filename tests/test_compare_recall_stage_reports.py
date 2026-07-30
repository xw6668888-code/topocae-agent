from __future__ import annotations

import unittest

from compare_recall_stage_reports import compare_stage_reports, extract_stage_metrics


class CompareRecallStageReportsTest(unittest.TestCase):
    def test_extract_stage_metrics_reads_best_final_survival_result(self):
        report = {
            "best_by_final_survival_recall": {
                "truth_count": 10,
                "proposal_stage": {"recall": 0.8, "matched_truth_count": 8},
                "classified_stage": {"recall": 0.7, "matched_truth_count": 7},
                "accepted_stage": {"recall": 0.6, "matched_truth_count": 6},
                "final_survival_stage": {"recall": 0.5, "matched_truth_count": 5},
            }
        }

        metrics = extract_stage_metrics(report)

        self.assertEqual(metrics["truth_count"], 10)
        self.assertEqual(metrics["proposal_recall"], 0.8)
        self.assertEqual(metrics["classified_recall"], 0.7)
        self.assertEqual(metrics["accepted_recall"], 0.6)
        self.assertEqual(metrics["final_survival_recall"], 0.5)

    def test_compare_stage_reports_computes_deltas(self):
        old = {
            "best_by_final_survival_recall": {
                "truth_count": 10,
                "proposal_stage": {"recall": 0.8, "matched_truth_count": 8},
                "classified_stage": {"recall": 0.6, "matched_truth_count": 6},
                "accepted_stage": {"recall": 0.5, "matched_truth_count": 5},
                "final_survival_stage": {"recall": 0.4, "matched_truth_count": 4},
            }
        }
        new = {
            "best_by_final_survival_recall": {
                "truth_count": 10,
                "proposal_stage": {"recall": 0.9, "matched_truth_count": 9},
                "classified_stage": {"recall": 0.7, "matched_truth_count": 7},
                "accepted_stage": {"recall": 0.6, "matched_truth_count": 6},
                "final_survival_stage": {"recall": 0.5, "matched_truth_count": 5},
            }
        }

        comparison = compare_stage_reports(old, new)

        self.assertAlmostEqual(comparison["delta"]["proposal_recall"], 0.1)
        self.assertAlmostEqual(comparison["delta"]["final_survival_recall"], 0.1)
        self.assertEqual(comparison["delta"]["final_survival_matched"], 1)


if __name__ == "__main__":
    unittest.main()
