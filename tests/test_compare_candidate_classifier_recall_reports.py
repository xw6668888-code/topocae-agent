from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from compare_candidate_classifier_recall_reports import metric_row, parse_named_report_arg, write_decision


def report(final_recall, accepted_recall, classifier_wrong):
    return {
        "best_by_final_survival_recall": {
            "proposal_stage": {"recall": 0.6814},
            "classified_stage": {"recall": 0.5},
            "accepted_stage": {"recall": accepted_recall},
            "final_survival_stage": {"recall": final_recall},
            "failure_stage_counts": {
                "no_area_compatible_proposal": 65,
                "proposal_exists_classifier_wrong_or_rejected": classifier_wrong,
                "classifier_accepts_but_final_changes_or_removes": 1,
                "survives_final_prediction": 100,
            },
        }
    }


class CompareCandidateClassifierRecallReportsTest(unittest.TestCase):
    def test_parse_named_report_arg(self):
        name, path = parse_named_report_arg("v5.17=some/path/report.json")
        self.assertEqual(name, "v5.17")
        self.assertEqual(str(path), "some/path/report.json" if "/" in str(path) else "some\\path\\report.json")

    def test_metric_row_reads_best_final_metrics(self):
        row = metric_row("v5.17", report(final_recall=0.55, accepted_recall=0.51, classifier_wrong=28))

        self.assertEqual(row["name"], "v5.17")
        self.assertAlmostEqual(row["accepted_recall"], 0.51)
        self.assertEqual(row["classifier_wrong_or_rejected"], 28)

    def test_decision_promotes_only_if_candidate_beats_v513(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decision.md"
            rows = [
                metric_row("v5.13", report(final_recall=0.5196, accepted_recall=0.4804, classifier_wrong=32)),
                metric_row("v5.17_plain", report(final_recall=0.50, accepted_recall=0.49, classifier_wrong=30)),
                metric_row("v5.17_rebalanced", report(final_recall=0.45, accepted_recall=0.44, classifier_wrong=40)),
            ]

            write_decision(path, rows)

            text = path.read_text(encoding="utf-8")
            self.assertIn("promote_v5.17_plain", text)


if __name__ == "__main__":
    unittest.main()
