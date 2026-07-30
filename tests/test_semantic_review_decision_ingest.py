from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from semantic_review_decision_ingest import (
    ALLOWED_USER_DECISIONS,
    build_review_rows,
    classify_decision_outcome,
    decide_next_action,
    read_csv_rows,
    summarize_decisions,
    validate_user_decision,
    write_review_template,
)


class SemanticReviewDecisionIngestTest(unittest.TestCase):
    def test_validate_user_decision_accepts_allowed_and_blank(self):
        self.assertEqual(validate_user_decision("accept_family_match"), "accept_family_match")
        self.assertEqual(validate_user_decision(" "), "")
        self.assertIn("keep_strict", ALLOWED_USER_DECISIONS)

    def test_validate_user_decision_rejects_unknown_value(self):
        with self.assertRaisesRegex(ValueError, "Invalid user_decision"):
            validate_user_decision("maybe")

    def test_classify_decision_outcome_maps_values(self):
        self.assertEqual(classify_decision_outcome("accept_family_match"), "policy_recovered")
        self.assertEqual(classify_decision_outcome("truth_label_should_change"), "truth_correction_needed")
        self.assertEqual(classify_decision_outcome("keep_strict"), "strict_failure")
        self.assertEqual(classify_decision_outcome("model_prediction_wrong"), "model_or_proposer_error")
        self.assertEqual(classify_decision_outcome("needs_face_level_review"), "unresolved_manual_review")
        self.assertEqual(classify_decision_outcome(""), "pending_user_review")

    def test_build_review_rows_adds_row_id_decision_and_prompt(self):
        source_rows = [
            {
                "sample": "part3",
                "truth_macro": "fillet",
                "candidate_macro": "blind_hole_round",
                "pair": "blind_hole_round|fillet",
                "review_category": "transition_vs_nontransition_conflict",
                "truth_area": "235.62",
                "candidate_area": "235.62",
                "relative_error": "0.0",
                "truth_face_count": "1",
                "candidate_face_count": "1",
                "candidate_source": "same_instance",
                "candidate_action": "rejected_label",
                "candidate_confidence": "0.64",
                "candidate_faces": "22",
            }
        ]

        rows = build_review_rows(source_rows)

        self.assertEqual(rows[0]["row_id"], "R001")
        self.assertEqual(rows[0]["user_decision"], "")
        self.assertIn("fillet/chamfer", rows[0]["review_prompt"])

    def test_write_review_template_round_trips_user_decision_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.csv"
            rows = build_review_rows([
                {
                    "sample": "part11",
                    "truth_macro": "slot_rect",
                    "candidate_macro": "pocket_rect",
                    "pair": "pocket_rect|slot_rect",
                    "review_category": "rectangular_recess_slot_vs_pocket",
                    "truth_area": "72.36",
                    "candidate_area": "72.36",
                    "relative_error": "0.0",
                    "truth_face_count": "3",
                    "candidate_face_count": "3",
                    "candidate_source": "hybrid_merged",
                    "candidate_action": "rejected_label",
                    "candidate_confidence": "0.78",
                    "candidate_faces": "28 29 30",
                }
            ])

            write_review_template(path, rows)
            loaded = read_csv_rows(path)

            self.assertEqual(loaded[0]["row_id"], "R001")
            self.assertIn("user_decision", loaded[0])
            self.assertIn("review_prompt", loaded[0])

    def test_summarize_decisions_counts_outcomes_and_categories(self):
        rows = [
            {"review_category": "transition_vs_nontransition_conflict", "pair": "a|b", "user_decision": "keep_strict"},
            {"review_category": "hole_subfeature_vs_composite_hole", "pair": "c|d", "user_decision": "accept_family_match"},
            {"review_category": "hole_subfeature_vs_composite_hole", "pair": "c|d", "user_decision": ""},
        ]

        summary = summarize_decisions(rows)

        self.assertEqual(summary["row_count"], 3)
        self.assertEqual(summary["by_outcome"]["strict_failure"], 1)
        self.assertEqual(summary["by_outcome"]["policy_recovered"], 1)
        self.assertEqual(summary["by_outcome"]["pending_user_review"], 1)
        self.assertEqual(summary["by_category"]["hole_subfeature_vs_composite_hole"], 2)

    def test_decide_next_action_waits_for_pending_review_first(self):
        summary = {"by_outcome": {"pending_user_review": 1, "policy_recovered": 5}}
        recommendation, _ = decide_next_action(summary)
        self.assertEqual(recommendation, "wait_for_user_semantic_review")

    def test_decide_next_action_prefers_truth_correction_when_largest(self):
        summary = {"by_outcome": {"truth_correction_needed": 4, "policy_recovered": 2}}
        recommendation, _ = decide_next_action(summary)
        self.assertEqual(recommendation, "correct_area_truth_table_then_rerun_metrics")


if __name__ == "__main__":
    unittest.main()
