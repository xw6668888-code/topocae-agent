from __future__ import annotations

import unittest


class CandidateClassifierErrorAtlasTest(unittest.TestCase):
    def test_apply_action_policy_distinguishes_reject_low_confidence_overlap_and_accept(self):
        from candidate_classifier_error_atlas import apply_action_policy

        records = [
            {"predicted_label": "through_hole_round", "confidence": 0.90, "face_indices": [1]},
            {"predicted_label": "reject", "confidence": 0.95, "face_indices": [2]},
            {"predicted_label": "fillet", "confidence": 0.20, "face_indices": [3]},
            {"predicted_label": "rib", "confidence": 0.85, "face_indices": [1, 4]},
        ]

        classified = apply_action_policy(records, min_confidence=0.50)
        actions = [record["action"] for record in classified]

        self.assertEqual(actions, ["accepted_remap", "rejected_label", "skipped_low_confidence", "skipped_overlap"])

    def test_compare_candidate_decisions_marks_old_accept_new_reject_for_matched_truth(self):
        from candidate_classifier_error_atlas import compare_candidate_decision

        candidate = {
            "sample": "part1",
            "proposal_macro": "through_hole_round",
            "proposal_sources": ["fallback_rule"],
            "face_indices": [7],
            "area": 200.0,
            "matched_truth_macros": ["through_hole_round"],
        }
        old_record = {
            "predicted_label": "through_hole_round",
            "confidence": 0.80,
            "action": "accepted_remap",
            "face_indices": [7],
        }
        new_record = {
            "predicted_label": "reject",
            "confidence": 0.91,
            "action": "rejected_label",
            "face_indices": [7],
        }

        comparison = compare_candidate_decision(candidate, old_record, new_record)

        self.assertEqual(comparison["error_category"], "old_correct_accept_new_reject")
        self.assertTrue(comparison["old_correct"])
        self.assertFalse(comparison["new_correct"])
        self.assertEqual(comparison["matched_truth_macros"], ["through_hole_round"])

    def test_audit_training_records_counts_label_rates_by_source_and_macro(self):
        from candidate_classifier_error_atlas import audit_training_records

        records = [
            {"source": "fallback_rule", "proposal_macro": "through_hole_round", "label": "reject"},
            {"source": "fallback_rule", "proposal_macro": "through_hole_round", "label": "through_hole_round"},
            {"source": "fallback_rule", "proposal_macro": "through_hole_round", "label": "reject"},
            {"source": "topology_rule", "proposal_macro": "fillet", "label": "fillet"},
        ]

        audit = audit_training_records(records)

        fb = audit["by_source_macro"]["fallback_rule"]["through_hole_round"]
        self.assertEqual(fb["record_count"], 3)
        self.assertEqual(fb["reject_count"], 2)
        self.assertEqual(fb["positive_count"], 1)
        self.assertAlmostEqual(fb["positive_rate"], 1.0 / 3.0)

        fillet = audit["by_source_macro"]["topology_rule"]["fillet"]
        self.assertEqual(fillet["record_count"], 1)
        self.assertEqual(fillet["positive_count"], 1)
        self.assertEqual(fillet["reject_count"], 0)


if __name__ == "__main__":
    unittest.main()
