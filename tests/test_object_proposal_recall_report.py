"""Tests for v5.11 object proposal recall diagnostics."""

from __future__ import annotations

import unittest
from unittest.mock import patch


class TestPureAreaMatching(unittest.TestCase):
    """Task 1: pure area matching tests — must fail before implementation."""

    def test_area_truth_matching_separates_proposal_classification_acceptance_and_final_survival(self):
        """A matching candidate at all stages → survives_final_prediction."""
        from object_proposal_recall_report import (
            candidate_matches_truth,
            classify_truth_failure_stage,
        )

        truth = {"macro": "rib", "area": 100.0, "face_count": 3}
        candidate = {
            "source": "assembled_instance_candidate",
            "proposal_macro": "rib",
            "classified_macro": "rib",
            "action": "accepted_remap",
            "face_indices": [1, 2, 3],
            "area": 100.5,
        }
        final_prediction = {
            "faces": [
                {"idx": 1, "macro_feature": "rib"},
                {"idx": 2, "macro_feature": "rib"},
                {"idx": 3, "macro_feature": "rib"},
            ]
        }

        # All four stages must match.
        for stage in ("proposal", "classified", "accepted", "final_survival"):
            self.assertTrue(
                candidate_matches_truth(
                    candidate,
                    truth,
                    stage,
                    absolute_area_tolerance=10.0,
                    relative_area_tolerance=0.02,
                    final_prediction=final_prediction,
                ),
                f"candidate should match truth at {stage} stage",
            )

        failure_stage = classify_truth_failure_stage(
            truth,
            [candidate],
            absolute_area_tolerance=10.0,
            relative_area_tolerance=0.02,
            final_prediction=final_prediction,
        )
        self.assertEqual(failure_stage, "survives_final_prediction")

    def test_failure_stage_distinguishes_rejected_classifier_from_missing_proposal(self):
        """Rejected classifier vs missing proposal get different failure stages."""
        from object_proposal_recall_report import (
            classify_truth_failure_stage,
        )

        truth_a = {"macro": "rib", "area": 100.0, "face_count": 3}
        truth_b = {"macro": "slot_u", "area": 200.0, "face_count": 2}

        # Candidate for truth A: proposal macro is rib but classified as reject.
        candidate_a = {
            "source": "assembled_instance_candidate",
            "proposal_macro": "rib",
            "classified_macro": "reject",
            "action": "rejected_label",
            "face_indices": [1, 2, 3],
            "area": 100.5,
        }
        # No candidate for truth B.
        candidates = [candidate_a]

        failure_a = classify_truth_failure_stage(
            truth_a, candidates, absolute_area_tolerance=10.0, relative_area_tolerance=0.02
        )
        failure_b = classify_truth_failure_stage(
            truth_b, candidates, absolute_area_tolerance=10.0, relative_area_tolerance=0.02
        )

        self.assertEqual(failure_a, "proposal_exists_classifier_wrong_or_rejected")
        self.assertEqual(failure_b, "no_area_compatible_proposal")


class TestCandidateAssemblyReplay(unittest.TestCase):
    """Task 3: candidate assembly replay tests."""

    def test_build_candidate_summaries_aligns_classifier_records_by_face_set_and_macro(self):
        """Candidates matched by sorted face tuple + proposal macro; face order doesn't matter."""
        from object_proposal_recall_report import (
            build_candidate_summaries,
            proposal_key,
        )

        proposals = [
            {
                "source": "same_instance_seed_component",
                "proposal_macro": "rib",
                "face_indices": [1, 2, 3],
                "area": 100.0,
            },
            {
                "source": "same_instance_seed_component",
                "proposal_macro": "slot_u",
                "face_indices": [4, 5],
                "area": 50.0,
            },
        ]

        classifier_records = [
            {
                "source": "same_instance_seed_component",
                "proposal_macro": "rib",
                "predicted_label": "rib",
                "confidence": 0.9,
                "face_indices": [3, 2, 1],  # different order
                "action": "accepted_remap",
            },
            {
                "source": "same_instance_seed_component",
                "proposal_macro": "slot_u",
                "predicted_label": "reject",
                "confidence": 0.7,
                "face_indices": [4, 5],
                "action": "rejected_label",
            },
        ]

        # Verify proposal_key extracts sorted faces + macro
        self.assertEqual(
            proposal_key(proposals[0]),
            ("rib", (1, 2, 3)),
        )
        self.assertEqual(
            proposal_key(classifier_records[0]),
            ("rib", (1, 2, 3)),
        )

        candidates = build_candidate_summaries(proposals, classifier_records)

        # Should produce 2 candidates
        self.assertEqual(len(candidates), 2)

        # Find by macro
        by_macro = {c["proposal_macro"]: c for c in candidates}

        rib = by_macro["rib"]
        self.assertEqual(rib["classified_macro"], "rib")
        self.assertEqual(rib["action"], "accepted_remap")

        slot = by_macro["slot_u"]
        self.assertEqual(slot["classified_macro"], "reject")
        self.assertEqual(slot["action"], "rejected_label")


class TestDatasetLevelResultBuilder(unittest.TestCase):
    """Task 5: dataset-level result builder tests."""

    @patch("object_proposal_recall_report.replay_candidates_for_sample")
    def test_build_threshold_result_reports_sample_and_macro_counts(self, mock_replay):
        """Build two samples: one with matching candidate, one without."""
        from object_proposal_recall_report import build_threshold_result

        # part1: one matching accepted candidate for rib truth
        part1_candidate = {
            "source": "assembled_instance_candidate",
            "proposal_macro": "rib",
            "classified_macro": "rib",
            "action": "accepted_remap",
            "face_indices": [1, 2, 3],
            "face_count": 3,
            "area": 100.5,
        }
        # part2: no candidates
        part2_candidates: list = []

        def replay_side_effect(graph, raw_prediction, classifier_bundle, strong_threshold, weak_threshold, min_confidence, proposal_mode="same_instance"):
            # Use the graph to determine which sample
            sample = graph.get("_sample", "part1")
            if sample == "part2":
                return part2_candidates
            return [part1_candidate]

        mock_replay.side_effect = replay_side_effect

        # Build minimal graphs and predictions for 2 samples
        graph_by_sample = {
            "part1": {"_sample": "part1", "nodes": [], "edges": []},
            "part2": {"_sample": "part2", "nodes": [], "edges": []},
        }
        raw_prediction_by_sample = {
            "part1": {},
            "part2": {},
        }
        truth_by_sample = {
            "part1": [
                {"sample": "part1", "macro": "rib", "area": 100.0, "face_count": 3, "truth_index": 0},
            ],
            "part2": [
                {"sample": "part2", "macro": "slot_u", "area": 200.0, "face_count": 2, "truth_index": 0},
            ],
        }
        final_prediction_by_sample = {
            "part1": {"faces": [
                {"idx": 1, "macro_feature": "rib"},
                {"idx": 2, "macro_feature": "rib"},
                {"idx": 3, "macro_feature": "rib"},
            ]},
        }

        result = build_threshold_result(
            graph_by_sample=graph_by_sample,
            raw_prediction_by_sample=raw_prediction_by_sample,
            truth_by_sample=truth_by_sample,
            final_prediction_by_sample=final_prediction_by_sample,
            classifier_bundle=None,
            strong_threshold=0.55,
            weak_threshold=0.35,
            min_confidence=0.5,
            absolute_area_tolerance=10.0,
            relative_area_tolerance=0.02,
        )

        self.assertEqual(result["truth_count"], 2)

        ps = result["proposal_stage"]
        self.assertEqual(ps["matched_truth_count"], 1)
        self.assertEqual(ps["truth_by_macro"], {"rib": 1, "slot_u": 1})
        self.assertEqual(ps["matched_by_macro"], {"rib": 1})
        self.assertEqual(ps["missed_by_macro"], {"slot_u": 1})

        self.assertEqual(result["failure_stage_counts"]["survives_final_prediction"], 1)
        self.assertEqual(result["failure_stage_counts"]["no_area_compatible_proposal"], 1)

    def test_threshold_pair_accounting_records_skipped_invalid_pairs(self):
        from object_proposal_recall_report import build_threshold_pairs

        pairs, skipped = build_threshold_pairs(
            strong_thresholds=[0.3, 0.4],
            weak_thresholds=[0.2, 0.3, 0.5],
        )

        self.assertEqual(
            pairs,
            [
                {"strong_threshold": 0.3, "weak_threshold": 0.2},
                {"strong_threshold": 0.4, "weak_threshold": 0.2},
                {"strong_threshold": 0.4, "weak_threshold": 0.3},
            ],
        )
        self.assertEqual(
            skipped,
            [
                {"strong_threshold": 0.3, "weak_threshold": 0.3, "reason": "weak_threshold_must_be_less_than_strong_threshold"},
                {"strong_threshold": 0.3, "weak_threshold": 0.5, "reason": "weak_threshold_must_be_less_than_strong_threshold"},
                {"strong_threshold": 0.4, "weak_threshold": 0.5, "reason": "weak_threshold_must_be_less_than_strong_threshold"},
            ],
        )

    @patch("object_proposal_recall_report.replay_candidates_for_sample")
    def test_build_threshold_result_reports_proposal_source_coverage(self, mock_replay):
        from object_proposal_recall_report import build_threshold_result

        mock_replay.return_value = [
            {
                "source": "hybrid_merged",
                "proposal_sources": ["same_instance", "raw_macro_component"],
                "proposal_macro": "rib",
                "classified_macro": "rib",
                "action": "accepted_remap",
                "face_indices": [1, 2, 3],
                "face_count": 3,
                "area": 100.0,
            }
        ]

        result = build_threshold_result(
            graph_by_sample={"part1": {"nodes": [], "edges": []}},
            raw_prediction_by_sample={"part1": {"faces": []}},
            truth_by_sample={"part1": [{"sample": "part1", "macro": "rib", "area": 100.0, "face_count": 3, "truth_index": 0}]},
            final_prediction_by_sample={"part1": {"faces": [{"idx": 1, "macro_feature": "rib"}, {"idx": 2, "macro_feature": "rib"}, {"idx": 3, "macro_feature": "rib"}]}},
            classifier_bundle=None,
            strong_threshold=0.3,
            weak_threshold=0.2,
            min_confidence=0.5,
            absolute_area_tolerance=10.0,
            relative_area_tolerance=0.02,
            proposal_mode="hybrid",
        )

        self.assertEqual(result["proposal_candidate_count_by_source"]["same_instance"], 1)
        self.assertEqual(result["proposal_candidate_count_by_source"]["raw_macro_component"], 1)
        self.assertEqual(result["proposal_matched_truth_count_by_source"]["same_instance"], 1)
        self.assertEqual(result["proposal_matched_truth_count_by_source"]["raw_macro_component"], 1)

    @patch("object_proposal_recall_report.replay_candidates_for_sample")
    def test_build_threshold_result_reports_proposal_source_coverage_by_macro(self, mock_replay):
        from object_proposal_recall_report import build_threshold_result

        mock_replay.return_value = [
            {
                "source": "fallback_rule",
                "proposal_sources": ["fallback_rule"],
                "proposal_macro": "through_hole_round",
                "classified_macro": "through_hole_round",
                "action": "accepted_remap",
                "face_indices": [7],
                "face_count": 1,
                "area": 200.0,
            }
        ]

        result = build_threshold_result(
            graph_by_sample={"part1": {"nodes": [], "edges": []}},
            raw_prediction_by_sample={"part1": {"faces": []}},
            truth_by_sample={"part1": [{"sample": "part1", "macro": "through_hole_round", "area": 200.0, "face_count": 1, "truth_index": 0}]},
            final_prediction_by_sample={"part1": {"faces": [{"idx": 7, "macro_feature": "through_hole_round"}]}},
            classifier_bundle=None,
            strong_threshold=0.3,
            weak_threshold=0.2,
            min_confidence=0.5,
            absolute_area_tolerance=10.0,
            relative_area_tolerance=0.02,
            proposal_mode="hybrid",
        )

        self.assertEqual(result["proposal_candidate_count_by_source_macro"]["fallback_rule"]["through_hole_round"], 1)
        self.assertEqual(result["proposal_matched_truth_count_by_source_macro"]["fallback_rule"]["through_hole_round"], 1)


class TestMacroNormalization(unittest.TestCase):
    """v5.11.1: area truth macros must be normalized to prediction vocabulary."""

    def test_truth_macro_aliases_are_normalized_to_prediction_vocabulary(self):
        from object_proposal_recall_report import normalize_truth_entries

        payload = {
            "schema": "cae_area_truth_table_v1",
            "source_xlsx": "W:\\python\\FEA\\dataset_generator\\test_part_s3\\Feature_Comparison_Table_s3_mm2.xlsx",
            "samples": {
                "part1": [
                    {"area": 100.0, "macro": "through_hole", "face_count": 1},
                    {"area": 101.0, "macro": "blind_hole", "face_count": 2},
                    {"area": 102.0, "macro": "stepped_hole", "face_count": 4},
                    {"area": 103.0, "macro": "cylindrical_boss", "face_count": 2},
                    {"area": 104.0, "macro": "rectangle_boss", "face_count": 5},
                    {"area": 105.0, "macro": "poctet_rect", "face_count": 5},
                ]
            },
        }

        normalized = normalize_truth_entries(payload)["part1"]
        macros = [entry["macro"] for entry in normalized]

        self.assertEqual(
            macros,
            [
                "through_hole_round",
                "blind_hole_round",
                "stepped_hole_round",
                "cylinder_boss",
                "rectangular_boss",
                "pocket_rect",
            ],
        )

    def test_candidate_and_final_prediction_macros_are_normalized_before_matching(self):
        from object_proposal_recall_report import candidate_matches_truth

        truth = {"macro": "through_hole_round", "area": 200.0, "face_count": 1}
        candidate = {
            "proposal_macro": "through_hole",
            "classified_macro": "through_hole",
            "action": "accepted_remap",
            "face_indices": [7],
            "face_count": 1,
            "area": 200.0,
        }
        final_prediction = {
            "faces": [
                {"idx": 7, "macro_feature": "through_hole"},
            ]
        }

        self.assertTrue(
            candidate_matches_truth(
                candidate,
                truth,
                "proposal",
                absolute_area_tolerance=10.0,
                relative_area_tolerance=0.02,
                final_prediction=final_prediction,
            )
        )
        self.assertTrue(
            candidate_matches_truth(
                candidate,
                truth,
                "final_survival",
                absolute_area_tolerance=10.0,
                relative_area_tolerance=0.02,
                final_prediction=final_prediction,
            )
        )


class TestHybridProposalMode(unittest.TestCase):
    @patch("object_proposal_recall_report.generate_hybrid_object_proposals")
    @patch("object_proposal_recall_report.assemble_instance_candidates")
    def test_replay_candidates_uses_hybrid_proposer_when_requested(self, mock_same_instance, mock_hybrid):
        from object_proposal_recall_report import replay_candidates_for_sample

        mock_same_instance.return_value = [
            {
                "source": "same_instance_seed_component",
                "proposal_macro": "rib",
                "face_indices": [1],
                "seed_face_indices": [1],
                "assembly_reasons": ["single_face_seed"],
            }
        ]
        mock_hybrid.return_value = [
            {
                "source": "raw_macro_component",
                "proposal_sources": ["raw_macro_component"],
                "proposal_macro": "rib",
                "face_indices": [1, 2, 3],
                "seed_face_indices": [1, 2, 3],
                "assembly_reasons": ["raw_macro_connected_component"],
            }
        ]

        candidates = replay_candidates_for_sample(
            graph={"nodes": [], "edges": []},
            raw_prediction={"faces": []},
            classifier_bundle=None,
            strong_threshold=0.3,
            weak_threshold=0.2,
            min_confidence=0.5,
            proposal_mode="hybrid",
        )

        mock_same_instance.assert_called_once()
        mock_hybrid.assert_called_once()
        self.assertEqual(candidates[0]["face_indices"], [1, 2, 3])
        self.assertEqual(candidates[0]["proposal_sources"], ["raw_macro_component"])


class TestMarkdownReportModeTitle(unittest.TestCase):
    def test_write_markdown_title_mentions_hybrid_mode(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from object_proposal_recall_report import write_markdown

        report = {
            "proposal_mode": "hybrid",
            "graph_dir": "graphs",
            "raw_prediction_dir": "raw",
            "truth_json": "truth.json",
            "classifier": "model.joblib",
            "final_prediction_dir": "predictions",
            "requested_threshold_pair_count": 30,
            "evaluated_threshold_pair_count": 23,
            "skipped_threshold_pairs": [{"strong_threshold": 0.3, "weak_threshold": 0.3}],
            "absolute_area_tolerance": 10.0,
            "relative_area_tolerance": 0.02,
            "threshold_results": [],
            "best_by_proposal_recall": {
                "strong_threshold": 0.3,
                "weak_threshold": 0.2,
                "proposal_stage": {"recall": 0.5, "truth_by_macro": {}, "matched_by_macro": {}},
                "final_survival_stage": {"recall": 0.4, "truth_by_macro": {}, "matched_by_macro": {}},
                "failure_stage_counts": {},
                "samples": [],
            },
            "best_by_final_survival_recall": {
                "strong_threshold": 0.3,
                "weak_threshold": 0.2,
                "proposal_stage": {"recall": 0.5, "truth_by_macro": {}, "matched_by_macro": {}},
                "final_survival_stage": {"recall": 0.4, "truth_by_macro": {}, "matched_by_macro": {}},
                "failure_stage_counts": {},
                "samples": [],
            },
        }

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.md"
            write_markdown(path, report)
            text = path.read_text(encoding="utf-8")

        self.assertTrue(text.startswith("# v5.13 Object Proposal Recall Diagnostic Report"))
        self.assertIn("- Proposal mode: `hybrid`", text)


if __name__ == "__main__":
    unittest.main()
