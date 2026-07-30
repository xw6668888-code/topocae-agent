from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from models.p1_b_candidate_topology_classifier import CandidateModelDimensions
from models.p1_c_candidate_topology_classifier_v2 import (
    BoundaryContextCandidateTopologyClassifier,
)
from utils.p1_c_candidate_iteration_paths import (
    CandidateIterationRunRoot,
    MODEL_NAME,
    PACKAGE_NAME,
    PIPELINE_NAME,
    SELECTION_NAME,
)
from utils.p1_c_run_paths import p1_c_run_dir
from utils.p1_c_topology_verifier_v5 import verify_candidate_prediction


class BoundaryContextModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dimensions = CandidateModelDimensions(
            face_embedding_dim=8,
            edge_embedding_dim=4,
            engineered_feature_dim=6,
            hidden_dim=16,
            semantic_role_count=3,
        )
        self.model = BoundaryContextCandidateTopologyClassifier(
            self.dimensions, macro_classes=["rib", "web"]
        )

    def test_forward_accepts_nonempty_boundary_context(self) -> None:
        result = self.model(
            torch.randn(3, 8),
            torch.randn(2, 4),
            torch.randn(3, 4),
            torch.randn(6),
            torch.randn(2, 8),
        )
        self.assertEqual(tuple(result["macro_logits"].shape), (2,))
        self.assertEqual(tuple(result["membership_logits"].shape), (3,))
        self.assertEqual(tuple(result["semantic_role_logits"].shape), (3, 3))

    def test_forward_accepts_empty_boundary_context(self) -> None:
        result = self.model(
            torch.randn(1, 8),
            torch.empty(0, 4),
            torch.empty(0, 4),
            torch.randn(6),
            torch.empty(0, 8),
        )
        self.assertTrue(torch.isfinite(result["macro_logits"]).all())


class CandidateIterationPathTests(unittest.TestCase):
    def test_all_iteration_artifacts_are_isolated_from_initial_candidate_run(self) -> None:
        seed = 20260718
        run = CandidateIterationRunRoot() / f"same_instance_seed_{seed}"
        self.assertEqual(run / "candidate_training_v1", p1_c_run_dir(seed) / PACKAGE_NAME)
        self.assertEqual(run / "candidate_model_v1", p1_c_run_dir(seed) / MODEL_NAME)
        self.assertEqual(run / "global_decode_v2_selection.json", p1_c_run_dir(seed) / SELECTION_NAME)
        self.assertEqual(run / "synthetic_pipeline_v1", p1_c_run_dir(seed) / PIPELINE_NAME)


class AuditOnlyConsensusVerifierTests(unittest.TestCase):
    def test_face_disagreement_is_reported_without_mutating_confidence(self) -> None:
        base = {
            "masked_top1": "rib",
            "refined_face_indices": [0, 1],
            "final_confidence": 0.9,
            "status": "accepted",
            "abstain": False,
            "predicted_macro": "rib",
            "abstention_reasons": [],
            "verifier_contract": {},
        }
        with patch(
            "utils.p1_c_topology_verifier_v5.revision_3.verify_candidate_prediction",
            return_value=base,
        ):
            result = verify_candidate_prediction(
                {},
                {},
                {"face_macro_predictions": ["web", "web"]},
                thresholds={},
            )
        self.assertEqual(result["face_candidate_macro_agreement"], 0.0)
        self.assertEqual(result["final_confidence"], 0.9)
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["predicted_macro"], "rib")


if __name__ == "__main__":
    unittest.main()
