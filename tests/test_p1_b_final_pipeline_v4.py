from __future__ import annotations

import unittest

from p1_b_general_recognition_pipeline_v4 import PIPELINE_SCHEMA_VERSION
from utils.p1_b_candidate_features import MACRO_CLASSES
from utils.p1_b_global_decode_v2 import decode_feature_objects
from utils.p1_b_topology_verifier_v4 import select_global_candidate_thresholds, verify_candidate_prediction


class P1BFinalPipelineV4Tests(unittest.TestCase):
    def test_face_candidate_disagreement_only_reduces_confidence(self) -> None:
        graph = _graph()
        prediction = _prediction("cylinder_boss")
        prediction["face_macro_predictions"] = {0: "pocket_circular", 1: "pocket_circular"}
        verified = verify_candidate_prediction(
            graph, {"candidate_id": "candidate", "face_indices": [0, 1]},
            prediction, thresholds=_thresholds(),
        )
        self.assertEqual(verified["masked_top1"], "cylinder_boss")
        self.assertEqual(verified["predicted_macro"], "unknown")
        self.assertTrue(verified["abstain"])
        self.assertFalse(verified["verifier_contract"]["hard_class_rewrite"])
        self.assertEqual(verified["face_candidate_macro_agreement"], 0.0)

    def test_global_decode_dominates_near_tie_same_macro_strict_subset(self) -> None:
        graph = _graph()
        decoded = decode_feature_objects(
            graph,
            [
                _verified("partial", [0], 0.99),
                _verified("complete", [0, 1], 0.91),
            ],
            subset_confidence_tolerance=0.1,
            selection_sha256="SYNTHETIC_VAL_SELECTION",
        )
        self.assertEqual(decoded["subset_dominated_count"], 1)
        self.assertEqual(decoded["feature_objects"][0]["face_indices"], [0, 1])
        self.assertEqual(decoded["global_decode_v2_contract"]["selection_split"], "synthetic_val")

    def test_global_decode_rejects_missing_selection_hash(self) -> None:
        with self.assertRaisesRegex(ValueError, "Synthetic-Val"):
            decode_feature_objects(
                _graph(), [_verified("a", [0], 0.9)],
                subset_confidence_tolerance=0.1, selection_sha256="",
            )

    def test_official_final_pipeline_schema_is_v4(self) -> None:
        self.assertEqual(PIPELINE_SCHEMA_VERSION, "topocae_p1_b_general_recognition_pipeline_4")


def _thresholds() -> dict:
    return select_global_candidate_thresholds(
        [
            {"split": "synthetic_val", "macro_confidence": 0.99, "validity_score": 0.99, "accept_label": 1},
            {"split": "synthetic_val", "macro_confidence": 0.01, "validity_score": 0.01, "accept_label": 0},
        ],
        threshold_grid=[0.5],
    )


def _prediction(macro: str) -> dict:
    probabilities = [0.001] * len(MACRO_CLASSES)
    probabilities[MACRO_CLASSES.index(macro)] = 0.999
    total = sum(probabilities)
    probabilities = [value / total for value in probabilities]
    return {
        "macro_classes": MACRO_CLASSES, "macro_probabilities": probabilities,
        "macro_logits": probabilities, "validity_score": 0.99,
        "membership_probabilities": [0.99, 0.99],
        "boundary_probabilities": [0.99, 0.99],
    }


def _verified(candidate_id: str, faces: list[int], confidence: float) -> dict:
    return {
        "candidate_id": candidate_id, "status": "accepted", "predicted_macro": "cylinder_boss",
        "refined_face_indices": faces, "boundary_face_indices": faces,
        "original_face_indices": faces, "final_confidence": confidence,
        "macro_confidence_before_correction": confidence, "candidate_validity_score": 1.0,
        "topology_ambiguity_penalty": 1.0, "topology_evidence": {}, "class_mask": {},
        "threshold_selection_sha256": "SYNTHETIC_VAL_SELECTION",
    }


def _graph() -> dict:
    nodes = [
        {"idx": 0, "primitive_surface": "cylinder", "attributes": {"topology_v2": {}}},
        {"idx": 1, "primitive_surface": "plane", "attributes": {"topology_v2": {}}},
    ]
    return {
        "schema_version": "topocae_graph_v2_relative_1_1", "nodes": nodes,
        "edges": [[0, 1], [1, 0]],
        "edge_attributes": [
            {"recognition_features_v2": {"material_side_ambiguous_evidence": 0.0}},
            {"recognition_features_v2": {"material_side_ambiguous_evidence": 0.0}},
        ],
    }


if __name__ == "__main__":
    unittest.main()
