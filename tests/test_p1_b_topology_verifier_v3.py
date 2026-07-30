from __future__ import annotations

import unittest

from p1_b_general_recognition_pipeline_v3 import PIPELINE_SCHEMA_VERSION
from utils.p1_b_candidate_features import MACRO_CLASSES
from utils.p1_b_topology_verifier_v3 import select_global_candidate_thresholds, verify_candidate_prediction


class P1BTopologyVerifierV3Tests(unittest.TestCase):
    def test_single_closed_torus_slot_is_not_rejected_by_face_count(self) -> None:
        graph = _graph(
            [_node(0, "torus", 3)],
            [],
        )
        verified = verify_candidate_prediction(
            graph,
            {"candidate_id": "single", "face_indices": [0]},
            _prediction("slot_o", 1),
            thresholds=_thresholds(),
        )
        self.assertEqual(verified["status"], "accepted")
        self.assertEqual(verified["predicted_macro"], "slot_o")
        self.assertIn("closed_single_surface_slot_evidence", verified["topology_evidence"])

    def test_parent_bridged_slot_membership_keeps_both_original_faces(self) -> None:
        graph = _graph(
            [_node(0, "cylinder", 3), _node(1, "plane", 4), _node(2, "cylinder", 3)],
            [(0, 1), (1, 0), (2, 1), (1, 2)],
        )
        verified = verify_candidate_prediction(
            graph,
            {"candidate_id": "bridged", "face_indices": [0, 2]},
            _prediction("slot_rect", 2),
            thresholds=_thresholds(),
        )
        self.assertEqual(verified["status"], "accepted")
        self.assertEqual(verified["refined_face_indices"], [0, 2])
        self.assertTrue(verified["topology_evidence"]["connected_via_shared_boundary_parent"])

    def test_official_pipeline_schema_is_v3(self) -> None:
        self.assertEqual(PIPELINE_SCHEMA_VERSION, "topocae_p1_b_general_recognition_pipeline_3")


def _thresholds() -> dict:
    return select_global_candidate_thresholds(
        [
            {"split": "synthetic_val", "macro_confidence": 0.99, "validity_score": 0.99, "accept_label": 1},
            {"split": "synthetic_val", "macro_confidence": 0.01, "validity_score": 0.01, "accept_label": 0},
        ],
        threshold_grid=[0.5],
    )


def _prediction(macro: str, face_count: int) -> dict:
    probabilities = [0.001] * len(MACRO_CLASSES)
    probabilities[MACRO_CLASSES.index(macro)] = 0.999
    total = sum(probabilities)
    probabilities = [value / total for value in probabilities]
    return {
        "macro_classes": MACRO_CLASSES,
        "macro_probabilities": probabilities,
        "macro_logits": probabilities,
        "validity_score": 0.99,
        "membership_probabilities": [0.99] * face_count,
        "boundary_probabilities": [0.99] * face_count,
    }


def _graph(nodes: list[dict], edges: list[tuple[int, int]]) -> dict:
    return {
        "schema_version": "topocae_graph_v2_relative_1_1",
        "nodes": nodes,
        "edges": [list(edge) for edge in edges],
        "edge_attributes": [{"recognition_features_v2": {"material_side_ambiguous_evidence": 0.0}} for _ in edges],
    }


def _node(index: int, primitive: str, outer_edge_count: int) -> dict:
    return {
        "idx": index,
        "primitive_surface": primitive,
        "attributes": {
            "topology_v2": {
                "wire_count": 1,
                "outer_loop_count": 1,
                "inner_loop_count": 0,
                "outer_loop_edge_count": outer_edge_count,
                "inner_loop_edge_count": 0,
            }
        },
    }


if __name__ == "__main__":
    unittest.main()
