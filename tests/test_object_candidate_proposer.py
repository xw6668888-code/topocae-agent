from __future__ import annotations

import unittest
from unittest.mock import patch


class ObjectCandidateProposerTest(unittest.TestCase):
    def test_merge_face_records_uses_prediction_macro_and_graph_area(self):
        from utils.object_candidate_proposer import merge_face_records

        graph = {
            "nodes": [
                {"idx": 0, "primitive_surface": "cylinder", "attributes": {"physical": {"area": 20.0}}},
                {"idx": 1, "primitive_surface": "plane", "attributes": {"physical": {"area": 5.0}}},
            ],
            "edges": [[0, 1]],
        }
        prediction = {
            "faces": [
                {"idx": 0, "macro_feature": "through_hole", "semantic_subtype": "through_hole_cylinder"},
                {"idx": 1, "macro_feature": "stock", "semantic_subtype": "stock_face"},
            ]
        }

        faces = merge_face_records(graph, prediction)
        by_idx = {face["idx"]: face for face in faces}

        self.assertEqual(by_idx[0]["macro_feature"], "through_hole_round")
        self.assertEqual(by_idx[0]["raw_macro_feature"], "through_hole")
        self.assertEqual(by_idx[0]["area"], 20.0)
        self.assertEqual(by_idx[0]["primitive_surface"], "cylinder")
        self.assertEqual(by_idx[1]["macro_feature"], "stock")

    def test_raw_macro_component_proposals_group_adjacent_same_macro_faces(self):
        from utils.object_candidate_proposer import raw_macro_component_proposals

        graph = {
            "nodes": [
                {"idx": 0, "primitive_surface": "plane", "attributes": {"physical": {"area": 10.0}}},
                {"idx": 1, "primitive_surface": "plane", "attributes": {"physical": {"area": 11.0}}},
                {"idx": 2, "primitive_surface": "plane", "attributes": {"physical": {"area": 100.0}}},
                {"idx": 3, "primitive_surface": "plane", "attributes": {"physical": {"area": 12.0}}},
            ],
            "edges": [[0, 1], [1, 2], [2, 3]],
        }
        faces = [
            {"idx": 0, "macro_feature": "rib", "area": 10.0},
            {"idx": 1, "macro_feature": "rib", "area": 11.0},
            {"idx": 2, "macro_feature": "stock", "area": 100.0},
            {"idx": 3, "macro_feature": "rib", "area": 12.0},
        ]

        proposals = raw_macro_component_proposals(graph, faces)

        self.assertEqual(
            [(p["proposal_macro"], p["face_indices"], p["proposal_sources"]) for p in proposals],
            [
                ("rib", [0, 1], ["raw_macro_component"]),
                ("rib", [3], ["raw_macro_component"]),
            ],
        )


    def test_topology_rule_proposals_wrap_existing_finders(self):
        from utils.object_candidate_proposer import topology_rule_proposals

        graph = {
            "nodes": [
                {"idx": 0, "primitive_surface": "cylinder", "attributes": {"physical": {"area": 30.0}}},
                {"idx": 1, "primitive_surface": "plane", "attributes": {"physical": {"area": 8.0}}},
            ],
            "edges": [[0, 1]],
        }
        faces = [
            {"idx": 0, "macro_feature": "stock", "area": 30.0, "primitive_surface": "cylinder"},
            {"idx": 1, "macro_feature": "stock", "area": 8.0, "primitive_surface": "plane"},
        ]

        with patch("utils.object_candidate_proposer.find_through_hole_opening_candidates", return_value=[[0]]), \
             patch("utils.object_candidate_proposer.find_blind_hole_candidates", return_value=[[0, 1]]), \
             patch("utils.object_candidate_proposer.find_counterbore_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_countersink_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_cylinder_boss_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_obround_hole_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_rectangular_prism_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_supported_rectangular_boss_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_u_slot_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_rectangular_slot_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_o_slot_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_v_slot_stock_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_mixed_rib_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_rib_stock_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_existing_rib_web_cap_completion_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_transition_cap_rib_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_pocket_embedded_rib_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_existing_rib_transition_absorption_candidates", return_value=[]):
            proposals = topology_rule_proposals(graph, faces)

        self.assertEqual(
            [(p["proposal_macro"], p["face_indices"], p["proposal_sources"]) for p in proposals],
            [
                ("through_hole_round", [0], ["topology_rule"]),
                ("blind_hole_round", [0, 1], ["topology_rule"]),
            ],
        )

    def test_deduplicate_proposals_merges_sources_for_same_macro_and_faces(self):
        from utils.object_candidate_proposer import deduplicate_proposals

        proposals = [
            {
                "source": "same_instance",
                "proposal_sources": ["same_instance"],
                "proposal_macro": "rib",
                "face_indices": [3, 1, 2],
                "seed_face_indices": [3, 1, 2],
                "assembly_reasons": ["strong_same_instance_edge"],
                "mean_same_instance_probability": 0.8,
                "min_same_instance_probability": 0.7,
            },
            {
                "source": "raw_macro_component",
                "proposal_sources": ["raw_macro_component"],
                "proposal_macro": "rib",
                "face_indices": [1, 2, 3],
                "seed_face_indices": [1, 2, 3],
                "assembly_reasons": ["raw_macro_connected_component"],
                "mean_same_instance_probability": None,
                "min_same_instance_probability": None,
            },
        ]

        merged = deduplicate_proposals(proposals)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["source"], "hybrid_merged")
        self.assertEqual(merged[0]["proposal_sources"], ["same_instance", "raw_macro_component"])
        self.assertEqual(merged[0]["face_indices"], [1, 2, 3])
        self.assertEqual(
            merged[0]["assembly_reasons"],
            ["strong_same_instance_edge", "raw_macro_connected_component"],
        )
        self.assertEqual(merged[0]["mean_same_instance_probability"], 0.8)
        self.assertEqual(merged[0]["min_same_instance_probability"], 0.7)


    def test_topology_rule_proposals_include_fillet_and_chamfer_finders(self):
        from utils.object_candidate_proposer import topology_rule_proposals

        graph = {
            "nodes": [
                {"idx": 0, "primitive_surface": "torus", "attributes": {"physical": {"area": 12.0}}},
                {"idx": 1, "primitive_surface": "plane", "attributes": {"physical": {"area": 4.0}}},
            ],
            "edges": [],
        }
        faces = [
            {"idx": 0, "macro_feature": "stock", "area": 12.0, "primitive_surface": "torus"},
            {"idx": 1, "macro_feature": "stock", "area": 4.0, "primitive_surface": "plane"},
        ]

        with patch("utils.object_candidate_proposer.find_through_hole_opening_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_blind_hole_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_counterbore_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_countersink_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_cylinder_boss_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_obround_hole_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_rectangular_prism_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_supported_rectangular_boss_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_u_slot_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_rectangular_slot_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_o_slot_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_v_slot_stock_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_mixed_rib_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_rib_stock_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_existing_rib_web_cap_completion_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_transition_cap_rib_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_pocket_embedded_rib_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_existing_rib_transition_absorption_candidates", return_value=[]), \
             patch("utils.object_candidate_proposer.find_fillet_stock_candidates", return_value=[[0]]), \
             patch("utils.object_candidate_proposer.find_chamfer_stock_candidates", return_value=[[1]]):
            proposals = topology_rule_proposals(graph, faces)

        self.assertEqual(
            [(p["proposal_macro"], p["face_indices"], p["assembly_reasons"]) for p in proposals],
            [
                ("fillet", [0], ["topology_fillet_stock_transition"]),
                ("chamfer", [1], ["topology_chamfer_stock_transition"]),
            ],
        )

    def test_opening_cylinder_fallback_proposes_stock_cylinder_with_two_circular_plane_boundaries(self):
        from utils.object_candidate_proposer import opening_cylinder_fallback_proposals

        graph = {
            "nodes": [
                {"idx": 0, "primitive_surface": "cylinder", "attributes": {"physical": {"area": 188.5}}},
                {"idx": 1, "primitive_surface": "plane", "attributes": {"physical": {"area": 1000.0}}},
                {"idx": 2, "primitive_surface": "plane", "attributes": {"physical": {"area": 1000.0}}},
                {"idx": 3, "primitive_surface": "cylinder", "attributes": {"physical": {"area": 900.0}}},
            ],
            "edges": [[0, 1], [0, 2], [3, 1]],
            "edge_attributes": [
                {"curve_type": "circle", "convexity": "convex"},
                {"curve_type": "circle", "convexity": "convex"},
                {"curve_type": "line", "convexity": "convex"},
            ],
        }
        faces = [
            {"idx": 0, "macro_feature": "stock", "area": 188.5, "primitive_surface": "cylinder"},
            {"idx": 1, "macro_feature": "stock", "area": 1000.0, "primitive_surface": "plane"},
            {"idx": 2, "macro_feature": "stock", "area": 1000.0, "primitive_surface": "plane"},
            {"idx": 3, "macro_feature": "stock", "area": 900.0, "primitive_surface": "cylinder"},
        ]

        proposals = opening_cylinder_fallback_proposals(graph, faces)

        self.assertEqual(
            [(p["proposal_macro"], p["face_indices"], p["assembly_reasons"]) for p in proposals],
            [("through_hole_round", [0], ["fallback_two_circular_plane_boundaries"])],
        )


if __name__ == "__main__":
    unittest.main()
