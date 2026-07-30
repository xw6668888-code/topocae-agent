from __future__ import annotations

import unittest

from utils.instance_candidate_features import extract_instance_candidate_features


def node(idx: int, primitive: str = "plane", area: float = 10.0):
    return {
        "idx": idx,
        "primitive_surface": primitive,
        "area": area,
        "attributes": {
            "physical": {"area": area},
            "shape_descriptors": {
                "projected_long_extent": 10.0,
                "projected_short_extent": 2.0,
                "projected_aspect_ratio": 5.0,
            },
        },
    }


class InstanceCandidateFeaturesV524Test(unittest.TestCase):
    def test_targeted_assembly_reasons_are_encoded_as_binary_features(self):
        graph_data = {"nodes": [node(1), node(2), node(3)], "edges": [], "edge_attributes": []}
        candidate = {
            "proposal_macro": "rib",
            "face_indices": [1, 2, 3],
            "source": "topology_rule",
            "proposal_sources": ["topology_rule"],
            "assembly_reasons": ["topology_review_targeted_rib_residual"],
        }

        features = extract_instance_candidate_features(graph_data, candidate, prediction_faces=[])

        self.assertEqual(features["assembly_reason_count"], 1.0)
        self.assertEqual(features["assembly_reason_topology_review_targeted_rib_residual"], 1.0)
        self.assertEqual(features["assembly_reason_topology_review_targeted_fillet_residual"], 0.0)
        self.assertEqual(features["assembly_reason_topology_review_targeted_chamfer_residual"], 0.0)

    def test_unknown_assembly_reason_sets_other_bucket(self):
        graph_data = {"nodes": [node(4, primitive="cylinder")], "edges": [], "edge_attributes": []}
        candidate = {
            "proposal_macro": "fillet",
            "face_indices": [4],
            "source": "topology_rule",
            "proposal_sources": ["topology_rule"],
            "assembly_reasons": ["new_reason_not_in_schema"],
        }

        features = extract_instance_candidate_features(graph_data, candidate, prediction_faces=[])

        self.assertEqual(features["assembly_reason_count"], 1.0)
        self.assertEqual(features["assembly_reason_other"], 1.0)


if __name__ == "__main__":
    unittest.main()
