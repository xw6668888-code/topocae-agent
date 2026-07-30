import unittest

import torch

from train_gnn_baseline import adapt_initial_state_dict
from upgrade_graph_edge_attributes_from_steps import merge_refreshed_graph


class UpgradeGraphEdgeAttributesTests(unittest.TestCase):
    def test_merge_preserves_labels_and_uses_refreshed_geometry_attributes(self):
        old_graph = {
            "nodes": [
                {
                    "idx": 0,
                    "macro_feature": "through_hole_round",
                    "macro_feature_id": 3,
                    "semantic_label": "round_hole_wall",
                    "instance_id": 7,
                    "attributes": {"area": 12.0},
                }
            ],
            "edges": [[0, 1], [1, 0]],
            "edge_attributes": [{"length": 1.0}, {"length": 1.0}],
            "feature_sets": [{"macro_feature": "through_hole_round", "faces": [0]}],
            "label_maps": {"macro_feature": {"key_to_id": {"through_hole_round": 3}}},
            "generation_quality": {"old": True},
        }
        refreshed_graph = {
            "nodes": [
                {
                    "idx": 0,
                    "primitive_surface": "cylinder",
                    "primitive_surface_id": 2,
                    "attributes": {"area": 12.5, "radius": 2.0},
                }
            ],
            "edges": [[0, 1], [1, 0]],
            "edge_attributes": [
                {"length": 1.2, "adjacent_plane_loop_role_id": 1},
                {"length": 1.2, "adjacent_plane_loop_role_id": 2},
            ],
            "graph_mode": "inference_unlabeled",
        }

        merged = merge_refreshed_graph(old_graph, refreshed_graph, "part.step")

        self.assertEqual(merged["nodes"][0]["macro_feature"], "through_hole_round")
        self.assertEqual(merged["nodes"][0]["semantic_label"], "round_hole_wall")
        self.assertEqual(merged["nodes"][0]["instance_id"], 7)
        self.assertEqual(merged["nodes"][0]["primitive_surface"], "cylinder")
        self.assertEqual(merged["nodes"][0]["attributes"]["radius"], 2.0)
        self.assertEqual(merged["feature_sets"], old_graph["feature_sets"])
        self.assertEqual(merged["label_maps"], old_graph["label_maps"])
        self.assertEqual(merged["edge_attributes"], refreshed_graph["edge_attributes"])
        self.assertEqual(merged["source_step_path"], "part.step")
        self.assertEqual(merged["loop_role_upgrade"]["status"], "merged")

    def test_merge_rejects_mismatched_topology(self):
        old_graph = {"nodes": [{"idx": 0}], "edges": [[0, 1]], "edge_attributes": [{}]}
        refreshed_graph = {"nodes": [{"idx": 0}, {"idx": 1}], "edges": [[0, 1]], "edge_attributes": [{}]}

        with self.assertRaisesRegex(ValueError, "node count mismatch"):
            merge_refreshed_graph(old_graph, refreshed_graph, "bad.step")


class InitialCheckpointAdaptationTests(unittest.TestCase):
    def test_adapts_edge_feature_input_columns_and_zeros_new_columns(self):
        current_state = {
            "layers.0.message_mlp.0.weight": torch.zeros((4, 8)),
            "edge_projection.0.weight": torch.zeros((4, 3)),
            "macro_head.bias": torch.zeros(2),
        }
        initial_state = {
            "layers.0.message_mlp.0.weight": torch.arange(24, dtype=torch.float32).reshape(4, 6),
            "edge_projection.0.weight": torch.arange(8, dtype=torch.float32).reshape(4, 2),
            "macro_head.bias": torch.tensor([1.0, 2.0]),
        }

        adapted, report = adapt_initial_state_dict(current_state, initial_state)

        self.assertEqual(report["expanded_tensor_count"], 2)
        self.assertTrue(torch.equal(adapted["layers.0.message_mlp.0.weight"][:, :6], initial_state["layers.0.message_mlp.0.weight"]))
        self.assertTrue(torch.equal(adapted["layers.0.message_mlp.0.weight"][:, 6:], torch.zeros((4, 2))))
        self.assertTrue(torch.equal(adapted["edge_projection.0.weight"][:, :2], initial_state["edge_projection.0.weight"]))
        self.assertTrue(torch.equal(adapted["edge_projection.0.weight"][:, 2:], torch.zeros((4, 1))))
        self.assertTrue(torch.equal(adapted["macro_head.bias"], initial_state["macro_head.bias"]))


if __name__ == "__main__":
    unittest.main()
