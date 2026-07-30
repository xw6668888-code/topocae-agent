import unittest

import torch

from run_gnn_inference import graph_to_tensors
from tests.test_training_export_rotation_invariant_v4 import make_node
from utils.training_export import (
    NODE_FEATURE_MODE_ROTATION_INVARIANT,
    build_node_feature_vector,
    node_feature_schema_for_mode,
)


class RunGNNInferenceRotationInvariantV4Test(unittest.TestCase):
    def test_graph_to_tensors_uses_checkpoint_node_feature_mode(self):
        schema = node_feature_schema_for_mode(NODE_FEATURE_MODE_ROTATION_INVARIANT)
        config = {
            "node_feature_mode": NODE_FEATURE_MODE_ROTATION_INVARIANT,
            "normalization": {
                "feature_mean": [0.0] * len(schema),
                "feature_std": [1.0] * len(schema),
            },
            "edge_feature_dim": 0,
        }
        node = make_node(
            center=[1.0, 2.0, 3.0],
            normal=[0.0, 0.0, 1.0],
            axis=[0.0, 0.0, 1.0],
            bbox=[10.0, 4.0, 0.0],
            uv=[10.0, 4.0],
        )
        graph = {"nodes": [node], "edges": [], "edge_attributes": []}

        tensors = graph_to_tensors(graph, config, ndigits=6, device=torch.device("cpu"))

        expected = build_node_feature_vector(
            node,
            ndigits=6,
            node_feature_mode=NODE_FEATURE_MODE_ROTATION_INVARIANT,
        )
        self.assertEqual(tensors["node_features"].shape, (1, len(schema)))
        self.assertTrue(
            torch.allclose(
                tensors["node_features"][0],
                torch.tensor(expected, dtype=torch.float32),
                atol=1e-6,
            )
        )


if __name__ == "__main__":
    unittest.main()
