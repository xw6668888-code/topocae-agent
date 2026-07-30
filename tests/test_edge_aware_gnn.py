import unittest

import torch

from models.gnn_baseline import EdgeAwareMultiHeadGNN
from train_gnn_baseline import GraphBatchCollator, compute_selection_score


class EdgeAwareGNNTest(unittest.TestCase):
    def test_model_forward_returns_face_and_edge_logits(self):
        model = EdgeAwareMultiHeadGNN(
            input_dim=4,
            edge_feature_dim=3,
            num_primitive_surfaces=4,
            num_macro_classes=5,
            num_semantic_classes=6,
            derived_target_dim=2,
            idealization_target_dim=1,
            hidden_dim=16,
            num_layers=2,
            primitive_embedding_dim=4,
            dropout=0.0,
        )
        node_features = torch.randn(3, 4)
        primitive_surface_ids = torch.tensor([1, 2, 1], dtype=torch.long)
        edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long)
        edge_features = torch.randn(3, 3)

        outputs = model(node_features, edge_index, primitive_surface_ids, edge_features)

        self.assertEqual(outputs["macro_logits"].shape, (3, 5))
        self.assertEqual(outputs["semantic_logits"].shape, (3, 6))
        self.assertEqual(outputs["edge_same_instance_logits"].shape, (3,))

    def test_collator_batches_edge_features_and_same_instance_labels(self):
        collator = GraphBatchCollator(
            feature_mean=torch.zeros(2),
            feature_std=torch.ones(2),
            edge_feature_mean=torch.zeros(3),
            edge_feature_std=torch.ones(3),
            derived_target_ids=[],
            idealization_target_ids=[],
        )
        batch = collator([
            {
                "sample_id": "a",
                "num_nodes": 2,
                "node_features": [[1.0, 2.0], [3.0, 4.0]],
                "primitive_surface_ids": [1, 2],
                "macro_feature_ids": [0, 1],
                "semantic_subtype_ids": [0, 5],
                "instance_ids": [0, 9],
                "derived_macro_feature_ids_per_node": [[], []],
                "idealization_tag_ids_per_node": [[], []],
                "edge_index": [[0, 1], [1, 0]],
                "edge_features": [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                "edge_same_instance_labels": [0, 0],
                "edge_supervision_mask": [1.0, 1.0],
            },
            {
                "sample_id": "b",
                "num_nodes": 1,
                "node_features": [[5.0, 6.0]],
                "primitive_surface_ids": [1],
                "macro_feature_ids": [0],
                "semantic_subtype_ids": [0],
                "instance_ids": [0],
                "derived_macro_feature_ids_per_node": [[]],
                "idealization_tag_ids_per_node": [[]],
                "edge_index": [],
                "edge_features": [],
                "edge_same_instance_labels": [],
                "edge_supervision_mask": [],
            },
        ])

        self.assertEqual(batch["edge_index"].shape, (2, 2))
        self.assertEqual(batch["edge_features"].shape, (2, 3))
        self.assertEqual(batch["edge_same_instance_labels"].tolist(), [0.0, 0.0])
        self.assertEqual(batch["edge_supervision_mask"].tolist(), [1.0, 1.0])

    def test_edge_aware_selection_score_includes_same_instance_f1(self):
        metrics = {
            "macro_macro_f1_active": 0.6,
            "semantic_macro_f1_active": 0.8,
            "derived_micro_f1": 1.0,
            "idealization_micro_f1": 0.4,
            "edge_micro_f1": 0.2,
        }

        score = compute_selection_score(
            metrics,
            use_derived=True,
            use_idealization=True,
            use_edge=True,
        )

        self.assertAlmostEqual(score, (0.6 + 0.8 + 1.0 + 0.4 + 0.2) / 5.0)


if __name__ == "__main__":
    unittest.main()
