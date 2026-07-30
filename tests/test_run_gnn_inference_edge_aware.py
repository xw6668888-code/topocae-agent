import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import torch

from models.gnn_baseline import EdgeAwareMultiHeadGNN
from run_gnn_inference import collect_graph_files, build_model_from_checkpoint, graph_to_tensors, predict_one_graph


class RunGNNInferenceEdgeAwareTest(unittest.TestCase):
    def test_collect_graph_files_skips_manifests_and_label_maps(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_path = root / "part1.json"
            graph_path.write_text("{}", encoding="utf-8")
            (root / "inference_graph_manifest.json").write_text("{}", encoding="utf-8")
            (root / "prediction_manifest.json").write_text("{}", encoding="utf-8")
            (root / "primitive_surface_label_map.json").write_text("{}", encoding="utf-8")

            files = collect_graph_files(root, recursive=False)

        self.assertEqual(files, [graph_path])

    def test_build_model_from_edge_aware_checkpoint(self):
        config = {
            "model_kind": "edge_aware",
            "hidden_dim": 8,
            "num_layers": 1,
            "primitive_embedding_dim": 4,
            "dropout": 0.0,
            "edge_feature_dim": 3,
            "normalization": {
                "feature_mean": [0.0, 0.0],
                "feature_std": [1.0, 1.0],
                "edge_feature_mean": [0.0, 0.0, 0.0],
                "edge_feature_std": [1.0, 1.0, 1.0],
            },
            "label_spaces": {
                "primitive_surface_id_to_key": {"0": "unknown", "1": "plane", "2": "cylinder"},
                "macro_feature_id_to_key": {"0": "stock", "1": "through_hole_round"},
                "semantic_subtype_id_to_key": {"0": "stock_face", "1": "through_hole_cylinder"},
                "idealization_tag_id_to_key": {},
                "derived_target_ids": [],
                "idealization_target_ids": [],
            },
        }
        model = EdgeAwareMultiHeadGNN(
            input_dim=2,
            edge_feature_dim=3,
            num_primitive_surfaces=3,
            num_macro_classes=2,
            num_semantic_classes=2,
            derived_target_dim=0,
            idealization_target_dim=0,
            hidden_dim=8,
            num_layers=1,
            primitive_embedding_dim=4,
            dropout=0.0,
        )
        checkpoint = {"config": config, "model_state_dict": model.state_dict()}

        loaded = build_model_from_checkpoint(checkpoint, torch.device("cpu"))

        self.assertIsInstance(loaded, EdgeAwareMultiHeadGNN)

    def test_graph_to_tensors_includes_edge_features_for_edge_aware_checkpoint(self):
        config = {
            "normalization": {
                "feature_mean": [0.0] * 50,
                "feature_std": [1.0] * 50,
                "edge_feature_mean": [0.0, 0.0],
                "edge_feature_std": [1.0, 1.0],
            }
        }
        graph = {
            "nodes": [
                {
                    "idx": 0,
                    "primitive_surface_id": 1,
                    "attributes": {
                        "type_onehot": [1] + [0] * 10,
                        "physical": {"area": 1.0, "center_of_mass": [0.0, 0.0, 0.0]},
                        "specific": {},
                        "shape_descriptors": {},
                    },
                },
                {
                    "idx": 1,
                    "primitive_surface_id": 2,
                    "attributes": {
                        "type_onehot": [0, 1] + [0] * 9,
                        "physical": {"area": 1.0, "center_of_mass": [1.0, 0.0, 0.0]},
                        "specific": {"axis": [0.0, 0.0, 1.0], "radius": 2.0},
                        "shape_descriptors": {},
                    },
                },
            ],
            "edges": [[0, 1]],
            "edge_attributes": [{"curve_type_id": 1, "length": 5.0}],
        }

        tensors = graph_to_tensors(graph, config, ndigits=6, device=torch.device("cpu"))

        self.assertIn("edge_features", tensors)
        self.assertEqual(tensors["edge_features"].shape, (1, 2))
        self.assertEqual(float(tensors["edge_features"][0, 1].item()), 5.0)

    def test_predict_one_graph_can_apply_topology_postprocess(self):
        class FixedPocketModel(torch.nn.Module):
            def forward(self, node_features, edge_index, primitive_surface_ids, edge_features=None):
                macro_logits = torch.zeros((2, 12), dtype=torch.float32)
                macro_logits[:, 11] = 8.0
                semantic_logits = torch.zeros((2, 25), dtype=torch.float32)
                semantic_logits[:, 24] = 8.0
                return {
                    "macro_logits": macro_logits,
                    "semantic_logits": semantic_logits,
                    "derived_logits": torch.zeros((2, 0), dtype=torch.float32),
                    "idealization_logits": torch.zeros((2, 0), dtype=torch.float32),
                    "edge_same_instance_logits": torch.tensor([4.0, 4.0], dtype=torch.float32),
                }

        graph = {
            "nodes": [
                {
                    "idx": 0,
                    "primitive_surface": "plane",
                    "primitive_surface_id": 1,
                    "attributes": {
                        "type_onehot": [1] + [0] * 10,
                        "physical": {"area": 1.0, "center_of_mass": [0.0, 0.0, 0.0]},
                        "specific": {"normal": [0.0, 0.0, 1.0]},
                        "shape_descriptors": {},
                    },
                },
                {
                    "idx": 1,
                    "primitive_surface": "plane",
                    "primitive_surface_id": 1,
                    "attributes": {
                        "type_onehot": [1] + [0] * 10,
                        "physical": {"area": 1.0, "center_of_mass": [1.0, 0.0, 0.0]},
                        "specific": {"normal": [0.0, 0.0, 1.0]},
                        "shape_descriptors": {},
                    },
                },
            ],
            "edges": [[0, 1], [1, 0]],
            "edge_attributes": [{"curve_type_id": 1, "length": 1.0}, {"curve_type_id": 1, "length": 1.0}],
        }
        config = {
            "normalization": {
                "feature_mean": [0.0] * 50,
                "feature_std": [1.0] * 50,
                "edge_feature_mean": [0.0, 0.0],
                "edge_feature_std": [1.0, 1.0],
            },
            "edge_feature_dim": 2,
            "label_spaces": {
                "macro_feature_id_to_key": {str(i): f"class_{i}" for i in range(12)} | {"0": "stock", "11": "pocket_rect"},
                "semantic_subtype_id_to_key": {str(i): f"semantic_{i}" for i in range(25)} | {"0": "stock_face", "24": "rectangular_pocket_bottom"},
                "idealization_tag_id_to_key": {},
                "derived_target_ids": [],
                "idealization_target_ids": [],
            },
        }

        prediction = predict_one_graph(
            model=FixedPocketModel(),
            graph_data=graph,
            graph_path=Path("sample.json"),
            checkpoint_config=config,
            ndigits=6,
            device=torch.device("cpu"),
            topology_postprocess=True,
        )

        self.assertEqual([face["macro_feature"] for face in prediction["faces"]], ["stock", "stock"])
        self.assertEqual(prediction["topology_postprocess"]["invalid_component_count"], 1)
        self.assertEqual(prediction["faces"][0]["raw_macro_feature"], "pocket_rect")

    def test_predict_one_graph_can_apply_candidate_classifier_before_topology_postprocess(self):
        class FixedFilletModel(torch.nn.Module):
            def forward(self, node_features, edge_index, primitive_surface_ids, edge_features=None):
                macro_logits = torch.zeros((1, 36), dtype=torch.float32)
                macro_logits[:, 25] = 8.0
                semantic_logits = torch.zeros((1, 34), dtype=torch.float32)
                semantic_logits[:, 33] = 8.0
                return {
                    "macro_logits": macro_logits,
                    "semantic_logits": semantic_logits,
                    "derived_logits": torch.zeros((1, 0), dtype=torch.float32),
                    "idealization_logits": torch.zeros((1, 0), dtype=torch.float32),
                    "edge_same_instance_logits": torch.zeros((0,), dtype=torch.float32),
                }

        class SlotOEstimator:
            classes_ = [0, 7]

            def predict(self, features):
                return [7 for _ in features]

            def predict_proba(self, features):
                return [[0.1, 0.9] for _ in features]

        graph = {
            "nodes": [
                {
                    "idx": 0,
                    "primitive_surface": "cylinder",
                    "primitive_surface_id": 2,
                    "attributes": {
                        "type_onehot": [0, 1] + [0] * 9,
                        "physical": {"area": 3416.28, "center_of_mass": [0.0, 0.0, 0.0]},
                        "specific": {"axis": [0.0, 1.0, 0.0], "radius": 2.5},
                        "shape_descriptors": {},
                    },
                }
            ],
            "edges": [],
            "edge_attributes": [],
        }
        config = {
            "normalization": {
                "feature_mean": [0.0] * 50,
                "feature_std": [1.0] * 50,
                "edge_feature_mean": [0.0, 0.0],
                "edge_feature_std": [1.0, 1.0],
            },
            "edge_feature_dim": 2,
            "label_spaces": {
                "macro_feature_id_to_key": {str(i): f"class_{i}" for i in range(36)} | {"0": "stock", "25": "fillet", "35": "slot_o"},
                "semantic_subtype_id_to_key": {str(i): f"semantic_{i}" for i in range(34)} | {"0": "stock_face", "33": "fillet_blend_surface"},
                "idealization_tag_id_to_key": {},
                "derived_target_ids": [],
                "idealization_target_ids": [],
            },
        }
        candidate_bundle = {
            "estimator": SlotOEstimator(),
            "metadata": {
                "feature_schema": ["area_ratio", "proposal_macro_fillet", "proposal_macro_slot_o"],
                "id_to_label": {"0": "reject", "7": "slot_o"},
            },
        }

        prediction = predict_one_graph(
            model=FixedFilletModel(),
            graph_data=graph,
            graph_path=Path("sample.json"),
            checkpoint_config=config,
            ndigits=6,
            device=torch.device("cpu"),
            topology_postprocess=False,
            candidate_classifier_bundle=candidate_bundle,
        )

        self.assertEqual(prediction["faces"][0]["macro_feature"], "slot_o")
        self.assertEqual(prediction["faces"][0]["raw_macro_feature"], "fillet")
        self.assertEqual(prediction["candidate_classifier_postprocess"]["accepted_count"], 1)



    def test_predict_one_graph_can_apply_rib_boundary_classifier_in_inference_pipeline(self):
        class FixedRibModel(torch.nn.Module):
            def forward(self, node_features, edge_index, primitive_surface_ids, edge_features=None):
                macro_logits = torch.zeros((3, 21), dtype=torch.float32)
                macro_logits[0, 20] = 8.0
                macro_logits[1, 20] = 8.0
                macro_logits[2, 0] = 8.0
                semantic_logits = torch.zeros((3, 35), dtype=torch.float32)
                semantic_logits[:, 0] = 8.0
                return {
                    "macro_logits": macro_logits,
                    "semantic_logits": semantic_logits,
                    "derived_logits": torch.zeros((3, 0), dtype=torch.float32),
                    "idealization_logits": torch.zeros((3, 0), dtype=torch.float32),
                    "edge_same_instance_logits": torch.ones((6,), dtype=torch.float32) * 4.0,
                }

        class RibMemberEstimator:
            classes_ = [0, 1]

            def predict(self, features):
                return [1 for _ in features]

            def predict_proba(self, features):
                return [[0.05, 0.95] for _ in features]

        def node(idx, area):
            return {
                "idx": idx,
                "primitive_surface": "plane",
                "primitive_surface_id": 1,
                "attributes": {
                    "type_onehot": [1] + [0] * 10,
                    "physical": {"area": area, "center_of_mass": [float(idx), 0.0, 0.0]},
                    "specific": {"normal": [0.0, 0.0, 1.0]},
                    "shape_descriptors": {"projected_aspect_ratio": 6.0},
                },
            }

        graph = {
            "nodes": [node(0, 20.0), node(1, 20.0), node(2, 30.0)],
            "edges": [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]],
            "edge_attributes": [
                {"source_face": 0, "target_face": 1, "curve_type": "line", "convexity": "convex", "length": 10.0},
                {"source_face": 1, "target_face": 0, "curve_type": "line", "convexity": "convex", "length": 10.0},
                {"source_face": 0, "target_face": 2, "curve_type": "line", "convexity": "convex", "length": 10.0},
                {"source_face": 2, "target_face": 0, "curve_type": "line", "convexity": "convex", "length": 10.0},
                {"source_face": 1, "target_face": 2, "curve_type": "line", "convexity": "convex", "length": 10.0},
                {"source_face": 2, "target_face": 1, "curve_type": "line", "convexity": "convex", "length": 10.0},
            ],
        }
        config = {
            "normalization": {
                "feature_mean": [0.0] * 50,
                "feature_std": [1.0] * 50,
                "edge_feature_mean": [0.0, 0.0],
                "edge_feature_std": [1.0, 1.0],
            },
            "edge_feature_dim": 2,
            "label_spaces": {
                "macro_feature_id_to_key": {str(i): f"class_{i}" for i in range(21)} | {"0": "stock", "20": "rib"},
                "semantic_subtype_id_to_key": {str(i): f"semantic_{i}" for i in range(35)} | {"0": "stock_face"},
                "idealization_tag_id_to_key": {},
                "derived_target_ids": [],
                "idealization_target_ids": [],
            },
        }
        rib_boundary_bundle = {
            "estimator": RibMemberEstimator(),
            "metadata": {
                "feature_schema": ["candidate_to_seed_area_ratio"],
                "id_to_label": {"0": "not_rib_member", "1": "rib_member"},
            },
        }

        prediction = predict_one_graph(
            model=FixedRibModel(),
            graph_data=graph,
            graph_path=Path("sample.json"),
            checkpoint_config=config,
            ndigits=6,
            device=torch.device("cpu"),
            topology_postprocess=False,
            rib_boundary_classifier_bundle=rib_boundary_bundle,
            rib_boundary_classifier_min_confidence=0.8,
        )

        self.assertEqual(prediction["faces"][2]["macro_feature"], "rib")
        self.assertEqual(prediction["rib_boundary_classifier_postprocess"]["expanded_count"], 1)

    def test_predict_one_graph_can_apply_instance_candidate_classifier_before_topology_postprocess(self):
        class FixedWebModel(torch.nn.Module):
            def forward(self, node_features, edge_index, primitive_surface_ids, edge_features=None):
                macro_logits = torch.zeros((2, 22), dtype=torch.float32)
                macro_logits[:, 21] = 8.0
                semantic_logits = torch.zeros((2, 35), dtype=torch.float32)
                semantic_logits[:, 0] = 8.0
                return {
                    "macro_logits": macro_logits,
                    "semantic_logits": semantic_logits,
                    "derived_logits": torch.zeros((2, 0), dtype=torch.float32),
                    "idealization_logits": torch.zeros((2, 0), dtype=torch.float32),
                    "edge_same_instance_logits": torch.tensor([4.0], dtype=torch.float32),
                }

        class RibObjectEstimator:
            classes_ = [0, 7]

            def predict(self, features):
                return [7 for _ in features]

            def predict_proba(self, features):
                return [[0.05, 0.95] for _ in features]

        def node(idx):
            return {
                "idx": idx,
                "primitive_surface": "plane",
                "primitive_surface_id": 1,
                "attributes": {
                    "type_onehot": [1] + [0] * 10,
                    "physical": {"area": 12.0, "center_of_mass": [float(idx), 0.0, 0.0]},
                    "specific": {"normal": [0.0, 0.0, 1.0]},
                    "shape_descriptors": {"projected_aspect_ratio": 5.0},
                },
            }

        graph = {
            "nodes": [node(0), node(1)],
            "edges": [[0, 1]],
            "edge_attributes": [
                {"source_face": 0, "target_face": 1, "curve_type": "line", "convexity": "convex", "length": 8.0},
            ],
        }
        config = {
            "normalization": {
                "feature_mean": [0.0] * 50,
                "feature_std": [1.0] * 50,
                "edge_feature_mean": [0.0, 0.0],
                "edge_feature_std": [1.0, 1.0],
            },
            "edge_feature_dim": 2,
            "label_spaces": {
                "macro_feature_id_to_key": {str(i): f"class_{i}" for i in range(22)} | {"0": "stock", "20": "rib", "21": "web"},
                "semantic_subtype_id_to_key": {str(i): f"semantic_{i}" for i in range(35)} | {"0": "stock_face"},
                "idealization_tag_id_to_key": {},
                "derived_target_ids": [],
                "idealization_target_ids": [],
            },
        }
        instance_bundle = {
            "estimator": RibObjectEstimator(),
            "metadata": {
                "feature_schema": ["face_count"],
                "id_to_label": {"0": "reject", "7": "rib"},
            },
        }

        prediction = predict_one_graph(
            model=FixedWebModel(),
            graph_data=graph,
            graph_path=Path("sample.json"),
            checkpoint_config=config,
            ndigits=6,
            device=torch.device("cpu"),
            topology_postprocess=False,
            instance_candidate_classifier_bundle=instance_bundle,
            instance_candidate_min_confidence=0.8,
            same_instance_assembly_threshold=0.55,
            same_instance_weak_threshold=0.35,
        )

        self.assertEqual([face["macro_feature"] for face in prediction["faces"]], ["rib", "rib"])
        self.assertEqual(prediction["faces"][0]["raw_macro_feature"], "web")
        self.assertEqual(prediction["instance_candidate_postprocess"]["accepted_count"], 1)
        self.assertEqual(prediction["instance_candidate_postprocess"]["strong_threshold"], 0.55)
if __name__ == "__main__":
    unittest.main()



