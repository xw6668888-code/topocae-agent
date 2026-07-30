import unittest

from utils.training_export import EDGE_FEATURE_SCHEMA, build_gnn_record


class TrainingExportEdgeFeaturesTest(unittest.TestCase):
    def test_edge_attributes_are_exported_as_numeric_edge_features(self):
        sample_record = {
            "sample_id": "sample",
            "graph_path": "graphs/sample.json",
            "step_path": "steps/sample.step",
        }
        graph_data = {
            "nodes": [
                {
                    "idx": 0,
                    "primitive_surface_id": 1,
                    "macro_feature_id": 1,
                    "semantic_label": 5,
                    "derived_macro_feature_ids": [],
                    "idealization_tag_ids": [],
                    "instance_id": 7,
                    "attributes": {
                        "type_onehot": [1] + [0] * 10,
                        "physical": {"area": 10.0, "center_of_mass": [0.0, 0.0, 0.0]},
                        "specific": {"normal": [0.0, 0.0, 1.0]},
                        "shape_descriptors": {},
                    },
                },
                {
                    "idx": 1,
                    "primitive_surface_id": 2,
                    "macro_feature_id": 1,
                    "semantic_label": 5,
                    "derived_macro_feature_ids": [],
                    "idealization_tag_ids": [],
                    "instance_id": 7,
                    "attributes": {
                        "type_onehot": [0, 1] + [0] * 9,
                        "physical": {"area": 12.0, "center_of_mass": [1.0, 0.0, 0.0]},
                        "specific": {"axis": [0.0, 0.0, 1.0], "radius": 3.0},
                        "shape_descriptors": {},
                    },
                },
            ],
            "edges": [[0, 1], [1, 0]],
            "edge_attributes": [
                {
                    "curve_type_id": 1,
                    "length": 5.0,
                    "is_line": 1.0,
                    "is_circle": 0.0,
                    "is_ellipse": 0.0,
                    "is_spline": 0.0,
                    "circle_radius": 0.0,
                    "convexity_id": 1,
                    "dihedral_angle": 1.570796,
                    "adjacent_face_normal_angle": 1.570796,
                },
                {
                    "curve_type_id": 1,
                    "length": 5.0,
                    "is_line": 1.0,
                    "is_circle": 0.0,
                    "is_ellipse": 0.0,
                    "is_spline": 0.0,
                    "circle_radius": 0.0,
                    "convexity_id": 1,
                    "dihedral_angle": 1.570796,
                    "adjacent_face_normal_angle": 1.570796,
                },
            ],
            "feature_sets": [],
        }

        record = build_gnn_record(sample_record, graph_data, ndigits=6)

        self.assertIn("edge_features", record)
        self.assertEqual(len(record["edge_features"]), 2)
        self.assertEqual(len(record["edge_features"][0]), len(EDGE_FEATURE_SCHEMA))
        self.assertEqual(record["edge_features"][0][EDGE_FEATURE_SCHEMA.index("length")], 5.0)
        self.assertEqual(record["edge_features"][0][EDGE_FEATURE_SCHEMA.index("is_line")], 1.0)

    def test_loop_role_edge_attributes_are_exported_as_numeric_edge_features(self):
        sample_record = {
            "sample_id": "sample",
            "graph_path": "graphs/sample.json",
            "step_path": "steps/sample.step",
        }
        graph_data = {
            "nodes": [
                {
                    "idx": 0,
                    "primitive_surface_id": 2,
                    "macro_feature_id": 1,
                    "semantic_label": 5,
                    "derived_macro_feature_ids": [],
                    "idealization_tag_ids": [],
                    "instance_id": 7,
                    "attributes": {
                        "type_onehot": [0, 1] + [0] * 9,
                        "physical": {"area": 12.0, "center_of_mass": [0.0, 0.0, 0.0]},
                        "specific": {"axis": [0.0, 0.0, 1.0], "radius": 3.0},
                        "shape_descriptors": {},
                    },
                },
                {
                    "idx": 1,
                    "primitive_surface_id": 1,
                    "macro_feature_id": 0,
                    "semantic_label": 0,
                    "derived_macro_feature_ids": [],
                    "idealization_tag_ids": [],
                    "instance_id": 0,
                    "attributes": {
                        "type_onehot": [1] + [0] * 10,
                        "physical": {"area": 100.0, "center_of_mass": [0.0, 0.0, 1.0]},
                        "specific": {"normal": [0.0, 0.0, 1.0]},
                        "shape_descriptors": {},
                    },
                },
            ],
            "edges": [[0, 1]],
            "edge_attributes": [
                {
                    "curve_type_id": 2,
                    "length": 18.849,
                    "is_circle": 1.0,
                    "adjacent_plane_loop_role": "inner_loop",
                    "adjacent_plane_loop_role_id": 1,
                    "adjacent_plane_loop_role_inner": 1.0,
                    "adjacent_plane_loop_role_outer": 0.0,
                    "cylindrical_wall_role_hint": "inner_hole_wall",
                    "cylindrical_wall_role_hint_id": 1,
                    "cylindrical_wall_role_inner_hint": 1.0,
                    "cylindrical_wall_role_outer_hint": 0.0,
                }
            ],
            "feature_sets": [],
        }

        record = build_gnn_record(sample_record, graph_data, ndigits=6)
        edge_features = record["edge_features"][0]

        self.assertIn("adjacent_plane_loop_role_inner", EDGE_FEATURE_SCHEMA)
        self.assertIn("cylindrical_wall_role_inner_hint", EDGE_FEATURE_SCHEMA)
        self.assertEqual(edge_features[EDGE_FEATURE_SCHEMA.index("adjacent_plane_loop_role_inner")], 1.0)
        self.assertEqual(edge_features[EDGE_FEATURE_SCHEMA.index("cylindrical_wall_role_inner_hint")], 1.0)

    def test_same_instance_edge_labels_ignore_pure_background_edges(self):
        sample_record = {
            "sample_id": "sample",
            "graph_path": "graphs/sample.json",
            "step_path": "steps/sample.step",
        }
        graph_data = {
            "nodes": [
                {
                    "idx": 0,
                    "primitive_surface_id": 1,
                    "macro_feature_id": 0,
                    "semantic_label": 0,
                    "derived_macro_feature_ids": [],
                    "idealization_tag_ids": [],
                    "instance_id": 0,
                    "attributes": {
                        "type_onehot": [1] + [0] * 10,
                        "physical": {"area": 10.0, "center_of_mass": [0.0, 0.0, 0.0]},
                        "specific": {},
                        "shape_descriptors": {},
                    },
                },
                {
                    "idx": 1,
                    "primitive_surface_id": 2,
                    "macro_feature_id": 1,
                    "semantic_label": 5,
                    "derived_macro_feature_ids": [],
                    "idealization_tag_ids": [],
                    "instance_id": 7,
                    "attributes": {
                        "type_onehot": [0, 1] + [0] * 9,
                        "physical": {"area": 12.0, "center_of_mass": [1.0, 0.0, 0.0]},
                        "specific": {"axis": [0.0, 0.0, 1.0], "radius": 3.0},
                        "shape_descriptors": {},
                    },
                },
                {
                    "idx": 2,
                    "primitive_surface_id": 1,
                    "macro_feature_id": 1,
                    "semantic_label": 7,
                    "derived_macro_feature_ids": [],
                    "idealization_tag_ids": [],
                    "instance_id": 7,
                    "attributes": {
                        "type_onehot": [1] + [0] * 10,
                        "physical": {"area": 8.0, "center_of_mass": [1.0, 0.0, -3.0]},
                        "specific": {"normal": [0.0, 0.0, 1.0]},
                        "shape_descriptors": {},
                    },
                },
                {
                    "idx": 3,
                    "primitive_surface_id": 1,
                    "macro_feature_id": 0,
                    "semantic_label": 0,
                    "derived_macro_feature_ids": [],
                    "idealization_tag_ids": [],
                    "instance_id": 0,
                    "attributes": {
                        "type_onehot": [1] + [0] * 10,
                        "physical": {"area": 20.0, "center_of_mass": [3.0, 0.0, 0.0]},
                        "specific": {},
                        "shape_descriptors": {},
                    },
                },
            ],
            "edges": [[1, 2], [2, 1], [1, 3], [0, 3]],
            "feature_sets": [],
        }

        record = build_gnn_record(sample_record, graph_data, ndigits=6)

        self.assertEqual(record["edge_same_instance_labels"], [1, 1, 0, -1])
        self.assertEqual(record["edge_supervision_mask"], [1.0, 1.0, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
