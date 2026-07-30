import math
import unittest

from utils.instance_candidate_features import extract_instance_candidate_features


def node(idx, primitive="plane", area=1.0, normal=None, center=None, descriptors=None):
    return {
        "idx": idx,
        "primitive_surface": primitive,
        "macro_feature": "truth_label_must_not_be_used",
        "instance_id": 99,
        "attributes": {
            "physical": {
                "area": area,
                "center_of_mass": center or [float(idx), 0.0, 0.0],
            },
            "specific": {"normal": normal or [0.0, 0.0, 1.0]},
            "shape_descriptors": descriptors or {},
        },
    }


def edge(src, dst, curve_type="line", convexity="concave", length=1.0):
    curve_ids = {"line": 1, "circle": 2, "ellipse": 3, "spline": 4}
    convexity_ids = {"convex": 1, "concave": 2, "smooth": 3}
    return [
        [src, dst],
        {
            "source_face": src,
            "target_face": dst,
            "curve_type": curve_type,
            "curve_type_id": curve_ids.get(curve_type, 0),
            "is_line": 1.0 if curve_type == "line" else 0.0,
            "is_circle": 1.0 if curve_type == "circle" else 0.0,
            "convexity": convexity,
            "convexity_id": convexity_ids.get(convexity, 0),
            "length": length,
        },
    ]


def graph(nodes, edge_defs):
    edges = []
    edge_attributes = []
    for pair, attr in edge_defs:
        edges.append(pair)
        edge_attributes.append(attr)
    return {
        "nodes": nodes,
        "edges": edges,
        "edge_attributes": edge_attributes,
        "feature_sets": [
            {
                "macro_feature": "truth_label_must_not_be_used",
                "instance_id": 123,
                "node_indices": [item["idx"] for item in nodes],
            }
        ],
    }


def role_node(idx, primitive="plane", area=10.0, normal=None, radius=None):
    specific = {}
    if normal is not None:
        specific["normal"] = normal
    if radius is not None:
        specific["radius"] = radius
        specific["axis"] = [0.0, 0.0, 1.0]
    return {
        "idx": idx,
        "primitive_surface": primitive,
        "attributes": {
            "physical": {"area": area, "center_of_mass": [float(idx), 0.0, 0.0]},
            "specific": specific,
            "shape_descriptors": {
                "projected_long_extent": 10.0,
                "projected_short_extent": 2.0,
                "projected_aspect_ratio": 5.0,
            },
        },
    }


def role_edge(src, dst, *, curve_type="circle", length=10.0, loop_role="inner_loop", wall_role="inner_hole_wall"):
    loop_inner = 1.0 if loop_role == "inner_loop" else 0.0
    loop_outer = 1.0 if loop_role == "outer_loop" else 0.0
    loop_unknown = 1.0 if loop_role == "unknown" else 0.0
    wall_inner = 1.0 if wall_role == "inner_hole_wall" else 0.0
    wall_outer = 1.0 if wall_role.startswith("outer") else 0.0
    wall_ambiguous = 1.0 if wall_role == "ambiguous" else 0.0
    return [src, dst], {
        "source_face": src,
        "target_face": dst,
        "curve_type": curve_type,
        "curve_type_id": 2 if curve_type == "circle" else 1,
        "is_circle": 1.0 if curve_type == "circle" else 0.0,
        "is_line": 1.0 if curve_type == "line" else 0.0,
        "length": length,
        "circle_radius": length / 6.283185307179586 if curve_type == "circle" else 0.0,
        "convexity": "smooth" if curve_type == "circle" else "convex",
        "convexity_id": 3 if curve_type == "circle" else 1,
        "adjacent_plane_loop_role": loop_role,
        "adjacent_plane_loop_role_inner": loop_inner,
        "adjacent_plane_loop_role_outer": loop_outer,
        "adjacent_plane_loop_role_unknown": loop_unknown,
        "cylindrical_wall_role_hint": wall_role,
        "cylindrical_wall_role_inner_hint": wall_inner,
        "cylindrical_wall_role_outer_hint": wall_outer,
        "cylindrical_wall_role_ambiguous_hint": wall_ambiguous,
    }


class InstanceCandidateFeaturesTest(unittest.TestCase):
    def test_rectangular_pocket_like_candidate_counts_area_primitives_edges_and_topology(self):
        graph_data = graph(
            [node(0, area=10.0)] + [node(idx, area=3.0) for idx in range(1, 5)] + [node(5, area=100.0)],
            [
                edge(0, 1, "line", "concave", 5.0),
                edge(0, 2, "line", "concave", 5.0),
                edge(0, 3, "line", "concave", 3.0),
                edge(0, 4, "line", "concave", 3.0),
                edge(1, 5, "circle", "convex", 2.0),
                edge(2, 5, "ellipse", "smooth", 2.0),
            ],
        )
        prediction_faces = [
            {"idx": idx, "macro_feature": "pocket_rect", "semantic_subtype": "rect_pocket_side_plane"}
            for idx in range(5)
        ]
        candidate = {
            "proposal_macro": "pocket_rect",
            "face_indices": [0, 1, 2, 3, 4],
            "mean_same_instance_probability": 0.82,
            "min_same_instance_probability": 0.61,
        }

        features = extract_instance_candidate_features(graph_data, candidate, prediction_faces=prediction_faces)

        self.assertEqual(features["face_count"], 5.0)
        self.assertEqual(features["candidate_area"], 22.0)
        self.assertAlmostEqual(features["area_ratio"], 22.0 / 122.0)
        self.assertEqual(features["area_min"], 3.0)
        self.assertEqual(features["area_max"], 10.0)
        self.assertAlmostEqual(features["area_mean"], 22.0 / 5.0)
        self.assertEqual(features["primitive_plane_ratio"], 1.0)
        self.assertEqual(features["internal_edge_count"], 4.0)
        self.assertEqual(features["boundary_edge_count"], 2.0)
        self.assertEqual(features["internal_concave_ratio"], 1.0)
        self.assertEqual(features["boundary_convex_ratio"], 0.5)
        self.assertEqual(features["boundary_smooth_ratio"], 0.5)
        self.assertEqual(features["internal_line_ratio"], 1.0)
        self.assertEqual(features["boundary_circle_ratio"], 0.5)
        self.assertEqual(features["boundary_ellipse_ratio"], 0.5)
        self.assertEqual(features["same_instance_probability_mean"], 0.82)
        self.assertEqual(features["same_instance_probability_min"], 0.61)
        self.assertEqual(features["predicted_macro_vote_fraction"], 1.0)
        self.assertEqual(features["topology_valid"], 1.0)

    def test_rib_candidate_exposes_convex_internal_edges_and_valid_thin_wall_topology(self):
        graph_data = graph(
            [
                node(0, area=20.0, normal=[1, 0, 0], center=[0, 0, 0], descriptors={
                    "projected_long_extent": 10.0,
                    "projected_short_extent": 3.0,
                    "projected_aspect_ratio": 3.33,
                }),
                node(1, area=20.0, normal=[1, 0, 0], center=[1, 0, 0], descriptors={
                    "projected_long_extent": 10.0,
                    "projected_short_extent": 3.0,
                    "projected_aspect_ratio": 3.33,
                }),
                node(2, area=4.0, normal=[0, 0, 1], center=[0.5, 0, 1.0], descriptors={
                    "projected_long_extent": 10.0,
                    "projected_short_extent": 1.0,
                }),
                node(3, area=100.0),
            ],
            [
                edge(0, 2, "line", "convex", 10.0),
                edge(1, 2, "line", "convex", 10.0),
                edge(2, 3, "spline", "smooth", 2.0),
            ],
        )
        prediction_faces = [{"idx": idx, "macro_feature": "rib"} for idx in [0, 1, 2]]

        features = extract_instance_candidate_features(
            graph_data,
            {"proposal_macro": "rib", "face_indices": [0, 1, 2]},
            prediction_faces=prediction_faces,
        )

        self.assertEqual(features["face_count"], 3.0)
        self.assertEqual(features["internal_edge_count"], 2.0)
        self.assertEqual(features["boundary_edge_count"], 1.0)
        self.assertEqual(features["internal_convex_ratio"], 1.0)
        self.assertEqual(features["boundary_spline_ratio"], 1.0)
        self.assertEqual(features["topology_valid"], 1.0)

    def test_single_fillet_candidate_is_stable_and_transition_topology_valid(self):
        graph_data = graph(
            [node(0, "cylinder", area=5.0), node(1, "plane", area=100.0)],
            [edge(0, 1, "circle", "smooth", 1.0)],
        )

        features = extract_instance_candidate_features(
            graph_data,
            {"proposal_macro": "fillet", "face_indices": [0]},
            prediction_faces=[{"idx": 0, "macro_feature": "fillet"}],
        )

        self.assertEqual(features["face_count"], 1.0)
        self.assertEqual(features["primitive_cylinder_ratio"], 1.0)
        self.assertEqual(features["boundary_smooth_ratio"], 1.0)
        self.assertEqual(features["topology_valid"], 1.0)

    def test_empty_candidate_returns_numeric_zero_features(self):
        features = extract_instance_candidate_features(
            {"nodes": [], "edges": [], "edge_attributes": []},
            {"proposal_macro": "pocket_rect", "face_indices": []},
        )

        self.assertEqual(features["face_count"], 0.0)
        self.assertEqual(features["candidate_area"], 0.0)
        self.assertEqual(features["area_ratio"], 0.0)
        self.assertEqual(features["topology_valid"], 0.0)
        self.assertTrue(all(isinstance(value, float) for value in features.values()))

    def test_prediction_vote_fraction_does_not_use_graph_truth_macro_or_feature_sets(self):
        graph_data = graph(
            [
                {**node(0, area=1.0), "macro_feature": "rib"},
                {**node(1, area=1.0), "macro_feature": "rib"},
            ],
            [edge(0, 1, "line", "convex", 1.0)],
        )
        graph_data["feature_sets"] = [{"macro_feature": "rib", "instance_id": 7, "node_indices": [0, 1]}]
        prediction_faces = [
            {"idx": 0, "macro_feature": "stock"},
            {"idx": 1, "macro_feature": "stock"},
        ]

        features = extract_instance_candidate_features(
            graph_data,
            {"proposal_macro": "rib", "face_indices": [0, 1]},
            prediction_faces=prediction_faces,
        )

        self.assertEqual(features["predicted_macro_vote_fraction"], 0.0)
        self.assertEqual(features["prediction_macro_rib_fraction"], 0.0)
        self.assertEqual(features["prediction_macro_stock_fraction"], 1.0)

    def test_prediction_vote_fraction_includes_pocket_circular_bucket(self):
        graph_data = graph(
            [node(0, area=2.0), node(1, area=3.0)],
            [edge(0, 1, "circle", "concave", 1.0)],
        )
        prediction_faces = [
            {"idx": 0, "macro_feature": "pocket_circular"},
            {"idx": 1, "macro_feature": "pocket_circular"},
        ]

        features = extract_instance_candidate_features(
            graph_data,
            {"proposal_macro": "pocket_circular", "face_indices": [0, 1]},
            prediction_faces=prediction_faces,
        )

        self.assertEqual(features["prediction_macro_pocket_circular_fraction"], 1.0)
        self.assertEqual(features["predicted_macro_vote_fraction"], 1.0)

    def test_extract_features_sanitizes_nan_and_inf_to_finite_floats(self):
        graph_data = graph(
            [
                node(0, area=float("inf")),
                node(1, area=float("nan")),
                node(2, area=5.0),
            ],
            [edge(0, 2, "line", "convex", 1.0)],
        )

        features = extract_instance_candidate_features(
            graph_data,
            {
                "proposal_macro": "rib",
                "face_indices": [0, 1],
                "mean_same_instance_probability": float("nan"),
                "min_same_instance_probability": float("inf"),
            },
            prediction_faces=[{"idx": 0, "macro_feature": "rib"}, {"idx": 1, "macro_feature": "rib"}],
        )

        self.assertTrue(all(isinstance(value, float) for value in features.values()))
        self.assertTrue(all(math.isfinite(value) for value in features.values()))
        self.assertEqual(features["same_instance_probability_mean"], 0.0)
        self.assertEqual(features["same_instance_probability_min"], 0.0)

    def test_extract_features_includes_hybrid_proposal_source_buckets(self):
        from utils.instance_candidate_features import extract_instance_candidate_features

        graph_data = {
            "nodes": [
                {
                    "idx": 0,
                    "primitive_surface": "cylinder",
                    "attributes": {
                        "physical": {"area": 12.0, "center_of_mass": [0.0, 0.0, 0.0]},
                        "shape_descriptors": {
                            "bbox_longest_extent": 4.0,
                            "bbox_middle_extent": 2.0,
                            "bbox_shortest_extent": 1.0,
                            "projected_long_extent": 4.0,
                            "projected_short_extent": 2.0,
                            "projected_aspect_ratio": 2.0,
                        },
                    },
                }
            ],
            "edges": [],
        }
        candidate = {
            "proposal_macro": "through_hole_round",
            "face_indices": [0],
            "proposal_sources": ["topology_rule", "fallback_rule"],
            "source": "hybrid_merged",
        }

        features = extract_instance_candidate_features(
            graph_data,
            candidate,
            prediction_faces=[{"idx": 0, "macro_feature": "stock"}],
            candidate_macro="through_hole_round",
        )

        self.assertEqual(features["proposal_source_count"], 2.0)
        self.assertEqual(features["proposal_source_same_instance"], 0.0)
        self.assertEqual(features["proposal_source_raw_macro_component"], 0.0)
        self.assertEqual(features["proposal_source_topology_rule"], 1.0)
        self.assertEqual(features["proposal_source_fallback_rule"], 1.0)
        self.assertEqual(features["proposal_source_hybrid_merged"], 1.0)

    def test_role_aware_features_identify_inner_cylindrical_wall(self):
        from utils.instance_candidate_features import extract_instance_candidate_features

        nodes = [
            role_node(0, "cylinder", area=20.0, radius=2.0),
            role_node(1, "plane", area=100.0, normal=[0.0, 0.0, 1.0]),
            role_node(2, "plane", area=100.0, normal=[0.0, 0.0, -1.0]),
        ]
        edges = []
        edge_attrs = []
        for pair, attr in [
            role_edge(0, 1, loop_role="inner_loop", wall_role="inner_hole_wall"),
            role_edge(1, 0, loop_role="inner_loop", wall_role="inner_hole_wall"),
            role_edge(0, 2, loop_role="inner_loop", wall_role="inner_hole_wall"),
            role_edge(2, 0, loop_role="inner_loop", wall_role="inner_hole_wall"),
        ]:
            edges.append(pair)
            edge_attrs.append(attr)

        graph = {"nodes": nodes, "edges": edges, "edge_attributes": edge_attrs}
        candidate = {"source": "fallback_rule", "proposal_macro": "through_hole_round", "face_indices": [0]}

        features = extract_instance_candidate_features(graph, candidate, candidate_macro="through_hole_round")

        self.assertEqual(features["boundary_unique_circle_count"], 2.0)
        self.assertEqual(features["boundary_loop_role_inner_count"], 2.0)
        self.assertEqual(features["boundary_loop_role_outer_count"], 0.0)
        self.assertEqual(features["boundary_cylinder_role_inner_count"], 2.0)
        self.assertEqual(features["boundary_cylinder_role_outer_count"], 0.0)
        self.assertAlmostEqual(features["candidate_boundary_inner_wall_face_fraction"], 1.0)
        self.assertAlmostEqual(features["candidate_boundary_outer_wall_face_fraction"], 0.0)
        self.assertEqual(features["topology_valid"], 1.0)

    def test_role_aware_features_identify_outer_cylindrical_wall(self):
        from utils.instance_candidate_features import extract_instance_candidate_features

        nodes = [
            role_node(0, "cylinder", area=30.0, radius=3.0),
            role_node(1, "plane", area=100.0, normal=[0.0, 0.0, 1.0]),
            role_node(2, "plane", area=100.0, normal=[0.0, 0.0, -1.0]),
        ]
        edges = []
        edge_attrs = []
        for pair, attr in [
            role_edge(0, 1, loop_role="outer_loop", wall_role="outer_cylindrical_wall"),
            role_edge(1, 0, loop_role="outer_loop", wall_role="outer_cylindrical_wall"),
            role_edge(0, 2, loop_role="outer_loop", wall_role="outer_cylindrical_wall"),
            role_edge(2, 0, loop_role="outer_loop", wall_role="outer_cylindrical_wall"),
        ]:
            edges.append(pair)
            edge_attrs.append(attr)

        graph = {"nodes": nodes, "edges": edges, "edge_attributes": edge_attrs}
        candidate = {"source": "fallback_rule", "proposal_macro": "through_hole_round", "face_indices": [0]}

        features = extract_instance_candidate_features(graph, candidate, candidate_macro="through_hole_round")

        self.assertEqual(features["boundary_unique_circle_count"], 2.0)
        self.assertEqual(features["boundary_loop_role_inner_count"], 0.0)
        self.assertEqual(features["boundary_loop_role_outer_count"], 2.0)
        self.assertEqual(features["boundary_cylinder_role_inner_count"], 0.0)
        self.assertEqual(features["boundary_cylinder_role_outer_count"], 2.0)
        self.assertAlmostEqual(features["candidate_boundary_inner_wall_face_fraction"], 0.0)
        self.assertAlmostEqual(features["candidate_boundary_outer_wall_face_fraction"], 1.0)

    def test_topology_reason_features_explain_invalid_through_hole_candidate(self):
        from utils.instance_candidate_features import extract_instance_candidate_features

        nodes = [
            role_node(0, "cylinder", area=20.0, radius=2.0),
            role_node(1, "plane", area=100.0, normal=[0.0, 0.0, 1.0]),
        ]
        pair, attr = role_edge(0, 1, loop_role="inner_loop", wall_role="inner_hole_wall")
        graph = {"nodes": nodes, "edges": [pair], "edge_attributes": [attr]}
        candidate = {"source": "fallback_rule", "proposal_macro": "through_hole_round", "face_indices": [0]}

        features = extract_instance_candidate_features(graph, candidate, candidate_macro="through_hole_round")

        self.assertEqual(features["topology_valid"], 0.0)
        self.assertGreaterEqual(features["topology_reason_count"], 1.0)
        self.assertEqual(features["topology_reason_requires_two_circular_opening_edges"], 1.0)

if __name__ == "__main__":
    unittest.main()
