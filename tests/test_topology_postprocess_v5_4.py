import unittest

from tests.test_topology_postprocess import face, graph, graph_node
from utils.topology_postprocess import apply_topology_postprocess


class TopologyPostprocessV54Test(unittest.TestCase):
    def test_stock_cylinder_with_two_circular_openings_is_promoted_to_through_hole(self):
        graph_data = graph(
            [
                graph_node(0, "cylinder", axis=[0, 0, 1], radius=8.0, descriptors={"area": 502.65}),
                graph_node(1, "plane", normal=[0, 0, 1], descriptors={"area": 2800.0}),
                graph_node(2, "plane", normal=[0, 0, -1], descriptors={"area": 2800.0}),
            ],
            [
                (0, 1, "circle", "smooth", 50.27),
                (1, 0, "circle", "smooth", 50.27),
                (0, 2, "circle", "smooth", 50.27),
                (2, 0, "circle", "smooth", 50.27),
            ],
        )
        prediction = {"sample_id": "sample", "faces": [face(0, macro_id=0, macro="stock")]}

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual(corrected["faces"][0]["macro_feature"], "through_hole_round")
        self.assertEqual(corrected["faces"][0]["semantic_subtype"], "through_hole_cylinder")

    def test_through_hole_with_inner_loop_roles_is_preserved(self):
        graph_data = {
            "nodes": [
                graph_node(0, "cylinder", axis=[0, 0, 1], radius=42.5, descriptors={"area": 2670.35, "uv_u_span": 6.283}),
                graph_node(1, "plane", normal=[0, 0, 1], descriptors={"area": 8811.38}),
                graph_node(2, "plane", normal=[0, 0, -1], descriptors={"area": 8811.38}),
            ],
            "edges": [[0, 1], [1, 0], [0, 2], [2, 0]],
            "edge_attributes": [
                {"source_face": 0, "target_face": 1, "curve_type": "circle", "curve_type_id": 2, "is_circle": 1.0, "convexity": "smooth", "convexity_id": 3, "length": 267.04, "adjacent_plane_loop_role": "inner_loop", "adjacent_plane_loop_role_id": 1, "adjacent_plane_loop_role_inner": 1.0, "cylindrical_wall_role_hint": "inner_hole_wall", "cylindrical_wall_role_hint_id": 1, "cylindrical_wall_role_inner_hint": 1.0},
                {"source_face": 1, "target_face": 0, "curve_type": "circle", "curve_type_id": 2, "is_circle": 1.0, "convexity": "smooth", "convexity_id": 3, "length": 267.04, "adjacent_plane_loop_role": "inner_loop", "adjacent_plane_loop_role_id": 1, "adjacent_plane_loop_role_inner": 1.0, "cylindrical_wall_role_hint": "inner_hole_wall", "cylindrical_wall_role_hint_id": 1, "cylindrical_wall_role_inner_hint": 1.0},
                {"source_face": 0, "target_face": 2, "curve_type": "circle", "curve_type_id": 2, "is_circle": 1.0, "convexity": "smooth", "convexity_id": 3, "length": 267.04, "adjacent_plane_loop_role": "inner_loop", "adjacent_plane_loop_role_id": 1, "adjacent_plane_loop_role_inner": 1.0, "cylindrical_wall_role_hint": "inner_hole_wall", "cylindrical_wall_role_hint_id": 1, "cylindrical_wall_role_inner_hint": 1.0},
                {"source_face": 2, "target_face": 0, "curve_type": "circle", "curve_type_id": 2, "is_circle": 1.0, "convexity": "smooth", "convexity_id": 3, "length": 267.04, "adjacent_plane_loop_role": "inner_loop", "adjacent_plane_loop_role_id": 1, "adjacent_plane_loop_role_inner": 1.0, "cylindrical_wall_role_hint": "inner_hole_wall", "cylindrical_wall_role_hint_id": 1, "cylindrical_wall_role_inner_hint": 1.0},
            ],
        }
        prediction = {"sample_id": "sample", "faces": [face(0, macro_id=1, macro="through_hole_round")]}

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual(corrected["faces"][0]["macro_feature"], "through_hole_round")

    def test_exterior_cylinder_with_outer_loop_roles_is_demoted_from_through_hole(self):
        graph_data = {
            "nodes": [
                graph_node(0, "cylinder", axis=[0, 0, 1], radius=60.0, descriptors={"area": 3769.91, "uv_u_span": 6.283}),
                graph_node(1, "plane", normal=[0, 0, 1], descriptors={"area": 8771.33}),
                graph_node(2, "plane", normal=[0, 0, -1], descriptors={"area": 9977.70}),
            ],
            "edges": [[0, 1], [1, 0], [0, 2], [2, 0]],
            "edge_attributes": [
                {"source_face": 0, "target_face": 1, "curve_type": "circle", "curve_type_id": 2, "is_circle": 1.0, "convexity": "smooth", "convexity_id": 3, "length": 376.99, "adjacent_plane_loop_role": "outer_loop", "adjacent_plane_loop_role_id": 2, "adjacent_plane_loop_role_outer": 1.0, "cylindrical_wall_role_hint": "outer_cylinder_wall", "cylindrical_wall_role_hint_id": 2, "cylindrical_wall_role_outer_hint": 1.0},
                {"source_face": 1, "target_face": 0, "curve_type": "circle", "curve_type_id": 2, "is_circle": 1.0, "convexity": "smooth", "convexity_id": 3, "length": 376.99, "adjacent_plane_loop_role": "outer_loop", "adjacent_plane_loop_role_id": 2, "adjacent_plane_loop_role_outer": 1.0, "cylindrical_wall_role_hint": "outer_cylinder_wall", "cylindrical_wall_role_hint_id": 2, "cylindrical_wall_role_outer_hint": 1.0},
                {"source_face": 0, "target_face": 2, "curve_type": "circle", "curve_type_id": 2, "is_circle": 1.0, "convexity": "smooth", "convexity_id": 3, "length": 376.99, "adjacent_plane_loop_role": "outer_loop", "adjacent_plane_loop_role_id": 2, "adjacent_plane_loop_role_outer": 1.0, "cylindrical_wall_role_hint": "outer_cylinder_wall", "cylindrical_wall_role_hint_id": 2, "cylindrical_wall_role_outer_hint": 1.0},
                {"source_face": 2, "target_face": 0, "curve_type": "circle", "curve_type_id": 2, "is_circle": 1.0, "convexity": "smooth", "convexity_id": 3, "length": 376.99, "adjacent_plane_loop_role": "outer_loop", "adjacent_plane_loop_role_id": 2, "adjacent_plane_loop_role_outer": 1.0, "cylindrical_wall_role_hint": "outer_cylinder_wall", "cylindrical_wall_role_hint_id": 2, "cylindrical_wall_role_outer_hint": 1.0},
            ],
        }
        prediction = {"sample_id": "sample", "faces": [face(0, macro_id=1, macro="through_hole_round")]}

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual(corrected["faces"][0]["macro_feature"], "stock")
        self.assertIn(
            "through_hole_functional_opening_role_required",
            corrected["topology_postprocess"]["invalid_components"][0]["reasons"],
        )

    def test_exterior_cylinder_and_chamfer_cone_are_not_repaired_to_countersink(self):
        graph_data = graph(
            [
                graph_node(0, "cylinder", axis=[0, 0, 1], radius=70.0, descriptors={"area": 3518.58}),
                graph_node(1, "cone", axis=[0, 0, 1], descriptors={"area": 617.56}),
                graph_node(2, "plane", normal=[0, 0, 1], descriptors={"area": 8811.38}),
                graph_node(3, "torus", descriptors={"area": 687.29}),
            ],
            [
                (0, 1, "circle", "smooth", 439.82),
                (1, 0, "circle", "smooth", 439.82),
                (0, 3, "circle", "smooth", 439.82),
                (3, 0, "circle", "smooth", 439.82),
                (1, 2, "circle", "smooth", 433.54),
                (2, 1, "circle", "smooth", 433.54),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(0, macro_id=1, macro="through_hole_round"),
                face(1, macro_id=26, macro="chamfer"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertNotEqual([item["macro_feature"] for item in corrected["faces"]], ["countersink_hole", "countersink_hole"])
        self.assertEqual(corrected["faces"][1]["macro_feature"], "chamfer")

    def test_huge_exterior_cylinder_is_not_repaired_to_single_face_o_slot(self):
        graph_data = graph(
            [
                graph_node(0, "cylinder", axis=[0, 1, 0], radius=100.0, descriptors={"area": 90400.97}),
                graph_node(1, "plane", normal=[0, 0, 1], descriptors={"area": 3397.72}),
                graph_node(2, "plane", normal=[0, 0, -1], descriptors={"area": 3397.72}),
                graph_node(3, "sphere", descriptors={"area": 62831.85}),
            ],
            [
                (0, 1, "line", "convex", 145.0),
                (1, 0, "line", "convex", 145.0),
                (0, 2, "line", "convex", 145.0),
                (2, 0, "line", "convex", 145.0),
                (0, 3, "circle", "smooth", 471.24),
                (3, 0, "circle", "smooth", 471.24),
            ],
        )
        prediction = {"sample_id": "sample", "faces": [face(0, macro_id=0, macro="stock")]}

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual(corrected["faces"][0]["macro_feature"], "stock")


    def test_partial_cylindrical_transition_with_two_planes_is_not_promoted_to_through_hole(self):
        graph_data = graph(
            [
                graph_node(
                    0,
                    "cylinder",
                    axis=[0, 1, 0],
                    radius=20.0,
                    descriptors={"area": 476.41, "uv_u_span": 2.38},
                ),
                graph_node(1, "plane", normal=[0, 0, 1], descriptors={"area": 2800.0}),
                graph_node(2, "plane", normal=[0, 0, -1], descriptors={"area": 2800.0}),
                graph_node(3, "plane", normal=[1, 0, 0], descriptors={"area": 476.0}),
                graph_node(4, "plane", normal=[-1, 0, 0], descriptors={"area": 476.0}),
            ],
            [
                (0, 1, "circle", "smooth", 47.64),
                (1, 0, "circle", "smooth", 47.64),
                (0, 2, "circle", "smooth", 47.64),
                (2, 0, "circle", "smooth", 47.64),
                (0, 3, "line", "convex", 10.0),
                (3, 0, "line", "convex", 10.0),
                (0, 4, "line", "convex", 10.0),
                (4, 0, "line", "convex", 10.0),
            ],
        )
        prediction = {"sample_id": "sample", "faces": [face(0, macro_id=0, macro="stock")]}

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual(corrected["faces"][0]["macro_feature"], "stock")

    def test_predicted_through_hole_with_transition_end_is_demoted_to_stock(self):
        graph_data = graph(
            [
                graph_node(0, "cylinder", axis=[0, 0, 1], radius=25.0, descriptors={"area": 13194.69, "uv_u_span": 6.283}),
                graph_node(1, "cone", axis=[0, 0, 1], descriptors={"area": 706.42}),
                graph_node(2, "torus", descriptors={"area": 772.50}),
            ],
            [
                (0, 1, "circle", "smooth", 157.08),
                (1, 0, "circle", "smooth", 157.08),
                (0, 2, "circle", "smooth", 157.08),
                (2, 0, "circle", "smooth", 157.08),
            ],
        )
        prediction = {"sample_id": "sample", "faces": [face(0, macro_id=1, macro="through_hole_round")]}

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual(corrected["faces"][0]["macro_feature"], "stock")

    def test_rib_component_splits_full_through_hole_cylinder(self):
        graph_data = graph(
            [
                graph_node(10, "plane", normal=[1, 0, 0], descriptors={"area": 180.93, "center": [48.75, 21.27, 7.24], "projected_aspect_ratio": 28.95, "projected_long_extent": 72.37, "projected_short_extent": 2.5}),
                graph_node(11, "cylinder", axis=[1, 0, 0], radius=10.0, descriptors={"area": 54.98, "center": [48.75, 41.67, -31.59], "uv_u_span": 2.199}),
                graph_node(12, "plane", normal=[1, 0, 0], descriptors={"area": 86.13, "center": [48.75, 17.23, -37.91], "projected_aspect_ratio": 13.78, "projected_long_extent": 34.45, "projected_short_extent": 2.5}),
                graph_node(13, "cylinder", axis=[1, 0, 0], radius=5.0, descriptors={"area": 78.54, "center": [48.75, 34.45, -27.91], "uv_u_span": 6.283}),
                graph_node(14, "plane", normal=[1, 0, 0], descriptors={"area": 1847.59, "center": [50.0, 16.0, -11.5], "projected_aspect_ratio": 1.81, "projected_long_extent": 82.0, "projected_short_extent": 45.2}),
                graph_node(15, "plane", normal=[1, 0, 0], descriptors={"area": 1847.59, "center": [47.5, 16.0, -11.5], "projected_aspect_ratio": 1.81, "projected_long_extent": 82.0, "projected_short_extent": 45.2}),
            ],
            [
                (10, 14, "line", "convex", 72.37), (14, 10, "line", "convex", 72.37),
                (10, 15, "line", "convex", 72.37), (15, 10, "line", "convex", 72.37),
                (10, 11, "line", "convex", 2.5), (11, 10, "line", "convex", 2.5),
                (11, 12, "line", "convex", 2.5), (12, 11, "line", "convex", 2.5),
                (11, 14, "circle", "smooth", 21.99), (14, 11, "circle", "smooth", 21.99),
                (11, 15, "circle", "smooth", 21.99), (15, 11, "circle", "smooth", 21.99),
                (12, 14, "line", "convex", 34.45), (14, 12, "line", "convex", 34.45),
                (12, 15, "line", "convex", 34.45), (15, 12, "line", "convex", 34.45),
                (13, 14, "circle", "smooth", 31.42), (14, 13, "circle", "smooth", 31.42),
                (13, 15, "circle", "smooth", 31.42), (15, 13, "circle", "smooth", 31.42),
            ],
        )
        prediction = {"sample_id": "sample", "faces": [face(idx, macro_id=20, macro="rib") for idx in [10, 11, 12, 13, 14, 15]]}

        corrected = apply_topology_postprocess(prediction, graph_data)
        by_idx = {item["idx"]: item for item in corrected["faces"]}

        self.assertEqual(by_idx[13]["macro_feature"], "through_hole_round")
        self.assertEqual(by_idx[13]["semantic_subtype"], "through_hole_cylinder")
        self.assertEqual({by_idx[idx]["macro_feature"] for idx in [10, 11, 12, 14, 15]}, {"rib"})

    def test_stock_parallel_planes_with_convex_transition_cap_are_repaired_to_rib(self):
        graph_data = graph(
            [
                graph_node(10, "cone", descriptors={"area": 197.7, "center": [0.0, 0.0, 0.0]}),
                graph_node(11, "plane", normal=[1, 0, 0], descriptors={"area": 1010.14, "center": [0.0, 0.0, 0.0], "projected_aspect_ratio": 5.15, "projected_long_extent": 102.5, "projected_short_extent": 19.9}),
                graph_node(12, "plane", normal=[1, 0, 0], descriptors={"area": 1010.14, "center": [1.8, 0.0, 0.0], "projected_aspect_ratio": 5.15, "projected_long_extent": 102.5, "projected_short_extent": 19.9}),
            ],
            [
                (10, 11, "line", "convex", 102.5), (11, 10, "line", "convex", 102.5),
                (10, 12, "line", "convex", 102.5), (12, 10, "line", "convex", 102.5),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(10, macro_id=25, macro="fillet"),
                face(11, macro_id=0, macro="stock"),
                face(12, macro_id=0, macro="stock"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual({item["macro_feature"] for item in corrected["faces"]}, {"rib"})

    def test_web_rib_body_absorbs_small_stock_end_caps_and_outputs_rib(self):
        graph_data = graph(
            [
                graph_node(47, "plane", normal=[0, 1, 0], descriptors={"area": 24.375, "center": [0, 0, 0], "projected_aspect_ratio": 10.8, "projected_long_extent": 16.25, "projected_short_extent": 1.5}),
                graph_node(48, "plane", normal=[0, 1, 0], descriptors={"area": 76.875, "center": [0, 1.5, 0], "projected_aspect_ratio": 34.0, "projected_long_extent": 51.25, "projected_short_extent": 1.5}),
                graph_node(49, "plane", normal=[0, 1, 0], descriptors={"area": 76.875, "center": [0, 2.0, 0], "projected_aspect_ratio": 34.0, "projected_long_extent": 51.25, "projected_short_extent": 1.5}),
                graph_node(50, "plane", normal=[0, 1, 0], descriptors={"area": 24.375, "center": [0, 3.5, 0], "projected_aspect_ratio": 10.8, "projected_long_extent": 16.25, "projected_short_extent": 1.5}),
                graph_node(55, "plane", normal=[0, 1, 0], descriptors={"area": 67.75, "center": [0, 1.75, 0], "projected_aspect_ratio": 3.1, "projected_long_extent": 103.0, "projected_short_extent": 33.0}),
            ],
            [
                (47, 48, "line", "concave", 1.5), (48, 47, "line", "concave", 1.5),
                (47, 55, "line", "convex", 16.25), (55, 47, "line", "convex", 16.25),
                (48, 55, "line", "convex", 51.25), (55, 48, "line", "convex", 51.25),
                (49, 55, "line", "convex", 51.25), (55, 49, "line", "convex", 51.25),
                (49, 50, "line", "concave", 1.5), (50, 49, "line", "concave", 1.5),
                (50, 55, "line", "convex", 16.25), (55, 50, "line", "convex", 16.25),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(47, macro_id=11, macro="pocket_rect"),
                face(48, macro_id=21, macro="web"),
                face(49, macro_id=21, macro="web"),
                face(50, macro_id=11, macro="pocket_rect"),
                face(55, macro_id=21, macro="web"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual({item["macro_feature"] for item in corrected["faces"]}, {"rib"})

    def test_pocket_embedded_thin_wall_faces_are_split_to_rib(self):
        graph_data = graph(
            [
                graph_node(21, "plane", normal=[0, 0, 1], descriptors={"area": 180.0, "center": [0, 0, 0], "projected_aspect_ratio": 9.0, "projected_long_extent": 45.0, "projected_short_extent": 5.0}),
                graph_node(22, "plane", normal=[0, 0, 1], descriptors={"area": 220.0, "center": [0, 10, 0], "projected_aspect_ratio": 22.0, "projected_long_extent": 66.0, "projected_short_extent": 3.0}),
                graph_node(26, "plane", normal=[0, 1, 0], descriptors={"area": 40.0, "center": [0, 0, 1.0], "projected_aspect_ratio": 12.0, "projected_long_extent": 24.0, "projected_short_extent": 2.0}),
                graph_node(27, "plane", normal=[0, 1, 0], descriptors={"area": 35.0, "center": [3, 0, 1.0], "projected_aspect_ratio": 10.0, "projected_long_extent": 20.0, "projected_short_extent": 2.0}),
                graph_node(28, "plane", normal=[0, 1, 0], descriptors={"area": 38.0, "center": [6, 0, 1.0], "projected_aspect_ratio": 11.0, "projected_long_extent": 22.0, "projected_short_extent": 2.0}),
                graph_node(29, "plane", normal=[0, 1, 0], descriptors={"area": 36.0, "center": [9, 0, 1.0], "projected_aspect_ratio": 10.0, "projected_long_extent": 20.0, "projected_short_extent": 2.0}),
                graph_node(30, "plane", normal=[0, 1, 0], descriptors={"area": 34.0, "center": [12, 0, 1.0], "projected_aspect_ratio": 10.0, "projected_long_extent": 20.0, "projected_short_extent": 2.0}),
                graph_node(46, "plane", normal=[0, 1, 0], descriptors={"area": 80.0, "center": [6.0, 0.5, 1.0], "projected_aspect_ratio": 3.0, "projected_long_extent": 50.0, "projected_short_extent": 16.0}),
            ],
            [
                (46, 26, "line", "convex", 20.0), (26, 46, "line", "convex", 20.0),
                (46, 27, "line", "convex", 20.0), (27, 46, "line", "convex", 20.0),
                (46, 28, "line", "convex", 20.0), (28, 46, "line", "convex", 20.0),
                (46, 29, "line", "convex", 20.0), (29, 46, "line", "convex", 20.0),
                (46, 30, "line", "convex", 20.0), (30, 46, "line", "convex", 20.0),
                (26, 27, "line", "concave", 2.0), (27, 26, "line", "concave", 2.0),
                (27, 28, "line", "concave", 2.0), (28, 27, "line", "concave", 2.0),
                (28, 29, "line", "concave", 2.0), (29, 28, "line", "concave", 2.0),
                (29, 30, "line", "concave", 2.0), (30, 29, "line", "concave", 2.0),
                (46, 21, "line", "convex", 1.0), (21, 46, "line", "convex", 1.0),
                (46, 22, "line", "convex", 1.0), (22, 46, "line", "convex", 1.0),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(21, macro_id=11, macro="pocket_rect"),
                face(22, macro_id=11, macro="pocket_rect"),
                face(26, macro_id=11, macro="pocket_rect"),
                face(27, macro_id=11, macro="pocket_rect"),
                face(28, macro_id=11, macro="pocket_rect"),
                face(29, macro_id=11, macro="pocket_rect"),
                face(30, macro_id=11, macro="pocket_rect"),
                face(46, macro_id=0, macro="stock"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)
        by_idx = {item["idx"]: item for item in corrected["faces"]}

        self.assertEqual({by_idx[idx]["macro_feature"] for idx in [26, 27, 28, 29, 30, 46]}, {"rib"})
        self.assertNotEqual(by_idx[21]["macro_feature"], "rib")
        self.assertNotEqual(by_idx[22]["macro_feature"], "rib")

    def test_extremely_thin_bottom_component_is_not_kept_as_rectangular_pocket(self):
        graph_data = graph(
            [
                graph_node(24, "plane", normal=[0, 0, 1], descriptors={"area": 294.0, "projected_aspect_ratio": 32.9, "projected_long_extent": 99.0, "projected_short_extent": 3.0}),
                graph_node(7, "plane", normal=[1, 0, 0], descriptors={"area": 357.0, "projected_aspect_ratio": 1.2}),
                graph_node(8, "plane", normal=[-1, 0, 0], descriptors={"area": 357.0, "projected_aspect_ratio": 1.2}),
                graph_node(9, "plane", normal=[0, 1, 0], descriptors={"area": 466.0, "projected_aspect_ratio": 1.8}),
                graph_node(10, "plane", normal=[0, -1, 0], descriptors={"area": 426.0, "projected_aspect_ratio": 1.6}),
            ],
            [
                (24, 7, "line", "concave"), (7, 24, "line", "concave"),
                (24, 8, "line", "concave"), (8, 24, "line", "concave"),
                (24, 9, "line", "concave"), (9, 24, "line", "concave"),
                (24, 10, "line", "concave"), (10, 24, "line", "concave"),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [face(idx, macro_id=11, macro="pocket_rect") for idx in [24, 7, 8, 9, 10]],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual({item["macro_feature"] for item in corrected["faces"]}, {"stock"})
        self.assertIn(
            "rectangular_pocket_bottom_aspect_ratio_too_high",
            corrected["topology_postprocess"]["invalid_components"][0]["reasons"],
        )

    def test_full_through_cylinder_is_not_absorbed_as_u_slot_bottom(self):
        graph_data = graph(
            [
                graph_node(0, "cylinder", axis=[0, 0, 1], radius=11.2, descriptors={"area": 7395.21, "uv_u_span": 6.283}),
                graph_node(17, "plane", normal=[1, 0, 0], descriptors={"area": 197.4, "projected_aspect_ratio": 20.0, "projected_long_extent": 50.0, "projected_short_extent": 2.5}),
                graph_node(19, "plane", normal=[-1, 0, 0], descriptors={"area": 197.4, "projected_aspect_ratio": 20.0, "projected_long_extent": 50.0, "projected_short_extent": 2.5}),
                graph_node(1, "plane", normal=[0, 0, 1], descriptors={"area": 5000.0}),
                graph_node(2, "plane", normal=[0, 0, -1], descriptors={"area": 5000.0}),
            ],
            [
                (0, 17, "line", "concave", 50.0), (17, 0, "line", "concave", 50.0),
                (0, 19, "line", "concave", 50.0), (19, 0, "line", "concave", 50.0),
                (0, 1, "circle", "smooth", 70.4), (1, 0, "circle", "smooth", 70.4),
                (0, 2, "circle", "smooth", 70.4), (2, 0, "circle", "smooth", 70.4),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(0, macro_id=1, macro="through_hole_round"),
                face(17, macro_id=10, macro="slot_rect"),
                face(19, macro_id=10, macro="slot_rect"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)
        by_idx = {item["idx"]: item for item in corrected["faces"]}

        self.assertEqual(by_idx[0]["macro_feature"], "through_hole_round")
        self.assertNotEqual({by_idx[idx]["macro_feature"] for idx in [0, 17, 19]}, {"slot_u"})

    def test_part11_like_overmerged_stepped_component_splits_to_counterbores_and_v_slot(self):
        graph_data = graph(
            [
                graph_node(7, "cylinder", axis=[0, 1, 0], radius=3.0, descriptors={"area": 84.82}),
                graph_node(8, "cylinder", axis=[0, 1, 0], radius=1.75, descriptors={"area": 38.48}),
                graph_node(9, "cylinder", axis=[0, 1, 0], radius=3.0, descriptors={"area": 84.82}),
                graph_node(10, "cylinder", axis=[0, 1, 0], radius=1.75, descriptors={"area": 38.48}),
                graph_node(11, "plane", normal=[0, -1, 0], descriptors={"area": 544.81}),
                graph_node(13, "plane", normal=[0, -1, 0], descriptors={"area": 18.65}),
                graph_node(14, "plane", normal=[0, -1, 0], descriptors={"area": 18.65}),
                graph_node(18, "plane", normal=[-0.4, 0.9, 0], descriptors={"area": 47.04, "projected_aspect_ratio": 57.0, "projected_long_extent": 52.0, "projected_short_extent": 0.9}),
                graph_node(19, "plane", normal=[-0.5, -0.8, 0], descriptors={"area": 50.66, "projected_aspect_ratio": 53.0, "projected_long_extent": 52.0, "projected_short_extent": 1.0}),
            ],
            [
                (7, 14, "circle", "smooth", 18.85), (14, 7, "circle", "smooth", 18.85),
                (8, 14, "circle", "smooth", 10.99), (14, 8, "circle", "smooth", 10.99),
                (9, 13, "circle", "smooth", 18.85), (13, 9, "circle", "smooth", 18.85),
                (10, 13, "circle", "smooth", 10.99), (13, 10, "circle", "smooth", 10.99),
                (7, 11, "circle", "smooth", 18.85), (11, 7, "circle", "smooth", 18.85),
                (9, 11, "circle", "smooth", 18.85), (11, 9, "circle", "smooth", 18.85),
                (18, 19, "line", "concave", 52.0, 0.9772), (19, 18, "line", "concave", 52.0, 0.9772),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                *[face(idx, macro_id=3, macro="stepped_hole_round") for idx in [7, 8, 9, 10, 11, 13, 14]],
                face(18, macro_id=26, macro="chamfer"),
                face(19, macro_id=26, macro="chamfer"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)
        by_idx = {item["idx"]: item for item in corrected["faces"]}

        self.assertEqual({by_idx[idx]["macro_feature"] for idx in [7, 8, 14]}, {"counterbore_hole"})
        self.assertEqual({by_idx[idx]["macro_feature"] for idx in [9, 10, 13]}, {"counterbore_hole"})
        self.assertEqual(by_idx[11]["macro_feature"], "stock")
        self.assertEqual({by_idx[idx]["macro_feature"] for idx in [18, 19]}, {"slot_v"})

    def test_through_like_partial_cylinder_with_two_concave_side_planes_is_repaired_to_u_slot(self):
        graph_data = graph(
            [
                graph_node(15, "cylinder", axis=[0, 0, 1], radius=0.52, descriptors={"area": 84.46, "uv_u_span": 3.1416}),
                graph_node(16, "plane", normal=[1, 0, 0], descriptors={"area": 52.0, "projected_aspect_ratio": 52.0, "projected_long_extent": 52.0, "projected_short_extent": 1.0}),
                graph_node(17, "plane", normal=[-1, 0, 0], descriptors={"area": 52.0, "projected_aspect_ratio": 52.0, "projected_long_extent": 52.0, "projected_short_extent": 1.0}),
                graph_node(2, "plane", normal=[0, 0, 1], descriptors={"area": 92.7}),
                graph_node(3, "plane", normal=[0, 0, -1], descriptors={"area": 92.7}),
            ],
            [
                (15, 16, "line", "concave", 52.0), (16, 15, "line", "concave", 52.0),
                (15, 17, "line", "concave", 52.0), (17, 15, "line", "concave", 52.0),
                (15, 2, "circle", "smooth", 1.62), (2, 15, "circle", "smooth", 1.62),
                (15, 3, "circle", "smooth", 1.62), (3, 15, "circle", "smooth", 1.62),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(15, macro_id=1, macro="through_hole_round"),
                face(16, macro_id=0, macro="stock"),
                face(17, macro_id=0, macro="stock"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)
        by_idx = {item["idx"]: item for item in corrected["faces"]}

        self.assertEqual({by_idx[idx]["macro_feature"] for idx in [15, 16, 17]}, {"slot_u"})
        self.assertEqual(by_idx[15]["semantic_subtype"], "slot_u_bottom_plane")
        self.assertEqual(by_idx[16]["semantic_subtype"], "slot_u_side_plane")
        self.assertEqual(by_idx[17]["semantic_subtype"], "slot_u_side_plane")

    def test_overmerged_rectangular_slot_keeps_three_consistent_long_faces(self):
        graph_data = graph(
            [
                graph_node(15, "plane", normal=[0, 0, 1], descriptors={"area": 1167.57, "projected_aspect_ratio": 5.58, "projected_long_extent": 11.16, "projected_short_extent": 2.0}),
                graph_node(17, "plane", normal=[0, 1, 0], descriptors={"area": 197.42, "projected_aspect_ratio": 59.08, "projected_long_extent": 108.0, "projected_short_extent": 1.83}),
                graph_node(18, "plane", normal=[-1, 0, 0], descriptors={"area": 216.0, "projected_aspect_ratio": 54.0, "projected_long_extent": 108.0, "projected_short_extent": 2.0}),
                graph_node(19, "plane", normal=[0, -1, 0], descriptors={"area": 197.42, "projected_aspect_ratio": 59.08, "projected_long_extent": 108.0, "projected_short_extent": 1.83}),
            ],
            [
                (15, 17, "line", "concave", 1.83), (17, 15, "line", "concave", 1.83),
                (15, 18, "line", "concave", 2.0), (18, 15, "line", "concave", 2.0),
                (15, 19, "line", "concave", 1.83), (19, 15, "line", "concave", 1.83),
                (17, 18, "line", "concave", 108.0), (18, 17, "line", "concave", 108.0),
                (18, 19, "line", "concave", 108.0), (19, 18, "line", "concave", 108.0),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [face(idx, macro_id=10, macro="slot_rect") for idx in [15, 17, 18, 19]],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)
        by_idx = {item["idx"]: item for item in corrected["faces"]}

        self.assertEqual(by_idx[15]["macro_feature"], "stock")
        self.assertEqual({by_idx[idx]["macro_feature"] for idx in [17, 18, 19]}, {"slot_rect"})

    def test_wide_predicted_chamfer_without_large_support_is_demoted_to_stock(self):
        graph_data = graph(
            [
                graph_node(4, "plane", normal=[1, 0, 0], descriptors={"area": 68.44, "projected_aspect_ratio": 39.5, "projected_long_extent": 52.0, "projected_short_extent": 1.32}),
                graph_node(2, "plane", normal=[0, 1, 0], descriptors={"area": 92.7}),
                graph_node(19, "plane", normal=[0, -1, 0], descriptors={"area": 50.66}),
                graph_node(22, "plane", normal=[1, 1, 0], descriptors={"area": 14.71}),
            ],
            [
                (4, 2, "line", "convex", 1.32, 1.5708), (2, 4, "line", "convex", 1.32, 1.5708),
                (4, 19, "line", "convex", 52.0, 1.0125), (19, 4, "line", "convex", 52.0, 1.0125),
                (4, 22, "line", "convex", 52.0, 0.7854), (22, 4, "line", "convex", 52.0, 0.7854),
            ],
        )
        prediction = {"sample_id": "sample", "faces": [face(4, macro_id=26, macro="chamfer")]}

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual(corrected["faces"][0]["macro_feature"], "stock")
        self.assertIn(
            "chamfer_requires_large_adjacent_support_face",
            corrected["topology_postprocess"]["invalid_components"][0]["reasons"],
        )

    def test_narrow_predicted_chamfer_with_large_support_is_preserved(self):
        graph_data = graph(
            [
                graph_node(21, "plane", normal=[1, 1, 0], descriptors={"area": 14.71, "projected_aspect_ratio": 183.0, "projected_long_extent": 52.0, "projected_short_extent": 0.28}),
                graph_node(11, "plane", normal=[0, 1, 0], descriptors={"area": 544.8}),
                graph_node(2, "plane", normal=[1, 0, 0], descriptors={"area": 92.7}),
                graph_node(3, "plane", normal=[-1, 0, 0], descriptors={"area": 92.7}),
                graph_node(12, "plane", normal=[0, -1, 0], descriptors={"area": 544.8}),
            ],
            [
                (21, 11, "line", "convex", 52.0, 0.7854), (11, 21, "line", "convex", 52.0, 0.7854),
                (21, 12, "line", "convex", 52.0, 0.7854), (12, 21, "line", "convex", 52.0, 0.7854),
                (21, 2, "line", "convex", 0.28, 1.5708), (2, 21, "line", "convex", 0.28, 1.5708),
                (21, 3, "line", "convex", 0.28, 1.5708), (3, 21, "line", "convex", 0.28, 1.5708),
            ],
        )
        prediction = {"sample_id": "sample", "faces": [face(21, macro_id=26, macro="chamfer")]}

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual(corrected["faces"][0]["macro_feature"], "chamfer")

    def test_invalid_single_stepped_small_partial_cylinder_is_repaired_to_fillet(self):
        graph_data = graph(
            [
                graph_node(60, "cylinder", axis=[0, 0, 1], radius=2.0, descriptors={"area": 9.4248, "center": [100.77, 0.77, -22.5], "uv_u_span": 1.5708}),
                graph_node(8, "plane", normal=[0, 0, 1], descriptors={"area": 357.16}),
                graph_node(19, "plane", normal=[0, 0, -1], descriptors={"area": 705.0}),
                graph_node(23, "plane", normal=[1, 0, 0], descriptors={"area": 86.0}),
                graph_node(24, "plane", normal=[0, 1, 0], descriptors={"area": 294.0}),
            ],
            [
                (60, 8, "circle", "smooth", 3.142), (8, 60, "circle", "smooth", 3.142),
                (60, 19, "circle", "smooth", 3.142), (19, 60, "circle", "smooth", 3.142),
                (60, 23, "line", "concave", 3.0), (23, 60, "line", "concave", 3.0),
                (60, 24, "line", "concave", 3.0), (24, 60, "line", "concave", 3.0),
            ],
        )
        stepped_face = face(60, macro_id=3, macro="stepped_hole_round")
        stepped_face["macro_confidence"] = 0.55
        prediction = {"sample_id": "sample", "faces": [stepped_face]}

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual(corrected["faces"][0]["macro_feature"], "fillet")
        self.assertEqual(corrected["faces"][0]["semantic_subtype"], "fillet_blend_surface")

    def test_invalid_single_stepped_full_cylinder_is_still_repaired_to_through_hole(self):
        graph_data = graph(
            [
                graph_node(0, "cylinder", axis=[0, 0, 1], radius=5.0, descriptors={"area": 314.159, "uv_u_span": 6.283}),
                graph_node(1, "plane", normal=[0, 0, 1], descriptors={"area": 2000.0}),
                graph_node(2, "plane", normal=[0, 0, -1], descriptors={"area": 2000.0}),
            ],
            [
                (0, 1, "circle", "smooth", 31.416), (1, 0, "circle", "smooth", 31.416),
                (0, 2, "circle", "smooth", 31.416), (2, 0, "circle", "smooth", 31.416),
            ],
        )
        prediction = {"sample_id": "sample", "faces": [face(0, macro_id=3, macro="stepped_hole_round")]}

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual(corrected["faces"][0]["macro_feature"], "through_hole_round")
        self.assertEqual(corrected["faces"][0]["semantic_subtype"], "through_hole_cylinder")

if __name__ == "__main__":
    unittest.main()


