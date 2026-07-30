import unittest

from tests.test_topology_postprocess import face, graph, graph_node
from utils.topology_postprocess import apply_topology_postprocess


def circular_edge(src, dst, *, role="unknown", wall_role="unknown", convexity="smooth", length=31.42):
    role_id = {"unknown": 0, "inner_loop": 1, "outer_loop": 2}[role]
    wall_role_id = {"unknown": 0, "inner_hole_wall": 1, "outer_cylinder_wall": 2}[wall_role]
    return {
        "source_face": src,
        "target_face": dst,
        "curve_type": "circle",
        "curve_type_id": 2,
        "is_circle": 1.0,
        "convexity": convexity,
        "convexity_id": {"convex": 1, "concave": 2, "smooth": 3}[convexity],
        "length": length,
        "adjacent_plane_loop_role": role,
        "adjacent_plane_loop_role_id": role_id,
        "adjacent_plane_loop_role_inner": 1.0 if role == "inner_loop" else 0.0,
        "adjacent_plane_loop_role_outer": 1.0 if role == "outer_loop" else 0.0,
        "cylindrical_wall_role_hint": wall_role,
        "cylindrical_wall_role_hint_id": wall_role_id,
        "cylindrical_wall_role_inner_hint": 1.0 if wall_role == "inner_hole_wall" else 0.0,
        "cylindrical_wall_role_outer_hint": 1.0 if wall_role == "outer_cylinder_wall" else 0.0,
    }


def role_graph(nodes, edge_specs):
    edges = []
    attrs = []
    for src, dst, kwargs in edge_specs:
        edges.append([src, dst])
        attrs.append(circular_edge(src, dst, **kwargs))
    return {"nodes": nodes, "edges": edges, "edge_attributes": attrs}


class TopologyPostprocessV56Test(unittest.TestCase):
    def test_inner_cylinder_boss_prediction_with_bottom_face_is_repaired_to_blind_hole(self):
        graph_data = role_graph(
            [
                graph_node(0, "cylinder", axis=[0, 0, 1], radius=5.0, descriptors={"area": 314.16, "uv_u_span": 6.283, "center": [0, 0, -5]}),
                graph_node(1, "plane", normal=[0, 0, 1], descriptors={"area": 78.54, "center": [0, 0, -10]}),
                graph_node(2, "plane", normal=[0, 0, 1], descriptors={"area": 4000.0, "center": [0, 0, 0]}),
            ],
            [
                (0, 1, {"role": "unknown", "wall_role": "inner_hole_wall", "convexity": "concave"}),
                (1, 0, {"role": "unknown", "wall_role": "inner_hole_wall", "convexity": "concave"}),
                (0, 2, {"role": "inner_loop", "wall_role": "inner_hole_wall", "convexity": "smooth"}),
                (2, 0, {"role": "inner_loop", "wall_role": "inner_hole_wall", "convexity": "smooth"}),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(0, macro_id=14, macro="cylinder_boss"),
                face(1, macro_id=14, macro="cylinder_boss"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)
        by_idx = {item["idx"]: item for item in corrected["faces"]}

        self.assertEqual({by_idx[idx]["macro_feature"] for idx in [0, 1]}, {"blind_hole_round"})
        self.assertEqual(by_idx[0]["semantic_subtype"], "blind_hole_cylinder")
        self.assertEqual(by_idx[1]["semantic_subtype"], "blind_hole_bottom")

    def test_inner_through_hole_prediction_as_cylinder_boss_is_repaired_to_through_hole(self):
        graph_data = role_graph(
            [
                graph_node(0, "cylinder", axis=[0, 0, 1], radius=4.0, descriptors={"area": 251.33, "uv_u_span": 6.283}),
                graph_node(1, "plane", normal=[0, 0, 1], descriptors={"area": 3000.0, "center": [0, 0, 0]}),
                graph_node(2, "plane", normal=[0, 0, -1], descriptors={"area": 3000.0, "center": [0, 0, 10]}),
            ],
            [
                (0, 1, {"role": "inner_loop", "wall_role": "inner_hole_wall", "convexity": "smooth"}),
                (1, 0, {"role": "inner_loop", "wall_role": "inner_hole_wall", "convexity": "smooth"}),
                (0, 2, {"role": "inner_loop", "wall_role": "inner_hole_wall", "convexity": "smooth"}),
                (2, 0, {"role": "inner_loop", "wall_role": "inner_hole_wall", "convexity": "smooth"}),
            ],
        )
        prediction = {"sample_id": "sample", "faces": [face(0, macro_id=14, macro="cylinder_boss")]}

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual(corrected["faces"][0]["macro_feature"], "through_hole_round")
        self.assertEqual(corrected["faces"][0]["semantic_subtype"], "through_hole_cylinder")

    def test_outer_loop_multi_radius_cylinders_are_not_kept_as_counterbore(self):
        graph_data = role_graph(
            [
                graph_node(0, "cylinder", axis=[0, 0, 1], radius=8.0, descriptors={"area": 502.65, "uv_u_span": 6.283}),
                graph_node(1, "plane", normal=[0, 0, 1], descriptors={"area": 150.0}),
                graph_node(2, "cylinder", axis=[0, 0, 1], radius=4.0, descriptors={"area": 251.33, "uv_u_span": 6.283}),
                graph_node(3, "plane", normal=[0, 0, 1], descriptors={"area": 4000.0}),
            ],
            [
                (0, 1, {"role": "outer_loop", "wall_role": "outer_cylinder_wall", "convexity": "smooth", "length": 50.27}),
                (1, 0, {"role": "outer_loop", "wall_role": "outer_cylinder_wall", "convexity": "smooth", "length": 50.27}),
                (1, 2, {"role": "outer_loop", "wall_role": "outer_cylinder_wall", "convexity": "smooth", "length": 25.13}),
                (2, 1, {"role": "outer_loop", "wall_role": "outer_cylinder_wall", "convexity": "smooth", "length": 25.13}),
                (0, 3, {"role": "outer_loop", "wall_role": "outer_cylinder_wall", "convexity": "smooth", "length": 50.27}),
                (3, 0, {"role": "outer_loop", "wall_role": "outer_cylinder_wall", "convexity": "smooth", "length": 50.27}),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(0, macro_id=4, macro="counterbore_hole"),
                face(1, macro_id=4, macro="counterbore_hole"),
                face(2, macro_id=4, macro="counterbore_hole"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertNotIn("counterbore_hole", {item["macro_feature"] for item in corrected["faces"]})
    def test_inner_hole_role_wins_over_exterior_cap_heuristic_for_blind_hole(self):
        graph_data = role_graph(
            [
                graph_node(0, "cylinder", axis=[0, 0, 1], radius=5.0, descriptors={"area": 1319.47, "uv_u_span": 6.283, "center": [0, 0, -8]}),
                graph_node(1, "plane", normal=[0, 0, 1], descriptors={"area": 78.54, "center": [0, 0, -16]}),
                graph_node(2, "plane", normal=[0, 0, 1], descriptors={"area": 5000.0, "center": [0, 0, 0]}),
                graph_node(3, "plane", normal=[0, 0, -1], descriptors={"area": 5000.0, "center": [0, 0, 10]}),
            ],
            [
                (0, 1, {"role": "outer_loop", "wall_role": "outer_cylinder_wall", "convexity": "smooth"}),
                (1, 0, {"role": "outer_loop", "wall_role": "outer_cylinder_wall", "convexity": "smooth"}),
                (0, 2, {"role": "inner_loop", "wall_role": "inner_hole_wall", "convexity": "smooth"}),
                (2, 0, {"role": "inner_loop", "wall_role": "inner_hole_wall", "convexity": "smooth"}),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(0, macro_id=14, macro="cylinder_boss"),
                face(1, macro_id=14, macro="cylinder_boss"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)
        by_idx = {item["idx"]: item for item in corrected["faces"]}

        self.assertEqual({by_idx[idx]["macro_feature"] for idx in [0, 1]}, {"blind_hole_round"})


    def test_stock_scale_plate_faces_are_not_accepted_as_rib(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[0, 1, 0], descriptors={
                    "area": 7829.96,
                    "center": [0, 0, 22],
                    "projected_aspect_ratio": 2.98,
                    "projected_long_extent": 149.26,
                    "projected_short_extent": 50.02,
                }),
                graph_node(1, "plane", normal=[0, 1, 0], descriptors={
                    "area": 6567.26,
                    "center": [0, 4, 23],
                    "projected_aspect_ratio": 1.79,
                    "projected_long_extent": 140.06,
                    "projected_short_extent": 78.47,
                }),
                graph_node(2, "plane", normal=[0.43, 0.5, -0.75], descriptors={
                    "area": 322.17,
                    "center": [42, 2, -3],
                    "projected_aspect_ratio": 15.11,
                    "projected_long_extent": 71.75,
                    "projected_short_extent": 4.75,
                }),
                graph_node(3, "plane", normal=[0, 0, 1], descriptors={"area": 9000.0}),
                graph_node(4, "plane", normal=[0, 0, -1], descriptors={"area": 8000.0}),
            ],
            [
                (0, 2, "line", "convex"),
                (2, 0, "line", "convex"),
                (1, 2, "line", "convex"),
                (2, 1, "line", "convex"),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(0, macro_id=20, macro="rib"),
                face(1, macro_id=20, macro="rib"),
                face(2, macro_id=20, macro="rib"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)
        by_idx = {item["idx"]: item for item in corrected["faces"]}

        self.assertEqual({by_idx[idx]["macro_feature"] for idx in [0, 1, 2]}, {"stock"})
        invalid_reasons = [
            reason
            for component in corrected["topology_postprocess"]["invalid_components"]
            for reason in component.get("reasons", [])
        ]
        self.assertIn("rib_component_contains_stock_scale_plate_faces", invalid_reasons)

    def test_mixed_rib_candidate_rejects_stock_scale_plate_faces(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[0, 1, 0], descriptors={
                    "area": 7829.96,
                    "center": [0, 0, 22],
                    "projected_aspect_ratio": 2.98,
                    "projected_long_extent": 149.26,
                    "projected_short_extent": 50.02,
                }),
                graph_node(1, "plane", normal=[0, 1, 0], descriptors={
                    "area": 6567.26,
                    "center": [0, 4, 23],
                    "projected_aspect_ratio": 1.79,
                    "projected_long_extent": 140.06,
                    "projected_short_extent": 78.47,
                }),
                graph_node(2, "plane", normal=[0.43, 0.5, -0.75], descriptors={
                    "area": 322.17,
                    "center": [42, 2, -3],
                    "projected_aspect_ratio": 15.11,
                    "projected_long_extent": 71.75,
                    "projected_short_extent": 4.75,
                }),
                graph_node(3, "plane", normal=[0, 0, 1], descriptors={"area": 9000.0}),
                graph_node(4, "plane", normal=[0, 0, -1], descriptors={"area": 8000.0}),
            ],
            [
                (2, 0, "line", "convex"),
                (0, 2, "line", "convex"),
                (2, 1, "line", "convex"),
                (1, 2, "line", "convex"),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(0, macro_id=0, macro="stock"),
                face(1, macro_id=0, macro="stock"),
                face(2, macro_id=20, macro="rib"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)
        by_idx = {item["idx"]: item for item in corrected["faces"]}

        self.assertEqual(by_idx[0]["macro_feature"], "stock")
        self.assertEqual(by_idx[1]["macro_feature"], "stock")
        self.assertNotEqual(by_idx[2]["macro_feature"], "rib")
    def test_mixed_rib_candidate_absorbs_adjacent_transition_faces(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[1, 0, 0], descriptors={
                    "area": 1200.0,
                    "center": [0, 0, 0],
                    "projected_aspect_ratio": 1.5,
                    "projected_long_extent": 55.0,
                    "projected_short_extent": 35.0,
                }),
                graph_node(1, "plane", normal=[1, 0, 0], descriptors={
                    "area": 1200.0,
                    "center": [4, 0, 0],
                    "projected_aspect_ratio": 1.5,
                    "projected_long_extent": 55.0,
                    "projected_short_extent": 35.0,
                }),
                graph_node(2, "plane", normal=[0, 0, 1], descriptors={
                    "area": 180.0,
                    "center": [2, 0, 18],
                    "projected_aspect_ratio": 9.0,
                    "projected_long_extent": 45.0,
                    "projected_short_extent": 5.0,
                }),
                graph_node(3, "cylinder", axis=[0, 1, 0], radius=2.0, descriptors={
                    "area": 90.0,
                    "center": [2, 1, 18],
                    "uv_u_span": 1.57,
                }),
                graph_node(4, "plane", normal=[0, 0, -1], descriptors={"area": 20000.0}),
            ],
            [
                (0, 1, "line", "convex"),
                (1, 0, "line", "convex"),
                (0, 2, "line", "convex"),
                (2, 0, "line", "convex"),
                (1, 2, "line", "convex"),
                (2, 1, "line", "convex"),
                (1, 3, "line", "convex"),
                (3, 1, "line", "convex"),
                (2, 3, "line", "convex"),
                (3, 2, "line", "convex"),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(0, macro_id=20, macro="rib"),
                face(1, macro_id=0, macro="stock"),
                face(2, macro_id=0, macro="stock"),
                face(3, macro_id=25, macro="fillet"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)
        by_idx = {item["idx"]: item for item in corrected["faces"]}

        self.assertEqual({by_idx[idx]["macro_feature"] for idx in [0, 1, 2, 3]}, {"rib"})
        self.assertEqual(by_idx[3]["semantic_subtype"], "rib_side_plane")
if __name__ == "__main__":
    unittest.main()





