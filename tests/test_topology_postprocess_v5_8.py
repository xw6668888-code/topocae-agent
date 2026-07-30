import unittest

from tests.test_topology_postprocess import face, graph_node
from tests.test_topology_postprocess_v5_6 import role_graph
from utils.topology_postprocess import apply_topology_postprocess


class TopologyPostprocessV58Test(unittest.TestCase):
    def test_invalid_stepped_component_preserves_inner_full_cylinder_as_through_hole(self):
        graph_data = role_graph(
            [
                graph_node(0, "cylinder", axis=[0, 0, 1], radius=25.0, descriptors={"area": 7853.98, "uv_u_span": 6.283}),
                graph_node(1, "plane", normal=[0, 0, 1], descriptors={"area": 4000.0}),
                graph_node(2, "plane", normal=[0, 0, -1], descriptors={"area": 4000.0}),
            ],
            [
                (0, 1, {"role": "inner_loop", "wall_role": "inner_hole_wall", "convexity": "smooth", "length": 157.08}),
                (1, 0, {"role": "inner_loop", "wall_role": "inner_hole_wall", "convexity": "smooth", "length": 157.08}),
                (0, 2, {"role": "inner_loop", "wall_role": "inner_hole_wall", "convexity": "smooth", "length": 157.08}),
                (2, 0, {"role": "inner_loop", "wall_role": "inner_hole_wall", "convexity": "smooth", "length": 157.08}),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(0, macro_id=3, macro="stepped_hole_round"),
                face(1, macro_id=3, macro="stepped_hole_round"),
                face(2, macro_id=3, macro="stepped_hole_round"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)
        by_idx = {item["idx"]: item for item in corrected["faces"]}

        self.assertEqual(by_idx[0]["macro_feature"], "through_hole_round")
        self.assertEqual(by_idx[0]["semantic_subtype"], "through_hole_cylinder")
        self.assertEqual(by_idx[1]["macro_feature"], "stock")
        self.assertEqual(by_idx[2]["macro_feature"], "stock")

    def test_invalid_counterbore_component_preserves_inner_full_cylinder_as_through_hole(self):
        graph_data = role_graph(
            [
                graph_node(0, "cylinder", axis=[0, 0, 1], radius=12.5, descriptors={"area": 8296.45, "uv_u_span": 6.283}),
                graph_node(1, "plane", normal=[0, 0, 1], descriptors={"area": 5000.0}),
                graph_node(2, "plane", normal=[0, 0, -1], descriptors={"area": 5000.0}),
                graph_node(3, "plane", normal=[1, 0, 0], descriptors={"area": 200.0}),
            ],
            [
                (0, 1, {"role": "inner_loop", "wall_role": "inner_hole_wall", "convexity": "smooth", "length": 78.54}),
                (1, 0, {"role": "inner_loop", "wall_role": "inner_hole_wall", "convexity": "smooth", "length": 78.54}),
                (0, 2, {"role": "inner_loop", "wall_role": "inner_hole_wall", "convexity": "smooth", "length": 78.54}),
                (2, 0, {"role": "inner_loop", "wall_role": "inner_hole_wall", "convexity": "smooth", "length": 78.54}),
                (1, 3, {"role": "outer_loop", "wall_role": "outer_cylinder_wall", "convexity": "smooth", "length": 20.0}),
                (3, 1, {"role": "outer_loop", "wall_role": "outer_cylinder_wall", "convexity": "smooth", "length": 20.0}),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(0, macro_id=4, macro="counterbore_hole"),
                face(1, macro_id=4, macro="counterbore_hole"),
                face(2, macro_id=4, macro="counterbore_hole"),
                face(3, macro_id=4, macro="counterbore_hole"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)
        by_idx = {item["idx"]: item for item in corrected["faces"]}

        self.assertEqual(by_idx[0]["macro_feature"], "through_hole_round")
        self.assertEqual(by_idx[0]["semantic_subtype"], "through_hole_cylinder")
        self.assertEqual(by_idx[1]["macro_feature"], "stock")
        self.assertEqual(by_idx[2]["macro_feature"], "stock")
        self.assertEqual(by_idx[3]["macro_feature"], "stock")

    def test_outer_cylinder_role_is_not_repaired_to_through_hole_from_invalid_hole_macro(self):
        graph_data = role_graph(
            [
                graph_node(0, "cylinder", axis=[0, 0, 1], radius=30.0, descriptors={"area": 9047.85, "uv_u_span": 6.283}),
                graph_node(1, "plane", normal=[0, 0, 1], descriptors={"area": 4000.0}),
                graph_node(2, "plane", normal=[0, 0, -1], descriptors={"area": 4000.0}),
            ],
            [
                (0, 1, {"role": "outer_loop", "wall_role": "outer_cylinder_wall", "convexity": "smooth", "length": 188.5}),
                (1, 0, {"role": "outer_loop", "wall_role": "outer_cylinder_wall", "convexity": "smooth", "length": 188.5}),
                (0, 2, {"role": "outer_loop", "wall_role": "outer_cylinder_wall", "convexity": "smooth", "length": 188.5}),
                (2, 0, {"role": "outer_loop", "wall_role": "outer_cylinder_wall", "convexity": "smooth", "length": 188.5}),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [face(0, macro_id=3, macro="stepped_hole_round")],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertNotEqual(corrected["faces"][0]["macro_feature"], "through_hole_round")


if __name__ == "__main__":
    unittest.main()
