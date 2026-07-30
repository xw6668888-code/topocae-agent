import unittest

from tests.test_topology_postprocess import face, graph, graph_node
from utils.topology_postprocess import apply_topology_postprocess


class TopologyPostprocessV53Test(unittest.TestCase):
    def test_mixed_rib_pocket_and_boss_predictions_are_repaired_to_rib_instance(self):
        graph_data = graph(
            [
                graph_node(
                    24,
                    "plane",
                    normal=[-0.707, -0.707, 0.0],
                    descriptors={
                        "area": 23.7588,
                        "projected_aspect_ratio": 66.0,
                        "projected_long_extent": 39.598,
                        "projected_short_extent": 0.6,
                        "center": [16.0, 16.0, 0.7],
                    },
                ),
                graph_node(
                    25,
                    "plane",
                    normal=[0.0, 0.0, 1.0],
                    descriptors={
                        "area": 391.5,
                        "projected_aspect_ratio": 2.07,
                        "projected_long_extent": 39.598,
                        "projected_short_extent": 19.09,
                        "center": [11.34, 11.34, 1.0],
                    },
                ),
                graph_node(
                    26,
                    "plane",
                    normal=[0.0, 0.0, 1.0],
                    descriptors={
                        "area": 392.0,
                        "projected_aspect_ratio": 2.0,
                        "projected_long_extent": 39.598,
                        "projected_short_extent": 19.8,
                        "center": [11.33, 11.33, 0.4],
                    },
                ),
            ],
            [
                (24, 25, "line", "convex", 39.598),
                (25, 24, "line", "convex", 39.598),
                (24, 26, "line", "convex", 39.598),
                (26, 24, "line", "convex", 39.598),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(24, macro_id=20, macro="rib"),
                face(25, macro_id=11, macro="pocket_rect"),
                face(26, macro_id=24, macro="rectangular_boss"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual([item["macro_feature"] for item in corrected["faces"]], ["rib", "rib", "rib"])
        self.assertEqual(
            [item["semantic_subtype"] for item in corrected["faces"]],
            ["rib_top_plane", "rib_side_plane", "rib_side_plane"],
        )

    def test_stepped_hole_like_obround_faces_are_repaired_to_obround_instance(self):
        graph_data = graph(
            [
                graph_node(16, "plane", normal=[0, 0, -1], descriptors={"area": 16.0}),
                graph_node(17, "plane", normal=[0, 0, 1], descriptors={"area": 16.0}),
                graph_node(18, "cylinder", axis=[1, 0, 0], radius=2.0, descriptors={"area": 12.5664}),
                graph_node(19, "cylinder", axis=[1, 0, 0], radius=2.0, descriptors={"area": 12.5664}),
            ],
            [
                (16, 18, "line", "concave", 2.0),
                (18, 16, "line", "concave", 2.0),
                (16, 19, "line", "concave", 2.0),
                (19, 16, "line", "concave", 2.0),
                (17, 18, "line", "concave", 2.0),
                (18, 17, "line", "concave", 2.0),
                (17, 19, "line", "concave", 2.0),
                (19, 17, "line", "concave", 2.0),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(16, macro_id=3, macro="stepped_hole_round"),
                face(17, macro_id=3, macro="stepped_hole_round"),
                face(18, macro_id=3, macro="stepped_hole_round"),
                face(19, macro_id=3, macro="stepped_hole_round"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual(
            [item["macro_feature"] for item in corrected["faces"]],
            ["obround_hole", "obround_hole", "obround_hole", "obround_hole"],
        )


    def test_narrow_pocket_rect_prediction_with_chamfer_edges_is_repaired_to_chamfer(self):
        graph_data = graph(
            [
                graph_node(
                    32,
                    "plane",
                    normal=[0.707, 0.707, 0.0],
                    descriptors={
                        "area": 39.598,
                        "projected_aspect_ratio": 19.8,
                        "projected_long_extent": 28.0,
                        "projected_short_extent": 1.414,
                    },
                ),
                graph_node(6, "plane", descriptors={"area": 666.0}),
                graph_node(7, "plane", descriptors={"area": 706.0}),
                graph_node(25, "plane", descriptors={"area": 391.5}),
                graph_node(28, "plane", descriptors={"area": 391.5}),
            ],
            [
                (32, 6, "line", "concave", 28.0, 0.785),
                (6, 32, "line", "concave", 28.0, 0.785),
                (32, 7, "line", "concave", 28.0, 0.785),
                (7, 32, "line", "concave", 28.0, 0.785),
                (32, 25, "line", "concave", 1.414),
                (25, 32, "line", "concave", 1.414),
                (32, 28, "line", "concave", 1.414),
                (28, 32, "line", "concave", 1.414),
            ],
        )
        prediction = {"sample_id": "sample", "faces": [face(32, macro_id=11, macro="pocket_rect")]}

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual(corrected["faces"][0]["macro_feature"], "chamfer")
        self.assertEqual(corrected["faces"][0]["semantic_subtype"], "chamfer_bevel_plane")
    def test_busy_stock_side_face_with_many_neighbors_is_not_repaired_to_chamfer(self):
        nodes = [
            graph_node(
                3,
                "plane",
                normal=[-1.0, 0.0, 0.0],
                descriptors={
                    "area": 59.116,
                    "projected_aspect_ratio": 15.0,
                    "projected_long_extent": 30.0,
                    "projected_short_extent": 2.0,
                },
            )
        ]
        nodes.extend(graph_node(idx, "plane", descriptors={"area": 400.0}) for idx in range(4, 12))
        edge_defs = []
        for idx in range(4, 12):
            angle = 0.785 if idx in {4, 5} else 1.571
            edge_defs.extend([(3, idx, "line", "convex", 2.0, angle), (idx, 3, "line", "convex", 2.0, angle)])
        graph_data = graph(nodes, edge_defs)
        prediction = {"sample_id": "sample", "faces": [face(3, macro_id=0, macro="stock")]}

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual(corrected["faces"][0]["macro_feature"], "stock")
if __name__ == "__main__":
    unittest.main()


