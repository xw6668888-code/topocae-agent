import unittest

from tests.test_topology_postprocess import face, graph, graph_node
from utils.topology_postprocess import (
    apply_topology_postprocess,
    find_countersink_candidates,
)


class TopologyPostprocessV52Test(unittest.TestCase):
    def test_complete_rectangular_pocket_is_not_expanded_into_adjacent_stock_face(self):
        graph_data = graph(
            [graph_node(idx, "plane", descriptors={"area": 100.0}) for idx in range(5)]
            + [graph_node(5, "plane", descriptors={"area": 350.0})],
            [
                (0, 1, "line", "concave"),
                (1, 0, "line", "concave"),
                (0, 2, "line", "concave"),
                (2, 0, "line", "concave"),
                (0, 3, "line", "concave"),
                (3, 0, "line", "concave"),
                (0, 4, "line", "concave"),
                (4, 0, "line", "concave"),
                (5, 1, "line", "concave"),
                (1, 5, "line", "concave"),
                (5, 2, "line", "concave"),
                (2, 5, "line", "concave"),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(0, macro_id=11, macro="pocket_rect"),
                face(1, macro_id=11, macro="pocket_rect"),
                face(2, macro_id=11, macro="pocket_rect"),
                face(3, macro_id=11, macro="pocket_rect"),
                face(4, macro_id=11, macro="pocket_rect"),
                face(5, macro_id=0, macro="stock"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual(
            [item["macro_feature"] for item in corrected["faces"]],
            ["pocket_rect", "pocket_rect", "pocket_rect", "pocket_rect", "pocket_rect", "stock"],
        )
    def test_small_countersink_cone_predicted_as_chamfer_is_repaired_when_attached_to_through_cylinder(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[0, 0, 1], descriptors={"area": 1000.0}),
                graph_node(1, "cylinder", axis=[0, 0, 1], radius=2.0, descriptors={"area": 15.1}),
                graph_node(2, "cone", axis=[0, 0, 1], descriptors={"area": 17.1}),
                graph_node(3, "plane", normal=[0, 0, -1], descriptors={"area": 1000.0}),
            ],
            [
                (1, 0, "circle", "smooth", 12.56),
                (0, 1, "circle", "smooth", 12.56),
                (1, 2, "circle", "smooth", 12.56),
                (2, 1, "circle", "smooth", 12.56),
                (2, 3, "circle", "smooth", 18.84),
                (3, 2, "circle", "smooth", 18.84),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(1, macro_id=0, macro="stock"),
                face(2, macro_id=26, macro="chamfer"),
            ],
        }

        candidates = find_countersink_candidates(prediction["faces"], graph_data, set())
        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual(candidates, [[1, 2]])
        self.assertEqual(
            [item["macro_feature"] for item in corrected["faces"]],
            ["countersink_hole", "countersink_hole"],
        )
        self.assertEqual(
            [item["semantic_subtype"] for item in corrected["faces"]],
            ["countersink_small_cylinder", "countersink_cone"],
        )

    def test_single_stepped_hole_cylinder_with_two_openings_is_repaired_to_through_hole(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[0, 0, 1]),
                graph_node(1, "cylinder", axis=[0, 0, 1], radius=5.0),
                graph_node(2, "plane", normal=[0, 0, -1]),
            ],
            [
                (1, 0, "circle", "smooth", 31.4),
                (0, 1, "circle", "smooth", 31.4),
                (1, 2, "circle", "smooth", 31.4),
                (2, 1, "circle", "smooth", 31.4),
            ],
        )
        prediction = {"sample_id": "sample", "faces": [face(1, macro_id=3, macro="stepped_hole_round")]}

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual(corrected["faces"][0]["macro_feature"], "through_hole_round")
        self.assertEqual(corrected["faces"][0]["semantic_subtype"], "through_hole_cylinder")

    def test_existing_rib_component_absorbs_adjacent_transition_faces(self):
        graph_data = graph(
            [
                graph_node(
                    0,
                    "plane",
                    normal=[0, 0.95, -0.31],
                    descriptors={
                        "area": 6261.8,
                        "projected_aspect_ratio": 62.0,
                        "projected_long_extent": 626.0,
                        "projected_short_extent": 10.0,
                        "center": [281.0, 96.7, -318.8],
                    },
                ),
                graph_node(
                    1,
                    "plane",
                    normal=[1, 0, 0],
                    descriptors={
                        "area": 57606.2,
                        "projected_aspect_ratio": 3.2,
                        "projected_long_extent": 614.8,
                        "projected_short_extent": 191.4,
                        "center": [286.0, 64.5, -219.6],
                    },
                ),
                graph_node(
                    2,
                    "plane",
                    normal=[-1, 0, 0],
                    descriptors={
                        "area": 57606.2,
                        "projected_aspect_ratio": 3.2,
                        "projected_long_extent": 614.8,
                        "projected_short_extent": 191.4,
                        "center": [276.0, 64.5, -219.6],
                    },
                ),
                graph_node(
                    3,
                    "cylinder",
                    axis=[1, 0, 0],
                    radius=10.0,
                    descriptors={"area": 900.0, "projected_aspect_ratio": 0.0},
                ),
            ],
            [
                (0, 1, "line", "convex", 626.0),
                (1, 0, "line", "convex", 626.0),
                (0, 2, "line", "convex", 626.0),
                (2, 0, "line", "convex", 626.0),
                (3, 0, "line", "convex", 2.0),
                (0, 3, "line", "convex", 2.0),
                (3, 1, "circle", "smooth", 21.6),
                (1, 3, "circle", "smooth", 21.6),
                (3, 2, "circle", "smooth", 21.6),
                (2, 3, "circle", "smooth", 21.6),
            ],
        )
        transition = face(3, macro_id=0, macro="pocket_circular")
        transition["semantic_subtype"] = "circular_pocket_side_cylinder"
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(0, macro_id=20, macro="rib"),
                face(1, macro_id=20, macro="rib"),
                face(2, macro_id=20, macro="rib"),
                transition,
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual([item["macro_feature"] for item in corrected["faces"]], ["rib", "rib", "rib", "rib"])
        self.assertEqual(corrected["faces"][3]["semantic_subtype"], "rib_side_plane")
    def test_rib_candidate_absorbs_adjacent_transition_faces(self):
        graph_data = graph(
            [
                graph_node(
                    0,
                    "plane",
                    normal=[0, 0.95, -0.31],
                    descriptors={
                        "area": 6261.8,
                        "projected_aspect_ratio": 62.0,
                        "projected_long_extent": 626.0,
                        "projected_short_extent": 10.0,
                        "center": [281.0, 96.7, -318.8],
                    },
                ),
                graph_node(
                    1,
                    "plane",
                    normal=[1, 0, 0],
                    descriptors={
                        "area": 57606.2,
                        "projected_aspect_ratio": 3.2,
                        "projected_long_extent": 614.8,
                        "projected_short_extent": 191.4,
                        "center": [286.0, 64.5, -219.6],
                    },
                ),
                graph_node(
                    2,
                    "plane",
                    normal=[-1, 0, 0],
                    descriptors={
                        "area": 57606.2,
                        "projected_aspect_ratio": 3.2,
                        "projected_long_extent": 614.8,
                        "projected_short_extent": 191.4,
                        "center": [276.0, 64.5, -219.6],
                    },
                ),
                graph_node(
                    3,
                    "plane",
                    normal=[0.7, 0.7, 0.0],
                    descriptors={
                        "area": 900.0,
                        "projected_aspect_ratio": 24.0,
                        "projected_long_extent": 600.0,
                        "projected_short_extent": 2.5,
                    },
                ),
                graph_node(4, "plane", normal=[0, 0, 1], descriptors={"area": 400000.0}),
                graph_node(5, "plane", normal=[0, 1, 0], descriptors={"area": 350000.0}),
            ],
            [
                (0, 1, "line", "convex", 626.0),
                (1, 0, "line", "convex", 626.0),
                (0, 2, "line", "convex", 626.0),
                (2, 0, "line", "convex", 626.0),
                (3, 1, "line", "convex", 600.0, 0.785),
                (1, 3, "line", "convex", 600.0, 0.785),
                (3, 0, "line", "convex", 600.0, 0.785),
                (0, 3, "line", "convex", 600.0, 0.785),
                (1, 4, "line", "concave", 614.8),
                (4, 1, "line", "concave", 614.8),
                (2, 5, "line", "concave", 614.8),
                (5, 2, "line", "concave", 614.8),
            ],
        )
        prediction = {
            "sample_id": "sample",
            "faces": [
                face(0, macro_id=0, macro="stock"),
                face(1, macro_id=0, macro="stock"),
                face(2, macro_id=0, macro="stock"),
                face(3, macro_id=26, macro="chamfer"),
                face(4, macro_id=0, macro="stock"),
                face(5, macro_id=0, macro="stock"),
            ],
        }

        corrected = apply_topology_postprocess(prediction, graph_data)

        self.assertEqual(
            [item["macro_feature"] for item in corrected["faces"]],
            ["rib", "rib", "rib", "rib", "stock", "stock"],
        )
        self.assertEqual(corrected["faces"][3]["semantic_subtype"], "rib_side_plane")


if __name__ == "__main__":
    unittest.main()



