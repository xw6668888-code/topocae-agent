import unittest

from tests.test_topology_postprocess import edge, face, graph, graph_node
from utils.candidate_topology_features import extract_candidate_topology_features


class CandidateTopologyFeaturesTest(unittest.TestCase):
    def test_large_base_candidate_exposes_area_ratio_and_raw_stock_fraction(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[0, 0, 1], descriptors={"area": 9807.45}),
                graph_node(1, "plane", normal=[0, -1, 0], descriptors={"area": 835.0}),
                graph_node(2, "plane", normal=[0, 0, 1], descriptors={"area": 12408.45}),
                graph_node(3, "plane", normal=[1, 0, 0], descriptors={"area": 2600.0}),
                graph_node(4, "plane", normal=[-1, 0, 0], descriptors={"area": 2600.0}),
            ],
            [
                (0, 1, "line", "convex", 167.0),
                (1, 0, "line", "convex", 167.0),
                (1, 2, "line", "convex", 167.0),
                (2, 1, "line", "convex", 167.0),
            ],
        )
        prediction_faces = [
            {**face(idx, macro_id=20, macro="rib"), "raw_macro_feature": "stock"}
            for idx in [0, 1, 2]
        ]

        features = extract_candidate_topology_features(
            graph_data,
            [0, 1, 2],
            prediction_faces=prediction_faces,
            candidate_macro="rib",
        )

        self.assertGreater(features["area_ratio"], 0.8)
        self.assertEqual(features["face_count"], 3.0)
        self.assertEqual(features["raw_stock_face_fraction"], 1.0)
        self.assertEqual(features["boundary_neighbor_count"], 0.0)

    def test_parent_plane_relation_separates_slot_like_and_fillet_like_cylinders(self):
        slot_graph = graph(
            [
                graph_node(0, "cylinder", axis=[0, -1, 0], radius=2.5, descriptors={"area": 3416.28}),
                graph_node(1, "plane", normal=[0, 0, 1]),
                graph_node(2, "plane", normal=[0, 0, -1]),
            ],
            [
                (0, 1, "line", "convex", 435.0),
                (1, 0, "line", "convex", 435.0),
                (0, 2, "line", "convex", 435.0),
                (2, 0, "line", "convex", 435.0),
            ],
        )
        fillet_graph = graph(
            [
                graph_node(0, "cylinder", axis=[1, 0, 0], radius=20.0, descriptors={"area": 1570.8}),
                graph_node(1, "plane", normal=[0, 0, 1]),
                graph_node(2, "plane", normal=[0, 1, 0]),
            ],
            [
                (0, 1, "line", "convex", 50.0),
                (1, 0, "line", "convex", 50.0),
                (0, 2, "line", "convex", 50.0),
                (2, 0, "line", "convex", 50.0),
            ],
        )

        slot_features = extract_candidate_topology_features(slot_graph, [0], candidate_macro="slot_o")
        fillet_features = extract_candidate_topology_features(fillet_graph, [0], candidate_macro="fillet")

        self.assertEqual(slot_features["boundary_parent_plane_parallel_pair_count"], 1.0)
        self.assertEqual(slot_features["boundary_parent_plane_orthogonal_pair_count"], 0.0)
        self.assertEqual(fillet_features["boundary_parent_plane_parallel_pair_count"], 0.0)
        self.assertEqual(fillet_features["boundary_parent_plane_orthogonal_pair_count"], 1.0)
        self.assertGreater(slot_features["cylindrical_angle_estimate_max"], 2.35)
        self.assertLess(fillet_features["cylindrical_angle_estimate_max"], 2.35)


if __name__ == "__main__":
    unittest.main()
