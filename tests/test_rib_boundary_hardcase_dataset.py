import unittest
from pathlib import Path

from build_rib_boundary_classifier_dataset import NOT_RIB_MEMBER_LABEL, RIB_MEMBER_LABEL
from build_rib_boundary_hardcase_dataset import build_hardcase_records_for_graph
from tests.test_topology_postprocess import graph, graph_node


class RibBoundaryHardcaseDatasetTest(unittest.TestCase):
    def test_build_hardcase_records_include_fragment_positive_and_contaminated_negative(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[1, 0, 0], descriptors={"area": 20.0}),
                graph_node(1, "plane", normal=[-1, 0, 0], descriptors={"area": 20.0}),
                graph_node(2, "plane", normal=[0, 0, 1], descriptors={"area": 30.0}),
                graph_node(3, "plane", normal=[0, 0, 1], descriptors={"area": 500.0}),
                graph_node(4, "plane", normal=[0, 1, 0], descriptors={"area": 60.0}),
            ],
            [
                (0, 1, "line", "convex", 10.0),
                (1, 0, "line", "convex", 10.0),
                (0, 2, "line", "convex", 10.0),
                (2, 0, "line", "convex", 10.0),
                (1, 2, "line", "convex", 10.0),
                (2, 1, "line", "convex", 10.0),
                (0, 3, "line", "convex", 20.0),
                (3, 0, "line", "convex", 20.0),
                (1, 4, "line", "concave", 12.0),
                (4, 1, "line", "concave", 12.0),
            ],
        )
        graph_data["feature_sets"] = [
            {
                "instance_id": 1,
                "macro_feature": "rib",
                "macro_feature_id": 20,
                "node_indices": [0, 1, 2],
            }
        ]

        records = build_hardcase_records_for_graph(
            sample_id="toy",
            graph_data=graph_data,
            graph_path=Path("toy.json"),
            max_contaminants=1,
        )

        sources = {record["source"] for record in records}
        self.assertIn("hard_fragment_positive", sources)
        self.assertIn("hard_contaminated_negative", sources)

        positives = [record for record in records if record["source"] == "hard_fragment_positive"]
        negatives = [record for record in records if record["source"] == "hard_contaminated_negative"]
        self.assertTrue(all(record["label"] == RIB_MEMBER_LABEL for record in positives))
        self.assertTrue(all(record["label"] == NOT_RIB_MEMBER_LABEL for record in negatives))
        self.assertTrue(any(3 in record["seed_face_indices"] or 4 in record["seed_face_indices"] for record in negatives))


if __name__ == "__main__":
    unittest.main()
