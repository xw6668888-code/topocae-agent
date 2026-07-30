import json
import tempfile
import unittest
from pathlib import Path

from build_rib_boundary_classifier_dataset import (
    build_rib_boundary_records_for_graph,
    split_rib_boundary_records,
    write_rib_boundary_dataset,
)
from tests.test_topology_postprocess import graph, graph_node


class RibBoundaryClassifierDatasetTest(unittest.TestCase):
    def test_build_records_creates_leave_one_out_rib_members_and_adjacent_non_members(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[1, 0, 0], descriptors={"area": 20.0, "projected_aspect_ratio": 8.0}),
                graph_node(1, "plane", normal=[-1, 0, 0], descriptors={"area": 20.0, "projected_aspect_ratio": 8.0}),
                graph_node(2, "plane", normal=[0, 0, 1], descriptors={"area": 30.0, "projected_aspect_ratio": 6.0}),
                graph_node(3, "plane", normal=[0, 0, 1], descriptors={"area": 500.0, "projected_aspect_ratio": 1.5}),
            ],
            [
                (0, 2, "line", "convex", 12.0),
                (2, 0, "line", "convex", 12.0),
                (1, 2, "line", "convex", 12.0),
                (2, 1, "line", "convex", 12.0),
                (0, 3, "line", "convex", 18.0),
                (3, 0, "line", "convex", 18.0),
                (1, 3, "line", "convex", 18.0),
                (3, 1, "line", "convex", 18.0),
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

        records = build_rib_boundary_records_for_graph(
            sample_id="toy",
            graph_data=graph_data,
            graph_path=Path("toy.json"),
        )

        positives = [record for record in records if record["label"] == "rib_member"]
        negatives = [record for record in records if record["label"] == "not_rib_member"]

        self.assertEqual({record["candidate_face_idx"] for record in positives}, {0, 1, 2})
        self.assertIn(3, {record["candidate_face_idx"] for record in negatives})
        self.assertEqual({len(record["seed_face_indices"]) for record in positives}, {2})
        self.assertEqual({len(record["seed_face_indices"]) for record in negatives}, {2})
        sample = positives[0]
        self.assertIn("seed_face_count", sample["features"])
        self.assertIn("candidate_to_seed_edge_count", sample["features"])
        self.assertIn("candidate_to_seed_area_ratio", sample["features"])
        self.assertGreater(sample["features"]["seed_face_count"], 0.0)

    def test_write_rib_boundary_dataset_creates_trainable_binary_splits(self):
        records = [
            {
                "sample_id": "a",
                "source": "truth_leave_one_out",
                "label": "rib_member",
                "seed_face_indices": [1, 2],
                "candidate_face_idx": 0,
                "features": {"seed_face_count": 2.0, "candidate_to_seed_edge_count": 1.0},
            },
            {
                "sample_id": "b",
                "source": "adjacent_truth_negative",
                "label": "not_rib_member",
                "seed_face_indices": [3, 4],
                "candidate_face_idx": 5,
                "features": {"seed_face_count": 2.0, "candidate_to_seed_edge_count": 1.0},
            },
        ]
        splits = split_rib_boundary_records(records, seed=1, train_ratio=0.5, val_ratio=0.25)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            metadata = write_rib_boundary_dataset(records, splits, output_dir)

            self.assertEqual(metadata["label_to_id"], {"not_rib_member": 0, "rib_member": 1})
            self.assertEqual(metadata["feature_schema"], ["candidate_to_seed_edge_count", "seed_face_count"])
            loaded = []
            for split_name in ["train", "val", "test"]:
                split_path = output_dir / f"{split_name}.jsonl"
                self.assertTrue(split_path.exists())
                loaded.extend(json.loads(line) for line in split_path.read_text(encoding="utf-8").splitlines() if line)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(len(loaded[0]["feature_vector"]), 2)


if __name__ == "__main__":
    unittest.main()

