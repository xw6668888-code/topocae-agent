import json
import tempfile
import unittest
from pathlib import Path

from build_rib_candidate_proposer_dataset import (
    build_records_from_review_template,
    split_rib_candidate_records,
    write_rib_candidate_dataset,
)
from tests.test_topology_postprocess import graph, graph_node


class RibCandidateProposerDatasetTest(unittest.TestCase):
    def test_build_records_from_colored_review_template(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[1, 0, 0], descriptors={"area": 20.0, "projected_aspect_ratio": 8.0}),
                graph_node(1, "plane", normal=[0, 0, 1], descriptors={"area": 500.0, "projected_aspect_ratio": 1.2}),
            ],
            [
                (0, 1, "line", "convex", 12.0),
                (1, 0, "line", "convex", 12.0),
            ],
        )
        template = {
            "samples": [
                {
                    "sample_id": "toy",
                    "graph_path": "toy.json",
                    "candidate_faces": [
                        {"face_idx": 0, "review": {"is_rib_face": True, "rib_instance_id": "rib_1"}},
                        {"face_idx": 1, "review": {"is_rib_face": False, "rib_instance_id": ""}},
                        {"face_idx": 2, "review": {"is_rib_face": None, "rib_instance_id": ""}},
                    ],
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_path = root / "toy.json"
            graph_path.write_text(json.dumps(graph_data), encoding="utf-8")
            records = build_records_from_review_template(template, root)

        self.assertEqual(len(records), 2)
        self.assertEqual({record["label"] for record in records}, {"rib_candidate", "not_rib_candidate"})
        self.assertEqual({record["candidate_face_idx"] for record in records}, {0, 1})
        self.assertIn("candidate_area_ratio", records[0]["features"])
        self.assertIn("candidate_convex_edge_count", records[0]["features"])

    def test_write_dataset_creates_trainable_binary_splits(self):
        records = [
            {
                "sample_id": "a",
                "source": "colored_rib_truth",
                "label": "rib_candidate",
                "candidate_face_idx": 0,
                "features": {"candidate_area_ratio": 0.1, "candidate_neighbor_count": 2.0},
            },
            {
                "sample_id": "b",
                "source": "colored_rib_truth",
                "label": "not_rib_candidate",
                "candidate_face_idx": 1,
                "features": {"candidate_area_ratio": 0.8, "candidate_neighbor_count": 4.0},
            },
        ]
        splits = split_rib_candidate_records(records, seed=1, train_ratio=0.5, val_ratio=0.25)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            metadata = write_rib_candidate_dataset(records, splits, output_dir)

            self.assertEqual(metadata["label_to_id"], {"not_rib_candidate": 0, "rib_candidate": 1})
            self.assertEqual(metadata["feature_schema"], ["candidate_area_ratio", "candidate_neighbor_count"])
            loaded = []
            for split_name in ["train", "val", "test"]:
                split_path = output_dir / f"{split_name}.jsonl"
                self.assertTrue(split_path.exists())
                loaded.extend(json.loads(line) for line in split_path.read_text(encoding="utf-8").splitlines() if line)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(len(loaded[0]["feature_vector"]), 2)


if __name__ == "__main__":
    unittest.main()
