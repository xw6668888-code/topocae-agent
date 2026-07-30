import json
import tempfile
import unittest
from pathlib import Path

from build_candidate_classifier_dataset import (
    DEFAULT_TARGET_MACROS,
    build_rule_proposals,
    build_candidate_records_for_graph,
    split_candidate_records,
    write_candidate_dataset,
)
from tests.test_topology_postprocess import face, graph, graph_node


class CandidateClassifierDatasetTest(unittest.TestCase):
    def test_default_target_macros_include_formal_counterbore_and_countersink_targets(self):
        self.assertIn("counterbore_hole", DEFAULT_TARGET_MACROS)
        self.assertIn("countersink_hole", DEFAULT_TARGET_MACROS)

    def test_rule_proposals_include_counterbore_candidates(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[0, 0, 1], descriptors={"area": 1000.0}),
                graph_node(1, "cylinder", axis=[0, 0, 1], radius=8.0, descriptors={"area": 120.0}),
                graph_node(2, "plane", normal=[0, 0, 1], descriptors={"area": 150.0}),
                graph_node(3, "cylinder", axis=[0, 0, 1], radius=4.0, descriptors={"area": 400.0}),
                graph_node(4, "plane", normal=[0, 0, -1], descriptors={"area": 1000.0}),
            ],
            [
                (1, 0, "circle", "smooth", 50.0),
                (0, 1, "circle", "smooth", 50.0),
                (1, 2, "circle", "concave", 50.0),
                (2, 1, "circle", "concave", 50.0),
                (2, 3, "circle", "convex", 25.0),
                (3, 2, "circle", "convex", 25.0),
                (3, 4, "circle", "smooth", 25.0),
                (4, 3, "circle", "smooth", 25.0),
            ],
        )
        prediction_data = {
            "faces": [
                face(1, macro_id=4, macro="counterbore_hole"),
                face(2, macro_id=0, macro="stock"),
                face(3, macro_id=1, macro="through_hole_round"),
            ],
        }

        proposals = build_rule_proposals(graph_data, prediction_data, {"counterbore_hole"})

        self.assertIn(
            {
                "source": "topology_rule",
                "proposal_macro": "counterbore_hole",
                "face_indices": [1, 2, 3],
            },
            proposals,
        )

    def test_build_records_labels_truth_candidate_and_reject_prediction(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[0, 0, 1], descriptors={"area": 100.0}),
                graph_node(1, "plane", normal=[1, 0, 0], descriptors={"area": 10.0}),
                graph_node(2, "plane", normal=[-1, 0, 0], descriptors={"area": 10.0}),
                graph_node(3, "cylinder", axis=[0, 1, 0], radius=2.0, descriptors={"area": 20.0}),
            ],
            [
                (0, 1, "line", "convex", 10.0),
                (1, 0, "line", "convex", 10.0),
                (0, 2, "line", "convex", 10.0),
                (2, 0, "line", "convex", 10.0),
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
        prediction_data = {
            "sample_id": "toy",
            "faces": [
                face(0, macro_id=20, macro="rib"),
                face(1, macro_id=20, macro="rib"),
                face(2, macro_id=20, macro="rib"),
                face(3, macro_id=25, macro="fillet"),
            ],
        }

        records = build_candidate_records_for_graph(
            sample_id="toy",
            graph_data=graph_data,
            prediction_data=prediction_data,
            graph_path=Path("toy.json"),
            prediction_path=Path("toy_pred.json"),
            target_macros={"rib", "fillet"},
        )

        labels_by_faces = {tuple(record["face_indices"]): record["label"] for record in records}
        self.assertEqual(labels_by_faces[(0, 1, 2)], "rib")
        self.assertEqual(labels_by_faces[(3,)], "reject")
        self.assertIn("area_ratio", records[0]["features"])
        self.assertIn("proposal_macro_rib", records[0]["features"])

    def test_write_candidate_dataset_creates_splits_metadata_and_vectors(self):
        records = [
            {
                "sample_id": "a",
                "source": "truth",
                "proposal_macro": "rib",
                "label": "rib",
                "face_indices": [0, 1, 2],
                "features": {"area_ratio": 0.1, "face_count": 3.0},
            },
            {
                "sample_id": "b",
                "source": "prediction_component",
                "proposal_macro": "fillet",
                "label": "reject",
                "face_indices": [3],
                "features": {"area_ratio": 0.02, "face_count": 1.0},
            },
        ]
        splits = split_candidate_records(records, seed=1, train_ratio=0.5, val_ratio=0.25)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            metadata = write_candidate_dataset(records, splits, output_dir)
            train_path = output_dir / "train.jsonl"
            val_path = output_dir / "val.jsonl"
            test_path = output_dir / "test.jsonl"
            metadata_path = output_dir / "metadata.json"

            self.assertTrue(train_path.exists())
            self.assertTrue(val_path.exists())
            self.assertTrue(test_path.exists())
            self.assertTrue(metadata_path.exists())
            self.assertEqual(metadata["feature_schema"], ["area_ratio", "face_count"])
            self.assertEqual(metadata["label_to_id"]["reject"], 0)

            loaded_records = []
            for path in [train_path, val_path, test_path]:
                loaded_records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
            self.assertEqual(len(loaded_records), 2)
            self.assertEqual(len(loaded_records[0]["feature_vector"]), 2)


if __name__ == "__main__":
    unittest.main()
