import json
import tempfile
import unittest
from pathlib import Path

from tests.test_topology_postprocess import face, graph, graph_node
from utils.rib_candidate_hardcases import (
    build_colored_rib_hardcase_records,
    merge_rib_candidate_records,
)


class RibCandidateHardcaseDatasetTest(unittest.TestCase):
    def test_colored_rib_truth_adds_positive_and_adjacent_negative_records(self):
        graph_data = graph(
            [
                graph_node(0, "plane", normal=[1, 0, 0], descriptors={"area": 20.0, "projected_aspect_ratio": 8.0}),
                graph_node(1, "plane", normal=[0, 0, 1], descriptors={"area": 500.0, "projected_aspect_ratio": 1.2}),
                graph_node(2, "plane", normal=[0, 1, 0], descriptors={"area": 450.0, "projected_aspect_ratio": 1.1}),
            ],
            [
                (0, 1, "line", "convex", 12.0),
                (1, 0, "line", "convex", 12.0),
                (0, 2, "circle", "convex", 6.0),
                (2, 0, "circle", "convex", 6.0),
            ],
        )
        prediction_data = {
            "sample_id": "toy",
            "faces": [
                face(0, macro_id=0, macro="stock"),
                face(1, macro_id=0, macro="stock"),
                face(2, macro_id=20, macro="rib"),
            ],
        }
        template = {
            "samples": [
                {
                    "sample_id": "toy",
                    "graph_path": "toy.json",
                    "candidate_faces": [
                        {"face_idx": 0, "review": {"is_rib_face": True, "rib_instance_id": "rib_1"}},
                        {"face_idx": 1, "review": {"is_rib_face": False, "rib_instance_id": ""}},
                    ],
                }
            ]
        }
        diagnostics = {
            "samples": [
                {
                    "sample_id": "toy",
                    "colored_rib_faces": [0],
                    "colored_background_faces": [1, 2],
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "toy.json").write_text(json.dumps(graph_data), encoding="utf-8")
            prediction_dir = root / "predictions"
            prediction_dir.mkdir()
            (prediction_dir / "toy.json").write_text(json.dumps(prediction_data), encoding="utf-8")
            records = build_colored_rib_hardcase_records(
                review_template=template,
                colored_diagnostics=diagnostics,
                root_dir=root,
                prediction_dir=prediction_dir,
            )

        positive = [record for record in records if record["label"] == "rib_candidate"]
        negative = [record for record in records if record["label"] == "not_rib_candidate"]

        self.assertEqual({record["candidate_face_idx"] for record in positive}, {0})
        self.assertIn(1, {record["candidate_face_idx"] for record in negative})
        self.assertIn(2, {record["candidate_face_idx"] for record in negative})
        self.assertIn("hole_cut_rib", positive[0]["hard_case_types"])
        self.assertIn("missed_or_non_rib_prediction", positive[0]["hard_case_types"])
        self.assertGreater(positive[0]["sample_weight"], 1.0)
        self.assertIn("adjacent_non_rib_negative", negative[0]["hard_case_types"])
        self.assertIn("features", positive[0])
        self.assertIn("candidate_area_ratio", positive[0]["features"])

    def test_merge_uses_hardcase_truth_when_base_record_conflicts(self):
        base_records = [
            {
                "sample_id": "toy",
                "candidate_face_idx": 0,
                "label": "not_rib_candidate",
                "source": "colored_rib_truth",
                "sample_weight": 1.0,
                "features": {"candidate_area_ratio": 0.1},
            }
        ]
        hard_records = [
            {
                "sample_id": "toy",
                "candidate_face_idx": 0,
                "label": "rib_candidate",
                "source": "colored_rib_hardcase_positive",
                "sample_weight": 5.0,
                "features": {"candidate_area_ratio": 0.1},
                "hard_case_types": ["complex_multi_face_rib"],
            }
        ]

        merged = merge_rib_candidate_records(base_records, hard_records)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["label"], "rib_candidate")
        self.assertEqual(merged[0]["sample_weight"], 5.0)
        self.assertIn("complex_multi_face_rib", merged[0]["hard_case_types"])


if __name__ == "__main__":
    unittest.main()
