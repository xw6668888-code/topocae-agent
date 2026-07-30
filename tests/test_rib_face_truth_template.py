import tempfile
import unittest
from pathlib import Path

from generate_rib_face_truth_template import (
    build_rib_boundary_records_from_review_template,
    build_rib_face_truth_template,
)
from tests.test_topology_postprocess import edge, graph_node


def graph_with_edges(nodes, edge_defs):
    edges = []
    attrs = []
    for edge_def in edge_defs:
        pair, attr = edge(*edge_def)
        edges.append(pair)
        attrs.append(attr)
    return {"nodes": nodes, "edges": edges, "edge_attributes": attrs}


class RibFaceTruthTemplateTest(unittest.TestCase):
    def test_template_selects_rib_error_samples_and_candidate_faces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_path = root / "graphs" / "partA.json"
            prediction_path = root / "predictions" / "partA.json"
            feature_path = root / "features" / "partA.json"
            graph_path.parent.mkdir()
            prediction_path.parent.mkdir()
            feature_path.parent.mkdir()

            graph_data = graph_with_edges(
                [
                    graph_node(0, "plane", normal=[1, 0, 0], descriptors={"area": 20.0, "projected_aspect_ratio": 12.0}),
                    graph_node(1, "plane", normal=[-1, 0, 0], descriptors={"area": 20.0, "projected_aspect_ratio": 12.0}),
                    graph_node(2, "plane", normal=[0, 0, 1], descriptors={"area": 30.0, "projected_aspect_ratio": 5.0}),
                    graph_node(3, "plane", normal=[0, 1, 0], descriptors={"area": 500.0, "projected_aspect_ratio": 1.4}),
                ],
                [
                    (0, 1, "line", "convex", 10.0),
                    (1, 0, "line", "convex", 10.0),
                    (1, 2, "line", "convex", 10.0),
                    (2, 1, "line", "convex", 10.0),
                    (2, 3, "line", "convex", 20.0),
                    (3, 2, "line", "convex", 20.0),
                ],
            )
            write_text_json(graph_path, graph_data)

            write_text_json(
                prediction_path,
                {
                    "sample_id": "partA",
                    "source_graph_path": str(graph_path),
                    "faces": [
                        {"idx": 0, "macro_feature": "rib", "macro_confidence": 0.91, "semantic_subtype": "rib_side_plane"},
                        {"idx": 1, "macro_feature": "stock", "macro_confidence": 0.65, "semantic_subtype": "stock_face"},
                        {"idx": 2, "macro_feature": "stock", "macro_confidence": 0.70, "semantic_subtype": "stock_face"},
                        {"idx": 3, "macro_feature": "stock", "macro_confidence": 0.99, "semantic_subtype": "stock_face"},
                    ],
                    "rib_boundary_classifier_postprocess": {
                        "decision_records": [
                            {
                                "stage": "expand",
                                "action": "skipped_low_confidence",
                                "seed_face_indices": [0],
                                "candidate_face_idx": 1,
                                "label": "rib_member",
                                "confidence": 0.62,
                            }
                        ]
                    },
                }
            )
            write_text_json(
                feature_path,
                {
                    "sample_id": "partA",
                    "source_step_path": "partA.stp",
                    "feature_objects": [
                        {"macro_feature": "rib", "face_indices": [0], "face_count": 1, "area_total": 20.0}
                    ],
                }
            )
            area_review = {
                "samples": [
                    {
                        "sample_id": "partA",
                        "source_step_path": "partA.stp",
                        "missed_truth": [{"macro": "rib", "face_count": 3, "area": 70.0}],
                        "false_positives": [],
                    }
                ]
            }

            template = build_rib_face_truth_template(
                area_review=area_review,
                graph_dir=graph_path.parent,
                prediction_dir=prediction_path.parent,
                feature_objects_dir=feature_path.parent,
            )

        self.assertEqual(template["schema"], "rib_face_truth_template_v1")
        self.assertEqual([sample["sample_id"] for sample in template["samples"]], ["partA"])
        candidate_by_idx = {
            candidate["face_idx"]: candidate
            for candidate in template["samples"][0]["candidate_faces"]
        }
        self.assertIn(0, candidate_by_idx)
        self.assertIn(1, candidate_by_idx)
        self.assertIn("predicted_rib_face", candidate_by_idx[0]["candidate_reasons"])
        self.assertIn("rib_boundary_decision_face", candidate_by_idx[1]["candidate_reasons"])
        self.assertIsNone(candidate_by_idx[1]["review"]["is_rib_face"])
        self.assertEqual(candidate_by_idx[1]["review"]["rib_instance_id"], "")

    def test_reviewed_template_builds_boundary_classifier_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_path = root / "partA.json"
            graph_data = graph_with_edges(
                [
                    graph_node(0, "plane", normal=[1, 0, 0], descriptors={"area": 20.0}),
                    graph_node(1, "plane", normal=[-1, 0, 0], descriptors={"area": 20.0}),
                    graph_node(2, "plane", normal=[0, 0, 1], descriptors={"area": 30.0}),
                    graph_node(3, "plane", normal=[0, 1, 0], descriptors={"area": 500.0}),
                ],
                [
                    (0, 1, "line", "convex", 10.0),
                    (1, 0, "line", "convex", 10.0),
                    (1, 2, "line", "convex", 10.0),
                    (2, 1, "line", "convex", 10.0),
                    (2, 3, "line", "convex", 20.0),
                    (3, 2, "line", "convex", 20.0),
                ],
            )
            write_text_json(graph_path, graph_data)
            template = {
                "schema": "rib_face_truth_template_v1",
                "samples": [
                    {
                        "sample_id": "partA",
                        "graph_path": str(graph_path),
                        "candidate_faces": [
                            {"face_idx": 0, "review": {"is_rib_face": True, "rib_instance_id": "rib_1"}},
                            {"face_idx": 1, "review": {"is_rib_face": True, "rib_instance_id": "rib_1"}},
                            {"face_idx": 2, "review": {"is_rib_face": True, "rib_instance_id": "rib_1"}},
                            {"face_idx": 3, "review": {"is_rib_face": False, "rib_instance_id": ""}},
                        ],
                    }
                ],
            }

            records = build_rib_boundary_records_from_review_template(template)

        labels = {record["label"] for record in records}
        self.assertIn("rib_member", labels)
        self.assertIn("not_rib_member", labels)
        self.assertTrue(any(record["candidate_face_idx"] == 3 for record in records))
        self.assertTrue(all(record["source"] == "reviewed_rib_face_truth" for record in records))


def write_text_json(path: Path, payload: dict) -> None:
    import json

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")



if __name__ == "__main__":
    unittest.main()


