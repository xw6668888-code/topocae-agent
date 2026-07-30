import json
import tempfile
import unittest
from pathlib import Path

from generate_failure_atlas import build_failure_atlas


class GenerateFailureAtlasTest(unittest.TestCase):
    def test_build_failure_atlas_merges_truth_graph_predictions_and_feature_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_dir = root / "graphs"
            pred_dir = root / "predictions"
            feature_dir = root / "features"
            graph_dir.mkdir()
            pred_dir.mkdir()
            feature_dir.mkdir()
            (graph_dir / "part6.json").write_text(
                json.dumps({
                    "nodes": [
                        {
                            "idx": 1,
                            "primitive_surface": "cylinder",
                            "attributes": {"physical": {"area": 15.0}, "specific": {"radius": 2.0}},
                        },
                        {
                            "idx": 2,
                            "primitive_surface": "cone",
                            "attributes": {"physical": {"area": 17.0}, "specific": {}},
                        },
                    ],
                    "edges": [[1, 2]],
                    "edge_attributes": [{"curve_type": "circle", "convexity": "smooth", "length": 12.56}],
                }),
                encoding="utf-8",
            )
            (pred_dir / "part6.json").write_text(
                json.dumps({
                    "faces": [
                        {"idx": 1, "macro_feature": "stock", "semantic_subtype": "stock_face"},
                        {"idx": 2, "macro_feature": "chamfer", "semantic_subtype": "chamfer_bevel_plane"},
                    ]
                }),
                encoding="utf-8",
            )
            (feature_dir / "part6.json").write_text(
                json.dumps({
                    "feature_objects": [
                        {
                            "macro_feature": "chamfer",
                            "face_indices": [2],
                            "face_count": 1,
                            "fingerprint": {"area_total": 17.0},
                        }
                    ]
                }),
                encoding="utf-8",
            )
            truth_path = root / "truth.json"
            truth_path.write_text(
                json.dumps({
                    "samples": {
                        "part6": [
                            {"area": 32.14, "macro": "countersink_hole", "face_count": 2}
                        ]
                    }
                }),
                encoding="utf-8",
            )

            atlas = build_failure_atlas(
                graph_dir=graph_dir,
                prediction_stages={"post": pred_dir},
                feature_object_stages={"post": feature_dir},
                truth_json=truth_path,
                sample_ids=["part6"],
            )

        self.assertEqual(atlas["sample_count"], 1)
        sample = atlas["samples"][0]
        self.assertEqual(sample["sample_id"], "part6")
        self.assertEqual(sample["truth"][0]["macro"], "countersink_hole")
        self.assertEqual(sample["faces"][1]["stage_predictions"]["post"]["macro_feature"], "chamfer")
        self.assertEqual(sample["feature_objects"]["post"][0]["face_indices"], [2])


if __name__ == "__main__":
    unittest.main()
