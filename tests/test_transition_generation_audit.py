import json
import tempfile
import unittest
from pathlib import Path

from utils.transition_generation_audit import audit_transition_generation


class TransitionGenerationAuditTest(unittest.TestCase):
    def test_audit_counts_transition_features_and_size_outliers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graphs_dir = Path(tmpdir)
            (graphs_dir / "sample.json").write_text(json.dumps({
                "stock_profile": {
                    "dimensions": {"L": 120.0, "W": 100.0, "H": 50.0},
                },
                "feature_sets": [
                    {
                        "macro_feature": "fillet",
                        "instance_id": 2,
                        "node_indices": [1],
                        "geometry_parameters": {"radius": 1.2},
                        "raw_parameters": {"radius": 1.2, "curve_type": "line"},
                    },
                    {
                        "macro_feature": "chamfer",
                        "instance_id": 3,
                        "node_indices": [2],
                        "geometry_parameters": {"distance": 6.5},
                        "raw_parameters": {"distance": 6.5, "curve_type": "circle"},
                    },
                ],
            }), encoding="utf-8")

            report = audit_transition_generation(graphs_dir, max_size_ratio=0.08)

        self.assertEqual(2, report["transition_total"])
        self.assertEqual({"chamfer": 1, "fillet": 1}, report["macro_counts"])
        self.assertEqual({"circle": 1, "line": 1}, report["curve_type_counts"])
        self.assertEqual(1, report["oversized_transition_count"])

    def test_audit_flags_forbidden_context_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graphs_dir = Path(tmpdir)
            (graphs_dir / "sample.json").write_text(json.dumps({
                "feature_sets": [
                    {
                        "macro_feature": "chamfer",
                        "instance_id": 1,
                        "node_indices": [4],
                        "raw_parameters": {
                            "distance": 1.0,
                            "transition_context": "hole_mouth",
                        },
                    },
                ],
            }), encoding="utf-8")

            report = audit_transition_generation(graphs_dir)

        self.assertEqual(1, report["forbidden_context_key_count"])
        self.assertEqual("transition_context", report["forbidden_context_keys"][0]["key"])


if __name__ == "__main__":
    unittest.main()
