import json
import tempfile
import unittest
from pathlib import Path

from utils.hole_generation_audit import audit_hole_generation


class HoleGenerationAuditTest(unittest.TestCase):
    def test_audit_flags_side_axis_hole_on_cylinder_stock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graphs_dir = Path(tmpdir)
            (graphs_dir / "sample.json").write_text(json.dumps({
                "stock_profile": {
                    "stock_type": "cylinder",
                },
                "feature_sets": [
                    {
                        "macro_feature": "stepped_hole_round",
                        "instance_id": 2,
                        "node_indices": [1, 2, 3],
                        "raw_parameters": {"axis": "x"},
                    }
                ],
            }), encoding="utf-8")

            report = audit_hole_generation(graphs_dir)

        self.assertEqual(1, report["hole_total"])
        self.assertEqual(1, report["internal_entry_risk_count"])
        self.assertEqual("sample.json", report["internal_entry_risks"][0]["graph"])

    def test_audit_accepts_side_axis_hole_on_block_stock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graphs_dir = Path(tmpdir)
            (graphs_dir / "sample.json").write_text(json.dumps({
                "stock_profile": {
                    "stock_type": "block",
                },
                "feature_sets": [
                    {
                        "macro_feature": "counterbore_hole",
                        "instance_id": 3,
                        "node_indices": [1, 2, 3],
                        "raw_parameters": {"axis": "y"},
                    }
                ],
            }), encoding="utf-8")

            report = audit_hole_generation(graphs_dir)

        self.assertEqual(1, report["hole_total"])
        self.assertEqual(0, report["internal_entry_risk_count"])
        self.assertEqual({"y": 1}, report["axis_counts"])


if __name__ == "__main__":
    unittest.main()
