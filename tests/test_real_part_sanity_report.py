from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from generate_real_part_sanity_report import (
    build_sanity_report,
    evaluate_truth_plausibility,
    load_graphs,
)


class RealPartSanityReportTest(unittest.TestCase):
    def test_truth_object_with_matching_feature_object_is_plausible(self) -> None:
        truth = {"sample": "part1", "macro": "fillet", "area": 100.0, "face_count": 1, "index": 0}
        feature_objects = [{"macro": "fillet", "area": 100.2, "face_indices": [3], "face_count": 1}]

        result = evaluate_truth_plausibility(truth, [90.0, 100.1], feature_objects)

        self.assertEqual(result["status"], "plausible")
        self.assertIn("matching_feature_object_area", result["reasons"])
        self.assertIn("matching_graph_face_area", result["reasons"])

    def test_truth_area_without_nearby_graph_or_object_area_is_suspicious(self) -> None:
        truth = {
            "sample": "part20",
            "macro": "counterbore_hole",
            "area": 3.0,
            "face_count": 3,
            "index": 0,
        }

        result = evaluate_truth_plausibility(truth, [1900.0, 2450.0], [])

        self.assertEqual(result["status"], "suspicious")
        self.assertIn("no_nearby_area_in_graph_or_feature_objects", result["reasons"])
        self.assertIn("nearest_area_ratio_close_to_in2_to_mm2", result["reasons"])

    def test_missing_graph_is_reported(self) -> None:
        summary = build_sanity_report({"part1": []}, {}, {})

        self.assertEqual(summary["samples"]["part1"]["flags"], ["missing_graph", "missing_truth", "missing_feature_objects"])

    def test_sample_with_many_suspicious_areas_gets_flagged(self) -> None:
        truth = {
            "part1": [
                {"sample": "part1", "macro": "fillet", "area": 10.0, "face_count": 1, "index": 0},
                {"sample": "part1", "macro": "chamfer", "area": 12.0, "face_count": 1, "index": 1},
            ]
        }
        graph = {"part1": {"nodes": [{"attributes": {"physical": {"area": 8000.0}}}]}}

        summary = build_sanity_report(truth, graph, {})

        self.assertIn("many_suspicious_truth_areas", summary["samples"]["part1"]["flags"])

    def test_load_graphs_ignores_label_map_json_without_nodes(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "part1.json").write_text('{"nodes": [], "edges": []}', encoding="utf-8")
            (root / "primitive_surface_label_map.json").write_text('{"plane": 1}', encoding="utf-8")

            graphs = load_graphs(root)

        self.assertEqual(sorted(graphs), ["part1"])

if __name__ == "__main__":
    unittest.main()
